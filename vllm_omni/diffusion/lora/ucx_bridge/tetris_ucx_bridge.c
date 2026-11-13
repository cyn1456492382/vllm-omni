#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <ucp/api/ucp.h>
#include <ucp/api/ucp_compat.h>
#include <ucs/type/status.h>
#include <cuda_runtime_api.h>

typedef struct tetris_mem_entry {
    ucp_mem_h memh;
    struct tetris_mem_entry *next;
} tetris_mem_entry_t;

typedef struct tetris_rkey_entry {
    ucp_rkey_h rkey;
    struct tetris_rkey_entry *next;
} tetris_rkey_entry_t;

typedef struct {
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    tetris_mem_entry_t *memories;
    tetris_rkey_entry_t *rkeys;
} tetris_ucx_context_t;

static int status_to_int(ucs_status_t status) {
    return (int)status;
}

int tetris_ucx_open(tetris_ucx_context_t **out) {
    if (out == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    *out = NULL;

    ucp_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_PARAM_FIELD_FEATURES |
                        UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    params.features = UCP_FEATURE_RMA;
    params.estimated_num_eps = 1;

    ucp_context_h context = NULL;
    ucs_status_t status = ucp_init(&params, NULL, &context);
    if (status != UCS_OK) {
        return status_to_int(status);
    }

    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

    ucp_worker_h worker = NULL;
    status = ucp_worker_create(context, &worker_params, &worker);
    if (status != UCS_OK) {
        ucp_cleanup(context);
        return status_to_int(status);
    }

    tetris_ucx_context_t *handle = calloc(1, sizeof(*handle));
    if (handle == NULL) {
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return UCS_ERR_NO_MEMORY;
    }
    handle->context = context;
    handle->worker = worker;
    *out = handle;
    return UCS_OK;
}

int tetris_ucx_worker_address(
    tetris_ucx_context_t *handle, void *buffer, size_t capacity, size_t *size_out) {
    if (handle == NULL || buffer == NULL || size_out == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    ucp_address_t *address = NULL;
    size_t address_length = 0;
    ucs_status_t status = ucp_worker_get_address(
        handle->worker, &address, &address_length);
    if (status != UCS_OK) {
        return status_to_int(status);
    }
    *size_out = address_length;
    if (address_length > capacity) {
        ucp_worker_release_address(handle->worker, address);
        return UCS_ERR_BUFFER_TOO_SMALL;
    }
    memcpy(buffer, address, address_length);
    ucp_worker_release_address(handle->worker, address);
    return UCS_OK;
}

int tetris_ucx_connect(
    tetris_ucx_context_t *handle, const void *address, size_t address_length) {
    if (handle == NULL || address == NULL || address_length == 0) {
        return UCS_ERR_INVALID_PARAM;
    }
    (void)address_length;
    ucp_ep_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    params.address = (const ucp_address_t *)address;
    ucs_status_t status = ucp_ep_create(handle->worker, &params, &handle->ep);
    return status_to_int(status);
}

int tetris_ucx_map_cuda(
    tetris_ucx_context_t *handle, void *address, size_t length, ucp_mem_h *memh_out) {
    if (handle == NULL || address == NULL || length == 0 || memh_out == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    ucp_mem_map_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_PROT |
                        UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    params.address = address;
    params.length = length;
    params.prot = UCP_MEM_MAP_PROT_LOCAL_READ |
                  UCP_MEM_MAP_PROT_LOCAL_WRITE |
                  UCP_MEM_MAP_PROT_REMOTE_WRITE;
    params.memory_type = UCS_MEMORY_TYPE_CUDA;
    ucs_status_t status = ucp_mem_map(handle->context, &params, memh_out);
    if (status != UCS_OK) {
        return status_to_int(status);
    }
    tetris_mem_entry_t *entry = calloc(1, sizeof(*entry));
    if (entry == NULL) {
        ucp_mem_unmap(handle->context, *memh_out);
        *memh_out = NULL;
        return UCS_ERR_NO_MEMORY;
    }
    entry->memh = *memh_out;
    entry->next = handle->memories;
    handle->memories = entry;
    return UCS_OK;
}

int tetris_ucx_unmap(tetris_ucx_context_t *handle, ucp_mem_h memh) {
    if (handle == NULL || memh == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    tetris_mem_entry_t **cursor = &handle->memories;
    while (*cursor != NULL && (*cursor)->memh != memh) {
        cursor = &(*cursor)->next;
    }
    if (*cursor == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    ucs_status_t status = ucp_mem_unmap(handle->context, memh);
    if (status == UCS_OK) {
        tetris_mem_entry_t *entry = *cursor;
        *cursor = entry->next;
        free(entry);
    }
    return status_to_int(status);
}

int tetris_ucx_pack_rkey(
    tetris_ucx_context_t *handle, ucp_mem_h memh, void *buffer, size_t capacity,
    size_t *size_out) {
    if (handle == NULL || memh == NULL || buffer == NULL || size_out == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    void *packed = NULL;
    size_t packed_size = 0;
    ucs_status_t status = ucp_rkey_pack(
        handle->context, memh, &packed, &packed_size);
    if (status != UCS_OK) {
        return status_to_int(status);
    }
    *size_out = packed_size;
    if (packed_size > capacity) {
        ucp_rkey_buffer_release(packed);
        return UCS_ERR_BUFFER_TOO_SMALL;
    }
    memcpy(buffer, packed, packed_size);
    ucp_rkey_buffer_release(packed);
    return UCS_OK;
}

int tetris_ucx_unpack_rkey(
    tetris_ucx_context_t *handle, const void *buffer, size_t size, ucp_rkey_h *rkey_out) {
    if (handle == NULL || handle->ep == NULL || buffer == NULL || size == 0 ||
        rkey_out == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    ucs_status_t status = ucp_ep_rkey_unpack(handle->ep, buffer, rkey_out);
    if (status != UCS_OK) {
        return status_to_int(status);
    }
    tetris_rkey_entry_t *entry = calloc(1, sizeof(*entry));
    if (entry == NULL) {
        ucp_rkey_destroy(*rkey_out);
        *rkey_out = NULL;
        return UCS_ERR_NO_MEMORY;
    }
    entry->rkey = *rkey_out;
    entry->next = handle->rkeys;
    handle->rkeys = entry;
    return UCS_OK;
}

int tetris_ucx_destroy_rkey(tetris_ucx_context_t *handle, ucp_rkey_h rkey) {
    if (handle == NULL || rkey == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    tetris_rkey_entry_t **cursor = &handle->rkeys;
    while (*cursor != NULL && (*cursor)->rkey != rkey) {
        cursor = &(*cursor)->next;
    }
    if (*cursor == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    tetris_rkey_entry_t *entry = *cursor;
    *cursor = entry->next;
    ucp_rkey_destroy(rkey);
    free(entry);
    return UCS_OK;
}

int tetris_ucx_put(
    tetris_ucx_context_t *handle, const void *local_buffer, size_t length,
    uint64_t remote_address, ucp_rkey_h rkey, ucp_mem_h local_memh,
    void **request_out) {
    if (handle == NULL || handle->ep == NULL || local_buffer == NULL ||
        length == 0 || rkey == NULL || request_out == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    ucp_request_param_t params;
    memset(&params, 0, sizeof(params));
    if (local_memh != NULL) {
        params.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        params.memh = local_memh;
    }
    ucs_status_ptr_t request = ucp_put_nbx(
        handle->ep, local_buffer, length, remote_address, rkey, &params);
    if (UCS_PTR_IS_ERR(request)) {
        return status_to_int(UCS_PTR_STATUS(request));
    }
    *request_out = request;
    return UCS_OK;
}

int tetris_ucx_progress(tetris_ucx_context_t *handle) {
    if (handle == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    return (int)ucp_worker_progress(handle->worker);
}

int tetris_ucx_request_done(tetris_ucx_context_t *handle, void *request) {
    if (handle == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    if (request == NULL) {
        return 1;
    }
    ucs_status_t status = ucp_request_check_status(request);
    if (status == UCS_INPROGRESS) {
        return 0;
    }
    ucp_request_free(request);
    return status == UCS_OK ? 1 : status_to_int(status);
}

int tetris_ucx_flush(tetris_ucx_context_t *handle) {
    if (handle == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }
    ucp_request_param_t params;
    memset(&params, 0, sizeof(params));
    ucs_status_ptr_t request = ucp_worker_flush_nbx(handle->worker, &params);
    if (UCS_PTR_IS_ERR(request)) {
        return status_to_int(UCS_PTR_STATUS(request));
    }
    if (request == NULL) {
        return UCS_OK;
    }
    for (;;) {
        ucp_worker_progress(handle->worker);
        ucs_status_t status = ucp_request_check_status(request);
        if (status == UCS_INPROGRESS) {
            continue;
        }
        ucp_request_free(request);
        return status_to_int(status);
    }
}

int tetris_ucx_flush_remote_writes(int device) {
    int previous_device = 0;
    cudaError_t error = cudaGetDevice(&previous_device);
    if (error != cudaSuccess) {
        return (int)error;
    }
    error = cudaSetDevice(device);
    if (error != cudaSuccess) {
        return (int)error;
    }
    error = cudaDeviceFlushGPUDirectRDMAWrites(
        cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
        cudaFlushGPUDirectRDMAWritesToOwner);
    cudaError_t restore_error = cudaSetDevice(previous_device);
    if (error != cudaSuccess) {
        return (int)error;
    }
    return restore_error == cudaSuccess ? 0 : (int)restore_error;
}

int tetris_ucx_close(tetris_ucx_context_t *handle) {
    if (handle == NULL) {
        return UCS_OK;
    }
    int result = tetris_ucx_flush(handle);
    if (handle->ep != NULL) {
        ucp_request_param_t params;
        memset(&params, 0, sizeof(params));
        ucs_status_ptr_t request = ucp_ep_close_nbx(handle->ep, &params);
        if (UCS_PTR_IS_ERR(request)) {
            result = status_to_int(UCS_PTR_STATUS(request));
        } else if (request != NULL) {
            for (;;) {
                ucp_worker_progress(handle->worker);
                ucs_status_t status = ucp_request_check_status(request);
                if (status == UCS_INPROGRESS) {
                    continue;
                }
                ucp_request_free(request);
                if (status != UCS_OK) {
                    result = status_to_int(status);
                }
                break;
            }
        }
        handle->ep = NULL;
    }
    while (handle->rkeys != NULL) {
        tetris_rkey_entry_t *entry = handle->rkeys;
        handle->rkeys = entry->next;
        ucp_rkey_destroy(entry->rkey);
        free(entry);
    }
    while (handle->memories != NULL) {
        tetris_mem_entry_t *entry = handle->memories;
        handle->memories = entry->next;
        ucs_status_t unmap_status = ucp_mem_unmap(handle->context, entry->memh);
        if (unmap_status != UCS_OK && result == UCS_OK) {
            result = status_to_int(unmap_status);
        }
        free(entry);
    }
    if (handle->worker != NULL) {
        ucp_worker_destroy(handle->worker);
    }
    if (handle->context != NULL) {
        ucp_cleanup(handle->context);
    }
    free(handle);
    return result;
}
