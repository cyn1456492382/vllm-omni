# TetrisLoRA UCX bridge

This directory contains the CUDA 12-compatible native bridge for the UCX 1.15
RMA path. It is intentionally not built or installed automatically.

The data path is:

```text
ucp_init(UCP_FEATURE_RMA)
  -> ucp_worker_create(UCS_THREAD_MODE_SINGLE)
  -> ucp_ep_create(remote worker address)
  -> ucp_mem_map(CUDA pointer)
  -> ucp_rkey_pack / ucp_ep_rkey_unpack
  -> ucp_put_nbx
  -> ucp_worker_progress + ucp_request_check_status
  -> ucp_worker_flush_nbx
  -> close/drain/unmap
```

The Python wrapper refuses to instantiate the endpoint unless UCX exposes an IB
device and an `rc_mlx5`, `dc_mlx5`, or `ud_mlx5` transport. It never falls back
to TCP, host staging, CUDA IPC, or another backend.

The worker uses `UCS_THREAD_MODE_SINGLE`. The caller owns the progress loop;
callbacks are not used for the RMA completion path, so completion ownership is
unambiguous: the caller progresses UCX, checks the request, then flushes before
reusing or unmapping a buffer.
