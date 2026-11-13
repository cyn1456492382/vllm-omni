# TetrisLoRA UCX compatibility record

## Environment checked

- CUDA runtime: 12.4
- PyTorch: 2.6.0+cu124
- UCX: 1.15.0
- UCX headers and libraries: `/opt/hpcx/ucx`
- CUDA headers and libraries: `/usr/local/cuda`
- GPU plugin: visible
- GDRCopy plugin: visible
- `/dev/infiniband`: absent in the checked container
- Verified `mlx5` transport: absent in the checked container

## Compile command

```bash
gcc -shared -fPIC -O2 \
  -I/opt/hpcx/ucx/include \
  -I/usr/local/cuda/include \
  vllm-omni/vllm_omni/diffusion/lora/ucx_bridge/tetris_ucx_bridge.c \
  -o /tmp/libtetris_ucx_check.so \
  -L/opt/hpcx/ucx/lib \
  -L/usr/local/cuda/lib64 \
  -lucp -lucs -luct -lcudart
```

The command completed successfully in the existing CUDA 12.4 container.
`ldd` resolved `libucp.so.0`, `libucs.so.0`, `libuct.so.0`, and
`libcudart.so.12` from the configured paths.

## Runtime boundary

The Python capability check correctly reports that UCX has no verified IB/RDMA
data plane in this environment. The UCX path therefore fails closed and does
not fall back to CUDA IPC, TCP payload, or host staging. A successful compile,
CUDA memory registration, or GDRCopy plugin alone is not an end-to-end
GPUDirect RDMA result.

Cross-node session handshake, remote completion, GPU payload path, correctness,
and performance remain unrun until a real IB/RDMA environment is provided.
