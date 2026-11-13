# TetrisLoRA deliverables and evidence boundary

The implementation keeps the existing CUDA 12.4 environment and does not
install dependencies or alter existing experiment artifacts.

Session resources are created during the runner's pre-measurement warmup.
Formal timing begins only after that warmup and reuses the persistent IPC
slots, imported tensor views, and interprocess event handles; per-operation
metadata, sequence numbers, and readiness/consumer synchronization remain in
the measured path.

## Delivered code

- `tetris_lora_twostage.py` provides a stream-ordered, two-stage BF16 path:
  Shrink and Expand+Add are separate Tensor-Core kernels, with FP32
  accumulation, compact per-branch H storage, bounded branch groups, and
  reusable descriptor/workspace metadata.
- `tetris_lora_runtime.py` rejects mixed fused/split packed-QKV layouts instead
  of silently dropping one layout, and maps independent Q/K/V results into
  their declared output slices.
- `my_module_mixfusion_token` routes LoRA through the MixFusion runtime at the
  same token-tile boundary as Base Linear work. The existing
  `system_b_mixfusion_patch` path remains full-sequence and is unchanged.
- `tetris_lora_runtime.py` uses the existing `dense_fused_lora_add` contract,
  creates a zero residual buffer, caches device-side adapter stacks, and keeps
  FP32 matmul behind `TETRIS_LORA_REFERENCE=1`.
- `tetris_lora_transport.py` carries request/row mapping, capacity, offset, and
  alpha-folded metadata in the descriptor.
- CUDA IPC uses persistent session-scoped X and Δ slot registrations: the
  allocation/export/import and interprocess event handle are created on first
  use per shape/dtype slot and reused for later operations; operation IDs and
  generations remain dynamic, and X is copied into a stable producer slot.
- `base_linear.py` dispatches TetrisLoRA inside the existing token-tile
  pack/base-GEMM/scatter path, issuing one remote operation per tile and never
  applying the local LoRA residual a second time.
- `run_tetris_lora_flux_eager_correctness.py` accepts separate native/Tetris
  implementation and tiling modes. `run_tetris_lora_matrix.sh` covers the
  requested 512²/1024²/2048² no-tile/no-fused and tile+fused comparisons.
- `forward_context.py` and the stepwise model runner pass request IDs, batch ID,
  and denoise step into the Tetris descriptor path.
- `measure_tetris_ipc_nvlink.py` records bidirectional IPC payload timing,
  exact receiver checksum, and before/after NVLink throughput counters. It
  marks attribution only when source Tx and target Rx deltas match the
  payload size per active link; it never infers NVLink use from `NV8` alone.
- `run_tetris_ucx_worker.py` and `configs/ucx_worker.example.json` provide the
  external-worker configuration boundary. The launcher fails closed because the
  current repository does not yet expose a complete Python-to-UCX control-plane
  handshake adapter; this is intentionally not reported as an implemented
  end-to-end RDMA backend.

## Evidence boundary

- CUDA IPC/P2P is a local transport. It is not automatically evidence of an
  NVLink payload path.
- An NVLink payload claim requires a live IPC experiment plus attributable,
  repeatable counter deltas matching direction and payload size.
- UCX bridge compilation and capability inspection are not cross-node GDR
  validation. Cross-node RDMA remains unvalidated without two nodes and an IB
  device.

The current host/container reports UCX 1.15.0 with CUDA and GDRCopy plugins,
but no `/dev/infiniband` and no verified `mlx5` transport. UCX therefore stays
fail-closed; it does not silently switch to IPC, TCP payload, or host staging.

## Token-tile scheduling contract

- The runner builds one `BatchTokenTilePlan` per diffusion step from each
  request's latent length, padded batch sequence length, and selected tile
  size. Fixed policy executes each ordinal tile independently; dynamic policy
  keeps the existing packed/queue path and records tile metadata.
- In `my_module_mixfusion_token`, each token tile enters one MixFusion
  descriptor containing Base Linear plus LoRA residual ownership. The tile is
  packed, computed, and scattered back to request/token positions.
- The boundary is valid for token-wise Linear projections, including QKV and
  output/MLP linears. Attention reduction and softmax are not tiled: they still
  consume the fully scattered sequence, preserving cross-token normalization.
- `system_b_mixfusion_patch` remains the source patch/full-sequence mode, and
  the remote Tetris/NVLink path is not implicitly switched by this change.

## 2026-11-10 local evidence

- GPU5 to GPU6 and GPU6 to GPU5 IPC payloads completed with exact checksum
  validation for 1 MiB and 64 MiB payloads; the extended run also covered
  16 MiB and 256 MiB.
- For every sampled run, all eight source links increased Tx and all eight
  target links increased Rx by exactly `payload_bytes / 1024 / 8` KiB per
  link. This is attributable payload-counter evidence for the measured IPC
  path, not merely topology evidence.
- Measured median effective bandwidth in the attribution run was 39.474 GiB/s
  for GPU5 to GPU6 at 64 MiB and 40.923 GiB/s for GPU6 to GPU5 at 64 MiB.
- The run used `CUDA_VISIBLE_DEVICES=5,6` inside the existing CUDA 12.4
  container and wrote results under
  `.tetris_lora_results/20261110_nvlink_validation/`.
- The same-kernel single-layer IPC run used the real EnvyFlux adapter and
  reported zero max/mean error against the co-located `dense_fused_lora_add`
  reference for `X=[32,3072]` and `delta=[32,12288]`. Numerical thresholds
  remain ungated pending user confirmation.

## 2026-11-11 implementation validation

- The CUDA 12.4 container imported the patched runtime and UCX modules.
- The IPC smoke ran three repeated sends through one persistent producer slot
  and one persistent imported consumer view on temporary visible GPUs 2/3;
  checksums matched on every receive. This validates slot/event reuse only and
  does not claim NVLink for that run.
- GPU0/GPU1 were left untouched because existing long-lived engine processes
  occupied them during this validation window.
