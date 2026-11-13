# MixFusion Implementation Audit

Date: 2026-11-06 (Asia/Shanghai)

## Implemented

- `MixFusionDescriptor` remains the ownership contract for Base, LoRA, token rows,
  ranks, dependencies, and residual Add.
- Base-only `mixfusion_both` dispatch now reaches the runtime instead of returning
  through the legacy Base path.
- Empty LoRA composition is normalized to one empty entry per request row owner.
- Mixed-rank `bucket` dispatch executes each exact-rank group once; `packed` remains
  explicitly reported as an exact-rank fallback.
- `mixfusion_transport.py` provides descriptor validation, reusable buffer
  registration, asynchronous loopback/P2P completion, mismatch detection, timeout
  handling, and fail-closed RDMA capability inspection.
- Optional attention numeric digests are emitted only when
  `VLLM_OMNI_ATTENTION_OUTPUT_TELEMETRY=1`; default timing behavior is unchanged.

## Validation

- Container: `vllm-adapter-a800-tp2-cu124-20261027`.
- `python -m compileall` for all changed Python files: passed.
- `git diff --check`: passed.
- Container Torch minimal harness covering P2P transfer and Base+LoRA MixFusion
  output: passed.
- Full pytest collection: blocked because the configured venv Python links point to
  missing paths and the container fallback Python lacks `aenum`/`transformers`.
- No dependency was installed and no other container or host environment was
  modified.

## Experiment gate

- Existing P6 artifacts remain the source of experiment history under
  `/mnt/nvme1/cyn/vllm/nfs_results/p6_mixfusion_token_tile_20261105/`.
- Existing MixFusion full-model artifacts report image-level correctness failure;
  they are not promoted to a performance or merge claim.
- Existing token-tile artifacts lack the numeric attention-output comparison. The
  new diagnostic digest is intended for a rerun, but no rerun is claimed here.
- Consequently the implementation is a diagnostic/review checkpoint, not a
  Phase-D performance promotion.

## Checkpoints

- `30bdf28f`: pre-completion checkpoint.
- `72b018bc`: pre-transport checkpoint.
- `48ce4aa9`: transport contract.
- `08a68751`: pre-validation checkpoint.
- `e67ce4d0`: transport validation and telemetry.
- `49bfb465`: pre-attention telemetry checkpoint.
- `152ba6af`: optional attention output telemetry.
- `58b9cc46`: pre-audit checkpoint.
