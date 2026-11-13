# Fixed Patch-Tile Smoke

This directory contains the fail-closed smoke driver for the fixed latent
patch metadata path. It does not claim that a Python `Transformer` call is a
single fused DiT CUDA kernel.

Run inside the cu124 container with the vLLM-Omni compatibility path first:

```bash
source /opt/tp2-cu124-state/venv-cu124/bin/activate
cd /workspace/vllm-omni
export PYTHONPATH=/opt/tp2-cu124-state/runtime-vllm-omni-compat-20261027:/workspace/vllm-omni
python benchmarks/diffusion/patch_tiling/run_patch_tiling_smoke.py
```

The metadata smoke must report `metadata=PASS`. `--require-fused` is expected
to fail until a real single-launch fused DiT operator is registered and wired
through the Flux pipeline. Both integration gates must be set only by the
implementation that owns the operator. That failure is intentional and blocks
formal experiments.

The current metadata path records these overheads:

- overlap token copies: duplicated valid token positions caused by overlap;
- packed padding: invalid tail positions in a rectangular patch buffer;
- device metadata: descriptor and gather/scatter index buffers;
- launch overhead: not yet measurable because the fused operator is absent.

The current Flux Transformer fallback is not an acceptable substitute: it
executes the regular per-layer attention, MLP, and projection operators and
therefore cannot satisfy the one-launch MixFusion contract.

For a real operator, profile the exact same command with:

```bash
/workspace/vllm-omni/scripts/patch_tiling/profile_patch_tiling_nsys.sh \
  /path/to/results/patch_tiling.nsys-rep \
  python <real_single_step_driver.py>
```

The resulting Nsight Systems report must show that increasing patch count does
not create a host loop of independent patch kernel launches. Formal T1 work
must additionally provide output parity/PSNR, endpoint latency, peak HBM, and
the exact launch-count comparison against the non-tiled baseline.
