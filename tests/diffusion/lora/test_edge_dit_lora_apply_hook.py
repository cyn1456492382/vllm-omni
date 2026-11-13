# vllm-omni/tests/diffusion/lora/test_edge_dit_lora_apply_hook.py
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch

_RT_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_omni"
    / "diffusion"
    / "lora"
    / "edge_dit_lora_runtime.py"
)


def _load_runtime():
    try:
        return importlib.import_module(
            "vllm_omni.diffusion.lora.edge_dit_lora_runtime"
        )
    except Exception:
        spec = importlib.util.spec_from_file_location(
            "edge_dit_lora_runtime", _RT_PATH
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["edge_dit_lora_runtime"] = mod
        spec.loader.exec_module(mod)
        return mod


class _FakeProvider:
    def compute_residual(self, module_name, x, out_features, precision=None):
        return torch.full(
            (x.shape[0], out_features), 2.0, dtype=x.dtype, device=x.device
        )


def test_resolve_lora_delta_prefers_edge():
    rt = _load_runtime()
    rt.clear_edge_dit_lora_runtime()
    rt.set_edge_dit_lora_runtime(
        {
            "req0": {
                "target_module_names": ["m0"],
                "placement_by_module": {"m0": "edge_remote"},
                "wire_precision": "fp16",
                "edge_worker_host": "127.0.0.1",
                "edge_worker_port": 1,
                "adapter_id": "a",
                "scale": 1.0,
                "fail_closed": True,
                "block_coalesce": False,
            }
        },
        providers={"req0": _FakeProvider()},
    )
    x = torch.randn(3, 8)
    called = {"local": 0}

    def local_fn():
        called["local"] += 1
        return torch.ones(3, 16)

    out = rt.resolve_lora_delta("m0", x, 16, local_fn)
    assert torch.allclose(out, torch.full((3, 16), 2.0))
    assert called["local"] == 0


def test_resolve_lora_delta_falls_back_local():
    rt = _load_runtime()
    rt.clear_edge_dit_lora_runtime()
    x = torch.randn(2, 4)

    def local_fn():
        return torch.ones(2, 5) * 3

    out = rt.resolve_lora_delta("unused", x, 5, local_fn)
    assert torch.allclose(out, torch.ones(2, 5) * 3)
