# vllm-omni/tests/diffusion/lora/test_edge_dit_lora_runtime.py
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
    def __init__(self):
        self.calls = 0

    def compute_residual(self, module_name, x, out_features, precision=None):
        self.calls += 1
        return torch.ones(
            x.shape[0], out_features, dtype=x.dtype, device=x.device
        )


def test_edge_remote_increments_counts():
    rt = _load_runtime()
    rt.clear_edge_dit_lora_runtime()
    fake = _FakeProvider()
    meta = {
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
    }
    rt.set_edge_dit_lora_runtime(meta, providers={"req0": fake})
    x = torch.randn(4, 8)
    out = rt.maybe_edge_dit_lora_residual("m0", x, out_features=16)
    assert out is not None and out.shape == (4, 16)
    assert rt.get_edge_dit_lora_call_counts()["calls"] == 1
    assert rt.maybe_edge_dit_lora_residual("other", x, 16) is None
