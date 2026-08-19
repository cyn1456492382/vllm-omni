# vllm-omni/tests/diffusion/lora/test_edge_dit_lora_metadata.py
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_META_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_omni"
    / "diffusion"
    / "lora"
    / "edge_dit_lora_metadata.py"
)


def _load_meta():
    try:
        return importlib.import_module(
            "vllm_omni.diffusion.lora.edge_dit_lora_metadata"
        )
    except Exception:
        spec = importlib.util.spec_from_file_location(
            "edge_dit_lora_metadata", _META_PATH
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["edge_dit_lora_metadata"] = mod
        spec.loader.exec_module(mod)
        return mod


def test_materialize_expands_modules_and_ports():
    meta = _load_meta()
    decisions = {
        "req0": {
            "modules": ["transformer.layers.0.attention.to_qkv"],
            "placement": "edge_remote",
            "wire_precision": "int8",
            "edge_worker_host": "127.0.0.1",
            "edge_worker_port": 9260,
            "adapter_id": "synth_r4",
            "scale": 1.0,
        }
    }
    out = meta.materialize_edge_dit_lora_runtime_metadata(decisions)
    m = out["req0"]
    assert m["target_module_names"] == [
        "transformer.layers.0.attention.to_qkv"
    ]
    assert (
        m["placement_by_module"]["transformer.layers.0.attention.to_qkv"]
        == "edge_remote"
    )
    assert m["edge_worker_port"] == 9260
    assert m["wire_precision"] == "int8"
    assert m["adapter_id"] == "synth_r4"
    assert m["fail_closed"] is True
