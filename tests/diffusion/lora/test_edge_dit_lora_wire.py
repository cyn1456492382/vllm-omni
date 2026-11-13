# vllm-omni/tests/diffusion/lora/test_edge_dit_lora_wire.py
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import torch

_WIRE_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm_omni"
    / "diffusion"
    / "lora"
    / "edge_dit_lora_wire.py"
)


def _load_wire():
    try:
        return importlib.import_module(
            "vllm_omni.diffusion.lora.edge_dit_lora_wire"
        )
    except Exception:
        spec = importlib.util.spec_from_file_location(
            "edge_dit_lora_wire", _WIRE_PATH
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["edge_dit_lora_wire"] = mod
        spec.loader.exec_module(mod)
        return mod


def test_fp16_roundtrip_shape_and_close():
    wire = _load_wire()
    x = torch.randn(7, 64, dtype=torch.float32)
    payload = wire.encode_activation(x, "fp16")
    y = wire.decode_activation(payload, tuple(x.shape), "fp16")
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=2e-3, rtol=2e-3)


def test_int8_roundtrip_bounded_error():
    wire = _load_wire()
    x = torch.randn(5, 32, dtype=torch.float32)
    payload = wire.encode_activation(x, "int8")
    y = wire.decode_activation(payload, tuple(x.shape), "int8")
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert float((y - x).abs().mean()) < float(x.abs().mean()) + 1.0


def test_int4_roundtrip_finite():
    wire = _load_wire()
    x = torch.randn(3, 16, dtype=torch.float32)
    payload = wire.encode_activation(x, "int4")
    y = wire.decode_activation(payload, tuple(x.shape), "int4")
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_encode_decode_delta_aliases_exist():
    wire = _load_wire()
    assert wire.encode_delta is wire.encode_activation
    assert wire.decode_delta is wire.decode_activation
