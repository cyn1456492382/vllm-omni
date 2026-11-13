# vllm-omni/tests/diffusion/lora/test_edge_dit_lora_provider_loopback.py
from __future__ import annotations

import importlib
import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path

import torch

_LORA_DIR = (
    Path(__file__).resolve().parents[3] / "vllm_omni" / "diffusion" / "lora"
)


def _load_mod(name: str):
    try:
        return importlib.import_module(f"vllm_omni.diffusion.lora.{name}")
    except Exception:
        path = _LORA_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        # Ensure wire is available under expected name for provider imports.
        if name != "edge_dit_lora_wire":
            _load_mod("edge_dit_lora_wire")
        spec.loader.exec_module(mod)
        return mod


def _serve_once(port: int, A: torch.Tensor, B: torch.Tensor, ready: threading.Event) -> None:
    wire = _load_mod("edge_dit_lora_wire")
    provider_mod = _load_mod("edge_dit_lora_provider")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    ready.set()
    conn, _ = srv.accept()
    header = json.loads(wire.recv_frame(conn).decode("utf-8"))
    x = wire.decode_activation(
        wire.recv_frame(conn),
        (int(header["rows"]), int(header["in_features"])),
        str(header["precision"]),
    )
    delta = provider_mod.local_lora_delta(x, A, B)
    wire.send_frame(conn, wire.encode_delta(delta, str(header["precision"])))
    conn.close()
    srv.close()


def test_provider_loopback_matches_local_gemm():
    provider_mod = _load_mod("edge_dit_lora_provider")
    rows, din, rank, dout = 8, 64, 4, 64
    x = torch.randn(rows, din)
    A = torch.randn(rank, din)
    B = torch.randn(dout, rank)
    port = 19260
    ready = threading.Event()
    t = threading.Thread(
        target=_serve_once, args=(port, A, B, ready), daemon=True
    )
    t.start()
    assert ready.wait(timeout=5)
    time.sleep(0.05)
    provider = provider_mod.EdgeDiTLoRAResidualProvider(
        host="127.0.0.1",
        port=port,
        adapter_id="synth",
        precision="fp16",
    )
    got = provider.compute_residual(module_name="m", x=x, out_features=dout)
    ref = provider_mod.local_lora_delta(x, A, B)
    assert torch.allclose(got.cpu().float(), ref, atol=2e-2, rtol=2e-2)
    t.join(timeout=5)
