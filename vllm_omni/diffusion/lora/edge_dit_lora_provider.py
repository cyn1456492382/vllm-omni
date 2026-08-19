# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cloud TCP client for edge-owned DiT LoRA residuals."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any

import torch

try:
    from vllm_omni.diffusion.lora import edge_dit_lora_wire as wire
except Exception:  # pragma: no cover - env missing optional package deps
    import importlib.util
    from pathlib import Path

    _path = Path(__file__).with_name("edge_dit_lora_wire.py")
    _spec = importlib.util.spec_from_file_location("edge_dit_lora_wire", _path)
    assert _spec is not None and _spec.loader is not None
    wire = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(wire)


def local_lora_delta(
    x: torch.Tensor, A: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    """Compute LoRA residual matching DiffusionBaseLinearLayerWithLoRA.

    Args:
        x: Activation ``[..., in_features]``.
        A: LoRA A with shape ``(rank, in_features)``.
        B: LoRA B with shape ``(out_features, rank)``.

    Returns:
        Delta with shape ``[rows, out_features]`` in float32.
    """
    x_flat = x.reshape(-1, x.shape[-1]).to(dtype=torch.float32)
    a = A.to(dtype=torch.float32)
    b = B.to(dtype=torch.float32)
    return (x_flat @ a.t()) @ b.t()


class EdgeDiTLoRAResidualProvider:
    """TCP client that requests LoRA residuals from an edge worker."""

    def __init__(
        self,
        host: str,
        port: int,
        adapter_id: str,
        precision: str = "fp16",
        timeout_s: float = 5.0,
        fail_closed: bool = True,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.adapter_id = str(adapter_id)
        self.precision = str(precision)
        self.timeout_s = float(timeout_s)
        self.fail_closed = bool(fail_closed)
        self._sock: socket.socket | None = None
        self.last_meta: dict[str, Any] | None = None
        self._debug_path = Path("/tmp/edge_dit_provider_debug.jsonl")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock
        timeout = None if self.timeout_s <= 0 else self.timeout_s
        sock = socket.create_connection((self.host, self.port), timeout=timeout)
        sock.settimeout(timeout)
        self._sock = sock
        return sock

    def _debug_event(self, event: str, **payload: Any) -> None:
        row = {"ts": time.time(), "event": event, **payload}
        try:
            self._debug_path.parent.mkdir(parents=True, exist_ok=True)
            with self._debug_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True, default=str))
                fh.write("\n")
        except Exception:
            pass

    def compute_residual(
        self,
        module_name: str,
        x: torch.Tensor,
        out_features: int,
        precision: str | None = None,
    ) -> torch.Tensor:
        """Send activation and receive LoRA delta from the edge worker.

        Args:
            module_name: Target LoRA module name.
            x: Activation tensor ``[..., in_features]``.
            out_features: Output feature size of the linear.
            precision: Optional wire precision override.

        Returns:
            Delta tensor on ``x.device`` / ``x.dtype`` with leading rows matching
            flattened ``x``.
        """
        prec = str(precision or self.precision)
        x_flat = x.reshape(-1, x.shape[-1])
        rows, in_features = int(x_flat.shape[0]), int(x_flat.shape[1])
        header: dict[str, Any] = {
            "op": "lora_residual",
            "module_name": str(module_name),
            "rows": rows,
            "in_features": in_features,
            "out_features": int(out_features),
            "precision": prec,
            "adapter_id": self.adapter_id,
        }
        request_payload = json.dumps(header, sort_keys=True).encode("utf-8")
        activation_payload = wire.encode_activation(x_flat, prec)
        request_frame_bytes = len(request_payload) + wire.FRAME_HEADER.size
        activation_frame_bytes = len(activation_payload) + wire.FRAME_HEADER.size
        timeout = None if self.timeout_s <= 0 else self.timeout_s
        self._debug_event(
            "send_begin",
            host=self.host,
            port=self.port,
            adapter_id=self.adapter_id,
            module_name=module_name,
            rows=rows,
            in_features=in_features,
            out_features=int(out_features),
            precision=prec,
            timeout_s=self.timeout_s,
            socket_timeout=timeout,
            request_payload_bytes=len(request_payload),
            activation_payload_bytes=len(activation_payload),
            request_frame_bytes=request_frame_bytes,
            activation_frame_bytes=activation_frame_bytes,
        )
        try:
            connect_t0 = time.perf_counter()
            # One connection per call for v1 smoke servers that accept once.
            sock = socket.create_connection((self.host, self.port), timeout=timeout)
            connect_ms = (time.perf_counter() - connect_t0) * 1000.0
            sock.settimeout(timeout)
            try:
                header_send_t0 = time.perf_counter()
                wire.send_frame(sock, request_payload)
                header_send_ms = (time.perf_counter() - header_send_t0) * 1000.0
                activation_send_t0 = time.perf_counter()
                wire.send_frame(sock, activation_payload)
                activation_send_ms = (
                    time.perf_counter() - activation_send_t0
                ) * 1000.0
                send_ms = header_send_ms + activation_send_ms
                recv_t0 = time.perf_counter()
                payload = wire.recv_frame(sock)
                recv_ms = (time.perf_counter() - recv_t0) * 1000.0
                if payload[:1] == b"{":
                    try:
                        err = json.loads(payload.decode("utf-8"))
                    except Exception as decode_exc:
                        raise RuntimeError(
                            "edge worker returned non-delta JSON payload"
                        ) from decode_exc
                    raise RuntimeError(
                        "edge worker error: {0}".format(
                            err.get("error", err)
                        )
                    )
            finally:
                sock.close()
            decode_t0 = time.perf_counter()
            delta = wire.decode_delta(payload, (rows, int(out_features)), prec)
            decode_ms = (time.perf_counter() - decode_t0) * 1000.0
            roundtrip_ms = connect_ms + send_ms + recv_ms + decode_ms
            self.last_meta = {
                "request_payload_bytes": request_frame_bytes + activation_frame_bytes,
                "request_header_bytes": len(request_payload),
                "activation_payload_bytes": len(activation_payload),
                "request_frame_bytes": request_frame_bytes,
                "activation_frame_bytes": activation_frame_bytes,
                "response_payload_bytes": len(payload),
                "provider_connect_ms": connect_ms,
                "provider_header_send_ms": header_send_ms,
                "provider_activation_send_ms": activation_send_ms,
                "provider_send_ms": send_ms,
                "provider_recv_ms": recv_ms,
                "provider_decode_ms": decode_ms,
                "provider_roundtrip_ms": roundtrip_ms,
            }
            self._debug_event(
                "send_ok",
                host=self.host,
                port=self.port,
                adapter_id=self.adapter_id,
                module_name=module_name,
                request_payload_bytes=len(request_payload),
                activation_payload_bytes=len(activation_payload),
                response_payload_bytes=len(payload),
                connect_ms=connect_ms,
                header_send_ms=header_send_ms,
                activation_send_ms=activation_send_ms,
                recv_ms=recv_ms,
                decode_ms=decode_ms,
                roundtrip_ms=roundtrip_ms,
            )
            return delta.to(device=x.device, dtype=x.dtype)
        except (BrokenPipeError, TimeoutError, socket.timeout) as exc:
            self._debug_event(
                "send_error",
                host=self.host,
                port=self.port,
                adapter_id=self.adapter_id,
                module_name=module_name,
                error_type=type(exc).__name__,
                error=str(exc),
                timeout_s=self.timeout_s,
                request_payload_bytes=len(request_payload),
                activation_payload_bytes=len(activation_payload),
                request_frame_bytes=request_frame_bytes,
                activation_frame_bytes=activation_frame_bytes,
            )
            raise RuntimeError(
                "edge worker I/O timed out or closed the connection "
                "(adapter_id={0!r} module={1!r}, timeout_s={2}, "
                "activation_bytes={3}). Increase edge timeout or check "
                "worker load/network throughput.".format(
                    self.adapter_id,
                    module_name,
                    self.timeout_s,
                    len(activation_payload),
                )
            ) from exc
        except BrokenPipeError as exc:
            self._debug_event(
                "send_error",
                host=self.host,
                port=self.port,
                adapter_id=self.adapter_id,
                module_name=module_name,
                error_type=type(exc).__name__,
                error=str(exc),
                request_payload_bytes=len(request_payload),
                activation_payload_bytes=len(activation_payload),
            )
            raise RuntimeError(
                "edge worker closed the connection mid-request "
                "(adapter_id={0!r} module={1!r}). Often means the Jetson "
                "worker was started with a single --adapter-dir and does not "
                "know this adapter; restart with --adapters-root covering the "
                "full catalog.".format(self.adapter_id, module_name)
            ) from exc
        except Exception as exc:
            self._debug_event(
                "send_error",
                host=self.host,
                port=self.port,
                adapter_id=self.adapter_id,
                module_name=module_name,
                error_type=type(exc).__name__,
                error=str(exc),
                request_payload_bytes=len(request_payload),
                activation_payload_bytes=len(activation_payload),
            )
            if self.fail_closed:
                raise
            raise
