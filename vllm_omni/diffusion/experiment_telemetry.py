# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort runtime telemetry for controlled diffusion experiments."""

from __future__ import annotations

import json
import fcntl
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


_TELEMETRY_DIR_ENV = "VLLM_OMNI_EXPERIMENT_TELEMETRY_DIR"
_RUN_ID_ENV = "VLLM_OMNI_EXPERIMENT_RUN_ID"
_PHASE_ENV = "VLLM_OMNI_EXPERIMENT_PHASE"
_SUPPRESS_LORA_EVENTS_ENV = "VLLM_OMNI_SUPPRESS_LORA_EVENTS"

_PER_MODULE_LORA_EVENTS = frozenset(
    {
        "batch_tile_lora_compute",
        "lora_batch_slot_applied",
        "lora_split_path_applied",
        "mixfusion_lora_tile_compute",
        # Per-kernel JSONL (~1k/step); gate validates via mixfusion_runtime_dispatch.
        "mixfusion_lora_kernel_dispatch",
        "punica_segmented_call",
        "punica_segmented_summary",
        "sequential_adapter_applied",
        "sequential_summary",
    }
)

_PER_MODULE_TILE_EVENTS = frozenset(
    {
        "batch_base_tile_compute",
    }
)


def _telemetry_dir() -> Path | None:
    value = os.environ.get(_TELEMETRY_DIR_ENV)
    if not value:
        return None
    return Path(value)


def _suppress_lora_event(telemetry_stream: str, event: str) -> bool:
    if telemetry_stream == "lora_events":
        suppressible = event in _PER_MODULE_LORA_EVENTS
    elif telemetry_stream == "dit_events":
        suppressible = event in _PER_MODULE_TILE_EVENTS
    else:
        suppressible = False
    if not suppressible:
        return False
    return os.environ.get(_SUPPRESS_LORA_EVENTS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _current_phase(directory: Path) -> str | None:
    phase_path = directory / ".phase"
    try:
        phase = phase_path.read_text(encoding="utf-8").strip()
    except OSError:
        phase = ""
    return phase or os.environ.get(_PHASE_ENV)


def tensor_metadata(
    tensor: torch.Tensor | None,
    *,
    role: str,
) -> dict[str, Any] | None:
    """Return runtime-derived tensor metadata without copying tensor data."""
    if tensor is None:
        return None
    shape = [int(value) for value in tensor.shape]
    element_size = int(tensor.element_size())
    numel = int(tensor.numel())
    return {
        "role": role,
        "shape": shape,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": numel,
        "element_size_bytes": element_size,
        "bytes": numel * element_size,
        "tensor_signature": "|".join(
            (role, str(tensor.dtype), "x".join(str(value) for value in shape))
        ),
    }


def tensor_digest(tensor: torch.Tensor) -> dict[str, Any]:
    """Return a deterministic numeric digest for diagnostic tensor comparison."""
    values = tensor.detach().reshape(-1).float().cpu().contiguous()
    payload = values.numpy().tobytes()
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(values.numel()),
        "sha256_float32": hashlib.sha256(payload).hexdigest(),
        "mean": float(values.mean().item()) if values.numel() else 0.0,
        "absmax": float(values.abs().max().item()) if values.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(values).item()),
    }


def cuda_memory_snapshot(device: torch.device | None = None) -> dict[str, int | None]:
    """Return allocator and free-memory values for the active CUDA device."""
    if not torch.cuda.is_available():
        return {
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_max_allocated_bytes": None,
            "cuda_max_reserved_bytes": None,
            "cuda_free_bytes": None,
            "cuda_total_bytes": None,
        }
    index = (
        device.index
        if device is not None and device.index is not None
        else torch.cuda.current_device()
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        "cuda_free_bytes": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
    }


def emit_event(telemetry_stream: str, event: str, **payload: Any) -> None:
    """Append one JSONL event when experiment telemetry is explicitly enabled."""
    if _suppress_lora_event(telemetry_stream, event) and not payload.get(
        "aggregate", False
    ):
        # #region agent log
        if (
            os.environ.get("CURSOR_AGENT_DEBUG") == "68c61b"
            and event == "lora_batch_slot_applied"
            and not getattr(emit_event, "_dbg_suppress_logged", False)
        ):
            try:
                import time as _time

                with open(
                    "/workspace/.cursor/debug-68c61b.log",
                    "a",
                    encoding="utf-8",
                ) as _dbg:
                    _dbg.write(
                        json.dumps(
                            {
                                "sessionId": "68c61b",
                                "hypothesisId": "H2",
                                "location": "experiment_telemetry.py:emit_event",
                                "message": "suppressed lora_batch_slot_applied",
                                "data": {
                                    "impl": payload.get("implementation"),
                                    "applied_slots": payload.get(
                                        "applied_slot_indices"
                                    ),
                                    "module": payload.get("module_name"),
                                    "aggregate": bool(payload.get("aggregate")),
                                },
                                "timestamp": int(_time.time() * 1000),
                                "runId": "diag-pre",
                            }
                        )
                        + "\n"
                    )
                emit_event._dbg_suppress_logged = True  # type: ignore[attr-defined]
            except Exception:
                pass
        # #endregion
        return
    directory = _telemetry_dir()
    if directory is None:
        return
    row = {
        "event": event,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get(_RUN_ID_ENV),
        "phase": _current_phase(directory),
        "process_id": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **payload,
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # TP workers share one telemetry file.  Lock the descriptor around the
        # complete encoded line; without this, large JSON rows can interleave
        # at the byte level and make the evidence unreadable.
        encoded = (json.dumps(row, sort_keys=True, default=str) + "\n").encode("utf-8")
        with (directory / f"{telemetry_stream}.jsonl").open("ab", buffering=0) as output:
            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
            try:
                output.write(encoded)
            finally:
                fcntl.flock(output.fileno(), fcntl.LOCK_UN)
    except OSError:
        # Instrumentation must not change serving behavior when its output path
        # becomes unavailable.
        return


def emit_event_rows(stream: str, rows: list[dict[str, Any]]) -> None:
    """Append multiple JSONL events with one file open."""
    directory = _telemetry_dir()
    if directory is None or not rows:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{stream}.jsonl").open("ab", buffering=0) as output:
            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
            for payload in rows:
                common = {
                    "run_id": os.environ.get(_RUN_ID_ENV),
                    "phase": _current_phase(directory),
                    "process_id": os.getpid(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
                row = {
                    "monotonic_ns": time.monotonic_ns(),
                    "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                    **common,
                    **payload,
                }
                encoded = (json.dumps(row, sort_keys=True, default=str) + "\n").encode("utf-8")
                output.write(encoded)
            fcntl.flock(output.fileno(), fcntl.LOCK_UN)
    except OSError:
        return
