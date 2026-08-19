# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Best-effort runtime telemetry for controlled diffusion experiments."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


_TELEMETRY_DIR_ENV = "VLLM_OMNI_EXPERIMENT_TELEMETRY_DIR"
_RUN_ID_ENV = "VLLM_OMNI_EXPERIMENT_RUN_ID"


def _telemetry_dir() -> Path | None:
    value = os.environ.get(_TELEMETRY_DIR_ENV)
    if not value:
        return None
    return Path(value)


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


def emit_event(stream: str, event: str, **payload: Any) -> None:
    """Append one JSONL event when experiment telemetry is explicitly enabled."""
    directory = _telemetry_dir()
    if directory is None:
        return
    row = {
        "event": event,
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get(_RUN_ID_ENV),
        "process_id": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **payload,
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{stream}.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True, default=str))
            output.write("\n")
    except OSError:
        # Instrumentation must not change serving behavior when its output path
        # becomes unavailable.
        return


def emit_event_rows(stream: str, rows: list[dict[str, Any]]) -> None:
    """Append multiple JSONL events with one file open."""
    directory = _telemetry_dir()
    if directory is None or not rows:
        return
    common = {
        "run_id": os.environ.get(_RUN_ID_ENV),
        "process_id": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{stream}.jsonl").open(
            "a", encoding="utf-8"
        ) as output:
            for payload in rows:
                row = {
                    "monotonic_ns": time.monotonic_ns(),
                    "wall_time_utc": datetime.now(timezone.utc).isoformat(),
                    **common,
                    **payload,
                }
                output.write(json.dumps(row, sort_keys=True, default=str))
                output.write("\n")
    except OSError:
        return
