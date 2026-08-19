# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred CUDA Event profiling for diffusion LoRA experiments."""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.experiment_telemetry import emit_event_rows

_ENABLED_ENV = "VLLM_OMNI_LORA_COMPUTE_BREAKDOWN"
_OPERATIONS_ENV = "VLLM_OMNI_LORA_COMPUTE_BREAKDOWN_OPERATIONS"
_BLOCK_PATTERN = re.compile(
    r"(?:^|\.)(noise_refiner|context_refiner|layers)\.(\d+)(?:\.|$)"
)


@dataclass
class PendingCudaInterval:
    """One interval resolved after the enclosing denoise synchronization."""

    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    host_start_ns: int
    host_end_ns: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


_CONTEXT: dict[str, Any] = {}
_PENDING: list[PendingCudaInterval] = []
_MEMORY_BEFORE: dict[str, int] = {}
_PROFILE_STEP_COUNT = 0



_FLOP_ENABLED_ENV = "VLLM_OMNI_EXPERIMENT_FLOPS"
_FLOP_TOTALS: dict[str, int] = {}
_NVTX_ENABLED_ENV = "VLLM_OMNI_LORA_NVTX_RANGES"
_TORCH_PROFILER_DIR_ENV = "VLLM_OMNI_LORA_TORCH_PROFILER_DIR"
_TORCH_PROFILER_MAX_STEPS_ENV = "VLLM_OMNI_LORA_TORCH_PROFILER_MAX_STEPS"
_TORCH_PROFILER_SKIP_STEPS_ENV = "VLLM_OMNI_LORA_TORCH_PROFILER_SKIP_STEPS"

_MEMORY_STAT_KEYS = (
    "active_bytes.all.current",
    "allocated_bytes.all.current",
    "reserved_bytes.all.current",
    "inactive_split_bytes.all.current",
    "segment.all.current",
    "allocation.all.current",
    "num_alloc_retries",
    "num_ooms",
    "num_sync_all_streams",
)


def flops_enabled() -> bool:
    """Return whether lightweight per-step FLOP accounting is enabled."""
    return os.environ.get(_FLOP_ENABLED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def record_lora_flops(
    *,
    lora_a_flops: int,
    lora_b_flops: int,
    residual_add_flops: int,
) -> None:
    """Accumulate local LoRA FLOPs without per-module telemetry writes."""
    if not flops_enabled():
        return
    _FLOP_TOTALS["lora_a_flops"] = _FLOP_TOTALS.get("lora_a_flops", 0) + int(
        lora_a_flops
    )
    _FLOP_TOTALS["lora_b_flops"] = _FLOP_TOTALS.get("lora_b_flops", 0) + int(
        lora_b_flops
    )
    _FLOP_TOTALS["residual_add_flops"] = _FLOP_TOTALS.get(
        "residual_add_flops", 0
    ) + int(residual_add_flops)

def enabled() -> bool:
    """Return whether breakdown profiling is explicitly enabled."""
    value = os.environ.get(_ENABLED_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"} and torch.cuda.is_available()


def _truthy_env(name: str) -> bool:
    """Return whether an environment variable is explicitly enabled."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _cuda_memory_stats() -> dict[str, int]:
    """Return allocator counters needed to diagnose fragmentation and retries."""
    if not torch.cuda.is_available():
        return {}
    stats = torch.cuda.memory_stats()
    return {key: int(stats.get(key, 0)) for key in _MEMORY_STAT_KEYS}


def operation_enabled(operation: str) -> bool:
    """Return whether one operation is included by the optional filter."""
    configured = os.environ.get(_OPERATIONS_ENV, "").strip()
    if not configured:
        return True
    allowed = {item.strip() for item in configured.split(",") if item.strip()}
    return operation in allowed


def infer_block_name(module_name: str | None) -> str | None:
    """Derive a stable Z-Image block name from a runtime module path."""
    if not module_name:
        return None
    match = _BLOCK_PATTERN.search(module_name)
    if match is None:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def begin_step(
    *,
    scheduler_step_id: int,
    request_ids: list[str],
    request_step_indices: list[int],
    actual_batch_size: int,
) -> None:
    """Start one denoise-step collection context."""
    if not enabled() and not flops_enabled():
        return
    _PENDING.clear()
    _FLOP_TOTALS.clear()
    _CONTEXT.clear()
    _MEMORY_BEFORE.clear()
    _MEMORY_BEFORE.update(_cuda_memory_stats())
    _CONTEXT.update(
        scheduler_step_id=int(scheduler_step_id),
        request_ids=list(request_ids),
        request_step_indices=[int(value) for value in request_step_indices],
        actual_batch_size=int(actual_batch_size),
    )


def start_interval(operation: str, **payload: Any) -> PendingCudaInterval | None:
    """Record a CUDA start marker without synchronizing."""
    if not enabled() or not operation_enabled(operation):
        return None
    module_name = payload.get("module_name")
    payload.setdefault("block_name", infer_block_name(module_name))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    interval = PendingCudaInterval(
        start_event=start_event,
        end_event=end_event,
        host_start_ns=time.monotonic_ns(),
        payload={"operation": operation, **payload},
    )
    if _truthy_env(_NVTX_ENABLED_ENV):
        torch.cuda.nvtx.range_push(f"edge_dit_lora:{operation}")
        interval.payload["nvtx_range"] = f"edge_dit_lora:{operation}"
    start_event.record()
    return interval


def finish_interval(
    interval: PendingCudaInterval | None,
    **payload: Any,
) -> None:
    """Record a CUDA end marker and defer duration resolution."""
    if interval is None:
        return
    interval.payload.update(payload)
    interval.end_event.record()
    if interval.payload.get("nvtx_range"):
        torch.cuda.nvtx.range_pop()
    interval.host_end_ns = time.monotonic_ns()
    _PENDING.append(interval)


@contextmanager
def maybe_profile_denoise_step():
    """Capture one optional high-detail PyTorch profiler trace per process."""
    global _PROFILE_STEP_COUNT

    trace_dir_value = os.environ.get(_TORCH_PROFILER_DIR_ENV, "").strip()
    if not trace_dir_value or not torch.cuda.is_available():
        yield
        return

    skip_steps = int(os.environ.get(_TORCH_PROFILER_SKIP_STEPS_ENV, "0"))
    max_steps = int(os.environ.get(_TORCH_PROFILER_MAX_STEPS_ENV, "1"))
    step_index = _PROFILE_STEP_COUNT
    _PROFILE_STEP_COUNT += 1
    if step_index < skip_steps or step_index >= skip_steps + max_steps:
        yield
        return

    trace_dir = Path(trace_dir_value)
    trace_dir.mkdir(parents=True, exist_ok=True)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )
    with profiler:
        yield

    trace_path = trace_dir / f"denoise_step_{step_index:04d}_pid_{os.getpid()}.json"
    profiler.export_chrome_trace(str(trace_path))
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages(group_by_input_shape=True):
        cuda_time_us = getattr(event, "self_device_time_total", None)
        if cuda_time_us is None:
            cuda_time_us = getattr(event, "self_cuda_time_total", 0.0)
        cuda_memory = getattr(event, "self_device_memory_usage", None)
        if cuda_memory is None:
            cuda_memory = getattr(event, "self_cuda_memory_usage", 0)
        rows.append(
            {
                "event": "lora_torch_profiler_operator",
                "operation_name": event.key,
                "count": int(event.count),
                "self_cpu_time_us": float(getattr(event, "self_cpu_time_total", 0.0)),
                "cpu_time_us": float(getattr(event, "cpu_time_total", 0.0)),
                "self_cuda_time_us": float(cuda_time_us or 0.0),
                "cuda_time_us": float(
                    getattr(event, "device_time_total", 0.0)
                    or getattr(event, "cuda_time_total", 0.0)
                ),
                "self_cuda_memory_bytes": int(cuda_memory or 0),
                "cuda_memory_bytes": int(
                    getattr(event, "device_memory_usage", 0)
                    or getattr(event, "cuda_memory_usage", 0)
                ),
                "input_shapes": str(getattr(event, "input_shapes", "")),
                "trace_path": str(trace_path),
                **_CONTEXT,
            }
        )
    emit_event_rows("lora_torch_profiler", rows)


def flush_step() -> int:
    """Resolve and emit intervals after the denoise stream is synchronized."""
    if not _PENDING and not _FLOP_TOTALS:
        return 0
    rows: list[dict[str, Any]] = []
    for sequence_index, interval in enumerate(_PENDING):
        try:
            cuda_duration_ms = float(
                interval.start_event.elapsed_time(interval.end_event)
            )
        except RuntimeError:
            cuda_duration_ms = None
        host_end_ns = interval.host_end_ns or interval.host_start_ns
        rows.append(
            {
                "event": "lora_compute_breakdown_interval",
                "sequence_index": sequence_index,
                "cuda_duration_ms": cuda_duration_ms,
                "host_enqueue_duration_ms": (
                    host_end_ns - interval.host_start_ns
                )
                / 1_000_000,
                **_CONTEXT,
                **interval.payload,
            }
        )
    if _FLOP_TOTALS:
        rows.append(
            {
                "event": "lora_compute_flops_step",
                "total_lora_flops": sum(_FLOP_TOTALS.values()),
                **_FLOP_TOTALS,
                **_CONTEXT,
            }
        )
    memory_after = _cuda_memory_stats()
    if _MEMORY_BEFORE or memory_after:
        rows.append(
            {
                "event": "lora_compute_memory_step",
                "before": _MEMORY_BEFORE,
                "after": memory_after,
                "delta": {
                    key: int(memory_after.get(key, 0))
                    - int(_MEMORY_BEFORE.get(key, 0))
                    for key in _MEMORY_STAT_KEYS
                },
                **_CONTEXT,
            }
        )
    emit_event_rows("lora_compute_breakdown_events", rows)
    count = len(rows)
    _PENDING.clear()
    _FLOP_TOTALS.clear()
    _CONTEXT.clear()
    _MEMORY_BEFORE.clear()
    return count
