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
    stream: torch.cuda.Stream | None
    device: torch.device
    host_start_ns: int
    host_end_ns: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


_CONTEXT: dict[str, Any] = {}
_PENDING: list[PendingCudaInterval] = []
_MEMORY_BEFORE: dict[str, int] = {}
_ATTENTION_TOTALS: dict[str, int] = {}
_PROFILE_STEP_COUNT = 0
_PROFILE_PHASE: str | None = None



_FLOP_ENABLED_ENV = "VLLM_OMNI_EXPERIMENT_FLOPS"
_FLOP_TOTALS: dict[str, int] = {}
_NVTX_ENABLED_ENV = "VLLM_OMNI_LORA_NVTX_RANGES"
_TORCH_PROFILER_DIR_ENV = "VLLM_OMNI_LORA_TORCH_PROFILER_DIR"
_TORCH_PROFILER_MAX_STEPS_ENV = "VLLM_OMNI_LORA_TORCH_PROFILER_MAX_STEPS"
_TORCH_PROFILER_SKIP_STEPS_ENV = "VLLM_OMNI_LORA_TORCH_PROFILER_SKIP_STEPS"
_TORCH_PROFILER_DEBUG_ENV = "VLLM_OMNI_LORA_TORCH_PROFILER_DEBUG"

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


@torch.compiler.disable
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


@torch.compiler.disable
def record_attention_flops(
    *,
    batch_size: int,
    query_tokens: int,
    key_tokens: int,
    num_heads: int,
    head_dim: int,
    module_name: str | None = None,
) -> None:
    """Record analytical QK/AV FLOPs for one executed attention call.

    The fused attention kernel does not expose FLOPs through
    ``torch.profiler``.  This records the exact executed tensor geometry, so
    the runner can add QK and AV work to the profiler's linear-operation
    count without pretending that the result came from a hardware counter.
    """
    if not (flops_enabled() or enabled()):
        return
    qk_flops = 2 * int(batch_size) * int(num_heads) * int(query_tokens) * int(
        key_tokens
    ) * int(head_dim)
    _ATTENTION_TOTALS["attention_qk_flops"] = (
        _ATTENTION_TOTALS.get("attention_qk_flops", 0) + qk_flops
    )
    _ATTENTION_TOTALS["attention_av_flops"] = (
        _ATTENTION_TOTALS.get("attention_av_flops", 0) + qk_flops
    )
    _ATTENTION_TOTALS["attention_calls"] = (
        _ATTENTION_TOTALS.get("attention_calls", 0) + 1
    )
    if module_name:
        key = f"attention_flops::{module_name}"
        _ATTENTION_TOTALS[key] = _ATTENTION_TOTALS.get(key, 0) + 2 * qk_flops

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


def _nvtx_range_name(operation: str) -> str:
    if operation.startswith("lora_"):
        return f"edge_dit_lora:{operation}"
    if operation.startswith("base_"):
        return f"edge_dit_base:{operation}"
    return f"edge_dit_model:{operation}"


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
    _ATTENTION_TOTALS.clear()
    _CONTEXT.clear()
    _MEMORY_BEFORE.clear()
    _MEMORY_BEFORE.update(_cuda_memory_stats())
    _CONTEXT.update(
        scheduler_step_id=int(scheduler_step_id),
        request_ids=list(request_ids),
        request_step_indices=[int(value) for value in request_step_indices],
        actual_batch_size=int(actual_batch_size),
    )


@torch.compiler.disable
def start_interval(
    operation: str,
    cuda_device: torch.device | int | None = None,
    **payload: Any,
) -> PendingCudaInterval | None:
    """Record a CUDA start marker without synchronizing."""
    if not enabled() or not operation_enabled(operation):
        return None
    if os.environ.get("VLLM_OMNI_DISABLE_LORA_CUDA_EVENTS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    module_name = payload.get("module_name")
    payload.setdefault("block_name", infer_block_name(module_name))
    device = torch.device(
        cuda_device if cuda_device is not None else torch.cuda.current_device()
    )
    with torch.cuda.device(device):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream()
    interval = PendingCudaInterval(
        start_event=start_event,
        end_event=end_event,
        stream=stream,
        device=device,
        host_start_ns=time.monotonic_ns(),
        payload={"operation": operation, **payload},
    )
    if _truthy_env(_NVTX_ENABLED_ENV):
        nvtx_range = _nvtx_range_name(operation)
        torch.cuda.nvtx.range_push(nvtx_range)
        interval.payload["nvtx_range"] = nvtx_range
    start_event.record(stream)
    return interval


@torch.compiler.disable
def finish_interval(
    interval: PendingCudaInterval | None,
    **payload: Any,
) -> None:
    """Record a CUDA end marker and defer duration resolution."""
    if interval is None:
        return
    if (
        interval.payload.get("operation") == "attention"
        and _truthy_env("VLLM_OMNI_BREAKDOWN_ATTENTION_SYNC")
    ):
        torch.cuda.synchronize(interval.device)
    interval.payload.update(payload)
    end_stream = interval.stream
    if interval.payload.get("operation") == "attention":
        end_stream = torch.cuda.current_stream(interval.device)
    interval.end_event.record(end_stream)
    if interval.payload.get("nvtx_range"):
        torch.cuda.nvtx.range_pop()
    interval.host_end_ns = time.monotonic_ns()
    _PENDING.append(interval)


@contextmanager
def maybe_profile_denoise_step():
    """Capture one optional high-detail PyTorch profiler trace per process."""
    global _PROFILE_PHASE, _PROFILE_STEP_COUNT

    trace_dir_value = os.environ.get(_TORCH_PROFILER_DIR_ENV, "").strip()
    debug_enabled = _truthy_env(_TORCH_PROFILER_DEBUG_ENV)
    cuda_available = torch.cuda.is_available() if trace_dir_value or debug_enabled else False
    if debug_enabled:
        emit_event_rows(
            "lora_torch_profiler_gate",
            [
                {
                    "event": "lora_torch_profiler_gate",
                    "trace_dir_configured": bool(trace_dir_value),
                    "cuda_available": cuda_available,
                    "step_index": _PROFILE_STEP_COUNT,
                    "process_id": os.getpid(),
                }
            ],
        )
    if not trace_dir_value or not cuda_available:
        yield
        return

    if not (Path(trace_dir_value) / ".active").is_file():
        yield
        return

    try:
        profile_phase = (Path(trace_dir_value) / ".." / ".phase").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        profile_phase = ""
    if profile_phase != _PROFILE_PHASE:
        _PROFILE_PHASE = profile_phase
        _PROFILE_STEP_COUNT = 0

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
        with_flops=True,
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
                "profiler_flops": (
                    int(event.flops)
                    if getattr(event, "flops", None)
                    else None
                ),
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
    if not _PENDING and not _FLOP_TOTALS and not _ATTENTION_TOTALS:
        return 0
    rows: list[dict[str, Any]] = []
    for sequence_index, interval in enumerate(_PENDING):
        cuda_duration_error = None
        try:
            with torch.cuda.device(interval.device):
                cuda_duration_ms = float(
                    interval.start_event.elapsed_time(interval.end_event)
                )
        except RuntimeError as error:
            cuda_duration_ms = None
            cuda_duration_error = str(error)
        if cuda_duration_ms is not None and (
            cuda_duration_ms < 0.0 or cuda_duration_ms != cuda_duration_ms
        ):
            cuda_duration_ms = None
        host_end_ns = interval.host_end_ns or interval.host_start_ns
        rows.append(
            {
                "event": "lora_compute_breakdown_interval",
                "sequence_index": sequence_index,
                "cuda_duration_ms": cuda_duration_ms,
                "cuda_duration_error": cuda_duration_error,
                "host_enqueue_duration_ms": (
                    host_end_ns - interval.host_start_ns
                )
                / 1_000_000,
                **_CONTEXT,
                **interval.payload,
            }
        )
    lora_path_intervals = [
        row
        for row in rows
        if row.get("event") == "lora_compute_breakdown_interval"
        and row.get("operation") == "lora_path_e2e"
        and row.get("execution_location")
        in {"multi_lora_operator", "served_lora_only"}
    ]
    valid_lora_path_ms = [
        float(row["cuda_duration_ms"])
        for row in lora_path_intervals
        if isinstance(row.get("cuda_duration_ms"), (int, float))
    ]
    operation_totals: dict[str, float] = {}
    for row in rows:
        operation = row.get("operation")
        duration = row.get("cuda_duration_ms")
        if isinstance(operation, str) and isinstance(duration, (int, float)):
            operation_totals[operation] = operation_totals.get(operation, 0.0) + float(
                duration
            )
    rows.append(
        {
            "event": "lora_path_cuda_step",
            "lora_path_cuda_event_ms": sum(valid_lora_path_ms),
            "lora_path_interval_count": len(lora_path_intervals),
            "lora_path_invalid_interval_count": (
                len(lora_path_intervals) - len(valid_lora_path_ms)
            ),
            "lora_path_execution_locations": sorted(
                {
                    str(row["execution_location"])
                    for row in lora_path_intervals
                    if row.get("execution_location")
                }
            ),
            **_CONTEXT,
        }
    )
    rows.append(
        {
            "event": "lora_compute_breakdown_step",
            "base_ms": operation_totals.get("base_gemm", 0.0),
            "lora_or_patch_ms": operation_totals.get("lora_path_e2e", 0.0),
            "attention_ms": operation_totals.get("attention", 0.0),
            "merge_or_barrier_ms": operation_totals.get(
                "merge_or_barrier", 0.0
            ),
            "merge_or_barrier_status": "fused_into_lora_path",
            "operation_totals_ms": operation_totals,
            **_CONTEXT,
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
    if _ATTENTION_TOTALS:
        rows.append(
            {
                "event": "attention_flops_step",
                "attention_flops": (
                    _ATTENTION_TOTALS.get("attention_qk_flops", 0)
                    + _ATTENTION_TOTALS.get("attention_av_flops", 0)
                ),
                **_ATTENTION_TOTALS,
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
    _ATTENTION_TOTALS.clear()
    _CONTEXT.clear()
    _MEMORY_BEFORE.clear()
    return count
