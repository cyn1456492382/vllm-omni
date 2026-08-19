# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime helpers for DiT-LoRA overlap experiments.

This module intentionally keeps the overlap experiment wiring out of the core
scheduler.  It stores request-scoped granularity, exports runtime-only module
inventories from instantiated models, and records lightweight wall-clock / CUDA
intervals for later CSV reporting.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_OVERLAP_GRANULARITY: ContextVar[int] = ContextVar("dit_lora_overlap_granularity", default=0)
_TASK_TRACE_ENABLED: ContextVar[bool] = ContextVar("dit_lora_task_trace_enabled", default=False)


@dataclass
class TaskInterval:
    name: str
    kind: str
    start_ms: float
    end_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def close(self, end_ms: float | None = None) -> None:
        self.end_ms = float(time.perf_counter() * 1000.0 if end_ms is None else end_ms)

    @property
    def duration_ms(self) -> float:
        if self.end_ms is None:
            return 0.0
        return max(0.0, self.end_ms - self.start_ms)


@dataclass
class DitLoRAOverlapRuntime:
    granularity: int = 0
    request_id: str | None = None
    batch_request_ids: tuple[str, ...] = ()
    batch_repeat: int = 1
    step_index: int = 0
    block_index: int = 0
    module_inventory: dict[str, list[str]] = field(default_factory=dict)
    task_intervals: list[TaskInterval] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def push_task(self, name: str, kind: str, **metadata: Any) -> TaskInterval:
        task = TaskInterval(name=name, kind=kind, start_ms=time.perf_counter() * 1000.0, metadata=dict(metadata))
        self.task_intervals.append(task)
        return task

    def close_task(self, task: TaskInterval) -> None:
        task.close()

    def record_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "batch_request_ids": list(self.batch_request_ids),
            "batch_repeat": int(self.batch_repeat),
            "granularity": int(self.granularity),
            "step_index": int(self.step_index),
            "block_index": int(self.block_index),
            "module_inventory": self.module_inventory,
            "task_intervals": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "duration_ms": item.duration_ms,
                    "metadata": item.metadata,
                }
                for item in self.task_intervals
            ],
            "metrics": self.metrics,
        }


_RUNTIME: DitLoRAOverlapRuntime = DitLoRAOverlapRuntime()


def runtime() -> DitLoRAOverlapRuntime:
    return _RUNTIME


def set_overlap_granularity(value: int) -> None:
    granularity = int(value)
    if granularity < 0 or granularity > 3:
        raise ValueError("overlap granularity must be in [0, 3]")
    _OVERLAP_GRANULARITY.set(granularity)
    _RUNTIME.granularity = granularity


def configure_overlap_runtime(*, enabled: bool, granularity: int) -> None:
    """Configure request-scoped overlap profiling for the active process."""
    set_overlap_granularity(granularity)
    _TASK_TRACE_ENABLED.set(bool(enabled))
    if not enabled:
        _RUNTIME.task_intervals = []
        _RUNTIME.metrics = {}


def overlap_granularity() -> int:
    return int(_OVERLAP_GRANULARITY.get())


@contextlib.contextmanager
def task_trace_enabled(enabled: bool = True) -> Iterator[None]:
    token = _TASK_TRACE_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _TASK_TRACE_ENABLED.reset(token)


@contextlib.contextmanager
def runtime_scope(request_id: str | None = None) -> Iterator[DitLoRAOverlapRuntime]:
    prev_request_id = _RUNTIME.request_id
    prev_step_index = _RUNTIME.step_index
    prev_block_index = _RUNTIME.block_index
    prev_intervals = list(_RUNTIME.task_intervals)
    prev_metrics = dict(_RUNTIME.metrics)
    try:
        if request_id is not None:
            _RUNTIME.request_id = request_id
        _RUNTIME.step_index = 0
        _RUNTIME.block_index = 0
        _RUNTIME.task_intervals = []
        _RUNTIME.metrics = {}
        yield _RUNTIME
    finally:
        _RUNTIME.request_id = prev_request_id
        _RUNTIME.step_index = prev_step_index
        _RUNTIME.block_index = prev_block_index
        _RUNTIME.task_intervals = prev_intervals
        _RUNTIME.metrics = prev_metrics


def task_tracing_enabled() -> bool:
    return bool(_TASK_TRACE_ENABLED.get())


@contextlib.contextmanager
def cuda_task(name: str, kind: str, **metadata: Any) -> Iterator[None]:
    task = _RUNTIME.push_task(name, kind, **metadata)
    try:
        yield
    finally:
        _RUNTIME.close_task(task)


@contextlib.contextmanager
def host_task(name: str, kind: str, **metadata: Any) -> Iterator[None]:
    task = _RUNTIME.push_task(name, f"host:{kind}", **metadata)
    try:
        yield
    finally:
        _RUNTIME.close_task(task)


class _ModulePathCollector:
    def __init__(self) -> None:
        self.paths: list[str] = []

    def visit(self, root: torch.nn.Module, prefix: str) -> None:
        for name, _module in root.named_modules(remove_duplicate=False):
            full_name = f"{prefix}.{name}" if name else prefix
            self.paths.append(full_name)


def export_module_inventory(pipeline: Any) -> dict[str, list[str]]:
    """Reflect the instantiated pipeline, never the filesystem.

    The caller is expected to already have a real pipeline instance. We only
    walk instantiated modules and return their names for later LoRA mapping.
    """
    inventory: dict[str, list[str]] = {}
    for component_name in ("text_encoder", "transformer", "vae"):
        component = getattr(pipeline, component_name, None)
        if isinstance(component, torch.nn.Module):
            collector = _ModulePathCollector()
            collector.visit(component, component_name)
            inventory[component_name] = collector.paths
    return inventory


def save_module_inventory(path: str | Path, inventory: dict[str, list[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")


def active_overlap_runtime() -> DitLoRAOverlapRuntime:
    return _RUNTIME
