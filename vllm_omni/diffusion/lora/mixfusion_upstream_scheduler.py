"""Pinned MixFusion scheduler method adapted to Omni request metadata.

Source: MixFusion e9aa899a1bd0885c100cc9b16011d987d3be4895,
``sduss/worker/scheduler/scheduler.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

MIXFUSION_SOURCE_COMMIT = "e9aa899a1bd0885c100cc9b16011d987d3be4895"
MIXFUSION_SCHEDULER_ENTRY = "sduss.worker.scheduler.scheduler.Scheduler.schedule"


@dataclass(frozen=True)
class SchedulerRequest:
    request_index: int
    adapter_signature: tuple[int, ...]


@dataclass(frozen=True)
class SchedulerOutput:
    request_indices: tuple[int, ...]
    adapter_signature: tuple[int, ...] | None


def _allow_heterogeneous_lora() -> bool:
    return os.getenv("MIXFUSION_MULTILORA_ALLOW_HETEROGENEOUS", "0") == "1"


def _treatment_b_enabled() -> bool:
    return os.getenv("MIXFUSION_MULTILORA_TREATMENT_B", "0") == "1"


def is_heterogeneous_mode() -> bool:
    """Heterogeneous only when BOTH ALLOW=1 and TREATMENT_B=1."""
    return _allow_heterogeneous_lora() and _treatment_b_enabled()


class _Policy:
    def __init__(self, requests: list[SchedulerRequest]) -> None:
        self.requests = requests

    def schedule_requests(self, max_num: int) -> SchedulerOutput:
        """Select up to ``max_num`` requests and remove them from the queue.

        Homogeneous mode (default): keep the first request's adapter signature
        and only include later requests that share it. Unselected requests stay
        in ``self.requests`` for a later cycle.
        Heterogeneous mode: take up to ``max_num`` requests and mark the batch
        heterogeneous via ``adapter_signature=None``.
        """
        if max_num <= 0 or not self.requests:
            return SchedulerOutput(request_indices=(), adapter_signature=None)
        if is_heterogeneous_mode():
            selected = self.requests[: int(max_num)]
            del self.requests[: len(selected)]
            return SchedulerOutput(
                request_indices=tuple(item.request_index for item in selected),
                adapter_signature=None,
            )
        seed = self.requests[0]
        selected = [seed]
        remaining: list[SchedulerRequest] = []
        for item in self.requests[1:]:
            if (
                len(selected) < int(max_num)
                and item.adapter_signature == seed.adapter_signature
            ):
                selected.append(item)
            else:
                remaining.append(item)
        self.requests = remaining
        return SchedulerOutput(
            request_indices=tuple(item.request_index for item in selected),
            adapter_signature=seed.adapter_signature,
        )


def _restrict_to_adapter_signature(
    output: SchedulerOutput,
    selected_signature: tuple[int, ...] | None,
    requests_snapshot: tuple[SchedulerRequest, ...],
) -> SchedulerOutput:
    """Drop any selected request that does not match the batch signature."""
    if selected_signature is None:
        return output
    allowed = {
        item.request_index
        for item in requests_snapshot
        if item.adapter_signature == selected_signature
    }
    filtered = tuple(
        index for index in output.request_indices if index in allowed
    )
    if not filtered:
        return SchedulerOutput(request_indices=(), adapter_signature=None)
    return SchedulerOutput(
        request_indices=filtered,
        adapter_signature=selected_signature,
    )


def _source_scheduler_schedule(self) -> SchedulerOutput:
    """Execute the pinned upstream ``Scheduler.schedule`` method body."""
    self.cycle_counter += 1
    # Snapshot before mutation for TREATMENT_B / restrict checks.
    snapshot = tuple(self.policy.requests)
    scheduler_output = self.policy.schedule_requests(max_num=self.max_batchsize)
    if is_heterogeneous_mode():
        return SchedulerOutput(
            request_indices=scheduler_output.request_indices,
            adapter_signature=None,
        )
    return _restrict_to_adapter_signature(
        scheduler_output, scheduler_output.adapter_signature, snapshot
    )


class NativeMixFusionScheduler:
    """Minimal source-compatible scheduler host for one Omni denoise batch."""

    def __init__(self, requests: Iterable[SchedulerRequest]) -> None:
        request_tuple = tuple(requests)
        seen: set[int] = set()
        for item in request_tuple:
            if item.request_index in seen:
                raise ValueError(
                    f"duplicate MixFusion request_index={item.request_index}"
                )
            seen.add(item.request_index)
        self.max_batchsize = len(request_tuple)
        self.policy = _Policy(list(request_tuple))
        self.cycle_counter = 0
        # Caller-owned map from request_index → full adapter signature.
        self.request_signatures = {
            item.request_index: item.adapter_signature for item in request_tuple
        }

    schedule = _source_scheduler_schedule

    def schedule_all_cycles(self) -> tuple[SchedulerOutput, ...]:
        """Drain the queue into homogeneous (or one hetero) cycles.

        Homogeneous ``(A,B,A)`` yields ``(0,2)`` then ``(1)``. Heterogeneous
        mode yields a single cycle covering all remaining requests.
        """
        cycles: list[SchedulerOutput] = []
        while self.policy.requests:
            output = self.schedule()
            if not output.request_indices:
                break
            cycles.append(output)
        return tuple(cycles)


__all__ = [
    "MIXFUSION_SCHEDULER_ENTRY",
    "MIXFUSION_SOURCE_COMMIT",
    "NativeMixFusionScheduler",
    "SchedulerOutput",
    "SchedulerRequest",
    "is_heterogeneous_mode",
]
