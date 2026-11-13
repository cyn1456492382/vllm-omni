"""Source-faithful MixFusion patch scheduling bridge.

The pinned MixFusion source schedules a request batch first and then splits the
denoising input in ``PatchTransformer2DModel`` using a deterministic row-major
patch walk.  This bridge keeps that executable ordering contract in Omni while
leaving LoRA math to Omni's registered Triton operator.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch

from vllm_omni.diffusion.lora.mixfusion_upstream_scheduler import (
    NativeMixFusionScheduler,
    SchedulerRequest,
    is_heterogeneous_mode,
)

MIXFUSION_SOURCE_COMMIT = "e9aa899a1bd0885c100cc9b16011d987d3be4895"
MIXFUSION_SCHEDULER_ENTRY = "sduss.worker.scheduler.scheduler.Scheduler.schedule"
MIXFUSION_PATCH_ENTRY = (
    "sduss.model_executor.modules.transformer.PatchTransformer2DModel.forward"
)
MIXFUSION_SPLIT_ENTRY = "sduss.model_executor.modules.utils.split_sample"


@dataclass(frozen=True)
class NativePatchWork:
    """One patch work item emitted by the source patch walk."""

    request_index: int
    patch_index: int
    row_start: int
    column_start: int
    valid_height: int
    valid_width: int
    grid_height: int
    grid_width: int


@dataclass(frozen=True)
class NativePatchSchedule:
    """Patch schedule returned by the source-compatible bridge."""

    work_items: tuple[NativePatchWork, ...]
    patch_height: int
    patch_width: int
    overlap: int
    request_grid_shapes: tuple[tuple[int, int], ...]
    scheduler_entry: str = MIXFUSION_SCHEDULER_ENTRY
    patch_entry: str = MIXFUSION_PATCH_ENTRY
    scheduler_policy: str = "fcfs_mixed_source_order_source_order_adapter"
    source_scheduler_called: bool = False
    source_split_called: bool = False
    source_split_entry: str = MIXFUSION_SPLIT_ENTRY
    source_split_input_shape: tuple[int, ...] | None = None
    source_split_output_shape: tuple[int, ...] | None = None
    source_split_patch_count: int = 0
    # Homogeneous sub-batch boundaries: ((0, 2), (1)) for (A,B,A).
    # Heterogeneous mode uses a single cycle covering every request.
    cycle_request_indices: tuple[tuple[int, ...], ...] = ()
    homogeneous: bool = True

    @property
    def patch_count(self) -> int:
        return len(self.work_items)

    def event_payload(self) -> dict[str, object]:
        return {
            "mixfusion_native_scheduler_called": self.source_scheduler_called,
            "mixfusion_native_scheduler_adapter_called": True,
            "mixfusion_native_scheduler_entry": self.scheduler_entry,
            "mixfusion_native_patch_entry": self.patch_entry,
            "mixfusion_native_source_commit": MIXFUSION_SOURCE_COMMIT,
            "mixfusion_native_scheduler_policy": self.scheduler_policy,
            "mixfusion_native_split_called": self.source_split_called,
            "mixfusion_native_split_consumed": False,
            "mixfusion_native_split_mode": (
                "verify_only" if self.source_split_called else "skipped"
            ),
            "mixfusion_native_split_entry": self.source_split_entry,
            "mixfusion_native_split_input_shape": (
                list(self.source_split_input_shape)
                if self.source_split_input_shape is not None
                else None
            ),
            "mixfusion_native_split_output_shape": (
                list(self.source_split_output_shape)
                if self.source_split_output_shape is not None
                else None
            ),
            "mixfusion_native_split_patch_count": self.source_split_patch_count,
            "mixfusion_native_patch_count": len(self.work_items),
            "mixfusion_native_patch_order": [
                [item.request_index, item.patch_index]
                for item in self.work_items
            ],
            "mixfusion_cycle_request_indices": [
                list(cycle) for cycle in self.cycle_request_indices
            ],
            "mixfusion_homogeneous": self.homogeneous,
        }


def _axis_starts(length: int, patch_size: int, overlap: int) -> tuple[int, ...]:
    if length <= patch_size:
        return (0,)
    stride = patch_size - overlap
    return tuple(range(0, length, stride))


def schedule_patch_work(
    request_grid_shapes: Iterable[tuple[int, int]],
    *,
    patch_height: int,
    patch_width: int,
    overlap: int,
    source_samples: dict[int, torch.Tensor] | None = None,
    source_indices: dict[str, list[str]] | None = None,
    adapter_signatures: Iterable[tuple[int, ...]] | None = None,
) -> NativePatchSchedule:
    """Execute the pinned MixFusion request-to-patch scheduling contract.

    This is the FLUX token-grid form of the source ``split_sample`` walk:
    requests are consumed in scheduler order and each request is traversed in
    row-major patch order.  No host-side tensor launch is performed here; the
    returned descriptors are consumed by the packed device path.
    """
    shapes = tuple((int(height), int(width)) for height, width in request_grid_shapes)
    if not shapes or any(height <= 0 or width <= 0 for height, width in shapes):
        raise ValueError("MixFusion patch scheduler requires positive grids")
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("MixFusion patch dimensions must be positive")
    if overlap < 0 or overlap >= min(patch_height, patch_width):
        raise ValueError("MixFusion patch overlap is out of range")

    source_scheduler_called = False
    source_split_called = False
    source_split_input_shape = None
    source_split_output_shape = None
    source_split_patch_count = 0
    signatures = tuple(adapter_signatures or ())
    if signatures and len(signatures) != len(shapes):
        raise ValueError(
            "MixFusion adapter signatures must match request grid shapes"
        )
    scheduler = NativeMixFusionScheduler(
        SchedulerRequest(
            request_index=request_index,
            adapter_signature=(
                signatures[request_index]
                if signatures
                else (request_index,)
            ),
        )
        for request_index in range(len(shapes))
    )
    # Drain every homogeneous cycle (or one hetero cycle). Do not drop
    # unselected requests from a single schedule() call.
    cycles = scheduler.schedule_all_cycles()
    source_scheduler_called = True
    homogeneous = not is_heterogeneous_mode()
    cycle_request_indices = tuple(
        tuple(int(index) for index in cycle.request_indices) for cycle in cycles
    )
    covered = [index for cycle in cycle_request_indices for index in cycle]
    if sorted(covered) != list(range(len(shapes))) or len(covered) != len(shapes):
        raise RuntimeError(
            "MixFusion scheduler missed or duplicated requests: "
            f"covered={covered} expected={list(range(len(shapes)))}"
        )
    # Execution order follows cycle boundaries; work_items stay grouped.
    request_order = tuple(covered)
    if os.environ.get("VLLM_OMNI_MIXFUSION_NATIVE_SOURCE_CALL", "")\
        .strip().lower() in {"1", "true", "yes", "on"}:
        from vllm_omni.diffusion.lora.mixfusion_upstream_split import (
            split_sample,
        )

        if source_samples is None or source_indices is None:
            raise RuntimeError(
                "MixFusion source split requires real BCHW samples and request "
                "indices; synthetic tensors are not admissible"
            )
        if overlap != 0 or any(height != width for height, width in shapes):
            raise ValueError(
                "pinned MixFusion split_sample requires square, non-overlapping "
                "patch geometry"
            )
        # Default: one source patch per request (full image). Historical M*
        # baselines did not pay for multi-patch split_sample. Opt into
        # finer splits with VLLM_OMNI_MIXFUSION_SOURCE_PATCH_SIZE (e.g. 256).
        default_source_patch = max(int(height) * 16 for height, _width in shapes)
        source_patch_env = os.environ.get("VLLM_OMNI_MIXFUSION_SOURCE_PATCH_SIZE")
        source_patch_size = int(
            source_patch_env if source_patch_env else default_source_patch
        )
        source_samples = {
            int(resolution): samples
            for resolution, samples in source_samples.items()
        }
        source_indices = {
            str(resolution): [str(request_id) for request_id in request_ids]
            for resolution, request_ids in source_indices.items()
        }
        if not source_samples:
            raise ValueError("MixFusion source split received no real samples")
        for resolution, samples in source_samples.items():
            if samples.ndim != 4 or samples.shape[0] == 0:
                raise ValueError(
                    "MixFusion source split expects non-empty BCHW samples: "
                    f"resolution={resolution} shape={tuple(samples.shape)}"
                )
            if samples.shape[2] != samples.shape[3]:
                raise ValueError(
                    "MixFusion source split requires square BCHW samples: "
                    f"resolution={resolution} shape={tuple(samples.shape)}"
                )
            if resolution % source_patch_size:
                raise ValueError(
                    "pinned MixFusion split_sample requires image resolution "
                    "divisible by the source patch size"
                )
        source_split_input_shape = tuple(
            int(value) for value in next(iter(source_samples.values())).shape
        )
        source_result = split_sample(
            source_samples, source_patch_size, source_indices
        )
        source_patch_ids = source_result[0]
        expected_patch_count = sum(
            (height * 16 // source_patch_size)
            * (width * 16 // source_patch_size)
            for height, width in shapes
        )
        if len(source_patch_ids) != expected_patch_count:
            raise RuntimeError(
                "MixFusion split_sample patch count mismatch: "
                f"got={len(source_patch_ids)} expected={expected_patch_count}"
            )
        # M1: split_sample is verify-only unless a consumer path is wired.
        # Latents still come from the original packed sequence; do not claim
        # patch-layout reproduction from source_split_called alone.
        _ = source_result[4]
        source_split_patch_count = len(source_patch_ids)
        source_split_output_shape = tuple(
            int(value) for value in source_result[4].shape
        )
        source_split_called = True

    work_items: list[NativePatchWork] = []
    for request_index in request_order:
        grid_height, grid_width = shapes[request_index]
        patch_index = 0
        for row_start in _axis_starts(grid_height, patch_height, overlap):
            for column_start in _axis_starts(grid_width, patch_width, overlap):
                work_items.append(
                    NativePatchWork(
                        request_index=request_index,
                        patch_index=patch_index,
                        row_start=row_start,
                        column_start=column_start,
                        valid_height=min(patch_height, grid_height - row_start),
                        valid_width=min(patch_width, grid_width - column_start),
                        grid_height=grid_height,
                        grid_width=grid_width,
                    )
                )
                patch_index += 1
    return NativePatchSchedule(
        work_items=tuple(work_items),
        patch_height=int(patch_height),
        patch_width=int(patch_width),
        overlap=int(overlap),
        request_grid_shapes=shapes,
        source_scheduler_called=source_scheduler_called,
        source_split_called=source_split_called,
        source_split_input_shape=source_split_input_shape,
        source_split_output_shape=source_split_output_shape,
        source_split_patch_count=source_split_patch_count,
        cycle_request_indices=cycle_request_indices,
        homogeneous=homogeneous,
    )



_ACTIVE_NATIVE_PATCH_SCHEDULE: ContextVar[NativePatchSchedule | None] = ContextVar(
    "native_patch_schedule", default=None
)


def active_native_patch_schedule() -> NativePatchSchedule | None:
    """Return the System B native patch schedule for the current forward."""
    return _ACTIVE_NATIVE_PATCH_SCHEDULE.get()


@contextmanager
def use_native_patch_schedule(
    schedule: NativePatchSchedule | None,
) -> Iterator[NativePatchSchedule | None]:
    """Bind the System B native patch schedule for one denoise forward."""
    token = _ACTIVE_NATIVE_PATCH_SCHEDULE.set(schedule)
    try:
        yield schedule
    finally:
        _ACTIVE_NATIVE_PATCH_SCHEDULE.reset(token)


def slice_lora_composition_for_cycle(
    root: torch.nn.Module,
    cycle_idx: tuple[int, ...],
) -> list[tuple[torch.nn.Module, tuple, tuple[int, ...] | None]]:
    """Temporarily publish sub-batch composition/row_layout for one cycle.

    Returns restore records for :func:`restore_lora_composition`.
    """
    saved: list[tuple[torch.nn.Module, tuple, tuple[int, ...] | None]] = []
    indices = tuple(int(i) for i in cycle_idx)
    for module in root.modules():
        composition = getattr(module, "_diffusion_lora_batch_composition", None)
        if composition is None:
            continue
        if len(composition) < max(indices, default=-1) + 1:
            raise RuntimeError(
                "MixFusion cycle composition shorter than cycle request index: "
                f"len={len(composition)} cycle={indices}"
            )
        row_layout = getattr(module, "_diffusion_lora_batch_row_layout", None)
        saved.append((module, tuple(composition), row_layout))
        sliced = tuple(composition[i] for i in indices)
        sliced_rows = (
            tuple(int(row_layout[i]) for i in indices)
            if row_layout is not None
            else None
        )
        setter = getattr(module, "set_batch_adapter_composition", None)
        if setter is None:
            raise RuntimeError(
                "MixFusion cycle remap requires set_batch_adapter_composition"
            )
        setter(sliced, row_layout=sliced_rows)
    return saved


def restore_lora_composition(
    saved: list[tuple[torch.nn.Module, tuple, tuple[int, ...] | None]],
) -> None:
    """Restore compositions saved by :func:`slice_lora_composition_for_cycle`."""
    for module, composition, row_layout in saved:
        setter = getattr(module, "set_batch_adapter_composition", None)
        if setter is not None:
            setter(composition, row_layout=row_layout)


def adapter_scale_signature_key(scale: float) -> int:
    """Stable integer key for adapter scale (no 1e-3 collision)."""
    import struct

    return int(struct.unpack("!I", struct.pack("!f", float(scale)))[0])


__all__ = [
    "MIXFUSION_PATCH_ENTRY",
    "MIXFUSION_SCHEDULER_ENTRY",
    "MIXFUSION_SOURCE_COMMIT",
    "MIXFUSION_SPLIT_ENTRY",
    "NativePatchSchedule",
    "NativePatchWork",
    "active_native_patch_schedule",
    "adapter_scale_signature_key",
    "restore_lora_composition",
    "schedule_patch_work",
    "slice_lora_composition_for_cycle",
    "use_native_patch_schedule",
]
