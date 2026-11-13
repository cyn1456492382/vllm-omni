"""MixFusion patch-level metadata adapter for Omni Multi-LoRA.

The pinned MixFusion source does not provide a LoRA kernel.  This module
therefore reuses its scheduler/patch ordering contract and dispatches the
result through Omni's dense fused Triton operator.  The separation is
intentional: the source semantics are authoritative for work ownership, while
the device math remains an explicitly named Omni backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import torch

from .ops import (
    BACKEND_DENSE_FUSED,
    BACKEND_TWO_STAGE_TC,
    LoRABackendError,
    get_dense_fused_ops,
    require_backend,
    resolve_backend_name,
)
from .tetris_lora_twostage import execute_two_stage_lora

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


def _allow_full_sequence_without_schedule() -> bool:
    """Fair M*: VLLM_OMNI_MIXFUSION_SCHEDULE=0 skips host patch scheduling."""
    return (os.environ.get("VLLM_OMNI_MIXFUSION_SCHEDULE") or "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


@dataclass(frozen=True)
class MixFusionMultiLoRAPlan:
    """Source-ordered ownership metadata for one flattened linear input."""

    row_layout: tuple[int, ...]
    row_spans: tuple[tuple[int, int, int], ...] | None
    patch_ordinals: tuple[int, ...]
    source_scheduler_called: bool
    source_split_called: bool
    ownership_mode: str

    @property
    def patch_count(self) -> int:
        return len(self.patch_ordinals)

    def validate(self, input_rows: int) -> None:
        if sum(self.row_layout) != int(input_rows):
            raise ValueError("MixFusion Multi-LoRA row ownership misses input rows")
        if not self.row_layout or any(rows <= 0 for rows in self.row_layout):
            raise ValueError("MixFusion Multi-LoRA row ownership must be positive")
        if not self.patch_ordinals:
            raise RuntimeError("MixFusion source patch plan is empty")
        if self.row_spans is None:
            if self.ownership_mode != "request_rows":
                raise ValueError("MixFusion Multi-LoRA ownership mode is invalid")
            # Fair M* / SCHEDULE=0 may omit the host MixFusion schedule while
            # still executing request-contiguous LoRA on the GPU.
            if not self.source_scheduler_called and not _allow_full_sequence_without_schedule():
                raise RuntimeError(
                    "MixFusion source scheduler did not produce a plan"
                )
            return
        if not self.source_scheduler_called:
            raise RuntimeError("MixFusion source scheduler did not produce a plan")
        if self.ownership_mode != "patch_row_spans":
            raise ValueError("MixFusion patch spans require patch ownership mode")
        # Interval overlap check (avoid O(tokens) Python scans on the hot path).
        intervals: list[tuple[int, int, int]] = []
        for start, count, request_index in self.row_spans:
            start_i = int(start)
            count_i = int(count)
            request_i = int(request_index)
            end_i = start_i + count_i
            if start_i < 0 or count_i <= 0 or end_i > input_rows:
                raise ValueError("MixFusion Multi-LoRA patch span is out of bounds")
            if not 0 <= request_i < len(self.row_layout):
                raise ValueError("MixFusion Multi-LoRA patch request is invalid")
            intervals.append((start_i, end_i, request_i))
        intervals.sort()
        covered_rows = 0
        previous_end = 0
        for start_i, end_i, _request_i in intervals:
            if start_i < previous_end:
                raise ValueError("MixFusion Multi-LoRA patch spans overlap")
            if start_i > previous_end:
                raise ValueError(
                    "MixFusion Multi-LoRA patch spans do not partition rows"
                )
            covered_rows += end_i - start_i
            previous_end = end_i
        if covered_rows != int(input_rows) or previous_end != int(input_rows):
            raise ValueError("MixFusion Multi-LoRA patch spans do not partition rows")


@dataclass(frozen=True)
class MixFusionMultiLoRADispatch:
    """Proof fields emitted for one source-ordered LoRA dispatch."""

    kernel_backend: str
    ownership_mode: str
    patch_count: int
    patch_row_span_count: int
    rank_bucket_blocks: tuple[int, ...]
    used_execute_tiled_fused_lora: bool = False


@lru_cache(maxsize=16)
def _cached_mixfusion_multilora_plan(
    row_layout: tuple[int, ...],
    request_shapes: tuple[tuple[int, int], ...],
    work_items: tuple[tuple[int, int, int, int, int, int], ...],
    source_scheduler_called: bool,
    source_split_called: bool,
) -> MixFusionMultiLoRAPlan:
    grid_rows = tuple(height * width for height, width in request_shapes)
    if len(request_shapes) != len(row_layout):
        raise ValueError(
            "MixFusion schedule request count does not match row_layout: "
            f"shapes={len(request_shapes)} layout={len(row_layout)}"
        )
    # The pinned latent patch schedule only applies to token-grid-sized
    # linears. Batch-sized conditioning layers (row_layout ones / small
    # sequences) keep request_rows ownership and ignore patch spans.
    if sum(row_layout) != sum(grid_rows):
        # #region agent log
        try:
            import json as _json
            import os as _os
            import time as _time

            if _os.environ.get("CURSOR_AGENT_DEBUG") == "68c61b":
                _key = (row_layout, grid_rows, source_scheduler_called)
                _seen = getattr(_cached_mixfusion_multilora_plan, "_dbg_seen", set())
                if _key not in _seen and len(_seen) < 8:
                    _seen.add(_key)
                    _cached_mixfusion_multilora_plan._dbg_seen = _seen  # type: ignore[attr-defined]
                    with open(
                        "/workspace/.cursor/debug-68c61b.log",
                        "a",
                        encoding="utf-8",
                    ) as _dbg:
                        _dbg.write(
                            _json.dumps(
                                {
                                    "sessionId": "68c61b",
                                    "hypothesisId": "H5",
                                    "location": "mixfusion_multi_lora.py:_cached_plan",
                                    "message": "non-grid linear under schedule",
                                    "data": {
                                        "row_layout": list(row_layout),
                                        "grid_rows": list(grid_rows),
                                        "work_items": len(work_items),
                                        "ownership": "request_rows",
                                    },
                                    "timestamp": int(_time.time() * 1000),
                                    "runId": "diag-pre",
                                }
                            )
                            + "\n"
                        )
        except Exception:
            pass
        # #endregion
        plan = MixFusionMultiLoRAPlan(
            row_layout=row_layout,
            row_spans=None,
            patch_ordinals=tuple(range(len(row_layout))),
            source_scheduler_called=source_scheduler_called,
            source_split_called=source_split_called,
            ownership_mode="request_rows",
        )
        plan.validate(sum(row_layout))
        return plan
    # Composition may carry a placeholder layout while the schedule owns the
    # real token grids — prefer grids for span addressing.
    if row_layout != grid_rows:
        if not work_items:
            raise ValueError(
                "MixFusion inputs are not request-contiguous relative to the "
                "patch schedule grids; refuse request_rows ownership without "
                f"real spans (row_layout={row_layout}, grid_rows={grid_rows})"
            )
        row_layout = grid_rows
    request_starts = []
    current = 0
    for rows in row_layout:
        request_starts.append(current)
        current += rows

    patch_spans: list[tuple[int, int, int]] = []
    for (
        request_index,
        _patch_index,
        row_start,
        column_start,
        valid_height,
        valid_width,
    ) in work_items:
        request_index = int(request_index)
        if not 0 <= request_index < len(row_layout):
            raise ValueError("MixFusion patch request index is invalid")
        _, grid_width = request_shapes[request_index]
        base_row = request_starts[request_index]
        for local_row in range(int(valid_height)):
            start = (
                base_row
                + (int(row_start) + local_row) * grid_width
                + int(column_start)
            )
            patch_spans.append((start, int(valid_width), request_index))

    # Full-request single spans that exactly match row_layout are still
    # request-contiguous — report request_rows so two_stage may run.
    trivial_request_rows = False
    if len(patch_spans) == len(row_layout):
        expected = []
        offset = 0
        for request_index, rows in enumerate(row_layout):
            expected.append((offset, int(rows), request_index))
            offset += int(rows)
        trivial_request_rows = tuple(patch_spans) == tuple(expected)
    if trivial_request_rows:
        ownership_mode = "request_rows"
        plan = MixFusionMultiLoRAPlan(
            row_layout=row_layout,
            row_spans=None,
            patch_ordinals=tuple(range(len(work_items))),
            source_scheduler_called=source_scheduler_called,
            source_split_called=source_split_called,
            ownership_mode=ownership_mode,
        )
    else:
        ownership_mode = "patch_row_spans"
        plan = MixFusionMultiLoRAPlan(
            row_layout=row_layout,
            row_spans=tuple(patch_spans),
            patch_ordinals=tuple(range(len(work_items))),
            source_scheduler_called=source_scheduler_called,
            source_split_called=source_split_called,
            ownership_mode=ownership_mode,
        )
    # #region agent log
    try:
        import json as _json
        import os as _os
        import time as _time

        if _os.environ.get("CURSOR_AGENT_DEBUG") == "68c61b":
            _seen2 = getattr(_cached_mixfusion_multilora_plan, "_dbg_own", set())
            _okey = (ownership_mode, len(work_items), sum(row_layout))
            if _okey not in _seen2 and len(_seen2) < 8:
                _seen2.add(_okey)
                _cached_mixfusion_multilora_plan._dbg_own = _seen2  # type: ignore[attr-defined]
                with open(
                    "/workspace/.cursor/debug-68c61b.log",
                    "a",
                    encoding="utf-8",
                ) as _dbg:
                    _dbg.write(
                        _json.dumps(
                            {
                                "sessionId": "68c61b",
                                "hypothesisId": "H5",
                                "location": "mixfusion_multi_lora.py:grid_plan",
                                "message": "schedule-bound ownership",
                                "data": {
                                    "ownership_mode": ownership_mode,
                                    "trivial_request_rows": bool(trivial_request_rows),
                                    "patch_spans": len(patch_spans),
                                    "work_items": len(work_items),
                                    "row_sum": int(sum(row_layout)),
                                    "grid_sum": int(sum(grid_rows)),
                                },
                                "timestamp": int(_time.time() * 1000),
                                "runId": "diag-pre",
                            }
                        )
                        + "\n"
                    )
    except Exception:
        pass
    # #endregion
    plan.validate(sum(row_layout))
    return plan


def build_mixfusion_multilora_plan(
    row_layout: tuple[int, ...],
    native_patch_schedule: object | None,
) -> MixFusionMultiLoRAPlan:
    """Build cached LoRA ownership from the pinned MixFusion patch schedule.

    A linear layer may receive either the complete request rows or a
    patch-packed row grid. Both are valid source contracts; the former keeps
    the source patch order at request ownership granularity, while the latter
    exposes exact patch row spans to the Triton metadata.
    """
    if native_patch_schedule is None:
        if not _allow_full_sequence_without_schedule():
            raise RuntimeError(
                "MixFusion Multi-LoRA requires a source patch schedule"
            )
        row_layout = tuple(int(rows) for rows in row_layout)
        plan = MixFusionMultiLoRAPlan(
            row_layout=row_layout,
            row_spans=None,
            patch_ordinals=tuple(range(len(row_layout))),
            source_scheduler_called=False,
            source_split_called=False,
            ownership_mode="request_rows",
        )
        plan.validate(sum(row_layout))
        return plan
    source_work_items = tuple(getattr(native_patch_schedule, "work_items", ()))
    if not source_work_items:
        raise RuntimeError("MixFusion Multi-LoRA source patch schedule is empty")
    source_scheduler_called = bool(
        getattr(native_patch_schedule, "source_scheduler_called", False)
    )
    source_split_called = bool(
        getattr(native_patch_schedule, "source_split_called", False)
    )
    if _is_true(os.environ.get("VLLM_OMNI_MIXFUSION_NATIVE_SOURCE_CALL")):
        if not source_split_called:
            raise RuntimeError(
                "MixFusion source split was requested but did not execute"
            )

    row_layout = tuple(int(rows) for rows in row_layout)
    request_shapes = tuple(
        (int(height), int(width))
        for height, width in getattr(
            native_patch_schedule, "request_grid_shapes", ()
        )
    )
    work_items = tuple(
        (
            int(item.request_index),
            int(getattr(item, "patch_index", item_index)),
            int(item.row_start),
            int(item.column_start),
            int(item.valid_height),
            int(item.valid_width),
        )
        for item_index, item in enumerate(source_work_items)
    )
    return _cached_mixfusion_multilora_plan(
        row_layout,
        request_shapes,
        work_items,
        source_scheduler_called,
        source_split_called,
    )


def apply_mixfusion_multilora(
    inputs: torch.Tensor,
    output: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    plan: MixFusionMultiLoRAPlan,
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: object,
    *,
    metadata_cache: dict | None = None,
) -> MixFusionMultiLoRADispatch:
    """Run source-ordered Multi-LoRA using the selected device backend.

    Kept outside Dynamo (``torch.compiler.disable``) so backend registry
    lookups cannot graph-break the surrounding Base GEMM compile.
    """
    return _apply_mixfusion_multilora_impl(
        inputs,
        output,
        a_stack,
        b_stack,
        plan,
        relations,
        scales,
        rank_table,
        metadata_cache=metadata_cache,
    )


try:
    apply_mixfusion_multilora = torch.compiler.disable(apply_mixfusion_multilora)  # type: ignore[misc]
except Exception:
    pass


def _apply_mixfusion_multilora_impl(
    inputs: torch.Tensor,
    output: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    plan: MixFusionMultiLoRAPlan,
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: object,
    *,
    metadata_cache: dict | None = None,
) -> MixFusionMultiLoRADispatch:
    """Implementation body (kept outside Dynamo via disable wrapper below)."""
    plan.validate(int(inputs.shape[0]))
    backend_name = resolve_backend_name(for_mixfusion=True)
    if backend_name == BACKEND_TWO_STAGE_TC:
        require_backend(BACKEND_TWO_STAGE_TC)
        # request_rows: contiguous row_layout. patch_row_spans: gather/scatter
        # inside execute_two_stage_lora (same span contract as dense_fused).
        if plan.ownership_mode not in {"request_rows", "patch_row_spans"}:
            raise LoRABackendError(
                "two_stage_tc unsupported ownership_mode="
                f"{plan.ownership_mode!r}"
            )
        if plan.ownership_mode == "request_rows" and plan.row_spans is not None:
            raise LoRABackendError(
                "two_stage_tc request_rows ownership forbids row_spans"
            )
        if plan.ownership_mode == "patch_row_spans" and not plan.row_spans:
            raise LoRABackendError(
                "two_stage_tc patch_row_spans ownership requires row_spans"
            )
        # K2: persist TetrisTwoStageWorkspace (H/records/metadata) + Δ buffer.
        cache = metadata_cache if metadata_cache is not None else {}
        tw_key = ("two_stage_workspace", str(inputs.device))
        tw_workspace = cache.get(tw_key)
        if tw_workspace is None:
            from vllm_omni.diffusion.lora.tetris_lora_twostage import (
                TetrisTwoStageWorkspace,
            )

            tw_workspace = TetrisTwoStageWorkspace()
            cache[tw_key] = tw_workspace
        workspace_key = (
            "two_stage_delta",
            int(output.shape[0]),
            int(output.shape[1]),
            str(output.device),
            str(output.dtype),
        )
        delta = cache.get(workspace_key)
        if (
            delta is None
            or not isinstance(delta, torch.Tensor)
            or delta.shape != output.shape
            or delta.device != output.device
            or delta.dtype != output.dtype
        ):
            delta = torch.empty_like(output)
            cache[workspace_key] = delta
        _, descriptor = execute_two_stage_lora(
            inputs,
            a_stack,
            b_stack,
            plan.row_layout,
            relations,
            scales,
            rank_table,
            output=delta,
            clear_output=True,
            workspace=tw_workspace,
            row_spans=plan.row_spans,
        )
        output.add_(delta)
        return MixFusionMultiLoRADispatch(
            kernel_backend="tetris_two_stage_tc",
            ownership_mode=plan.ownership_mode,
            patch_count=plan.patch_count,
            patch_row_span_count=len(plan.row_spans or ()),
            rank_bucket_blocks=descriptor.rank_blocks,
        )
    if backend_name != BACKEND_DENSE_FUSED:
        raise LoRABackendError(
            f"MixFusion Multi-LoRA performance path does not accept "
            f"backend={backend_name!r}; use dense_fused or two_stage_tc "
            "(reference is not a silent fallback)."
        )
    dense_mod = get_dense_fused_ops()
    rank_bucket_blocks = dense_mod.dense_fused_lora_add(
        inputs,
        output,
        a_stack,
        b_stack,
        plan.row_layout,
        relations,
        scales,
        rank_table,
        metadata_cache=metadata_cache,
        row_spans=plan.row_spans,
    )
    return MixFusionMultiLoRADispatch(
        kernel_backend="omni_dense_fused_lora_triton",
        ownership_mode=plan.ownership_mode,
        patch_count=plan.patch_count,
        patch_row_span_count=len(plan.row_spans or ()),
        rank_bucket_blocks=tuple(int(block) for block in rank_bucket_blocks),
    )


__all__ = [
    "MixFusionMultiLoRADispatch",
    "MixFusionMultiLoRAPlan",
    "apply_mixfusion_multilora",
    "build_mixfusion_multilora_plan",
]
