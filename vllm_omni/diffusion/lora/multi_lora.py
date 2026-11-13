"""Vendored into Omni for official-pip vllm.

Provenance: copied from parent `vllm/lora/multi_lora.py` (cyn fork helper).

**Reference / correctness path only.** ``execute_lora``, ``execute_fused_lora``,
and ``execute_tiled_fused_lora`` all implement the same per-request torch
matmul ``y += scale * (x @ A.T) @ B.T``. ``execute_tiled_fused_lora`` consumes
``tile_plan.tile_size`` to chunk rows (not a fused/tiled performance kernel).
Performance No-Tile backends are ``dense_fused`` / ``two_stage_tc`` via
``ops.backend``; do not treat these helpers as fused or tiled kernels.
"""

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class MultiLoRAAdapter:
    adapter_id: int
    lora_a: torch.Tensor
    lora_b: torch.Tensor


@dataclass(frozen=True)
class MultiLoRABatchPlan:
    input_row_layout: tuple[int, ...]
    relations: tuple[tuple[int, ...], ...]
    scales: tuple[tuple[float, ...], ...]
    registry: dict[int, MultiLoRAAdapter]


def _get_lora_slot_tensors(
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    slot: int,
    rank_table: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return one adapter's exact-rank views from shared storage."""
    a = a_stack[int(slot), 0]
    b = b_stack[int(slot), 0]
    rank = int(rank_table[int(slot)])
    if rank <= 0:
        return None
    if rank > a.shape[0] or rank > b.shape[1]:
        raise ValueError(
            "LoRA rank metadata exceeds stacked storage: "
            f"slot={slot}, rank={rank}, a_rank={a.shape[0]}, "
            f"b_rank={b.shape[1]}"
        )
    return a[:rank, :], b[:, :rank]


def prepare_lora_batch(
    *,
    module_identity: str,
    input_row_layout: tuple[int, ...],
    request_to_adapter_relation: list[list[int]] | tuple[tuple[int, ...], ...],
    adapter_registry_snapshot: dict[int, MultiLoRAAdapter],
    request_scales: list[list[float]] | tuple[tuple[float, ...], ...],
    execution_ownership: str,
) -> MultiLoRABatchPlan:
    del module_identity, execution_ownership
    return MultiLoRABatchPlan(
        tuple(int(x) for x in input_row_layout),
        tuple(tuple(int(x) for x in row) for row in request_to_adapter_relation),
        tuple(tuple(float(x) for x in row) for row in request_scales),
        dict(adapter_registry_snapshot),
    )


def _apply_rows(
    x: torch.Tensor,
    y: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    row_layout: tuple[int, ...],
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: Any,
    output_offset: int = 0,
    output_size: int | None = None,
    *,
    tile_size: int | None = None,
) -> None:
    row_start = 0
    for request_index, rows in enumerate(row_layout):
        row_end = row_start + rows
        if rows <= 0:
            row_start = row_end
            continue
        chunk = int(tile_size) if tile_size is not None else rows
        if chunk <= 0:
            raise ValueError(f"invalid tile_size={tile_size}")
        for tile_start in range(row_start, row_end, chunk):
            tile_end = min(tile_start + chunk, row_end)
            target = y[tile_start:tile_end]
            for slot, scale in zip(relations[request_index], scales[request_index]):
                adapter_tensors = _get_lora_slot_tensors(
                    a_stack, b_stack, slot, rank_table
                )
                if adapter_tensors is None:
                    continue
                a, b = adapter_tensors
                delta = (
                    x[tile_start:tile_end].to(a.dtype) @ a.transpose(0, 1)
                ) @ b.transpose(0, 1)
                if output_size is None:
                    output_size = int(delta.shape[1])
                target[:, output_offset : output_offset + output_size].add_(
                    delta[:, :output_size].to(dtype=target.dtype), alpha=float(scale)
                )
        row_start = row_end


def execute_lora(
    x: torch.Tensor,
    y: torch.Tensor,
    plan: MultiLoRABatchPlan,
    *,
    gemm_timer: Callable[[str, Callable[[], torch.Tensor]], torch.Tensor] | None = None,
) -> None:
    for slot, adapter in plan.registry.items():
        del slot
        _ = adapter
    # The plan registry tensors are equivalent to the stacked representation;
    # execute one adapter at a time while preserving request-to-slot routing.
    for request_index, rows in enumerate(plan.input_row_layout):
        start = sum(plan.input_row_layout[:request_index])
        end = start + rows
        for slot, scale in zip(plan.relations[request_index], plan.scales[request_index]):
            adapter = plan.registry[int(slot)]
            if adapter.lora_a.numel() == 0 or adapter.lora_b.numel() == 0:
                continue
            input_rows = x[start:end].to(adapter.lora_a.dtype)
            hidden = (
                gemm_timer(
                    "lora_a",
                    lambda: input_rows @ adapter.lora_a.transpose(0, 1),
                )
                if gemm_timer is not None
                else input_rows @ adapter.lora_a.transpose(0, 1)
            )
            delta = (
                gemm_timer(
                    "lora_b",
                    lambda: hidden @ adapter.lora_b.transpose(0, 1),
                )
                if gemm_timer is not None
                else hidden @ adapter.lora_b.transpose(0, 1)
            )
            y[start:end, : delta.shape[1]].add_(delta.to(y.dtype), alpha=float(scale))


def execute_fused_lora(
    x: torch.Tensor,
    y: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    row_layout: tuple[int, ...],
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: Any,
) -> None:
    """Reference torch matmul (name retained for API compatibility)."""
    _apply_rows(x, y, a_stack, b_stack, row_layout, relations, scales, rank_table)


def execute_tiled_fused_lora(
    x: torch.Tensor,
    y: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    row_layout: tuple[int, ...],
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: Any,
    *,
    tile_plan: Any = None,
) -> None:
    """Reference torch matmul that consumes ``tile_plan.tile_size`` row chunks."""
    if tile_plan is None:
        raise ValueError(
            "execute_tiled_fused_lora requires tile_plan "
            "(tile_size must change execution granularity)"
        )
    tile_size = int(getattr(tile_plan, "tile_size"))
    if tile_size <= 0:
        raise ValueError(f"tile_plan.tile_size must be positive, got {tile_size}")
    _apply_rows(
        x,
        y,
        a_stack,
        b_stack,
        row_layout,
        relations,
        scales,
        rank_table,
        tile_size=tile_size,
    )
