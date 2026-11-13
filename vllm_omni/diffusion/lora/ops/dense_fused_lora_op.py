# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense diffusion LoRA shrink-expand-add Triton operator.

PROVENANCE
----------
Copied from parent fork path:
  vllm/lora/ops/triton_ops/dense_fused_lora_op.py
Introducing commit: bf69e4612 (cyn1456492382)
  feat: add auditable dense fused LoRA operator
Blob at copy time: e9844a9852e2dfd068ca6ced4b2de11f948afe46
Modifications vs source: package location only; owned by vllm_omni Backend
Registry so MixFusion/Tetris do not hard-import the parent fork path.
This file is NOT upstream vLLM; it is a TetrisLoRA/Omni performance backend.

SHRINK REPETITION
-----------------
``_dense_fused_lora_kernel`` grids ``pid_n`` over ``cdiv(N, BLOCK_N)`` with
``BLOCK_N=64``. Shrink (``X @ A``) does not depend on the output tile, but the
GEMM sits inside that N-tiled program, so every N-block recomputes it. The
shared-input kernel has the same ``pid_n`` axis. Host launch does not loop
over N: one rank bucket keeps a single launch, and several rank buckets
already loop in ``dense_fused_lora_add`` (each launch still repeats Shrink).
MixFusion defaults to ``two_stage_tc``, which shrinks once per rank block
into H and then expands. ``VLLM_OMNI_FORCE_DENSE_FUSED=1`` or
``VLLM_OMNI_LORA_BACKEND=dense_fused`` keeps this kernel for contrast.
See ``.tetris_lora_results/acceptance_20261112/closure_20261112/SHRINK_VS_TWOSTAGE.md``.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _dense_fused_lora_kernel(
    x_ptr,
    y_ptr,
    a_ptr,
    b_ptr,
    row_starts_ptr,
    row_counts_ptr,
    slots_ptr,
    scales_ptr,
    ranks_ptr,
    input_hidden,
    output_hidden,
    max_relations,
    a_slot_stride,
    a_rank_stride,
    a_hidden_stride,
    b_slot_stride,
    b_output_stride,
    b_rank_stride,
    x_row_stride,
    x_hidden_stride,
    y_row_stride,
    y_output_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    request_id = tl.program_id(axis=2)
    row_start = tl.load(row_starts_ptr + request_id)
    row_count = tl.load(row_counts_ptr + request_id)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = row_start + row_offsets
    output_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_offsets < row_count
    output_mask = output_offsets < output_hidden
    if pid_m * BLOCK_M >= row_count:
        return

    residual = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for relation_id in range(max_relations):
        metadata_offset = request_id * max_relations + relation_id
        slot = tl.load(slots_ptr + metadata_offset)
        rank = tl.load(ranks_ptr + metadata_offset)
        scale = tl.load(scales_ptr + metadata_offset)
        if (slot >= 0) & (rank > 0):
            hidden = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
            rank_offsets = tl.arange(0, BLOCK_R)
            for hidden_start in range(0, input_hidden, BLOCK_H):
                hidden_offsets = hidden_start + tl.arange(0, BLOCK_H)
                x_values = tl.load(
                    x_ptr
                    + rows[:, None] * x_row_stride
                    + hidden_offsets[None, :] * x_hidden_stride,
                    mask=(
                        row_mask[:, None]
                        & (hidden_offsets[None, :] < input_hidden)
                    ),
                    other=0.0,
                )
                a_values = tl.load(
                    a_ptr
                    + slot * a_slot_stride
                    + rank_offsets[None, :] * a_rank_stride
                    + hidden_offsets[:, None] * a_hidden_stride,
                    mask=(
                        (rank_offsets[None, :] < rank)
                        & (hidden_offsets[:, None] < input_hidden)
                    ),
                    other=0.0,
                )
                hidden = tl.dot(x_values, a_values, acc=hidden)

            b_values = tl.load(
                b_ptr
                + slot * b_slot_stride
                + output_offsets[:, None] * b_output_stride
                + rank_offsets[None, :] * b_rank_stride,
                mask=(
                    output_mask[:, None]
                    & (rank_offsets[None, :] < rank)
                ),
                other=0.0,
            )
            delta = tl.dot(
                hidden.to(x_ptr.dtype.element_ty),
                tl.trans(b_values),
            )
            residual += delta * scale

    y_values = tl.load(
        y_ptr + rows[:, None] * y_row_stride
        + output_offsets[None, :] * y_output_stride,
        mask=row_mask[:, None] & output_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        y_ptr + rows[:, None] * y_row_stride
        + output_offsets[None, :] * y_output_stride,
        y_values + residual,
        mask=row_mask[:, None] & output_mask[None, :],
    )


@triton.jit
def _shared_input_lora_kernel(
    x_ptr,
    y_ptr,
    a_ptr,
    b_ptr,
    row_starts_ptr,
    row_counts_ptr,
    slots_ptr,
    scales_ptr,
    ranks_ptr,
    input_hidden,
    output_hidden,
    max_relations,
    a_slot_stride,
    a_rank_stride,
    a_hidden_stride,
    b_slot_stride,
    b_output_stride,
    b_rank_stride,
    x_row_stride,
    x_hidden_stride,
    y_row_stride,
    y_output_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MAX_RELATIONS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    request_id = tl.program_id(axis=2)
    row_start = tl.load(row_starts_ptr + request_id)
    row_count = tl.load(row_counts_ptr + request_id)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = row_start + row_offsets
    output_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_offsets < row_count
    output_mask = output_offsets < output_hidden
    if pid_m * BLOCK_M >= row_count:
        return

    rank_offsets = tl.arange(0, BLOCK_R)
    hidden_by_relation = tl.zeros(
        (MAX_RELATIONS, BLOCK_M, BLOCK_R), dtype=tl.float32
    )
    for hidden_start in range(0, input_hidden, BLOCK_H):
        hidden_offsets = hidden_start + tl.arange(0, BLOCK_H)
        x_values = tl.load(
            x_ptr
            + rows[:, None] * x_row_stride
            + hidden_offsets[None, :] * x_hidden_stride,
            mask=(row_mask[:, None] & (hidden_offsets[None, :] < input_hidden)),
            other=0.0,
        )
        for relation_id in range(MAX_RELATIONS):
            relation_valid = relation_id < max_relations
            metadata_offset = request_id * max_relations + relation_id
            slot = tl.load(
                slots_ptr + metadata_offset, mask=relation_valid, other=-1
            )
            rank = tl.load(
                ranks_ptr + metadata_offset, mask=relation_valid, other=0
            )
            a_values = tl.load(
                a_ptr
                + slot * a_slot_stride
                + rank_offsets[None, :] * a_rank_stride
                + hidden_offsets[:, None] * a_hidden_stride,
                mask=(
                    relation_valid
                    & (slot >= 0)
                    & (rank_offsets[None, :] < rank)
                    & (hidden_offsets[:, None] < input_hidden)
                ),
                other=0.0,
            )
            hidden_by_relation = hidden_by_relation + tl.where(
                (
                    tl.arange(0, MAX_RELATIONS)[:, None, None] == relation_id
                )
                & relation_valid,
                tl.dot(x_values, a_values)[None, :, :],
                tl.zeros((MAX_RELATIONS, BLOCK_M, BLOCK_R), dtype=tl.float32),
            )

    residual = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for relation_id in range(MAX_RELATIONS):
        relation_valid = relation_id < max_relations
        metadata_offset = request_id * max_relations + relation_id
        slot = tl.load(
            slots_ptr + metadata_offset, mask=relation_valid, other=-1
        )
        rank = tl.load(
            ranks_ptr + metadata_offset, mask=relation_valid, other=0
        )
        scale = tl.load(
            scales_ptr + metadata_offset, mask=relation_valid, other=0.0
        )
        b_values = tl.load(
            b_ptr
            + slot * b_slot_stride
            + output_offsets[:, None] * b_output_stride
            + rank_offsets[None, :] * b_rank_stride,
            mask=(
                relation_valid
                & (slot >= 0)
                & output_mask[:, None]
                & (rank_offsets[None, :] < rank)
            ),
            other=0.0,
        )
        relation_mask = (
            tl.arange(0, MAX_RELATIONS)[:, None, None] == relation_id
        ).to(tl.float32)
        hidden_for_relation = tl.sum(
            hidden_by_relation * relation_mask,
            axis=0,
        )
        residual += tl.dot(
            hidden_for_relation.to(x_ptr.dtype.element_ty),
            tl.trans(b_values),
        ) * scale

    y_values = tl.load(
        y_ptr + rows[:, None] * y_row_stride
        + output_offsets[None, :] * y_output_stride,
        mask=row_mask[:, None] & output_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        y_ptr + rows[:, None] * y_row_stride
        + output_offsets[None, :] * y_output_stride,
        y_values + residual,
        mask=row_mask[:, None] & output_mask[None, :],
    )


def _validate_lora_activation_contract(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    row_layout: tuple[int, ...],
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: Any,
    name: str,
    row_spans: tuple[tuple[int, int, int], ...] | None = None,
    allow_empty_relations: bool = True,
) -> tuple[int, tuple[int, ...]]:
    """Shared device/shape/slot validation for fused LoRA entry points."""
    if not (x.is_cuda and y.is_cuda):
        raise RuntimeError(f"{name} requires CUDA tensors")
    if x.device != y.device or a_stack.device != x.device or b_stack.device != x.device:
        raise ValueError(f"{name} requires x/y/A/B on the same CUDA device")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"{name} expects flattened 2-D tensors")
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"{name} y rows must match x rows")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{name} supports FP16 and BF16 inputs")
    if y.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"{name} supports FP16/BF16/FP32 outputs")
    if a_stack.dtype != x.dtype or b_stack.dtype != x.dtype:
        raise ValueError(f"{name} expects weights matching input dtype")
    if len(row_layout) != len(relations) or len(relations) != len(scales):
        raise ValueError("LoRA row metadata lengths do not match")
    if any(int(rows) < 0 for rows in row_layout):
        raise ValueError("LoRA row counts cannot be negative")
    if sum(row_layout) != x.shape[0]:
        raise ValueError("LoRA row layout does not cover flattened inputs")
    if a_stack.ndim != 4 or b_stack.ndim != 4:
        raise ValueError(f"{name} expects stacked 4-D weights")
    if a_stack.shape[0] != b_stack.shape[0]:
        raise ValueError("LoRA A/B slot capacities do not match")
    if int(a_stack.shape[1]) != 1 or int(b_stack.shape[1]) != 1:
        raise ValueError(f"{name} requires A/B slice dimension == 1")
    if a_stack.shape[3] != x.shape[1] or b_stack.shape[2] != y.shape[1]:
        raise ValueError("LoRA weight dimensions do not match activations")
    max_rank = int(a_stack.shape[2])
    if max_rank <= 0 or max_rank > 64 or b_stack.shape[3] < max_rank:
        raise ValueError(f"{name} supports stored ranks from 1 through 64")
    rank_signature = tuple(int(rank) for rank in rank_table)
    slot_capacity = int(a_stack.shape[0])
    for request_relations, request_scales in zip(relations, scales):
        if len(request_relations) != len(request_scales):
            raise ValueError("LoRA relation and scale lengths do not match")
        for slot in request_relations:
            slot_i = int(slot)
            if not 0 <= slot_i < len(rank_signature):
                raise ValueError(f"LoRA slot is outside rank table: {slot_i}")
            if slot_i >= slot_capacity:
                raise ValueError(
                    f"LoRA slot {slot_i} exceeds A/B capacity {slot_capacity}"
                )
            rank = int(rank_signature[slot_i])
            if rank < 0 or rank > max_rank or rank > int(b_stack.shape[3]):
                raise ValueError(f"invalid LoRA rank {rank} for slot {slot_i}")
            # rank==0 means missing target / zero contribution; still a valid slot.
    max_relations = max((len(row) for row in relations), default=0)
    if max_relations == 0 and not allow_empty_relations:
        raise ValueError(f"{name} received empty relation metadata")
    if row_spans is not None:
        if not row_spans:
            return max_relations, rank_signature
        covered: list[tuple[int, int]] = []
        for start, count, request_index in row_spans:
            start_i = int(start)
            count_i = int(count)
            request_i = int(request_index)
            if (
                start_i < 0
                or count_i <= 0
                or start_i + count_i > x.shape[0]
                or not 0 <= request_i < len(relations)
            ):
                raise ValueError("invalid LoRA patch row span")
            span_end = start_i + count_i
            for prev_start, prev_end in covered:
                if start_i < prev_end and prev_start < span_end:
                    raise ValueError(
                        "overlapping LoRA row spans are unsupported "
                        f"({start_i}:{span_end} vs {prev_start}:{prev_end})"
                    )
            covered.append((start_i, span_end))
    return max_relations, rank_signature


@torch.compiler.disable
def _device_metadata(
    *,
    x: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    row_layout: tuple[int, ...],
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: Any,
    cache: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] | None,
    row_spans: tuple[tuple[int, int, int], ...] | None = None,
    rank_lower: int = 0,
    rank_upper: int = 32,
) -> tuple[torch.Tensor, ...]:
    rank_signature = tuple(int(rank) for rank in rank_table)
    # Do not read Tensor._version: inference / inductor tensors may not track it,
    # and Dynamo would still capture the attribute access into the graph.
    key = (
        tuple(row_layout),
        row_spans,
        relations,
        scales,
        rank_signature,
        str(x.device),
        int(a_stack.data_ptr()),
        int(b_stack.data_ptr()),
        int(rank_lower),
        int(rank_upper),
    )
    if cache is not None and key in cache:
        return cache[key]

    max_relations = max((len(row) for row in relations), default=0)
    if max_relations == 0:
        return ()
    if row_spans is None:
        row_starts = []
        current_row = 0
        span_request_indices = tuple(range(len(row_layout)))
        for row_count in row_layout:
            row_starts.append(current_row)
            current_row += int(row_count)
        row_counts = tuple(int(row_count) for row_count in row_layout)
    else:
        row_starts = [int(start) for start, _, _ in row_spans]
        row_counts = tuple(int(count) for _, count, _ in row_spans)
        span_request_indices = tuple(int(request) for _, _, request in row_spans)
    slot_values = [[-1] * max_relations for _ in row_counts]
    scale_values = [[0.0] * max_relations for _ in row_counts]
    rank_values = [[0] * max_relations for _ in row_counts]
    for metadata_index, request_id in enumerate(span_request_indices):
        request_slots = relations[request_id]
        request_scales = scales[request_id]
        if len(request_slots) != len(request_scales):
            raise ValueError("LoRA relation and scale lengths do not match")
        for relation_id, (slot, scale) in enumerate(
            zip(request_slots, request_scales)
        ):
            slot = int(slot)
            if not 0 <= slot < len(rank_signature):
                raise ValueError(f"LoRA slot is outside rank table: {slot}")
            if slot >= a_stack.shape[0] or slot >= b_stack.shape[0]:
                raise ValueError(f"LoRA slot exceeds weight capacity: {slot}")
            rank = rank_signature[slot]
            if rank < 0 or rank > a_stack.shape[2] or rank > b_stack.shape[3]:
                raise ValueError(f"invalid LoRA rank {rank} for slot {slot}")
            if not int(rank_lower) < rank <= int(rank_upper):
                continue
            slot_values[metadata_index][relation_id] = slot
            scale_values[metadata_index][relation_id] = float(request_scales[relation_id])
            rank_values[metadata_index][relation_id] = rank

    device = x.device
    metadata = (
        torch.as_tensor(row_starts, device=device, dtype=torch.int32),
        torch.as_tensor(row_counts, device=device, dtype=torch.int32),
        torch.as_tensor(slot_values, device=device, dtype=torch.int32),
        torch.as_tensor(scale_values, device=device, dtype=torch.float32),
        torch.as_tensor(rank_values, device=device, dtype=torch.int32),
    )
    if cache is not None:
        if len(cache) >= 32:
            cache.clear()
        cache[key] = metadata
    return metadata


def dense_fused_lora_add(
    x: torch.Tensor,
    y: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    row_layout: tuple[int, ...],
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: Any,
    *,
    metadata_cache: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] | None = None,
    row_spans: tuple[tuple[int, int, int], ...] | None = None,
) -> tuple[int, ...]:
    """Add a mixed-rank LoRA residual with a Tensor-Core Triton kernel."""
    max_relations, rank_signature = _validate_lora_activation_contract(
        x=x,
        y=y,
        a_stack=a_stack,
        b_stack=b_stack,
        row_layout=row_layout,
        relations=relations,
        scales=scales,
        rank_table=rank_table,
        name="dense_fused LoRA",
        row_spans=row_spans,
    )
    if max_relations == 0 or x.shape[0] == 0:
        return ()
    if row_spans is not None and not row_spans:
        return ()

    active_ranks = sorted(
        {
            rank_signature[int(slot)]
            for request_relations in relations
            for slot in request_relations
            if rank_signature[int(slot)] > 0
        }
    )
    rank_buckets = ((0, 8, 16), (8, 16, 16), (16, 32, 32), (32, 64, 64))
    selected_buckets = tuple(
        bucket
        for bucket in rank_buckets
        if any(bucket[0] < rank <= bucket[1] for rank in active_ranks)
    )
    # All referenced slots/ranks were validated above. Empty selected_buckets
    # means every referenced rank is 0 (explicit zero-Δ), not illegal metadata.
    if not selected_buckets:
        return ()
    # One bucket: one launch. Multiple buckets: this loop. Do not add an N
    # loop here; Shrink is already repeated inside each BLOCK_N program.
    # MixFusion avoids that by selecting two_stage_tc (see ops/backend.py).
    block_m = 16
    block_n = 64
    block_hidden = 64
    max_rows = max(
        (int(count) for _, count, _ in row_spans), default=0
    ) if row_spans is not None else max(row_layout, default=0)
    metadata_rows = len(row_spans) if row_spans is not None else len(row_layout)
    grid = (
        triton.cdiv(max_rows, block_m),
        triton.cdiv(y.shape[1], block_n),
        metadata_rows,
    )
    for rank_lower, rank_upper, block_rank in selected_buckets:
        row_starts, row_counts, slots, rank_scales, ranks = _device_metadata(
            x=x,
            a_stack=a_stack,
            b_stack=b_stack,
            row_layout=row_layout,
            relations=relations,
            scales=scales,
            rank_table=rank_table,
            cache=metadata_cache,
            row_spans=row_spans,
            rank_lower=rank_lower,
            rank_upper=rank_upper,
        )
        _dense_fused_lora_kernel[grid](
            x,
            y,
            a_stack,
            b_stack,
            row_starts,
            row_counts,
            slots,
            rank_scales,
            ranks,
            x.shape[1],
            y.shape[1],
            max_relations,
            a_stack.stride(0),
            a_stack.stride(2),
            a_stack.stride(3),
            b_stack.stride(0),
            b_stack.stride(2),
            b_stack.stride(3),
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_R=block_rank,
            BLOCK_H=block_hidden,
            num_warps=4,
            num_stages=2,
        )
    return tuple(bucket[2] for bucket in selected_buckets)


def shared_input_lora_add(
    x: torch.Tensor,
    y: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    row_layout: tuple[int, ...],
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: Any,
    *,
    metadata_cache: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] | None = None,
) -> None:
    """Add LoRA residuals while loading each input tile once per H chunk."""
    max_relations, _rank_signature = _validate_lora_activation_contract(
        x=x,
        y=y,
        a_stack=a_stack,
        b_stack=b_stack,
        row_layout=row_layout,
        relations=relations,
        scales=scales,
        rank_table=rank_table,
        name="shared-input LoRA",
    )
    if max_relations == 0 or x.shape[0] == 0:
        return
    if max_relations > 4:
        raise ValueError("shared-input LoRA supports at most four adapters per row")

    row_starts, row_counts, slots, rank_scales, ranks = _device_metadata(
        x=x,
        a_stack=a_stack,
        b_stack=b_stack,
        row_layout=row_layout,
        relations=relations,
        scales=scales,
        rank_table=rank_table,
        cache=metadata_cache,
    )
    block_m = 16
    block_n = 64
    # Match dense_fused max stored rank (64); kernel BLOCK_R is constexpr.
    block_rank = 64 if int(a_stack.shape[2]) > 32 else 32
    block_hidden = 64
    # Triton block dims must be powers of two; keep logical max_relations.
    relation_block = triton.next_power_of_2(max_relations)
    grid = (
        triton.cdiv(max(row_layout, default=0), block_m),
        triton.cdiv(y.shape[1], block_n),
        len(row_layout),
    )
    _shared_input_lora_kernel[grid](
        x,
        y,
        a_stack,
        b_stack,
        row_starts,
        row_counts,
        slots,
        rank_scales,
        ranks,
        x.shape[1],
        y.shape[1],
        max_relations,
        a_stack.stride(0),
        a_stack.stride(2),
        a_stack.stride(3),
        b_stack.stride(0),
        b_stack.stride(2),
        b_stack.stride(3),
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_rank,
        BLOCK_H=block_hidden,
        MAX_RELATIONS=relation_block,
        num_warps=4,
        num_stages=2,
    )


__all__ = ["dense_fused_lora_add", "shared_input_lora_add"]
