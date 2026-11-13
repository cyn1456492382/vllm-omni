"""Two-stage Tensor-Core Multi-LoRA execution for diffusion Linear layers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence

import torch
from vllm.triton_utils import tl, triton


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _max_branch_group() -> int:
    value = int(os.environ.get("TETRIS_LORA_MAX_BRANCHES", "3"))
    if value < 1 or value > 3:
        raise ValueError("TETRIS_LORA_MAX_BRANCHES must be between 1 and 3")
    return value


@dataclass(frozen=True)
class TetrisTwoStageDescriptor:
    """Resolved ownership and compact H layout for one Linear invocation."""

    module_name: str
    request_ids: tuple[str, ...]
    row_layout: tuple[int, ...]
    relations: tuple[tuple[int, ...], ...]
    scales: tuple[tuple[float, ...], ...]
    ranks: tuple[tuple[int, ...], ...]
    output_offset: int
    output_features: int
    h_bytes: int
    branch_count: int
    group_count: int
    max_branch_group: int
    rank_blocks: tuple[int, ...]


@dataclass
class TetrisTwoStageWorkspace:
    """Reusable device buffers and metadata for one stream-ordered invocation."""

    h: torch.Tensor | None = None
    metadata: dict[tuple[str, tuple[object, ...], str], torch.Tensor] | None = None
    records: dict[
        TetrisTwoStageDescriptor,
        tuple[list[dict[str, int | float]], list[dict[str, int]], list[dict[str, int]]],
    ] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.records is None:
            self.records = {}

    def ensure_h(
        self, elements: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if (
            self.h is None
            or self.h.device != device
            or self.h.dtype != dtype
            or self.h.numel() < int(elements)
        ):
            self.h = torch.empty((int(elements),), device=device, dtype=dtype)
        return self.h[: int(elements)]


@triton.jit
def _shrink_group_kernel(
    x_ptr,
    a_ptr,
    h_ptr,
    group_row_starts_ptr,
    group_row_counts_ptr,
    group_branch_starts_ptr,
    group_branch_counts_ptr,
    branch_slots_ptr,
    branch_ranks_ptr,
    branch_h_offsets_ptr,
    branch_h_strides_ptr,
    input_hidden,
    a_slot_stride,
    a_rank_stride,
    a_hidden_stride,
    x_row_stride,
    x_hidden_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MAX_BRANCHES: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    group_id = tl.program_id(axis=1)
    row_start = tl.load(group_row_starts_ptr + group_id)
    row_count = tl.load(group_row_counts_ptr + group_id)
    branch_start = tl.load(group_branch_starts_ptr + group_id)
    branch_count = tl.load(group_branch_counts_ptr + group_id)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = row_start + row_offsets
    row_mask = row_offsets < row_count
    if pid_m * BLOCK_M >= row_count:
        return

    rank_offsets = tl.arange(0, BLOCK_R)
    hidden0 = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    hidden1 = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    hidden2 = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    active0 = branch_count > 0
    active1 = branch_count > 1
    active2 = branch_count > 2
    slot0 = tl.load(branch_slots_ptr + branch_start, mask=active0, other=0)
    slot1 = tl.load(branch_slots_ptr + branch_start + 1, mask=active1, other=0)
    slot2 = tl.load(branch_slots_ptr + branch_start + 2, mask=active2, other=0)
    rank0 = tl.load(branch_ranks_ptr + branch_start, mask=active0, other=0)
    rank1 = tl.load(branch_ranks_ptr + branch_start + 1, mask=active1, other=0)
    rank2 = tl.load(branch_ranks_ptr + branch_start + 2, mask=active2, other=0)

    for hidden_start in range(0, input_hidden, BLOCK_H):
        hidden_offsets = hidden_start + tl.arange(0, BLOCK_H)
        x_values = tl.load(
            x_ptr
            + rows[:, None] * x_row_stride
            + hidden_offsets[None, :] * x_hidden_stride,
            mask=row_mask[:, None] & (hidden_offsets[None, :] < input_hidden),
            other=0.0,
        )
        if MAX_BRANCHES >= 1:
            a0 = tl.load(
                a_ptr
                + slot0 * a_slot_stride
                + rank_offsets[None, :] * a_rank_stride
                + hidden_offsets[:, None] * a_hidden_stride,
                mask=active0
                & (rank_offsets[None, :] < rank0)
                & (hidden_offsets[:, None] < input_hidden),
                other=0.0,
            )
            hidden0 = tl.dot(x_values, a0, acc=hidden0)
        if MAX_BRANCHES >= 2:
            a1 = tl.load(
                a_ptr
                + slot1 * a_slot_stride
                + rank_offsets[None, :] * a_rank_stride
                + hidden_offsets[:, None] * a_hidden_stride,
                mask=active1
                & (rank_offsets[None, :] < rank1)
                & (hidden_offsets[:, None] < input_hidden),
                other=0.0,
            )
            hidden1 = tl.dot(x_values, a1, acc=hidden1)
        if MAX_BRANCHES >= 3:
            a2 = tl.load(
                a_ptr
                + slot2 * a_slot_stride
                + rank_offsets[None, :] * a_rank_stride
                + hidden_offsets[:, None] * a_hidden_stride,
                mask=active2
                & (rank_offsets[None, :] < rank2)
                & (hidden_offsets[:, None] < input_hidden),
                other=0.0,
            )
            hidden2 = tl.dot(x_values, a2, acc=hidden2)

    h_offset0 = tl.load(branch_h_offsets_ptr + branch_start, mask=active0, other=0)
    h_offset1 = tl.load(
        branch_h_offsets_ptr + branch_start + 1, mask=active1, other=0
    )
    h_offset2 = tl.load(
        branch_h_offsets_ptr + branch_start + 2, mask=active2, other=0
    )
    h_stride0 = tl.load(branch_h_strides_ptr + branch_start, mask=active0, other=0)
    h_stride1 = tl.load(
        branch_h_strides_ptr + branch_start + 1, mask=active1, other=0
    )
    h_stride2 = tl.load(
        branch_h_strides_ptr + branch_start + 2, mask=active2, other=0
    )
    if MAX_BRANCHES >= 1:
        tl.store(
            h_ptr + h_offset0 + row_offsets[:, None] * h_stride0 + rank_offsets[None, :],
            hidden0,
            mask=active0
            & row_mask[:, None]
            & (rank_offsets[None, :] < rank0),
        )
    if MAX_BRANCHES >= 2:
        tl.store(
            h_ptr + h_offset1 + row_offsets[:, None] * h_stride1 + rank_offsets[None, :],
            hidden1,
            mask=active1
            & row_mask[:, None]
            & (rank_offsets[None, :] < rank1),
        )
    if MAX_BRANCHES >= 3:
        tl.store(
            h_ptr + h_offset2 + row_offsets[:, None] * h_stride2 + rank_offsets[None, :],
            hidden2,
            mask=active2
            & row_mask[:, None]
            & (rank_offsets[None, :] < rank2),
        )


@triton.jit
def _expand_add_kernel(
    h_ptr,
    b_ptr,
    delta_ptr,
    group_row_starts_ptr,
    group_row_counts_ptr,
    group_branch_starts_ptr,
    group_branch_counts_ptr,
    branch_slots_ptr,
    branch_ranks_ptr,
    branch_scales_ptr,
    branch_h_offsets_ptr,
    branch_h_strides_ptr,
    output_hidden,
    b_slot_stride,
    b_output_stride,
    b_rank_stride,
    delta_row_stride,
    delta_output_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MAX_BRANCHES: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    group_id = tl.program_id(axis=2)
    row_start = tl.load(group_row_starts_ptr + group_id)
    row_count = tl.load(group_row_counts_ptr + group_id)
    branch_start = tl.load(group_branch_starts_ptr + group_id)
    branch_count = tl.load(group_branch_counts_ptr + group_id)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_offsets < row_count
    output_mask = output_offsets < output_hidden
    if pid_m * BLOCK_M >= row_count:
        return

    rank_offsets = tl.arange(0, BLOCK_R)
    residual = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for branch_index in range(MAX_BRANCHES):
        active = branch_index < branch_count
        metadata_index = branch_start + branch_index
        slot = tl.load(branch_slots_ptr + metadata_index, mask=active, other=0)
        rank = tl.load(branch_ranks_ptr + metadata_index, mask=active, other=0)
        scale = tl.load(branch_scales_ptr + metadata_index, mask=active, other=0.0)
        h_offset = tl.load(
            branch_h_offsets_ptr + metadata_index, mask=active, other=0
        )
        h_stride = tl.load(
            branch_h_strides_ptr + metadata_index, mask=active, other=0
        )
        hidden = tl.load(
            h_ptr
            + h_offset
            + row_offsets[:, None] * h_stride
            + rank_offsets[None, :],
            mask=active
            & row_mask[:, None]
            & (rank_offsets[None, :] < rank),
            other=0.0,
        )
        b_values = tl.load(
            b_ptr
            + slot * b_slot_stride
            + output_offsets[:, None] * b_output_stride
            + rank_offsets[None, :] * b_rank_stride,
            mask=active
            & output_mask[:, None]
            & (rank_offsets[None, :] < rank),
            other=0.0,
        )
        residual += tl.dot(hidden, tl.trans(b_values)) * scale

    tl.store(
        delta_ptr
        + (row_start + row_offsets[:, None]) * delta_row_stride
        + output_offsets[None, :] * delta_output_stride,
        residual,
        mask=row_mask[:, None] & output_mask[None, :],
    )


def _cached_device_int(
    cache: dict[tuple[str, tuple[object, ...], str], torch.Tensor],
    name: str,
    values: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    key = (str(name), tuple(int(value) for value in values), str(device))
    tensor = cache.get(key)
    if tensor is None:
        tensor = torch.tensor(key[1], device=device, dtype=torch.int32)
        cache[key] = tensor
    return tensor


def _cached_device_float(
    cache: dict[tuple[str, tuple[object, ...], str], torch.Tensor],
    name: str,
    values: Sequence[float],
    device: torch.device,
) -> torch.Tensor:
    normalized = tuple(float(value) for value in values)
    key = (str(name), tuple(value.hex() for value in normalized), str(device))
    tensor = cache.get(key)
    if tensor is None:
        tensor = torch.tensor(normalized, device=device, dtype=torch.float32)
        cache[key] = tensor
    return tensor


def build_two_stage_descriptor(
    *,
    module_name: str,
    request_ids: Sequence[str],
    row_layout: Sequence[int],
    relations: Sequence[Sequence[int]],
    scales: Sequence[Sequence[float]],
    rank_table: Sequence[int],
    output_offset: int,
    output_features: int,
    input_features: int,
) -> TetrisTwoStageDescriptor:
    """Build a validated descriptor before launching either stage."""
    del input_features
    normalized_layout = tuple(int(value) for value in row_layout)
    normalized_relations = tuple(
        tuple(int(slot) for slot in entries) for entries in relations
    )
    normalized_scales = tuple(
        tuple(float(scale) for scale in entries) for entries in scales
    )
    if len(normalized_layout) != len(normalized_relations):
        raise ValueError("row layout and relations must have equal request counts")
    if len(normalized_scales) != len(normalized_relations):
        raise ValueError("scales and relations must have equal request counts")
    if any(len(a) != len(b) for a, b in zip(normalized_relations, normalized_scales)):
        raise ValueError("relation and scale counts must match per request")
    if any(rows < 0 for rows in normalized_layout):
        raise ValueError("row layout cannot contain negative counts")
    normalized_ranks = []
    for entries in normalized_relations:
        request_ranks = []
        for slot in entries:
            if slot < 0 or slot >= len(rank_table):
                raise ValueError(f"LoRA slot {slot} exceeds rank metadata")
            rank = int(rank_table[slot])
            if rank < 0:
                raise ValueError("LoRA rank cannot be negative")
            request_ranks.append(rank)
        normalized_ranks.append(tuple(request_ranks))
    normalized_ranks = tuple(normalized_ranks)
    if any(not math.isfinite(scale) for entries in normalized_scales for scale in entries):
        raise ValueError("LoRA scales must be finite")
    if sum(normalized_layout) <= 0:
        raise ValueError("two-stage LoRA requires at least one input row")
    if len(request_ids) not in {0, len(normalized_layout)}:
        raise ValueError("request id count must match row layout")
    branch_count = sum(len(entries) for entries in normalized_relations)
    # K3: specialize MAX_BRANCHES to actual K (capped by env), not always 3.
    ceiling = _max_branch_group()
    actual_k = max((len(entries) for entries in normalized_relations), default=1)
    max_group = max(1, min(ceiling, actual_k))
    if any(len(entries) > ceiling for entries in normalized_relations):
        raise ValueError(
            "active LoRA branches exceed TETRIS_LORA_MAX_BRANCHES; "
            "increase the bounded group limit explicitly"
        )
    rank_blocks = tuple(
        sorted(
            {
                16 if rank <= 16 else _round_up(rank, 64)
                for ranks in normalized_ranks
                for rank in ranks
                if rank > 0
            }
        )
    )
    if any(rank_block > 128 for rank_block in rank_blocks):
        raise ValueError("ranks above 128 are not supported by the TC path")
    if int(output_offset) < 0 or int(output_features) <= 0:
        raise ValueError("output offset/features are invalid")
    h_elements = 0
    for rows, ranks in zip(normalized_layout, normalized_ranks):
        for rank in ranks:
            if rank <= 0:
                continue
            h_elements += int(rows) * (16 if rank <= 16 else _round_up(rank, 64))
    return TetrisTwoStageDescriptor(
        module_name=str(module_name),
        request_ids=tuple(str(value) for value in request_ids),
        row_layout=normalized_layout,
        relations=normalized_relations,
        scales=normalized_scales,
        ranks=normalized_ranks,
        output_offset=int(output_offset),
        output_features=int(output_features),
        h_bytes=h_elements * 2,
        branch_count=branch_count,
        group_count=len(normalized_layout),
        max_branch_group=max_group,
        rank_blocks=rank_blocks,
    )


def _branch_records(
    descriptor: TetrisTwoStageDescriptor,
) -> tuple[
    list[dict[str, int | float]],
    list[dict[str, int]],
    list[dict[str, int]],
]:
    records: list[dict[str, int | float]] = []
    shrink_groups: list[dict[str, int]] = []
    expand_groups: list[dict[str, int]] = []
    row_start = 0
    h_offset = 0
    for request_index, (row_count, slots, scales, ranks) in enumerate(
        zip(
            descriptor.row_layout,
            descriptor.relations,
            descriptor.scales,
            descriptor.ranks,
        )
    ):
        request_records: list[int] = []
        for slot, scale, rank in zip(slots, scales, ranks):
            if rank <= 0:
                continue
            padded_rank = 16 if rank <= 16 else _round_up(rank, 64)
            records.append(
                {
                    "request_index": request_index,
                    "row_start": row_start,
                    "row_count": int(row_count),
                    "slot": int(slot),
                    "rank": int(rank),
                    "scale": float(scale),
                    "h_offset": h_offset,
                    "h_stride": padded_rank,
                    "rank_block": padded_rank,
                }
            )
            request_records.append(len(records) - 1)
            h_offset += int(row_count) * padded_rank
        if request_records:
            expand_groups.append(
                {
                    "request_index": request_index,
                    "record_start": request_records[0],
                    "record_count": len(request_records),
                }
            )
        cursor = 0
        while cursor < len(request_records):
            first_index = request_records[cursor]
            rank_block = int(records[first_index]["rank_block"])
            end = cursor
            while end < len(request_records):
                record_index = request_records[end]
                if int(records[record_index]["rank_block"]) != rank_block:
                    break
                if end - cursor >= descriptor.max_branch_group:
                    break
                end += 1
            selected = request_records[cursor:end]
            shrink_groups.append(
                {
                    "request_index": request_index,
                    "record_start": selected[0],
                    "record_count": len(selected),
                }
            )
            cursor = end
        row_start += int(row_count)
    return records, shrink_groups, expand_groups


def _launch_shrink(
    x: torch.Tensor,
    a_stack: torch.Tensor,
    h: torch.Tensor,
    records: list[dict[str, int | float]],
    groups: list[dict[str, int]],
    rank_block: int,
    *,
    max_branch_group: int,
    metadata_cache: dict[tuple[str, tuple[int, ...], str], torch.Tensor],
) -> None:
    selected_groups = [
        group
        for group in groups
        if all(
            int(records[index]["rank_block"]) == rank_block
            for index in range(
                int(group["record_start"]),
                int(group["record_start"]) + int(group["record_count"]),
            )
        )
    ]
    if not selected_groups:
        return
    device = x.device
    group_rows = []
    group_counts = []
    group_branch_starts = []
    group_branch_counts = []
    branch_slots = []
    branch_ranks = []
    branch_h_offsets = []
    branch_h_strides = []
    for group in selected_groups:
        start = len(branch_slots)
        first = int(group["record_start"])
        count = int(group["record_count"])
        group_rows.append(int(records[first]["row_start"]))
        group_counts.append(int(records[first]["row_count"]))
        group_branch_starts.append(start)
        group_branch_counts.append(count)
        for index in range(first, first + count):
            record = records[index]
            branch_slots.append(int(record["slot"]))
            branch_ranks.append(int(record["rank"]))
            branch_h_offsets.append(int(record["h_offset"]))
            branch_h_strides.append(int(record["h_stride"]))
    block_m = int(os.environ.get("TETRIS_LORA_SHRINK_BLOCK_M", "32"))
    block_h = int(os.environ.get("TETRIS_LORA_SHRINK_BLOCK_H", "128"))
    grid = (triton.cdiv(max(group_counts), block_m), len(selected_groups))
    _shrink_group_kernel[grid](
        x,
        a_stack[:, 0],
        h,
        _cached_device_int(metadata_cache, "shrink.group_rows", group_rows, device),
        _cached_device_int(metadata_cache, "shrink.group_counts", group_counts, device),
        _cached_device_int(metadata_cache, "shrink.group_branch_starts", group_branch_starts, device),
        _cached_device_int(metadata_cache, "shrink.group_branch_counts", group_branch_counts, device),
        _cached_device_int(metadata_cache, "shrink.branch_slots", branch_slots, device),
        _cached_device_int(metadata_cache, "shrink.branch_ranks", branch_ranks, device),
        _cached_device_int(metadata_cache, "shrink.branch_h_offsets", branch_h_offsets, device),
        _cached_device_int(metadata_cache, "shrink.branch_h_strides", branch_h_strides, device),
        x.shape[1],
        a_stack.stride(0),
        a_stack.stride(2),
        a_stack.stride(3),
        x.stride(0),
        x.stride(1),
        BLOCK_M=block_m,
        BLOCK_R=rank_block,
        BLOCK_H=block_h,
        MAX_BRANCHES=max_branch_group,
        num_warps=int(os.environ.get("TETRIS_LORA_SHRINK_WARPS", "4")),
        num_stages=2,
    )


def _launch_expand(
    h: torch.Tensor,
    b_stack: torch.Tensor,
    delta: torch.Tensor,
    records: list[dict[str, int | float]],
    groups: list[dict[str, int]],
    *,
    max_branch_group: int,
    metadata_cache: dict[tuple[str, tuple[int, ...], str], torch.Tensor],
) -> None:
    device = delta.device
    group_rows = []
    group_counts = []
    group_branch_starts = []
    group_branch_counts = []
    branch_slots = []
    branch_ranks = []
    branch_scales = []
    branch_h_offsets = []
    branch_h_strides = []
    for group in groups:
        start = len(branch_slots)
        first = int(group["record_start"])
        count = int(group["record_count"])
        record = records[first]
        group_rows.append(int(record["row_start"]))
        group_counts.append(int(record["row_count"]))
        group_branch_starts.append(start)
        group_branch_counts.append(count)
        for index in range(first, first + count):
            record = records[index]
            branch_slots.append(int(record["slot"]))
            branch_ranks.append(int(record["rank"]))
            branch_scales.append(float(record["scale"]))
            branch_h_offsets.append(int(record["h_offset"]))
            branch_h_strides.append(int(record["h_stride"]))
    if not groups:
        return
    # K4: launch expand per rank_block so rank-2 groups are not inflated by
    # a global max_rank_block from mixed-rank batches.
    from collections import defaultdict

    groups_by_block: dict[int, list[int]] = defaultdict(list)
    for group_index, group in enumerate(groups):
        first = int(group["record_start"])
        groups_by_block[int(records[first]["rank_block"])].append(group_index)
    block_m = int(os.environ.get("TETRIS_LORA_EXPAND_BLOCK_M", "32"))
    block_n = int(os.environ.get("TETRIS_LORA_EXPAND_BLOCK_N", "128"))
    for rank_block, group_indices in sorted(groups_by_block.items()):
        sub_rows = [group_rows[i] for i in group_indices]
        sub_counts = [group_counts[i] for i in group_indices]
        sub_starts = [group_branch_starts[i] for i in group_indices]
        sub_bcounts = [group_branch_counts[i] for i in group_indices]
        grid = (
            triton.cdiv(max(sub_counts), block_m),
            triton.cdiv(delta.shape[1], block_n),
            len(group_indices),
        )
        cache_tag = f"expand.r{rank_block}"
        _expand_add_kernel[grid](
            h,
            b_stack[:, 0],
            delta,
            _cached_device_int(
                metadata_cache, f"{cache_tag}.group_rows", sub_rows, device
            ),
            _cached_device_int(
                metadata_cache, f"{cache_tag}.group_counts", sub_counts, device
            ),
            _cached_device_int(
                metadata_cache, f"{cache_tag}.group_branch_starts", sub_starts, device
            ),
            _cached_device_int(
                metadata_cache, f"{cache_tag}.group_branch_counts", sub_bcounts, device
            ),
            _cached_device_int(metadata_cache, "expand.branch_slots", branch_slots, device),
            _cached_device_int(metadata_cache, "expand.branch_ranks", branch_ranks, device),
            _cached_device_float(
                metadata_cache, "expand.branch_scales", branch_scales, device
            ),
            _cached_device_int(
                metadata_cache, "expand.branch_h_offsets", branch_h_offsets, device
            ),
            _cached_device_int(
                metadata_cache, "expand.branch_h_strides", branch_h_strides, device
            ),
            delta.shape[1],
            b_stack.stride(0),
            b_stack.stride(2),
            b_stack.stride(3),
            delta.stride(0),
            delta.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_R=int(rank_block),
            MAX_BRANCHES=max_branch_group,
            num_warps=int(os.environ.get("TETRIS_LORA_EXPAND_WARPS", "4")),
            num_stages=2,
        )


def _contiguous_from_row_spans(
    x: torch.Tensor,
    row_spans: Sequence[tuple[int, int, int]],
    n_requests: int,
) -> tuple[torch.Tensor, tuple[int, ...], list[tuple[int, int, int]]]:
    """Gather patch spans into request-contiguous X; return scatter map.

    Scatter entries are ``(contig_start, count, src_start)`` for writing Δ
    back onto the original row order.
    """
    if n_requests <= 0:
        raise ValueError("row_spans require a positive request count")
    per_req: list[list[tuple[int, int]]] = [[] for _ in range(n_requests)]
    covered: list[tuple[int, int]] = []
    for start, count, request_index in row_spans:
        start_i = int(start)
        count_i = int(count)
        req_i = int(request_index)
        if (
            start_i < 0
            or count_i <= 0
            or start_i + count_i > x.shape[0]
            or not 0 <= req_i < n_requests
        ):
            raise ValueError(
                f"invalid two_stage row span "
                f"(start={start_i}, count={count_i}, request={req_i})"
            )
        span_end = start_i + count_i
        for prev_start, prev_end in covered:
            if start_i < prev_end and prev_start < span_end:
                raise ValueError(
                    "overlapping LoRA row spans are unsupported "
                    f"({start_i}:{span_end} vs {prev_start}:{prev_end})"
                )
        covered.append((start_i, span_end))
        per_req[req_i].append((start_i, count_i))
    chunks: list[torch.Tensor] = []
    scatter: list[tuple[int, int, int]] = []
    layout: list[int] = []
    cursor = 0
    for req_i in range(n_requests):
        n_rows = 0
        for start_i, count_i in per_req[req_i]:
            chunks.append(x[start_i : start_i + count_i])
            scatter.append((cursor, count_i, start_i))
            cursor += count_i
            n_rows += count_i
        layout.append(n_rows)
    if not chunks:
        empty = x.new_empty((0, x.shape[1]))
        return empty, tuple(layout), scatter
    return torch.cat(chunks, dim=0), tuple(layout), scatter


def execute_two_stage_lora(
    x: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    row_layout: Sequence[int],
    relations: Sequence[Sequence[int]],
    scales: Sequence[Sequence[float]],
    rank_table: Sequence[int],
    *,
    module_name: str = "",
    request_ids: Sequence[str] = (),
    output: torch.Tensor | None = None,
    output_offset: int = 0,
    descriptor: TetrisTwoStageDescriptor | None = None,
    workspace: TetrisTwoStageWorkspace | None = None,
    clear_output: bool = True,
    row_spans: Sequence[tuple[int, int, int]] | None = None,
) -> tuple[torch.Tensor, TetrisTwoStageDescriptor]:
    """Run compact BF16 H shrink followed by TC Expand+Add into pure delta.

    When ``row_spans`` is set (MixFusion ``patch_row_spans`` ownership), rows
    are gathered into request-contiguous order, the contiguous path runs, then
    Δ is scattered back. ``row_layout`` may still describe request sizes but is
    recomputed from spans when spans are provided.
    """
    if x.ndim != 2 or not x.is_cuda:
        raise ValueError("Tetris two-stage LoRA expects a 2D CUDA input")
    if a_stack.ndim != 4 or b_stack.ndim != 4:
        raise ValueError("LoRA stacks must have shape [slot, slice, dim, dim]")
    if a_stack.dtype != x.dtype or b_stack.dtype != x.dtype:
        raise ValueError("Tetris two-stage LoRA requires matching BF16 tensors")
    scatter_map: list[tuple[int, int, int]] | None = None
    x_work = x
    layout_work: Sequence[int] = row_layout
    out_full = output
    if row_spans is not None:
        if descriptor is not None:
            raise ValueError(
                "two_stage row_spans path builds its own descriptor; "
                "do not pass a prebuilt descriptor"
            )
        x_work, layout_work, scatter_map = _contiguous_from_row_spans(
            x, row_spans, n_requests=len(relations)
        )
        if out_full is None:
            out_full = torch.zeros(
                (x.shape[0], b_stack.shape[2]), device=x.device, dtype=x.dtype
            )
        elif clear_output:
            out_full.zero_()
        # Contiguous delta workspace; scatter into out_full after.
        output = None
        clear_output = True
    if (
        len(layout_work) != len(relations)
        or sum(int(rows) for rows in layout_work) != x_work.shape[0]
    ):
        raise ValueError("row layout does not cover the flattened input")
    if a_stack.shape[1] != 1 or b_stack.shape[1] != 1:
        raise ValueError("the two-stage TC path currently accepts one projection slice")
    if a_stack.shape[3] != x_work.shape[1]:
        raise ValueError("LoRA A input width does not match X")
    if b_stack.shape[3] != a_stack.shape[2]:
        raise ValueError("LoRA B rank capacity does not match LoRA A")
    if output is None:
        if int(output_offset) != 0:
            raise ValueError("nonzero output offsets require an output view")
        delta = torch.zeros(
            (x_work.shape[0], b_stack.shape[2]),
            device=x_work.device,
            dtype=x_work.dtype,
        )
    else:
        if output.ndim != 2 or tuple(output.shape) != (
            x_work.shape[0],
            b_stack.shape[2],
        ):
            raise ValueError("output shape does not match the selected LoRA slice")
        if int(output_offset) != 0:
            raise ValueError("nonzero output offsets require an output view")
        delta = output
        if clear_output:
            delta.zero_()
    if descriptor is None:
        descriptor = build_two_stage_descriptor(
            module_name=module_name,
            request_ids=request_ids,
            row_layout=layout_work,
            relations=relations,
            scales=scales,
            rank_table=rank_table,
            output_offset=output_offset,
            output_features=int(delta.shape[1]),
            input_features=int(x_work.shape[1]),
        )
    elif descriptor.output_features != int(delta.shape[1]):
        raise ValueError("provided descriptor output width does not match output")
    if descriptor.output_offset != int(output_offset):
        raise ValueError("provided descriptor output offset does not match output")
    for ranks in descriptor.ranks:
        for rank in ranks:
            if rank > a_stack.shape[2] or rank > b_stack.shape[3]:
                raise ValueError("LoRA rank exceeds resident stack capacity")
    if descriptor.branch_count == 0:
        return delta, descriptor
    active_slots = [slot for entries in descriptor.relations for slot in entries]
    if max(active_slots, default=-1) >= a_stack.shape[0]:
        raise ValueError("LoRA slot exceeds resident stack")
    if workspace is None:
        workspace = TetrisTwoStageWorkspace()
    cached_records = workspace.records.get(descriptor)
    if cached_records is None:
        cached_records = _branch_records(descriptor)
        workspace.records[descriptor] = cached_records
    records, shrink_groups, expand_groups = cached_records
    h = workspace.ensure_h(
        descriptor.h_bytes // 2, x_work.device, x_work.dtype
    )
    for rank_block in descriptor.rank_blocks:
        _launch_shrink(
            x_work,
            a_stack,
            h,
            records,
            shrink_groups,
            rank_block,
            max_branch_group=descriptor.max_branch_group,
            metadata_cache=workspace.metadata,
        )
    _launch_expand(
        h,
        b_stack,
        delta,
        records,
        expand_groups,
        max_branch_group=descriptor.max_branch_group,
        metadata_cache=workspace.metadata,
    )
    if scatter_map is not None:
        assert out_full is not None
        for contig_start, count, src_start in scatter_map:
            out_full[src_start : src_start + count].copy_(
                delta[contig_start : contig_start + count]
            )
        return out_full, descriptor
    return delta, descriptor


__all__ = [
    "TetrisTwoStageDescriptor",
    "TetrisTwoStageWorkspace",
    "build_two_stage_descriptor",
    "execute_two_stage_lora",
]
