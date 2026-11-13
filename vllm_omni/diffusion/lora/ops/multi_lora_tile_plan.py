"""Vendored into Omni for official-pip vllm.

Provenance: copied from parent `vllm/lora/ops/triton_ops/multi_lora_tile_plan.py` (cyn fork helper).

Reference tile-plan builder for the diffusion Multi-LoRA contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MultiLoRATilePlan:
    row_layout: tuple[int, ...]
    relations: tuple[tuple[int, ...], ...]
    scales: tuple[tuple[float, ...], ...]
    rank_signature: tuple[int, ...]
    output_size: int
    tile_size: int


def build_multi_lora_tile_plan(
    row_layout,
    relations,
    scales,
    rank_signature,
    output_size,
    device,
    tile_size=64,
):
    del device
    return MultiLoRATilePlan(
        tuple(row_layout), tuple(tuple(x) for x in relations),
        tuple(tuple(x) for x in scales), tuple(rank_signature),
        int(output_size), int(tile_size),
    )
