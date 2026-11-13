#!/usr/bin/env python3
"""Measure the two-stage TC Multi-LoRA operator on FLUX.1.D QKV shapes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time

import torch

from vllm_omni.diffusion.lora.tetris_lora_twostage import (
    TetrisTwoStageWorkspace,
    build_two_stage_descriptor,
    execute_two_stage_lora,
)


def _token_rows(batch: int, resolution: int) -> int:
    image_tokens = (resolution // 16) ** 2
    text_tokens = 512
    return batch * (image_tokens + text_tokens)


def _p90(samples: list[float]) -> float:
    if len(samples) < 2:
        return float(samples[0])
    return float(statistics.quantiles(samples, n=10, method="inclusive")[8])


def _measure(
    x: torch.Tensor,
    a_stack: torch.Tensor,
    b_stack: torch.Tensor,
    row_layout: tuple[int, ...],
    relations: tuple[tuple[int, ...], ...],
    scales: tuple[tuple[float, ...], ...],
    rank_table: tuple[int, ...],
    warmup: int,
    iterations: int,
) -> tuple[list[float], object, dict[str, object]]:
    descriptor = build_two_stage_descriptor(
        module_name="transformer_blocks.0.attn.to_qkv",
        request_ids=tuple(f"request-{index}" for index in range(len(row_layout))),
        row_layout=row_layout,
        relations=relations,
        scales=scales,
        rank_table=rank_table,
        output_offset=0,
        output_features=int(b_stack.shape[2]),
        input_features=int(x.shape[1]),
    )
    workspace = TetrisTwoStageWorkspace()
    h = workspace.ensure_h(descriptor.h_bytes // 2, x.device, x.dtype)
    del h
    delta = torch.empty(
        (x.shape[0], b_stack.shape[2]), device=x.device, dtype=x.dtype
    )
    for _ in range(warmup):
        _, descriptor = execute_two_stage_lora(
            x,
            a_stack,
            b_stack,
            row_layout,
            relations,
            scales,
            rank_table,
            descriptor=descriptor,
            workspace=workspace,
            output=delta,
            clear_output=False,
        )
    torch.cuda.synchronize(x.device)
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(x.device))
        execute_two_stage_lora(
            x,
            a_stack,
            b_stack,
            row_layout,
            relations,
            scales,
            rank_table,
            descriptor=descriptor,
            workspace=workspace,
            output=delta,
            clear_output=False,
        )
        end.record(torch.cuda.current_stream(x.device))
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    assert descriptor is not None
    return samples, descriptor, {
        "descriptor_prepared": True,
        "workspace_preallocated": True,
        "output_preallocated": True,
        "allocations_excluded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    input_features = 3072
    output_features = 9216
    max_rank = 64
    rows_by_resolution = {
        512: _token_rows(1, 512),
        1024: _token_rows(1, 1024),
    }
    generator = torch.Generator(device=device).manual_seed(20261111)
    results: list[dict[str, object]] = []
    for batch in (1, 4):
        for resolution in (512, 1024):
            rows_per_request = rows_by_resolution[resolution]
            rows = batch * rows_per_request
            row_layout = (rows_per_request,) * batch
            x = torch.randn(
                rows, input_features,
                device=device, dtype=torch.bfloat16, generator=generator
            )
            for rank in (2, 64):
                for branches in (1, 3):
                    rank_table = (rank,) * branches
                    a_stack = torch.randn(
                        branches, 1, max_rank, input_features,
                        device=device, dtype=torch.bfloat16, generator=generator
                    )
                    b_stack = torch.randn(
                        branches, 1, output_features, max_rank,
                        device=device, dtype=torch.bfloat16, generator=generator
                    )
                    relations = (tuple(range(branches)),) * batch
                    scales = (
                        tuple(1.0 - 0.1 * branch for branch in range(branches)),
                    ) * batch
                    samples, descriptor, timing_flags = _measure(
                        x, a_stack, b_stack, row_layout, relations, scales,
                        rank_table, args.warmup, args.iterations
                    )
                    padded_rank = 16 if rank <= 16 else 64
                    effective_flops = 2 * rows * branches * rank * (
                        input_features + output_features
                    )
                    padded_flops = 2 * rows * branches * padded_rank * (
                        input_features + output_features
                    )
                    results.append(
                        {
                            "batch": batch,
                            "resolution": resolution,
                            "rows": rows,
                            "rows_per_request": rows_per_request,
                            "rank": rank,
                            "branches": branches,
                            "input_features": input_features,
                            "output_features": output_features,
                            "dtype": "bfloat16",
                            "h_bytes": descriptor.h_bytes,
                            "h_write_bytes": descriptor.h_bytes,
                            "h_read_bytes": descriptor.h_bytes,
                            "h_total_bytes": 2 * descriptor.h_bytes,
                            "rank_blocks": list(descriptor.rank_blocks),
                            "effective_flops": effective_flops,
                            "padded_mma_flops": padded_flops,
                            "theoretical_mma_utilization": effective_flops / padded_flops,
                            "median_ms": statistics.median(samples),
                            "p90_ms": _p90(samples),
                            "samples_ms": samples,
                            **timing_flags,
                        }
                    )
                    del a_stack, b_stack
                    torch.cuda.synchronize(device)

    report = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "bfloat16",
        "shape_source": "FLUX.1.D canonical QKV hidden=3072, output=3*3072; image tokens=(resolution/16)^2, text tokens=512",
        "timing": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "cuda_event": True,
            "allocations_excluded": True,
            "descriptor_and_workspace_prepared_before_events": True,
            "compile_and_weight_load_excluded": True,
        },
        "results": results,
        "created_unix": time.time(),
    }
    with open(args.output, "x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    csv_path = args.output.removesuffix(".json") + ".csv"
    with open(csv_path, "x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})


if __name__ == "__main__":
    main()
