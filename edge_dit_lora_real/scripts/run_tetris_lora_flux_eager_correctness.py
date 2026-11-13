#!/usr/bin/env python
"""Compare native and TetrisLoRA FLUX output on one real adapter request."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


def _generate(
    model: Path,
    deploy_config: Path,
    adapter: Path,
    output_path: Path,
    *,
    enabled: bool,
    width: int,
    height: int,
    steps: int,
    seed: int,
    prompt: str,
    compile_mode: str,
    base_device: int,
    lora_device: int,
    all_targets: bool,
    implementation: str,
    base_tiling: bool,
    sample_nvlink: bool = False,
    repeats: int = 3,
    batch_size: int = 1,
    mixfusion_schedule: bool = False,
    patch_grid: int | None = None,
    token_tile_size: int | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if base_tiling:
        raise ValueError(
            "VLLM_OMNI_BASE_TILING token-tile path was removed; use "
            "--mixfusion-schedule / --patch-grid instead"
        )
    # If Slurm/Workflow B already constrained visibility, keep ordinals 0/1
    # inside that set. Otherwise pin the requested physical pair.
    existing = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not existing or existing == "NoDevFiles":
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{base_device},{lora_device}"
    if compile_mode == "eager":
        os.environ["VLLM_OMNI_DISABLE_DIFFUSION_COMPILE"] = "1"
    else:
        os.environ.pop("VLLM_OMNI_DISABLE_DIFFUSION_COMPILE", None)
    os.environ["VLLM_OMNI_TETRIS_LORA"] = "1" if enabled else "0"
    os.environ["VLLM_OMNI_MULTI_LORA_IMPL"] = implementation
    os.environ["VLLM_OMNI_BASE_TILING"] = "0"
    if mixfusion_schedule:
        os.environ["VLLM_OMNI_MIXFUSION_SCHEDULE"] = "1"
        os.environ["VLLM_OMNI_MIXFUSION_NATIVE_SOURCE_CALL"] = "1"
        if patch_grid is not None:
            os.environ["VLLM_OMNI_MIXFUSION_PATCH_GRID"] = str(int(patch_grid))
    else:
        os.environ["VLLM_OMNI_MIXFUSION_SCHEDULE"] = "0"
        os.environ["VLLM_OMNI_MIXFUSION_NATIVE_SOURCE_CALL"] = "0"
        os.environ.pop("VLLM_OMNI_MIXFUSION_PATCH_GRID", None)
    if token_tile_size is not None:
        os.environ["VLLM_OMNI_MULTI_LORA_TILE_SIZE"] = str(int(token_tile_size))
    os.environ["TETRIS_LORA_ALL_TARGETS"] = "1" if all_targets else "0"
    os.environ["TETRIS_LORA_BASE_GPU"] = "0"
    os.environ["TETRIS_LORA_LORA_GPU"] = "1"
    stats_path = output_path.with_suffix(".tetris.jsonl")
    if stats_path.exists():
        raise FileExistsError(f"refusing to overwrite {stats_path}")
    os.environ["TETRIS_LORA_STATS_PATH"] = str(stats_path)
    telemetry_dir = output_path.parent / ("telemetry_tetris" if enabled else "telemetry_native")
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_OMNI_EXPERIMENT_TELEMETRY_DIR"] = str(telemetry_dir)
    os.environ["VLLM_OMNI_LORA_COMPUTE_BREAKDOWN"] = "1"

    from vllm.lora.request import LoRARequest
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    prompts = [
        {"prompt": prompt if index == 0 else f"{prompt} variant {index}"}
        for index in range(batch_size)
    ]
    omni = Omni(model=str(model), deploy_config=str(deploy_config))
    try:
        lora_request = LoRARequest(
            lora_name=adapter.name,
            lora_int_id=1001,
            lora_path=str(adapter),
        )
        add_results = omni.engine.collective_rpc(
            method="add_lora", args=(lora_request,)
        )
        if not add_results or not all(bool(result) for result in add_results):
            raise RuntimeError(f"Omni adapter registration failed: {add_results}")
        sampling = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,
            seed=seed,
            num_outputs_per_prompt=1,
            lora_request=lora_request,
            lora_scale=1.0,
            extra_args={},
        )
        omni.generate(prompts, sampling, use_tqdm=False)
        repeat_root = output_path.parent / f"{output_path.stem}_repeats"
        repeat_root.mkdir(parents=True, exist_ok=False)
        samples: list[dict[str, object]] = []
        first_image: np.ndarray | None = None
        for repeat_index in range(repeats):
            for telemetry_file in telemetry_dir.glob("*.jsonl"):
                telemetry_file.unlink()
            stats_path.write_text("", encoding="utf-8")
            nvlink_before = (
                _nvlink_counters(base_device, lora_device)
                if sample_nvlink
                else None
            )
            started = time.perf_counter()
            step_wall_start_ns = time.perf_counter_ns()
            outputs = omni.generate(prompts, sampling, use_tqdm=False)
            step_wall_end_ns = time.perf_counter_ns()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            nvlink_after = (
                _nvlink_counters(base_device, lora_device)
                if sample_nvlink
                else None
            )
            if len(outputs) != batch_size:
                raise RuntimeError(
                    f"FLUX generation returned {len(outputs)} outputs, "
                    f"expected batch_size={batch_size}"
                )
            if any(output.error is not None for output in outputs):
                raise RuntimeError(f"FLUX generation failed: {outputs}")
            image = np.asarray(outputs[0].images[0]).astype(np.float32)
            if first_image is None:
                first_image = image
                outputs[0].images[0].save(output_path)
            repeat_dir = repeat_root / f"repeat_{repeat_index + 1:02d}"
            repeat_dir.mkdir()
            outputs[0].images[0].save(repeat_dir / output_path.name)
            for batch_index, output in enumerate(outputs):
                output.images[0].save(
                    repeat_dir / f"batch{batch_index:02d}_{output_path.name}"
                )
            repeat_telemetry = repeat_dir / telemetry_dir.name
            shutil.copytree(telemetry_dir, repeat_telemetry)
            repeat_stats_path = repeat_dir / stats_path.name
            shutil.copy2(stats_path, repeat_stats_path)
            samples.append(
                {
                    "repeat": repeat_index + 1,
                    "step_ms": elapsed_ms,
                    "step_wall_start_ns": int(step_wall_start_ns),
                    "step_wall_end_ns": int(step_wall_end_ns),
                    "step_wall_ms": float(elapsed_ms),
                    "batch_size": batch_size,
                    "image": image,
                    "nvlink": (
                        _nvlink_report(
                            nvlink_before,
                            nvlink_after,
                            elapsed_ms,
                            base_device,
                            lora_device,
                        )
                        if nvlink_before is not None and nvlink_after is not None
                        else None
                    ),
                    "output_dir": str(repeat_dir),
                    "stats_path": str(repeat_stats_path),
                    "telemetry_dir": str(repeat_telemetry),
                }
            )
        if first_image is None:
            raise RuntimeError("no measured repeats were collected")
        return first_image, samples
    finally:
        omni.close()


def _read_breakdown(
    output_path: Path | None = None, telemetry_dir: Path | None = None
) -> dict[str, object]:
    if telemetry_dir is None:
        if output_path is None:
            raise ValueError("output_path or telemetry_dir is required")
        telemetry_dir = output_path.parent / (
            "telemetry_tetris"
            if output_path.name == "tetris.png"
            else "telemetry_native"
        )
    path = telemetry_dir / "lora_compute_breakdown_events.jsonl"
    rows = []
    if path.exists():
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    intervals = [
        row for row in rows if row.get("event") == "lora_compute_breakdown_interval"
    ]
    step_rows = [
        row for row in rows if row.get("event") == "lora_compute_breakdown_step"
    ]
    operation_totals: dict[str, float] = {}
    for row in intervals:
        operation = row.get("operation")
        duration = row.get("cuda_duration_ms")
        if isinstance(operation, str) and isinstance(duration, (int, float)):
            operation_totals[operation] = operation_totals.get(operation, 0.0) + float(duration)
    return {
        "path": str(path),
        "interval_count": len(intervals),
        "operation_totals_ms": operation_totals,
        "last_step": step_rows[-1] if step_rows else None,
    }


def _union_intervals_ms(intervals: list[tuple[int, int]]) -> float:
    """Return wall occupancy of half-open [start_ns, end_ns) intervals in ms."""
    cleaned = [
        (int(start), int(end))
        for start, end in intervals
        if int(end) > int(start)
    ]
    if not cleaned:
        return 0.0
    cleaned.sort()
    total_ns = 0
    cur_start, cur_end = cleaned[0]
    for start, end in cleaned[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
            continue
        total_ns += cur_end - cur_start
        cur_start, cur_end = start, end
    total_ns += cur_end - cur_start
    return float(total_ns) / 1_000_000.0


def _read_tetris_breakdown(path: Path) -> dict[str, object]:
    phases: dict[str, object] = {}
    payloads: dict[str, int] = {}
    counts: dict[str, int] = {}
    intervals: dict[str, list[tuple[int, int]]] = {}
    if not path.exists():
        return phases
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("kind") != "transport_phase":
            continue
        phase = event.get("phase")
        duration_us = event.get("duration_us")
        if isinstance(phase, str) and isinstance(duration_us, (int, float)):
            phases[phase] = float(phases.get(phase, 0.0)) + float(duration_us) / 1000.0
            counts[phase] = counts.get(phase, 0) + 1
            payload_bytes = event.get("payload_bytes")
            if isinstance(payload_bytes, int):
                payloads[phase] = payloads.get(phase, 0) + payload_bytes
            start_ns = event.get("start_ns")
            end_ns = event.get("end_ns")
            if isinstance(start_ns, int) and isinstance(end_ns, int) and end_ns > start_ns:
                intervals.setdefault(phase, []).append((start_ns, end_ns))
    phases["_payload_bytes"] = payloads
    phases["_counts"] = counts
    phases["_interval_union_ms"] = {
        phase: _union_intervals_ms(spans) for phase, spans in intervals.items()
    }
    phases["_intervals"] = intervals
    return phases


def _critical_path_breakdown(
    phases: dict[str, object],
    *,
    tetris_step_ms: float,
    coupled_step_ms: float,
) -> dict[str, object]:
    """Build sum vs critical-path fields from transport phase aggregates.

    Base and Worker use separate monotonic clocks; only same-process interval
    unions are treated as critical-path wall occupancy. CUDA Event phase sums
    remain cumulative upper bounds and are labeled as such.
    """
    unions = phases.get("_interval_union_ms")
    if not isinstance(unions, dict):
        unions = {}

    def sum_ms(name: str) -> float:
        value = phases.get(name, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0

    def union_ms(name: str) -> float:
        value = unions.get(name, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0

    control_ack_sum = sum_ms("control_ack")
    ready_wait_sum = sum_ms("remote_compute_ready_wait")
    ready_wait_cuda_sum = sum_ms("remote_compute_ready_wait_cuda")
    ready_span_cuda_sum = sum_ms("remote_compute_ready_span")
    recv_wait_sum = sum_ms("control_receive_wait")
    remote_compute_sum = sum_ms("remote_compute")
    decoupled_lora_gemm_sum = sum_ms("pure_lora_gemm")
    x_peer_sum = sum_ms("x_peer_copy")
    delta_p2p_sum = sum_ms("delta_p2p_copy")
    publish_sum = sum_ms("control_publish")
    merge_sum = sum_ms("merge")

    # Base-side host critical path (same monotonic domain).
    base_host_blocking = _union_intervals_ms(
        list(phases.get("_intervals", {}).get("operation_host_total", []))
        if isinstance(phases.get("_intervals"), dict)
        else []
    )
    remote_ready_cp = union_ms("remote_compute_ready_wait")
    publish_cp = union_ms("control_publish")
    delta_recv_cp = union_ms("delta_descriptor_receive") or union_ms(
        "delta_ipc_receive"
    )
    merge_cp = union_ms("merge")
    control_cp = publish_cp + delta_recv_cp
    wait_cp = remote_ready_cp

    # Worker-side host blocking (worker clock only; not additive to Base E2E).
    worker_intervals: list[tuple[int, int]] = []
    interval_map = phases.get("_intervals")
    if isinstance(interval_map, dict):
        for name in ("control_receive_wait", "control_ack"):
            spans = interval_map.get(name, [])
            if isinstance(spans, list):
                worker_intervals.extend(spans)
    worker_host_blocking = _union_intervals_ms(worker_intervals)
    control_receive_cp = union_ms("control_receive_wait")

    # CUDA Event sums are cumulative GPU work, not interval-union wall.
    # Critical-path comm uses the sum as an upper-bound proxy and is excluded
    # from ready_wait to avoid claiming both nested wait and GPU copy.
    critical_path_x_peer = x_peer_sum
    critical_path_delta_p2p = delta_p2p_sum
    critical_path_comm = critical_path_x_peer + critical_path_delta_p2p
    critical_path_remote_compute = remote_compute_sum
    critical_path_control = control_cp
    critical_path_wait = wait_cp
    critical_path_merge = merge_cp
    # Measured non-comm CP pieces on Base host timeline (publish/recv/wait/merge).
    # Remote CUDA compute is nested under ready_wait from Base's POV — do not
    # add remote_compute_sum into noncomm (would double-count with wait_cp).
    critical_path_noncomm_measured = (
        critical_path_control + critical_path_wait + critical_path_merge
    )
    overhead = float(tetris_step_ms) - float(coupled_step_ms)
    # CUDA Event sums (comm / remote_compute) are nested under Base ready_wait
    # from the Base POV. Subtracting both wait_cp and CUDA sums double-counts.
    # Primary unexplained closes against Base host union pieces only.
    unexplained = overhead - critical_path_noncomm_measured
    unexplained_with_cuda_proxy = (
        overhead - critical_path_comm - critical_path_noncomm_measured
    )
    critical_path_other = max(
        0.0,
        base_host_blocking
        - critical_path_control
        - critical_path_wait
        - critical_path_merge,
    )

    return {
        "scope_note": {
            "sum_ms": "cumulative_per_operation",
            "critical_path_host_union": "same_process_monotonic_interval_union",
            "critical_path_cuda_sums": (
                "upper_bound_proxy_not_interval_union; nested under ready_wait "
                "from Base POV for X-path work"
            ),
            "cross_process": (
                "Worker and Base monotonic clocks are not comparable; Worker "
                "unions are not added into Base E2E critical path"
            ),
        },
        "control_ack_sum_ms": control_ack_sum,
        "remote_compute_ready_wait_sum_ms": ready_wait_sum,
        "control_receive_wait_sum_ms": recv_wait_sum,
        "remote_compute_sum_ms": remote_compute_sum,
        "decoupled_lora_gemm_cuda_ms": decoupled_lora_gemm_sum,
        "remote_compute_ready_wait_cuda_ms": ready_wait_cuda_sum,
        "remote_compute_ready_span_cuda_ms": ready_span_cuda_sum,
        "wait_comm_ms": ready_span_cuda_sum - decoupled_lora_gemm_sum,
        "x_peer_copy_sum_ms": x_peer_sum,
        "delta_p2p_copy_sum_ms": delta_p2p_sum,
        "control_publish_sum_ms": publish_sum,
        "merge_sum_ms": merge_sum,
        "base_host_blocking_ms": base_host_blocking,
        "worker_host_blocking_ms": worker_host_blocking,
        "gpu_dependency_wait_ms": remote_ready_cp,
        "remote_ready_critical_path_ms": remote_ready_cp,
        "remote_compute_ready_wait_critical_path_ms": remote_ready_cp,
        "control_receive_critical_path_ms": control_receive_cp,
        "critical_path_x_peer_copy_ms": critical_path_x_peer,
        "critical_path_delta_p2p_copy_ms": critical_path_delta_p2p,
        "critical_path_p2p_ms": critical_path_comm,
        "critical_path_comm_ms": critical_path_comm,
        "critical_path_remote_compute_ms": critical_path_remote_compute,
        "critical_path_control_ms": critical_path_control,
        "critical_path_wait_ms": critical_path_wait,
        "critical_path_merge_ms": critical_path_merge,
        "critical_path_other_ms": critical_path_other,
        "critical_path_noncomm_ms": critical_path_noncomm_measured,
        "tetris_step_ms": float(tetris_step_ms),
        "coupled_step_ms": float(coupled_step_ms),
        "overhead_ms": overhead,
        "unexplained_ms": unexplained,
        "unexplained_with_cuda_proxy_ms": unexplained_with_cuda_proxy,
        "unexplained_ms_note": (
            "unexplained_ms = overhead - critical_path_noncomm "
            "(Base host unions: control+wait+merge). CUDA comm/remote sums are "
            "reported separately and nested under ready_wait — not subtracted "
            "again. Negative values mean host CP pieces exceed overhead because "
            "ready_wait replaces part of coupled LoRA compute rather than "
            "stacking purely on top; do not force to 0. "
            "unexplained_with_cuda_proxy_ms also subtracts CUDA sum proxies and "
            "will over-count when wait_cp > 0."
        ),
    }


def _read_denoise_timing(telemetry_dir: Path) -> dict[str, float]:
    path = telemetry_dir / "dit_events.jsonl"
    if not path.exists():
        return {}
    end_rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "dit_step_end":
            end_rows.append(event)
    if not end_rows:
        return {}
    row = end_rows[-1]
    result = {}
    for key in ("denoise_elapsed_ms", "denoise_cuda_event_ms"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result


def _median_sample(samples: list[dict[str, object]]) -> dict[str, object]:
    return sorted(samples, key=lambda row: float(row["step_ms"]))[len(samples) // 2]


def _sample_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    values = [float(row["step_ms"]) for row in samples]
    return {
        "samples_ms": values,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _nvlink_counters(*devices: int) -> dict[int, dict[int, dict[str, int]]]:
    counters: dict[int, dict[int, dict[str, int]]] = {}
    for device in devices:
        output = subprocess.check_output(
            ["nvidia-smi", "nvlink", "-i", str(device), "-gt", "d"],
            text=True,
        )
        links: dict[int, dict[str, int]] = {}
        for line in output.splitlines():
            if "Data Tx:" not in line and "Data Rx:" not in line:
                continue
            link = int(line.split("Link", 1)[1].split(":", 1)[0].strip())
            direction = "tx" if "Data Tx:" in line else "rx"
            value = int(line.rsplit(":", 1)[1].strip().split()[0])
            links.setdefault(link, {})[direction] = value
        if len(links) != 8 or any(set(value) != {"tx", "rx"} for value in links.values()):
            raise RuntimeError(f"incomplete NVLink counters for physical GPU {device}")
        counters[device] = links
    return counters


def _nvlink_report(
    before: dict[int, dict[int, dict[str, int]]],
    after: dict[int, dict[int, dict[str, int]]],
    elapsed_ms: float,
    base_device: int,
    lora_device: int,
) -> dict[str, object]:
    per_gpu: dict[str, object] = {}
    endpoint_bytes: dict[int, dict[str, int]] = {}
    for device in (base_device, lora_device):
        links = {}
        endpoint_bytes[device] = {"tx": 0, "rx": 0}
        for link in sorted(before[device]):
            tx_kib = after[device][link]["tx"] - before[device][link]["tx"]
            rx_kib = after[device][link]["rx"] - before[device][link]["rx"]
            links[str(link)] = {"tx_kib": tx_kib, "rx_kib": rx_kib}
            endpoint_bytes[device]["tx"] += tx_kib * 1024
            endpoint_bytes[device]["rx"] += rx_kib * 1024
        per_gpu[str(device)] = links
    seconds = elapsed_ms / 1000.0
    per_direction_capacity = 8 * 25_000_000_000
    return {
        "unit": "KiB",
        "devices": per_gpu,
        "per_endpoint_bytes": endpoint_bytes,
        "per_endpoint_gib_s": {
            str(device): {
                direction: byte_count / 2**30 / seconds if seconds else 0.0
                for direction, byte_count in values.items()
            }
            for device, values in endpoint_bytes.items()
        },
        "per_endpoint_utilization_of_8x25GBs": {
            str(device): {
                direction: byte_count / seconds / per_direction_capacity
                if seconds
                else 0.0
                for direction, byte_count in values.items()
            }
            for device, values in endpoint_bytes.items()
        },
        "measurement_window_ms": elapsed_ms,
        "attribution": "full measured decoupled denoise; includes GPU-side cross-device traffic and any unrelated NVLink activity",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--compile-mode",
        choices=("eager", "regional"),
        default="eager",
        help="Use the independent eager override or the existing regional config.",
    )
    parser.add_argument(
        "--base-device",
        type=int,
        default=5,
        help="Physical GPU for the Omni/Base process.",
    )
    parser.add_argument(
        "--lora-device",
        type=int,
        default=6,
        help="Physical GPU for the external LoRA worker.",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Dispatch every real adapter target instead of only the first target.",
    )
    parser.add_argument(
        "--native-implementation",
        choices=("torch", "fused", "tile_fused", "dense_fused"),
        default="fused",
    )
    parser.add_argument(
        "--tetris-implementation",
        choices=("torch", "fused", "tile_fused", "dense_fused"),
        default="fused",
    )
    parser.add_argument(
        "--native-base-tiling",
        action="store_true",
        help="Removed path; kept for CLI compat (raises).",
    )
    parser.add_argument(
        "--tetris-base-tiling",
        action="store_true",
        help="Removed path; kept for CLI compat (raises).",
    )
    parser.add_argument(
        "--mixfusion-schedule",
        action="store_true",
        help="Enable MixFusion SCHEDULE=1 + native source call (Tile/MF mid).",
    )
    parser.add_argument(
        "--patch-grid",
        type=int,
        default=None,
        help="VLLM_OMNI_MIXFUSION_PATCH_GRID pixels (256/512/1024).",
    )
    parser.add_argument(
        "--token-tile-size",
        type=int,
        default=None,
        help="VLLM_OMNI_MULTI_LORA_TILE_SIZE image tokens (256/1024/4096).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of measured steady-state repeats after one warmup.",
    )
    parser.add_argument(
        "--prompt",
        default="a still life photograph of a yellow flower in a glass vase",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of concurrent prompts in one generate() (deploy max_num_seqs must cover this).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    original_impl = os.environ.get("VLLM_OMNI_MULTI_LORA_IMPL")
    original_operator = os.environ.get("VLLM_OMNI_ENABLE_MULTI_LORA_OPERATOR")
    original_base_tiling = os.environ.get("VLLM_OMNI_BASE_TILING")
    os.environ["VLLM_OMNI_ENABLE_MULTI_LORA_OPERATOR"] = "1"
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3 for steady-state reporting")
    baseline, native_samples = _generate(
        args.model,
        args.deploy_config,
        args.adapter,
        args.output_dir / "baseline.png",
        enabled=False,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        prompt=args.prompt,
        compile_mode=args.compile_mode,
        base_device=args.base_device,
        lora_device=args.lora_device,
        all_targets=args.all_targets,
        implementation=args.native_implementation,
        base_tiling=args.native_base_tiling,
        repeats=args.repeats,
        batch_size=args.batch_size,
        mixfusion_schedule=args.mixfusion_schedule,
        patch_grid=args.patch_grid,
        token_tile_size=args.token_tile_size,
    )
    native_summary = _sample_summary(native_samples)
    baseline_ms = float(native_summary["median_ms"])
    native_median_sample = _median_sample(native_samples)
    native_breakdown = _read_breakdown(
        telemetry_dir=Path(str(native_median_sample["telemetry_dir"]))
    )
    if original_impl is None:
        os.environ.pop("VLLM_OMNI_MULTI_LORA_IMPL", None)
    else:
        os.environ["VLLM_OMNI_MULTI_LORA_IMPL"] = original_impl
    if original_operator is None:
        os.environ.pop("VLLM_OMNI_ENABLE_MULTI_LORA_OPERATOR", None)
    else:
        os.environ["VLLM_OMNI_ENABLE_MULTI_LORA_OPERATOR"] = original_operator
    if original_base_tiling is None:
        os.environ.pop("VLLM_OMNI_BASE_TILING", None)
    else:
        os.environ["VLLM_OMNI_BASE_TILING"] = original_base_tiling
    decoupled, tetris_samples = _generate(
        args.model,
        args.deploy_config,
        args.adapter,
        args.output_dir / "tetris.png",
        enabled=True,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        prompt=args.prompt,
        compile_mode=args.compile_mode,
        base_device=args.base_device,
        lora_device=args.lora_device,
        all_targets=args.all_targets,
        implementation=args.tetris_implementation,
        base_tiling=args.tetris_base_tiling,
        sample_nvlink=True,
        batch_size=args.batch_size,
        repeats=args.repeats,
        mixfusion_schedule=args.mixfusion_schedule,
        patch_grid=args.patch_grid,
        token_tile_size=args.token_tile_size,
    )
    tetris_summary = _sample_summary(tetris_samples)
    decoupled_ms = float(tetris_summary["median_ms"])
    tetris_median_sample = _median_sample(tetris_samples)
    tetris_event_path = Path(str(tetris_median_sample["stats_path"]))
    if not tetris_event_path.exists():
        raise RuntimeError("TetrisLoRA was not dispatched in the target layer")
    tetris_breakdown = _read_breakdown(
        telemetry_dir=Path(str(tetris_median_sample["telemetry_dir"]))
    )
    tetris_phases = _read_tetris_breakdown(tetris_event_path)
    critical_path = _critical_path_breakdown(
        tetris_phases,
        tetris_step_ms=decoupled_ms,
        coupled_step_ms=baseline_ms,
    )
    # Full interval lists are large; keep unions only in published JSON.
    tetris_phases_published = {
        key: value
        for key, value in tetris_phases.items()
        if key != "_intervals"
    }
    tetris_denoise = _read_denoise_timing(
        Path(str(tetris_median_sample["telemetry_dir"]))
    )
    tetris_events = [
        json.loads(line)
        for line in tetris_event_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not tetris_events:
        raise RuntimeError("TetrisLoRA stats file contains no dispatch events")
    unique_targets = {
        str(event.get("normalized_target", event.get("module_name", "")))
        .removeprefix("transformer.")
        for event in tetris_events
        if event.get("kind") == "begin_delta"
    }
    expanded_targets = set(unique_targets)
    for target in unique_targets:
        if target.endswith(".to_qkv"):
            prefix = target[: -len(".to_qkv")]
            expanded_targets.update(f"{prefix}.to_{axis}" for axis in ("q", "k", "v"))
        elif target.endswith(".add_kv_proj"):
            prefix = target[: -len(".add_kv_proj")]
            expanded_targets.update(
                f"{prefix}.add_{axis}_proj" for axis in ("q", "k", "v")
            )
    missing_targets: list[str] = []
    if args.all_targets:
        from safetensors import safe_open

        expected_targets = set()
        with safe_open(
            str(args.adapter / "adapter_model.safetensors"),
            framework="pt",
            device="cpu",
        ) as tensors:
            marker = ".lora_A.weight"
            for key in tensors.keys():
                if key.startswith("base_model.model.") and key.endswith(marker):
                    expected_targets.add(
                        key[len("base_model.model.") : -len(marker)]
                    )
        missing_targets = sorted(expected_targets - expanded_targets)
        if missing_targets:
            raise RuntimeError(
                "TetrisLoRA did not dispatch every adapter target: "
                + ", ".join(missing_targets)
            )
    delta = np.abs(baseline - decoupled)
    mse = float(np.square(baseline - decoupled).mean())
    exact_match = mse == 0.0
    psnr_db_cap = 100.0
    psnr_db = (
        psnr_db_cap
        if exact_match
        else min(
            psnr_db_cap,
            20.0 * math.log10(255.0) - 10.0 * math.log10(mse),
        )
    )
    result = {
        "status": "measured_ungated",
        "model": str(args.model),
        "adapter": str(args.adapter),
        "compile_mode": args.compile_mode,
        "native_implementation": args.native_implementation,
        "tetris_implementation": args.tetris_implementation,
        "native_base_tiling": args.native_base_tiling,
        "tetris_base_tiling": args.tetris_base_tiling,
        "mixfusion_schedule": bool(args.mixfusion_schedule),
        "patch_grid": args.patch_grid,
        "token_tile_size": args.token_tile_size,
        "batch_size": args.batch_size,
        "base_physical_gpu": args.base_device,
        "lora_physical_gpu": args.lora_device,
        "steps": args.steps,
        "resolution": [args.width, args.height],
        "seed": args.seed,
        "baseline_step_ms": baseline_ms,
        "tetris_step_ms": decoupled_ms,
        "native_steady_state": native_summary,
        "tetris_steady_state": tetris_summary,
        "warmup": {
            "count_per_path": 1,
            "excluded_from_measured": True,
            "scope": "one generate after model/session initialization",
        },
        "nvlink_counters": tetris_median_sample["nvlink"],
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "mse": mse,
        "psnr_db": psnr_db,
        "psnr_db_is_lower_bound": exact_match or psnr_db == psnr_db_cap,
        "exact_match": exact_match,
        "tetris_execution": tetris_events[0],
        "tetris_dispatch_count": sum(
            event.get("kind") == "begin_delta" for event in tetris_events
        ),
        "tetris_event_count": len(tetris_events),
        "native_breakdown": native_breakdown,
        "tetris_breakdown": tetris_breakdown,
        "tetris_transport_phases_ms": tetris_phases_published,
        "tetris_denoise_timing_ms": tetris_denoise,
        "timing_critical_path": critical_path,
        "step_wall": {
            "native": {
                "step_wall_start_ns": native_median_sample.get("step_wall_start_ns"),
                "step_wall_end_ns": native_median_sample.get("step_wall_end_ns"),
                "step_wall_ms": native_median_sample.get(
                    "step_wall_ms", baseline_ms
                ),
            },
            "tetris": {
                "step_wall_start_ns": tetris_median_sample.get("step_wall_start_ns"),
                "step_wall_end_ns": tetris_median_sample.get("step_wall_end_ns"),
                "step_wall_ms": tetris_median_sample.get(
                    "step_wall_ms", decoupled_ms
                ),
            },
            "scope": "omni.generate wall; includes all denoise steps in --steps",
        },
        "p0_telemetry_scope": {
            "pure_p2p_copy_ms": "x_peer_copy_ms + delta_p2p_copy_ms",
            "control_plane_ms": (
                "control_publish_ms + control_receive_ms + control_ack_ms"
            ),
            "control_receive_ms": "JSON decode only; blocking receive is separate",
            "control_receive_wait_ms": (
                "blocking socket receive; includes inter-dispatch wait and is not "
                "control processing"
            ),
            "host_sync_ms": "worker_host_sync_ms + base_host_sync_ms",
            "session_setup_ms": "one-time process/IPC/pool initialization",
            "steady_state_operation_ms": (
                "per-operation host/control/descriptor/merge management; not session setup"
            ),
            "base_wait_idle_ms": (
                "remote_compute_ready_wait_ms; nested wait interval, not additive"
            ),
            "overlap_efficiency": (
                "not auto-accepted until per-operation base CUDA intervals are aligned"
            ),
            "sum_vs_critical_path": (
                "timing_breakdown keeps legacy sums; timing_critical_path adds "
                "*_sum_ms and Base monotonic interval unions. See "
                "TIMING_AUDIT_CONTROL_WAIT.md"
            ),
        },
        "communication_metrics": {
            "operation_count": sum(
                1
                for event in tetris_events
                if event.get("kind") == "begin_delta"
            ),
            "tile_count": sum(
                int(event.get("token_tile_count", 0))
                for event in tetris_events
                if "token_tile_count" in event
            ),
            "phase_payload_bytes": tetris_phases.get("_payload_bytes", {}),
            "phase_counts": tetris_phases.get("_counts", {}),
        },
        "timing_breakdown": {
            "native_lora_step_ms": baseline_ms,
            "coupled_ms": baseline_ms,
            "tetris_decoupled_step_ms": decoupled_ms,
            "overhead_ms": decoupled_ms - baseline_ms,
            "bidirectional_gpu_communication_ms": (
                tetris_phases.get("x_peer_copy", 0.0)
                + tetris_phases.get("delta_p2p_copy", 0.0)
            ),
            "pure_p2p_copy_ms": (
                tetris_phases.get("x_peer_copy", 0.0)
                + tetris_phases.get("delta_p2p_copy", 0.0)
            ),
            "pure_p2p_ms": (
                tetris_phases.get("x_peer_copy", 0.0)
                + tetris_phases.get("delta_p2p_copy", 0.0)
            ),
            "remote_lora_compute_ms": tetris_phases.get("remote_compute", 0.0),
            "decoupled_lora_gemm_cuda_ms": tetris_phases.get(
                "pure_lora_gemm", 0.0
            ),
            "coupled_lora_gemm_cuda_ms": native_breakdown.get(
                "operation_totals_ms", {}
            ).get("pure_lora_gemm", 0.0),
            "wait_comm_ms": (
                tetris_phases.get("remote_compute_ready_span", 0.0)
                - tetris_phases.get("pure_lora_gemm", 0.0)
            ),
            "remote_compute_ready_span_cuda_ms": tetris_phases.get(
                "remote_compute_ready_span", 0.0
            ),
            "remote_compute_ready_wait_cuda_ms": tetris_phases.get(
                "remote_compute_ready_wait_cuda", 0.0
            ),
            "control_publish_mailbox_ms": tetris_phases.get(
                "control_publish_mailbox", 0.0
            ),
            "control_publish_ms": tetris_phases.get("control_publish", 0.0),
            "control_receive_ms": tetris_phases.get("control_receive", 0.0),
            "control_receive_wait_ms": tetris_phases.get(
                "control_receive_wait", 0.0
            ),
            "control_ack_ms": tetris_phases.get("control_ack", 0.0),
            "control_plane_ms": (
                tetris_phases.get("control_publish", 0.0)
                + tetris_phases.get("control_receive", 0.0)
                + tetris_phases.get("control_ack", 0.0)
            ),
            "control_ms": (
                tetris_phases.get("control_publish", 0.0)
                + tetris_phases.get("control_receive", 0.0)
                + tetris_phases.get("control_ack", 0.0)
            ),
            "other_ms": (
                (decoupled_ms - baseline_ms)
                - (
                    tetris_phases.get("x_peer_copy", 0.0)
                    + tetris_phases.get("delta_p2p_copy", 0.0)
                )
                - (
                    tetris_phases.get("control_publish", 0.0)
                    + tetris_phases.get("control_receive", 0.0)
                    + tetris_phases.get("control_ack", 0.0)
                )
            ),
            "steady_state_operation_management_ms": (
                tetris_phases.get("control_publish", 0.0)
                + tetris_phases.get("control_receive", 0.0)
                + tetris_phases.get("control_ack", 0.0)
                + tetris_phases.get("x_descriptor_receive", 0.0)
                + tetris_phases.get("delta_descriptor_receive", 0.0)
                + tetris_phases.get("merge", 0.0)
            ),
            "worker_host_sync_ms": tetris_phases.get("worker_host_sync", 0.0),
            "base_host_sync_ms": tetris_phases.get("base_host_sync", 0.0),
            "host_sync_ms": (
                tetris_phases.get("worker_host_sync", 0.0)
                + tetris_phases.get("base_host_sync", 0.0)
            ),
            "ipc_sync_preparation_ms": (
                tetris_phases.get("x_ipc_publish", 0.0)
                + tetris_phases.get("delta_ipc_receive", 0.0)
                + tetris_phases.get("x_slot_copy", 0.0)
            ),
            "session_setup_ms": tetris_phases.get("session_setup", 0.0),
            "remote_compute_ready_wait_ms": tetris_phases.get(
                "remote_compute_ready_wait", 0.0
            ),
            "base_wait_idle_ms": tetris_phases.get(
                "remote_compute_ready_wait", 0.0
            ),
            "delta_consumer_wait_ms": tetris_phases.get(
                "delta_consumer_wait", 0.0
            ),
            "merge_ms": tetris_phases.get("merge", 0.0),
            "operation_host_total_ms": tetris_phases.get(
                "operation_host_total", 0.0
            ),
            "inter_dispatch_gap_ms": max(
                0.0,
                tetris_denoise.get("denoise_elapsed_ms", decoupled_ms)
                - tetris_phases.get("operation_host_total", 0.0),
            ),
            "overhead_to_comm_ratio": (
                (decoupled_ms - baseline_ms)
                / max(
                    1e-9,
                    tetris_phases.get("x_peer_copy", 0.0)
                    + tetris_phases.get("delta_p2p_copy", 0.0),
                )
            ),
            "overhead_to_native_ratio": (
                (decoupled_ms - baseline_ms) / max(1e-9, baseline_ms)
            ),
            "overlap_efficiency": None,
            # Corrected sum vs critical-path fields (legacy keys above retained).
            "metric_scope": {
                "legacy_phase_ms": "cumulative_per_operation",
                "critical_path": "see timing_critical_path",
            },
            "step_wall_start_ns": tetris_median_sample.get("step_wall_start_ns"),
            "step_wall_end_ns": tetris_median_sample.get("step_wall_end_ns"),
            "step_wall_ms": tetris_median_sample.get("step_wall_ms", decoupled_ms),
            "control_ack_sum_ms": critical_path["control_ack_sum_ms"],
            "remote_compute_ready_wait_sum_ms": critical_path[
                "remote_compute_ready_wait_sum_ms"
            ],
            "remote_compute_ready_wait_cuda_ms": critical_path[
                "remote_compute_ready_wait_cuda_ms"
            ],
            "remote_compute_ready_span_cuda_ms": critical_path[
                "remote_compute_ready_span_cuda_ms"
            ],
            "control_receive_wait_sum_ms": critical_path[
                "control_receive_wait_sum_ms"
            ],
            "remote_compute_sum_ms": critical_path["remote_compute_sum_ms"],
            "decoupled_lora_gemm_cuda_ms": critical_path[
                "decoupled_lora_gemm_cuda_ms"
            ],
            "wait_comm_ms": critical_path["wait_comm_ms"],
            "x_peer_copy_sum_ms": critical_path["x_peer_copy_sum_ms"],
            "delta_p2p_copy_sum_ms": critical_path["delta_p2p_copy_sum_ms"],
            "base_host_blocking_ms": critical_path["base_host_blocking_ms"],
            "worker_host_blocking_ms": critical_path["worker_host_blocking_ms"],
            "gpu_dependency_wait_ms": critical_path["gpu_dependency_wait_ms"],
            "remote_ready_critical_path_ms": critical_path[
                "remote_ready_critical_path_ms"
            ],
            "remote_compute_ready_wait_critical_path_ms": critical_path[
                "remote_compute_ready_wait_critical_path_ms"
            ],
            "control_receive_critical_path_ms": critical_path[
                "control_receive_critical_path_ms"
            ],
            "critical_path_p2p_ms": critical_path["critical_path_p2p_ms"],
            "critical_path_x_peer_copy_ms": critical_path[
                "critical_path_x_peer_copy_ms"
            ],
            "critical_path_delta_p2p_copy_ms": critical_path[
                "critical_path_delta_p2p_copy_ms"
            ],
            "critical_path_comm_ms": critical_path["critical_path_comm_ms"],
            "critical_path_remote_compute_ms": critical_path[
                "critical_path_remote_compute_ms"
            ],
            "critical_path_control_ms": critical_path["critical_path_control_ms"],
            "critical_path_wait_ms": critical_path["critical_path_wait_ms"],
            "critical_path_merge_ms": critical_path["critical_path_merge_ms"],
            "critical_path_other_ms": critical_path["critical_path_other_ms"],
            "critical_path_noncomm_ms": critical_path["critical_path_noncomm_ms"],
            "tetris_step_ms": critical_path["tetris_step_ms"],
            "coupled_step_ms": critical_path["coupled_step_ms"],
            "unexplained_ms": critical_path["unexplained_ms"],
            "unexplained_ms_note": critical_path["unexplained_ms_note"],
        },
        "tetris_dispatch_unique_targets": sorted(expanded_targets),
        "tetris_missing_targets": missing_targets,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
