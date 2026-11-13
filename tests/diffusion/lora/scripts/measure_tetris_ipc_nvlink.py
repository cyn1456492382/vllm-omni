#!/usr/bin/env python
"""Measure CUDA IPC payloads and attributable NVLink counter deltas."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import statistics
import subprocess
import time
import uuid
from multiprocessing.connection import Connection

import torch


def nvidia_smi(*args: str) -> str:
    return subprocess.check_output(["nvidia-smi", *args], text=True)


def counters(device: int) -> dict[int, tuple[int, int]]:
    values: dict[int, list[int]] = {}
    for line in nvidia_smi("nvlink", "-i", str(device), "-gt", "d").splitlines():
        if "Data Tx:" not in line and "Data Rx:" not in line:
            continue
        link = int(line.split("Link", 1)[1].split(":", 1)[0].strip())
        value = int(line.rsplit(":", 1)[1].strip().split()[0])
        pair = values.setdefault(link, [0, 0])
        pair[0 if "Data Tx:" in line else 1] = value
    return {link: (pair[0], pair[1]) for link, pair in values.items()}


def counter_delta(
    before: dict[int, tuple[int, int]], after: dict[int, tuple[int, int]]
) -> dict[int, tuple[int, int]]:
    return {
        link: (after[link][0] - before[link][0], after[link][1] - before[link][1])
        for link in before.keys() & after.keys()
    }


def _receiver(sock: socket.socket, device: int) -> None:
    from vllm_omni.diffusion.lora.tetris_lora_transport import (
        CudaIpcTransfer,
        TensorDescriptor,
        _IpcTensorPayload,
        _validate_payload,
        import_cuda_tensor,
    )

    torch.cuda.set_device(device)
    connection = Connection(sock.detach(), readable=True, writable=True)
    transfer = CudaIpcTransfer(connection, role="consumer", device=device)
    tensor_cache = {}
    output_cache = {}
    try:
        while True:
            raw = json.loads(connection.recv_bytes().decode("utf-8"))
            if raw.get("kind") == "close":
                break
            if raw.get("kind") not in {"tensor", "tensor_ref"}:
                raise RuntimeError(f"unexpected probe message: {raw}")
            descriptor = TensorDescriptor.from_json(raw["descriptor"])
            cache_key = str(raw.get("cache_key") or descriptor.buffer_token)
            cached = tensor_cache.get(cache_key)
            if raw.get("kind") == "tensor_ref":
                if cached is None:
                    raise RuntimeError(f"unknown persistent IPC slot {cache_key}")
                received, event = cached
            else:
                payload = _IpcTensorPayload.from_json(raw["payload"])
                payload = _IpcTensorPayload(
                    **{**payload.__dict__, "target_device": device}
                )
                received, event = import_cuda_tensor(payload)
                tensor_cache[cache_key] = (received, event)
            _validate_payload(descriptor, received)
            if event is not None:
                torch.cuda.current_stream(device).wait_event(event)
            output = output_cache.get(cache_key)
            if output is None:
                output = torch.empty_like(received, device=device)
                output_cache[cache_key] = output
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device)
            start.record(stream)
            output.copy_(received, non_blocking=True)
            end.record(stream)
            end.synchronize()
            checksum = None
            if raw.get("kind") == "tensor":
                checksum = float(output.float().sum().item())
            transfer.acknowledge(descriptor, consumer_complete=True)
            connection.send_bytes(
                json.dumps(
                    {
                        "kind": "checksum",
                        "checksum": checksum,
                        "gpu_copy_ms": float(start.elapsed_time(end)),
                    }
                ).encode()
            )
    finally:
        transfer.close()
        connection.close()


class IpcProbe:
    def __init__(self, source: int, target: int) -> None:
        from vllm_omni.diffusion.lora.tetris_lora_transport import (
            CudaIpcTransfer,
            TensorDescriptor,
        )
        self.source = source
        self.target = target
        parent_socket, child_socket = socket.socketpair()
        self.context = mp.get_context("spawn")
        self.process = self.context.Process(target=_receiver, args=(child_socket, target))
        self.process.start()
        child_socket.close()
        self.connection = Connection(parent_socket.detach(), readable=True, writable=True)
        self.send = CudaIpcTransfer(self.connection, role="producer", device=source)
        self.tensor_descriptor = TensorDescriptor
        self.sequence = 0

    def transfer(self, payload: torch.Tensor) -> dict[str, float | None]:
        self.sequence += 1
        bytes_count = payload.numel() * payload.element_size()
        descriptor = self.tensor_descriptor(
            session_epoch="nvlink-counter-validation",
            operation_id=uuid.uuid4().hex,
            request_id="nvlink-counter-validation",
            batch_id="nvlink-counter-validation",
            denoise_step=0,
            layer_name="ipc.payload_probe",
            invocation_id=self.sequence,
            slot_id=0,
            slot_generation=self.sequence,
            shape=tuple(payload.shape),
            stride=tuple(payload.stride()),
            dtype=str(payload.dtype),
            layout="contiguous",
            payload_bytes=bytes_count,
            buffer_token=f"persistent-probe:{bytes_count}",
            adapter_id="counter-probe",
            adapter_version="1",
            rank=1,
            scale=1.0,
            branch_index=0,
            output_offset=0,
            completion_sequence=self.sequence,
            buffer_capacity=bytes_count,
        )
        started = time.perf_counter()
        self.send.send(
            descriptor,
            payload,
            cache_key=descriptor.buffer_token,
        )
        self.send.wait_ack(descriptor, timeout_s=60.0)
        response = json.loads(self.connection.recv_bytes().decode("utf-8"))
        torch.cuda.synchronize(self.source)
        elapsed = time.perf_counter() - started
        if response.get("checksum") is not None:
            expected_checksum = payload.float().sum().item()
            actual_checksum = float(response["checksum"])
            if not torch.isfinite(torch.tensor(actual_checksum)):
                raise RuntimeError("receiver produced a non-finite checksum")
            if actual_checksum != expected_checksum:
                raise RuntimeError(
                    f"IPC payload checksum mismatch: {actual_checksum} != "
                    f"{expected_checksum}"
                )
        return {
            "host_elapsed_ms": elapsed * 1000.0,
            "gpu_copy_ms": float(response["gpu_copy_ms"]),
            "checksum": response.get("checksum"),
        }

    def close(self) -> None:
        try:
            self.connection.send_bytes(json.dumps({"kind": "close"}).encode())
        except (BrokenPipeError, OSError):
            pass
        self.connection.close()
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.exitcode != 0:
            raise RuntimeError(f"IPC receiver exited with {self.process.exitcode}")


def physical_peer_map(physical_source: int, physical_target: int) -> dict[str, object]:
    return {
        "source": physical_source,
        "target": physical_target,
        "topology": nvidia_smi("topo", "-m"),
        "nvlink_remote_map": nvidia_smi("nvlink", "-R"),
    }


def measure_direction(
    source: int,
    target: int,
    physical_source: int,
    physical_target: int,
    sizes: list[int],
    warmup: int,
    iterations: int,
) -> list[dict[str, object]]:
    if not torch.cuda.can_device_access_peer(source, target):
        raise RuntimeError(f"CUDA P2P is unavailable for visible {source}->{target}")
    measurements = []
    for mib in sizes:
        payload = torch.arange(
            mib * 1024 * 1024 // 2,
            dtype=torch.bfloat16,
            device=f"cuda:{source}",
        )
        probe = IpcProbe(source, target)
        try:
            for _ in range(warmup):
                probe.transfer(payload)
            baseline_before = counters(physical_source)
            baseline_target_before = counters(physical_target)
            time.sleep(1.0)
            baseline_after = counters(physical_source)
            baseline_target_after = counters(physical_target)
            samples = []
            for _ in range(iterations):
                source_before = counters(physical_source)
                target_before = counters(physical_target)
                timing = probe.transfer(payload)
                source_after = counters(physical_source)
                target_after = counters(physical_target)
                samples.append(
                    {
                        "elapsed_ms": timing["gpu_copy_ms"],
                        "host_elapsed_ms": timing["host_elapsed_ms"],
                        "source_counter_delta": counter_delta(source_before, source_after),
                        "target_counter_delta": counter_delta(target_before, target_after),
                    }
                )
            expected_per_link_kib = payload.numel() * payload.element_size() // 1024 // 8
            for sample in samples:
                source_values = list(sample["source_counter_delta"].values())
                target_values = list(sample["target_counter_delta"].values())
                sample["counter_attribution"] = {
                    "expected_kib_per_link": expected_per_link_kib,
                    "source_tx_matches": all(
                        tx == expected_per_link_kib and rx == 0
                        for tx, rx in source_values
                    ),
                    "target_rx_matches": all(
                        tx == 0 and rx == expected_per_link_kib
                        for tx, rx in target_values
                    ),
                }
            median_s = statistics.median(item["elapsed_ms"] for item in samples) / 1000
            measurements.append(
                {
                    "payload_mib": mib,
                    "payload_bytes": payload.numel() * payload.element_size(),
                    "baseline_counter_delta_1s": {
                        "source": counter_delta(baseline_before, baseline_after),
                        "target": counter_delta(baseline_target_before, baseline_target_after),
                    },
                    "runs": samples,
                    "median_ms": median_s * 1000,
                    "effective_gib_s": payload.numel() * payload.element_size() / 2**30 / median_s,
                }
            )
        finally:
            probe.close()
            del payload
            torch.cuda.empty_cache()
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--physical-gpus", default="5,6")
    parser.add_argument("--sizes-mib", default="1,16,64,256")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    physical = [int(value) for value in args.physical_gpus.split(",")]
    if len(physical) != 2 or torch.cuda.device_count() < 2:
        raise SystemExit("need two visible GPUs and two physical GPU IDs")
    report = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_ids": physical,
        "gpu_inventory": nvidia_smi("--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total", "--format=csv,noheader"),
        "topology": nvidia_smi("topo", "-m"),
        "nvlink_remote_map": nvidia_smi("nvlink", "-R"),
        "directions": [],
        "counter_claim": "NVLink payload attribution requires repeatable directional deltas above baseline; NV8 alone is insufficient",
    }
    sizes = [int(value) for value in args.sizes_mib.split(",")]
    for source, target, physical_source, physical_target in (
        (args.source, args.target, physical[0], physical[1]),
        (args.target, args.source, physical[1], physical[0]),
    ):
        report["directions"].append(
            {
                "visible_source": source,
                "visible_target": target,
                "physical_source": physical_source,
                "physical_target": physical_target,
                "peer_evidence": physical_peer_map(physical_source, physical_target),
                "measurements": measure_direction(
                    source, target, physical_source, physical_target, sizes, args.warmup, args.iterations
                ),
            }
        )
    with open(args.output, "x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
