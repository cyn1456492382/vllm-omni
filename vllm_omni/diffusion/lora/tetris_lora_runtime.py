"""Process-separated TetrisLoRA residual execution for FLUX."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import torch

from .tetris_lora_transport import (
    ControlInbox,
    ControlMailbox,
    CudaIpcTransfer,
    IpcSignalRing,
    OperationIdAllocator,
    SessionPlanRegistry,
    TensorDescriptor,
    TransportError,
    TransportUnavailableError,
    binary_wire_enabled,
    compact_control_enabled,
    decode_control_bytes,
    mailbox_control_enabled,
    operation_batch_ack_enabled,
    wire_json_dumps,
    wire_json_loads,
)
from .ops import (
    BACKEND_DENSE_FUSED,
    BACKEND_REFERENCE,
    BACKEND_TWO_STAGE_TC,
    LoRABackendError,
    get_dense_fused_ops,
    require_backend,
    resolve_backend_name,
)
from .tetris_lora_twostage import execute_two_stage_lora


def _signal_ring_capacity() -> int:
    return max(8, int(os.environ.get("TETRIS_LORA_SIGNAL_RING_SIZE", "64")))


def _share_control_registry(
    endpoints: tuple[CudaIpcTransfer, ...],
) -> SessionPlanRegistry:
    """Point every local endpoint at one plan registry."""
    if not endpoints:
        raise TransportError("control registry requires an endpoint")
    registry = endpoints[0].registry
    for endpoint in endpoints[1:]:
        endpoint.attach_registry(registry)
    return registry


def _attach_compact_control(
    connection: Any,
    *,
    device: int,
    endpoints: tuple[CudaIpcTransfer, ...],
    initiate: bool,
) -> IpcSignalRing | None:
    """Session-scoped Signal-ring (+ optional mailbox) handshake."""
    registry = _share_control_registry(endpoints)
    if not compact_control_enabled():
        return None
    inbox = ControlInbox(connection, registry)
    ring = IpcSignalRing(device, capacity=_signal_ring_capacity())
    mailbox: ControlMailbox | None = None
    if mailbox_control_enabled():
        if initiate:
            mailbox = ControlMailbox(ring.capacity, create=True)
        # non-initiate imports peer mailbox after offer
    for endpoint in endpoints:
        endpoint.attach_inbox(inbox)
        endpoint.attach_signal_ring(ring)
    if initiate:
        offer_msg: dict[str, Any] = {
            "kind": "event_ring_offer",
            "capacity": ring.capacity,
            "handles": ring.export_handles(),
            "generation_shm_name": ring.export_generation_shm_name(),
        }
        if mailbox is not None:
            offer_msg["mailbox_shm_name"] = mailbox.export_shm_name()
        _send_json(connection, offer_msg)
        reply = _recv_json(connection)
        if reply.get("kind") != "event_ring_reply":
            raise TransportError(f"expected event_ring_reply, got {reply.get('kind')}")
        ring.import_peer_handles([str(handle) for handle in reply["handles"]])
        generation_shm_name = reply.get("generation_shm_name")
        if not generation_shm_name:
            raise TransportError(
                "event_ring_reply missing generation_shm_name for ACK fencing"
            )
        ring.import_peer_generations(str(generation_shm_name))
        if mailbox is not None:
            for endpoint in endpoints:
                endpoint.attach_mailbox(mailbox)
        return ring
    offer = _recv_json(connection)
    if offer.get("kind") != "event_ring_offer":
        raise TransportError(f"expected event_ring_offer, got {offer.get('kind')}")
    if int(offer.get("capacity", -1)) != ring.capacity:
        raise TransportError("signal ring capacity mismatch with peer")
    ring.import_peer_handles([str(handle) for handle in offer["handles"]])
    generation_shm_name = offer.get("generation_shm_name")
    if not generation_shm_name:
        raise TransportError(
            "event_ring_offer missing generation_shm_name for ACK fencing"
        )
    ring.import_peer_generations(str(generation_shm_name))
    if mailbox_control_enabled():
        mailbox_shm_name = offer.get("mailbox_shm_name")
        if not mailbox_shm_name:
            raise TransportError(
                "event_ring_offer missing mailbox_shm_name for steady control"
            )
        mailbox = ControlMailbox(
            ring.capacity, create=False, shm_name=str(mailbox_shm_name)
        )
        for endpoint in endpoints:
            endpoint.attach_mailbox(mailbox)
    reply_msg: dict[str, Any] = {
        "kind": "event_ring_reply",
        "capacity": ring.capacity,
        "handles": ring.export_handles(),
        "generation_shm_name": ring.export_generation_shm_name(),
    }
    _send_json(connection, reply_msg)
    return ring


def _enabled() -> bool:
    return os.environ.get("VLLM_OMNI_TETRIS_LORA", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _target_module() -> str:
    return os.environ.get(
        "TETRIS_LORA_TARGET_MODULE", "transformer_blocks.0.ff.net.0.proj"
    )


def _normalize_target_module(module_name: str) -> str:
    value = str(module_name)
    if value.startswith("transformer."):
        return value[len("transformer.") :]
    return value


def _resolve_target_modules(module_name: str) -> tuple[str, ...]:
    target = _normalize_target_module(module_name)
    if target.endswith(".to_qkv"):
        prefix = target[: -len(".to_qkv")]
        return tuple(f"{prefix}.to_{axis}" for axis in ("q", "k", "v"))
    if target.endswith(".add_kv_proj"):
        prefix = target[: -len(".add_kv_proj")]
        return tuple(f"{prefix}.add_{axis}_proj" for axis in ("q", "k", "v"))
    return (target,)


def _qkv_slice_index(packed_target: str, target: str) -> int | None:
    """Return the packed Q/K/V slice selected by an independent target."""
    packed_target = _normalize_target_module(packed_target)
    target = _normalize_target_module(target)
    if target == packed_target:
        return None
    if packed_target.endswith(".to_qkv"):
        prefix = packed_target[: -len("to_qkv")]
        target_prefix = "to_"
    elif packed_target.endswith(".add_kv_proj"):
        prefix = packed_target[: -len("add_kv_proj")]
        target_prefix = "add_"
    else:
        return None
    for index, axis in enumerate(("q", "k", "v")):
        if target == prefix + f"{target_prefix}{axis}" + (
            "_proj" if target_prefix == "add_" else ""
        ):
            return index
    return None


def _resolve_target_layout(
    packed_target: str,
    active_slots: tuple[int, ...],
    adapter_weights: dict[tuple[int, str], object],
) -> tuple[str, ...]:
    """Resolve one packed target without silently mixing fused and split LoRA."""
    packed_target = _normalize_target_module(packed_target)
    if not packed_target.endswith((".to_qkv", ".add_kv_proj")):
        return (packed_target,)
    split_targets = _resolve_target_modules(packed_target)
    fused_slots = {
        slot for slot in active_slots if (slot, packed_target) in adapter_weights
    }
    split_slots = {
        slot
        for slot in active_slots
        if any((slot, target) in adapter_weights for target in split_targets)
    }
    if fused_slots and split_slots:
        raise TransportError(
            f"mixed fused and split LoRA layouts are unsupported for {packed_target}"
        )
    if fused_slots:
        return (packed_target,)
    return tuple(
        target
        for target in split_targets
        if any((slot, target) in adapter_weights for slot in active_slots)
    )


def _is_target_module(module_name: str) -> bool:
    normalized = _normalize_target_module(module_name)
    if os.environ.get("TETRIS_LORA_ALL_TARGETS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return normalized.startswith(
            ("transformer_blocks.", "single_transformer_blocks.")
        )
    target = _normalize_target_module(_target_module())
    return normalized == target or normalized.endswith("." + target)


def _send_json(connection: Any, message: dict[str, Any]) -> None:
    connection.send_bytes(wire_json_dumps(message, sort_keys=True).encode("utf-8"))


def _recv_json(connection: Any) -> dict[str, Any]:
    message = wire_json_loads(connection.recv_bytes())
    if not isinstance(message, dict):
        raise TransportError("control message must be an object")
    return message


def _recv_json_timed(
    connection: Any,
    registry: SessionPlanRegistry | None = None,
) -> tuple[dict[str, Any], float, float]:
    """Receive one control message and split wait from decode time.

    Binary plan frames are consumed until a tensor, ACK, or JSON control
    message is available. Handshake callers pass no registry.
    """
    plans = registry if registry is not None else SessionPlanRegistry()
    receive_started_ns = time.monotonic_ns()
    payload = connection.recv_bytes()
    receive_finished_ns = time.monotonic_ns()
    message = decode_control_bytes(payload, plans)
    while message is None:
        payload = connection.recv_bytes()
        message = decode_control_bytes(payload, plans)
    decode_finished_ns = time.monotonic_ns()
    return (
        message,
        (receive_finished_ns - receive_started_ns) / 1000.0,
        (decode_finished_ns - receive_finished_ns) / 1000.0,
    )


_ADAPTER_VERSION_CACHE: dict[str, str] = {}
# Lock shared with _CLIENTS (defined later); created early for invalidate().
_CLIENTS_LOCK = threading.Lock()
_SHA256_STATS: dict[str, float] = {
    "calls": 0.0,
    "bytes": 0.0,
    "ms": 0.0,
}
_STATS_BUFFER: list[str] = []
_STATS_BUFFER_LOCK = threading.Lock()
_STATS_BUFFER_FLUSH_LINES = 64
_WORKER_TIMING_POOL_SIZE = 64


def _stats_buffer_enabled() -> bool:
    value = os.environ.get("TETRIS_LORA_STATS_BUFFER", "1").lower()
    return value not in {"0", "false", "no", "off"}


def _sha256_stats_enabled() -> bool:
    return os.environ.get("TETRIS_LORA_SHA256_STATS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def flush_runtime_event_buffer(path: str | None = None) -> int:
    """Flush buffered `_record_runtime_event` lines to disk."""
    target = path or os.environ.get("TETRIS_LORA_STATS_PATH")
    with _STATS_BUFFER_LOCK:
        if not _STATS_BUFFER:
            return 0
        if not target:
            _STATS_BUFFER.clear()
            return 0
        lines = list(_STATS_BUFFER)
        _STATS_BUFFER.clear()
    with Path(target).open("a", encoding="utf-8") as output:
        output.writelines(lines)
    return len(lines)


def _record_runtime_event(kind: str, **payload: Any) -> None:
    path = os.environ.get("TETRIS_LORA_STATS_PATH")
    if not path:
        return
    record = {"kind": kind, **payload}
    line = json.dumps(record, sort_keys=True) + "\n"
    if not _stats_buffer_enabled():
        with Path(path).open("a", encoding="utf-8") as output:
            output.write(line)
        return
    with _STATS_BUFFER_LOCK:
        _STATS_BUFFER.append(line)
        if len(_STATS_BUFFER) >= _STATS_BUFFER_FLUSH_LINES:
            lines = list(_STATS_BUFFER)
            _STATS_BUFFER.clear()
        else:
            lines = []
    if lines:
        with Path(path).open("a", encoding="utf-8") as output:
            output.writelines(lines)


def _sha256_file(path: Path) -> str:
    started = time.perf_counter() if _sha256_stats_enabled() else None
    digest = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            total_bytes += len(block)
            digest.update(block)
    if started is not None:
        _SHA256_STATS["calls"] += 1.0
        _SHA256_STATS["bytes"] += float(total_bytes)
        _SHA256_STATS["ms"] += (time.perf_counter() - started) * 1000.0
        _record_runtime_event(
            "sha256_stats",
            calls=int(_SHA256_STATS["calls"]),
            bytes=int(_SHA256_STATS["bytes"]),
            ms=float(_SHA256_STATS["ms"]),
            path=str(path),
        )
    return digest.hexdigest()


def register_adapter_version(path: str | Path) -> str:
    """Hash ``adapter_model.safetensors`` once per resolved adapter directory.

    Hot-path callers (``begin_tetris_lora``) must use the cache only; do not
    version adapters via per-target ``os.stat``.
    """
    resolved = str(Path(path).resolve())
    cached = _ADAPTER_VERSION_CACHE.get(resolved)
    if cached is not None:
        return cached
    tensor_path = Path(resolved) / "adapter_model.safetensors"
    version = _sha256_file(tensor_path)
    _ADAPTER_VERSION_CACHE[resolved] = version
    return version


def invalidate_adapter_version(path: str | Path) -> None:
    """Drop the cached adapter content hash and bump generation.

    After weights at ``path`` change, callers must invalidate before the next
    ``begin_tetris_lora`` so workers keyed by the old version are not reused.
    """
    resolved = str(Path(path).resolve())
    _ADAPTER_VERSION_CACHE.pop(resolved, None)
    with _CLIENTS_LOCK:
        stale = [
            key
            for key, client in _CLIENTS.items()
            if any(str(Path(p).resolve()) == resolved for _, p in key[0])
        ]
        for key in stale:
            client = _CLIENTS.pop(key, None)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass


class _PendingTimingQueue:
    """Bounded CUDA Event timing queue (Base ``_drain_delta_copy_timings`` style).

    Only calls ``elapsed_time`` after ``end_event.query()`` is True. If the pool
    is full, telemetry samples are dropped (zeros / skip) without failing compute.
    Never synchronizes the device.
    """

    def __init__(self, pool_size: int = _WORKER_TIMING_POOL_SIZE) -> None:
        self.pool_size = max(1, int(pool_size))
        self._pool: list[tuple[Any, Any]] = []
        self._in_use: list[bool] = []
        self._cursor = 0
        self._pending: list[dict[str, Any]] = []
        self._cuda_ready = False

    def _ensure_pool(self) -> None:
        if self._cuda_ready:
            return
        self._pool = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(self.pool_size)
        ]
        self._in_use = [False] * self.pool_size
        self._cuda_ready = True

    def acquire(self) -> tuple[Any, Any, int] | None:
        self._ensure_pool()
        for offset in range(self.pool_size):
            index = (self._cursor + offset) % self.pool_size
            if not self._in_use[index]:
                self._in_use[index] = True
                self._cursor = (index + 1) % self.pool_size
                start_event, end_event = self._pool[index]
                return start_event, end_event, index
        return None

    def submit(
        self,
        *,
        start_event: Any,
        end_event: Any,
        event_index: int,
        phase: str,
        **meta: Any,
    ) -> None:
        self._pending.append(
            {
                "start_event": start_event,
                "end_event": end_event,
                "event_index": event_index,
                "phase": phase,
                **meta,
            }
        )

    def drain(self) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for item in self._pending:
            end_event = item["end_event"]
            if not end_event.query():
                remaining.append(item)
                continue
            duration_us = float(item["start_event"].elapsed_time(end_event)) * 1000.0
            event_index = int(item["event_index"])
            if 0 <= event_index < len(self._in_use):
                self._in_use[event_index] = False
            completed.append({**item, "duration_us": duration_us})
        self._pending = remaining
        return completed


def _localize_x_mode() -> str:
    mode = os.environ.get("TETRIS_LORA_LOCALIZE_X", "auto").lower()
    if mode not in {"auto", "1", "0"}:
        raise TransportError(
            "TETRIS_LORA_LOCALIZE_X must be one of auto, 1, or 0"
        )
    return mode


def _logical_device_index(physical_device: int) -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return physical_device
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    try:
        physical_text = str(int(physical_device))
    except (TypeError, ValueError):
        return None
    for logical_index, entry in enumerate(entries):
        if entry == physical_text:
            return logical_index
    return None


def _physical_device_for_logical(logical_device: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
        if 0 <= logical_device < len(entries) and entries[logical_device].isdigit():
            return int(entries[logical_device])
    return logical_device


def _physical_peer_access(base_device: int, peer_device: int) -> bool:
    logical_peer = _logical_device_index(peer_device)
    if logical_peer is not None and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if (
            0 <= base_device < device_count
            and 0 <= logical_peer < device_count
            and base_device != logical_peer
        ):
            return bool(torch.cuda.can_device_access_peer(base_device, logical_peer))

    try:
        output = subprocess.run(
            ["nvidia-smi", "topo", "-p2p", "r"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    header_index = None
    headers: list[int] = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        matches = [int(value) for value in re.findall(r"GPU(\d+)", line)]
        if len(matches) >= 2:
            header_index = index
            headers = matches
            break
    if header_index is None or peer_device not in headers:
        return False
    row_prefix = f"GPU{_physical_device_for_logical(base_device)}"
    row = next(
        (
            line
            for line in lines[header_index + 1 :]
            if line.strip().startswith(row_prefix)
        ),
        None,
    )
    if row is None:
        return False
    values = row.split()[1:]
    try:
        value = values[headers.index(peer_device)]
    except (IndexError, ValueError):
        return False
    return value == "OK"


class _WorkerXWorkspace:
    def __init__(self) -> None:
        self._buffers: dict[tuple[Any, ...], torch.Tensor] = {}
        self._events: dict[
            tuple[Any, ...],
            tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event],
        ] = {}

    def get(self, source: torch.Tensor) -> torch.Tensor:
        key = (str(source.dtype), tuple(source.shape), tuple(source.stride()))
        buffer = self._buffers.get(key)
        if buffer is None or buffer.numel() < source.numel():
            buffer = torch.empty(
                source.shape, dtype=source.dtype, device=source.device
            )
            self._buffers[key] = buffer
        return buffer

    def events_for(
        self, source: torch.Tensor
    ) -> tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]:
        key = (str(source.dtype), tuple(source.shape), tuple(source.stride()))
        events = self._events.get(key)
        if events is None:
            events = (
                torch.cuda.Event(enable_timing=False),
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=False, interprocess=True),
            )
            self._events[key] = events
        return events


def _copy_x_to_local_workspace(
    remote_x: torch.Tensor,
    workspace: _WorkerXWorkspace,
    copy_stream: torch.cuda.Stream,
    *,
    producer_ready: torch.cuda.Event | None = None,
    timing_start: Any | None = None,
    timing_end: Any | None = None,
    ready_span_start: Any | None = None,
) -> tuple[
    torch.Tensor,
    torch.cuda.Event,
    torch.cuda.Event,
    torch.cuda.Event,
]:
    """Copy remote X into a local workspace with explicit producer readiness.

    ``producer_ready`` is the Base-side write-complete Event returned by
    ``CudaIpcTransfer.receive*``. A locally recorded workspace event cannot
    substitute for that producer fence.
    """
    compute_stream = torch.cuda.current_stream(remote_x.device)
    _local_ready, copy_start, copy_end, copy_release = workspace.events_for(remote_x)
    if producer_ready is not None:
        copy_stream.wait_event(producer_ready)
        compute_stream.wait_event(producer_ready)
    local_x = workspace.get(remote_x)
    copy_start.record(copy_stream)
    if timing_start is not None:
        timing_start.record(copy_stream)
    if ready_span_start is not None:
        ready_span_start.record(copy_stream)
    with torch.cuda.stream(copy_stream):
        local_x.copy_(remote_x, non_blocking=True)
    copy_end.record(copy_stream)
    if timing_end is not None:
        timing_end.record(copy_stream)
    copy_release.record(copy_stream)
    compute_stream.wait_event(copy_end)
    return local_x, copy_start, copy_end, copy_release


def _worker_main(
    connection: Any,
    physical_device: int,
    adapter_paths_json: str,
    target_module: str,
    adapter_id: str,
    all_targets: bool,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
    from safetensors import safe_open

    torch.cuda.set_device(0)
    if os.environ.get("TETRIS_LORA_REFERENCE", "").lower() not in {
        "1", "true", "yes", "on"
    } and not torch.cuda.is_available():
        raise RuntimeError("TetrisLoRA two_stage_tc performance mode requires CUDA")
    adapter_paths = {
        int(slot): str(path)
        for slot, path in json.loads(adapter_paths_json).items()
    }
    if not adapter_paths:
        raise RuntimeError("TetrisLoRA worker received no adapter paths")
    adapter_weights: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, int]] = {}
    adapter_versions: dict[int, str] = {}
    prefix = "base_model.model."
    available_targets: set[str] = set()
    for slot, adapter_path in sorted(adapter_paths.items()):
        config = json.loads(
            (Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8")
        )
        alpha = float(config["lora_alpha"])
        tensor_path = Path(adapter_path) / "adapter_model.safetensors"
        adapter_versions[slot] = register_adapter_version(adapter_path)
        with safe_open(str(tensor_path), framework="pt", device="cpu") as tensors:
            if all_targets:
                requested_targets = {
                    key[len(prefix) : -len(".lora_A.weight")]
                    for key in tensors.keys()
                    if key.startswith(prefix) and key.endswith(".lora_A.weight")
                }
            else:
                requested_targets = set(_resolve_target_modules(target_module))
                requested_targets.update(
                    key[len(prefix) : -len(".lora_A.weight")]
                    for key in tensors.keys()
                    if key.startswith(prefix)
                    and key.endswith(".lora_A.weight")
                    and key[len(prefix) : -len(".lora_A.weight")] == target_module
                )
            for target in sorted(requested_targets):
                a_key = f"{prefix}{target}.lora_A.weight"
                b_key = f"{prefix}{target}.lora_B.weight"
                if a_key not in tensors.keys() or b_key not in tensors.keys():
                    continue
                a = tensors.get_tensor(a_key).to("cuda:0").contiguous()
                b = tensors.get_tensor(b_key).to("cuda:0").contiguous()
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                    raise RuntimeError(
                        f"invalid LoRA shapes for slot {slot} target {target}: "
                        f"A={tuple(a.shape)}, B={tuple(b.shape)}"
                    )
                rank = int(a.shape[0])
                adapter_weights[(slot, target)] = (
                    a.reshape(1, 1, rank, a.shape[1]),
                    (b * (alpha / rank)).contiguous().reshape(
                        1, 1, b.shape[0], rank
                    ),
                    rank,
                )
                available_targets.add(target)
    if not adapter_weights:
        raise RuntimeError(f"adapters contain no LoRA tensors: {adapter_paths}")

    target_stack_cache: dict[
        tuple[str, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]
    ] = {}
    reference_plan_cache: dict[tuple[Any, ...], Any] = {}

    def get_target_stack(
        target: str, active_slots: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        cache_key = (target, active_slots)
        cached = target_stack_cache.get(cache_key)
        if cached is not None:
            return cached
        entries = [adapter_weights.get((slot, target)) for slot in active_slots]
        if not any(entries):
            raise RuntimeError(f"no active adapter contains target {target}")
        present = [entry for entry in entries if entry is not None]
        assert present
        max_rank = max(int(entry[2]) for entry in present)
        input_hidden = int(present[0][0].shape[-1])
        output_hidden = int(present[0][1].shape[-2])
        a_stack = torch.zeros(
            (max(adapter_paths) + 1, 1, max_rank, input_hidden),
            device="cuda:0",
            dtype=present[0][0].dtype,
        )
        b_stack = torch.zeros(
            (max(adapter_paths) + 1, 1, output_hidden, max_rank),
            device="cuda:0",
            dtype=present[0][1].dtype,
        )
        ranks = [0] * (max(adapter_paths) + 1)
        for slot, entry in zip(active_slots, entries):
            if entry is None:
                continue
            a, b, rank = entry
            a_stack[slot, 0, :rank, : a.shape[-1]].copy_(a[0, 0])
            b_stack[slot, 0, : b.shape[-2], :rank].copy_(b[0, 0])
            ranks[slot] = rank
        result = (a_stack, b_stack, tuple(ranks))
        target_stack_cache[cache_key] = result
        return result

    producer = CudaIpcTransfer(connection, role="producer", device=0)
    consumer = CudaIpcTransfer(connection, role="consumer", device=0)
    metadata_cache: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] = {}
    localize_mode = _localize_x_mode()
    localize_active = os.environ.get("TETRIS_LORA_LOCALIZE_X_ACTIVE", "0") == "1"
    if localize_mode == "1" and not localize_active:
        raise TransportError(
            "TETRIS_LORA_LOCALIZE_X=1 requires CUDA peer access between GPUs"
        )
    x_workspace = _WorkerXWorkspace()
    x_copy_stream = torch.cuda.Stream(device=0) if localize_active else None
    delta_workspace: dict[tuple[Any, ...], torch.Tensor] = {}
    adapter_version = hashlib.sha256(
        "".join(
            f"{slot}:{adapter_versions[slot]}" for slot in sorted(adapter_versions)
        ).encode("utf-8")
    ).hexdigest()
    first_rank = next(int(entry[2]) for entry in adapter_weights.values())
    _send_json(
        connection,
        {
            "kind": "ready",
            "rank": first_rank,
            "target": target_module,
            "targets": sorted(available_targets),
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "adapter_versions": adapter_versions,
        },
    )
    _attach_compact_control(
        connection,
        device=0,
        endpoints=(producer, consumer),
        initiate=False,
    )
    compute_release = torch.cuda.Event(enable_timing=False, interprocess=True)
    timing_queue = _PendingTimingQueue(_WORKER_TIMING_POOL_SIZE)
    worker_inbox = consumer._inbox
    # First X completion_sequence is 2 (allocator starts at 1, seq=id*2).
    next_mailbox_x_seq = 2
    use_mailbox = (
        mailbox_control_enabled()
        and consumer._mailbox is not None
        and consumer._signal_ring is not None
        and consumer._signal_ring.peer_generations is not None
    )
    host_poll_s = (
        max(
            0.00005,
            float(os.environ.get("TETRIS_LORA_CONTROL_POLL_S", "0.0001")),
        )
        if use_mailbox
        else 0.05
    )
    try:
        while True:
            for sample in timing_queue.drain():
                _record_runtime_event(
                    "transport_phase",
                    phase=sample["phase"],
                    operation_id=sample.get("operation_id"),
                    layer_name=sample.get("layer_name"),
                    payload_bytes=sample.get("payload_bytes"),
                    delta_bytes=sample.get("delta_bytes"),
                    x_residency=sample.get("x_residency"),
                    x_peer_copy_us=sample.get("x_peer_copy_us"),
                    worker_receive_wait_us=sample.get("worker_receive_wait_us"),
                    duration_us=sample["duration_us"],
                    device="cuda:0",
                    path=sample.get("path"),
                    timing_scope=sample.get("timing_scope"),
                )
            receive_started_ns = time.monotonic_ns()
            control: dict[str, Any] | None = None
            descriptor = None
            remote_x = None
            x_ready_event = None
            from_mailbox = False
            # Multiplex host pipe (cold/close) with mailbox wake (steady).
            host_pending = False
            try:
                host_pending = bool(connection.poll(host_poll_s))
            except Exception:
                host_pending = False
            if (
                use_mailbox
                and not host_pending
                and consumer.try_poll_mailbox_x(next_mailbox_x_seq)
            ):
                consumer.wait_mailbox_publish(
                    next_mailbox_x_seq, delta=False, timeout_s=1.0
                )
                (
                    descriptor,
                    remote_x,
                    x_ready_event,
                    mailbox_compute,
                ) = consumer.materialize_from_mailbox(
                    next_mailbox_x_seq, delta=False
                )
                if not isinstance(mailbox_compute, dict):
                    raise TransportError(
                        "mailbox X publish is missing folded compute"
                    )
                control = mailbox_compute
                next_mailbox_x_seq += 2
                from_mailbox = True
                receive_finished_ns = time.monotonic_ns()
                control_receive_wait_us = (
                    receive_finished_ns - receive_started_ns
                ) / 1000.0
                control_receive_us = 0.0
            elif not host_pending and use_mailbox:
                # Nothing on pipe and mailbox not ready yet: retry.
                continue
            if not from_mailbox:
                if worker_inbox is not None:
                    control = worker_inbox.recv_control()
                    receive_finished_ns = time.monotonic_ns()
                    control_receive_wait_us = (
                        receive_finished_ns - receive_started_ns
                    ) / 1000.0
                    control_receive_us = 0.0
                else:
                    control, control_receive_wait_us, control_receive_us = (
                        _recv_json_timed(connection, consumer.registry)
                    )
                    receive_finished_ns = time.monotonic_ns()
            assert control is not None
            if control.get("kind") == "close":
                return
            operation_started_ns = time.monotonic_ns()
            descriptor_receive_started_ns = time.monotonic_ns()
            if from_mailbox:
                # Mailbox path already materialized X + folded compute.
                pass
            elif control.get("kind") in {"tensor", "tensor_ref"}:
                # Folded compute: one host message carries compute + X ref.
                compute_ctrl = control.get("compute")
                if not isinstance(compute_ctrl, dict):
                    raise TransportError(
                        "folded tensor control is missing compute payload"
                    )
                descriptor, remote_x, x_ready_event = consumer.receive_from_raw(
                    control
                )
                control = compute_ctrl
                # Keep mailbox sequence aligned after a cold host X.
                next_mailbox_x_seq = int(descriptor.completion_sequence) + 2
            elif control.get("kind") == "compute":
                descriptor, remote_x, x_ready_event = consumer.receive()
                next_mailbox_x_seq = int(descriptor.completion_sequence) + 2
            else:
                raise TransportError(f"unexpected worker command: {control}")
            descriptor_receive_finished_ns = time.monotonic_ns()
            _record_runtime_event(
                "transport_phase",
                phase="control_receive",
                operation_id=str(descriptor.operation_id),
                layer_name=str(descriptor.layer_name),
                duration_us=control_receive_us,
                start_ns=receive_finished_ns,
                end_ns=receive_finished_ns
                + max(0, int(control_receive_us * 1000.0)),
                direction="base_to_worker",
                timing_scope="json_decode_only",
                clock_domain="worker_monotonic",
            )
            _record_runtime_event(
                "transport_phase",
                phase="control_receive_wait",
                operation_id=str(descriptor.operation_id),
                layer_name=str(descriptor.layer_name),
                duration_us=control_receive_wait_us,
                start_ns=receive_started_ns,
                end_ns=receive_finished_ns,
                direction="base_to_worker",
                timing_scope="blocking_socket_receive_includes_inter_dispatch_wait",
                clock_domain="worker_monotonic",
            )
            _record_runtime_event(
                "transport_phase",
                phase="x_descriptor_receive",
                operation_id=str(descriptor.operation_id),
                layer_name=str(descriptor.layer_name),
                duration_us=(descriptor_receive_finished_ns - descriptor_receive_started_ns)
                / 1000.0,
                direction="base_to_worker",
                timing_scope="ipc_descriptor_receive_and_import",
            )
            x_for_compute = remote_x
            x_residency = "remote_ipc"
            pending_copy_timing: dict[str, Any] | None = None
            ready_span_timing = timing_queue.acquire()
            ready_span_start = ready_span_end = None
            ready_span_index = -1
            if ready_span_timing is not None:
                ready_span_start, ready_span_end, ready_span_index = (
                    ready_span_timing
                )
            if localize_active:
                assert x_copy_stream is not None
                copy_timing = timing_queue.acquire()
                timing_start = timing_end = None
                timing_index = -1
                if copy_timing is not None:
                    timing_start, timing_end, timing_index = copy_timing
                (
                    x_for_compute,
                    _copy_start,
                    _copy_end,
                    source_release_event,
                ) = _copy_x_to_local_workspace(
                    remote_x,
                    x_workspace,
                    x_copy_stream,
                    producer_ready=x_ready_event,
                    timing_start=timing_start,
                    timing_end=timing_end,
                    ready_span_start=ready_span_start,
                )
                x_residency = "local_p2p"
                if copy_timing is not None and _copy_end is not None:
                    pending_copy_timing = {
                        "start_event": timing_start,
                        "end_event": timing_end,
                        "event_index": timing_index,
                        "payload_bytes": int(
                            remote_x.numel() * remote_x.element_size()
                        ),
                    }
            elif x_ready_event is not None:
                # Direct remote-X read: compute stream must wait Base write.
                # No copy ran; do not invent timing tickets or mislabel residency.
                if ready_span_start is not None:
                    ready_span_start.record(torch.cuda.current_stream(0))
                torch.cuda.current_stream(0).wait_event(x_ready_event)
                x_residency = "remote_ipc"
            target = _normalize_target_module(str(control["target_module"]))
            requested_composition = tuple(
                tuple((int(slot), float(scale)) for slot, scale in entries)
                for entries in descriptor.request_composition
            )
            if not requested_composition:
                requested_composition = (
                    ((int(descriptor.slot_id), float(descriptor.scale)),),
                )
            row_layout = descriptor.row_layout or (int(remote_x.shape[0]),)
            request_ids = descriptor.request_ids or (descriptor.request_id,)
            active_slots = tuple(
                sorted(
                    {
                        int(slot)
                        for entries in requested_composition
                        for slot, _ in entries
                    }
                )
            )
            targets = _resolve_target_layout(target, active_slots, adapter_weights)
            if not targets:
                raise RuntimeError(f"worker has no active target(s) for {target}")
            # Packed Q/K/V Δ layout must follow the full Base projection, not
            # only the active LoRA targets (e.g. K+V still uses indices 1/2).
            full_split_targets = _resolve_target_modules(target)
            if target.endswith((".to_qkv", ".add_kv_proj")) and len(full_split_targets) > 1:
                descriptor_slices = tuple(
                    int(size) for size in descriptor.output_slices
                )
                if descriptor_slices and len(descriptor_slices) == len(
                    full_split_targets
                ):
                    output_slices = descriptor_slices
                elif descriptor_slices and len(descriptor_slices) == len(targets):
                    raise RuntimeError(
                        "independent Q/K/V LoRA output_slices must describe the "
                        f"full packed layout ({len(full_split_targets)}), not "
                        f"only active targets ({len(targets)})"
                    )
                else:
                    # Infer full packed widths; missing LoRA targets contribute 0.
                    inferred: list[int] = []
                    for split_target in full_split_targets:
                        if any(
                            (slot, split_target) in adapter_weights
                            for slot in active_slots
                        ):
                            inferred.append(
                                int(
                                    get_target_stack(split_target, active_slots)[
                                        1
                                    ].shape[-2]
                                )
                            )
                        elif descriptor_slices:
                            raise RuntimeError(
                                "missing LoRA target in packed projection but "
                                "descriptor.output_slices is incomplete; "
                                "cannot infer zero-Δ slice width"
                            )
                        else:
                            raise RuntimeError(
                                "independent Q/K/V LoRA requires output_slices "
                                "for the full packed Base projection layout"
                            )
                    output_slices = tuple(inferred)
            else:
                output_slices = tuple(
                    int(size) for size in descriptor.output_slices
                )
            x_flat = x_for_compute.reshape(-1, x_for_compute.shape[-1])
            compute_timing = timing_queue.acquire()
            compute_start = compute_end = None
            compute_timing_index = -1
            if compute_timing is not None:
                compute_start, compute_end, compute_timing_index = compute_timing
                compute_start.record(torch.cuda.current_stream(0))
            if target.endswith((".to_qkv", ".add_kv_proj")) and output_slices:
                output_features = sum(output_slices)
            else:
                output_features = None
            delta = None
            delta_cache_key: tuple[Any, ...] | None = None
            source_acked_after_lora = False
            signal_only_source_ack = (
                compact_control_enabled()
                and consumer._signal_ring is not None
                and consumer._signal_ring.peer_generations is not None
            )
            rank_tables = descriptor.slice_rank_tables
            for target_index, target_name in enumerate(targets):
                a_stack, b_stack, rank_table = get_target_stack(
                    target_name, active_slots
                )
                if x_flat.shape[-1] != a_stack.shape[-1]:
                    raise RuntimeError(
                        f"LoRA input shape mismatch for {target_name}: "
                        f"X={tuple(x_flat.shape)}, A={tuple(a_stack.shape)}"
                    )
                slice_index = _qkv_slice_index(target, target_name)
                rank_index = slice_index if slice_index is not None else 0
                if rank_tables and rank_index < len(rank_tables):
                    requested_ranks = rank_tables[rank_index]
                    rank_table = tuple(
                        int(requested_ranks[slot]) if slot < len(requested_ranks) else 0
                        for slot in range(len(rank_table))
                    )
                relations = tuple(
                    tuple(slot for slot, _ in entries)
                    for entries in requested_composition
                )
                scales = tuple(
                    tuple(scale for _, scale in entries)
                    for entries in requested_composition
                )
                # Preserve historical Tetris default (two_stage_tc) when unset.
                backend_name = resolve_backend_name(
                    explicit=(
                        (os.environ.get("VLLM_OMNI_LORA_BACKEND") or "").strip()
                        or BACKEND_TWO_STAGE_TC
                    )
                )
                if backend_name == BACKEND_TWO_STAGE_TC:
                    require_backend(BACKEND_TWO_STAGE_TC)
                    # K2: one TetrisTwoStageWorkspace per worker process.
                    tw_ws = getattr(_worker_main, "_two_stage_workspace", None)
                    if tw_ws is None:
                        from vllm_omni.diffusion.lora.tetris_lora_twostage import (
                            TetrisTwoStageWorkspace,
                        )

                        tw_ws = TetrisTwoStageWorkspace()
                        _worker_main._two_stage_workspace = tw_ws  # type: ignore[attr-defined]
                    worker_spans = tuple(
                        tuple(int(v) for v in span)
                        for span in (descriptor.row_spans or ())
                    )
                    delta_part, _ = execute_two_stage_lora(
                        x_flat,
                        a_stack,
                        b_stack,
                        row_layout,
                        relations,
                        scales,
                        rank_table,
                        module_name=target_name,
                        request_ids=request_ids,
                        workspace=tw_ws,
                        row_spans=worker_spans or None,
                    )
                elif backend_name == BACKEND_DENSE_FUSED:
                    dense_mod = get_dense_fused_ops()
                    delta_part = torch.zeros(
                        (x_flat.shape[0], int(b_stack.shape[-2])),
                        device=x_flat.device,
                        dtype=x_flat.dtype,
                    )
                    worker_spans = tuple(
                        tuple(int(v) for v in span)
                        for span in (descriptor.row_spans or ())
                    )
                    dense_mod.dense_fused_lora_add(
                        x_flat,
                        delta_part,
                        a_stack,
                        b_stack,
                        row_layout,
                        relations,
                        scales,
                        rank_table,
                        row_spans=worker_spans or None,
                    )
                elif backend_name == BACKEND_REFERENCE:
                    require_backend(BACKEND_REFERENCE)
                    from vllm_omni.diffusion.lora.multi_lora import (
                        MultiLoRAAdapter,
                        execute_lora,
                        prepare_lora_batch,
                    )

                    delta_part = torch.zeros(
                        (x_flat.shape[0], int(b_stack.shape[-2])),
                        device=x_flat.device,
                        dtype=x_flat.dtype,
                    )
                    plan_key = (
                        target_name,
                        tuple(active_slots),
                        tuple(int(rows) for rows in row_layout),
                        relations,
                        scales,
                        tuple(rank_table),
                        int(a_stack.data_ptr()),
                        int(b_stack.data_ptr()),
                    )
                    plan = reference_plan_cache.get(plan_key)
                    if plan is None:
                        registry = {}
                        for slot in sorted(
                            {slot for row in relations for slot in row}
                        ):
                            rank = (
                                int(rank_table[slot])
                                if slot < len(rank_table)
                                else 0
                            )
                            registry[slot] = MultiLoRAAdapter(
                                adapter_id=int(slot),
                                lora_a=a_stack[slot, 0, :rank, :],
                                lora_b=b_stack[slot, 0, :, :rank],
                            )
                        plan = prepare_lora_batch(
                            module_identity=target_name,
                            input_row_layout=tuple(
                                int(rows) for rows in row_layout
                            ),
                            request_to_adapter_relation=relations,
                            adapter_registry_snapshot=registry,
                            request_scales=scales,
                            execution_ownership="tetris_worker",
                        )
                        reference_plan_cache[plan_key] = plan
                    last_gemm_end: list[Any | None] = [None]
                    gemm_calls = 0
                    expected_gemm_calls = sum(
                        2
                        for request_slots in plan.relations
                        for slot in request_slots
                        if int(slot) in plan.registry
                        and plan.registry[int(slot)].lora_a.numel() > 0
                        and plan.registry[int(slot)].lora_b.numel() > 0
                    )

                    def time_worker_gemm(gemm_name, operation):
                        nonlocal gemm_calls, source_acked_after_lora
                        gemm_timing = timing_queue.acquire()
                        if gemm_timing is None:
                            result = operation()
                            gemm_end = torch.cuda.Event(enable_timing=False)
                            gemm_end.record(torch.cuda.current_stream(0))
                            last_gemm_end[0] = gemm_end
                        else:
                            gemm_start, gemm_end, gemm_index = gemm_timing
                            stream = torch.cuda.current_stream(0)
                            gemm_start.record(stream)
                            result = operation()
                            gemm_end.record(stream)
                            last_gemm_end[0] = gemm_end
                            timing_queue.submit(
                                start_event=gemm_start,
                                end_event=gemm_end,
                                event_index=gemm_index,
                                phase="pure_lora_gemm",
                                operation_id=descriptor.operation_id,
                                layer_name=target_name,
                                execution_location="decoupled_worker",
                                executor="execute_lora",
                                gemm=gemm_name,
                                timing_scope="pure_lora_cuda_event",
                                device="cuda:0",
                            )
                        gemm_calls += 1
                        if (
                            signal_only_source_ack
                            and target_index == len(targets) - 1
                            and gemm_name == "lora_b"
                            and gemm_calls == expected_gemm_calls
                        ):
                            consumer.acknowledge(
                                descriptor,
                                consumer_complete=True,
                                completion_event=gemm_end,
                                host=False,
                            )
                            source_acked_after_lora = True
                        return result

                    execute_lora(
                        x_flat,
                        delta_part,
                        plan,
                        gemm_timer=time_worker_gemm,
                    )
                    if (
                        target_index == len(targets) - 1
                        and ready_span_end is not None
                    ):
                        ready_span_end.record(torch.cuda.current_stream(0))
                        timing_queue.submit(
                            start_event=ready_span_start,
                            end_event=ready_span_end,
                            event_index=ready_span_index,
                            phase="remote_compute_ready_span",
                            operation_id=descriptor.operation_id,
                            layer_name=target_name,
                            timing_scope=(
                                "worker_cuda_span_x_copy_start_to_last_lora_gemm_end"
                            ),
                            device="cuda:0",
                        )
                    if (
                        signal_only_source_ack
                        and target_index == len(targets) - 1
                        and not source_acked_after_lora
                    ):
                        if last_gemm_end[0] is None:
                            last_gemm_end[0] = torch.cuda.Event(enable_timing=False)
                            last_gemm_end[0].record(torch.cuda.current_stream(0))
                        consumer.acknowledge(
                            descriptor,
                            consumer_complete=True,
                            completion_event=last_gemm_end[0],
                            host=False,
                        )
                        source_acked_after_lora = True
                else:
                    raise LoRABackendError(
                        f"TetrisLoRA worker performance path refuses "
                        f"backend={backend_name!r}; use dense_fused, "
                        "two_stage_tc, or reference"
                    )
                if output_features is None:
                    output_features = int(delta_part.shape[1])
                if delta is None:
                    delta_cache_key = (
                        target,
                        tuple(int(value) for value in x_flat.shape),
                        int(output_features),
                        str(x_flat.dtype),
                    )
                    delta = delta_workspace.get(delta_cache_key)
                    if delta is None or delta.device != x_flat.device:
                        delta = torch.empty(
                            (x_flat.shape[0], output_features),
                            device=x_flat.device,
                            dtype=x_flat.dtype,
                        )
                        delta_workspace[delta_cache_key] = delta
                    # D6: full single-slice overwrite does not need zero_;
                    # QKV partial writes still require a clean buffer.
                    if slice_index is not None or len(targets) > 1:
                        delta.zero_()
                if slice_index is None:
                    if delta_part.shape[1] != output_features:
                        raise RuntimeError(
                            f"fused LoRA output width mismatch for {target}: "
                            f"{delta_part.shape[1]} != {output_features}"
                        )
                    delta.copy_(delta_part)
                else:
                    if not output_slices or slice_index >= len(output_slices):
                        raise RuntimeError(
                            "independent Q/K/V LoRA requires output_slices metadata"
                        )
                    output_offset = sum(output_slices[:slice_index])
                    output_width = output_slices[slice_index]
                    if delta_part.shape[1] != output_width:
                        raise RuntimeError(
                            f"QKV slice width mismatch for {target_name}: "
                            f"{delta_part.shape[1]} != {output_width}"
                        )
                    delta[:, output_offset : output_offset + output_width].copy_(
                        delta_part
                    )
            if delta is None:
                if output_features is None:
                    raise RuntimeError(f"worker has no output shape for {target}")
                delta = torch.zeros(
                    (x_flat.shape[0], output_features),
                    device=x_flat.device,
                    dtype=x_flat.dtype,
                )
                delta_cache_key = (
                    target,
                    tuple(int(value) for value in x_flat.shape),
                    int(output_features),
                    str(x_flat.dtype),
                )
            delta = delta.reshape(*x_for_compute.shape[:-1], delta.shape[-1])
            if compute_end is not None:
                compute_end.record(torch.cuda.current_stream(0))
            if not localize_active:
                compute_release.record(torch.cuda.current_stream(0))
                source_release_event = compute_release
            output_descriptor = TensorDescriptor(
                **{
                    **descriptor.__dict__,
                    "operation_id": f"{descriptor.operation_id}:delta",
                    "buffer_token": f"{descriptor.buffer_token}:delta",
                    "shape": tuple(int(value) for value in delta.shape),
                    "stride": tuple(int(value) for value in delta.stride()),
                    "payload_bytes": int(delta.numel() * delta.element_size()),
                    "buffer_capacity": int(delta.numel() * delta.element_size()),
                    "completion_sequence": descriptor.completion_sequence + 1,
                    "layer_name": target,
                    "rank": max(
                        (
                            int(rank)
                            for request_rank in descriptor.request_ranks
                            for rank in request_rank
                        ),
                        default=int(descriptor.rank),
                    ),
                }
            )
            delta_cache = (
                f"delta:{delta_cache_key}"
                if delta_cache_key is not None
                else output_descriptor.buffer_token
            )
            # Compact+ring: X release is generation-fenced GPU doorbell only
            # (no host source_ack piggyback). Legacy compact without a peer
            # generation table still piggybacks on the Δ host frame.
            pipeline_delta_first = compact_control_enabled()
            skip_delta_host_wait = (
                pipeline_delta_first and operation_batch_ack_enabled()
            )
            ack_started_ns = time.monotonic_ns()
            if signal_only_source_ack:
                if not source_acked_after_lora:
                    consumer.acknowledge(
                        descriptor,
                        consumer_complete=True,
                        completion_event=source_release_event,
                        host=False,
                    )
                ack_finished_ns = time.monotonic_ns()
                delta_publish_ns = time.monotonic_ns()
                producer.send(
                    output_descriptor, delta, cache_key=delta_cache
                )
            elif pipeline_delta_first:
                source_ack = consumer.build_compact_ack(
                    descriptor,
                    consumer_complete=True,
                    completion_event=source_release_event,
                )
                delta_publish_ns = time.monotonic_ns()
                producer.send(
                    output_descriptor,
                    delta,
                    cache_key=delta_cache,
                    source_ack=source_ack,
                )
                ack_finished_ns = time.monotonic_ns()
            else:
                # Producer waits for this ACK next; flush immediately so batching
                # cannot stall progress behind a future target.
                consumer.acknowledge(
                    descriptor,
                    consumer_complete=True,
                    completion_event=source_release_event,
                    flush=True,
                )
                delta_publish_ns = time.monotonic_ns()
                producer.send(
                    output_descriptor, delta, cache_key=delta_cache
                )
                ack_finished_ns = time.monotonic_ns()
            output_wait_started_ns = time.monotonic_ns()
            if skip_delta_host_wait:
                # GPU fence: Base holds the client lock across begin→finish, so
                # the next folded X publish (producer-ready Event on the Base
                # stream) happens-after the previous Δ P2P copy+merge enqueue.
                # Worker only overwrites the Δ workspace after that next X
                # materialize wait; do not host-wait the Δ consumer ACK here.
                output_wait_finished_ns = output_wait_started_ns
            else:
                producer.wait_ack(output_descriptor)
                output_wait_finished_ns = time.monotonic_ns()
            # Never elapsed_time() here: Events may still be in-flight after
            # skip ACK. Defer to the pending timing queue (query-then-read).
            if pending_copy_timing is not None:
                timing_queue.submit(
                    start_event=pending_copy_timing["start_event"],
                    end_event=pending_copy_timing["end_event"],
                    event_index=int(pending_copy_timing["event_index"]),
                    phase="x_peer_copy",
                    operation_id=descriptor.operation_id,
                    layer_name=target,
                    payload_bytes=int(pending_copy_timing["payload_bytes"]),
                    device="cuda:0",
                    path="cuda_p2p",
                    timing_scope="deferred_event_query",
                )
            if (
                compute_start is not None
                and compute_end is not None
                and compute_timing_index >= 0
            ):
                timing_queue.submit(
                    start_event=compute_start,
                    end_event=compute_end,
                    event_index=int(compute_timing_index),
                    phase="remote_compute",
                    operation_id=descriptor.operation_id,
                    layer_name=target,
                    payload_bytes=int(remote_x.numel() * remote_x.element_size()),
                    delta_bytes=int(delta.numel() * delta.element_size()),
                    timing_scope="deferred_event_query",
                )
            # Host-enqueue markers only; GPU durations come from drain().
            remote_compute_us = 0.0
            x_peer_copy_us = 0.0
            _record_runtime_event(
                "transport_phase",
                phase="remote_compute",
                operation_id=descriptor.operation_id,
                layer_name=target,
                payload_bytes=int(remote_x.numel() * remote_x.element_size()),
                delta_bytes=int(delta.numel() * delta.element_size()),
                x_residency=x_residency,
                x_peer_copy_us=x_peer_copy_us,
                worker_receive_wait_us=(descriptor_receive_finished_ns - operation_started_ns)
                / 1000.0,
                duration_us=0.0,
                timing_scope="host_enqueue_only_gpu_duration_deferred",
                device="cuda:0",
            )
            _record_runtime_event(
                "transport_phase",
                phase="lora_compute",
                operation_id=descriptor.operation_id,
                layer_name=target,
                payload_bytes=int(remote_x.numel() * remote_x.element_size()),
                delta_bytes=int(delta.numel() * delta.element_size()),
                x_residency=x_residency,
                x_peer_copy_us=x_peer_copy_us,
                duration_us=0.0,
                timing_scope="host_enqueue_only_same_interval_as_remote_compute",
                device="cuda:0",
            )
            _record_runtime_event(
                "transport_phase",
                phase="control_ack",
                operation_id=descriptor.operation_id,
                layer_name=target,
                duration_us=(
                    0.0
                    if signal_only_source_ack
                    else (ack_finished_ns - ack_started_ns) / 1000.0
                ),
                start_ns=ack_started_ns,
                end_ns=ack_finished_ns,
                direction="worker_to_base_source_release",
                mechanism=(
                    "signal_ring_generation_fenced"
                    if signal_only_source_ack
                    else "host_ack_or_piggyback"
                ),
                clock_domain="worker_monotonic",
            )
            _record_runtime_event(
                "transport_phase",
                phase="delta_consumer_wait",
                operation_id=descriptor.operation_id,
                layer_name=target,
                duration_us=(output_wait_finished_ns - output_wait_started_ns)
                / 1000.0,
                direction="worker_waits_for_base_merge_event",
                mechanism=(
                    "next_compute_happens_after_merge"
                    if skip_delta_host_wait
                    else "host_ack"
                ),
            )
            _record_runtime_event(
                "transport_phase",
                phase="worker_host_sync",
                operation_id=descriptor.operation_id,
                layer_name=target,
                duration_us=0.0,
                call_count=0,
                mechanism="cuda_event_dependency",
            )
            _record_runtime_event(
                "transport_phase",
                phase="delta_ipc_lifetime",
                operation_id=descriptor.operation_id,
                layer_name=target,
                payload_bytes=int(delta.numel() * delta.element_size()),
                start_ns=delta_publish_ns,
                end_ns=time.monotonic_ns(),
                timing_scope="publish_through_base_consumer_ack",
            )
    finally:
        producer.close()
        consumer.close()


class TetrisLoRAWorkerClient:
    """Persistent Base-side client for one adapter worker."""

    def __init__(
        self,
        adapter_paths: dict[int, str],
        *,
        adapter_id: str,
        physical_device: int,
        base_device: int = 0,
    ) -> None:
        session_setup_started_ns = time.monotonic_ns()
        self.transport = os.environ.get("TETRIS_LORA_TRANSPORT", "ipc").lower()
        if self.transport != "ipc":
            raise TransportError(
                f"transport {self.transport!r} is not yet wired to the worker "
                "client; refusing implicit IPC fallback"
            )
        self.adapter_paths = {
            int(slot): str(Path(path).resolve())
            for slot, path in adapter_paths.items()
        }
        if not self.adapter_paths:
            raise ValueError("TetrisLoRA worker requires at least one adapter path")
        # Hash once at session init so begin_tetris_lora only hits the cache.
        for adapter_path in self.adapter_paths.values():
            register_adapter_version(adapter_path)
        self.adapter_id = str(adapter_id)
        self.physical_device = int(physical_device)
        self.base_device = int(base_device)
        localize_mode = _localize_x_mode()
        peer_access = _physical_peer_access(
            self.base_device, self.physical_device
        )
        if localize_mode == "1" and not peer_access:
            raise TransportError(
                "TETRIS_LORA_LOCALIZE_X=1 requires CUDA peer access between "
                f"base GPU {self.base_device} and LoRA GPU {self.physical_device}"
            )
        self.localize_x = localize_mode != "0" and peer_access
        self.localize_x_fallback_reason = (
            "peer_access_unavailable"
            if localize_mode == "auto" and not peer_access
            else None
        )
        parent_socket, child_socket = socket.socketpair()
        worker_log = None
        worker_log_path = os.environ.get("TETRIS_LORA_WORKER_LOG")
        if worker_log_path:
            worker_log = open(worker_log_path, "ab", buffering=0)
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from multiprocessing.connection import Connection; "
                "from vllm_omni.diffusion.lora.tetris_lora_runtime import "
                "_worker_main; import sys; "
                "_worker_main(Connection(int(sys.argv[1])), int(sys.argv[2]), "
                "sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6] == '1')",
                str(child_socket.fileno()),
                str(self.physical_device),
                json.dumps(self.adapter_paths, sort_keys=True),
                _target_module(),
                self.adapter_id,
                "1"
                if os.environ.get("TETRIS_LORA_ALL_TARGETS", "").lower()
                in {"1", "true", "yes", "on"}
                else "0",
            ],
            pass_fds=(child_socket.fileno(),),
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(self.physical_device),
                "TETRIS_LORA_LOCALIZE_X_ACTIVE": "1"
                if self.localize_x
                else "0",
            },
            stdin=subprocess.DEVNULL,
            stdout=worker_log,
            stderr=worker_log,
        )
        if worker_log is not None:
            worker_log.close()
        child_socket.close()
        self._parent = Connection(parent_socket.detach(), readable=True, writable=True)
        ready = _recv_json(self._parent)
        if ready.get("kind") != "ready":
            self.close()
            raise RuntimeError(f"TetrisLoRA worker failed to initialize: {ready}")
        if ready.get("adapter_id") != self.adapter_id:
            self.close()
            raise RuntimeError("TetrisLoRA worker adapter identity mismatch")
        self._send = CudaIpcTransfer(
            self._parent, role="producer", device=self.base_device
        )
        self._receive = CudaIpcTransfer(
            self._parent, role="consumer", device=self.base_device
        )
        _attach_compact_control(
            self._parent,
            device=self.base_device,
            endpoints=(self._send, self._receive),
            initiate=True,
        )
        self._delta_workspace: dict[tuple[Any, ...], torch.Tensor] = {}
        self._delta_copy_stream = torch.cuda.Stream(device=self.base_device)
        event_pool_size = max(
            8, int(os.environ.get("TETRIS_LORA_EVENT_POOL_SIZE", "512"))
        )
        self._delta_copy_event_pool = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(event_pool_size)
        ]
        self._delta_copy_event_in_use = [False] * event_pool_size
        self._delta_copy_event_cursor = 0
        self._merge_start = torch.cuda.Event(enable_timing=True)
        self._merge_end = torch.cuda.Event(enable_timing=False, interprocess=True)
        # D9: bounded fence Event pool (independent of timing pool).
        fence_pool_size = max(
            8, int(os.environ.get("TETRIS_LORA_FENCE_POOL_SIZE", "64"))
        )
        self._fence_event_pool = [
            torch.cuda.Event(enable_timing=False) for _ in range(fence_pool_size)
        ]
        self._fence_event_cursor = 0
        self._delta_workspace_in_use = False
        self._pending_delta_copy_timings: list[dict[str, Any]] = []
        self._x_slots: dict[tuple[Any, ...], torch.Tensor] = {}
        self._x_slot_warm: set[tuple[Any, ...]] = set()
        # D2→multi-slot: bounded in-flight ops with independent generation and
        # explicit per-slot Δ consume fences (do not assume next-X ⇒ prior-Δ done).
        self._op_slot_count = max(
            1, min(8, int(os.environ.get("TETRIS_LORA_OP_SLOTS", "2")))
        )
        self._op_slot_free = [True] * self._op_slot_count
        self._op_slot_generation = [0] * self._op_slot_count
        self._op_slot_merge_end = [
            torch.cuda.Event(enable_timing=False, interprocess=True)
            for _ in range(self._op_slot_count)
        ]
        self._op_slot_delta_workspace: list[dict[tuple[Any, ...], torch.Tensor]] = [
            {} for _ in range(self._op_slot_count)
        ]
        self._op_slot_delta_in_use = [False] * self._op_slot_count
        self._lock = threading.Lock()
        self._inflight_cond = threading.Condition(self._lock)
        self._inflight_ops: dict[str, int] = {}
        self._sequence = 0
        self._op_ids = OperationIdAllocator()
        self._send.bind_wire_session(self._op_ids.epoch)
        self._receive.bind_wire_session(self._op_ids.epoch)
        self._rank = int(ready["rank"])
        self.session_epoch = uuid.uuid4().hex
        self.adapter_version = str(ready["adapter_version"])
        _record_runtime_event(
            "transport_capability",
            backend="ipc",
            x_residency="local_p2p" if self.localize_x else "remote_ipc",
            localize_x_mode=localize_mode,
            fallback_reason=self.localize_x_fallback_reason,
            base_device=self.base_device,
            lora_device=self.physical_device,
        )
        _record_runtime_event(
            "transport_phase",
            phase="session_setup",
            duration_us=(time.monotonic_ns() - session_setup_started_ns) / 1000.0,
            timing_scope="worker_process_connection_ipc_registration_and_pool_init",
            persistent_session=True,
        )

    def begin_delta(
        self,
        x: torch.Tensor,
        scale: float,
        *,
        module_name: str,
        context: dict[str, Any] | None = None,
        composition: tuple[tuple[tuple[int, float], ...], ...] = (),
        rank_tables: tuple[tuple[int, ...], ...] = (),
        output_slices: tuple[int, ...] = (),
        row_layout: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        if x.ndim != 2 or not x.is_cuda:
            raise ValueError("TetrisLoRA expects a flattened CUDA activation")
        self._inflight_cond.acquire()
        slot_idx: int | None = None
        operation_id: str | None = None
        try:
            while not any(self._op_slot_free):
                if not self._inflight_cond.wait(timeout=120.0):
                    raise TransportError(
                        "TetrisLoRA op-slot backpressure timeout; "
                        f"inflight={len(self._inflight_ops)} "
                        f"slots={self._op_slot_count}"
                    )
            slot_idx = next(
                index
                for index, free in enumerate(self._op_slot_free)
                if free
            )
            self._op_slot_free[slot_idx] = False
            self._op_slot_generation[slot_idx] += 1
            slot_generation = int(self._op_slot_generation[slot_idx])
            operation_started_ns = time.monotonic_ns()
            self._sequence += 1
            operation_id = str(self._op_ids.next_id())
            self._send.bind_wire_session(self._op_ids.epoch)
            self._receive.bind_wire_session(self._op_ids.epoch)
            context = dict(context or {})
            normalized_composition = tuple(
                tuple((int(slot), float(branch_scale)) for slot, branch_scale in entries)
                for entries in composition
            )
            normalized_row_layout = tuple(int(rows) for rows in row_layout)
            if not normalized_composition:
                normalized_composition = (((0, float(scale)),),)
            if not normalized_row_layout:
                normalized_row_layout = (int(x.shape[0]),)
            if len(normalized_row_layout) != len(normalized_composition):
                raise ValueError(
                    "TetrisLoRA composition/row_layout length mismatch: "
                    f"{len(normalized_composition)} vs {len(normalized_row_layout)}"
                )
            if sum(normalized_row_layout) != int(x.shape[0]):
                raise ValueError(
                    "TetrisLoRA row_layout does not cover X rows: "
                    f"sum={sum(normalized_row_layout)} x_rows={int(x.shape[0])}"
                )
            request_ids_ctx = tuple(context.get("request_ids", ()))
            if request_ids_ctx and len(request_ids_ctx) != len(normalized_composition):
                raise ValueError(
                    "TetrisLoRA request_ids length does not match composition"
                )
            primary_ranks = rank_tables[0] if rank_tables else (self._rank,)
            request_ranks = tuple(
                tuple(
                    int(primary_ranks[slot])
                    if slot < len(primary_ranks)
                    else self._rank
                    for slot, _ in entries
                )
                for entries in normalized_composition
            )
            # Mixed batch may start with an empty (no-LoRA) request; descriptor
            # scalars come from the first non-empty branch, empty rows stay 0-Δ.
            descriptor_slot = 0
            descriptor_scale = float(scale)
            descriptor_rank = int(self._rank)
            for entries, ranks in zip(normalized_composition, request_ranks):
                if entries:
                    descriptor_slot, descriptor_scale = entries[0]
                    descriptor_rank = int(ranks[0]) if ranks else int(self._rank)
                    break
            descriptor = TensorDescriptor(
                session_epoch=self.session_epoch,
                operation_id=operation_id,
                request_id=str(context.get("request_id", operation_id)),
                batch_id=str(context.get("batch_id", operation_id)),
                denoise_step=int(context.get("denoise_step", -1)),
                layer_name=_normalize_target_module(module_name),
                invocation_id=self._sequence,
                slot_id=int(descriptor_slot),
                slot_generation=int(slot_generation),
                shape=tuple(int(value) for value in x.shape),
                stride=tuple(int(value) for value in x.stride()),
                dtype=str(x.dtype),
                layout="strided",
                payload_bytes=int(x.numel() * x.element_size()),
                buffer_token="pending",
                adapter_id=self.adapter_id,
                adapter_version=self.adapter_version,
                rank=int(descriptor_rank),
                scale=float(descriptor_scale),
                branch_index=0,
                output_offset=0,
                completion_sequence=self._sequence * 2,
                request_ids=request_ids_ctx,
                row_spans=tuple(
                    tuple(span) for span in context.get("row_spans", ())
                ),
                buffer_capacity=int(x.numel() * x.element_size()),
                offset=0,
                alpha_folded=True,
                request_composition=normalized_composition,
                request_ranks=request_ranks,
                row_layout=normalized_row_layout,
                slice_rank_tables=tuple(
                    tuple(int(rank) for rank in ranks) for ranks in rank_tables
                ),
                output_slices=tuple(int(size) for size in output_slices),
            )
            # Per op-slot X buffer: reuse only after that slot's prior Δ consume.
            if self._op_slot_delta_in_use[slot_idx]:
                torch.cuda.current_stream(x.device).wait_event(
                    self._op_slot_merge_end[slot_idx]
                )
                self._op_slot_delta_in_use[slot_idx] = False
            slot_key = (
                int(slot_idx),
                tuple(int(value) for value in x.shape),
                tuple(int(value) for value in x.stride()),
                str(x.dtype),
            )
            x_slot = self._x_slots.get(slot_key)
            if x_slot is None:
                x_slot = torch.empty_like(x, device=x.device)
                self._x_slots[slot_key] = x_slot
                _record_runtime_event(
                    "transport_phase",
                    phase="x_slot_registration",
                    operation_id=operation_id,
                    payload_bytes=descriptor.payload_bytes,
                    duration_us=0.0,
                    setup="persistent_x_slot_registration",
                    op_slot=int(slot_idx),
                )
            descriptor = TensorDescriptor(
                **{
                    **descriptor.__dict__,
                    "shape": tuple(int(value) for value in x_slot.shape),
                    "stride": tuple(int(value) for value in x_slot.stride()),
                    "payload_bytes": int(x_slot.numel() * x_slot.element_size()),
                    "buffer_capacity": int(x_slot.numel() * x_slot.element_size()),
                    "buffer_token": (
                        "xslot:"
                        + f"{slot_idx}:"
                        + ":".join(str(value) for value in slot_key[1])
                        + ":"
                        + ":".join(str(value) for value in slot_key[2])
                        + ":"
                        + slot_key[3]
                    ),
                }
            )
            copy_start_ns = time.monotonic_ns()
            x_slot.copy_(x, non_blocking=True)
            publish_started_ns = time.monotonic_ns()
            compute_payload = {
                "kind": "compute",
                "scale": float(scale),
                "target_module": _normalize_target_module(module_name),
                "context": context,
            }
            if binary_wire_enabled() or compact_control_enabled():
                # One host message: folded compute + session X slot ref.
                # Binary mode carries that message as a plan id plus a frame.
                self._send.send(
                    descriptor,
                    x_slot,
                    cache_key=descriptor.buffer_token,
                    compute=compute_payload,
                )
            else:
                _send_json(self._parent, compute_payload)
                self._send.send(
                    descriptor,
                    x_slot,
                    cache_key=descriptor.buffer_token,
                )
            publish_finished_ns = time.monotonic_ns()
            _record_runtime_event(
                "transport_phase",
                phase="control_publish",
                operation_id=operation_id,
                layer_name=descriptor.layer_name,
                duration_us=(publish_finished_ns - publish_started_ns) / 1000.0,
                start_ns=publish_started_ns,
                end_ns=publish_finished_ns,
                direction="base_to_worker",
                clock_domain="base_monotonic",
                mechanism=getattr(self._send, "last_publish_mechanism", "unknown"),
            )
            if getattr(self._send, "last_publish_mechanism", None) == "mailbox_ring":
                _record_runtime_event(
                    "transport_phase",
                    phase="control_publish_mailbox",
                    operation_id=operation_id,
                    layer_name=descriptor.layer_name,
                    duration_us=float(self._send.last_mailbox_publish_us),
                    direction="base_to_worker",
                    clock_domain="base_monotonic",
                    mechanism="mailbox_ring",
                    timing_scope="mailbox_entry_write_and_ring_doorbell_only",
                )
            _record_runtime_event(
                "transport_phase",
                phase="x_slot_copy",
                operation_id=operation_id,
                payload_bytes=descriptor.payload_bytes,
                start_ns=copy_start_ns,
                end_ns=publish_started_ns,
                duration_us=(publish_started_ns - copy_start_ns) / 1000.0,
                timing_scope="base_device_copy_into_persistent_ipc_slot",
            )
            _record_runtime_event(
                "transport_phase",
                phase="x_ipc_publish",
                operation_id=operation_id,
                layer_name=descriptor.layer_name,
                payload_bytes=descriptor.payload_bytes,
                start_ns=publish_started_ns,
                end_ns=publish_finished_ns,
                duration_us=(publish_finished_ns - publish_started_ns) / 1000.0,
                timing_scope="descriptor_and_cuda_ipc_handle_publish_not_payload_copy",
                op_slot=int(slot_idx),
            )
            self._inflight_ops[operation_id] = int(slot_idx)
            return {
                "descriptor": descriptor,
                "active": True,
                "begin_host_started_ns": operation_started_ns,
                "begin_host_finished_ns": time.monotonic_ns(),
                "op_slot": int(slot_idx),
                "slot_generation": int(slot_generation),
            }
        except BaseException:
            if slot_idx is not None:
                self._op_slot_free[slot_idx] = True
            if operation_id is not None:
                self._inflight_ops.pop(operation_id, None)
            self._inflight_cond.notify_all()
            raise
        finally:
            self._inflight_cond.release()

    def finish_delta(
        self, operation: dict[str, Any], y: torch.Tensor
    ) -> torch.Tensor:
        if operation.get("cancelled"):
            raise TransportError(
                "TetrisLoRA operation was cancelled; refuse finish on broken op"
            )
        if not operation.get("active"):
            raise TransportError("TetrisLoRA operation is not active")
        descriptor = operation["descriptor"]
        slot_idx = int(operation.get("op_slot", 0))
        merge_end = self._op_slot_merge_end[slot_idx]
        slot_workspace = self._op_slot_delta_workspace[slot_idx]
        try:
            finish_host_started_ns = time.monotonic_ns()
            self._drain_delta_copy_timings()
            wait_started_ns = time.monotonic_ns()
            signal_only_wait = (
                compact_control_enabled()
                and binary_wire_enabled()
                and self._send._signal_ring is not None
                and self._send._signal_ring.peer_generations is not None
            )
            ready_wait_timing = (
                self._acquire_delta_copy_events() if signal_only_wait else None
            )
            if ready_wait_timing is not None:
                ready_wait_start, ready_wait_end, ready_wait_event_index = (
                    ready_wait_timing
                )
                ready_wait_start.record(torch.cuda.current_stream(y.device))
            if signal_only_wait:
                self._send.wait_ack_signal(descriptor)
            else:
                self._send.wait_ack(descriptor)
            if ready_wait_timing is not None:
                ready_wait_end.record(torch.cuda.current_stream(y.device))
                self._pending_delta_copy_timings.append(
                    {
                        "operation_id": descriptor.operation_id,
                        "layer_name": descriptor.layer_name,
                        "payload_bytes": int(descriptor.payload_bytes),
                        "start_event": ready_wait_start,
                        "end_event": ready_wait_end,
                        "event_index": ready_wait_event_index,
                        "phase": "remote_compute_ready_wait_cuda",
                        "timing_scope": "base_cuda_stream_wait_for_worker_lora_ready",
                    }
                )
            remote_ready_ns = time.monotonic_ns()
            delta_receive_started_ns = time.monotonic_ns()
            delta_seq = int(descriptor.completion_sequence) + 1
            use_delta_mailbox = (
                mailbox_control_enabled()
                and self._receive._mailbox is not None
                and self._receive._signal_ring is not None
                and self._receive._signal_ring.peer_generations is not None
            )
            used_delta_mailbox = False
            if use_delta_mailbox:
                # Blocking wait: Worker publishes Δ mailbox after X-release
                # (host cold path also mirrors the wake).
                self._receive.wait_mailbox_publish(
                    delta_seq, delta=True, timeout_s=30.0
                )
                if self._receive.try_poll_mailbox_delta(delta_seq):
                    try:
                        output_descriptor, delta, delta_ready_event, _compute = (
                            self._receive.materialize_from_mailbox(
                                delta_seq, delta=True
                            )
                        )
                        used_delta_mailbox = True
                    except TransportError as exc:
                        # Cold path: plan is known but IPC cache still needs
                        # the host TENSOR import for this registration.
                        if "unknown IPC cache key" not in str(exc):
                            raise
                        output_descriptor, delta, delta_ready_event = (
                            self._receive.receive()
                        )
                else:
                    # Mailbox wake beat the host PLAN frame; drain host.
                    output_descriptor, delta, delta_ready_event = (
                        self._receive.receive()
                    )
            else:
                output_descriptor, delta, delta_ready_event = self._receive.receive()
            receive_finished_ns = time.monotonic_ns()
            _record_runtime_event(
                "transport_phase",
                phase="delta_descriptor_receive",
                operation_id=descriptor.operation_id,
                layer_name=descriptor.layer_name,
                duration_us=(receive_finished_ns - delta_receive_started_ns) / 1000.0,
                start_ns=delta_receive_started_ns,
                end_ns=receive_finished_ns,
                direction="worker_to_base",
                timing_scope=(
                    "mailbox_delta_materialize"
                    if used_delta_mailbox
                    else "ipc_descriptor_receive_and_import"
                ),
                mechanism=(
                    "signal_ring_mailbox" if used_delta_mailbox else "host_pipe"
                ),
                clock_domain="base_monotonic",
            )
            if output_descriptor.operation_id != f"{descriptor.operation_id}:delta":
                raise TransportError("LoRA delta operation id mismatch")
            # D7: reshape must alias original storage or Δ is lost.
            y_flat = y.reshape(-1, y.shape[-1])
            if y_flat.data_ptr() != y.data_ptr() or y_flat.storage_offset() != y.storage_offset():
                raise TransportError(
                    "TetrisLoRA finish_delta requires y.reshape to view the same "
                    "storage; got a copy (non-contiguous or unsupported layout)"
                )
            delta_flat = delta.reshape(-1, delta.shape[-1])
            if y_flat.shape != delta_flat.shape:
                raise TransportError(
                    f"LoRA residual shape {tuple(delta_flat.shape)} does not match "
                    f"Base output {tuple(y_flat.shape)}"
                )
            workspace_key = (
                str(delta.dtype),
                tuple(int(value) for value in delta.shape),
                tuple(int(value) for value in y_flat.shape),
            )
            local_delta = slot_workspace.get(workspace_key)
            if local_delta is None or local_delta.device != y_flat.device:
                local_delta = torch.empty_like(delta, device=y_flat.device)
                slot_workspace[workspace_key] = local_delta
                self._delta_workspace[workspace_key] = local_delta
            current_stream = torch.cuda.current_stream(y.device)
            self._drain_delta_copy_timings()
            timing_pair = self._acquire_delta_copy_events()
            # Fence Events are independent of timing-pool Events so telemetry
            # exhaustion never blocks the copy or invents a host sync.
            fence_end = self._fence_event_pool[
                self._fence_event_cursor % len(self._fence_event_pool)
            ]
            self._fence_event_cursor += 1
            with torch.cuda.stream(self._delta_copy_stream):
                if delta_ready_event is not None:
                    self._delta_copy_stream.wait_event(delta_ready_event)
                if self._op_slot_delta_in_use[slot_idx]:
                    self._delta_copy_stream.wait_event(merge_end)
                if timing_pair is not None:
                    delta_copy_start, delta_copy_end, event_index = timing_pair
                    delta_copy_start.record(self._delta_copy_stream)
                local_delta.copy_(delta, non_blocking=True)
                if timing_pair is not None:
                    delta_copy_end.record(self._delta_copy_stream)
                fence_end.record(self._delta_copy_stream)
            current_stream.wait_event(fence_end)
            merge_enqueue_started_ns = time.monotonic_ns()
            self._merge_start.record(current_stream)
            y_flat.add_(local_delta.reshape_as(y_flat).to(dtype=y_flat.dtype))
            merge_end.record(current_stream)
            self._merge_end = merge_end
            self._op_slot_delta_in_use[slot_idx] = True
            self._delta_workspace_in_use = True
            merge_enqueue_finished_ns = time.monotonic_ns()
            _record_runtime_event(
                "transport_phase",
                phase="remote_compute_ready_wait",
                operation_id=descriptor.operation_id,
                layer_name=descriptor.layer_name,
                payload_bytes=int(delta.numel() * delta.element_size()),
                start_ns=wait_started_ns,
                end_ns=remote_ready_ns,
                duration_us=(remote_ready_ns - wait_started_ns) / 1000.0,
                timing_scope="remote_compute_through_worker_ready_ack",
                clock_domain="base_monotonic",
            )
            _record_runtime_event(
                "transport_phase",
                phase="delta_ipc_receive",
                operation_id=descriptor.operation_id,
                layer_name=descriptor.layer_name,
                payload_bytes=int(delta.numel() * delta.element_size()),
                start_ns=delta_receive_started_ns,
                end_ns=receive_finished_ns,
                duration_us=(receive_finished_ns - delta_receive_started_ns) / 1000.0,
                timing_scope="descriptor_and_ipc_mapping_not_payload_copy",
            )
            if timing_pair is not None:
                self._pending_delta_copy_timings.append(
                    {
                        "operation_id": descriptor.operation_id,
                        "layer_name": descriptor.layer_name,
                        "payload_bytes": int(
                            delta.numel() * delta.element_size()
                        ),
                        "start_event": delta_copy_start,
                        "end_event": delta_copy_end,
                        "event_index": event_index,
                    }
                )
            _record_runtime_event(
                "transport_phase",
                phase="merge",
                operation_id=descriptor.operation_id,
                layer_name=descriptor.layer_name,
                payload_bytes=int(delta.numel() * delta.element_size()),
                duration_us=(merge_enqueue_finished_ns - merge_enqueue_started_ns)
                / 1000.0,
                start_ns=merge_enqueue_started_ns,
                end_ns=merge_enqueue_finished_ns,
                device=str(y.device),
                timing_scope="host_enqueue_only_stream_ordered_merge",
                clock_domain="base_monotonic",
            )
            # Worker fences Δ reuse on the Signal ring; skip per-op host ACK
            # bytes when compact batch ACK is enabled (GPU Signal is enough).
            signal_only_delta_ack = (
                compact_control_enabled()
                and operation_batch_ack_enabled()
                and self._receive._signal_ring is not None
            )
            self._receive.acknowledge(
                output_descriptor,
                consumer_complete=True,
                completion_event=merge_end,
                host=not signal_only_delta_ack,
            )
            operation["active"] = False
            _record_runtime_event(
                "transport_phase",
                phase="operation_host_total",
                operation_id=descriptor.operation_id,
                layer_name=descriptor.layer_name,
                payload_bytes=int(delta.numel() * delta.element_size()),
                start_ns=operation["begin_host_started_ns"],
                end_ns=time.monotonic_ns(),
                duration_us=(time.monotonic_ns() - operation["begin_host_started_ns"])
                / 1000.0,
                begin_duration_us=(
                    operation["begin_host_finished_ns"]
                    - operation["begin_host_started_ns"]
                )
                / 1000.0,
                finish_duration_us=(time.monotonic_ns() - finish_host_started_ns)
                / 1000.0,
                timing_scope="base_host_operation_including_control_plane",
                op_slot=int(slot_idx),
            )
            _record_runtime_event(
                "transport_phase",
                phase="base_host_sync",
                operation_id=descriptor.operation_id,
                layer_name=descriptor.layer_name,
                duration_us=0.0,
                call_count=0,
                mechanism="cuda_stream_wait_event",
            )
            return y
        finally:
            with self._inflight_cond:
                self._op_slot_free[slot_idx] = True
                self._inflight_ops.pop(str(descriptor.operation_id), None)
                self._inflight_cond.notify_all()

    def close(self) -> None:
        if getattr(self, "_parent", None) is None:
            return
        # Bounded drain: stop new work, wait in-flight X/Δ copy + merge, then
        # flush telemetry before releasing IPC mappings / exporters / Events.
        try:
            if getattr(self, "_receive", None) is not None:
                self._receive.flush_acks()
        except Exception:
            pass
        try:
            self._delta_copy_stream.synchronize()
            self._drain_delta_copy_timings()
            if getattr(self, "_merge_end", None) is not None:
                self._merge_end.synchronize()
        except Exception:
            pass
        try:
            flush_runtime_event_buffer()
        except Exception:
            pass
        try:
            _send_json(self._parent, {"kind": "close"})
        except Exception:
            pass
        for endpoint in (
            getattr(self, "_send", None),
            getattr(self, "_receive", None),
        ):
            if endpoint is not None:
                endpoint.close()
        process = getattr(self, "_process", None)
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def _acquire_delta_copy_events(
        self,
    ) -> tuple[torch.cuda.Event, torch.cuda.Event, int] | None:
        """Acquire a timing Event pair, or None when the pool is exhausted.

        Telemetry must never raise or force a host sync when the pool is full.
        Synchronization fences use independent non-timing Events.
        """
        pool_size = len(self._delta_copy_event_pool)
        for offset in range(pool_size):
            index = (self._delta_copy_event_cursor + offset) % pool_size
            if not self._delta_copy_event_in_use[index]:
                self._delta_copy_event_in_use[index] = True
                self._delta_copy_event_cursor = (index + 1) % pool_size
                start_event, end_event = self._delta_copy_event_pool[index]
                return start_event, end_event, index
        return None

    def _drain_delta_copy_timings(self) -> None:
        remaining: list[dict[str, Any]] = []
        for item in self._pending_delta_copy_timings:
            end_event = item["end_event"]
            if not end_event.query():
                remaining.append(item)
                continue
            duration_us = float(
                item["start_event"].elapsed_time(end_event)
            ) * 1000.0
            _record_runtime_event(
                "transport_phase",
                phase=item.get("phase", "delta_p2p_copy"),
                operation_id=item["operation_id"],
                layer_name=item["layer_name"],
                payload_bytes=item["payload_bytes"],
                duration_us=duration_us,
                timing_scope=item.get(
                    "timing_scope", "explicit_lora_gpu_to_base_gpu_p2p_copy"
                ),
            )
            self._delta_copy_event_in_use[item["event_index"]] = False
        self._pending_delta_copy_timings = remaining


_CLIENTS: dict[
    tuple[
        tuple[tuple[int, str], ...],
        tuple[tuple[int, str], ...],
        int,
        int,
    ],
    TetrisLoRAWorkerClient,
] = {}
_NIXL_RUNTIME: Any | None = None


def _load_explicit_nixl_runtime() -> None:
    """Load the NIXL runtime for an explicit nixl/rdma transport opt-in.

    The process default stays IPC. ``TETRIS_LORA_RDMA_MODE=mock`` is refused.
    Constructor failures propagate; this function does not open the IPC client.
    """
    global _NIXL_RUNTIME
    requested = os.environ.get("TETRIS_LORA_RDMA_MODE")
    if requested is not None and requested.lower() not in {"nixl", "rdma"}:
        raise TransportError(
            "TETRIS_LORA_TRANSPORT=nixl/rdma refuses "
            f"TETRIS_LORA_RDMA_MODE={requested!r}; no MOCK/TCP fallback"
        )
    from .tetris_lora_rdma_runtime import RdmaMode, TetrisRdmaRuntime

    with _CLIENTS_LOCK:
        runtime = _NIXL_RUNTIME
        if runtime is None:
            runtime = TetrisRdmaRuntime(mode=RdmaMode.NIXL)
            _NIXL_RUNTIME = runtime
        status = runtime.capability.status
    raise TransportUnavailableError(
        "TETRIS_LORA_TRANSPORT=nixl/rdma loaded TetrisRdmaRuntime "
        f"status={status}; refusing IPC/MOCK/TCP fallback"
    )


def begin_tetris_lora(
    module_name: str,
    x: torch.Tensor,
    composition: tuple[tuple[tuple[int, float], ...], ...] | None,
    slot_paths: dict[int, str] | None,
    rank_tables: tuple[tuple[int, ...], ...] = (),
    output_slices: tuple[int, ...] = (),
    row_layout: tuple[int, ...] = (),
) -> tuple[TetrisLoRAWorkerClient, dict[str, Any]] | None:
    """Start remote LoRA work before the Base GEMM is submitted."""
    if not _enabled() or not _is_target_module(module_name):
        return None
    if composition is None or not composition or not any(composition):
        return None
    active_slots = tuple(
        sorted(
            {
                int(slot)
                for entries in composition
                for slot, _ in entries
            }
        )
    )
    if slot_paths is None or any(slot not in slot_paths for slot in active_slots):
        raise RuntimeError(
            "TetrisLoRA cannot resolve every active adapter slot to a path"
        )
    active_paths = {
        slot: str(Path(slot_paths[slot]).resolve()) for slot in active_slots
    }
    override_path = os.environ.get("TETRIS_LORA_ADAPTER_PATH")
    if override_path and len(active_paths) == 1 and str(
        Path(override_path).resolve()
    ) != next(iter(active_paths.values())):
        raise RuntimeError(
            "TETRIS_LORA_ADAPTER_PATH does not match the active LoRA slot path"
        )
    if not _is_target_module(module_name):
        return None
    physical_device = int(os.environ.get("TETRIS_LORA_LORA_GPU", "1"))
    base_device = int(os.environ.get("TETRIS_LORA_BASE_GPU", "0"))
    transport = os.environ.get("TETRIS_LORA_TRANSPORT", "ipc").lower()
    if transport in {"nixl", "rdma"}:
        _load_explicit_nixl_runtime()
    if transport != "ipc":
        if transport != "ucx":
            raise TransportError(f"unsupported TETRIS_LORA_TRANSPORT={transport}")
        raise TransportError(
            "UCX dataplane is IMPLEMENTATION-INCOMPLETE this round; "
            "NIXL is the wired backend; refusing IPC fallback"
        )
    path_key = tuple(sorted(active_paths.items()))
    # Session-cached content hash only (register_adapter_version). Never
    # re-read adapter_model.safetensors on the per-target begin hot path.
    version_key = tuple(
        sorted(
            (slot, register_adapter_version(path))
            for slot, path in active_paths.items()
        )
    )
    # D5: one resident worker per (lora_gpu, base_gpu). Adapter/version changes
    # replace that worker instead of growing a combinatorial client cache.
    key = (physical_device, base_device)
    adapter_id = ",".join(f"{slot}:{path}" for slot, path in path_key)
    adapter_id = f"{adapter_id}|v={version_key}"
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is not None and client.adapter_id != adapter_id:
            try:
                client.close()
            except Exception:
                pass
            client = None
            _CLIENTS.pop(key, None)
        if client is None:
            client = TetrisLoRAWorkerClient(
                active_paths,
                adapter_id=adapter_id,
                physical_device=physical_device,
                base_device=base_device,
            )
            _CLIENTS[key] = client
    context: dict[str, Any] = {}
    try:
        from vllm_omni.diffusion.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available():
            forward_context = get_forward_context()
            request_ids = tuple(forward_context.tetris_request_ids)
            context = {
                "request_id": request_ids[0] if request_ids else "unknown-request",
                "request_ids": request_ids,
                "batch_id": forward_context.tetris_batch_id or "unknown-batch",
                "denoise_step": (
                    -1
                    if forward_context.denoise_step_idx is None
                    else int(forward_context.denoise_step_idx)
                ),
            }
    except (ImportError, AttributeError):
        pass
    primary_scale = next(
        (float(branch_scale) for entries in composition for _, branch_scale in entries),
        1.0,
    )
    _record_runtime_event(
        "begin_delta",
        module_name=module_name,
        normalized_target=_normalize_target_module(module_name),
        adapter_paths=active_paths,
        active_slots=list(active_slots),
        scale=primary_scale,
        shape=list(x.shape),
    )
    return client, client.begin_delta(
        x,
        primary_scale,
        module_name=module_name,
        context=context,
        composition=composition,
        rank_tables=rank_tables,
        output_slices=output_slices,
        row_layout=row_layout,
    )


def finish_tetris_lora(
    operation: tuple[TetrisLoRAWorkerClient, dict[str, Any]], y: torch.Tensor
) -> torch.Tensor:
    """Wait for delta and merge after the Base GEMM is enqueued."""
    client, handle = operation
    return client.finish_delta(handle, y)


def cancel_tetris_lora(
    operation: tuple[TetrisLoRAWorkerClient, dict[str, Any]] | None,
) -> None:
    """Release an in-flight op-slot after Base failure without merging Δ.

    Must be called exactly when ``begin_tetris_lora`` succeeded but
    ``finish_tetris_lora`` will not run. Marks the handle inactive so a
    later finish cannot reuse a broken operation.
    """
    if operation is None:
        return
    client, handle = operation
    if not handle.get("active") and handle.get("cancelled"):
        return
    handle["active"] = False
    handle["cancelled"] = True
    slot_idx = handle.get("op_slot")
    descriptor = handle.get("descriptor")
    op_id = getattr(descriptor, "operation_id", None) if descriptor is not None else None
    cond = getattr(client, "_inflight_cond", None)
    if cond is not None:
        with cond:
            if slot_idx is not None and hasattr(client, "_op_slot_free"):
                free = client._op_slot_free
                if 0 <= int(slot_idx) < len(free):
                    free[int(slot_idx)] = True
            if op_id is not None and hasattr(client, "_inflight_ops"):
                client._inflight_ops.pop(str(op_id), None)
            cond.notify_all()
        return
    # Legacy single-lock clients (unit doubles).
    lock = getattr(client, "_lock", None)
    if lock is not None and lock.locked():
        lock.release()


def close_tetris_lora_clients() -> None:
    global _NIXL_RUNTIME
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
        runtime = _NIXL_RUNTIME
        _NIXL_RUNTIME = None
    for client in clients:
        client.close()
    if runtime is not None:
        runtime.close()
    try:
        flush_runtime_event_buffer()
    except Exception:
        pass


atexit.register(close_tetris_lora_clients)


__all__ = [
    "TetrisLoRAWorkerClient",
    "begin_tetris_lora",
    "cancel_tetris_lora",
    "close_tetris_lora_clients",
    "finish_tetris_lora",
    "invalidate_adapter_version",
    "register_adapter_version",
]
