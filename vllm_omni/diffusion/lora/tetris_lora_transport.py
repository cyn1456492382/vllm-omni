"""Transport contracts for process-separated Base and LoRA execution.

Steady-state send, receive, and ACK use a fixed binary frame once a plan is
registered. Init handshake and the signal-ring offer stay JSON. Plan updates
use msgpack.
"""

from __future__ import annotations

import base64
import json
import os
import struct
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor


class TransportError(RuntimeError):
    """A transport operation failed without a safe local fallback."""


class TransportUnavailableError(TransportError):
    """The requested data plane is not available in this process."""


class BufferState(str, Enum):
    """Lifecycle states for one bounded transport slot."""

    FREE = "free"
    PRODUCER_READY = "producer_ready"
    TRANSFER_IN_FLIGHT = "transfer_in_flight"
    TARGET_VISIBLE = "target_visible"
    CONSUMER_IN_FLIGHT = "consumer_in_flight"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class TensorDescriptor:
    """Validated control-plane metadata for one GPU tensor transfer."""

    session_epoch: str
    operation_id: str
    request_id: str
    batch_id: str
    denoise_step: int
    layer_name: str
    invocation_id: int
    slot_id: int
    slot_generation: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    layout: str
    payload_bytes: int
    buffer_token: str
    adapter_id: str
    adapter_version: str
    rank: int
    scale: float
    branch_index: int
    output_offset: int
    completion_sequence: int
    protocol_version: int = 1
    request_ids: tuple[str, ...] = ()
    row_spans: tuple[tuple[int, int, int], ...] = ()
    buffer_capacity: int = 0
    offset: int = 0
    alpha_folded: bool = False
    request_composition: tuple[tuple[tuple[int, float], ...], ...] = ()
    request_ranks: tuple[tuple[int, ...], ...] = ()
    slice_rank_tables: tuple[tuple[int, ...], ...] = ()
    row_layout: tuple[int, ...] = ()
    output_slices: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.protocol_version != 1:
            raise ValueError(f"unsupported protocol version {self.protocol_version}")
        if not all(
            (
                self.session_epoch,
                self.operation_id,
                self.request_id,
                self.batch_id,
                self.layer_name,
                self.buffer_token,
                self.adapter_id,
                self.adapter_version,
            )
        ):
            raise ValueError("transport identifiers must be non-empty")
        if len(self.shape) != len(self.stride):
            raise ValueError("shape and stride ranks differ")
        if any(int(dim) < 0 for dim in self.shape):
            raise ValueError("tensor shape cannot contain negative dimensions")
        if any(int(stride) < 0 for stride in self.stride):
            raise ValueError("negative tensor strides are unsupported")
        if self.rank < 1 or self.slot_id < 0 or self.slot_generation < 0:
            raise ValueError("invalid rank or slot metadata")
        if self.payload_bytes < 0 or self.completion_sequence < 0:
            raise ValueError("invalid transfer size or completion sequence")
        if self.offset < 0 or (
            self.buffer_capacity > 0
            and self.buffer_capacity < self.offset + self.payload_bytes
        ):
            raise ValueError("payload exceeds declared buffer capacity")
        if self.request_composition and self.row_layout:
            if len(self.request_composition) != len(self.row_layout):
                raise ValueError("request composition and row layout differ")
        if self.request_ranks and self.request_composition:
            if len(self.request_ranks) != len(self.request_composition):
                raise ValueError("request ranks and composition differ")
            for entries, ranks in zip(self.request_composition, self.request_ranks):
                if len(entries) != len(ranks):
                    raise ValueError("request rank counts do not match branches")
        if any(int(rows) < 0 for rows in self.row_layout):
            raise ValueError("row layout cannot contain negative counts")
        if self.output_offset < 0:
            raise ValueError("output offset cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for embedding in an outer control message (no nested JSON)."""
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_obj(cls, payload: str | dict[str, Any]) -> "TensorDescriptor":
        """Parse a descriptor from a dict or legacy JSON string."""
        raw = json.loads(payload) if isinstance(payload, str) else dict(payload)
        raw["shape"] = tuple(int(value) for value in raw["shape"])
        raw["stride"] = tuple(int(value) for value in raw["stride"])
        raw["request_ids"] = tuple(str(value) for value in raw.get("request_ids", ()))
        raw["row_spans"] = tuple(
            tuple(int(value) for value in span)
            for span in raw.get("row_spans", ())
        )
        raw["request_composition"] = tuple(
            tuple((int(slot), float(scale)) for slot, scale in entries)
            for entries in raw.get("request_composition", ())
        )
        raw["request_ranks"] = tuple(
            tuple(int(rank) for rank in ranks)
            for ranks in raw.get("request_ranks", ())
        )
        raw["slice_rank_tables"] = tuple(
            tuple(int(rank) for rank in ranks)
            for ranks in raw.get("slice_rank_tables", ())
        )
        raw["row_layout"] = tuple(int(rows) for rows in raw.get("row_layout", ()))
        raw["output_slices"] = tuple(
            int(size) for size in raw.get("output_slices", ())
        )
        descriptor = cls(**raw)
        descriptor.validate()
        return descriptor

    @classmethod
    def from_json(cls, payload: str) -> "TensorDescriptor":
        return cls.from_obj(payload)


@dataclass(frozen=True)
class TransferResult:
    """Observed transfer path and byte accounting."""

    backend: str
    payload_bytes: int
    host_staged: bool
    path: str
    elapsed_ms: float


@dataclass(frozen=True)
class _IpcTensorPayload:
    """JSON-safe reconstruction arguments produced by PyTorch's IPC reducer."""

    size: tuple[int, ...]
    stride: tuple[int, ...]
    tensor_offset: int
    dtype: str
    storage_device: int
    target_device: int
    storage_size_bytes: int
    storage_offset_bytes: int
    storage_handle: str
    ref_counter_handle: str
    ref_counter_offset: int
    event_handle: str | None
    event_sync_required: bool
    producer_ready_event_handle: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_obj(cls, payload: str | dict[str, Any]) -> "_IpcTensorPayload":
        raw = json.loads(payload) if isinstance(payload, str) else dict(payload)
        raw["size"] = tuple(int(value) for value in raw["size"])
        raw["stride"] = tuple(int(value) for value in raw["stride"])
        return cls(**raw)

    @classmethod
    def from_json(cls, payload: str) -> "_IpcTensorPayload":
        return cls.from_obj(payload)


def _b64(value: bytes | None) -> str | None:
    if value is None:
        return None
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str | None) -> bytes | None:
    if value is None:
        return None
    return base64.b64decode(value.encode("ascii"))


def export_cuda_tensor(tensor: torch.Tensor, target_device: int) -> _IpcTensorPayload:
    """Export a CUDA tensor using PyTorch's CUDA IPC reducer."""
    if not tensor.is_cuda:
        raise ValueError("CUDA IPC requires a CUDA tensor")
    reducer, args = reduce_tensor(tensor.detach())
    if reducer is not rebuild_cuda_tensor or len(args) != 15:
        raise TransportError("PyTorch did not return its CUDA IPC tensor reducer")
    (
        tensor_type,
        tensor_size,
        tensor_stride,
        tensor_offset,
        storage_type,
        dtype,
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        requires_grad,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = args
    del tensor_type, storage_type, requires_grad
    if not isinstance(storage_device, int):
        storage_device = int(torch.device(storage_device).index or 0)
    return _IpcTensorPayload(
        size=tuple(int(value) for value in tensor_size),
        stride=tuple(int(value) for value in tensor_stride),
        tensor_offset=int(tensor_offset),
        dtype=str(dtype).split(".")[-1],
        storage_device=int(storage_device),
        target_device=int(target_device),
        storage_size_bytes=int(storage_size_bytes),
        storage_offset_bytes=int(storage_offset_bytes),
        storage_handle=_b64(storage_handle) or "",
        ref_counter_handle=_b64(ref_counter_handle) or "",
        ref_counter_offset=int(ref_counter_offset),
        event_handle=_b64(event_handle),
        event_sync_required=bool(event_sync_required),
        producer_ready_event_handle=None,
    )


def import_cuda_tensor(payload: _IpcTensorPayload) -> tuple[torch.Tensor, torch.cuda.Event | None]:
    """Map an IPC allocation into the consumer process and return its event."""
    if not torch.cuda.is_available():
        raise TransportError("CUDA is unavailable in the consumer process")
    dtype = getattr(torch, payload.dtype, None)
    if dtype is None:
        raise ValueError(f"unsupported IPC dtype {payload.dtype}")
    storage = rebuild_cuda_tensor(
        torch.Tensor,
        payload.size,
        payload.stride,
        payload.tensor_offset,
        torch.storage.TypedStorage,
        dtype,
        int(payload.target_device),
        _unb64(payload.storage_handle),
        payload.storage_size_bytes,
        payload.storage_offset_bytes,
        False,
        _unb64(payload.ref_counter_handle),
        payload.ref_counter_offset,
        _unb64(payload.event_handle),
        payload.event_sync_required,
    )
    event = None
    if payload.producer_ready_event_handle is not None:
        event = torch.cuda.Event.from_ipc_handle(
            payload.target_device,
            _unb64(payload.producer_ready_event_handle),
        )
    return storage, event


def _validate_payload(descriptor: TensorDescriptor, tensor: torch.Tensor) -> None:
    descriptor.validate()
    if tuple(int(value) for value in tensor.shape) != descriptor.shape:
        raise ValueError("tensor shape does not match descriptor")
    if tuple(int(value) for value in tensor.stride()) != descriptor.stride:
        raise ValueError("tensor stride does not match descriptor")
    if str(tensor.dtype) != descriptor.dtype:
        raise ValueError("tensor dtype does not match descriptor")
    payload_bytes = int(tensor.numel() * tensor.element_size())
    if payload_bytes != descriptor.payload_bytes:
        raise ValueError("tensor byte count does not match descriptor")


class SlotLease:
    """Generation-aware slot ownership with quarantine until DMA drains."""

    def __init__(self, slot_id: int) -> None:
        self.slot_id = int(slot_id)
        self.generation = 0
        self.state = BufferState.FREE

    def begin(self) -> int:
        if self.state not in {BufferState.FREE, BufferState.QUARANTINED}:
            raise TransportError(f"slot {self.slot_id} is not reusable: {self.state}")
        if self.state == BufferState.QUARANTINED:
            raise TransportError(f"slot {self.slot_id} is quarantined until drain")
        self.generation += 1
        self.state = BufferState.PRODUCER_READY
        return self.generation

    def mark_in_flight(self) -> None:
        if self.state != BufferState.PRODUCER_READY:
            raise TransportError("slot is not producer-ready")
        self.state = BufferState.TRANSFER_IN_FLIGHT

    def mark_target_visible(self) -> None:
        if self.state != BufferState.TRANSFER_IN_FLIGHT:
            raise TransportError("slot transfer is not in flight")
        self.state = BufferState.TARGET_VISIBLE

    def mark_consumer_in_flight(self) -> None:
        if self.state != BufferState.TARGET_VISIBLE:
            raise TransportError("target buffer is not visible")
        self.state = BufferState.CONSUMER_IN_FLIGHT

    def complete(self) -> None:
        if self.state != BufferState.CONSUMER_IN_FLIGHT:
            raise TransportError("consumer has not completed")
        self.state = BufferState.FREE

    def quarantine(self) -> None:
        if self.state != BufferState.FREE:
            self.state = BufferState.QUARANTINED


class IpcSignalRing:
    """Session-scoped IPC event ring.

    Mirrors custom_all_reduce Signal/RankSignals registration: export handles
    once at session start, then steady-state only passes event_index +
    generation (no per-op cudaIpc handle bytes on the control plane).

    Generation fencing uses a POSIX shared-memory ``uint64[capacity]`` table
    (host-visible across Base/Worker). The producer writes generation before
    recording the CUDA IPC event; the consumer waits the event then checks the
    host table so recycled ring slots cannot be observed stale.
    """

    def __init__(self, device: int, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("signal ring capacity must be positive")
        self.device = int(device)
        self.capacity = int(capacity)
        previous = torch.cuda.current_device()
        torch.cuda.set_device(self.device)
        try:
            self.local_events = [
                torch.cuda.Event(enable_timing=False, interprocess=True)
                for _ in range(self.capacity)
            ]
        finally:
            torch.cuda.set_device(previous)
        self.peer_events: list[torch.cuda.Event | None] = [None] * self.capacity
        # Local generation table (exported by name); peer table imported later.
        self._gen_shm = shared_memory.SharedMemory(
            create=True, size=int(self.capacity) * 8
        )
        # Avoid double-tracking when the peer also attaches by name; we unlink
        # explicitly in close_generation_tables.
        try:
            from multiprocessing import resource_tracker as _rt

            _rt.unregister(self._gen_shm._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass
        self._gen_host = np.ndarray(
            (self.capacity,), dtype=np.uint64, buffer=self._gen_shm.buf
        )
        self._gen_host[:] = np.uint64(0)
        self._peer_gen_shm: shared_memory.SharedMemory | None = None
        self._peer_gen_host: np.ndarray | None = None
        self._cursor = 0
        self._lock = threading.Lock()

    @property
    def peer_generations(self) -> np.ndarray | None:
        """Peer host generation table, or None before import."""
        return self._peer_gen_host

    def export_handles(self) -> list[str]:
        previous = torch.cuda.current_device()
        torch.cuda.set_device(self.device)
        try:
            return [_b64(event.ipc_handle()) or "" for event in self.local_events]
        finally:
            torch.cuda.set_device(previous)

    def export_generation_shm_name(self) -> str:
        """Return the POSIX shm name for this ring's generation table."""
        return str(self._gen_shm.name)

    def import_peer_handles(self, handles: list[str]) -> None:
        if len(handles) != self.capacity:
            raise TransportError(
                f"signal ring size mismatch: got {len(handles)} want {self.capacity}"
            )
        previous = torch.cuda.current_device()
        torch.cuda.set_device(self.device)
        try:
            imported: list[torch.cuda.Event | None] = []
            for handle in handles:
                raw = _unb64(handle)
                if raw is None:
                    raise TransportError("signal ring handle is empty")
                imported.append(
                    torch.cuda.Event.from_ipc_handle(self.device, raw)
                )
            self.peer_events = imported
        finally:
            torch.cuda.set_device(previous)

    def import_peer_generations(self, shm_name: str) -> None:
        """Attach the peer generation table by POSIX shared-memory name."""
        if not shm_name:
            raise TransportError("peer generation shm name is empty")
        if self._peer_gen_shm is not None:
            raise TransportError("peer generation table already imported")
        peer = shared_memory.SharedMemory(name=str(shm_name))
        try:
            from multiprocessing import resource_tracker as _rt

            _rt.unregister(peer._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:
            pass
        if int(peer.size) < int(self.capacity) * 8:
            peer.close()
            raise TransportError(
                "signal ring generation shm too small: "
                f"got {peer.size} want {int(self.capacity) * 8}"
            )
        self._peer_gen_shm = peer
        self._peer_gen_host = np.ndarray(
            (self.capacity,), dtype=np.uint64, buffer=peer.buf
        )

    def close_generation_tables(self) -> None:
        """Release local/peer generation shm (session teardown)."""
        if self._peer_gen_shm is not None:
            try:
                self._peer_gen_shm.close()
            except Exception:
                pass
            self._peer_gen_shm = None
            self._peer_gen_host = None
        try:
            name = self._gen_shm.name
            self._gen_shm.close()
            shared_memory.SharedMemory(name=name).unlink()
        except Exception:
            try:
                self._gen_shm.close()
            except Exception:
                pass

    def next_index(self) -> int:
        with self._lock:
            index = self._cursor % self.capacity
            self._cursor += 1
            return index

    def record_local(
        self,
        index: int,
        stream: torch.cuda.Stream | None = None,
        *,
        generation: int | None = None,
    ) -> None:
        """Record the local doorbell; optionally publish a generation fence first."""
        stream = stream or torch.cuda.current_stream(self.device)
        if generation is not None:
            # Host publish happens-before the CUDA IPC event record.
            self._gen_host[int(index)] = np.uint64(int(generation))
        self.local_events[int(index)].record(stream)

    def wait_peer(self, index: int, stream: torch.cuda.Stream | None = None) -> None:
        event = self.peer_events[int(index)]
        if event is None:
            raise TransportError(f"peer signal {index} is not imported")
        (stream or torch.cuda.current_stream(self.device)).wait_event(event)

    @torch.compiler.disable
    def wait_peer_with_generation(
        self,
        index: int,
        expected_generation: int,
        stream: torch.cuda.Stream | None = None,
        *,
        timeout_s: float = 30.0,
    ) -> None:
        """Wait until the peer publishes ``expected_generation``, then GPU-fence.

        Host generation is polled first so a sticky/recycled CUDA IPC event
        cannot return success with a stale (or zero) generation. After the
        generation matches, ``wait_peer`` orders the consumer stream behind the
        producer's recorded doorbell.
        """
        if self._peer_gen_host is None:
            raise TransportError("peer generation table is not imported")
        stream = stream or torch.cuda.current_stream(self.device)
        want = int(expected_generation)
        if want <= 0:
            raise TransportError(
                f"signal ring generation must be positive, got {want}"
            )
        deadline = time.monotonic() + float(timeout_s)
        poll_s = _control_poll_s()
        idx = int(index)
        while True:
            got = int(self._peer_gen_host[idx])
            if got == want:
                # Host generation publish happens-before the producer
                # event.record; GPU wait orders the consumer stream only.
                self.wait_peer(idx, stream)
                return
            if time.monotonic() >= deadline:
                raise TransportError(
                    "signal ring generation wait timed out: "
                    f"index={idx} got={int(self._peer_gen_host[idx])} want={want}"
                )
            time.sleep(poll_s)


_MAILBOX_DTYPE = np.dtype(
    [
        ("seq", np.uint64),
        ("plan_id", np.uint64),
        ("registration_id", np.uint64),
        ("slot_id", np.uint64),
        ("slot_generation", np.uint64),
        ("op_id", np.uint64),
        ("flags", np.uint64),
        ("denoise_step", np.int32),
        ("_pad", np.int32),
    ]
)
MAILBOX_FLAG_HAS_COMPUTE = 1 << 0
MAILBOX_FLAG_DELTA = 1 << 1


def mailbox_control_enabled() -> bool:
    """Steady-state shared mailbox (no per-op host pipe). Default on with compact."""
    compact = os.environ.get("TETRIS_LORA_COMPACT_CONTROL", "1").lower()
    if compact in {"0", "false", "no", "off"}:
        return False
    binary = os.environ.get("TETRIS_LORA_BINARY_WIRE", "1").lower()
    if binary in {"0", "false", "no", "off"}:
        return False
    value = os.environ.get("TETRIS_LORA_MAILBOX_CONTROL", "1").lower()
    return value not in {"0", "false", "no", "off"}


class ControlMailbox:
    """POSIX-shm control mailbox for steady X/Δ publishes (no host pipe).

    Layout: ``x_entries[capacity]`` then ``delta_entries[capacity]``. The
    publisher writes payload fields first and ``seq`` last as the publish
    marker; the peer waits on the signal ring generation equal to ``seq``.
    """

    def __init__(
        self,
        capacity: int,
        *,
        create: bool = True,
        shm_name: str | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("mailbox capacity must be positive")
        self.capacity = int(capacity)
        nbytes = int(2 * self.capacity * _MAILBOX_DTYPE.itemsize)
        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
            try:
                from multiprocessing import resource_tracker as _rt

                _rt.unregister(self._shm._name, "shared_memory")  # type: ignore[attr-defined]
            except Exception:
                pass
            self._owner = True
        else:
            if not shm_name:
                raise TransportError("mailbox import requires shm_name")
            self._shm = shared_memory.SharedMemory(name=str(shm_name))
            try:
                from multiprocessing import resource_tracker as _rt

                _rt.unregister(self._shm._name, "shared_memory")  # type: ignore[attr-defined]
            except Exception:
                pass
            if int(self._shm.size) < nbytes:
                self._shm.close()
                raise TransportError(
                    f"mailbox shm too small: got {self._shm.size} want {nbytes}"
                )
            self._owner = False
        self.x_entries = np.ndarray(
            (self.capacity,), dtype=_MAILBOX_DTYPE, buffer=self._shm.buf, offset=0
        )
        self.delta_entries = np.ndarray(
            (self.capacity,),
            dtype=_MAILBOX_DTYPE,
            buffer=self._shm.buf,
            offset=int(self.capacity * _MAILBOX_DTYPE.itemsize),
        )
        if create:
            self.x_entries[:] = np.zeros(self.capacity, dtype=_MAILBOX_DTYPE)
            self.delta_entries[:] = np.zeros(self.capacity, dtype=_MAILBOX_DTYPE)

    def export_shm_name(self) -> str:
        return str(self._shm.name)

    def close(self) -> None:
        name = self._shm.name
        try:
            self._shm.close()
        except Exception:
            pass
        if self._owner:
            try:
                shared_memory.SharedMemory(name=name).unlink()
            except Exception:
                pass

    def _publish(
        self,
        entries: np.ndarray,
        *,
        seq: int,
        plan_id: int,
        registration_id: int,
        slot_id: int,
        slot_generation: int,
        op_id: int,
        flags: int,
        denoise_step: int,
    ) -> int:
        idx = ring_index_for_sequence(seq, self.capacity)
        entries[idx]["plan_id"] = np.uint64(plan_id)
        entries[idx]["registration_id"] = np.uint64(registration_id)
        entries[idx]["slot_id"] = np.uint64(slot_id)
        entries[idx]["slot_generation"] = np.uint64(slot_generation)
        entries[idx]["op_id"] = np.uint64(op_id)
        entries[idx]["flags"] = np.uint64(flags)
        entries[idx]["denoise_step"] = np.int32(denoise_step)
        entries[idx]["_pad"] = np.int32(0)
        # Publish marker last so peers observing seq see consistent fields.
        entries[idx]["seq"] = np.uint64(seq)
        return idx

    def publish_x(self, **kwargs: Any) -> int:
        return self._publish(self.x_entries, **kwargs)

    def publish_delta(self, **kwargs: Any) -> int:
        return self._publish(self.delta_entries, **kwargs)

    def read_x(self, seq: int) -> np.void:
        idx = ring_index_for_sequence(seq, self.capacity)
        row = self.x_entries[idx]
        if int(row["seq"]) != int(seq):
            raise TransportError(
                f"mailbox X seq mismatch: index={idx} got={int(row['seq'])} want={seq}"
            )
        return row

    def read_delta(self, seq: int) -> np.void:
        idx = ring_index_for_sequence(seq, self.capacity)
        row = self.delta_entries[idx]
        if int(row["seq"]) != int(seq):
            raise TransportError(
                f"mailbox Δ seq mismatch: index={idx} got={int(row['seq'])} want={seq}"
            )
        return row


def _control_poll_s() -> float:
    """Host poll slice for control demux (default 10us).

    P0 all-targets spent seconds in 10ms poll granularity across 342 targets.
    """
    return max(
        0.000005,
        float(os.environ.get("TETRIS_LORA_CONTROL_POLL_S", "0.00001")),
    )


def compact_control_enabled() -> bool:
    """Steady-state compact control plane (default on)."""
    value = os.environ.get("TETRIS_LORA_COMPACT_CONTROL", "1").lower()
    return value not in {"0", "false", "no", "off"}


def operation_batch_ack_enabled() -> bool:
    """Batch host ACK bytes; GPU Signal ring remains the steady-state fence."""
    if not compact_control_enabled():
        return False
    value = os.environ.get("TETRIS_LORA_OPERATION_BATCH_ACK", "1").lower()
    return value not in {"0", "false", "no", "off"}


def operation_ack_batch_size() -> int:
    return max(1, int(os.environ.get("TETRIS_LORA_ACK_BATCH_SIZE", "32")))


def ring_index_for_sequence(completion_sequence: int, capacity: int) -> int:
    """Map a descriptor completion sequence onto a session signal-ring slot."""
    if capacity < 1:
        raise ValueError("signal ring capacity must be positive")
    return int(completion_sequence) % int(capacity)


_U64_MAX = (1 << 64) - 1
_I32_MIN = -2147483648
_I32_MAX = 2147483647
_WIRE_MAGIC = b"TLC2"
_WIRE_VERSION = 2
# magic, version, msg_type, flags, len, denoise_step, session, op_id,
# plan_id, slot, generation, completion_seq, event_index, registration_id.
_WIRE_HEADER = struct.Struct("<4sBBHIiQQIIQQII")
_ACK_RECORD = struct.Struct("<QQIII")
FLAG_COMPACT = 1 << 0
FLAG_CONSUMER_COMPLETE = 1 << 1
FLAG_DELTA = 1 << 2
FLAG_HAS_COMPUTE = 1 << 3
FLAG_HAS_SOURCE_ACK = 1 << 4
_PER_OP_PLAN_FIELDS = (
    "operation_id",
    "invocation_id",
    "slot_generation",
    "completion_sequence",
    "denoise_step",
    # Per-generate request identity must not allocate a new plan_id; otherwise
    # every measured generate re-hosts all PLAN frames and mailbox-only never
    # engages after the cold first registration of each layer.
    "request_id",
    "batch_id",
    "request_ids",
)
_DELTA_SUFFIX = ":delta"
_WIRE_JSON_LOCK = threading.Lock()
_WIRE_JSON_COUNTS = {"dumps": 0, "loads": 0}


class WireMsgType(IntEnum):
    """Binary control message types carried after the magic header."""

    PLAN = 1
    TENSOR = 2
    TENSOR_REF = 3
    ACK = 4
    ACK_BATCH = 5
    CLOSE = 6
    # Cache-hit steady publish: fixed header only (no descriptor/handle body).
    DOORBELL = 7


class WireHeader(NamedTuple):
    """Fixed little-endian prefix of one steady-state control frame."""

    version: int
    msg_type: int
    flags: int
    length: int
    denoise_step: int
    session: int
    op_id: int
    plan_id: int
    slot: int
    generation: int
    completion_seq: int
    event_index: int
    registration_id: int


def binary_wire_enabled() -> bool:
    """Return whether steady-state control uses the binary frame.

    Default is on. Set ``TETRIS_LORA_BINARY_WIRE=0`` to keep JSON frames.
    """
    value = os.environ.get("TETRIS_LORA_BINARY_WIRE", "1").lower()
    return value not in {"0", "false", "no", "off"}


def wire_json_probe_enabled() -> bool:
    """Return whether wire-path ``json.dumps`` / ``json.loads`` are counted."""
    value = os.environ.get("TETRIS_LORA_COUNT_WIRE_JSON", "").lower()
    return value in {"1", "true", "yes", "on"}


def note_wire_json(kind: str) -> None:
    """Count one wire-path JSON call when the probe is enabled.

    Args:
        kind: ``dumps`` or ``loads``.

    Raises:
        ValueError: If ``kind`` is not a wire JSON counter.
    """
    if kind not in {"dumps", "loads"}:
        raise ValueError("wire JSON probe kind must be dumps or loads")
    if not wire_json_probe_enabled():
        return
    with _WIRE_JSON_LOCK:
        _WIRE_JSON_COUNTS[kind] += 1


def wire_json_counts() -> dict[str, int]:
    """Return dumps/loads counted while ``TETRIS_LORA_COUNT_WIRE_JSON=1``."""
    with _WIRE_JSON_LOCK:
        return {
            "dumps": int(_WIRE_JSON_COUNTS["dumps"]),
            "loads": int(_WIRE_JSON_COUNTS["loads"]),
        }


def reset_wire_json_counts() -> None:
    """Zero the wire JSON probe counters."""
    with _WIRE_JSON_LOCK:
        _WIRE_JSON_COUNTS["dumps"] = 0
        _WIRE_JSON_COUNTS["loads"] = 0


def wire_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize one JSON wire message and count it when the probe is on."""
    note_wire_json("dumps")
    return json.dumps(obj, **kwargs)


def wire_json_loads(payload: str | bytes | bytearray) -> Any:
    """Parse one JSON wire message and count it when the probe is on."""
    note_wire_json("loads")
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def _fit_u64(value: int, name: str) -> int:
    value = int(value)
    if value < 0 or value > _U64_MAX:
        raise TransportError(f"{name} is outside the u64 range")
    return value


def _fit_u32(value: int, name: str) -> int:
    value = int(value)
    if value < 0 or value > 0xFFFFFFFF:
        raise TransportError(f"{name} is outside the u32 range")
    return value


def _fit_i32(value: int, name: str) -> int:
    value = int(value)
    if value < _I32_MIN or value > _I32_MAX:
        raise TransportError(f"{name} is outside the int32 range")
    return value


def parse_operation_id(operation_id: str) -> tuple[int, bool]:
    """Parse a decimal u64 operation id.

    The ``:delta`` suffix is a flag, not part of the id. Non-decimal ids fail
    closed so a string hash cannot become an operation id.

    Args:
        operation_id: Decimal id, optionally suffixed with ``:delta``.

    Returns:
        The u64 id and whether the delta flag is set.

    Raises:
        TransportError: If the id is not a canonical decimal u64.
    """
    text = str(operation_id)
    delta = text.endswith(_DELTA_SUFFIX)
    if delta:
        text = text[: -len(_DELTA_SUFFIX)]
    if not text.isdigit():
        raise TransportError(
            "operation id must be a monotonic decimal u64, not a string hash"
        )
    if len(text) > 1 and text.startswith("0"):
        raise TransportError("operation id must not contain leading zeros")
    value = int(text)
    if value < 1 or value > _U64_MAX:
        raise TransportError("operation id is outside the u64 range")
    return value, delta


def _operation_text(op_id: int, flags: int) -> str:
    text = str(int(op_id))
    if flags & FLAG_DELTA:
        return f"{text}{_DELTA_SUFFIX}"
    return text


class OperationIdAllocator:
    """Monotonic u64 operation ids for one Base session.

    Overflow bumps ``epoch`` and restarts at 1. If the epoch itself cannot
    advance, allocation fails closed. Ids are never derived by hashing strings.
    """

    def __init__(self, *, start: int = 1, epoch: int = 1) -> None:
        start = int(start)
        epoch = int(epoch)
        if start < 1 or epoch < 1:
            raise ValueError("operation ids and epochs start at 1")
        if start > _U64_MAX + 1 or epoch > _U64_MAX:
            raise ValueError("operation id start is outside the u64 range")
        self._next = start
        self.epoch = epoch
        self._lock = threading.Lock()

    def next_id(self) -> int:
        """Return the next operation id.

        Returns:
            A positive u64 id.

        Raises:
            TransportError: If the epoch cannot bump after ``u64`` overflow.
        """
        with self._lock:
            if self._next > _U64_MAX:
                if self.epoch >= _U64_MAX:
                    raise TransportError("operation id epoch exhausted")
                self.epoch += 1
                self._next = 1
            value = self._next
            self._next += 1
            return value


def _freeze_plan(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = tuple(
            (str(key), _freeze_plan(value[key]))
            for key in sorted(value, key=str)
        )
        return ("dict", items)
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_freeze_plan(item) for item in value))
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TransportError(
        f"plan value is not msgpack-safe: {type(value).__name__}"
    )


def _thaw_plan(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "dict":
        return {key: _thaw_plan(item) for key, item in value[1]}
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "seq":
        return [_thaw_plan(item) for item in value[1]]
    return value


def _msgpack_module() -> Any:
    try:
        import msgpack
        return msgpack
    except ImportError:
        pass
    # Workspace-vendored wheel (container venv may lack msgpack).
    # tetris_lora_transport.py → .../vllm-omni/vllm_omni/diffusion/lora → parents[4]=repo root
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root
        / ".tetris_lora_results/acceptance_20261112/closure_20261112/vendor",
        Path(
            "/workspace/.tetris_lora_results/acceptance_20261112/closure_20261112/vendor"
        ),
        Path(
            "/mnt/nvme1/cyn/vllm/.tetris_lora_results/acceptance_20261112/closure_20261112/vendor"
        ),
    ]
    import sys

    for path in candidates:
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
        try:
            import msgpack

            return msgpack
        except ImportError:
            continue
    raise TransportError(
        "plan updates require the msgpack package; pickle is not used"
    )


def _pack_msgpack(payload: Mapping[str, Any]) -> bytes:
    packed = _msgpack_module().packb(dict(payload), use_bin_type=True)
    if not isinstance(packed, bytes):
        raise TransportError("msgpack did not return bytes")
    return packed


def _unpack_msgpack(payload: bytes) -> dict[str, Any]:
    try:
        unpacked = _msgpack_module().unpackb(payload, raw=False)
    except Exception as exc:
        raise TransportError("plan payload is not msgpack") from exc
    if not isinstance(unpacked, dict):
        raise TransportError("msgpack control payload must be a map")
    return unpacked


class SessionPlanRegistry:
    """Session-local map from a canonical plan body to a numeric plan id."""

    def __init__(self) -> None:
        self._by_body: dict[Any, int] = {}
        self._by_id: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self.wire_session = 1
        self._lock = threading.Lock()

    def register_plan(
        self,
        plan: Mapping[str, Any],
        plan_id: int | None = None,
    ) -> int:
        """Register ``plan`` and return its id.

        Identical canonical bodies reuse the existing id. An explicit
        ``plan_id`` is the peer-assigned id from a plan frame.

        Args:
            plan: Msgpack-safe plan body.
            plan_id: Peer id, or None to allocate the next local id.

        Returns:
            The stable plan id.

        Raises:
            TransportError: If the id space is exhausted or the body/id pair
                disagrees with an existing registration.
        """
        frozen = _freeze_plan(dict(plan))
        with self._lock:
            existing = self._by_body.get(frozen)
            if existing is not None:
                if plan_id is not None and int(plan_id) != existing:
                    raise TransportError(
                        "plan body is already registered under another id"
                    )
                return existing
            if plan_id is None:
                if self._next_id > 0xFFFFFFFF:
                    raise TransportError("plan id space exhausted")
                assigned = self._next_id
                self._next_id += 1
            else:
                assigned = _fit_u32(plan_id, "plan_id")
                if assigned < 1:
                    raise TransportError("plan id must be positive")
                if assigned in self._by_id:
                    raise TransportError(
                        "plan id is already registered with a different body"
                    )
                self._next_id = max(self._next_id, assigned + 1)
            self._by_body[frozen] = assigned
            self._by_id[assigned] = _thaw_plan(frozen)
            return assigned

    def has_plan(self, plan_id: int) -> bool:
        """Return True when ``plan_id`` is already registered locally."""
        with self._lock:
            return int(plan_id) in self._by_id

    def plan(self, plan_id: int) -> dict[str, Any]:
        """Return the plan body registered at ``plan_id``.

        Raises:
            TransportError: If ``plan_id`` was never registered.
        """
        with self._lock:
            try:
                return self._by_id[int(plan_id)]
            except KeyError as exc:
                raise TransportError(f"unknown plan_id {plan_id}") from exc

    def observe_session(self, session: int) -> None:
        """Record the peer wire session.

        Args:
            session: Positive u64 session (operation-id epoch).

        Raises:
            TransportError: If ``session`` moves backwards or leaves u64.
        """
        session = _fit_u64(session, "session")
        if session < 1:
            raise TransportError("wire session must be positive")
        with self._lock:
            if session < self.wire_session:
                raise TransportError("wire session epoch moved backwards")
            self.wire_session = session


def _plan_from_descriptor(
    descriptor: TensorDescriptor,
    *,
    cache_key: str,
    target_module: str,
    compute_scale: float | None,
) -> dict[str, Any]:
    if int(descriptor.invocation_id) != int(descriptor.slot_generation):
        raise TransportError(
            "binary wire requires invocation_id to match slot generation"
        )
    raw = descriptor.to_dict()
    for key in _PER_OP_PLAN_FIELDS:
        raw.pop(key, None)
    raw["cache_key"] = str(cache_key)
    raw["target_module"] = str(target_module)
    if compute_scale is not None:
        raw["compute_scale"] = float(compute_scale)
    return raw


def _descriptor_from_plan(
    plan: Mapping[str, Any],
    header: WireHeader,
    *,
    delta: bool,
) -> tuple[TensorDescriptor, str, str]:
    raw = dict(plan)
    cache_key = str(raw.pop("cache_key"))
    target_module = str(raw.pop("target_module", raw.get("layer_name", "")))
    raw.pop("compute_scale", None)
    if int(header.slot) != int(raw["slot_id"]):
        raise TransportError("wire slot does not match the registered plan")
    flags = FLAG_DELTA if delta else 0
    raw["operation_id"] = _operation_text(header.op_id, flags)
    raw["invocation_id"] = int(header.generation)
    raw["slot_generation"] = int(header.generation)
    raw["completion_sequence"] = int(header.completion_seq)
    raw["denoise_step"] = int(header.denoise_step)
    # Per-op request identity is stripped from the plan body; restore stable
    # placeholders so TensorDescriptor validation still passes on mailbox hits.
    raw.setdefault("request_id", f"op:{int(header.op_id)}")
    raw.setdefault("batch_id", f"session:{int(header.session)}")
    raw.setdefault("request_ids", ())
    return TensorDescriptor.from_obj(raw), cache_key, target_module


def pack_wire_frame(header: WireHeader, body: bytes = b"") -> bytes:
    """Pack one binary control frame.

    Args:
        header: Fixed prefix. ``length`` must equal ``len(body)``.
        body: Msgpack plan/IPC blob, or a packed ACK record. Empty on the
            steady tensor-ref and compact ACK path.

    Returns:
        Magic-prefixed frame bytes.

    Raises:
        TransportError: If the header and body disagree or a field overflows.
    """
    if header.version != _WIRE_VERSION:
        raise TransportError(f"unsupported control wire version {header.version}")
    if len(body) != int(header.length):
        raise TransportError("wire length does not match body")
    try:
        prefix = _WIRE_HEADER.pack(
            _WIRE_MAGIC,
            int(header.version),
            int(header.msg_type),
            int(header.flags),
            int(header.length),
            int(header.denoise_step),
            int(header.session),
            int(header.op_id),
            int(header.plan_id),
            int(header.slot),
            int(header.generation),
            int(header.completion_seq),
            int(header.event_index),
            int(header.registration_id),
        )
    except struct.error as exc:
        raise TransportError(
            "control frame field is outside its integer range"
        ) from exc
    return prefix + body


def unpack_wire_header(payload: bytes) -> tuple[WireHeader, bytes]:
    """Split one binary frame into its header and body.

    Args:
        payload: Bytes beginning with the control magic.

    Returns:
        The header and the body of ``header.length`` bytes.

    Raises:
        TransportError: If the magic, version, or length is wrong.
    """
    if len(payload) < _WIRE_HEADER.size:
        raise TransportError("truncated control frame")
    if not payload.startswith(_WIRE_MAGIC):
        raise TransportError("control frame magic mismatch")
    (
        _magic,
        version,
        msg_type,
        flags,
        length,
        denoise_step,
        session,
        op_id,
        plan_id,
        slot,
        generation,
        completion_seq,
        event_index,
        registration_id,
    ) = _WIRE_HEADER.unpack_from(payload)
    header = WireHeader(
        version=int(version),
        msg_type=int(msg_type),
        flags=int(flags),
        length=int(length),
        denoise_step=int(denoise_step),
        session=int(session),
        op_id=int(op_id),
        plan_id=int(plan_id),
        slot=int(slot),
        generation=int(generation),
        completion_seq=int(completion_seq),
        event_index=int(event_index),
        registration_id=int(registration_id),
    )
    body = payload[_WIRE_HEADER.size :]
    if header.version != _WIRE_VERSION:
        raise TransportError(f"unsupported control wire version {header.version}")
    if len(body) != header.length:
        raise TransportError("control frame length mismatch")
    return header, body


def _ack_dict(
    op_id: int,
    generation: int,
    completion_seq: int,
    flags: int,
    event_index: int,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "kind": "ack",
        "operation_id": _operation_text(op_id, flags),
        "slot_generation": int(generation),
        "consumer_complete": bool(flags & FLAG_CONSUMER_COMPLETE),
        "completion_sequence": int(completion_seq),
    }
    if flags & FLAG_COMPACT:
        message["compact"] = True
        message["event_index"] = int(event_index)
    return message


def _ack_flags(message: Mapping[str, Any]) -> tuple[int, int]:
    op_id, delta = parse_operation_id(str(message.get("operation_id", "")))
    flags = FLAG_DELTA if delta else 0
    if message.get("consumer_complete"):
        flags |= FLAG_CONSUMER_COMPLETE
    if message.get("compact"):
        flags |= FLAG_COMPACT
    return op_id, flags


def _pack_ack_record(message: Mapping[str, Any]) -> bytes:
    op_id, flags = _ack_flags(message)
    try:
        return _ACK_RECORD.pack(
            op_id,
            _fit_u64(int(message.get("slot_generation", 0)), "generation"),
            _fit_u64(int(message.get("completion_sequence", 0)), "completion_seq"),
            flags,
            _fit_u32(int(message.get("event_index", 0)), "event_index"),
        )
    except struct.error as exc:
        raise TransportError("ack record field is outside its integer range") from exc


def encode_plan_frame(
    plan_id: int,
    plan: Mapping[str, Any],
    *,
    session: int,
) -> bytes:
    """Encode one plan update as a binary header plus a msgpack body.

    Args:
        plan_id: Id the peer must register.
        plan: Canonical plan body.
        session: Wire session epoch.

    Returns:
        One control frame. The body is msgpack, not JSON or pickle.
    """
    body = _pack_msgpack(plan)
    header = WireHeader(
        version=_WIRE_VERSION,
        msg_type=int(WireMsgType.PLAN),
        flags=0,
        length=len(body),
        denoise_step=0,
        session=_fit_u64(session, "session"),
        op_id=0,
        plan_id=_fit_u32(plan_id, "plan_id"),
        slot=0,
        generation=0,
        completion_seq=0,
        event_index=0,
        registration_id=0,
    )
    return pack_wire_frame(header, body)


def encode_ack_frame(message: Mapping[str, Any], *, session: int) -> bytes:
    """Encode one ACK.

    Compact ACKs are header-only. A non-compact IPC event handle rides in a
    msgpack body so the steady path still does not call JSON.

    Args:
        message: ACK dict produced by ``build_compact_ack``.
        session: Wire session epoch.

    Returns:
        One ACK frame.
    """
    op_id, flags = _ack_flags(message)
    handle = message.get("completion_event_handle")
    body = b""
    if handle:
        body = _pack_msgpack({"completion_event_handle": str(handle)})
    header = WireHeader(
        version=_WIRE_VERSION,
        msg_type=int(WireMsgType.ACK),
        flags=flags,
        length=len(body),
        denoise_step=0,
        session=_fit_u64(session, "session"),
        op_id=op_id,
        plan_id=0,
        slot=0,
        generation=_fit_u64(int(message.get("slot_generation", 0)), "generation"),
        completion_seq=_fit_u64(
            int(message.get("completion_sequence", 0)), "completion_seq"
        ),
        event_index=_fit_u32(int(message.get("event_index", 0)), "event_index"),
        registration_id=0,
    )
    return pack_wire_frame(header, body)


def encode_ack_batch_frame(
    acks: list[dict[str, Any]],
    *,
    session: int,
) -> bytes:
    """Encode one or more ACKs.

    A single entry uses the ACK frame. Several entries pack fixed records
    after an ACK_BATCH header.

    Args:
        acks: ACK dicts, in send order.
        session: Wire session epoch.

    Returns:
        One control frame.

    Raises:
        TransportError: If ``acks`` is empty.
    """
    if not acks:
        raise TransportError("ack batch is empty")
    if len(acks) == 1:
        return encode_ack_frame(acks[0], session=session)
    body = b"".join(_pack_ack_record(ack) for ack in acks)
    first_op, first_flags = _ack_flags(acks[0])
    header = WireHeader(
        version=_WIRE_VERSION,
        msg_type=int(WireMsgType.ACK_BATCH),
        flags=first_flags,
        length=len(body),
        denoise_step=0,
        session=_fit_u64(session, "session"),
        op_id=first_op,
        plan_id=0,
        slot=0,
        generation=_fit_u64(int(acks[0].get("slot_generation", 0)), "generation"),
        completion_seq=_fit_u64(
            int(acks[0].get("completion_sequence", 0)), "completion_seq"
        ),
        event_index=_fit_u32(int(acks[0].get("event_index", 0)), "event_index"),
        registration_id=0,
    )
    return pack_wire_frame(header, body)


def build_steady_tensor_frames(
    registry: SessionPlanRegistry,
    plans_sent: set[int],
    descriptor: TensorDescriptor,
    *,
    session: int,
    cache_key: str,
    registration_id: int,
    first_registration: bool,
    ipc_payload: Mapping[str, Any] | None = None,
    compute: Mapping[str, Any] | None = None,
    source_ack: Mapping[str, Any] | None = None,
) -> tuple[int, list[bytes]]:
    """Build the frames for one tensor send.

    The first time ``plan_id`` is announced, a msgpack plan frame precedes the
    tensor frame. Later sends of the same plan are a single binary tensor ref.

    Args:
        registry: Registry that allocates or reuses ``plan_id``.
        plans_sent: Plan ids already written to this peer. Not mutated.
        descriptor: Per-operation tensor metadata.
        session: Wire session epoch.
        cache_key: IPC cache key stored in the plan.
        registration_id: Producer registration counter.
        first_registration: When true, ``ipc_payload`` is sent as msgpack.
        ipc_payload: CUDA IPC export dict for the first registration.
        compute: Folded compute control. Steady refs keep only its target.
        source_ack: Piggybacked ACK. Steady refs pack it as a fixed record.

    Returns:
        The plan id and the frames to write, in order.

    Raises:
        TransportError: If the operation id is not decimal, or the first
            registration has no IPC payload.
    """
    op_id, delta = parse_operation_id(descriptor.operation_id)
    target_module = descriptor.layer_name
    compute_scale = None
    if compute is not None:
        target_module = str(compute.get("target_module") or target_module)
        if "scale" in compute:
            compute_scale = float(compute["scale"])
    plan = _plan_from_descriptor(
        descriptor,
        cache_key=cache_key,
        target_module=target_module,
        compute_scale=compute_scale,
    )
    plan_id = registry.register_plan(plan)
    flags = FLAG_DELTA if delta else 0
    if compute is not None:
        flags |= FLAG_HAS_COMPUTE
    if source_ack is not None:
        flags |= FLAG_HAS_SOURCE_ACK
    if first_registration:
        if ipc_payload is None:
            raise TransportError("first IPC registration requires a payload")
        flags |= 1 << 5
        blob: dict[str, Any] = {
            "cache_key": str(cache_key),
            "payload": dict(ipc_payload),
        }
        if source_ack is not None:
            blob["source_ack"] = dict(source_ack)
        body = _pack_msgpack(blob)
        msg_type = int(WireMsgType.TENSOR)
    elif source_ack is not None:
        # Legacy piggyback ACK body on a tensor ref (pre-ring-only path).
        body = _pack_ack_record(source_ack)
        msg_type = int(WireMsgType.TENSOR_REF)
    elif compact_control_enabled() and binary_wire_enabled():
        # Steady cache-hit: header-only doorbell (no descriptor/handle bytes).
        body = b""
        msg_type = int(WireMsgType.DOORBELL)
    else:
        body = b""
        msg_type = int(WireMsgType.TENSOR_REF)
    header = WireHeader(
        version=_WIRE_VERSION,
        msg_type=msg_type,
        flags=flags,
        length=len(body),
        denoise_step=_fit_i32(descriptor.denoise_step, "denoise_step"),
        session=_fit_u64(session, "session"),
        op_id=op_id,
        plan_id=plan_id,
        slot=_fit_u32(descriptor.slot_id, "slot"),
        generation=_fit_u64(descriptor.slot_generation, "generation"),
        completion_seq=_fit_u64(descriptor.completion_sequence, "completion_seq"),
        event_index=0,
        registration_id=_fit_u32(registration_id, "registration_id"),
    )
    frames: list[bytes] = []
    if plan_id not in plans_sent:
        frames.append(
            encode_plan_frame(
                plan_id, registry.plan(plan_id), session=session
            )
        )
    frames.append(pack_wire_frame(header, body))
    return plan_id, frames


def _decode_ack_body(header: WireHeader, body: bytes) -> dict[str, Any]:
    message = _ack_dict(
        header.op_id,
        header.generation,
        header.completion_seq,
        header.flags,
        header.event_index,
    )
    if body:
        extra = _unpack_msgpack(body)
        handle = extra.get("completion_event_handle")
        if handle:
            message["completion_event_handle"] = str(handle)
    return message


def _decode_tensor_frame(
    registry: SessionPlanRegistry,
    header: WireHeader,
    body: bytes,
) -> dict[str, Any]:
    plan = registry.plan(header.plan_id)
    delta = bool(header.flags & FLAG_DELTA)
    descriptor, cache_key, target_module = _descriptor_from_plan(
        plan, header, delta=delta
    )
    if header.msg_type == int(WireMsgType.TENSOR):
        kind = "tensor"
    elif header.msg_type in {
        int(WireMsgType.TENSOR_REF),
        int(WireMsgType.DOORBELL),
    }:
        kind = "tensor_ref"
    else:
        raise TransportError(
            f"tensor frame decode got unexpected msg_type {header.msg_type}"
        )
    message: dict[str, Any] = {
        "kind": kind,
        "cache_key": cache_key,
        "registration_id": int(header.registration_id),
        "descriptor": descriptor.to_dict(),
    }
    if header.msg_type == int(WireMsgType.DOORBELL) and body:
        raise TransportError("doorbell frames must have an empty body")
    if header.flags & FLAG_HAS_COMPUTE:
        scale = plan.get("compute_scale", plan.get("scale", 1.0))
        message["compute"] = {
            "kind": "compute",
            "scale": float(scale),
            "target_module": target_module,
            "context": {},
        }
    if header.msg_type == int(WireMsgType.TENSOR):
        blob = _unpack_msgpack(body)
        message["cache_key"] = str(blob.get("cache_key") or cache_key)
        if "payload" not in blob:
            raise TransportError("tensor registration is missing its IPC payload")
        message["payload"] = dict(blob["payload"])
        source_ack = blob.get("source_ack")
        if isinstance(source_ack, dict):
            message["source_ack"] = source_ack
    elif header.flags & FLAG_HAS_SOURCE_ACK:
        if len(body) != _ACK_RECORD.size:
            raise TransportError("tensor ref source ack is the wrong size")
        op_id, generation, completion_seq, flags, event_index = _ACK_RECORD.unpack(
            body
        )
        message["source_ack"] = _ack_dict(
            int(op_id),
            int(generation),
            int(completion_seq),
            int(flags),
            int(event_index),
        )
    return message


def decode_control_bytes(
    payload: bytes,
    registry: SessionPlanRegistry,
) -> dict[str, Any] | None:
    """Decode one control datagram.

    Binary frames are unpacked with ``struct``. Plan frames are registered and
    return None so the caller reads the following steady-state frame. JSON is
    only used when the payload is not a binary frame.

    Args:
        payload: One ``send_bytes`` datagram.
        registry: Registry updated by plan frames and session epochs.

    Returns:
        A control dict, or None when the datagram was a consumed plan update.

    Raises:
        TransportError: If a binary frame is malformed or its plan is unknown.
    """
    if payload.startswith(_WIRE_MAGIC):
        header, body = unpack_wire_header(payload)
        registry.observe_session(header.session)
        msg_type = header.msg_type
        if msg_type == int(WireMsgType.PLAN):
            registry.register_plan(_unpack_msgpack(body), plan_id=header.plan_id)
            return None
        if msg_type == int(WireMsgType.CLOSE):
            return {"kind": "close"}
        if msg_type == int(WireMsgType.ACK):
            return _decode_ack_body(header, body)
        if msg_type == int(WireMsgType.ACK_BATCH):
            if not body or len(body) % _ACK_RECORD.size != 0:
                raise TransportError("ack batch payload is truncated")
            acks = []
            for offset in range(0, len(body), _ACK_RECORD.size):
                op_id, generation, completion_seq, flags, event_index = (
                    _ACK_RECORD.unpack_from(body, offset)
                )
                acks.append(
                    _ack_dict(
                        int(op_id),
                        int(generation),
                        int(completion_seq),
                        int(flags),
                        int(event_index),
                    )
                )
            return {"kind": "ack_batch", "acks": acks}
        if msg_type in {
            int(WireMsgType.TENSOR),
            int(WireMsgType.TENSOR_REF),
            int(WireMsgType.DOORBELL),
        }:
            return _decode_tensor_frame(registry, header, body)
        raise TransportError(f"unsupported control message type {msg_type}")
    message = wire_json_loads(payload)
    if not isinstance(message, dict):
        raise TransportError("control message must be an object")
    return message


class ControlInbox:
    """Demultiplexes ack and tensor control messages on one duplex pipe.

    Allows wait_ack / receive to tolerate reordering (completion-queue style)
    without dropping peer messages. Supports ``ack_batch`` and source-ack
    piggybacks on tensor messages so steady-state is not one host RTT per
    target.
    """

    def __init__(
        self,
        connection: Any,
        registry: SessionPlanRegistry | None = None,
    ) -> None:
        self.connection = connection
        self.registry = registry if registry is not None else SessionPlanRegistry()
        self._lock = threading.Lock()
        self._acks: dict[tuple[str, int], dict[str, Any]] = {}
        self._tensors: list[dict[str, Any]] = []
        self._controls: list[dict[str, Any]] = []

    def _store_ack(self, raw: dict[str, Any]) -> None:
        key = (str(raw.get("operation_id")), int(raw.get("slot_generation", -1)))
        self._acks[key] = raw

    def _ingest(self, raw: dict[str, Any]) -> None:
        kind = raw.get("kind")
        if kind == "ack":
            self._store_ack(raw)
            return
        if kind == "ack_batch":
            for entry in raw.get("acks", ()):
                if not isinstance(entry, dict):
                    raise TransportError("ack_batch entry must be an object")
                ack = dict(entry)
                ack.setdefault("kind", "ack")
                self._store_ack(ack)
            return
        if kind in {"tensor", "tensor_ref"}:
            source_ack = raw.get("source_ack")
            if isinstance(source_ack, dict):
                ack = dict(source_ack)
                ack.setdefault("kind", "ack")
                self._store_ack(ack)
            self._tensors.append(raw)
            return
        if kind in {"compute", "close"}:
            self._controls.append(raw)
            return
        raise TransportError(f"unexpected control message kind: {kind}")

    def _pump(self, timeout_s: float) -> bool:
        if not self.connection.poll(timeout_s):
            return False
        message = decode_control_bytes(self.connection.recv_bytes(), self.registry)
        if message is None:
            return True
        self._ingest(message)
        return True

    def wait_ack(
        self,
        operation_id: str,
        slot_generation: int,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        key = (str(operation_id), int(slot_generation))
        deadline = time.monotonic() + timeout_s
        poll_s = _control_poll_s()
        with self._lock:
            while time.monotonic() < deadline:
                if key in self._acks:
                    return self._acks.pop(key)
                remaining = max(0.0, deadline - time.monotonic())
                if not self._pump(min(poll_s, remaining)):
                    continue
            raise TimeoutError(f"IPC acknowledgement timed out: {operation_id}")

    def recv_tensor(self, timeout_s: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        poll_s = _control_poll_s()
        with self._lock:
            while time.monotonic() < deadline:
                if self._tensors:
                    return self._tensors.pop(0)
                remaining = max(0.0, deadline - time.monotonic())
                if not self._pump(min(poll_s, remaining)):
                    continue
            raise TimeoutError("IPC tensor receive timed out")

    def recv_control(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Receive a compute/close control message (or folded tensor+compute)."""
        deadline = time.monotonic() + timeout_s
        poll_s = _control_poll_s()
        with self._lock:
            while time.monotonic() < deadline:
                if self._controls:
                    return self._controls.pop(0)
                if self._tensors:
                    # Folded compute path: tensor carries the compute payload.
                    return self._tensors.pop(0)
                remaining = max(0.0, deadline - time.monotonic())
                if not self._pump(min(poll_s, remaining)):
                    continue
            raise TimeoutError("IPC control receive timed out")


class CudaIpcTransfer:
    """CUDA IPC transfer with session-scoped tensor/event registrations."""

    def __init__(self, control_connection: Any, *, role: str, device: int) -> None:
        if role not in {"producer", "consumer"}:
            raise ValueError("role must be producer or consumer")
        self.connection = control_connection
        self.role = role
        self.device = int(device)
        self.closed = False
        self._send_lock = threading.Lock()
        # cache_key -> (payload, event, exported_tensor, registration_id)
        self._producer_cache: dict[
            str, tuple[_IpcTensorPayload, torch.cuda.Event, torch.Tensor, int]
        ] = {}
        self._consumer_cache: dict[
            str, tuple[torch.Tensor, torch.cuda.Event | None, int]
        ] = {}
        # D9: bound IPC registration caches (session growth fail-closed recycle).
        # Default must cover a full Flux all-targets session (~340 unique X
        # slots); a too-small cap clears the cache mid-step and forces every
        # later publish back onto the cold host TENSOR path.
        self._cache_max_entries = max(
            16, int(os.environ.get("TETRIS_LORA_IPC_CACHE_MAX", "4096"))
        )
        self._registration_seq = 0
        self._signal_ring: IpcSignalRing | None = None
        self._mailbox: ControlMailbox | None = None
        self._inbox: ControlInbox | None = None
        self._pending_acks: list[dict[str, Any]] = []
        # Legacy single-connection wait_ack stash so ack_batch entries are not dropped.
        self._ack_stash: dict[tuple[str, int], dict[str, Any]] = {}
        self._plans = SessionPlanRegistry()
        self._plans_sent: set[int] = set()
        self._steady_plan_ids: dict[tuple[Any, ...], int] = {}
        self.last_publish_mechanism = "none"
        self.last_mailbox_publish_us = 0.0
        self._wire_session = 1
        self._wire_session_bound = False
        # Plans the peer is known to hold (via prior host PLAN frames).
        self._plans_known_peer: set[int] = set()

    @property
    def registry(self) -> SessionPlanRegistry:
        """Plan registry shared with the peer inbox on this connection."""
        return self._plans

    def attach_registry(self, registry: SessionPlanRegistry) -> None:
        """Use ``registry`` for plan ids on this endpoint.

        Args:
            registry: Session registry shared by the pipe's local endpoints.
        """
        self._plans = registry
        self._steady_plan_ids.clear()

    def bind_wire_session(self, session: int) -> None:
        """Pin the numeric wire session (operation-id epoch) for sends.

        Args:
            session: Positive u64 session. Unbound endpoints echo the peer.
        """
        session = _fit_u64(session, "session")
        if session < 1:
            raise TransportError("wire session must be positive")
        self._wire_session = session
        self._wire_session_bound = True

    def _session_id(self) -> int:
        if self._wire_session_bound:
            return self._wire_session
        return int(self._plans.wire_session)

    def attach_signal_ring(self, ring: IpcSignalRing) -> None:
        """Bind a session-scoped IPC signal ring for compact acknowledgements."""
        self._signal_ring = ring

    def attach_mailbox(self, mailbox: ControlMailbox) -> None:
        """Bind the session shared control mailbox (steady no-host path)."""
        self._mailbox = mailbox

    def _maybe_mailbox_ring_publish(
        self,
        descriptor: TensorDescriptor,
        *,
        cache_key: str,
        registration_id: int,
        compute: dict[str, Any] | None,
        plan_id: int | None = None,
    ) -> bool:
        """Publish X/Δ mailbox entry + ring wake. Returns True when published."""
        if (
            not mailbox_control_enabled()
            or self._mailbox is None
            or self._signal_ring is None
        ):
            return False
        publish_started_ns = time.monotonic_ns()
        target_module = descriptor.layer_name
        compute_scale = None
        if compute is not None:
            target_module = str(compute.get("target_module") or target_module)
            if "scale" in compute:
                compute_scale = float(compute["scale"])
        if plan_id is None:
            plan = _plan_from_descriptor(
                descriptor,
                cache_key=cache_key,
                target_module=target_module,
                compute_scale=compute_scale,
            )
            plan_id = int(self._plans.register_plan(plan))
        op_id, is_delta = parse_operation_id(descriptor.operation_id)
        flags = 0
        if compute is not None:
            flags |= MAILBOX_FLAG_HAS_COMPUTE
        if is_delta:
            flags |= MAILBOX_FLAG_DELTA
        seq = int(descriptor.completion_sequence)
        publish = (
            self._mailbox.publish_delta if is_delta else self._mailbox.publish_x
        )
        idx = publish(
            seq=seq,
            plan_id=int(plan_id),
            registration_id=int(registration_id),
            slot_id=int(descriptor.slot_id),
            slot_generation=int(descriptor.slot_generation),
            op_id=int(op_id),
            flags=int(flags),
            denoise_step=int(descriptor.denoise_step),
        )
        self._signal_ring.record_local(
            idx,
            torch.cuda.current_stream(self.device),
            generation=seq,
        )
        self.last_mailbox_publish_us = (
            time.monotonic_ns() - publish_started_ns
        ) / 1000.0
        return True

    def attach_inbox(self, inbox: ControlInbox) -> None:
        """Share a demux inbox across producer/consumer endpoints on one pipe."""
        self._inbox = inbox
        self.attach_registry(inbox.registry)

    def send(
        self,
        descriptor: TensorDescriptor,
        tensor: torch.Tensor,
        *,
        cache_key: str | None = None,
        compute: dict[str, Any] | None = None,
        source_ack: dict[str, Any] | None = None,
    ) -> None:
        self.last_publish_mechanism = "host_pipe"
        self.last_mailbox_publish_us = 0.0
        if self.role != "producer":
            raise TransportError("only producer may send")
        _validate_payload(descriptor, tensor)
        if tensor.device.index != self.device:
            raise ValueError("tensor device does not match producer device")
        previous_device = torch.cuda.current_device()
        changed_device = previous_device != self.device
        if changed_device:
            torch.cuda.set_device(self.device)
        try:
            self._send_registered(
                descriptor,
                tensor,
                cache_key,
                compute=compute,
                source_ack=source_ack,
            )
        finally:
            if changed_device:
                torch.cuda.set_device(previous_device)

    def _send_registered(
        self,
        descriptor: TensorDescriptor,
        tensor: torch.Tensor,
        cache_key: str | None,
        *,
        compute: dict[str, Any] | None = None,
        source_ack: dict[str, Any] | None = None,
    ) -> None:
        cache_key = cache_key or descriptor.buffer_token
        cached = self._producer_cache.get(cache_key)
        if cached is None:
            event = torch.cuda.Event(enable_timing=False, interprocess=True)
            # Hold a strong reference to the exact exported allocation so a
            # later same-key tensor cannot silently reuse a stale IPC handle.
            exported = tensor
            payload = export_cuda_tensor(exported, target_device=self.device)
            payload = _IpcTensorPayload(
                **{
                    **asdict(payload),
                    "target_device": self.device,
                    "producer_ready_event_handle": _b64(event.ipc_handle()),
                }
            )
            self._registration_seq += 1
            registration_id = self._registration_seq
            if len(self._producer_cache) >= self._cache_max_entries:
                # Session recycle: drop all entries rather than risk stale
                # handle reuse after partial eviction.
                self._producer_cache.clear()
            self._producer_cache[cache_key] = (
                payload,
                event,
                exported,
                registration_id,
            )
            first_registration = True
        else:
            payload, event, exported, registration_id = cached
            if exported is not tensor:
                if exported.device != tensor.device:
                    raise TransportError(
                        f"IPC cache device mismatch for {cache_key}: "
                        f"{exported.device} vs {tensor.device}"
                    )
                if (
                    exported.untyped_storage().data_ptr()
                    != tensor.untyped_storage().data_ptr()
                ):
                    raise TransportError(
                        f"IPC cache storage identity changed for {cache_key}; "
                        "refuse silent re-registration of a live handle"
                    )
                if exported.storage_offset() != tensor.storage_offset():
                    raise TransportError(
                        f"IPC cache storage offset changed for {cache_key}"
                    )
                if payload.size != tuple(int(value) for value in tensor.shape):
                    raise TransportError(f"IPC cache shape mismatch for {cache_key}")
                if payload.stride != tuple(int(value) for value in tensor.stride()):
                    raise TransportError(f"IPC cache stride mismatch for {cache_key}")
                if payload.dtype != str(tensor.dtype).split(".")[-1]:
                    raise TransportError(f"IPC cache dtype mismatch for {cache_key}")
                if tuple(int(v) for v in exported.shape) != tuple(
                    int(v) for v in tensor.shape
                ) or tuple(int(v) for v in exported.stride()) != tuple(
                    int(v) for v in tensor.stride()
                ):
                    raise TransportError(
                        f"IPC cache view layout changed for {cache_key}"
                    )
            first_registration = False
        event.record(torch.cuda.current_stream(self.device))
        # Steady mailbox path: cache-hit + peer already has the plan → no pipe.
        if (
            not first_registration
            and source_ack is None
            and mailbox_control_enabled()
            and self._mailbox is not None
            and self._signal_ring is not None
        ):
            target_module = descriptor.layer_name
            compute_scale = None
            if compute is not None:
                target_module = str(compute.get("target_module") or target_module)
                if "scale" in compute:
                    compute_scale = float(compute["scale"])
            descriptor_signature = (
                descriptor.session_epoch,
                descriptor.layer_name,
                descriptor.slot_id,
                descriptor.shape,
                descriptor.stride,
                descriptor.dtype,
                descriptor.layout,
                descriptor.payload_bytes,
                descriptor.adapter_id,
                descriptor.adapter_version,
                descriptor.rank,
                descriptor.scale,
                descriptor.branch_index,
                descriptor.output_offset,
                descriptor.row_spans,
                descriptor.buffer_capacity,
                descriptor.offset,
                descriptor.alpha_folded,
                descriptor.request_composition,
                descriptor.request_ranks,
                descriptor.slice_rank_tables,
                descriptor.row_layout,
                descriptor.output_slices,
                descriptor.protocol_version,
            )
            plan_cache_key = (
                descriptor_signature,
                str(cache_key),
                str(target_module),
                compute_scale,
            )
            plan_id = self._steady_plan_ids.get(plan_cache_key)
            if plan_id is None:
                plan = _plan_from_descriptor(
                    descriptor,
                    cache_key=cache_key,
                    target_module=target_module,
                    compute_scale=compute_scale,
                )
                plan_id = self._plans.register_plan(plan)
                self._steady_plan_ids[plan_cache_key] = int(plan_id)
            if plan_id in self._plans_sent or plan_id in self._plans_known_peer:
                if self._maybe_mailbox_ring_publish(
                    descriptor,
                    cache_key=cache_key,
                    registration_id=int(registration_id),
                    compute=compute,
                    plan_id=int(plan_id),
                ):
                    self.last_publish_mechanism = "mailbox_ring"
                    return
        if binary_wire_enabled():
            with self._send_lock:
                plan_id, frames = build_steady_tensor_frames(
                    self._plans,
                    self._plans_sent,
                    descriptor,
                    session=self._session_id(),
                    cache_key=cache_key,
                    registration_id=int(registration_id),
                    first_registration=first_registration,
                    ipc_payload=asdict(payload) if first_registration else None,
                    compute=compute,
                    source_ack=source_ack,
                )
                for frame in frames:
                    self.connection.send_bytes(frame)
                self._plans_sent.add(plan_id)
                self._plans_known_peer.add(plan_id)
            # Always mirror a mailbox wake after host send (except piggyback
            # ACK frames). Peers gate materialize on has_plan / IPC cache and
            # fall back to the host frames when this wake is cold.
            if source_ack is None:
                self._maybe_mailbox_ring_publish(
                    descriptor,
                    cache_key=cache_key,
                    registration_id=int(registration_id),
                    compute=compute,
                    plan_id=int(plan_id),
                )
            return
        # Legacy JSON path. Descriptor fields are a nested dict, not a second
        # JSON string inside the outer message.
        message: dict[str, Any] = {
            "kind": "tensor" if first_registration else "tensor_ref",
            "cache_key": cache_key,
            "registration_id": int(registration_id),
            "descriptor": descriptor.to_dict(),
        }
        if first_registration:
            message["payload"] = asdict(payload)
        if compute is not None:
            message["compute"] = compute
        if source_ack is not None:
            message["source_ack"] = source_ack
        encoded = wire_json_dumps(
            message, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        with self._send_lock:
            self.connection.send_bytes(encoded)

    def _materialize(
        self, raw: dict[str, Any]
    ) -> tuple[TensorDescriptor, torch.Tensor, torch.cuda.Event | None]:
        if raw.get("kind") not in {"tensor", "tensor_ref"}:
            raise TransportError(f"unexpected IPC message kind: {raw.get('kind')}")
        descriptor = TensorDescriptor.from_obj(raw["descriptor"])
        cache_key = str(raw.get("cache_key") or descriptor.buffer_token)
        registration_id = int(raw.get("registration_id", -1))
        cached = self._consumer_cache.get(cache_key)
        if raw.get("kind") == "tensor_ref":
            if cached is None:
                raise TransportError(f"unknown IPC cache key {cache_key}")
            tensor, event, cached_registration = cached
            if registration_id >= 0 and cached_registration != registration_id:
                raise TransportError(
                    f"stale IPC registration for {cache_key}: "
                    f"have={cached_registration} got={registration_id}"
                )
        else:
            payload = _IpcTensorPayload.from_obj(raw["payload"])
            payload = _IpcTensorPayload(**{**asdict(payload), "target_device": self.device})
            tensor, event = import_cuda_tensor(payload)
            if registration_id < 0:
                self._registration_seq += 1
                registration_id = self._registration_seq
            if len(self._consumer_cache) >= self._cache_max_entries:
                self._consumer_cache.clear()
            self._consumer_cache[cache_key] = (tensor, event, registration_id)
        _validate_payload(descriptor, tensor)
        # Producer-ready IPC Event: consumer must wait before reading X/Δ.
        # On the skip_delta_host_wait path the worker intentionally does not
        # wait_ack the prior Δ consume Signal; the next X materialize wait here
        # is the GPU fence for Base having finished the previous Δ P2P copy
        # (Base records X ready only after finish_delta under the client lock).
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)
        return descriptor, tensor, event

    def receive(self) -> tuple[TensorDescriptor, torch.Tensor, torch.cuda.Event | None]:
        if self.role != "consumer":
            raise TransportError("only consumer may receive")
        if self._inbox is not None:
            raw = self._inbox.recv_tensor()
        else:
            raw = self._read_control_message()
        return self._materialize(raw)

    def _read_control_message(self) -> dict[str, Any]:
        message = decode_control_bytes(self.connection.recv_bytes(), self._plans)
        while message is None:
            message = decode_control_bytes(self.connection.recv_bytes(), self._plans)
        return message

    def receive_from_raw(
        self, raw: dict[str, Any]
    ) -> tuple[TensorDescriptor, torch.Tensor, torch.cuda.Event | None]:
        """Materialize a tensor message already pulled from the inbox/pipe."""
        if self.role != "consumer":
            raise TransportError("only consumer may receive")
        return self._materialize(raw)

    def wait_mailbox_publish(
        self,
        seq: int,
        *,
        delta: bool,
        timeout_s: float = 30.0,
    ) -> None:
        """Wait for a mailbox X/Δ publish fenced by the peer signal ring."""
        if self._mailbox is None or self._signal_ring is None:
            raise TransportError("mailbox wait requires mailbox + signal ring")
        idx = ring_index_for_sequence(seq, self._signal_ring.capacity)
        self._signal_ring.wait_peer_with_generation(
            idx, int(seq), timeout_s=timeout_s
        )
        if delta:
            self._mailbox.read_delta(int(seq))
        else:
            self._mailbox.read_x(int(seq))

    def materialize_from_mailbox(
        self, seq: int, *, delta: bool
    ) -> tuple[
        TensorDescriptor,
        torch.Tensor,
        torch.cuda.Event | None,
        dict[str, Any] | None,
    ]:
        """Materialize a cache-hit tensor from the shared mailbox entry."""
        if self.role != "consumer":
            raise TransportError("only consumer may materialize from mailbox")
        if self._mailbox is None:
            raise TransportError("mailbox is not attached")
        row = (
            self._mailbox.read_delta(int(seq))
            if delta
            else self._mailbox.read_x(int(seq))
        )
        flags = int(row["flags"])
        header = WireHeader(
            version=_WIRE_VERSION,
            msg_type=int(WireMsgType.DOORBELL),
            flags=FLAG_DELTA if delta else 0,
            length=0,
            denoise_step=int(row["denoise_step"]),
            session=self._session_id(),
            op_id=int(row["op_id"]),
            plan_id=int(row["plan_id"]),
            slot=int(row["slot_id"]),
            generation=int(row["slot_generation"]),
            completion_seq=int(row["seq"]),
            event_index=0,
            registration_id=int(row["registration_id"]),
        )
        descriptor, cache_key, target_module = _descriptor_from_plan(
            self._plans.plan(int(row["plan_id"])), header, delta=delta
        )
        raw: dict[str, Any] = {
            "kind": "tensor_ref",
            "cache_key": cache_key,
            "registration_id": int(row["registration_id"]),
            "descriptor": descriptor.to_dict(),
        }
        compute = None
        if flags & MAILBOX_FLAG_HAS_COMPUTE:
            plan = self._plans.plan(int(row["plan_id"]))
            scale = plan.get("compute_scale", plan.get("scale", 1.0))
            compute = {
                "kind": "compute",
                "scale": float(scale),
                "target_module": target_module,
                "context": {},
            }
            raw["compute"] = compute
        descriptor_out, tensor, event = self._materialize(raw)
        return descriptor_out, tensor, event, compute

    def try_poll_mailbox_x(self, seq: int) -> bool:
        """Return True if mailbox X entry ``seq`` is published and plan-known.

        Plan-known gating avoids racing a host PLAN frame that is still in the
        pipe when a premature mailbox wake is visible.
        """
        if self._mailbox is None:
            return False
        idx = ring_index_for_sequence(seq, self._mailbox.capacity)
        row = self._mailbox.x_entries[idx]
        if int(row["seq"]) != int(seq):
            return False
        return self._plans.has_plan(int(row["plan_id"]))

    def try_poll_mailbox_delta(self, seq: int) -> bool:
        """Return True if mailbox Δ entry ``seq`` is published and plan-known."""
        if self._mailbox is None:
            return False
        idx = ring_index_for_sequence(seq, self._mailbox.capacity)
        row = self._mailbox.delta_entries[idx]
        if int(row["seq"]) != int(seq):
            return False
        return self._plans.has_plan(int(row["plan_id"]))

    def build_compact_ack(
        self,
        descriptor: TensorDescriptor,
        *,
        consumer_complete: bool,
        completion_event: torch.cuda.Event | None = None,
    ) -> dict[str, Any]:
        """Build a compact ACK dict and record the session signal-ring event."""
        message: dict[str, Any] = {
            "kind": "ack",
            "operation_id": descriptor.operation_id,
            "slot_generation": descriptor.slot_generation,
            "consumer_complete": bool(consumer_complete),
            "completion_sequence": int(descriptor.completion_sequence),
        }
        use_compact = (
            compact_control_enabled()
            and self._signal_ring is not None
            and completion_event is not None
        )
        if use_compact:
            assert self._signal_ring is not None
            event_index = ring_index_for_sequence(
                descriptor.completion_sequence, self._signal_ring.capacity
            )
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(completion_event)
            # Fence with completion_sequence (monotonic per op), not slot
            # generation which repeats across early ops on the same slot.
            self._signal_ring.record_local(
                event_index,
                stream,
                generation=int(descriptor.completion_sequence),
            )
            message["compact"] = True
            message["event_index"] = int(event_index)
        elif completion_event is not None:
            message["completion_event_handle"] = _b64(completion_event.ipc_handle())
        return message

    def acknowledge(
        self,
        descriptor: TensorDescriptor,
        *,
        consumer_complete: bool,
        completion_event: torch.cuda.Event | None = None,
        batch: bool | None = None,
        host: bool = True,
        flush: bool = False,
    ) -> None:
        """Record completion; optionally defer/batch or skip the host ACK bytes.

        Args:
            descriptor: Operation metadata being acknowledged.
            consumer_complete: Whether the consumer finished the payload.
            completion_event: GPU event that must complete before the peer
                observes this ACK.
            batch: Override operation-batch ACK coalescing.
            host: When False and compact Signal fencing is active, only the
                ring event is recorded (no host pipe write).
            flush: When True, flush pending host ACKs immediately (size>=1) so a
                waiting producer cannot stall on a deferred batch.
        """
        if self.role != "consumer":
            raise TransportError("only consumer may acknowledge")
        message = self.build_compact_ack(
            descriptor,
            consumer_complete=consumer_complete,
            completion_event=completion_event,
        )
        if not host:
            if not message.get("compact"):
                raise TransportError(
                    "host=False acknowledge requires compact signal fencing"
                )
            # Skip-host path: still progress any previously batched host ACKs.
            if self._pending_acks:
                with self._send_lock:
                    self._flush_acks_locked()
            return
        use_batch = operation_batch_ack_enabled() if batch is None else bool(batch)
        if use_batch and message.get("compact"):
            with self._send_lock:
                self._pending_acks.append(message)
                if flush or len(self._pending_acks) >= operation_ack_batch_size():
                    self._flush_acks_locked()
            return
        with self._send_lock:
            if self._pending_acks:
                self._flush_acks_locked()
            self._send_ack_locked(message)

    def flush_acks(self) -> int:
        """Flush deferred operation-batch ACK host messages."""
        with self._send_lock:
            return self._flush_acks_locked()

    def _flush_acks_locked(self) -> int:
        if not self._pending_acks:
            return 0
        batch = list(self._pending_acks)
        self._pending_acks.clear()
        if binary_wire_enabled():
            self.connection.send_bytes(
                encode_ack_batch_frame(batch, session=self._session_id())
            )
            return len(batch)
        if len(batch) == 1:
            payload: dict[str, Any] = batch[0]
        else:
            payload = {"kind": "ack_batch", "acks": batch}
        self._send_json_locked(payload)
        return len(batch)

    def _send_ack_locked(self, message: dict[str, Any]) -> None:
        if binary_wire_enabled():
            self.connection.send_bytes(
                encode_ack_frame(message, session=self._session_id())
            )
            return
        self._send_json_locked(message)

    def _send_json_locked(self, message: dict[str, Any]) -> None:
        self.connection.send_bytes(
            wire_json_dumps(message, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )

    def wait_ack(self, descriptor: TensorDescriptor, timeout_s: float = 30.0) -> None:
        if self.role != "producer":
            raise TransportError("only producer may wait for acknowledgement")
        want_key = (str(descriptor.operation_id), int(descriptor.slot_generation))
        if self._inbox is not None:
            raw = self._inbox.wait_ack(
                descriptor.operation_id,
                descriptor.slot_generation,
                timeout_s=timeout_s,
            )
        else:
            raw = self._ack_stash.pop(want_key, None)
            deadline = time.monotonic() + timeout_s
            poll_s = _control_poll_s()
            while raw is None and time.monotonic() < deadline:
                if not self.connection.poll(poll_s):
                    continue
                message = self._read_control_message()
                if message.get("kind") == "ack_batch":
                    # Preserve every ack_batch entry; do not drop siblings.
                    for entry in message.get("acks", ()):
                        if not isinstance(entry, dict):
                            raise TransportError("ack_batch entry must be an object")
                        ack = dict(entry)
                        ack.setdefault("kind", "ack")
                        key = (
                            str(ack.get("operation_id")),
                            int(ack.get("slot_generation", -1)),
                        )
                        if key == want_key and raw is None:
                            raw = ack
                        else:
                            self._ack_stash[key] = ack
                    continue
                if message.get("kind") == "ack":
                    key = (
                        str(message.get("operation_id")),
                        int(message.get("slot_generation", -1)),
                    )
                    if key == want_key:
                        raw = message
                    else:
                        self._ack_stash[key] = message
                    continue
                raise TransportError(
                    f"unexpected IPC acknowledgement kind: {message.get('kind')}"
                )
            if raw is None:
                raise TimeoutError(
                    f"IPC acknowledgement timed out: {descriptor.operation_id}"
                )
        if raw.get("kind") != "ack":
            raise TransportError("unexpected IPC acknowledgement")
        if raw.get("operation_id") != descriptor.operation_id:
            raise TransportError("IPC acknowledgement operation mismatch")
        if int(raw.get("slot_generation", -1)) != descriptor.slot_generation:
            raise TransportError("IPC acknowledgement generation mismatch")
        if not raw.get("consumer_complete", False):
            raise TransportError("consumer did not report completion")
        if raw.get("compact") and self._signal_ring is not None:
            if self._signal_ring.peer_generations is not None:
                self._signal_ring.wait_peer_with_generation(
                    int(raw["event_index"]),
                    int(descriptor.completion_sequence),
                )
            else:
                self._signal_ring.wait_peer(int(raw["event_index"]))
            return
        event_handle = raw.get("completion_event_handle")
        if event_handle:
            completion_event = torch.cuda.Event.from_ipc_handle(
                self.device, _unb64(event_handle)
            )
            torch.cuda.current_stream(self.device).wait_event(completion_event)

    @torch.compiler.disable
    def wait_ack_signal(
        self,
        descriptor: TensorDescriptor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Wait for a generation-fenced signal-ring ACK (no host ticket).

        Requires compact control, an attached signal ring, and an imported peer
        generation table. Mismatch or missing setup is fail-closed.
        """
        if self.role != "producer":
            raise TransportError("only producer may wait for acknowledgement")
        if not compact_control_enabled():
            raise TransportError(
                "wait_ack_signal requires TETRIS_LORA_COMPACT_CONTROL"
            )
        if self._signal_ring is None:
            raise TransportError("wait_ack_signal requires an attached signal ring")
        if self._signal_ring.peer_generations is None:
            raise TransportError(
                "wait_ack_signal requires a peer generation table"
            )
        event_index = ring_index_for_sequence(
            descriptor.completion_sequence, self._signal_ring.capacity
        )
        self._signal_ring.wait_peer_with_generation(
            event_index,
            int(descriptor.completion_sequence),
            stream=stream,
        )

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                with self._send_lock:
                    self._flush_acks_locked()
            except Exception:
                pass
            self._producer_cache.clear()
            self._consumer_cache.clear()
            try:
                self.connection.close()
            except Exception:
                pass


def inspect_local_ipc_capability() -> dict[str, Any]:
    """Return evidence without claiming NVLink or RDMA."""
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA is unavailable"}
    device_count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    p2p = []
    for source in range(device_count):
        for target in range(device_count):
            if source == target:
                continue
            p2p.append(
                {
                    "source": source,
                    "target": target,
                    "can_access_peer": bool(
                        torch.cuda.can_device_access_peer(source, target)
                    ),
                }
            )
    return {
        "available": device_count >= 2,
        "cuda_device_count": device_count,
        "cuda_visible_devices": visible,
        "p2p": p2p,
        "nvlink_confirmed": False,
        "actual_transfer_path_confirmed": False,
    }


__all__ = [
    "BufferState",
    "ControlInbox",
    "CudaIpcTransfer",
    "ControlMailbox",
    "IpcSignalRing",
    "mailbox_control_enabled",
    "SlotLease",
    "TensorDescriptor",
    "TransportError",
    "TransportUnavailableError",
    "TransferResult",
    "OperationIdAllocator",
    "SessionPlanRegistry",
    "WireHeader",
    "WireMsgType",
    "binary_wire_enabled",
    "build_steady_tensor_frames",
    "compact_control_enabled",
    "decode_control_bytes",
    "export_cuda_tensor",
    "import_cuda_tensor",
    "inspect_local_ipc_capability",
    "operation_ack_batch_size",
    "operation_batch_ack_enabled",
    "pack_wire_frame",
    "parse_operation_id",
    "reset_wire_json_counts",
    "ring_index_for_sequence",
    "unpack_wire_header",
    "wire_json_counts",
    "wire_json_dumps",
    "wire_json_loads",
]
