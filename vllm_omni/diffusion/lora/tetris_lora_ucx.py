"""CUDA 12-compatible UCX RMA backend with strict RDMA gating."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import queue
import shutil
import socket
import struct
import subprocess
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .tetris_lora_transport import TransportError, TransportUnavailableError


def _progress_owned(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run UCX worker calls on its single owning thread."""

    def wrapped(self: "UcxRdmaEndpoint", *args: Any, **kwargs: Any) -> Any:
        loop = getattr(self, "_progress_loop", None)
        if loop is None or threading.current_thread() is loop._thread:
            return method(self, *args, **kwargs)
        return loop.submit(lambda: method(self, *args, **kwargs))

    return wrapped


@dataclass(frozen=True)
class UcxCapability:
    """Evidence collected before enabling the RDMA data plane."""

    ucx_version: str
    ib_device_visible: bool
    ib_transport_visible: bool
    cuda_plugin_visible: bool
    gdrcopy_plugin_visible: bool
    reason: str


class UcxTransferState(str, Enum):
    """States acknowledged by both sides of one GPU RMA operation."""

    FREE = "free"
    PUT_SUBMITTED = "put_submitted"
    PUT_LOCAL_COMPLETE = "put_local_complete"
    REMOTE_WRITE_COMPLETE = "remote_write_complete"
    REMOTE_VISIBLE = "remote_visible"
    CONSUMER_IN_FLIGHT = "consumer_in_flight"
    CONSUMER_COMPLETE = "consumer_complete"
    RECLAIMED = "reclaimed"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class UcxControlMessage:
    """Versioned control-plane message; payload bytes never travel here."""

    kind: str
    session_epoch: str
    operation_id: str
    slot_id: int
    slot_generation: int
    payload: dict[str, Any]
    protocol_version: int = 1

    def encode(self) -> bytes:
        raw = json.dumps(
            {
                "kind": self.kind,
                "session_epoch": self.session_epoch,
                "operation_id": self.operation_id,
                "slot_id": self.slot_id,
                "slot_generation": self.slot_generation,
                "payload": self.payload,
                "protocol_version": self.protocol_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return struct.pack("!I", len(raw)) + raw

    @classmethod
    def decode(cls, raw: bytes) -> "UcxControlMessage":
        message = json.loads(raw.decode("utf-8"))
        result = cls(
            kind=str(message["kind"]),
            session_epoch=str(message["session_epoch"]),
            operation_id=str(message["operation_id"]),
            slot_id=int(message["slot_id"]),
            slot_generation=int(message["slot_generation"]),
            payload=dict(message.get("payload", {})),
            protocol_version=int(message.get("protocol_version", 0)),
        )
        if result.protocol_version != 1:
            raise TransportError(
                f"unsupported UCX control protocol {result.protocol_version}"
            )
        if not result.session_epoch or not result.operation_id:
            raise TransportError("UCX control identifiers must be non-empty")
        return result


class UcxControlChannel:
    """Length-prefixed TCP control channel for descriptor and completion events."""

    _MAX_MESSAGE_BYTES = 4 * 1024 * 1024

    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection
        self._connection.settimeout(None)
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def connect(cls, host: str, port: int, timeout_s: float = 10.0) -> "UcxControlChannel":
        connection = socket.create_connection((host, int(port)), timeout=timeout_s)
        return cls(connection)

    @classmethod
    def accept(cls, listener: socket.socket) -> "UcxControlChannel":
        connection, _ = listener.accept()
        return cls(connection)

    def send(self, message: UcxControlMessage) -> None:
        encoded = message.encode()
        with self._lock:
            if self._closed:
                raise TransportError("UCX control channel is closed")
            self._connection.sendall(encoded)

    def recv(self, timeout_s: float | None = None) -> UcxControlMessage:
        previous_timeout = self._connection.gettimeout()
        self._connection.settimeout(timeout_s)
        try:
            header = self._recv_exact(4)
            if not header:
                raise EOFError("UCX control peer closed the connection")
            size = struct.unpack("!I", header)[0]
            if size <= 0 or size > self._MAX_MESSAGE_BYTES:
                raise TransportError(f"invalid UCX control message size {size}")
            return UcxControlMessage.decode(self._recv_exact(size))
        finally:
            self._connection.settimeout(previous_timeout)

    def _recv_exact(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = int(size)
        while remaining:
            chunk = self._connection.recv(remaining)
            if not chunk:
                raise EOFError("UCX control peer closed during a message")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._connection.shutdown(socket.SHUT_RDWR)
                self._connection.close()


@dataclass
class UcxTransferRecord:
    """Local lifecycle record; generation never substitutes for DMA draining."""

    session_epoch: str
    operation_id: str
    slot_id: int
    slot_generation: int
    state: UcxTransferState = UcxTransferState.FREE
    request: ctypes.c_void_p | None = None
    quarantined: bool = False

    def require(self, *states: UcxTransferState) -> None:
        if self.state not in states:
            allowed = ", ".join(state.value for state in states)
            raise TransportError(
                f"operation {self.operation_id} is {self.state.value}, expected {allowed}"
            )

    def advance(self, state: UcxTransferState) -> None:
        self.state = state


@dataclass(frozen=True)
class UcxRemoteRegionDescriptor:
    """Control-plane descriptor for one registered remote GPU allocation."""

    session_epoch: str
    worker_address: bytes
    remote_address: int
    capacity_bytes: int
    packed_rkey: bytes
    memory_type: str = "cuda"
    protocol_version: int = 1

    def to_payload(self) -> dict[str, Any]:
        if self.protocol_version != 1 or self.capacity_bytes <= 0:
            raise TransportError("invalid UCX remote region descriptor")
        return {
            "worker_address": base64.b64encode(self.worker_address).decode("ascii"),
            "remote_address": int(self.remote_address),
            "capacity_bytes": int(self.capacity_bytes),
            "packed_rkey": base64.b64encode(self.packed_rkey).decode("ascii"),
            "memory_type": self.memory_type,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_payload(
        cls, session_epoch: str, payload: dict[str, Any]
    ) -> "UcxRemoteRegionDescriptor":
        result = cls(
            session_epoch=session_epoch,
            worker_address=base64.b64decode(payload["worker_address"]),
            remote_address=int(payload["remote_address"]),
            capacity_bytes=int(payload["capacity_bytes"]),
            packed_rkey=base64.b64decode(payload["packed_rkey"]),
            memory_type=str(payload["memory_type"]),
            protocol_version=int(payload["protocol_version"]),
        )
        if result.protocol_version != 1 or result.capacity_bytes <= 0:
            raise TransportError("invalid UCX remote region descriptor")
        if result.memory_type != "cuda" or not result.packed_rkey:
            raise TransportError("UCX descriptor must reference registered CUDA memory")
        return result


class UcxSlotPool:
    """Bounded transfer credits; timed-out generations stay quarantined."""

    def __init__(self, capacity: int) -> None:
        if int(capacity) < 1:
            raise ValueError("slot capacity must be positive")
        self._condition = threading.Condition()
        self._free = list(range(int(capacity) - 1, -1, -1))
        self._generation = [0] * int(capacity)
        self._quarantined: set[int] = set()

    def acquire(self, timeout_s: float | None = None) -> tuple[int, int]:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            while not self._free:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("no free UCX transfer credits")
                self._condition.wait(remaining)
            slot_id = self._free.pop()
            self._generation[slot_id] += 1
            return slot_id, self._generation[slot_id]

    def release(self, slot_id: int, generation: int) -> None:
        with self._condition:
            self._validate_generation(slot_id, generation)
            if slot_id in self._quarantined or slot_id in self._free:
                raise TransportError("cannot release a free or quarantined UCX slot")
            self._free.append(slot_id)
            self._condition.notify()

    def quarantine(self, slot_id: int, generation: int) -> None:
        with self._condition:
            self._validate_generation(slot_id, generation)
            self._quarantined.add(slot_id)
            if slot_id in self._free:
                self._free.remove(slot_id)

    def reclaim_after_drain(self, slot_id: int, generation: int) -> None:
        with self._condition:
            self._validate_generation(slot_id, generation)
            if slot_id not in self._quarantined:
                raise TransportError("only quarantined slots can be reclaimed")
            self._quarantined.remove(slot_id)
            self._free.append(slot_id)
            self._condition.notify()

    def _validate_generation(self, slot_id: int, generation: int) -> None:
        if not 0 <= int(slot_id) < len(self._generation):
            raise TransportError(f"invalid UCX slot {slot_id}")
        if self._generation[int(slot_id)] != int(generation):
            raise TransportError("stale UCX slot generation")

class UcxWorkerProgressLoop:
    """Single-owner UCX progress loop for ``UCS_THREAD_MODE_SINGLE`` workers."""

    def __init__(self, endpoint: "UcxRdmaEndpoint", interval_s: float = 0.00005) -> None:
        self._endpoint = endpoint
        self._interval_s = float(interval_s)
        self._commands: queue.Queue[tuple[Callable[[], Any], Future[Any]] | None] = queue.Queue()
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="tetris-ucx-progress",
            daemon=False,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                command = self._commands.get(timeout=self._interval_s)
            except queue.Empty:
                if self._endpoint.active:
                    self._endpoint._progress_owned()
                continue
            if command is None:
                break
            callback, future = command
            if future.cancelled():
                continue
            try:
                future.set_result(callback())
            except BaseException as exc:
                future.set_exception(exc)
        if self._endpoint.active:
            self._endpoint._drain_owned()

    def submit(self, callback: Callable[[], Any]) -> Any:
        if self._stopping.is_set():
            raise TransportError("UCX progress loop is stopping")
        future: Future[Any] = Future()
        self._commands.put((callback, future))
        return future.result()

    def close(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        self._commands.put(None)
        self._thread.join(timeout=30.0)
        if self._thread.is_alive():
            raise TransportError("UCX progress loop did not drain before shutdown")


class UcxRdmaTransfer:
    """One-sided GPU PUT plus explicit remote visibility and consumer ack."""

    def __init__(
        self,
        endpoint: "UcxRdmaEndpoint",
        channel: UcxControlChannel,
        *,
        session_epoch: str,
        role: str,
        slot_count: int = 1,
        device: int = 0,
    ) -> None:
        if role not in {"producer", "consumer"}:
            raise ValueError("role must be producer or consumer")
        self.endpoint = endpoint
        self.channel = channel
        self.session_epoch = str(session_epoch)
        self.role = role
        self.device = int(device)
        self._records: dict[str, UcxTransferRecord] = {}
        self._slots = UcxSlotPool(slot_count)

    def send_region_descriptor(
        self, descriptor: UcxRemoteRegionDescriptor
    ) -> None:
        if descriptor.session_epoch != self.session_epoch:
            raise TransportError("UCX region descriptor session mismatch")
        self.channel.send(
            UcxControlMessage(
                "region_descriptor",
                self.session_epoch,
                "session:init",
                -1,
                0,
                descriptor.to_payload(),
            )
        )

    def receive_region_descriptor(
        self, timeout_s: float = 10.0
    ) -> UcxRemoteRegionDescriptor:
        message = self.channel.recv(timeout_s)
        if message.kind != "region_descriptor":
            raise TransportError(
                f"expected region_descriptor, received {message.kind}"
            )
        if message.session_epoch != self.session_epoch:
            raise TransportError("UCX region descriptor session mismatch")
        return UcxRemoteRegionDescriptor.from_payload(
            self.session_epoch, message.payload
        )

    def put(
        self,
        *,
        operation_id: str,
        source_address: int,
        payload_bytes: int,
        remote_address: int,
        remote_rkey: ctypes.c_void_p,
        metadata: dict[str, Any],
        producer_ready: Callable[[], None] | None = None,
        local_memh: ctypes.c_void_p | None = None,
        timeout_s: float = 30.0,
    ) -> UcxTransferRecord:
        if self.role != "producer":
            raise TransportError("only the producer may submit a UCX PUT")
        slot_id, generation = self._slots.acquire(timeout_s)
        record = UcxTransferRecord(
            self.session_epoch, str(operation_id), int(slot_id), generation
        )
        record.advance(UcxTransferState.PUT_SUBMITTED)
        self._records[record.operation_id] = record
        deadline = time.monotonic() + float(timeout_s)
        try:
            if producer_ready is None:
                raise TransportError("producer_ready callback is required")
            producer_ready()
            record.request = self.endpoint.put(
                source_address,
                payload_bytes,
                remote_address,
                remote_rkey,
                local_memh,
            )
            while not self.endpoint.request_done(record.request):
                if time.monotonic() >= deadline:
                    raise TimeoutError("UCX PUT local completion timed out")
                time.sleep(0.0001)
            record.request = None
            record.advance(UcxTransferState.PUT_LOCAL_COMPLETE)
            self.endpoint.flush()
            self.channel.send(
                UcxControlMessage(
                    "remote_write_complete",
                    self.session_epoch,
                    record.operation_id,
                    record.slot_id,
                    record.slot_generation,
                    {"bytes": int(payload_bytes), **metadata},
                )
            )
            visibility = self._recv_matching(record, deadline, "remote_visible")
            record.advance(UcxTransferState.REMOTE_VISIBLE)
            ack = self._recv_matching(record, deadline, "consumer_complete")
            record.advance(UcxTransferState.CONSUMER_COMPLETE)
            record.advance(UcxTransferState.RECLAIMED)
            self._slots.release(record.slot_id, record.slot_generation)
            return record
        except BaseException:
            record.quarantined = True
            record.advance(UcxTransferState.QUARANTINED)
            self._slots.quarantine(record.slot_id, record.slot_generation)
            try:
                self.endpoint.flush()
            except BaseException:
                pass
            raise

    def drain_quarantined(self) -> None:
        """Drain local DMA while retaining slots isolated from reuse."""
        self.endpoint.flush()
        for record in self._records.values():
            if record.quarantined:
                record.require(UcxTransferState.QUARANTINED)
        raise TransportError(
            "quarantined UCX slots remain isolated until consumer_complete or "
            "session teardown proves no late DMA can target them"
        )

    def receive_and_consume(
        self,
        *,
        consumer: Callable[[UcxControlMessage], Any],
        wait_consumer: Callable[[Any], None] | None = None,
        timeout_s: float = 30.0,
    ) -> Any:
        if self.role != "consumer":
            raise TransportError("only the consumer may receive a PUT notification")
        deadline = time.monotonic() + float(timeout_s)
        notice = self.channel.recv(max(0.0, deadline - time.monotonic()))
        if notice.kind != "remote_write_complete":
            raise TransportError(f"unexpected remote-write message: {notice.kind}")
        if notice.session_epoch != self.session_epoch:
            raise TransportError("stale UCX session epoch")
        record = UcxTransferRecord(
            notice.session_epoch,
            notice.operation_id,
            notice.slot_id,
            notice.slot_generation,
            UcxTransferState.REMOTE_WRITE_COMPLETE,
        )
        self._records[record.operation_id] = record
        try:
            if wait_consumer is None:
                raise TransportError("wait_consumer callback is required")
            self.endpoint.flush_remote_writes(self.device)
            record.advance(UcxTransferState.REMOTE_VISIBLE)
            self.channel.send(
                UcxControlMessage(
                    "remote_visible",
                    record.session_epoch,
                    record.operation_id,
                    record.slot_id,
                    record.slot_generation,
                    {"cuda_visibility": "cudaDeviceFlushGPUDirectRDMAWrites"},
                )
            )
            record.advance(UcxTransferState.CONSUMER_IN_FLIGHT)
            result = consumer(notice)
            wait_consumer(result)
            record.advance(UcxTransferState.CONSUMER_COMPLETE)
            self.channel.send(
                UcxControlMessage(
                    "consumer_complete",
                    record.session_epoch,
                    record.operation_id,
                    record.slot_id,
                    record.slot_generation,
                    {"status": "ok"},
                )
            )
            record.advance(UcxTransferState.RECLAIMED)
            return result
        except BaseException:
            record.quarantined = True
            record.advance(UcxTransferState.QUARANTINED)
            raise
        finally:
            if time.monotonic() > deadline:
                record.quarantined = True
                record.advance(UcxTransferState.QUARANTINED)

    def _recv_matching(
        self,
        record: UcxTransferRecord,
        deadline: float,
        expected_kind: str,
    ) -> UcxControlMessage:
        while time.monotonic() < deadline:
            message = self.channel.recv(max(0.0, deadline - time.monotonic()))
            if message.session_epoch != record.session_epoch:
                continue
            if message.operation_id != record.operation_id:
                continue
            if (
                message.slot_id != record.slot_id
                or message.slot_generation != record.slot_generation
            ):
                raise TransportError("UCX completion slot generation mismatch")
            if message.kind != expected_kind:
                raise TransportError(
                    f"unexpected UCX completion {message.kind}, expected {expected_kind}"
                )
            return message
        raise TimeoutError("consumer completion acknowledgment timed out")


def inspect_ucx_capability() -> UcxCapability:
    """Inspect the installed UCX without changing the environment."""
    ucx_info = shutil.which("ucx_info")
    version = "unknown"
    transports = ""
    if ucx_info:
        try:
            version = subprocess.run(
                [ucx_info, "-v"], capture_output=True, text=True, check=False
            ).stdout.strip()
            transports = subprocess.run(
                [ucx_info, "-d"], capture_output=True, text=True, check=False
            ).stdout
        except OSError:
            pass
    lib_dir = Path(os.environ.get("UCX_INSTALL_PATH", "/opt/hpcx/ucx")) / "lib"
    cuda_plugins = list(lib_dir.glob("ucx/libuct_cuda*.so*"))
    ib_visible = Path("/dev/infiniband").exists()
    ib_transport = any(
        token in transports for token in ("rc_mlx5", "dc_mlx5", "ud_mlx5")
    )
    if not ib_visible or not ib_transport:
        reason = "UCX has no verified IB/RDMA data plane in this environment"
    else:
        reason = "UCX IB transport and device are visible; registration still requires a data-plane test"
    return UcxCapability(
        ucx_version=version,
        ib_device_visible=ib_visible,
        ib_transport_visible=ib_transport,
        cuda_plugin_visible=any("cuda" in path.name for path in cuda_plugins),
        gdrcopy_plugin_visible=any("gdrcopy" in path.name for path in cuda_plugins),
        reason=reason,
    )


class UcxRdmaEndpoint:
    """Actual UCX RMA endpoint wrapper; never falls back silently."""

    def __init__(self, bridge_path: str | None = None) -> None:
        self.capability = inspect_ucx_capability()
        if not self.capability.ib_device_visible or not self.capability.ib_transport_visible:
            raise TransportUnavailableError(self.capability.reason)
        path = bridge_path or os.environ.get("TETRIS_LORA_UCX_BRIDGE")
        if not path:
            raise TransportUnavailableError(
                "TETRIS_LORA_UCX_BRIDGE is not configured; build the reviewed UCX bridge first"
            )
        self._lib = ctypes.CDLL(path)
        self._configure_abi()
        self._context = ctypes.c_void_p()
        status = self._lib.tetris_ucx_open(ctypes.byref(self._context))
        if status != 0:
            raise TransportError(f"ucp_init/worker_create failed with status {status}")
        self._closed = False
        self._active = True
        self._requests: dict[int, ctypes.c_void_p] = {}
        self._request_lock = threading.Lock()
        self._memories: dict[int, ctypes.c_void_p] = {}
        self._rkeys: dict[int, ctypes.c_void_p] = {}
        self._progress_loop = UcxWorkerProgressLoop(self)

    def _configure_abi(self) -> None:
        self._lib.tetris_ucx_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.tetris_ucx_open.restype = ctypes.c_int
        self._lib.tetris_ucx_close.argtypes = [ctypes.c_void_p]
        self._lib.tetris_ucx_close.restype = ctypes.c_int
        self._lib.tetris_ucx_progress.argtypes = [ctypes.c_void_p]
        self._lib.tetris_ucx_progress.restype = ctypes.c_int
        self._lib.tetris_ucx_flush.argtypes = [ctypes.c_void_p]
        self._lib.tetris_ucx_flush.restype = ctypes.c_int
        self._lib.tetris_ucx_flush_remote_writes.argtypes = [ctypes.c_int]
        self._lib.tetris_ucx_flush_remote_writes.restype = ctypes.c_int
        self._lib.tetris_ucx_worker_address.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._lib.tetris_ucx_worker_address.restype = ctypes.c_int
        self._lib.tetris_ucx_connect.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._lib.tetris_ucx_connect.restype = ctypes.c_int
        self._lib.tetris_ucx_map_cuda.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.tetris_ucx_map_cuda.restype = ctypes.c_int
        self._lib.tetris_ucx_unmap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.tetris_ucx_unmap.restype = ctypes.c_int
        self._lib.tetris_ucx_pack_rkey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._lib.tetris_ucx_pack_rkey.restype = ctypes.c_int
        self._lib.tetris_ucx_unpack_rkey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.tetris_ucx_unpack_rkey.restype = ctypes.c_int
        self._lib.tetris_ucx_destroy_rkey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.tetris_ucx_destroy_rkey.restype = ctypes.c_int
        self._lib.tetris_ucx_put.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.tetris_ucx_put.restype = ctypes.c_int
        self._lib.tetris_ucx_request_done.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.tetris_ucx_request_done.restype = ctypes.c_int

    @_progress_owned
    def worker_address(self) -> bytes:
        buffer = ctypes.create_string_buffer(64 * 1024)
        size = ctypes.c_size_t()
        status = int(
            self._lib.tetris_ucx_worker_address(
                self._context, buffer, len(buffer), ctypes.byref(size)
            )
        )
        if status != 0:
            raise TransportError(f"UCX worker address failed: {status}")
        return bytes(buffer.raw[: size.value])

    @_progress_owned
    def connect(self, address: bytes) -> None:
        raw = ctypes.create_string_buffer(address)
        status = int(
            self._lib.tetris_ucx_connect(
                self._context, raw, len(address)
            )
        )
        if status != 0:
            raise TransportError(f"UCX endpoint connect failed: {status}")

    @_progress_owned
    def register_cuda(self, address: int, length: int) -> ctypes.c_void_p:
        memh = ctypes.c_void_p()
        status = int(
            self._lib.tetris_ucx_map_cuda(
                self._context,
                ctypes.c_void_p(address),
                ctypes.c_size_t(length),
                ctypes.byref(memh),
            )
        )
        if status != 0:
            raise TransportError(f"UCX CUDA memory registration failed: {status}")
        self._memories[int(memh.value)] = memh
        return memh

    @_progress_owned
    def unregister_cuda(self, memh: ctypes.c_void_p) -> None:
        status = int(self._lib.tetris_ucx_unmap(self._context, memh))
        if status != 0:
            raise TransportError(f"UCX CUDA memory unmap failed: {status}")
        self._memories.pop(int(memh.value), None)

    @_progress_owned
    def pack_rkey(self, memh: ctypes.c_void_p) -> bytes:
        buffer = ctypes.create_string_buffer(64 * 1024)
        size = ctypes.c_size_t()
        status = int(
            self._lib.tetris_ucx_pack_rkey(
                self._context,
                memh,
                buffer,
                len(buffer),
                ctypes.byref(size),
            )
        )
        if status != 0:
            raise TransportError(f"UCX rkey pack failed: {status}")
        return bytes(buffer.raw[: size.value])

    @_progress_owned
    def unpack_rkey(self, packed: bytes) -> ctypes.c_void_p:
        raw = ctypes.create_string_buffer(packed)
        rkey = ctypes.c_void_p()
        status = int(
            self._lib.tetris_ucx_unpack_rkey(
                self._context, raw, len(packed), ctypes.byref(rkey)
            )
        )
        if status != 0:
            raise TransportError(f"UCX rkey unpack failed: {status}")
        self._rkeys[int(rkey.value)] = rkey
        return rkey

    @_progress_owned
    def destroy_rkey(self, rkey: ctypes.c_void_p) -> None:
        status = int(self._lib.tetris_ucx_destroy_rkey(self._context, rkey))
        if status != 0:
            raise TransportError(f"UCX rkey destroy failed: {status}")
        self._rkeys.pop(int(rkey.value), None)

    @_progress_owned
    def put(
        self,
        address: int,
        length: int,
        remote_address: int,
        rkey: ctypes.c_void_p,
        local_memh: ctypes.c_void_p | None = None,
    ) -> ctypes.c_void_p | None:
        request = ctypes.c_void_p()
        status = int(
            self._lib.tetris_ucx_put(
                self._context,
                ctypes.c_void_p(address),
                ctypes.c_size_t(length),
                ctypes.c_uint64(remote_address),
                rkey,
                local_memh or ctypes.c_void_p(),
                ctypes.byref(request),
            )
        )
        if status != 0:
            raise TransportError(f"UCX PUT failed: {status}")
        return None if not request.value else request

    @_progress_owned
    def request_done(self, request: ctypes.c_void_p | None) -> bool:
        value = request if request is not None else ctypes.c_void_p()
        status = int(self._lib.tetris_ucx_request_done(self._context, value))
        if status == 0:
            return False
        if status != 1:
            raise TransportError(f"UCX request failed: {status}")
        return True

    @_progress_owned
    def progress(self) -> int:
        if self._closed:
            raise TransportError("UCX endpoint is closed")
        return int(self._lib.tetris_ucx_progress(self._context))

    @_progress_owned
    def flush(self) -> None:
        if self._closed:
            return
        status = int(self._lib.tetris_ucx_flush(self._context))
        if status != 0:
            raise TransportError(f"UCX worker flush failed with status {status}")

    def flush_remote_writes(self, device: int) -> None:
        status = int(self._lib.tetris_ucx_flush_remote_writes(int(device)))
        if status != 0:
            raise TransportError(
                "CUDA GPUDirect RDMA remote-write visibility flush failed: "
                f"cuda_status={status}"
            )

    def _progress_owned(self) -> None:
        if self._closed:
            return
        status = int(self._lib.tetris_ucx_progress(self._context))
        if status < 0:
            raise TransportError(f"UCX progress failed with status {status}")

    @property
    def active(self) -> bool:
        return self._active

    def _drain_owned(self) -> None:
        if not self._closed:
            self.flush()

    def _close_owned(self) -> None:
        if self._closed:
            return
        status = int(self._lib.tetris_ucx_close(self._context))
        self._closed = True
        self._active = False
        self._rkeys.clear()
        self._memories.clear()
        self._context = ctypes.c_void_p()
        if status != 0:
            raise TransportError(f"UCX close/drain failed with status {status}")

    def close(self) -> None:
        if not self._closed:
            self._progress_loop.submit(self._close_owned)
            self._progress_loop.close()


def build_command() -> list[str]:
    """Return the reviewed, non-executed CUDA 12/UCX 1.15 build command."""
    root = Path(__file__).with_name("ucx_bridge")
    return [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        f"-I{os.environ.get('UCX_INSTALL_PATH', '/opt/hpcx/ucx')}/include",
        str(root / "tetris_ucx_bridge.c"),
        "-o",
        str(root / "libtetris_ucx.so"),
        f"-L{os.environ.get('UCX_INSTALL_PATH', '/opt/hpcx/ucx')}/lib",
        "-lucp",
        "-lucs",
        "-luct",
        "-lcudart",
    ]


__all__ = [
    "UcxCapability",
    "UcxControlChannel",
    "UcxControlMessage",
    "UcxRdmaEndpoint",
    "UcxRdmaTransfer",
    "UcxRemoteRegionDescriptor",
    "UcxSlotPool",
    "UcxTransferRecord",
    "UcxTransferState",
    "UcxWorkerProgressLoop",
    "build_command",
    "inspect_ucx_capability",
]
