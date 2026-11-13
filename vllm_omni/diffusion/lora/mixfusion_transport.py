"""Transport contracts used by the MixFusion LoRA runtime."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch


class TransportError(RuntimeError):
    """Base error for MixFusion transport failures."""


class TransportUnavailableError(TransportError):
    """Raised when a requested transport is not available."""


@dataclass(frozen=True)
class TransportDescriptor:
    """Control-plane metadata for one activation or residual transfer."""

    request_id: str
    step_id: int
    block_id: str
    tile_id: int | None
    shape: tuple[int, ...]
    dtype: str
    adapter_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    payload_bytes: int
    buffer_token: str
    completion_sequence: int
    protocol_version: int = 1

    def validate(self) -> None:
        """Validate descriptor fields before a data-plane operation."""
        if self.protocol_version != 1:
            raise ValueError(
                f"unsupported transport protocol: {self.protocol_version}"
            )
        if not self.request_id or not self.block_id or not self.buffer_token:
            raise ValueError("transport descriptor identifiers must be non-empty")
        if any(int(dimension) < 0 for dimension in self.shape):
            raise ValueError("transport descriptor shape must be non-negative")
        if not self.dtype:
            raise ValueError("transport descriptor dtype must be non-empty")
        if len(self.adapter_ids) != len(self.ranks):
            raise ValueError("adapter and rank metadata lengths do not match")
        if any(int(rank) < 0 for rank in self.ranks):
            raise ValueError("transport descriptor ranks must be non-negative")
        if int(self.payload_bytes) < 0:
            raise ValueError("transport payload size must be non-negative")
        if int(self.completion_sequence) < 0:
            raise ValueError("transport completion sequence must be non-negative")


@dataclass(frozen=True)
class TransportRegistration:
    """Result of registering a tensor with an endpoint."""

    buffer_token: str
    device: str
    bytes: int
    registered_gpu_buffer: bool
    payload_not_host_staged: bool


class TransportRequest:
    """Completion handle returned by asynchronous endpoint operations."""

    def __init__(self, descriptor: TransportDescriptor) -> None:
        self.descriptor = descriptor
        self._done = False
        self._error: BaseException | None = None

    @property
    def done(self) -> bool:
        """Return whether the operation has completed."""
        return self._done

    def _complete(self, error: BaseException | None = None) -> None:
        self._error = error
        self._done = True

    def wait(self, timeout: float | None = None) -> None:
        """Wait for completion and raise the recorded transport error."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._done:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "transport request timed out at completion sequence "
                    f"{self.descriptor.completion_sequence}"
                )
            time.sleep(0.0005)
        if self._error is not None:
            raise TransportError(str(self._error)) from self._error


class TransportEndpoint(ABC):
    """Asynchronous activation transport interface."""

    @abstractmethod
    def connect(self) -> None:
        """Connect the endpoint."""

    @abstractmethod
    def close(self) -> None:
        """Close the endpoint and fail pending work."""

    @abstractmethod
    def register_buffer(
        self, buffer_token: str, device_buffer: torch.Tensor
    ) -> TransportRegistration:
        """Register a reusable activation or residual buffer."""

    @abstractmethod
    def deregister_buffer(self, buffer_token: str) -> None:
        """Deregister a previously registered buffer."""

    @abstractmethod
    def send_async(
        self,
        descriptor: TransportDescriptor,
        device_buffer: torch.Tensor,
        completion_event: Any | None = None,
    ) -> TransportRequest:
        """Submit an asynchronous send."""

    @abstractmethod
    def recv_async(
        self,
        descriptor: TransportDescriptor,
        device_buffer: torch.Tensor,
        completion_event: Any | None = None,
    ) -> TransportRequest:
        """Submit an asynchronous receive."""

    @abstractmethod
    def poll(self) -> int:
        """Progress pending operations and return the completed count."""

    def wait(self, request: TransportRequest, timeout: float | None = None) -> None:
        """Progress and wait for one request."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not request.done:
            self.poll()
            if request.done:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "transport request timed out at completion sequence "
                    f"{request.descriptor.completion_sequence}"
                )
            time.sleep(0.0005)
        request.wait()


@dataclass
class _PendingTransfer:
    descriptor: TransportDescriptor
    request: TransportRequest
    payload: torch.Tensor | None = None
    destination: torch.Tensor | None = None
    event: Any | None = None


class LoopbackTransportEndpoint(TransportEndpoint):
    """Same-process endpoint for validating descriptor and lifetime semantics."""

    def __init__(self) -> None:
        self._connected = False
        self._buffers: dict[str, torch.Tensor] = {}
        self._sends: dict[int, _PendingTransfer] = {}
        self._receives: dict[int, _PendingTransfer] = {}
        self._closed_requests: deque[TransportRequest] = deque()

    def connect(self) -> None:
        """Mark the loopback endpoint connected."""
        if self._connected:
            return
        self._connected = True

    def close(self) -> None:
        """Fail pending operations and release registrations."""
        error = TransportError("transport endpoint closed")
        for pending in (*self._sends.values(), *self._receives.values()):
            pending.request._complete(error)
        self._sends.clear()
        self._receives.clear()
        self._buffers.clear()
        self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise TransportError("transport endpoint is not connected")

    @staticmethod
    def _validate_tensor(
        descriptor: TransportDescriptor, device_buffer: torch.Tensor
    ) -> None:
        descriptor.validate()
        if tuple(int(value) for value in device_buffer.shape) != descriptor.shape:
            raise ValueError("transport buffer shape does not match descriptor")
        if str(device_buffer.dtype) != descriptor.dtype:
            raise ValueError("transport buffer dtype does not match descriptor")
        if device_buffer.numel() * device_buffer.element_size() != int(
            descriptor.payload_bytes
        ):
            raise ValueError("transport payload size does not match tensor")

    def register_buffer(
        self, buffer_token: str, device_buffer: torch.Tensor
    ) -> TransportRegistration:
        """Register a reusable buffer without claiming RDMA capability."""
        self._require_connected()
        if not buffer_token:
            raise ValueError("buffer token must be non-empty")
        if buffer_token in self._buffers:
            raise ValueError(f"buffer token is already registered: {buffer_token}")
        self._buffers[buffer_token] = device_buffer
        is_gpu = device_buffer.is_cuda
        return TransportRegistration(
            buffer_token=buffer_token,
            device=str(device_buffer.device),
            bytes=int(device_buffer.numel() * device_buffer.element_size()),
            registered_gpu_buffer=is_gpu,
            payload_not_host_staged=is_gpu,
        )

    def deregister_buffer(self, buffer_token: str) -> None:
        """Deregister a reusable buffer."""
        self._require_connected()
        if buffer_token not in self._buffers:
            raise KeyError(buffer_token)
        del self._buffers[buffer_token]

    def send_async(
        self,
        descriptor: TransportDescriptor,
        device_buffer: torch.Tensor,
        completion_event: Any | None = None,
    ) -> TransportRequest:
        """Capture a device-local payload for a paired loopback receive."""
        self._require_connected()
        self._validate_tensor(descriptor, device_buffer)
        sequence = int(descriptor.completion_sequence)
        if sequence in self._sends:
            raise TransportError(f"duplicate send sequence: {sequence}")
        request = TransportRequest(descriptor)
        self._sends[sequence] = _PendingTransfer(
            descriptor=descriptor,
            request=request,
            payload=device_buffer.detach().clone(),
            event=completion_event,
        )
        return request

    def recv_async(
        self,
        descriptor: TransportDescriptor,
        device_buffer: torch.Tensor,
        completion_event: Any | None = None,
    ) -> TransportRequest:
        """Pair a destination buffer with a previously submitted send."""
        self._require_connected()
        self._validate_tensor(descriptor, device_buffer)
        sequence = int(descriptor.completion_sequence)
        if sequence in self._receives:
            raise TransportError(f"duplicate receive sequence: {sequence}")
        request = TransportRequest(descriptor)
        self._receives[sequence] = _PendingTransfer(
            descriptor=descriptor,
            request=request,
            destination=device_buffer,
            event=completion_event,
        )
        return request

    @staticmethod
    def _event_ready(event: Any | None) -> bool:
        if event is None:
            return True
        query = getattr(event, "query", None)
        return bool(query()) if callable(query) else True

    @staticmethod
    def _compatible(
        send: TransportDescriptor, receive: TransportDescriptor
    ) -> bool:
        return (
            send.request_id == receive.request_id
            and send.step_id == receive.step_id
            and send.block_id == receive.block_id
            and send.tile_id == receive.tile_id
            and send.shape == receive.shape
            and send.dtype == receive.dtype
            and send.adapter_ids == receive.adapter_ids
            and send.ranks == receive.ranks
            and send.payload_bytes == receive.payload_bytes
        )

    def poll(self) -> int:
        """Complete ready loopback transfers."""
        completed = 0
        for sequence in tuple(self._sends):
            send = self._sends[sequence]
            receive = self._receives.get(sequence)
            if not self._event_ready(send.event) or (
                receive is not None and not self._event_ready(receive.event)
            ):
                continue
            if receive is None:
                continue
            if not self._compatible(send.descriptor, receive.descriptor):
                error = TransportError(
                    "send and receive descriptors do not describe the same work"
                )
                send.request._complete(error)
                receive.request._complete(error)
                del self._sends[sequence]
                del self._receives[sequence]
                completed += 2
                continue
            try:
                assert send.payload is not None
                assert receive.destination is not None
                receive.destination.copy_(send.payload)
                send.request._complete()
                receive.request._complete()
            except BaseException as exc:
                send.request._complete(exc)
                receive.request._complete(exc)
            del self._sends[sequence]
            del self._receives[sequence]
            completed += 2
        return completed


class P2PTransportEndpoint(LoopbackTransportEndpoint):
    """Local P2P contract with loopback semantics for the first implementation."""


@dataclass(frozen=True)
class TransportCapability:
    """Observed capability state for a requested transport."""

    transport: str
    available: bool
    registered_gpu_buffer: bool
    payload_not_host_staged: bool
    reason: str
    hardware_data_plane: bool = False


def inspect_transport_capability(transport: str) -> TransportCapability:
    """Inspect transport capability without claiming unavailable hardware."""
    normalized = transport.strip().lower()
    if normalized in {"p2p", "loopback"}:
        return TransportCapability(
            normalized,
            True,
            False,
            False,
            "loopback contract only; hardware P2P is not implemented",
            False,
        )
    if normalized == "tcp":
        return TransportCapability(
            normalized,
            True,
            False,
            False,
            "existing TCP wire fallback",
            False,
        )
    if normalized == "rdma":
        enabled = os.environ.get("VLLM_OMNI_GPUDIRECT_RDMA", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return TransportCapability(
                normalized,
                False,
                False,
                False,
                "GPUDirect RDMA capability was not declared",
                False,
            )
        return TransportCapability(
            normalized,
            False,
            False,
            False,
            "RDMA endpoint is not implemented in this runtime",
            False,
        )
    raise ValueError(f"unsupported transport: {transport!r}")


def create_transport_endpoint(transport: str) -> TransportEndpoint:
    """Create only transports whose data-plane semantics are implemented."""
    normalized = transport.strip().lower()
    if normalized in {"p2p", "loopback"}:
        return P2PTransportEndpoint()
    capability = inspect_transport_capability(normalized)
    raise TransportUnavailableError(
        f"transport {normalized!r} is unavailable: {capability.reason}"
    )


__all__ = [
    "LoopbackTransportEndpoint",
    "P2PTransportEndpoint",
    "TransportCapability",
    "TransportDescriptor",
    "TransportEndpoint",
    "TransportError",
    "TransportRegistration",
    "TransportRequest",
    "TransportUnavailableError",
    "create_transport_endpoint",
    "inspect_transport_capability",
]
