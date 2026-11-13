"""Cross-host Base↔LoRA X/Δ runtime (NIXL dataplane, mock protocol).

This is not PD disaggregation. Tetris moves per-layer activations X and
residuals Δ. One backend is wired: NIXL (register, metadata handshake,
prepared descriptor lists, ``make_prepped_xfer`` + ``transfer`` or
``initialize_xfer`` + ``transfer``, poll, release). UCX is fail-closed as
``IMPLEMENTATION-INCOMPLETE`` this round.

Status labels:

* ``DEPENDENCY-BLOCKED`` — NIXL Python API cannot be imported or constructed.
* ``HARDWARE-BLOCKED`` — NIXL imports, but no IB device (or the agent cannot
  open on the visible device). Not a fallback to mock or TCP.
* ``IMPLEMENTATION-INCOMPLETE`` — UCX runtime is not wired this round.
* ``WIRED`` — NIXL submit path is bound. Dual-host completion is not claimed
  as ``VERIFIED``.
* ``VERIFIED`` — reserved for a transfer whose NIXL status reached ``DONE``
  on a non-injected agent. This process does not self-assign it.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable

from .tetris_lora_transport import (
    TensorDescriptor,
    TransportError,
    TransportUnavailableError,
)

# Line-level provenance copied from vLLM NIXL connector patterns.
# Paths are relative to the vllm tree root.
NIXL_LIFECYCLE_PROVENANCE: tuple[tuple[str, str], ...] = (
    (
        "register_memory",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:1003-1005",
    ),
    (
        "register_memory (alt)",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:1225-1227",
    ),
    (
        "get_agent_metadata",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:1015",
    ),
    (
        "prep_xfer_dlist",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:1486-1488",
    ),
    (
        "prep_xfer_dlist (remote)",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:1691-1694",
    ),
    (
        "make_prepped_xfer + transfer (READ)",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:322-332",
    ),
    (
        "make_prepped_xfer + transfer (WRITE)",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py:669-677",
    ),
    (
        "check_xfer_state + release",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2223-2228",
    ),
    (
        "deregister_memory",
        "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:2603-2604",
    ),
    (
        "NixlWrapper lazy import",
        "vllm/distributed/nixl_utils.py:50-77",
    ),
)

UCX_LIFECYCLE_PROVENANCE: tuple[tuple[str, str], ...] = (
    (
        "inspect_ucx_capability",
        "vllm_omni/diffusion/lora/tetris_lora_ucx.py:564-596",
    ),
    (
        "UcxRdmaEndpoint open/map/rma",
        "vllm_omni/diffusion/lora/tetris_lora_ucx.py:599-620",
    ),
)

# CPU control metadata (TensorDescriptor) vs GPU data-plane payload bytes.
METADATA_VS_GPU_PAYLOAD: tuple[tuple[str, str, str], ...] = (
    ("session_epoch", "cpu_metadata", "session fencing"),
    ("operation_id", "cpu_metadata", "async ticket key / poll handle"),
    ("request_id / batch_id / request_ids", "cpu_metadata", "request composition"),
    ("denoise_step", "cpu_metadata", "DiT step index"),
    ("layer_name / invocation_id", "cpu_metadata", "per-layer X/Δ target"),
    ("slot_id / slot_generation", "cpu_metadata", "buffer slot lifecycle"),
    ("shape / stride / dtype / layout", "cpu_metadata", "tensor view contract"),
    ("payload_bytes / offset / buffer_capacity", "cpu_metadata", "byte accounting"),
    ("buffer_token", "cpu_metadata", "registered MR key"),
    ("adapter_id / adapter_version / rank / scale", "cpu_metadata", "LoRA binding"),
    ("branch_index / output_offset / completion_sequence", "cpu_metadata", "ordering"),
    (
        "request_composition / request_ranks / row_layout / output_slices",
        "cpu_metadata",
        "mixed-rank packing",
    ),
    ("activation X bytes", "gpu_payload", "Base→LoRA layer input"),
    ("residual Δ bytes", "gpu_payload", "LoRA→Base addend"),
)

# NIXL Python API status strings (nixl_agent.transfer / check_xfer_state).
_NIXL_DONE = "DONE"
_NIXL_PROC = "PROC"
_NIXL_ERR = "ERR"
_ACCEPTED_LAYOUTS = frozenset({"row_major", "contiguous"})


class DataplaneStatus(str, Enum):
    """Why a non-mock dataplane is or is not usable."""

    VERIFIED = "VERIFIED"
    WIRED = "WIRED"
    IMPLEMENTATION_INCOMPLETE = "IMPLEMENTATION-INCOMPLETE"
    DEPENDENCY_BLOCKED = "DEPENDENCY-BLOCKED"
    HARDWARE_BLOCKED = "HARDWARE-BLOCKED"


_BLOCKING_STATUSES = frozenset(
    {
        DataplaneStatus.IMPLEMENTATION_INCOMPLETE,
        DataplaneStatus.DEPENDENCY_BLOCKED,
        DataplaneStatus.HARDWARE_BLOCKED,
    }
)


class RdmaMode(str, Enum):
    """Runtime data-plane selection."""

    MOCK = "mock"
    NIXL = "nixl"
    UCX = "ucx"


class TransferKind(str, Enum):
    X = "x"
    DELTA = "delta"


@dataclass(frozen=True)
class MemoryRegion:
    """Session-scoped registered buffer (no per-op reg_mr)."""

    token: str
    kind: TransferKind
    payload_bytes: int
    device: str
    registered: bool
    nixl_desc: Any | None = None
    data_ptr: int | None = None
    xfer_descs: Any | None = None
    local_side: Any | None = None


@dataclass
class PendingTransfer:
    """Outstanding async transfer ticket."""

    operation_id: str
    kind: TransferKind
    descriptor: TensorDescriptor
    submitted_ns: int
    done: bool = False
    error: BaseException | None = None
    nixl_handle: Any | None = None
    nixl_state: str | None = None
    direction: str = "send"
    released: bool = False
    mock_payload: Any | None = None


@dataclass(frozen=True)
class RdmaCapability:
    """Evidence gate before enabling a non-mock data plane."""

    mode: RdmaMode
    nixl_importable: bool
    ucx_endpoint_available: bool
    ib_device_visible: bool
    reason: str
    hardware_blocked: bool
    provenance: str
    status: DataplaneStatus
    nixl_agent_bound: bool = False


@dataclass(frozen=True)
class DenoiseStepPlan:
    """All-targets semantic schedule for one denoise step (MOCK-capable)."""

    targets: tuple[str, ...]
    denoise_step: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    payload_bytes: int
    adapter_id: str = "adapter"
    adapter_version: str = "v1"
    rank: int = 8
    scale: float = 1.0
    request_id: str = "req-all-targets"
    batch_id: str = "batch-1"


@dataclass
class DenoiseStepResult:
    """Observed completions from one MOCK/all-targets denoise schedule."""

    denoise_step: int
    targets: tuple[str, ...]
    x_send_ops: list[str] = field(default_factory=list)
    delta_recv_ops: list[str] = field(default_factory=list)
    completion_sequence: int = 0


class NixlQuarantineError(TransportError):
    """Registration cleanup failed; the buffer may still be in DMA."""


def _provenance(pairs: tuple[tuple[str, str], ...]) -> str:
    return "; ".join(f"{name}@{path}" for name, path in pairs)


def _ib_visible() -> bool:
    """True when sysfs or /dev lists at least one IB device node."""
    for path in ("/sys/class/infiniband", "/dev/infiniband"):
        if not os.path.isdir(path):
            continue
        try:
            with os.scandir(path) as entries:
                if any(True for _ in entries):
                    return True
        except OSError:
            continue
    return False


def _try_import_nixl() -> tuple[bool, Any | None, Any | None, str]:
    """Import nixl agent factory when present.

    Returns:
        Tuple of import-ok, agent class, config class, and a short reason.
        Import success is not hardware evidence.
    """
    try:
        from nixl._api import nixl_agent as nixl_agent_cls  # type: ignore
        from nixl._api import nixl_agent_config as nixl_config_cls  # type: ignore

        return True, nixl_agent_cls, nixl_config_cls, "nixl._api.nixl_agent importable"
    except Exception:
        pass
    try:
        import nixl  # type: ignore

        agent_cls = getattr(nixl, "nixl_agent", None) or getattr(
            nixl, "NixlAgent", None
        )
        config_cls = getattr(nixl, "nixl_agent_config", None)
        if agent_cls is not None:
            return True, agent_cls, config_cls, "nixl module exposes agent factory"
        return True, None, config_cls, "nixl importable but agent factory missing"
    except Exception:
        return False, None, None, "NIXL python module unavailable"


def _probe_ucx() -> tuple[bool, bool, str]:
    """Return (endpoint_available, ib_visible, reason) without opening endpoints."""
    ib_visible = _ib_visible()
    try:
        from .tetris_lora_ucx import inspect_ucx_capability

        cap = inspect_ucx_capability()
        ucx_ok = bool(
            getattr(cap, "ib_device_visible", False)
            and getattr(cap, "ib_transport_visible", False)
        )
        return ucx_ok, ib_visible, str(getattr(cap, "reason", "ucx probe"))
    except Exception as exc:
        return False, ib_visible, f"UCX probe failed: {exc}"


def _capability(
    *,
    mode: RdmaMode,
    status: DataplaneStatus,
    reason: str,
    provenance: str,
    nixl_importable: bool,
    ucx_endpoint_available: bool,
    ib_device_visible: bool,
    nixl_agent_bound: bool = False,
) -> RdmaCapability:
    return RdmaCapability(
        mode=mode,
        nixl_importable=nixl_importable,
        ucx_endpoint_available=ucx_endpoint_available,
        ib_device_visible=ib_device_visible,
        reason=reason,
        hardware_blocked=status == DataplaneStatus.HARDWARE_BLOCKED,
        provenance=provenance,
        status=status,
        nixl_agent_bound=nixl_agent_bound,
    )


def probe_rdma_capability(mode: RdmaMode | str | None = None) -> RdmaCapability:
    """Fail-closed capability probe.

    A missing NIXL module is ``DEPENDENCY-BLOCKED``. Missing IB after a
    successful import is ``HARDWARE-BLOCKED``. UCX is
    ``IMPLEMENTATION-INCOMPLETE`` whether or not a bridge is installed.
    """
    if mode is None:
        requested = os.environ.get("TETRIS_LORA_RDMA_MODE", "mock").lower()
    elif isinstance(mode, RdmaMode):
        requested = mode.value
    else:
        requested = str(mode).lower()
    try:
        selected = RdmaMode(requested)
    except ValueError as exc:
        raise TransportError(f"unknown TETRIS_LORA_RDMA_MODE={requested!r}") from exc

    nixl_ok, nixl_agent_cls, _nixl_config_cls, nixl_reason = _try_import_nixl()
    ucx_ok, ib_visible, ucx_reason = _probe_ucx()
    ib_visible = ib_visible or _ib_visible()

    if selected == RdmaMode.MOCK:
        return _capability(
            mode=selected,
            status=DataplaneStatus.WIRED,
            nixl_importable=nixl_ok,
            ucx_endpoint_available=ucx_ok,
            ib_device_visible=ib_visible,
            reason="explicit mock loopback",
            provenance="mode=mock; MOCK loopback; no IB required",
        )
    if selected == RdmaMode.UCX:
        return _capability(
            mode=selected,
            status=DataplaneStatus.IMPLEMENTATION_INCOMPLETE,
            nixl_importable=nixl_ok,
            ucx_endpoint_available=ucx_ok,
            ib_device_visible=ib_visible,
            reason=(
                "UCX dataplane is not wired this round "
                f"({ucx_reason}); NIXL is the selected backend"
            ),
            provenance=(
                "mode=ucx IMPLEMENTATION-INCOMPLETE; "
                + _provenance(UCX_LIFECYCLE_PROVENANCE)
            ),
        )
    if not nixl_ok or nixl_agent_cls is None:
        return _capability(
            mode=selected,
            status=DataplaneStatus.DEPENDENCY_BLOCKED,
            nixl_importable=nixl_ok and nixl_agent_cls is not None,
            ucx_endpoint_available=ucx_ok,
            ib_device_visible=ib_visible,
            reason=nixl_reason,
            provenance=(
                "mode=nixl DEPENDENCY-BLOCKED; "
                + _provenance(NIXL_LIFECYCLE_PROVENANCE[:3])
            ),
        )
    if not ib_visible:
        return _capability(
            mode=selected,
            status=DataplaneStatus.HARDWARE_BLOCKED,
            nixl_importable=True,
            ucx_endpoint_available=ucx_ok,
            ib_device_visible=False,
            reason="NIXL importable but no IB device visible for dual-host RDMA",
            provenance=(
                "mode=nixl HARDWARE-BLOCKED; nixl present; "
                "/sys/class/infiniband|/dev/infiniband has no device"
            ),
        )
    return _capability(
        mode=selected,
        status=DataplaneStatus.WIRED,
        nixl_importable=True,
        ucx_endpoint_available=ucx_ok,
        ib_device_visible=True,
        reason=(
            "NIXL importable and IB visible; submit path wired; "
            "dual-host DONE is not VERIFIED"
        ),
        provenance="mode=nixl WIRED; " + _provenance(NIXL_LIFECYCLE_PROVENANCE),
    )


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = [0] * len(shape)
    acc = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = acc
        acc *= int(shape[index])
    return tuple(stride)


def _assert_contiguous_layout(tensor: Any) -> None:
    """Reject views NIXL cannot describe as one contiguous region."""
    is_contig = getattr(tensor, "is_contiguous", None)
    if not callable(is_contig):
        raise TransportError("tensor layout is not observable")
    if not bool(is_contig()):
        raise TransportError("NIXL tensor descriptors require contiguous tensors")
    shape = getattr(tensor, "shape", None)
    stride_fn = getattr(tensor, "stride", None)
    if shape is None or not callable(stride_fn):
        raise TransportError("tensor shape/stride are not observable")
    shape_t = tuple(int(dim) for dim in shape)
    stride_t = tuple(int(item) for item in stride_fn())
    if any(item < 0 for item in stride_t):
        raise TransportError("negative tensor strides are unsupported")
    if stride_t != _contiguous_stride(shape_t):
        raise TransportError("layout is not contiguous row-major")


def _payload_nbytes(payload: Any) -> int:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    nbytes = getattr(payload, "nbytes", None)
    if isinstance(nbytes, int) and not isinstance(nbytes, bool):
        return int(nbytes)
    raise TransportError(
        "payload byte length is not observable; pass bytes or a tensor"
    )


def _is_cuda_payload(payload: Any) -> bool:
    flag = getattr(payload, "is_cuda", False)
    return isinstance(flag, bool) and flag


def _data_ptr(payload: Any) -> int | None:
    getter = getattr(payload, "data_ptr", None)
    if not callable(getter):
        return None
    return int(getter())


def _mock_key(
    descriptor: TensorDescriptor, kind: TransferKind
) -> tuple[str, str, int]:
    return (descriptor.operation_id, kind.value, int(descriptor.slot_generation))


class _MockMailbox:
    """Bounded in-process payload mailbox shared by paired mock runtimes."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise TransportError("mock mailbox capacity must be positive")
        self.capacity = int(capacity)
        self._payloads: dict[tuple[str, str, int], tuple[TensorDescriptor, Any]] = {}
        self._lock = threading.Lock()

    def publish(
        self,
        key: tuple[str, str, int],
        descriptor: TensorDescriptor,
        payload: Any,
    ) -> None:
        with self._lock:
            if key in self._payloads:
                raise TransportError(
                    f"duplicate mock payload for operation {key[0]} "
                    f"kind={key[1]} generation={key[2]}"
                )
            if len(self._payloads) >= self.capacity:
                raise TransportError(
                    "mock mailbox at capacity; refusing unbounded queue"
                )
            self._payloads[key] = (descriptor, payload)

    def pop(
        self, key: tuple[str, str, int]
    ) -> tuple[TensorDescriptor, Any] | None:
        with self._lock:
            return self._payloads.pop(key, None)

    def pop_match(
        self,
        operation_id: str,
        kind: TransferKind,
        slot_generation: int | None,
    ) -> tuple[TensorDescriptor, Any]:
        with self._lock:
            matches = [
                item
                for item in self._payloads
                if item[0] == operation_id
                and item[1] == kind.value
                and (slot_generation is None or item[2] == slot_generation)
            ]
            if len(matches) != 1:
                raise TransportError(f"mock inbox miss for {(operation_id, kind)}")
            return self._payloads.pop(matches[0])


class _NixlSession:
    """NIXL agent lifecycle: open, register, handshake, post, poll, release.

    API names match ``nixl._api.nixl_agent`` as used by
    ``references/nixl/examples/python/basic_two_peers.py`` and the
    transfers-polling-cleanup reference. Push is ``WRITE``. Pull is ``READ``.
    """

    def __init__(self, agent_cls: Any, config_cls: Any | None) -> None:
        self._agent_cls = agent_cls
        self._config_cls = config_cls
        self.agent: Any | None = None
        self.agent_metadata: bytes | None = None
        self.remote_agent_name: str | None = None
        self.bind_error: str | None = None
        self.bind_status = DataplaneStatus.DEPENDENCY_BLOCKED
        self._regs: list[tuple[str, Any]] = []
        self._local_dlists: list[tuple[str, Any]] = []
        self._remote_dlists: list[tuple[str, Any]] = []
        self.remote_sides: dict[str, Any] = {}
        self.remote_descs: dict[str, Any] = {}
        self._try_open()

    def _try_open(self) -> None:
        config_cls = self._config_cls
        if config_cls is None:
            self.bind_error = "nixl_agent_config unavailable"
            self.bind_status = DataplaneStatus.DEPENDENCY_BLOCKED
            return
        name = f"tetris-{uuid.uuid4().hex[:8]}"
        try:
            # Out-of-band metadata exchange; do not open a public listener.
            # backends=["UCX"] is NIXL's plugin, not the Tetris UCX runtime.
            config = config_cls(
                enable_prog_thread=True,
                enable_listen_thread=False,
                listen_port=0,
                backends=["UCX"],
            )
            self.agent = self._agent_cls(name, config)
            getter = getattr(self.agent, "get_agent_metadata", None)
            if not callable(getter):
                raise TransportError("nixl agent missing get_agent_metadata")
            metadata = getter()
            if not isinstance(metadata, (bytes, bytearray)) or len(metadata) == 0:
                raise TransportError(
                    "nixl get_agent_metadata did not return metadata bytes"
                )
            self.agent_metadata = bytes(metadata)
            self.bind_status = DataplaneStatus.WIRED
            self.bind_error = None
        except Exception as exc:
            self.agent = None
            self.agent_metadata = None
            self.bind_error = f"nixl agent open failed: {exc}"
            if isinstance(exc, (ImportError, ModuleNotFoundError)):
                self.bind_status = DataplaneStatus.DEPENDENCY_BLOCKED
            else:
                self.bind_status = DataplaneStatus.HARDWARE_BLOCKED

    @property
    def bound(self) -> bool:
        return self.agent is not None and self.agent_metadata is not None

    def _require_agent(self) -> Any:
        if self.agent is None:
            raise TransportUnavailableError(
                f"nixl agent not bound ({self.bind_status.value}): "
                f"{self.bind_error or 'open failed'}"
            )
        return self.agent

    def register_tensor(self, token: str, tensor: Any) -> tuple[Any, Any, Any]:
        """Register one contiguous tensor and prepare the local xfer side.

        Returns:
            Registration descriptors, transfer descriptors, and the local
            ``prep_xfer_dlist`` handle.

        Raises:
            TransportError: Registration or local prepare failed.
            NixlQuarantineError: Deregister failed after a partial register.
        """
        agent = self._require_agent()
        register = getattr(agent, "register_memory", None)
        get_xfer = getattr(agent, "get_xfer_descs", None)
        prep = getattr(agent, "prep_xfer_dlist", None)
        if not callable(register) or not callable(get_xfer) or not callable(prep):
            raise TransportUnavailableError(
                "nixl register_memory/get_xfer_descs/prep_xfer_dlist unavailable "
                "(IMPLEMENTATION-INCOMPLETE)"
            )
        reg_descs = register(tensor)
        if not reg_descs:
            raise TransportError("nixl register_memory returned no descriptors")
        self._regs.append((token, reg_descs))
        try:
            xfer_descs = get_xfer(tensor)
            if not xfer_descs:
                raise TransportError("nixl get_xfer_descs returned no descriptors")
            # Empty agent name is the local side (NIXL_INIT_AGENT).
            local_side = prep("", xfer_descs)
            if local_side is None:
                raise TransportError("nixl prep_xfer_dlist returned no local handle")
        except Exception as exc:
            self._rollback_register(token, reg_descs, exc)
            raise
        self._local_dlists.append((token, local_side))
        return reg_descs, xfer_descs, local_side

    def register_raw(
        self,
        token: str,
        *,
        data_ptr: int,
        payload_bytes: int,
        device_id: int,
    ) -> tuple[Any, Any, Any]:
        """Register a trusted VRAM address tuple owned by the caller."""
        agent = self._require_agent()
        register = getattr(agent, "register_memory", None)
        get_xfer = getattr(agent, "get_xfer_descs", None)
        prep = getattr(agent, "prep_xfer_dlist", None)
        if not callable(register) or not callable(get_xfer) or not callable(prep):
            raise TransportUnavailableError(
                "nixl register_memory/get_xfer_descs/prep_xfer_dlist unavailable "
                "(IMPLEMENTATION-INCOMPLETE)"
            )
        reg_list = [(int(data_ptr), int(payload_bytes), int(device_id), "")]
        xfer_list = [(int(data_ptr), int(payload_bytes), int(device_id))]
        reg_descs = register(reg_list, mem_type="VRAM")
        if not reg_descs:
            raise TransportError("nixl register_memory returned no descriptors")
        self._regs.append((token, reg_descs))
        try:
            xfer_descs = get_xfer(xfer_list, mem_type="VRAM")
            if not xfer_descs:
                raise TransportError("nixl get_xfer_descs returned no descriptors")
            local_side = prep("", xfer_descs, mem_type="VRAM")
            if local_side is None:
                raise TransportError("nixl prep_xfer_dlist returned no local handle")
        except Exception as exc:
            self._rollback_register(token, reg_descs, exc)
            raise
        self._local_dlists.append((token, local_side))
        return reg_descs, xfer_descs, local_side

    def _rollback_register(self, token: str, reg_descs: Any, exc: Exception) -> None:
        self._regs = [item for item in self._regs if item[0] != token]
        dereg = getattr(self.agent, "deregister_memory", None) if self.agent else None
        if not callable(dereg):
            raise NixlQuarantineError(
                f"nixl prepare failed: {exc}; deregister_memory unavailable; "
                "DMA quarantined"
            ) from exc
        try:
            dereg(reg_descs)
        except Exception as dereg_exc:
            raise NixlQuarantineError(
                f"nixl prepare failed: {exc}; deregister_memory failed: "
                f"{dereg_exc}; DMA quarantined"
            ) from exc

    def export_xfer_descs(self, xfer_descs: Any) -> bytes:
        """Serialize descriptors for a trusted control-plane peer."""
        agent = self._require_agent()
        serialize = getattr(agent, "get_serialized_descs", None)
        if not callable(serialize):
            raise TransportUnavailableError(
                "nixl get_serialized_descs unavailable (IMPLEMENTATION-INCOMPLETE)"
            )
        blob = serialize(xfer_descs)
        if not isinstance(blob, (bytes, bytearray)) or len(blob) == 0:
            raise TransportError("nixl get_serialized_descs returned empty bytes")
        return bytes(blob)

    def attach_remote(self, remote_metadata: bytes) -> str:
        """Load peer metadata. ``add_remote_agent`` returns the peer name."""
        if not isinstance(remote_metadata, (bytes, bytearray)) or not remote_metadata:
            raise TransportError("remote NIXL metadata must be non-empty bytes")
        agent = self._require_agent()
        add = getattr(agent, "add_remote_agent", None)
        if not callable(add):
            raise TransportUnavailableError("nixl add_remote_agent unavailable")
        remote_name = add(bytes(remote_metadata))
        if not isinstance(remote_name, str) or not remote_name:
            raise TransportError("nixl add_remote_agent did not return a peer name")
        self.remote_agent_name = remote_name
        return remote_name

    def install_remote_descs(self, token: str, blob: bytes) -> Any:
        """Deserialize trusted peer descriptors and prepare the remote side.

        ``blob`` must come from the local control plane after
        ``get_serialized_descs``. Descriptor serialization uses pickle inside
        NIXL; this method does not accept notification payloads.
        """
        if self.remote_agent_name is None:
            raise TransportUnavailableError(
                "nixl remote peer metadata is not attached"
            )
        if not isinstance(blob, (bytes, bytearray)) or not blob:
            raise TransportError("remote xfer descriptor blob is empty")
        agent = self._require_agent()
        decode = getattr(agent, "deserialize_descs", None)
        prep = getattr(agent, "prep_xfer_dlist", None)
        if not callable(decode) or not callable(prep):
            raise TransportUnavailableError(
                "nixl deserialize_descs/prep_xfer_dlist unavailable "
                "(IMPLEMENTATION-INCOMPLETE)"
            )
        remote_descs = decode(bytes(blob))
        if not remote_descs:
            raise TransportError("nixl deserialize_descs returned no descriptors")
        remote_side = prep(self.remote_agent_name, remote_descs)
        if remote_side is None:
            raise TransportError("nixl prep_xfer_dlist returned no remote handle")
        previous = self.remote_sides.get(token)
        if previous is not None:
            self._release_dlist(previous)
        self.remote_sides[token] = remote_side
        self.remote_descs[token] = remote_descs
        self._remote_dlists = [
            item for item in self._remote_dlists if item[0] != token
        ]
        self._remote_dlists.append((token, remote_side))
        return remote_side

    def post(
        self,
        *,
        operation: str,
        local_side: Any,
        remote_side: Any,
        local_descs: Any,
        remote_descs: Any,
        notif_msg: bytes,
    ) -> tuple[Any, str]:
        """Post one READ or WRITE and return ``(handle, status)``.

        Prefers prepared sides (``make_prepped_xfer`` + ``transfer``). Falls
        back to ``initialize_xfer`` + ``transfer`` when those descriptors are
        present and prepared handles are not.
        """
        if operation not in {"READ", "WRITE"}:
            raise TransportError(f"unsupported nixl operation {operation}")
        if self.remote_agent_name is None:
            raise TransportUnavailableError(
                "nixl remote peer metadata is not attached"
            )
        agent = self._require_agent()
        make = getattr(agent, "make_prepped_xfer", None)
        initialize = getattr(agent, "initialize_xfer", None)
        transfer = getattr(agent, "transfer", None)
        if not callable(transfer):
            raise TransportUnavailableError(
                "nixl transfer() unavailable (IMPLEMENTATION-INCOMPLETE)"
            )
        if local_side is not None and remote_side is not None and callable(make):
            handle = make(
                operation=operation,
                local_xfer_side=local_side,
                local_indices=[0],
                remote_xfer_side=remote_side,
                remote_indices=[0],
                notif_msg=notif_msg,
            )
        elif callable(initialize):
            if local_descs is None or remote_descs is None:
                raise TransportUnavailableError(
                    "nixl xfer dlist handles are not installed and "
                    "initialize_xfer descriptors are missing"
                )
            handle = initialize(
                operation=operation,
                local_descs=local_descs,
                remote_descs=remote_descs,
                remote_agent=self.remote_agent_name,
                notif_msg=notif_msg,
            )
        else:
            raise TransportUnavailableError(
                "nixl make_prepped_xfer/initialize_xfer unavailable "
                "(IMPLEMENTATION-INCOMPLETE)"
            )
        if handle is None:
            raise TransportError("nixl transfer handle was not created")
        state = transfer(handle, notif_msg=notif_msg)
        if not isinstance(state, str):
            raise TransportError(
                f"nixl transfer() returned non-str status {state!r}"
            )
        return handle, state

    def check(self, handle: Any) -> str:
        agent = self._require_agent()
        checker = getattr(agent, "check_xfer_state", None)
        if not callable(checker):
            raise TransportUnavailableError("nixl check_xfer_state unavailable")
        state = checker(handle)
        if not isinstance(state, str):
            raise TransportError(f"nixl check_xfer_state returned {state!r}")
        return state

    def release_xfer(self, handle: Any) -> None:
        agent = self._require_agent()
        release = getattr(agent, "release_xfer_handle", None)
        if not callable(release):
            raise TransportError("nixl release_xfer_handle unavailable")
        release(handle)

    def _release_dlist(self, handle: Any) -> None:
        agent = self._require_agent()
        release = getattr(agent, "release_dlist_handle", None)
        if not callable(release):
            raise TransportError("nixl release_dlist_handle unavailable")
        release(handle)

    def shutdown(
        self, tickets: list[PendingTransfer], uncertain: set[str]
    ) -> list[str]:
        """Cancel in-flight posts, release handles, drop the peer, deregister.

        Args:
            tickets: Outstanding transfers. Active handles are released
                (NIXL cancels a posted transfer inside ``release_xfer_handle``).
            uncertain: Tokens whose DMA state could not be proven idle. The
                caller must keep those tensor owners.

        Returns:
            Cleanup errors. Empty when every owned object was released.
        """
        errors: list[str] = []
        if self.agent is None:
            return errors
        for ticket in tickets:
            if ticket.released or ticket.nixl_handle is None:
                continue
            token = ticket.descriptor.buffer_token
            try:
                self.release_xfer(ticket.nixl_handle)
            except Exception as exc:
                uncertain.add(token)
                errors.append(
                    f"release_xfer_handle {ticket.operation_id}: {exc}"
                )
            else:
                ticket.released = True
        for token, handle in list(self._local_dlists):
            if token in uncertain:
                continue
            try:
                self._release_dlist(handle)
            except Exception as exc:
                uncertain.add(token)
                errors.append(f"release_dlist_handle local {token}: {exc}")
        for token, handle in list(self._remote_dlists):
            if token in uncertain:
                continue
            try:
                self._release_dlist(handle)
            except Exception as exc:
                uncertain.add(token)
                errors.append(f"release_dlist_handle remote {token}: {exc}")
        if self.remote_agent_name is not None:
            remove = getattr(self.agent, "remove_remote_agent", None)
            if not callable(remove):
                errors.append("remove_remote_agent unavailable")
            else:
                try:
                    remove(self.remote_agent_name)
                except Exception as exc:
                    errors.append(f"remove_remote_agent: {exc}")
            self.remote_agent_name = None
        dereg = getattr(self.agent, "deregister_memory", None)
        for token, desc in list(self._regs):
            if token in uncertain:
                continue
            if not callable(dereg):
                uncertain.add(token)
                errors.append(f"deregister_memory unavailable for {token}")
                continue
            try:
                dereg(desc)
            except Exception as exc:
                uncertain.add(token)
                errors.append(f"deregister_memory {token}: {exc}")
        self._regs.clear()
        self._local_dlists.clear()
        self._remote_dlists.clear()
        self.remote_sides.clear()
        self.remote_descs.clear()
        self.agent = None
        self.agent_metadata = None
        return errors


class TetrisRdmaRuntime:
    """Minimal send_x/recv_x/send_delta/recv_delta session.

    CPU metadata stays on ``TensorDescriptor``. Non-mock modes refuse host
    tensors. Push (``send_*``) posts NIXL ``WRITE``. Pull (``recv_*``) posts
    NIXL ``READ`` and becomes done only after status ``DONE``. Mock recv
    stays pending until a matching payload is published.
    """

    def __init__(
        self,
        *,
        session_epoch: str | None = None,
        mode: RdmaMode | str | None = None,
        nixl_agent_cls: Any | None = None,
        nixl_agent_config_cls: Any | None = None,
        mock_mailbox_capacity: int = 32,
    ) -> None:
        self.session_epoch = session_epoch or uuid.uuid4().hex
        self._accepting = True
        self._closed = False
        self._regions: dict[str, MemoryRegion] = {}
        self._pending: dict[str, PendingTransfer] = {}
        self._tensor_owners: dict[str, Any] = {}
        self.quarantine: dict[str, tuple[Any, str]] = {}
        self._lock = threading.Lock()
        self._completion_seq = 0
        self._nixl: _NixlSession | None = None
        self._mailbox: _MockMailbox | None = None
        self._mock_cap = int(mock_mailbox_capacity)
        if nixl_agent_cls is not None:
            if mode not in (None, RdmaMode.NIXL, RdmaMode.NIXL.value):
                raise TransportError("injected NIXL agent requires mode=nixl")
            self.capability = _capability(
                mode=RdmaMode.NIXL,
                status=DataplaneStatus.WIRED,
                nixl_importable=True,
                ucx_endpoint_available=False,
                ib_device_visible=False,
                reason="injected nixl agent; hardware VERIFIED is not claimed",
                provenance="mode=nixl WIRED; injected agent; submit path under test",
            )
            agent_cls = nixl_agent_cls
            config_cls = nixl_agent_config_cls
        else:
            self.capability = probe_rdma_capability(mode)
            if self.capability.status in _BLOCKING_STATUSES:
                raise TransportUnavailableError(
                    f"RDMA mode {self.capability.mode.value} "
                    f"{self.capability.status.value}: {self.capability.reason}"
                )
            agent_cls = None
            config_cls = None
            if self.capability.mode == RdmaMode.NIXL:
                _ok, agent_cls, config_cls, reason = _try_import_nixl()
                if agent_cls is None:
                    raise TransportUnavailableError(
                        "RDMA mode nixl DEPENDENCY-BLOCKED: "
                        f"{reason}; no MOCK/TCP fallback"
                    )
        self._backend_note = self.capability.provenance
        if self.capability.mode == RdmaMode.NIXL:
            if agent_cls is None:
                raise TransportUnavailableError(
                    "RDMA mode nixl DEPENDENCY-BLOCKED: agent factory missing; "
                    "no MOCK/TCP fallback"
                )
            self._nixl = _NixlSession(agent_cls, config_cls)
            if not self._nixl.bound:
                status = self._nixl.bind_status
                detail = self._nixl.bind_error or "agent open failed"
                self._nixl = None
                raise TransportUnavailableError(
                    f"RDMA mode nixl {status.value}: {detail}; no MOCK/TCP fallback"
                )
            self.capability = replace(self.capability, nixl_agent_bound=True)
            self._backend_note = self.capability.provenance

    def _ensure_mailbox(self) -> _MockMailbox:
        if self._mailbox is None:
            self._mailbox = _MockMailbox(self._mock_cap)
        return self._mailbox

    def attach_mock_peer(self, peer: TetrisRdmaRuntime) -> None:
        """Share one bounded mailbox. Recv completes only from that mailbox."""
        if (
            self.capability.mode != RdmaMode.MOCK
            or peer.capability.mode != RdmaMode.MOCK
        ):
            raise TransportError("attach_mock_peer requires mode=mock")
        box = self._mailbox or peer._mailbox or _MockMailbox(
            min(self._mock_cap, peer._mock_cap)
        )
        self._mailbox = box
        peer._mailbox = box

    def register_buffer(
        self,
        *,
        kind: TransferKind,
        payload_bytes: int,
        device: str,
        token: str | None = None,
        data_ptr: int | None = None,
        device_id: int | None = None,
        tensor: Any | None = None,
    ) -> MemoryRegion:
        """Register one session buffer.

        In NIXL mode ``registered`` is true only after ``register_memory``
        returns descriptors. The tensor is held until deregister or quarantine.
        Mock mode records a logical buffer after layout checks; it is not an MR.
        """
        if self._closed or not self._accepting:
            raise TransportError("runtime is closed; submits stopped")
        nbytes = int(payload_bytes)
        if nbytes < 0:
            raise TransportError("payload_bytes must be non-negative")
        buffer_token = token or f"{kind.value}:{uuid.uuid4().hex}"
        with self._lock:
            if buffer_token in self._regions or buffer_token in self.quarantine:
                raise TransportError(f"buffer {buffer_token} already registered")
        if tensor is not None:
            _assert_contiguous_layout(tensor)
            tensor_nbytes = getattr(tensor, "nbytes", None)
            if isinstance(tensor_nbytes, int) and int(tensor_nbytes) != nbytes:
                raise TransportError(
                    f"tensor nbytes {tensor_nbytes} != payload_bytes {nbytes}"
                )
        if self.capability.mode != RdmaMode.MOCK:
            if tensor is None and (data_ptr is None or device_id is None):
                raise TransportError(
                    "nixl register_buffer requires a contiguous tensor or "
                    "a trusted data_ptr and device_id"
                )
            if tensor is not None and not _is_cuda_payload(tensor):
                raise TransportError(
                    "non-mock RDMA refuses host-staged GPU payloads"
                )
            if tensor is None and not str(device).startswith("cuda"):
                raise TransportError(
                    "non-mock RDMA refuses host-staged GPU payloads"
                )
        nixl_desc = None
        xfer_descs = None
        local_side = None
        recorded_ptr = data_ptr if data_ptr is not None else (
            _data_ptr(tensor) if tensor is not None else None
        )
        registered = self.capability.mode == RdmaMode.MOCK
        if self._nixl is not None:
            if tensor is not None:
                self._tensor_owners[buffer_token] = tensor
            try:
                if tensor is not None:
                    nixl_desc, xfer_descs, local_side = self._nixl.register_tensor(
                        buffer_token, tensor
                    )
                else:
                    assert data_ptr is not None and device_id is not None
                    nixl_desc, xfer_descs, local_side = self._nixl.register_raw(
                        buffer_token,
                        data_ptr=int(data_ptr),
                        payload_bytes=nbytes,
                        device_id=int(device_id),
                    )
            except NixlQuarantineError as exc:
                if tensor is not None:
                    self.quarantine[buffer_token] = (tensor, str(exc))
                raise
            except Exception:
                self._tensor_owners.pop(buffer_token, None)
                raise
            if not nixl_desc:
                self._tensor_owners.pop(buffer_token, None)
                raise TransportError("nixl register_memory failed")
            registered = True
        region = MemoryRegion(
            token=buffer_token,
            kind=kind,
            payload_bytes=nbytes,
            device=str(device),
            registered=registered,
            nixl_desc=nixl_desc,
            data_ptr=recorded_ptr,
            xfer_descs=xfer_descs,
            local_side=local_side,
        )
        with self._lock:
            if not region.registered:
                raise TransportError(
                    f"buffer {buffer_token} was not registered"
                )
            self._regions[buffer_token] = region
        return region

    def attach_nixl_remote(
        self, remote_metadata: bytes, name: str | None = None
    ) -> str:
        """Attach a remote NIXL agent. The peer name comes from the API.

        Args:
            remote_metadata: Bytes from the peer ``get_agent_metadata``.
            name: Ignored. ``add_remote_agent`` returns the canonical name.
        """
        del name
        if self._nixl is None:
            raise TransportError("attach_nixl_remote requires mode=nixl")
        return self._nixl.attach_remote(remote_metadata)

    def attach_nixl_peer(
        self,
        remote_metadata: bytes,
        remote_xfer_descs: dict[str, bytes] | None = None,
    ) -> str:
        """Exchange agent metadata and install trusted remote xfer descriptors.

        Args:
            remote_metadata: Bytes from the peer's ``get_agent_metadata``.
            remote_xfer_descs: Map of local buffer token to the peer's
                ``get_serialized_descs`` blob for that buffer.

        Returns:
            Remote agent name from ``add_remote_agent``.
        """
        remote_name = self.attach_nixl_remote(remote_metadata)
        for token, blob in (remote_xfer_descs or {}).items():
            if self._nixl is None:
                raise TransportError("nixl session closed during handshake")
            self._nixl.install_remote_descs(token, blob)
        return remote_name

    def export_nixl_agent_metadata(self) -> bytes:
        """Local agent metadata for the trusted control-plane handshake."""
        if self._nixl is None or self._nixl.agent_metadata is None:
            raise TransportError("nixl agent metadata unavailable")
        return self._nixl.agent_metadata

    def export_nixl_xfer_descs(self, token: str) -> bytes:
        """Serialize one local transfer descriptor list."""
        if self._nixl is None:
            raise TransportError("export_nixl_xfer_descs requires mode=nixl")
        region = self._regions.get(token)
        if region is None:
            raise TransportError(f"buffer {token} is not registered")
        region = self._require_region(token, region.kind)
        if region.xfer_descs is None:
            raise TransportError(f"buffer {token} has no xfer descriptors")
        return self._nixl.export_xfer_descs(region.xfer_descs)

    def _require_region(self, token: str, kind: TransferKind) -> MemoryRegion:
        region = self._regions.get(token)
        if region is None:
            raise TransportError(f"buffer {token} is not registered")
        if not region.registered:
            raise TransportError(f"buffer {token} is not registered")
        if region.kind != kind:
            raise TransportError(f"buffer {token} kind mismatch")
        return region

    def _reject_if_closed(self) -> None:
        if self._closed or not self._accepting:
            raise TransportError("runtime is closed; submits stopped")

    def _submit(
        self,
        *,
        kind: TransferKind,
        descriptor: TensorDescriptor,
        payload: Any | None,
        direction: str,
    ) -> PendingTransfer:
        self._reject_if_closed()
        descriptor.validate()
        if descriptor.layout not in _ACCEPTED_LAYOUTS:
            raise TransportError(f"unsupported layout {descriptor.layout}")
        if descriptor.session_epoch != self.session_epoch:
            raise TransportError("session_epoch mismatch")
        region = self._require_region(descriptor.buffer_token, kind)
        if descriptor.payload_bytes > region.payload_bytes:
            raise TransportError("payload exceeds registered buffer")
        if direction == "send":
            if payload is None:
                raise TransportError("send requires a payload")
            actual = _payload_nbytes(payload)
            if actual != int(descriptor.payload_bytes):
                raise TransportError(
                    f"payload bytes {actual} != descriptor "
                    f"{descriptor.payload_bytes}"
                )
        if self.capability.mode != RdmaMode.MOCK and payload is not None:
            if not _is_cuda_payload(payload):
                raise TransportError(
                    "non-mock RDMA refuses host-staged GPU payloads"
                )
            if region.data_ptr is not None:
                ptr = _data_ptr(payload)
                if ptr is not None and ptr != int(region.data_ptr):
                    raise TransportError("payload is not the registered buffer")
        ticket = PendingTransfer(
            operation_id=descriptor.operation_id,
            kind=kind,
            descriptor=descriptor,
            submitted_ns=time.monotonic_ns(),
            direction=direction,
        )
        with self._lock:
            self._reject_if_closed()
            if descriptor.operation_id in self._pending:
                raise TransportError(
                    f"duplicate operation_id {descriptor.operation_id}"
                )
            self._pending[descriptor.operation_id] = ticket
            try:
                if self.capability.mode == RdmaMode.MOCK:
                    self._submit_mock(ticket, payload)
                elif self.capability.mode == RdmaMode.NIXL:
                    self._submit_nixl(ticket, payload)
                else:
                    self._submit_ucx(ticket)
            except Exception as exc:
                if ticket.error is None:
                    ticket.error = exc
                # A post that never started must not look like a completion.
                if ticket.nixl_handle is None and ticket.mock_payload is None:
                    ticket.done = False
                raise
            if ticket.error is not None:
                raise ticket.error
        return ticket

    def _submit_mock(self, ticket: PendingTransfer, payload: Any | None) -> None:
        key = _mock_key(ticket.descriptor, ticket.kind)
        mailbox = self._ensure_mailbox()
        if ticket.direction == "send":
            mailbox.publish(key, ticket.descriptor, payload)
            ticket.mock_payload = payload
            ticket.done = True
            self._completion_seq += 1
            return
        delivered = mailbox.pop(key)
        if delivered is None:
            # Pull waits for a matching send. Do not self-complete.
            ticket.done = False
            return
        _desc, found = delivered
        actual = _payload_nbytes(found)
        if actual != int(ticket.descriptor.payload_bytes):
            raise TransportError(
                f"delivered payload bytes {actual} != descriptor "
                f"{ticket.descriptor.payload_bytes}"
            )
        ticket.mock_payload = found
        ticket.done = True
        self._completion_seq += 1

    def _complete_mock_recv(self, ticket: PendingTransfer) -> None:
        if ticket.done or ticket.error is not None or ticket.direction != "recv":
            return
        if self._mailbox is None:
            return
        delivered = self._mailbox.pop(_mock_key(ticket.descriptor, ticket.kind))
        if delivered is None:
            return
        _desc, found = delivered
        actual = _payload_nbytes(found)
        if actual != int(ticket.descriptor.payload_bytes):
            ticket.error = TransportError(
                f"delivered payload bytes {actual} != descriptor "
                f"{ticket.descriptor.payload_bytes}"
            )
            ticket.done = True
            return
        ticket.mock_payload = found
        ticket.done = True
        self._completion_seq += 1

    def _submit_nixl(self, ticket: PendingTransfer, payload: Any | None) -> None:
        """Post WRITE (push) or READ (pull). Save the handle on the ticket."""
        del payload  # Bytes stay in the registered MR; NIXL does not copy here.
        if self._nixl is None or not self._nixl.bound:
            raise TransportUnavailableError(
                "nixl session not bound; no MOCK/TCP fallback"
            )
        if self._nixl.remote_agent_name is None:
            raise TransportUnavailableError(
                "nixl remote peer metadata is not attached"
            )
        region = self._require_region(ticket.descriptor.buffer_token, ticket.kind)
        remote_side = self._nixl.remote_sides.get(region.token)
        remote_descs = self._nixl.remote_descs.get(region.token)
        if remote_side is None and remote_descs is None:
            raise TransportUnavailableError(
                "nixl remote xfer descs are not installed for "
                f"{region.token}"
            )
        # send = push WRITE, recv = pull READ. Completion is a NIXL status.
        operation = "WRITE" if ticket.direction == "send" else "READ"
        notif = (
            f"{self.session_epoch}:{ticket.operation_id}:"
            f"{ticket.kind.value}:{ticket.descriptor.slot_generation}"
        ).encode()
        handle, state = self._nixl.post(
            operation=operation,
            local_side=region.local_side,
            remote_side=remote_side,
            local_descs=region.xfer_descs,
            remote_descs=remote_descs,
            notif_msg=notif,
        )
        ticket.nixl_handle = handle
        self._apply_nixl_state(ticket, state)

    def _submit_ucx(self, ticket: PendingTransfer) -> None:
        del ticket
        raise TransportUnavailableError(
            "UCX dataplane is IMPLEMENTATION-INCOMPLETE this round; "
            "NIXL is the wired backend; no MOCK/TCP fallback"
        )

    def _apply_nixl_state(self, ticket: PendingTransfer, state: str) -> None:
        """Map NIXL ``DONE`` / ``PROC`` / ``ERR`` onto the ticket."""
        if ticket.done or ticket.error is not None:
            return
        normalized = state.strip().upper()
        ticket.nixl_state = normalized
        if normalized == _NIXL_DONE:
            ticket.done = True
            self._completion_seq += 1
            self._release_nixl_handle(ticket)
            return
        if normalized == _NIXL_PROC:
            ticket.done = False
            return
        if normalized == _NIXL_ERR:
            ticket.error = TransportError(f"nixl xfer failed: {state}")
            ticket.done = True
            self._release_nixl_handle(ticket)
            return
        ticket.error = TransportError(f"nixl xfer unexpected status: {state}")
        ticket.done = True
        self._release_nixl_handle(ticket)

    def _release_nixl_handle(self, ticket: PendingTransfer) -> None:
        if ticket.released or ticket.nixl_handle is None or self._nixl is None:
            return
        try:
            self._nixl.release_xfer(ticket.nixl_handle)
        except Exception as exc:
            token = ticket.descriptor.buffer_token
            owner = self._tensor_owners.get(token)
            if owner is not None:
                self.quarantine[token] = (
                    owner,
                    f"release_xfer_handle failed: {exc}",
                )
            ticket.error = TransportError(
                f"nixl release_xfer_handle failed: {exc}"
            )
            return
        ticket.released = True

    def send_x(self, descriptor: TensorDescriptor, payload: Any) -> PendingTransfer:
        return self._submit(
            kind=TransferKind.X,
            descriptor=descriptor,
            payload=payload,
            direction="send",
        )

    def recv_x(self, descriptor: TensorDescriptor) -> PendingTransfer:
        return self._submit(
            kind=TransferKind.X,
            descriptor=descriptor,
            payload=None,
            direction="recv",
        )

    def send_delta(
        self, descriptor: TensorDescriptor, payload: Any
    ) -> PendingTransfer:
        return self._submit(
            kind=TransferKind.DELTA,
            descriptor=descriptor,
            payload=payload,
            direction="send",
        )

    def recv_delta(self, descriptor: TensorDescriptor) -> PendingTransfer:
        return self._submit(
            kind=TransferKind.DELTA,
            descriptor=descriptor,
            payload=None,
            direction="recv",
        )

    def poll(self, operation_id: str) -> PendingTransfer:
        """Progress one ticket: mock delivery or NIXL ``check_xfer_state``."""
        with self._lock:
            ticket = self._pending.get(operation_id)
            if ticket is None:
                raise TransportError(f"unknown transfer {operation_id}")
            if (
                self.capability.mode == RdmaMode.MOCK
                and ticket.direction == "recv"
                and not ticket.done
                and ticket.error is None
            ):
                self._complete_mock_recv(ticket)
            if (
                ticket.nixl_handle is not None
                and self._nixl is not None
                and not ticket.done
                and ticket.error is None
                and not ticket.released
            ):
                try:
                    state = self._nixl.check(ticket.nixl_handle)
                except Exception as exc:
                    ticket.error = TransportError(
                        f"nixl check_xfer_state failed: {exc}"
                    )
                    ticket.done = True
                else:
                    self._apply_nixl_state(ticket, state)
            error = ticket.error
        if error is not None:
            raise error
        return ticket

    def take_mock_payload(
        self,
        operation_id: str,
        kind: TransferKind,
        *,
        slot_generation: int | None = None,
    ) -> tuple[TensorDescriptor, Any]:
        """Remove one undelivered mock payload. Does not complete a recv."""
        if self._mailbox is None:
            raise TransportError(
                f"mock inbox miss for {(operation_id, kind)}"
            )
        return self._mailbox.pop_match(operation_id, kind, slot_generation)

    def _make_descriptor(
        self,
        *,
        plan: DenoiseStepPlan,
        operation_id: str,
        layer_name: str,
        buffer_token: str,
        completion_sequence: int,
        invocation_id: int,
    ) -> TensorDescriptor:
        return TensorDescriptor(
            session_epoch=self.session_epoch,
            operation_id=operation_id,
            request_id=plan.request_id,
            batch_id=plan.batch_id,
            denoise_step=plan.denoise_step,
            layer_name=layer_name,
            invocation_id=invocation_id,
            slot_id=0,
            slot_generation=1,
            shape=plan.shape,
            stride=plan.stride,
            dtype=plan.dtype,
            layout="row_major",
            payload_bytes=plan.payload_bytes,
            buffer_token=buffer_token,
            adapter_id=plan.adapter_id,
            adapter_version=plan.adapter_version,
            rank=plan.rank,
            scale=plan.scale,
            branch_index=0,
            output_offset=0,
            completion_sequence=completion_sequence,
        )

    def run_mock_all_targets_denoise_step(
        self,
        plan: DenoiseStepPlan,
        *,
        payload_factory: Callable[[str, TransferKind], Any] | None = None,
    ) -> DenoiseStepResult:
        """One denoise step across N targets on a paired mock mailbox.

        Base pushes X, the peer pulls X, the peer pushes Δ, Base pulls Δ.
        Recv completes only when the matching operation, kind, and generation
        were published. This schedule is separate from the protocol tests.
        """
        if self.capability.mode != RdmaMode.MOCK:
            raise TransportError(
                "run_mock_all_targets_denoise_step requires mode=mock"
            )
        if not plan.targets:
            raise TransportError("DenoiseStepPlan.targets must be non-empty")
        if plan.payload_bytes < 0:
            raise TransportError("payload_bytes must be non-negative")

        peer = TetrisRdmaRuntime(
            session_epoch=self.session_epoch,
            mode=RdmaMode.MOCK,
            mock_mailbox_capacity=self._mock_cap,
        )
        self.attach_mock_peer(peer)
        try:
            x_region = self.register_buffer(
                kind=TransferKind.X,
                payload_bytes=plan.payload_bytes,
                device="mock:0",
                token=f"x-all:{self.session_epoch}",
            )
            delta_region = self.register_buffer(
                kind=TransferKind.DELTA,
                payload_bytes=plan.payload_bytes,
                device="mock:0",
                token=f"delta-all:{self.session_epoch}",
            )
            peer.register_buffer(
                kind=TransferKind.X,
                payload_bytes=plan.payload_bytes,
                device="mock:0",
                token=x_region.token,
            )
            peer.register_buffer(
                kind=TransferKind.DELTA,
                payload_bytes=plan.payload_bytes,
                device="mock:0",
                token=delta_region.token,
            )
            result = DenoiseStepResult(
                denoise_step=plan.denoise_step,
                targets=tuple(plan.targets),
            )
            factory = payload_factory or (
                lambda _layer, _kind: bytes(plan.payload_bytes)
            )
            for index, layer_name in enumerate(plan.targets):
                x_op = f"x:{plan.denoise_step}:{index}:{layer_name}"
                delta_op = f"delta:{plan.denoise_step}:{index}:{layer_name}"
                x_desc = self._make_descriptor(
                    plan=plan,
                    operation_id=x_op,
                    layer_name=layer_name,
                    buffer_token=x_region.token,
                    completion_sequence=self._completion_seq,
                    invocation_id=index,
                )
                x_payload = factory(layer_name, TransferKind.X)
                send_ticket = self.send_x(x_desc, x_payload)
                if not self.poll(x_op).done:
                    raise TransportError(f"mock X send incomplete for {layer_name}")
                got_x = peer.recv_x(x_desc)
                if not got_x.done or got_x.mock_payload != x_payload:
                    raise TransportError("mock X payload was not delivered")
                if got_x.descriptor.layer_name != layer_name:
                    raise TransportError("mock X layer_name mismatch")
                result.x_send_ops.append(send_ticket.operation_id)

                delta_desc = self._make_descriptor(
                    plan=plan,
                    operation_id=delta_op,
                    layer_name=layer_name,
                    buffer_token=delta_region.token,
                    completion_sequence=self._completion_seq,
                    invocation_id=index,
                )
                delta_payload = factory(layer_name, TransferKind.DELTA)
                peer.send_delta(delta_desc, delta_payload)
                recv_ticket = self.recv_delta(delta_desc)
                if (
                    not self.poll(delta_op).done
                    or recv_ticket.mock_payload != delta_payload
                ):
                    raise TransportError(
                        f"mock recv_delta incomplete for {layer_name}"
                    )
                if recv_ticket.descriptor.layer_name != layer_name:
                    raise TransportError("mock Δ layer_name mismatch")
                result.delta_recv_ops.append(recv_ticket.operation_id)
            result.completion_sequence = self._completion_seq
            return result
        finally:
            peer.close()

    def close(self) -> None:
        """Stop submits, cancel in-flight DMA, then drop safe tensor owners.

        Order: reject new submits, ``release_xfer_handle`` (cancels active
        posts), ``release_dlist_handle``, ``remove_remote_agent``,
        ``deregister_memory``, drop the agent, then drop tensor owners.
        A buffer whose release or deregister fails stays in ``quarantine``.
        """
        errors: list[str] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._accepting = False
            uncertain: set[str] = set()
            if self._nixl is not None:
                errors = self._nixl.shutdown(
                    list(self._pending.values()), uncertain
                )
                self._nixl = None
            owners = self._tensor_owners
            self._tensor_owners = {}
            for token, tensor in owners.items():
                if token in uncertain or token in self.quarantine:
                    reason = "uncertain DMA during close"
                    existing = self.quarantine.get(token)
                    if existing is not None:
                        reason = existing[1]
                    self.quarantine[token] = (tensor, reason)
            owners.clear()
            self._pending.clear()
            self._mailbox = None
        if errors:
            raise TransportError("nixl close failed: " + "; ".join(errors))


__all__ = [
    "METADATA_VS_GPU_PAYLOAD",
    "NIXL_LIFECYCLE_PROVENANCE",
    "UCX_LIFECYCLE_PROVENANCE",
    "DataplaneStatus",
    "DenoiseStepPlan",
    "DenoiseStepResult",
    "MemoryRegion",
    "PendingTransfer",
    "RdmaCapability",
    "RdmaMode",
    "TetrisRdmaRuntime",
    "TransferKind",
    "probe_rdma_capability",
]
