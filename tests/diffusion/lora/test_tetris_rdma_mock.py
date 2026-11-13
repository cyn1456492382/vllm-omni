"""MOCK protocol tests and NIXL/UCX dataplane status gates."""

from __future__ import annotations

import os

import pytest

from vllm_omni.diffusion.lora import tetris_lora_rdma_runtime as rdma_runtime
from vllm_omni.diffusion.lora.tetris_lora_rdma_runtime import (
    METADATA_VS_GPU_PAYLOAD,
    NIXL_LIFECYCLE_PROVENANCE,
    DataplaneStatus,
    DenoiseStepPlan,
    RdmaMode,
    TetrisRdmaRuntime,
    TransferKind,
    probe_rdma_capability,
)
from vllm_omni.diffusion.lora.tetris_lora_transport import (
    TensorDescriptor,
    TransportError,
    TransportUnavailableError,
)

def _descriptor(
    session_epoch: str,
    operation_id: str,
    token: str,
    *,
    slot_generation: int = 1,
    payload_bytes: int = 64,
) -> TensorDescriptor:
    return TensorDescriptor(
        session_epoch=session_epoch,
        operation_id=operation_id,
        request_id="req-1",
        batch_id="batch-1",
        denoise_step=0,
        layer_name="blocks.0.attn.to_q",
        invocation_id=0,
        slot_id=0,
        slot_generation=slot_generation,
        shape=(4, 8),
        stride=(8, 1),
        dtype="bfloat16",
        layout="row_major",
        payload_bytes=payload_bytes,
        buffer_token=token,
        adapter_id="adapter",
        adapter_version="v1",
        rank=8,
        scale=1.0,
        branch_index=0,
        output_offset=0,
        completion_sequence=0,
    )


class _CudaBuf:
    """Contiguous stand-in for a device tensor. Not a host byte copy."""

    def __init__(self, nbytes: int, *, contiguous: bool = True) -> None:
        self.nbytes = nbytes
        self.is_cuda = True
        self.shape = (nbytes,)
        self._contiguous = contiguous
        self._stride = (1,) if contiguous else (2,)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def data_ptr(self) -> int:
        return 0x1000


class _FakeConfig:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeAgent:
    """Records NIXL agent calls. Status strings match the Python API."""

    def __init__(self, name: str, cfg: _FakeConfig | None = None) -> None:
        self.name = name
        self.cfg = cfg
        self.calls: list[tuple] = []
        self.fail_release = False
        self._next = "PROC"

    def get_agent_metadata(self) -> bytes:
        self.calls.append(("get_agent_metadata",))
        return b"agent-md"

    def register_memory(self, reg_list, mem_type=None, backends=None):
        self.calls.append(("register_memory", reg_list, mem_type))
        if getattr(self, "empty_reg", False):
            return None
        return ("reg", reg_list)

    def get_xfer_descs(self, descs, mem_type=None):
        self.calls.append(("get_xfer_descs", mem_type))
        return ("xfer", descs)

    def get_serialized_descs(self, descs) -> bytes:
        self.calls.append(("get_serialized_descs",))
        return b"xfer-blob"

    def deserialize_descs(self, serialized_descs: bytes):
        self.calls.append(("deserialize_descs", serialized_descs))
        return ("remote-descs", serialized_descs)

    def prep_xfer_dlist(self, agent_name, xfer_list, mem_type=None, backends=None):
        self.calls.append(("prep_xfer_dlist", agent_name))
        return ("dlist", agent_name)

    def add_remote_agent(self, metadata: bytes) -> str:
        self.calls.append(("add_remote_agent", metadata))
        return "peer"

    def make_prepped_xfer(
        self,
        operation,
        local_xfer_side,
        local_indices,
        remote_xfer_side,
        remote_indices,
        notif_msg=b"",
        backends=None,
        skip_desc_merge=False,
    ):
        self.calls.append(
            ("make_prepped_xfer", operation, list(local_indices), list(remote_indices))
        )
        return ("xfer-handle", operation)

    def initialize_xfer(
        self,
        operation,
        local_descs,
        remote_descs,
        remote_agent,
        notif_msg=b"",
        backends=None,
    ):
        self.calls.append(("initialize_xfer", operation, remote_agent))
        return ("init-handle", operation)

    def transfer(self, handle, notif_msg=b"") -> str:
        self.calls.append(("transfer", handle, notif_msg))
        return self._next

    def check_xfer_state(self, handle) -> str:
        self.calls.append(("check_xfer_state", handle))
        return "DONE"

    def release_xfer_handle(self, handle) -> None:
        self.calls.append(("release_xfer_handle", handle))
        if self.fail_release:
            raise RuntimeError("cancel failed")

    def release_dlist_handle(self, handle) -> None:
        self.calls.append(("release_dlist_handle", handle))

    def deregister_memory(self, dereg_list, backends=None) -> None:
        self.calls.append(("deregister_memory", dereg_list))

    def remove_remote_agent(self, agent: str) -> bool:
        self.calls.append(("remove_remote_agent", agent))
        return True


def _nixl_runtime(agent_cls=_FakeAgent) -> TetrisRdmaRuntime:
    return TetrisRdmaRuntime(
        session_epoch="epoch-nixl",
        mode=RdmaMode.NIXL,
        nixl_agent_cls=agent_cls,
        nixl_agent_config_cls=_FakeConfig,
    )


def _handshake(runtime: TetrisRdmaRuntime, token: str) -> _FakeAgent:
    blob = runtime.export_nixl_xfer_descs(token)
    runtime.attach_nixl_peer(b"remote-md", {token: blob})
    agent = runtime._nixl.agent
    assert isinstance(agent, _FakeAgent)
    return agent


def test_probe_default_is_mock(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "mock")
    cap = probe_rdma_capability()
    assert cap.mode == RdmaMode.MOCK
    assert cap.hardware_blocked is False
    assert cap.status == DataplaneStatus.WIRED
    assert "mock" in cap.provenance


def test_nixl_mode_dependency_blocked_without_module(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "nixl")
    cap = probe_rdma_capability()
    if not cap.nixl_importable:
        assert cap.status == DataplaneStatus.DEPENDENCY_BLOCKED
        assert cap.hardware_blocked is False
        assert "DEPENDENCY-BLOCKED" in cap.provenance
        with pytest.raises(TransportUnavailableError, match="DEPENDENCY-BLOCKED"):
            TetrisRdmaRuntime()
    elif not cap.ib_device_visible:
        assert cap.status == DataplaneStatus.HARDWARE_BLOCKED
        assert "HARDWARE-BLOCKED" in cap.provenance
        with pytest.raises(TransportUnavailableError, match="HARDWARE-BLOCKED"):
            TetrisRdmaRuntime()
    else:
        assert cap.status == DataplaneStatus.WIRED
        assert cap.hardware_blocked is False


def test_status_taxonomy_is_not_all_hardware_blocked(monkeypatch) -> None:
    monkeypatch.setattr(rdma_runtime, "_probe_ucx", lambda: (False, False, "stub"))
    monkeypatch.setattr(rdma_runtime, "_ib_visible", lambda: False)
    monkeypatch.setattr(
        rdma_runtime,
        "_try_import_nixl",
        lambda: (False, None, None, "NIXL python module unavailable"),
    )
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "nixl")
    missing = probe_rdma_capability()
    assert missing.status == DataplaneStatus.DEPENDENCY_BLOCKED
    assert missing.hardware_blocked is False

    monkeypatch.setattr(
        rdma_runtime,
        "_try_import_nixl",
        lambda: (True, object, object, "nixl importable"),
    )
    no_ib = probe_rdma_capability()
    assert no_ib.status == DataplaneStatus.HARDWARE_BLOCKED
    assert no_ib.hardware_blocked is True

    monkeypatch.setattr(rdma_runtime, "_ib_visible", lambda: True)
    monkeypatch.setattr(rdma_runtime, "_probe_ucx", lambda: (False, True, "stub"))
    wired = probe_rdma_capability()
    assert wired.status == DataplaneStatus.WIRED
    assert wired.hardware_blocked is False
    assert "VERIFIED" not in wired.provenance

    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "ucx")
    monkeypatch.setattr(
        rdma_runtime,
        "_probe_ucx",
        lambda: (True, True, "ib visible but ucx runtime not wired"),
    )
    ucx = probe_rdma_capability()
    assert ucx.status == DataplaneStatus.IMPLEMENTATION_INCOMPLETE
    assert ucx.hardware_blocked is False
    assert ucx.ucx_endpoint_available is True
    with pytest.raises(TransportUnavailableError, match="IMPLEMENTATION-INCOMPLETE"):
        TetrisRdmaRuntime()


def test_ucx_mode_is_implementation_incomplete(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "ucx")
    monkeypatch.setenv("TETRIS_LORA_UCX_BRIDGE", "/tmp/not-used")
    monkeypatch.setattr(
        rdma_runtime,
        "_probe_ucx",
        lambda: (True, True, "bridge present"),
    )
    cap = probe_rdma_capability()
    assert cap.status == DataplaneStatus.IMPLEMENTATION_INCOMPLETE
    assert "IMPLEMENTATION-INCOMPLETE" in cap.provenance
    assert cap.hardware_blocked is False
    with pytest.raises(TransportUnavailableError, match="IMPLEMENTATION-INCOMPLETE"):
        TetrisRdmaRuntime()


def test_mock_send_x_loopback(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "mock")
    runtime = TetrisRdmaRuntime(session_epoch="epoch-test")
    region = runtime.register_buffer(
        kind=TransferKind.X, payload_bytes=64, device="cuda:0"
    )
    payload = b"\x11" * 64
    descriptor = _descriptor(runtime.session_epoch, "op-x-1", region.token)
    ticket = runtime.send_x(descriptor, payload=payload)
    assert runtime.poll("op-x-1").done is True
    got, found = runtime.take_mock_payload("op-x-1", TransferKind.X)
    assert got.operation_id == "op-x-1"
    assert found == payload
    assert os.environ.get("TETRIS_LORA_RDMA_MODE") == "mock"
    assert ticket.done is True


def test_metadata_vs_gpu_payload_table_covers_descriptor_planes() -> None:
    planes = {row[1] for row in METADATA_VS_GPU_PAYLOAD}
    assert planes == {"cpu_metadata", "gpu_payload"}
    assert any("activation X" in row[0] for row in METADATA_VS_GPU_PAYLOAD)
    assert any("residual" in row[0] for row in METADATA_VS_GPU_PAYLOAD)
    assert NIXL_LIFECYCLE_PROVENANCE
    assert any("register_memory" in step for step, _ in NIXL_LIFECYCLE_PROVENANCE)
    assert any("make_prepped_xfer" in step for step, _ in NIXL_LIFECYCLE_PROVENANCE)


def test_mock_all_targets_single_denoise_step(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "mock")
    runtime = TetrisRdmaRuntime(session_epoch="epoch-all-targets")
    # Semantic stand-in for an all-targets schedule (truncated for unit speed).
    targets = tuple(f"blocks.{i}.attn.to_q" for i in range(8)) + tuple(
        f"blocks.{i}.attn.to_out.0" for i in range(4)
    )
    plan = DenoiseStepPlan(
        targets=targets,
        denoise_step=0,
        shape=(4, 8),
        stride=(8, 1),
        dtype="bfloat16",
        payload_bytes=64,
    )
    result = runtime.run_mock_all_targets_denoise_step(plan)
    assert result.denoise_step == 0
    assert result.targets == targets
    assert len(result.x_send_ops) == len(targets)
    assert len(result.delta_recv_ops) == len(targets)
    assert result.completion_sequence >= 2 * len(targets)
    assert runtime.capability.mode == RdmaMode.MOCK


def test_mock_protocol_recv_waits_for_matching_payload(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "mock")
    base = TetrisRdmaRuntime(session_epoch="epoch-proto")
    peer = TetrisRdmaRuntime(session_epoch="epoch-proto", mode=RdmaMode.MOCK)
    base.attach_mock_peer(peer)
    region = base.register_buffer(
        kind=TransferKind.X, payload_bytes=64, device="mock:0", token="x-proto"
    )
    peer.register_buffer(
        kind=TransferKind.X, payload_bytes=64, device="mock:0", token=region.token
    )
    payload = b"\xab" * 64
    desc = _descriptor(base.session_epoch, "op-match", region.token)
    waiting = peer.recv_x(desc)
    assert waiting.done is False
    assert peer.poll("op-match").done is False
    base.send_x(desc, payload)
    done = peer.poll("op-match")
    assert done.done is True
    assert done.mock_payload == payload
    peer.close()
    base.close()


def test_mock_protocol_generation_mismatch_does_not_deliver(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "mock")
    base = TetrisRdmaRuntime(session_epoch="epoch-gen")
    peer = TetrisRdmaRuntime(session_epoch="epoch-gen", mode=RdmaMode.MOCK)
    base.attach_mock_peer(peer)
    token = "x-gen"
    base.register_buffer(
        kind=TransferKind.DELTA, payload_bytes=64, device="mock:0", token=token
    )
    peer.register_buffer(
        kind=TransferKind.DELTA, payload_bytes=64, device="mock:0", token=token
    )
    payload = b"\xcd" * 64
    stale = _descriptor(
        base.session_epoch, "op-gen", token, slot_generation=2
    )
    current = _descriptor(
        base.session_epoch, "op-gen", token, slot_generation=1
    )
    peer.recv_delta(stale)
    base.send_delta(current, payload)
    assert peer.poll("op-gen").done is False
    got, found = base.take_mock_payload(
        "op-gen", TransferKind.DELTA, slot_generation=1
    )
    assert got.slot_generation == 1
    assert found == payload
    peer.close()
    base.close()


def test_mock_protocol_rejects_payload_length_mismatch(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "mock")
    runtime = TetrisRdmaRuntime(session_epoch="epoch-bytes")
    region = runtime.register_buffer(
        kind=TransferKind.X, payload_bytes=64, device="mock:0"
    )
    desc = _descriptor(runtime.session_epoch, "op-short", region.token)
    with pytest.raises(TransportError, match="payload bytes"):
        runtime.send_x(desc, b"\x00" * 8)


def test_mock_protocol_duplicate_operation_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "mock")
    runtime = TetrisRdmaRuntime(session_epoch="epoch-dup")
    region = runtime.register_buffer(
        kind=TransferKind.X, payload_bytes=64, device="mock:0"
    )
    desc = _descriptor(runtime.session_epoch, "op-dup", region.token)
    payload = b"\x01" * 64
    runtime.send_x(desc, payload)
    with pytest.raises(TransportError, match="duplicate operation_id"):
        runtime.send_x(desc, payload)
    _got, found = runtime.take_mock_payload("op-dup", TransferKind.X)
    assert found == payload


def test_mock_protocol_mailbox_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "mock")
    runtime = TetrisRdmaRuntime(session_epoch="epoch-cap", mock_mailbox_capacity=1)
    region = runtime.register_buffer(
        kind=TransferKind.X, payload_bytes=4, device="mock:0"
    )
    first = _descriptor(
        runtime.session_epoch, "op-cap-1", region.token, payload_bytes=4
    )
    second = _descriptor(
        runtime.session_epoch, "op-cap-2", region.token, payload_bytes=4
    )
    runtime.send_x(first, b"\x02" * 4)
    with pytest.raises(TransportError, match="capacity"):
        runtime.send_x(second, b"\x03" * 4)


def test_fake_agent_submit_posts_prepped_xfer() -> None:
    runtime = _nixl_runtime()
    buf = _CudaBuf(64)
    region = runtime.register_buffer(
        kind=TransferKind.X,
        payload_bytes=64,
        device="cuda:0",
        token="x-nixl",
        tensor=buf,
    )
    assert region.registered is True
    assert runtime._tensor_owners["x-nixl"] is buf
    agent = _handshake(runtime, region.token)
    assert agent.cfg is not None
    assert agent.cfg.kwargs["backends"] == ["UCX"]
    assert agent.cfg.kwargs["enable_listen_thread"] is False
    assert ("get_agent_metadata",) in agent.calls
    assert ("register_memory", buf, None) in agent.calls
    assert ("prep_xfer_dlist", "") in agent.calls
    assert ("add_remote_agent", b"remote-md") in agent.calls
    assert ("prep_xfer_dlist", "peer") in agent.calls

    desc = _descriptor(runtime.session_epoch, "op-write", region.token)
    ticket = runtime.send_x(desc, buf)
    assert ticket.nixl_handle == ("xfer-handle", "WRITE")
    assert ticket.done is False
    assert ticket.nixl_state == "PROC"
    assert ("make_prepped_xfer", "WRITE", [0], [0]) in agent.calls
    assert any(
        call[0] == "transfer" and call[1] == ticket.nixl_handle for call in agent.calls
    )
    assert not any(call[0] == "initialize_xfer" for call in agent.calls)
    assert runtime.poll("op-write").done is True
    assert ("check_xfer_state", ("xfer-handle", "WRITE")) in agent.calls
    assert ("release_xfer_handle", ("xfer-handle", "WRITE")) in agent.calls

    read = _descriptor(runtime.session_epoch, "op-read", region.token)
    recv = runtime.recv_x(read)
    assert recv.direction == "recv"
    assert recv.nixl_handle == ("xfer-handle", "READ")
    assert recv.done is False
    assert ("make_prepped_xfer", "READ", [0], [0]) in agent.calls
    assert runtime.poll("op-read").done is True

    runtime.close()
    assert ("remove_remote_agent", "peer") in agent.calls
    assert any(call[0] == "deregister_memory" for call in agent.calls)
    assert any(call[0] == "release_dlist_handle" for call in agent.calls)
    assert "x-nixl" not in runtime._tensor_owners
    assert "x-nixl" not in runtime.quarantine
    with pytest.raises(TransportError, match="closed"):
        runtime.send_x(desc, buf)


def test_fake_agent_falls_back_to_initialize_xfer() -> None:
    class _InitOnly(_FakeAgent):
        make_prepped_xfer = None  # type: ignore[assignment]

    runtime = _nixl_runtime(_InitOnly)
    buf = _CudaBuf(64)
    region = runtime.register_buffer(
        kind=TransferKind.DELTA,
        payload_bytes=64,
        device="cuda:0",
        token="d-nixl",
        tensor=buf,
    )
    agent = _handshake(runtime, region.token)
    desc = _descriptor(runtime.session_epoch, "op-init", region.token)
    ticket = runtime.send_delta(desc, buf)
    assert ticket.nixl_handle == ("init-handle", "WRITE")
    assert ("initialize_xfer", "WRITE", "peer") in agent.calls
    assert any(call[0] == "transfer" for call in agent.calls)
    assert ticket.done is False
    runtime.close()


def test_nixl_register_requires_real_registration() -> None:
    class _EmptyReg(_FakeAgent):
        empty_reg = True

    runtime = _nixl_runtime(_EmptyReg)
    buf = _CudaBuf(64)
    with pytest.raises(TransportError, match="register_memory"):
        runtime.register_buffer(
            kind=TransferKind.X,
            payload_bytes=64,
            device="cuda:0",
            token="missing",
            tensor=buf,
        )
    assert "missing" not in runtime._regions
    assert "missing" not in runtime._tensor_owners
    bad = _CudaBuf(64, contiguous=False)
    with pytest.raises(TransportError, match="contiguous"):
        runtime.register_buffer(
            kind=TransferKind.X,
            payload_bytes=64,
            device="cuda:0",
            token="bad-layout",
            tensor=bad,
        )
    assert not any(
        call[0] == "register_memory" and call[1] is bad
        for call in runtime._nixl.agent.calls
    )


def test_nixl_recv_does_not_self_complete_without_post() -> None:
    runtime = _nixl_runtime()
    buf = _CudaBuf(64)
    region = runtime.register_buffer(
        kind=TransferKind.X,
        payload_bytes=64,
        device="cuda:0",
        token="x-wait",
        tensor=buf,
    )
    desc = _descriptor(runtime.session_epoch, "op-wait", region.token)
    with pytest.raises(TransportUnavailableError, match="not attached"):
        runtime.recv_x(desc)
    ticket = runtime._pending["op-wait"]
    assert ticket.nixl_handle is None
    assert ticket.done is False
    assert ticket.error is not None
    with pytest.raises(TransportUnavailableError):
        runtime.poll("op-wait")


def test_close_quarantines_uncertain_dma() -> None:
    runtime = _nixl_runtime()
    buf = _CudaBuf(64)
    region = runtime.register_buffer(
        kind=TransferKind.X,
        payload_bytes=64,
        device="cuda:0",
        token="x-q",
        tensor=buf,
    )
    agent = _handshake(runtime, region.token)
    agent.fail_release = True
    desc = _descriptor(runtime.session_epoch, "op-q", region.token)
    ticket = runtime.send_x(desc, buf)
    assert ticket.nixl_handle is not None
    assert ticket.done is False
    with pytest.raises(TransportError, match="cancel failed"):
        runtime.close()
    assert runtime.quarantine["x-q"][0] is buf
    assert "x-q" not in runtime._tensor_owners
    assert not any(call[0] == "deregister_memory" for call in agent.calls)


def test_begin_tetris_lora_nixl_loads_runtime_without_ipc(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_TETRIS_LORA", "1")
    monkeypatch.setenv("TETRIS_LORA_ALL_TARGETS", "1")
    monkeypatch.delenv("TETRIS_LORA_RDMA_MODE", raising=False)
    seen: dict[str, object] = {}

    def _boom(*_args, **kwargs):
        seen["mode"] = kwargs.get("mode")
        seen["env_mode"] = os.environ.get("TETRIS_LORA_RDMA_MODE")
        raise TransportUnavailableError("DEPENDENCY-BLOCKED nixl import")

    monkeypatch.setattr(rdma_runtime, "TetrisRdmaRuntime", _boom)
    from vllm_omni.diffusion.lora.tetris_lora_runtime import begin_tetris_lora

    for transport in ("nixl", "rdma"):
        monkeypatch.setenv("TETRIS_LORA_TRANSPORT", transport)
        with pytest.raises(TransportUnavailableError, match="DEPENDENCY-BLOCKED"):
            begin_tetris_lora(
                "transformer_blocks.0.ff.net.0.proj",
                object(),  # type: ignore[arg-type]
                (((0, 1.0),),),
                {0: "/tmp/tetris-adapter-not-read"},
            )
        assert seen["mode"] == RdmaMode.NIXL
        assert seen["env_mode"] is None

    monkeypatch.setenv("TETRIS_LORA_RDMA_MODE", "mock")

    def _should_not_construct(*_args, **_kwargs):
        raise AssertionError("mock mode must not construct a fallback runtime")

    monkeypatch.setattr(rdma_runtime, "TetrisRdmaRuntime", _should_not_construct)
    with pytest.raises(TransportError, match="no MOCK/TCP fallback"):
        begin_tetris_lora(
            "transformer_blocks.0.ff.net.0.proj",
            object(),  # type: ignore[arg-type]
            (((0, 1.0),),),
            {0: "/tmp/tetris-adapter-not-read"},
        )

    monkeypatch.setenv("TETRIS_LORA_TRANSPORT", "ucx")
    with pytest.raises(TransportError, match="IMPLEMENTATION-INCOMPLETE"):
        begin_tetris_lora(
            "transformer_blocks.0.ff.net.0.proj",
            object(),  # type: ignore[arg-type]
            (((0, 1.0),),),
            {0: "/tmp/tetris-adapter-not-read"},
        )
