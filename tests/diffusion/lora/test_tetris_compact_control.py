"""Unit tests for compact IPC control inbox (no GPU required)."""

from __future__ import annotations

import json
import threading
from multiprocessing.connection import Pipe

import pytest

from vllm_omni.diffusion.lora.tetris_lora_transport import (
    ControlInbox,
    CudaIpcTransfer,
    OperationIdAllocator,
    SessionPlanRegistry,
    TensorDescriptor,
    TransportError,
    WireMsgType,
    build_steady_tensor_frames,
    operation_batch_ack_enabled,
    reset_wire_json_counts,
    ring_index_for_sequence,
    unpack_wire_header,
    wire_json_counts,
)


def test_control_inbox_demuxes_ack_and_tensor_out_of_order() -> None:
    parent, child = Pipe()
    inbox = ControlInbox(parent)

    def peer() -> None:
        child.send_bytes(
            json.dumps(
                {
                    "kind": "tensor_ref",
                    "cache_key": "x",
                    "descriptor": "{}",
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )
        child.send_bytes(
            json.dumps(
                {
                    "kind": "ack",
                    "operation_id": "op-1",
                    "slot_generation": 3,
                    "consumer_complete": True,
                    "compact": True,
                    "event_index": 1,
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )

    thread = threading.Thread(target=peer)
    thread.start()
    ack = inbox.wait_ack("op-1", 3, timeout_s=5.0)
    assert ack["compact"] is True
    assert int(ack["event_index"]) == 1
    tensor = inbox.recv_tensor(timeout_s=5.0)
    assert tensor["kind"] == "tensor_ref"
    assert tensor["cache_key"] == "x"
    thread.join(timeout=5.0)
    parent.close()
    child.close()


def test_control_inbox_ack_batch_and_source_ack_piggyback() -> None:
    parent, child = Pipe()
    inbox = ControlInbox(parent)

    def peer() -> None:
        child.send_bytes(
            json.dumps(
                {
                    "kind": "ack_batch",
                    "acks": [
                        {
                            "kind": "ack",
                            "operation_id": "op-a",
                            "slot_generation": 1,
                            "consumer_complete": True,
                            "compact": True,
                            "event_index": 0,
                        },
                        {
                            "kind": "ack",
                            "operation_id": "op-b",
                            "slot_generation": 2,
                            "consumer_complete": True,
                            "compact": True,
                            "event_index": 1,
                        },
                    ],
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )
        child.send_bytes(
            json.dumps(
                {
                    "kind": "tensor_ref",
                    "cache_key": "delta",
                    "descriptor": "{}",
                    "source_ack": {
                        "kind": "ack",
                        "operation_id": "op-x",
                        "slot_generation": 7,
                        "consumer_complete": True,
                        "compact": True,
                        "event_index": 3,
                    },
                    "compute": {
                        "kind": "compute",
                        "scale": 1.0,
                        "target_module": "transformer_blocks.0.ff.net.0.proj",
                    },
                },
                separators=(",", ":"),
            ).encode("utf-8")
        )

    thread = threading.Thread(target=peer)
    thread.start()
    assert inbox.wait_ack("op-a", 1, timeout_s=5.0)["event_index"] == 0
    assert inbox.wait_ack("op-b", 2, timeout_s=5.0)["event_index"] == 1
    folded = inbox.recv_control(timeout_s=5.0)
    assert folded["kind"] == "tensor_ref"
    assert folded["compute"]["kind"] == "compute"
    assert inbox.wait_ack("op-x", 7, timeout_s=5.0)["event_index"] == 3
    thread.join(timeout=5.0)
    parent.close()
    child.close()


def test_compact_control_env_default_on(monkeypatch) -> None:
    from vllm_omni.diffusion.lora.tetris_lora_transport import compact_control_enabled

    monkeypatch.delenv("TETRIS_LORA_COMPACT_CONTROL", raising=False)
    assert compact_control_enabled() is True
    monkeypatch.setenv("TETRIS_LORA_COMPACT_CONTROL", "0")
    assert compact_control_enabled() is False
    assert operation_batch_ack_enabled() is False
    monkeypatch.delenv("TETRIS_LORA_COMPACT_CONTROL", raising=False)
    monkeypatch.delenv("TETRIS_LORA_OPERATION_BATCH_ACK", raising=False)
    assert operation_batch_ack_enabled() is True
    assert ring_index_for_sequence(65, 64) == 1


def _numeric_descriptor(operation_id: str) -> TensorDescriptor:
    return TensorDescriptor(
        session_epoch="epoch",
        operation_id=operation_id,
        request_id="request",
        batch_id="batch",
        denoise_step=3,
        layer_name="transformer_blocks.0.ff.net.0.proj",
        invocation_id=1,
        slot_id=0,
        slot_generation=1,
        shape=(2, 3),
        stride=(3, 1),
        dtype="torch.bfloat16",
        layout="contiguous",
        payload_bytes=12,
        buffer_token="buffer",
        adapter_id="adapter",
        adapter_version="version",
        rank=4,
        scale=1.0,
        branch_index=0,
        output_offset=0,
        completion_sequence=4,
        request_ids=("request",),
        row_spans=((0, 2, 0),),
        buffer_capacity=12,
        offset=0,
        alpha_folded=True,
    )


def test_operation_ids_are_monotonic_and_overflow_bumps_epoch() -> None:
    allocator = OperationIdAllocator()
    assert [allocator.next_id() for _ in range(3)] == [1, 2, 3]
    assert allocator.epoch == 1

    wrapped = OperationIdAllocator(start=1 << 64, epoch=4)
    assert wrapped.next_id() == 1
    assert wrapped.epoch == 5
    assert wrapped.next_id() == 2

    exhausted = OperationIdAllocator(start=1 << 64, epoch=(1 << 64) - 1)
    with pytest.raises(TransportError, match="epoch"):
        exhausted.next_id()


def test_register_plan_reuses_id_and_rejects_string_hash_ops() -> None:
    registry = SessionPlanRegistry()
    first = registry.register_plan({"layer_name": "proj", "rank": 4})
    second = registry.register_plan({"rank": 4, "layer_name": "proj"})
    assert first == second == 1
    assert registry.register_plan({"layer_name": "other", "rank": 4}) == 2
    descriptor = _numeric_descriptor("tetris:deadbeef")
    with pytest.raises(TransportError, match="decimal"):
        build_steady_tensor_frames(
            registry,
            set(),
            descriptor,
            session=1,
            cache_key="buffer",
            registration_id=1,
            first_registration=False,
        )


def test_binary_steady_state_has_zero_json_after_plan_register(monkeypatch) -> None:
    """Steady send/receive/ACK after plan registration does not call JSON."""
    monkeypatch.setenv("TETRIS_LORA_BINARY_WIRE", "1")
    monkeypatch.setenv("TETRIS_LORA_COUNT_WIRE_JSON", "1")
    reset_wire_json_counts()
    import vllm_omni.diffusion.lora.tetris_lora_transport as wire

    def _forbid_json(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("json on the steady-state wire")

    monkeypatch.setattr(wire.json, "dumps", _forbid_json)
    monkeypatch.setattr(wire.json, "loads", _forbid_json)

    parent, child = Pipe()
    inbox = ControlInbox(parent)
    producer = CudaIpcTransfer(parent, role="producer", device=0)
    producer.attach_inbox(inbox)
    consumer = CudaIpcTransfer(child, role="consumer", device=0)
    registry = SessionPlanRegistry()
    plans_sent: set[int] = set()
    first = _numeric_descriptor("1")
    plan_id, first_frames = build_steady_tensor_frames(
        registry,
        plans_sent,
        first,
        session=1,
        cache_key=first.buffer_token,
        registration_id=1,
        first_registration=False,
        compute={"kind": "compute", "scale": 1.0, "target_module": first.layer_name},
    )
    plans_sent.add(plan_id)
    assert len(first_frames) == 2
    plan_header, plan_body = unpack_wire_header(first_frames[0])
    assert plan_header.msg_type == int(WireMsgType.PLAN)
    assert plan_header.plan_id == plan_id
    unpacked = wire._msgpack_module().unpackb(plan_body, raw=False)
    assert unpacked["layer_name"] == first.layer_name
    steady_header, steady_body = unpack_wire_header(first_frames[1])
    assert steady_header.msg_type == int(WireMsgType.DOORBELL)
    assert steady_body == b""
    assert steady_header.op_id == 1
    assert b"transformer" not in first_frames[1]

    second = _numeric_descriptor("2")
    source_ack = {
        "kind": "ack",
        "operation_id": "7",
        "slot_generation": 1,
        "consumer_complete": True,
        "completion_sequence": 4,
        "compact": True,
        "event_index": 6,
    }
    second_id, second_frames = build_steady_tensor_frames(
        registry,
        plans_sent,
        second,
        session=1,
        cache_key=second.buffer_token,
        registration_id=1,
        first_registration=False,
        compute={"kind": "compute", "scale": 1.0, "target_module": second.layer_name},
        source_ack=source_ack,
    )
    assert second_id == plan_id
    assert len(second_frames) == 1
    second_header, _ = unpack_wire_header(second_frames[0])
    assert second_header.op_id == 2
    assert second_header.op_id > steady_header.op_id

    for frame in (*first_frames, *second_frames):
        child.send_bytes(frame)
    earlier = inbox.recv_tensor(timeout_s=5.0)
    later = inbox.recv_tensor(timeout_s=5.0)
    earlier_descriptor = TensorDescriptor.from_obj(earlier["descriptor"])
    later_descriptor = TensorDescriptor.from_obj(later["descriptor"])
    assert earlier_descriptor.operation_id == first.operation_id
    assert earlier_descriptor.layer_name == first.layer_name
    assert earlier_descriptor.request_id == "op:1"
    assert earlier_descriptor.batch_id == "session:1"
    assert earlier_descriptor.request_ids == ()
    assert later_descriptor.operation_id == second.operation_id
    assert later_descriptor.layer_name == second.layer_name
    assert later_descriptor.request_id == "op:2"
    assert later_descriptor.batch_id == "session:1"
    assert later_descriptor.request_ids == ()
    assert earlier["compute"]["target_module"] == first.layer_name
    assert earlier["kind"] == "tensor_ref"
    piggyback = inbox.wait_ack("7", 1, timeout_s=5.0)
    assert piggyback["compact"] is True
    assert piggyback["event_index"] == 6

    consumer.acknowledge(second, consumer_complete=True, batch=False)
    producer.wait_ack(second, timeout_s=5.0)
    ack_a = {
        "kind": "ack",
        "operation_id": "8",
        "slot_generation": 1,
        "consumer_complete": True,
        "completion_sequence": 4,
    }
    ack_b = {
        "kind": "ack",
        "operation_id": "9:delta",
        "slot_generation": 2,
        "consumer_complete": True,
        "completion_sequence": 5,
        "compact": True,
        "event_index": 3,
    }
    consumer._pending_acks.extend((ack_a, ack_b))
    assert consumer.flush_acks() == 2
    assert inbox.wait_ack("8", 1, timeout_s=5.0)["consumer_complete"] is True
    batched = inbox.wait_ack("9:delta", 2, timeout_s=5.0)
    assert batched["compact"] is True
    assert batched["event_index"] == 3
    assert wire_json_counts() == {"dumps": 0, "loads": 0}
    parent.close()
    child.close()


def test_wire_json_probe_counts_legacy_inbox_load(monkeypatch) -> None:
    monkeypatch.setenv("TETRIS_LORA_COUNT_WIRE_JSON", "1")
    reset_wire_json_counts()
    parent, child = Pipe()
    inbox = ControlInbox(parent)
    child.send_bytes(
        json.dumps(
            {
                "kind": "ack",
                "operation_id": "legacy",
                "slot_generation": 1,
                "consumer_complete": True,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    ack = inbox.wait_ack("legacy", 1, timeout_s=5.0)
    assert ack["operation_id"] == "legacy"
    assert wire_json_counts()["loads"] == 1
    assert wire_json_counts()["dumps"] == 0
    parent.close()
    child.close()
