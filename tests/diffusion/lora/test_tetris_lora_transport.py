from __future__ import annotations

import pytest

from vllm_omni.diffusion.lora.tetris_lora_transport import (
    BufferState,
    SlotLease,
    TensorDescriptor,
)


def _descriptor() -> TensorDescriptor:
    return TensorDescriptor(
        session_epoch="epoch",
        operation_id="operation",
        request_id="request",
        batch_id="batch",
        denoise_step=0,
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
        completion_sequence=1,
        request_ids=("request",),
        row_spans=((0, 2, 0),),
        buffer_capacity=12,
        offset=0,
        alpha_folded=True,
    )


def test_descriptor_round_trip_preserves_shape_and_stride():
    descriptor = _descriptor()
    restored = TensorDescriptor.from_json(descriptor.to_json())
    assert restored == descriptor


def test_descriptor_rejects_payload_outside_declared_capacity():
    descriptor = _descriptor()
    invalid = TensorDescriptor(
        **{**descriptor.__dict__, "buffer_capacity": descriptor.payload_bytes - 1}
    )
    with pytest.raises(ValueError, match="capacity"):
        invalid.validate()


def test_slot_requires_consumer_completion_before_reuse():
    slot = SlotLease(0)
    assert slot.begin() == 1
    slot.mark_in_flight()
    slot.mark_target_visible()
    slot.mark_consumer_in_flight()
    slot.complete()
    assert slot.state is BufferState.FREE
    assert slot.begin() == 2


def test_quarantined_slot_cannot_be_reused():
    slot = SlotLease(0)
    slot.begin()
    slot.mark_in_flight()
    slot.quarantine()
    with pytest.raises(Exception, match="quarantined"):
        slot.begin()


def test_wait_ack_signal_requires_signal_ring():
    from vllm_omni.diffusion.lora.tetris_lora_transport import (
        CudaIpcTransfer,
        TransportError,
    )

    transfer = CudaIpcTransfer(object(), role="producer", device=0)
    with pytest.raises(TransportError, match="signal ring"):
        transfer.wait_ack_signal(_descriptor())
