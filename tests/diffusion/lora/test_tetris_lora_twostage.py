from __future__ import annotations

import pytest

from vllm_omni.diffusion.lora.tetris_lora_transport import TensorDescriptor
from vllm_omni.diffusion.lora.tetris_lora_twostage import (
    _branch_records,
    build_two_stage_descriptor,
)


def test_two_stage_descriptor_preserves_ragged_multi_lora_metadata(monkeypatch):
    monkeypatch.setenv("TETRIS_LORA_MAX_BRANCHES", "3")
    descriptor = build_two_stage_descriptor(
        module_name="transformer_blocks.0.attn.to_qkv",
        request_ids=("req-a", "req-b"),
        row_layout=(5, 3),
        relations=((0, 1), (1,)),
        scales=((0.5, 1.25), (0.75,)),
        rank_table=(2, 64),
        output_offset=0,
        output_features=9216,
        input_features=3072,
    )

    assert descriptor.row_layout == (5, 3)
    assert descriptor.ranks == ((2, 64), (64,))
    assert descriptor.h_bytes == 2 * (5 * 16 + 5 * 64 + 3 * 64)
    assert descriptor.rank_blocks == (16, 64)

    records, shrink_groups, expand_groups = _branch_records(descriptor)
    assert len(records) == 3
    assert [group["record_count"] for group in shrink_groups] == [1, 1, 1]
    assert [group["record_count"] for group in expand_groups] == [2, 1]


def test_two_stage_descriptor_enforces_branch_group_limit(monkeypatch):
    monkeypatch.setenv("TETRIS_LORA_MAX_BRANCHES", "2")
    with pytest.raises(ValueError, match="active LoRA branches"):
        build_two_stage_descriptor(
            module_name="transformer_blocks.0.attn.to_qkv",
            request_ids=("req",),
            row_layout=(4,),
            relations=((0, 1, 2),),
            scales=((1.0, 1.0, 1.0),),
            rank_table=(2, 2, 2),
            output_offset=0,
            output_features=9216,
            input_features=3072,
        )


def test_transport_descriptor_round_trip_keeps_two_stage_fields():
    descriptor = TensorDescriptor(
        session_epoch="epoch",
        operation_id="operation",
        request_id="request",
        batch_id="batch",
        denoise_step=0,
        layer_name="transformer_blocks.0.attn.to_qkv",
        invocation_id=1,
        slot_id=0,
        slot_generation=1,
        shape=(8, 3072),
        stride=(3072, 1),
        dtype="torch.bfloat16",
        layout="strided",
        payload_bytes=8 * 3072 * 2,
        buffer_token="buffer",
        adapter_id="0:/adapter-a,1:/adapter-b",
        adapter_version="version",
        rank=2,
        scale=1.0,
        branch_index=0,
        output_offset=0,
        completion_sequence=1,
        request_ids=("request", "request-2"),
        row_layout=(5, 3),
        request_composition=(((0, 1.0), (1, 0.5)), ((1, 1.25),)),
        request_ranks=((2, 64), (64,)),
        slice_rank_tables=((2, 64), (2, 64), (2, 64)),
        output_slices=(3072, 3072, 3072),
        buffer_capacity=8 * 3072 * 2,
    )

    assert TensorDescriptor.from_json(descriptor.to_json()) == descriptor
