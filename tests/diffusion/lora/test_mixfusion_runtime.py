from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.lora.mixfusion_runtime import (
    MixFusionDescriptor,
    MixFusionRuntime,
    multi_lora_rank_policy,
    multi_lora_tile_size,
    runtime_variant,
)
from vllm_omni.diffusion.lora.mixfusion_multi_lora import (
    MixFusionMultiLoRAPlan,
    apply_mixfusion_multilora,
    build_mixfusion_multilora_plan,
)
from vllm_omni.diffusion.lora.ops import LoRABackendError
from vllm_omni.diffusion.lora.layers.base_linear import (
    _selected_multi_lora_implementation,
)


def test_mixfusion_multilora_plan_uses_source_patch_rows() -> None:
    schedule = SimpleNamespace(
        work_items=(
            SimpleNamespace(
                request_index=0,
                row_start=0,
                column_start=0,
                valid_height=1,
                valid_width=2,
            ),
            SimpleNamespace(
                request_index=0,
                row_start=1,
                column_start=0,
                valid_height=1,
                valid_width=2,
            ),
        ),
        request_grid_shapes=((2, 2),),
        source_scheduler_called=True,
        source_split_called=True,
    )

    plan = build_mixfusion_multilora_plan((4,), schedule)

    assert plan.ownership_mode == "patch_row_spans"
    assert plan.row_spans == ((0, 2, 0), (2, 2, 0))


def test_mixfusion_multilora_plan_keeps_request_row_contract() -> None:
    schedule = SimpleNamespace(
        work_items=(
            SimpleNamespace(
                request_index=0,
                row_start=0,
                column_start=0,
                valid_height=1,
                valid_width=4,
            ),
        ),
        request_grid_shapes=((1, 4),),
        source_scheduler_called=True,
        source_split_called=True,
    )

    plan = build_mixfusion_multilora_plan((4,), schedule)

    assert plan.ownership_mode == "request_rows"
    assert plan.row_spans is None


def _cpu_lora_tensors(rows: int):
    inputs = torch.zeros(rows, 2)
    output = torch.zeros(rows, 2)
    a_stack = torch.zeros(1, 1, 4, 2)
    b_stack = torch.zeros(1, 1, 2, 4)
    return inputs, output, a_stack, b_stack


def test_two_stage_refuses_patch_row_spans(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_LORA_BACKEND", "two_stage_tc")
    monkeypatch.delenv("VLLM_OMNI_FORCE_DENSE_FUSED", raising=False)
    plan = MixFusionMultiLoRAPlan(
        row_layout=(4,),
        row_spans=((0, 2, 0), (2, 2, 0)),
        patch_ordinals=(0, 1),
        source_scheduler_called=True,
        source_split_called=True,
        ownership_mode="patch_row_spans",
    )
    inputs, output, a_stack, b_stack = _cpu_lora_tensors(4)

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("two_stage must not run on patch_row_spans")

    monkeypatch.setattr(
        "vllm_omni.diffusion.lora.mixfusion_multi_lora.execute_two_stage_lora",
        _must_not_run,
    )
    with pytest.raises(LoRABackendError, match="request-contiguous"):
        apply_mixfusion_multilora(
            inputs,
            output,
            a_stack,
            b_stack,
            plan,
            ((0,),),
            ((1.0,),),
            (4,),
        )


def test_two_stage_accepts_request_rows(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_LORA_BACKEND", "two_stage_tc")
    monkeypatch.delenv("VLLM_OMNI_FORCE_DENSE_FUSED", raising=False)
    plan = MixFusionMultiLoRAPlan(
        row_layout=(4,),
        row_spans=None,
        patch_ordinals=(0,),
        source_scheduler_called=True,
        source_split_called=True,
        ownership_mode="request_rows",
    )
    inputs, output, a_stack, b_stack = _cpu_lora_tensors(4)

    def _fake(*_args, output=None, **_kwargs):
        return torch.zeros_like(output), SimpleNamespace(rank_blocks=(16,))

    monkeypatch.setattr(
        "vllm_omni.diffusion.lora.mixfusion_multi_lora.execute_two_stage_lora",
        _fake,
    )
    dispatch = apply_mixfusion_multilora(
        inputs,
        output,
        a_stack,
        b_stack,
        plan,
        ((0,),),
        ((1.0,),),
        (4,),
    )
    assert dispatch.kernel_backend == "tetris_two_stage_tc"
    assert dispatch.ownership_mode == "request_rows"
    assert dispatch.patch_row_span_count == 0
    assert dispatch.rank_bucket_blocks == (16,)


def test_mixfusion_runtime_preserves_base_lora_add() -> None:
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    base_weight = torch.eye(2)
    lora_a = torch.tensor([[[[1.0, 0.0]]]])
    lora_b = torch.tensor([[[[2.0], [3.0]]]])
    output = torch.empty_like(inputs)
    descriptor = MixFusionDescriptor(
        request_id="request-0",
        step_id=0,
        block_name="block-0",
        module_name="module-0",
        input_shape=(2, 2),
        output_shape=(2, 2),
        row_layout=(2,),
        base_operation="torch.mm",
        adapter_ids=((0,),),
        ranks=((1,),),
        scales=((0.5,),),
        token_tiles=((0, 2),),
        dependency_events=("input_ready",),
        stream=None,
        output_add_owner="mixfusion_runtime",
    )

    MixFusionRuntime().execute(
        descriptor,
        inputs,
        output,
        base_forward=lambda value, bias: value @ base_weight.t(),
        bias=None,
        lora_forward=None,
        lora_a_stacked=(lora_a,),
        lora_b_stacked=(lora_b,),
    )

    expected = inputs + (inputs @ lora_a[0, 0].t()) @ lora_b[0, 0].t() * 0.5
    torch.testing.assert_close(output, expected)


def test_mixfusion_runtime_supports_base_only() -> None:
    inputs = torch.tensor([[1.0, 2.0]])
    output = torch.empty((1, 2))
    descriptor = MixFusionDescriptor(
        request_id="request-0",
        step_id=0,
        block_name="block-0",
        module_name="module-0",
        input_shape=(1, 2),
        output_shape=(1, 2),
        row_layout=(1,),
        base_operation="torch.mm",
        adapter_ids=((),),
        ranks=((),),
        scales=((),),
        token_tiles=((0, 1),),
        dependency_events=("input_ready",),
        stream=None,
        output_add_owner="mixfusion_runtime",
    )

    MixFusionRuntime().execute(
        descriptor,
        inputs,
        output,
        base_forward=lambda value, bias: value + 1,
        bias=None,
        lora_forward=None,
        lora_a_stacked=(),
        lora_b_stacked=(),
    )

    torch.testing.assert_close(output, inputs + 1)


def test_mixfusion_runtime_supports_lora_only() -> None:
    inputs = torch.tensor([[1.0, 2.0]])
    lora_a = torch.tensor([[[[1.0, 0.0]]]])
    lora_b = torch.tensor([[[[2.0], [3.0]]]])
    output = torch.empty((1, 2))
    descriptor = MixFusionDescriptor(
        request_id="request-0",
        step_id=0,
        block_name="block-0",
        module_name="module-0",
        input_shape=(1, 2),
        output_shape=(1, 2),
        row_layout=(1,),
        base_operation="torch.mm",
        adapter_ids=((0,),),
        ranks=((1,),),
        scales=((0.5,),),
        token_tiles=((0, 1),),
        dependency_events=("input_ready",),
        stream=None,
        output_add_owner="mixfusion_runtime",
    )

    MixFusionRuntime().execute(
        descriptor,
        inputs,
        output,
        base_forward=None,
        bias=None,
        lora_forward=None,
        lora_a_stacked=(lora_a,),
        lora_b_stacked=(lora_b,),
    )

    expected = (inputs @ lora_a[0, 0].t()) @ lora_b[0, 0].t() * 0.5
    torch.testing.assert_close(output, expected)


def test_mixfusion_runtime_uses_slice_specific_ranks() -> None:
    inputs = torch.tensor([[1.0, 2.0]])
    first_a = torch.tensor([[[[1.0, 0.0]]]])
    first_b = torch.tensor([[[[2.0], [3.0]]]])
    second_a = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    second_b = torch.tensor([[[[4.0, 0.0], [0.0, 5.0]]]])
    output = torch.empty((1, 4))
    descriptor = MixFusionDescriptor(
        request_id="request-0",
        step_id=0,
        block_name="block-0",
        module_name="module-0",
        input_shape=(1, 2),
        output_shape=(1, 4),
        row_layout=(1,),
        base_operation="torch.mm",
        adapter_ids=((0,),),
        ranks=((1,),),
        scales=((1.0,),),
        token_tiles=((0, 1),),
        dependency_events=("input_ready",),
        stream=None,
        output_add_owner="mixfusion_runtime",
        slice_ranks=(((1,),), ((2,),)),
    )

    MixFusionRuntime().execute(
        descriptor,
        inputs,
        output,
        base_forward=lambda value, bias: torch.cat((value, value), dim=1),
        bias=None,
        lora_forward=None,
        lora_a_stacked=(first_a, second_a),
        lora_b_stacked=(first_b, second_b),
    )

    expected = torch.cat(
        (
            inputs
            + (inputs @ first_a[0, 0].t()) @ first_b[0, 0].t(),
            inputs + (inputs @ second_a[0, 0].t()) @ second_b[0, 0].t(),
        ),
        dim=1,
    )
    torch.testing.assert_close(output, expected)


def test_mixfusion_descriptor_rejects_gapped_token_tiles() -> None:
    descriptor = MixFusionDescriptor(
        request_id=None,
        step_id=None,
        block_name=None,
        module_name=None,
        input_shape=(4, 2),
        output_shape=(4, 2),
        row_layout=(4,),
        base_operation="torch.mm",
        adapter_ids=((),),
        ranks=((),),
        scales=((),),
        token_tiles=((0, 2), (3, 4)),
        dependency_events=(),
        stream=None,
        output_add_owner="mixfusion_runtime",
    )

    try:
        descriptor.validate()
    except ValueError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("gapped token tiles must be rejected")


def test_mixfusion_configuration_uses_explicit_rank_policy(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_RUNTIME_VARIANT", "mixfusion_both")
    monkeypatch.setenv("VLLM_OMNI_MULTI_LORA_TILE_SIZE", "128")
    monkeypatch.setenv("VLLM_OMNI_MULTI_LORA_RANK_POLICY", "bucket")

    assert runtime_variant() == "mixfusion_both"
    assert multi_lora_tile_size() == 128
    assert multi_lora_rank_policy() == "bucket"


def test_mixfusion_performance_mode_rejects_missing_lora_forward(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_RUNTIME_VARIANT", "mixfusion_both")
    inputs = torch.tensor([[1.0, 2.0]])
    lora_a = torch.tensor([[[[1.0, 0.0]]]])
    lora_b = torch.tensor([[[[2.0], [3.0]]]])
    output = torch.empty((1, 2))
    descriptor = MixFusionDescriptor(
        request_id="request-0",
        step_id=0,
        block_name="block-0",
        module_name="module-0",
        input_shape=(1, 2),
        output_shape=(1, 2),
        row_layout=(1,),
        base_operation="torch.mm",
        adapter_ids=((0,),),
        ranks=((1,),),
        scales=((0.5,),),
        token_tiles=((0, 1),),
        dependency_events=("input_ready",),
        stream=None,
        output_add_owner="mixfusion_runtime",
    )
    try:
        MixFusionRuntime().execute(
            descriptor,
            inputs,
            output,
            base_forward=None,
            bias=None,
            lora_forward=None,
            lora_a_stacked=(lora_a,),
            lora_b_stacked=(lora_b,),
        )
    except RuntimeError as exc:
        assert "requires lora_forward" in str(exc)
    else:
        raise AssertionError("performance mode must reject missing lora_forward")


def test_system_b_variant_routes_lora_through_mixfusion(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_RUNTIME_VARIANT", "system_b_mixfusion_patch")
    monkeypatch.setenv("VLLM_OMNI_MULTI_LORA_IMPL", "fused")
    assert _selected_multi_lora_implementation() == "mixfusion"


def test_token_tile_variant_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_RUNTIME_VARIANT", "token_tile_both")
    try:
        runtime_variant()
    except ValueError as exc:
        assert "token-tile path removed" in str(exc)
    else:
        raise AssertionError("token_tile_both must fail closed")


def test_my_module_token_variant_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_RUNTIME_VARIANT", "my_module_mixfusion_token")
    try:
        runtime_variant()
    except ValueError as exc:
        assert "token-tile path removed" in str(exc)
    else:
        raise AssertionError("my_module_mixfusion_token must fail closed")
