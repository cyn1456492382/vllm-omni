"""Source-audited patch adapter coordinating Base and LoRA in Omni.

The pinned MixFusion source contributes patch scheduling/execution semantics.
It does not contain a LoRA fused kernel, so this runtime never reports the
Omni callback path as a source-kernel result.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.experiment_telemetry import emit_event
from vllm_omni.diffusion.lora.mixfusion_source_adapter import (
    MixFusionSourcePatchContract,
)
from vllm_omni.diffusion.lora.mixfusion_transport import (
    inspect_transport_capability,
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_VALID_VARIANTS = frozenset(
    {
        "naive",
        "system_b_mixfusion_patch",
        "multi_lora_both",
        "reference_both",
        "mixfusion_both",
        "dense_fused_both",
        "shared_input_both",
    }
)
_VARIANT_ALIASES = {
    "reference_both": "naive",
    "mixfusion_both": "system_b_mixfusion_patch",
}
_REMOVED_TOKEN_TILE_VARIANTS = frozenset(
    {"token_tile_both", "my_module_mixfusion_token"}
)
_VALID_TRANSPORTS = frozenset({"tcp", "p2p", "rdma"})
_VALID_RPC_GRANULARITIES = frozenset({"module", "block"})
_VALID_IMPLEMENTATIONS = frozenset(
    {
        "reference",
        "fused",
        "fused_pack_input",
        "fused_pack_lora",
        "dense_fused",
        "shared_input",
        "mixfusion",
        "torch",
        "tile_fused",
        "punica_segmented",
        "punica_ref",
        "sequential",
    }
)
_VALID_TILE_POLICIES = frozenset({"fixed", "dynamic"})
_VALID_RANK_POLICIES = frozenset({"separate", "bucket", "packed"})
_VALID_TILE_SIZES = frozenset({64, 128, 256, 512})


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


def runtime_variant() -> str:
    """返回当前实验路径，并拒绝含义不明确的路径名称。"""
    variant = os.environ.get("VLLM_OMNI_RUNTIME_VARIANT", "reference_both")
    variant = variant.strip().lower()
    if variant in _REMOVED_TOKEN_TILE_VARIANTS:
        raise ValueError(
            f"VLLM_OMNI_RUNTIME_VARIANT={variant!r} is unsupported: "
            "token-tile path removed; only System B MixFusion "
            "(mixfusion_both / system_b_mixfusion_patch) remains"
        )
    if variant not in _VALID_VARIANTS:
        raise ValueError(
            "VLLM_OMNI_RUNTIME_VARIANT must be one of "
            f"{sorted(_VALID_VARIANTS)}, got {variant!r}"
        )
    return variant


def canonical_runtime_variant() -> str:
    """Return the report-facing name for the selected execution path."""
    variant = runtime_variant()
    return _VARIANT_ALIASES.get(variant, variant)


def uses_mixfusion_runtime() -> bool:
    """Return whether Base+LoRA uses source patch ownership at full sequence."""
    return canonical_runtime_variant() == "system_b_mixfusion_patch"


def mixfusion_runtime_enabled() -> bool:
    """返回是否显式启用 Base+LoRA MixFusion runtime。"""
    return _is_true(os.environ.get("VLLM_OMNI_MIXFUSION_RUNTIME")) or uses_mixfusion_runtime()


def _enum_env(name: str, default: str, values: frozenset[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in values:
        raise ValueError(f"{name} must be one of {sorted(values)}, got {value!r}")
    return value


def lora_transport() -> str:
    """Return the selected LoRA transport without probing hardware."""
    return _enum_env("VLLM_OMNI_LORA_TRANSPORT", "tcp", _VALID_TRANSPORTS)


def lora_rpc_granularity() -> str:
    """Return the request coalescing granularity."""
    return _enum_env(
        "VLLM_OMNI_LORA_RPC_GRANULARITY",
        "module",
        _VALID_RPC_GRANULARITIES,
    )


def multi_lora_implementation() -> str:
    """Return the explicitly selected dense LoRA implementation."""
    implementation = _enum_env(
        "VLLM_OMNI_MULTI_LORA_IMPL", "fused", _VALID_IMPLEMENTATIONS
    )
    if implementation in {"fused_pack_input", "fused_pack_lora"}:
        raise ValueError(
            f"{implementation} is not implemented as a distinct kernel; "
            "use fused or tile_fused until a measured kernel exists"
        )
    return implementation


def multi_lora_tile_policy() -> str:
    """Return the fixed or dynamic token tile policy."""
    default = os.environ.get("VLLM_OMNI_BASE_TILE_POLICY", "fixed")
    if default.strip().lower() == "adaptive_dynamic":
        default = "dynamic"
    return _enum_env("VLLM_OMNI_MULTI_LORA_TILE_POLICY", default, _VALID_TILE_POLICIES)


def multi_lora_tile_size() -> int:
    """Return the supported token tile size."""
    value = os.environ.get("VLLM_OMNI_MULTI_LORA_TILE_SIZE", "64")
    try:
        tile_size = int(value)
    except ValueError as exc:
        raise ValueError(
            "VLLM_OMNI_MULTI_LORA_TILE_SIZE must be an integer"
        ) from exc
    if tile_size not in _VALID_TILE_SIZES:
        raise ValueError(
            "VLLM_OMNI_MULTI_LORA_TILE_SIZE must be one of "
            f"{sorted(_VALID_TILE_SIZES)}, got {tile_size}"
        )
    return tile_size


def multi_lora_rank_policy() -> str:
    """Return the rank grouping policy used by the LoRA operator."""
    return _enum_env(
        "VLLM_OMNI_MULTI_LORA_RANK_POLICY",
        "separate",
        _VALID_RANK_POLICIES,
    )


@dataclass(frozen=True)
class MixFusionDescriptor:
    """描述一个完整的 Base+LoRA 输出 ownership 单元。"""

    request_id: str | None
    step_id: int | None
    block_name: str | None
    module_name: str | None
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    row_layout: tuple[int, ...]
    base_operation: str
    adapter_ids: tuple[tuple[int, ...], ...]
    ranks: tuple[tuple[int, ...], ...]
    scales: tuple[tuple[float, ...], ...]
    token_tiles: tuple[tuple[int, int], ...]
    dependency_events: tuple[str, ...]
    stream: str | None
    output_add_owner: str
    slice_ranks: tuple[tuple[tuple[int, ...], ...], ...] = ()
    variant: str | None = None
    token_tile_segments: tuple[tuple[int, int, int], ...] = ()
    source_patch_contract: MixFusionSourcePatchContract | None = None

    def validate(self) -> None:
        """Validate row ownership and metadata before dispatching work."""
        if len(self.input_shape) != 2 or len(self.output_shape) != 2:
            raise ValueError("MixFusion descriptor tensors must be 2D")
        if any(int(rows) <= 0 for rows in self.row_layout):
            raise ValueError("MixFusion row ownership must be positive")
        if sum(self.row_layout) != self.input_shape[0]:
            raise ValueError("MixFusion row ownership does not cover inputs")
        request_count = len(self.row_layout)
        if not (
            len(self.adapter_ids) == request_count
            and len(self.ranks) == request_count
            and len(self.scales) == request_count
        ):
            raise ValueError("MixFusion request metadata lengths do not match")
        for adapters, ranks, scales in zip(
            self.adapter_ids, self.ranks, self.scales
        ):
            if not (len(adapters) == len(ranks) == len(scales)):
                raise ValueError("MixFusion adapter metadata lengths do not match")
            if any(int(rank) < 0 for rank in ranks):
                raise ValueError("MixFusion ranks must be non-negative")
        if self.slice_ranks and any(
            len(slice_ranks) != request_count for slice_ranks in self.slice_ranks
        ):
            raise ValueError("MixFusion slice rank metadata is invalid")
        previous_end = 0
        for start, end in self.token_tiles:
            if start != previous_end or end <= start:
                raise ValueError("MixFusion token tiles must be contiguous")
            previous_end = end
        if self.token_tiles and previous_end != self.input_shape[0]:
            raise ValueError("MixFusion token tiles do not cover inputs")


BaseForward = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]
LoRAForward = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        int,
        tuple[tuple[int, ...], ...],
        tuple[tuple[float, ...], ...],
    ],
    None,
]


class MixFusionRuntime:
    """在一个 descriptor 中执行 Base、LoRA residual 和就地 Add。"""

    def __init__(self, *, allow_reference: bool | None = None) -> None:
        # Explicit opt-in, or the configured reference/naive variant.
        if allow_reference is None:
            allow_reference = canonical_runtime_variant() in {"naive"}
        self._allow_reference = bool(allow_reference)

    def _allow_reference_lora(self) -> bool:
        return self._allow_reference

    def execute(
        self,
        descriptor: MixFusionDescriptor,
        inputs: torch.Tensor,
        output: torch.Tensor,
        *,
        base_forward: BaseForward | None,
        bias: torch.Tensor | None,
        lora_forward: LoRAForward | None,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
    ) -> None:
        """执行 descriptor 并把结果写入 output。"""
        descriptor.validate()
        if descriptor.source_patch_contract is not None:
            descriptor.source_patch_contract.validate(int(inputs.shape[0]))
        if inputs.ndim != 2 or output.ndim != 2:
            raise ValueError("MixFusion runtime expects flattened 2D tensors")
        if tuple(inputs.shape) != descriptor.input_shape:
            raise ValueError("MixFusion descriptor input shape does not match")
        if tuple(output.shape) != descriptor.output_shape:
            raise ValueError("MixFusion descriptor output shape does not match")
        metadata_length = len(descriptor.row_layout)
        if not (
            len(descriptor.adapter_ids) == metadata_length
            and len(descriptor.ranks) == metadata_length
            and len(descriptor.scales) == metadata_length
        ):
            raise ValueError("MixFusion request metadata lengths do not match")
        if len(lora_a_stacked) != len(lora_b_stacked):
            raise ValueError("LoRA A/B slice count does not match")

        # Validate LoRA execution mode before mutating output.
        if lora_a_stacked and lora_forward is None and not self._allow_reference_lora():
            raise RuntimeError(
                "MixFusion performance mode requires lora_forward; "
                "set MixFusionRuntime(allow_reference=True) or an explicit "
                "reference variant to use the mathematical fallback"
            )

        if base_forward is None:
            output.zero_()
        else:
            base_output = base_forward(inputs, bias)
            if tuple(base_output.shape) != tuple(output.shape):
                raise ValueError("Base output shape does not match MixFusion output")
            output.copy_(base_output)

        if not lora_a_stacked:
            if base_forward is None:
                raise ValueError("MixFusion requires Base or LoRA work")
            self._emit_dispatch_event(
                descriptor,
                base_executed=True,
                lora_executed=False,
            )
            return

        if descriptor.slice_ranks:
            if len(descriptor.slice_ranks) != len(lora_a_stacked):
                raise ValueError("MixFusion slice rank metadata does not match")
            slice_ranks = descriptor.slice_ranks
        else:
            slice_ranks = tuple(descriptor.ranks for _ in lora_a_stacked)
        for ranks in slice_ranks:
            if len(ranks) != metadata_length:
                raise ValueError("MixFusion slice request metadata is invalid")
            if any(
                len(request_ranks) != len(request_adapters)
                for request_ranks, request_adapters in zip(
                    ranks, descriptor.adapter_ids
                )
            ):
                raise ValueError("MixFusion slice rank metadata is invalid")

        output_offset = 0
        for slice_index, (a_stack, b_stack) in enumerate(
            zip(lora_a_stacked, lora_b_stacked)
        ):
            slice_size = int(b_stack.shape[2])
            slice_output = output[:, output_offset : output_offset + slice_size]
            relations = descriptor.adapter_ids
            scales = descriptor.scales
            if lora_forward is not None:
                lora_forward(
                    inputs,
                    slice_output,
                    slice_index,
                    relations,
                    scales,
                )
            else:
                self._execute_reference_lora(
                    inputs,
                    slice_output,
                    a_stack,
                    b_stack,
                    descriptor,
                    slice_ranks[slice_index],
                )
            output_offset += slice_size
        if output_offset != output.shape[1]:
            raise ValueError("LoRA slices do not cover MixFusion output features")

        self._emit_dispatch_event(
            descriptor,
            base_executed=base_forward is not None,
            lora_executed=True,
        )

    @staticmethod
    def _emit_dispatch_event(
        descriptor: MixFusionDescriptor,
        *,
        base_executed: bool,
        lora_executed: bool,
    ) -> None:
        suppress = os.environ.get(
            "VLLM_OMNI_SUPPRESS_LORA_EVENTS", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        active_slots = sorted(
            {
                int(slot)
                for request_adapters in descriptor.adapter_ids
                for slot in request_adapters
            }
        )
        # Fair M*: one compact verification event per process. Full per-module
        # JSONL costs hundreds of ms; the gate only needs adapter_ids/ranks.
        if suppress:
            if (
                base_executed
                and lora_executed
                and active_slots
                and not getattr(MixFusionRuntime, "_fair_verify_emitted", False)
            ):
                # Only mark emitted if telemetry actually has a sink; otherwise
                # retry on a later layer once the experiment dir is bound.
                from vllm_omni.diffusion.experiment_telemetry import (
                    _telemetry_dir,
                )

                if _telemetry_dir() is not None:
                    MixFusionRuntime._fair_verify_emitted = True
                    emit_event(
                        "lora_events",
                        "mixfusion_runtime_dispatch",
                        module_name=descriptor.module_name,
                        adapter_ids=[
                            list(item) for item in descriptor.adapter_ids
                        ],
                        ranks=[list(item) for item in descriptor.ranks],
                        slice_ranks=[
                            [list(item) for item in ranks]
                            for ranks in descriptor.slice_ranks
                        ],
                        base_executed=True,
                        lora_executed=True,
                        active_lora_slot_indices=active_slots,
                        fair_compact_once=True,
                        aggregate=True,
                    )
            return
        # M9: session-cache transport capability (static; not per-layer probe).
        transport_name = lora_transport()
        cap_cache = getattr(MixFusionRuntime, "_transport_capability_cache", None)
        if cap_cache is None:
            cap_cache = {}
            MixFusionRuntime._transport_capability_cache = cap_cache
        capability = cap_cache.get(transport_name)
        if capability is None:
            capability = inspect_transport_capability(transport_name)
            cap_cache[transport_name] = capability
        logical_rank = sum(
            int(rank)
            for request_ranks in descriptor.slice_ranks
            for ranks in request_ranks
            for rank in ranks
            if int(rank) > 0
        )
        rank_alignment = 16
        padded_rank = sum(
            ((int(rank) + rank_alignment - 1) // rank_alignment) * rank_alignment
            for request_ranks in descriptor.slice_ranks
            for ranks in request_ranks
            for rank in ranks
            if int(rank) > 0
        )
        tile_size = multi_lora_tile_size()
        valid_rows = sum(int(rows) for rows in descriptor.row_layout)
        padded_rows = sum(
            ((int(rows) + tile_size - 1) // tile_size) * tile_size
            for rows in descriptor.row_layout
        )
        emit_event(
            "lora_events",
            "mixfusion_runtime_dispatch",
            variant=descriptor.variant or runtime_variant(),
            module_name=descriptor.module_name,
            block_name=descriptor.block_name,
            request_id=descriptor.request_id,
            step_id=descriptor.step_id,
            input_shape=list(descriptor.input_shape),
            output_shape=list(descriptor.output_shape),
            row_layout=list(descriptor.row_layout),
            adapter_ids=[list(item) for item in descriptor.adapter_ids],
            ranks=[list(item) for item in descriptor.ranks],
            token_tiles=[list(item) for item in descriptor.token_tiles],
            token_tile_segments=[list(item) for item in descriptor.token_tile_segments],
            attention_tiled=False,
            slice_ranks=[
                [list(item) for item in ranks]
                for ranks in descriptor.slice_ranks
            ],
            dependency_events=list(descriptor.dependency_events),
            stream=descriptor.stream,
            output_add_owner=descriptor.output_add_owner,
            **(
                descriptor.source_patch_contract.event_payload()
                if descriptor.source_patch_contract is not None
                else {
                    "mixfusion_source_kernel_available": False,
                    "mixfusion_source_execution_mode": "not_source_mapped",
                    "attention_semantics": "full_sequence_attention",
                }
            ),
            base_executed=base_executed,
            lora_executed=lora_executed,
            base_lora_same_runtime=base_executed and lora_executed,
            actual_batch_size=len(descriptor.row_layout),
            active_lora_slots=len(active_slots),
            active_lora_slot_indices=active_slots,
            transport=lora_transport(),
            transport_available=capability.available,
            registered_gpu_buffer=capability.registered_gpu_buffer,
            payload_not_host_staged=capability.payload_not_host_staged,
            transport_capability_reason=capability.reason,
            rpc_granularity=lora_rpc_granularity(),
            tile_size=multi_lora_tile_size(),
            tile_policy=multi_lora_tile_policy(),
            rank_policy=multi_lora_rank_policy(),
            descriptor_count=1,
            runtime_queue_wait_ms=None,
            completion_overlap=None,
            host_dispatch_count=1,
            valid_rows=sum(descriptor.row_layout),
            padded_rows=padded_rows,
            rank_logical=logical_rank,
            rank_padded_to_16=padded_rank,
            rank_padding_ratio=(
                float(padded_rank) / float(logical_rank)
                if logical_rank
                else 1.0
            ),
            row_padding_ratio=(
                float(padded_rows) / float(valid_rows) if valid_rows else 1.0
            ),
            rank_diagnostic_status="analytical_metadata",
            hmma_status="not_measured",
            memory_traffic_status="not_measured",
        )

    @staticmethod
    def _execute_reference_lora(
        inputs: torch.Tensor,
        output: torch.Tensor,
        a_stack: torch.Tensor,
        b_stack: torch.Tensor,
        descriptor: MixFusionDescriptor,
        slice_ranks: tuple[tuple[int, ...], ...],
    ) -> None:
        """提供不依赖 Triton 的 correctness fallback。"""
        row_offset = 0
        for request_index, (row_count, relations, scales) in enumerate(
            zip(descriptor.row_layout, descriptor.adapter_ids, descriptor.scales)
        ):
            row_slice = slice(row_offset, row_offset + row_count)
            for relation_index, (slot, scale) in enumerate(zip(relations, scales)):
                ranks = slice_ranks[request_index]
                rank = int(ranks[relation_index])
                if rank <= 0:
                    continue
                hidden = inputs[row_slice] @ a_stack[slot, 0, :rank].transpose(0, 1)
                output[row_slice].add_(
                    (hidden @ b_stack[slot, 0, :, :rank].transpose(0, 1)) * float(scale)
                )
            row_offset += row_count


__all__ = [
    "MixFusionDescriptor",
    "MixFusionRuntime",
    "canonical_runtime_variant",
    "lora_rpc_granularity",
    "lora_transport",
    "mixfusion_runtime_enabled",
    "multi_lora_implementation",
    "multi_lora_rank_policy",
    "multi_lora_tile_policy",
    "multi_lora_tile_size",
    "runtime_variant",
    "uses_mixfusion_runtime",
]
