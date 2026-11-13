"""Auditable adapter for the patch execution contract in MixFusion.

The pinned MixFusion commit supplies the scheduler, patch transformer, and
split implementation, but it does not supply a LoRA fused CUDA/Triton kernel.
Omni therefore extends that source contract to Multi-LoRA while keeping the
device backend explicitly named as Omni's dense fused Triton operator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.lora.mixfusion_native_bridge import (
    MIXFUSION_SPLIT_ENTRY,
)

MIXFUSION_SOURCE_REPOSITORY = "/mnt/nvme1/cyn/MixFusion"
MIXFUSION_SOURCE_COMMIT = "e9aa899a1bd0885c100cc9b16011d987d3be4895"
MIXFUSION_SCHEDULER_ENTRY = "sduss.worker.scheduler.scheduler.Scheduler.schedule"
MIXFUSION_SIGNATURE_ENTRY = (
    "sduss.worker.scheduler.policy.Policy.restrict_to_adapter_signature"
)
MIXFUSION_DENOISE_ENTRY = (
    "sduss.worker.runner._model_runner.ModelRunner.exec_denoising_stage"
)
MIXFUSION_PATCH_ENTRY = (
    "sduss.model_executor.modules.transformer.PatchTransformer2DModel.forward"
)
MIXFUSION_SELF_ATTENTION_ENTRY = (
    "sduss.model_executor.modules.attention.PatchSelfAttention.forward"
)
MIXFUSION_CROSS_ATTENTION_ENTRY = (
    "sduss.model_executor.modules.attention.PatchCrossAttention.forward"
)


def _registered_kernel_name() -> str:
    return os.environ.get(
        "VLLM_OMNI_MIXFUSION_SOURCE_KERNEL_OP",
        "vllm_omni.mixfusion_lora_patch",
    )


def source_kernel_available() -> bool:
    """Return whether a device MixFusion LoRA operator is registered."""
    namespace, separator, operation = _registered_kernel_name().partition(".")
    if not separator or not namespace or not operation:
        return False
    try:
        return hasattr(getattr(torch.ops, namespace), operation)
    except AttributeError:
        return False


@dataclass(frozen=True)
class MixFusionSourcePatchContract:
    """Source-mapped scheduling and execution metadata for one invocation."""

    source_repository: str
    source_commit: str
    scheduler_entry: str
    signature_entry: str
    denoise_entry: str
    patch_entry: str
    self_attention_entry: str
    cross_attention_entry: str
    module_name: str | None
    patch_count: int
    patch_ordinals: tuple[int, ...]
    row_layout: tuple[int, ...]
    adapter_ids: tuple[tuple[int, ...], ...]
    rank_signatures: tuple[tuple[tuple[int, ...], ...], ...]
    execution_mode: str
    source_kernel_name: str
    source_kernel_available: bool
    attention_semantics: str
    merge_boundary: str
    native_scheduler_called: bool = False
    native_scheduler_entry: str = MIXFUSION_SCHEDULER_ENTRY
    native_patch_entry: str = MIXFUSION_PATCH_ENTRY
    native_split_called: bool = False
    native_split_entry: str = MIXFUSION_SPLIT_ENTRY
    native_split_input_shape: tuple[int, ...] | None = None
    native_split_output_shape: tuple[int, ...] | None = None
    native_split_patch_count: int = 0
    lora_kernel_backend: str = "none"
    lora_semantics: str = "none"

    def validate(self, input_rows: int) -> None:
        if not self.source_commit or len(self.source_commit) != 40:
            raise ValueError("MixFusion source commit must be a full SHA")
        if self.patch_count <= 0:
            raise ValueError("MixFusion source patch count must be positive")
        if self.patch_ordinals != tuple(range(self.patch_count)):
            raise ValueError("MixFusion patch ordinals must be contiguous")
        if sum(self.row_layout) != int(input_rows):
            raise ValueError(
                "MixFusion source row ownership does not cover input rows: "
                f"layout={self.row_layout}, rows={input_rows}"
            )
        if len(self.adapter_ids) != len(self.row_layout):
            raise ValueError("MixFusion source adapter rows do not match ownership")
        if self.attention_semantics != "full_sequence_attention":
            raise ValueError(
                "MixFusion adapter cannot run with per-tile attention semantics"
            )
        if self.merge_boundary != "attention_cannot_patch_full_sequence_boundary":
            raise ValueError("MixFusion attention merge boundary is not source mapped")

    def event_payload(self) -> dict[str, Any]:
        return {
            "mixfusion_source_repository": self.source_repository,
            "mixfusion_source_commit": self.source_commit,
            "mixfusion_scheduler_entry": self.scheduler_entry,
            "mixfusion_signature_entry": self.signature_entry,
            "mixfusion_denoise_entry": self.denoise_entry,
            "mixfusion_patch_entry": self.patch_entry,
            "mixfusion_attention_entries": [
                self.self_attention_entry,
                self.cross_attention_entry,
            ],
            "mixfusion_source_patch_count": self.patch_count,
            "mixfusion_source_patch_ordinals": list(self.patch_ordinals),
            "mixfusion_source_row_layout": list(self.row_layout),
            "mixfusion_source_adapter_ids": [
                list(entries) for entries in self.adapter_ids
            ],
            "mixfusion_source_rank_signatures": [
                [list(signature) for signature in slice_signatures]
                for slice_signatures in self.rank_signatures
            ],
            "mixfusion_source_execution_mode": self.execution_mode,
            "mixfusion_source_kernel_name": self.source_kernel_name,
            "mixfusion_source_kernel_available": self.source_kernel_available,
            "mixfusion_native_scheduler_called": self.native_scheduler_called,
            "mixfusion_native_scheduler_adapter_called": True,
            "mixfusion_native_scheduler_entry": self.native_scheduler_entry,
            "mixfusion_native_patch_entry": self.native_patch_entry,
            "mixfusion_native_split_called": self.native_split_called,
            "mixfusion_native_split_entry": self.native_split_entry,
            "mixfusion_native_split_input_shape": (
                list(self.native_split_input_shape)
                if self.native_split_input_shape is not None
                else None
            ),
            "mixfusion_native_split_output_shape": (
                list(self.native_split_output_shape)
                if self.native_split_output_shape is not None
                else None
            ),
            "mixfusion_native_split_patch_count": self.native_split_patch_count,
            "mixfusion_lora_kernel_backend": self.lora_kernel_backend,
            "mixfusion_lora_semantics": self.lora_semantics,
            "attention_semantics": self.attention_semantics,
            "attention_fallback": self.attention_semantics,
            "mixfusion_merge_boundary": self.merge_boundary,
        }


def build_source_patch_contract(
    *,
    patch_plan: Any | None,
    input_rows: int,
    row_layout: tuple[int, ...],
    adapter_ids: tuple[tuple[int, ...], ...],
    rank_signatures: tuple[tuple[tuple[int, ...], ...], ...],
    module_name: str | None,
    native_patch_schedule: Any | None = None,
) -> MixFusionSourcePatchContract:
    """Build the source-mapped contract used by the Omni host adapter.

    The source MixFusion contract is split into two auditable pieces here. The
    pinned source ``split_sample`` implementation is called directly by the
    bridge, while the source scheduler is not called until an Omni request
    adapter is available. Keeping those flags separate prevents source-order
    emulation from being reported as an upstream scheduler invocation.
    """
    patch_count = int(
        getattr(
            patch_plan,
            "patch_count",
            len(getattr(native_patch_schedule, "work_items", ())) or 1,
        )
    )
    if patch_plan is not None:
        if patch_count != len(getattr(patch_plan, "descriptors", ())):
            raise ValueError("MixFusion patch plan descriptor count is inconsistent")
        if patch_count <= 0:
            raise ValueError("MixFusion patch plan has no descriptors")
    contract = MixFusionSourcePatchContract(
        source_repository=MIXFUSION_SOURCE_REPOSITORY,
        source_commit=MIXFUSION_SOURCE_COMMIT,
        scheduler_entry=MIXFUSION_SCHEDULER_ENTRY,
        signature_entry=MIXFUSION_SIGNATURE_ENTRY,
        denoise_entry=MIXFUSION_DENOISE_ENTRY,
        patch_entry=MIXFUSION_PATCH_ENTRY,
        self_attention_entry=MIXFUSION_SELF_ATTENTION_ENTRY,
        cross_attention_entry=MIXFUSION_CROSS_ATTENTION_ENTRY,
        module_name=module_name,
        patch_count=patch_count,
        patch_ordinals=tuple(range(patch_count)),
        row_layout=tuple(int(rows) for rows in row_layout),
        adapter_ids=adapter_ids,
        rank_signatures=rank_signatures,
        execution_mode=(
            "source_patch_scheduler_plus_omni_multilora_triton"
            if patch_plan is not None or native_patch_schedule is not None
            else "source_patch_contract_module_adapter"
        ),
        source_kernel_name=_registered_kernel_name(),
        source_kernel_available=source_kernel_available(),
        attention_semantics="full_sequence_attention",
        merge_boundary="attention_cannot_patch_full_sequence_boundary",
        native_scheduler_called=bool(
            getattr(patch_plan, "source_scheduler_called", False)
            or getattr(native_patch_schedule, "source_scheduler_called", False)
        ),
        native_scheduler_entry=str(
            getattr(patch_plan, "source_scheduler_entry", MIXFUSION_SCHEDULER_ENTRY)
        ),
        native_patch_entry=str(
            getattr(patch_plan, "source_patch_entry", MIXFUSION_PATCH_ENTRY)
        ),
        native_split_called=bool(
            getattr(native_patch_schedule, "source_split_called", False)
        ),
        native_split_entry=str(
            getattr(
                native_patch_schedule,
                "source_split_entry",
                MIXFUSION_SPLIT_ENTRY,
            )
        ),
        native_split_input_shape=(
            tuple(int(value) for value in native_patch_schedule.source_split_input_shape)
            if getattr(native_patch_schedule, "source_split_input_shape", None)
            is not None
            else None
        ),
        native_split_output_shape=(
            tuple(int(value) for value in native_patch_schedule.source_split_output_shape)
            if getattr(native_patch_schedule, "source_split_output_shape", None)
            is not None
            else None
        ),
        native_split_patch_count=int(
            getattr(native_patch_schedule, "source_split_patch_count", 0)
        ),
        lora_kernel_backend=(
            "omni_dense_fused_lora_triton"
            if patch_plan is not None or native_patch_schedule is not None
            else "none"
        ),
        lora_semantics=(
            "source_patch_order_multi_lora_omni_kernel"
            if patch_plan is not None or native_patch_schedule is not None
            else "none"
        ),
    )
    contract.validate(input_rows)
    return contract


def require_source_kernel_for_formal() -> None:
    """Fail closed when a caller explicitly requires the real device kernel."""
    required = os.environ.get("VLLM_OMNI_REQUIRE_MIXFUSION_SOURCE_KERNEL", "")
    if required.strip().lower() in {"1", "true", "yes", "on"} and not source_kernel_available():
        raise RuntimeError(
            "formal MixFusion source-kernel mode requested, but no registered "
            f"device operator exists: {_registered_kernel_name()}"
        )


__all__ = [
    "MIXFUSION_DENOISE_ENTRY",
    "MIXFUSION_PATCH_ENTRY",
    "MIXFUSION_SCHEDULER_ENTRY",
    "MIXFUSION_SOURCE_COMMIT",
    "MIXFUSION_SOURCE_REPOSITORY",
    "MixFusionSourcePatchContract",
    "build_source_patch_contract",
    "require_source_kernel_for_formal",
    "source_kernel_available",
]
