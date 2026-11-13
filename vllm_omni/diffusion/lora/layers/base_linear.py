# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re
from collections.abc import Callable

import torch
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.ops.triton_ops import (
    LoRAKernelMeta,
    lora_expand,
    lora_shrink,
)
from vllm_omni.diffusion.lora.multi_lora import (
    MultiLoRAAdapter,
    execute_fused_lora,
    execute_tiled_fused_lora,
    execute_lora,
    prepare_lora_batch,
)
from vllm_omni.diffusion.lora.ops.multi_lora_tile_plan import (
    build_multi_lora_tile_plan,
)
from vllm_omni.diffusion.experiment_telemetry import (
    emit_event,
    emit_event_rows,
    tensor_metadata,
)
from vllm_omni.diffusion.lora.lora_compute_breakdown import (
    finish_interval,
    record_lora_flops,
    start_interval,
)
from vllm_omni.diffusion.lora.mixfusion_runtime import (
    MixFusionDescriptor,
    MixFusionRuntime,
    mixfusion_runtime_enabled,
    multi_lora_implementation,
    multi_lora_rank_policy,
    multi_lora_tile_size,
    runtime_variant,
    uses_mixfusion_runtime,
)
from vllm_omni.diffusion.lora.mixfusion_multi_lora import (
    apply_mixfusion_multilora,
    build_mixfusion_multilora_plan,
)
from vllm_omni.diffusion.lora.mixfusion_native_bridge import (
    active_native_patch_schedule,
)
from vllm_omni.diffusion.lora.mixfusion_source_adapter import (
    build_source_patch_contract,
)


def _multi_lora_operator_enabled() -> bool:
    if runtime_variant() in {
        "system_b_mixfusion_patch",
        "multi_lora_both",
        "mixfusion_both",
        "dense_fused_both",
        "shared_input_both",
    }:
        return True
    value = os.environ.get("VLLM_OMNI_ENABLE_MULTI_LORA_OPERATOR", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _lora_diagnostic_mode() -> str:
    enabled = os.environ.get("VLLM_OMNI_ENABLE_LORA_DIAGNOSTIC", "")
    if enabled.strip().lower() not in {"1", "true", "yes", "on"}:
        return "normal"
    return os.environ.get("VLLM_OMNI_LORA_DIAGNOSTIC_MODE", "normal").strip().lower()


def _reject_removed_token_tile_env() -> None:
    """Fail closed when legacy token-tile / patch-tiling env knobs are set."""
    base_tiling = os.environ.get("VLLM_OMNI_BASE_TILING", "").strip().lower()
    patch_tiling = os.environ.get("VLLM_OMNI_PATCH_TILING", "").strip().lower()
    if base_tiling in {"1", "true", "yes", "on"} or patch_tiling in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise ValueError(
            "token-tile path removed; only System B MixFusion "
            "(mixfusion_both / system_b_mixfusion_patch) remains"
        )


def _selected_multi_lora_implementation() -> str:
    """Resolve the device backend while preserving MixFusion ownership modes."""
    implementation = multi_lora_implementation()
    if implementation == "reference":
        implementation = "torch"
    if uses_mixfusion_runtime():
        return "mixfusion"
    return implementation


class DiffusionBaseLinearLayerWithLoRA(BaseLinearLayerWithLoRA):
    """
    Diffusion-specific base that overrides apply() to use direct torch matmul
    instead of punica_wrapper.

    punica_wrapper is used to hold multiple LoRA slots and slices efficiently.

    This matches the semantics of PunicaWrapperGPU.add_lora_linear():
    - Shrink: buffer = (x @ lora_a.T)
    - Expand: y += buffer @ lora_b.T

    All other functionality (weight management, TP slicing, forward logic)
    is inherited from vLLM's BaseLinearLayerWithLoRA.
    """

    @staticmethod
    def _lora_activation_enabled(module_name: str | None) -> bool:
        """Apply optional experiment-wide timestep/block LoRA masks."""
        step_start = int(os.environ.get("VLLM_OMNI_LORA_STEP_START", "0"))
        step_end = int(os.environ.get("VLLM_OMNI_LORA_STEP_END", "-1"))
        block_start = int(os.environ.get("VLLM_OMNI_LORA_BLOCK_START", "0"))
        block_end = int(os.environ.get("VLLM_OMNI_LORA_BLOCK_END", "-1"))
        try:
            from vllm_omni.diffusion.lora.dit_lora_overlap import runtime

            step_index = int(runtime().step_index)
        except Exception:
            step_index = 0
        if step_end >= step_start and not step_start <= step_index <= step_end:
            return False
        if block_end < block_start:
            return True
        match = re.search(
            r"(?:layers|transformer_blocks|single_transformer_blocks|"
            r"noise_refiner|context_refiner)\.(\d+)",
            str(module_name),
        )
        if match is None:
            return False
        block_index = int(match.group(1))
        return block_start <= block_index <= block_end

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config,
        model_config=None,
    ) -> None:
        super().create_lora_weights(max_loras, lora_config, model_config)
        # Keep a direct reference for attribute forwarding: `base_layer` is a
        # registered submodule (stored under `_modules`), so direct access via
        # `object.__getattribute__` will not find it. We stash a ref in
        # `__dict__` for robust lookups in `__getattr__`.
        modules = object.__getattribute__(self, "_modules")
        base_layer = modules.get("base_layer") or object.__getattribute__(self, "__dict__").get("base_layer")
        object.__setattr__(self, "_diffusion_base_layer_ref", base_layer)
        n_slices = getattr(self, "n_slices", 1)
        self._diffusion_lora_active_slices = (False,) * int(n_slices)
        self._diffusion_lora_slot_ranks_by_slice = tuple(
            [0] * int(max_loras) for _ in range(int(n_slices))
        )
        self._diffusion_lora_batch_row_layout = None

    def _refresh_diffusion_lora_active_slices(self) -> None:
        rank_tables = getattr(self, "_diffusion_lora_slot_ranks_by_slice", None)
        if rank_tables is None:
            return
        self._diffusion_lora_active_slices = tuple(
            any(int(rank) > 0 for rank in ranks) for ranks in rank_tables
        )

    def reset_lora(self, index: int):
        super().reset_lora(index)
        rank_tables = getattr(self, "_diffusion_lora_slot_ranks_by_slice", None)
        if rank_tables is not None:
            for ranks in rank_tables:
                ranks[index] = 0
            self._refresh_diffusion_lora_active_slices()
        else:
            n_slices = getattr(self, "n_slices", 1)
            self._diffusion_lora_active_slices = (False,) * int(n_slices)

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor | None],
        lora_b: torch.Tensor | list[torch.Tensor | None],
    ):
        super().set_lora(index, lora_a, lora_b)  # type: ignore[arg-type]

        n_slices = getattr(self, "n_slices", 1)
        if isinstance(lora_a, list) or isinstance(lora_b, list):
            assert isinstance(lora_a, list)
            assert isinstance(lora_b, list)
            rank_tables = getattr(
                self, "_diffusion_lora_slot_ranks_by_slice", None
            )
            if rank_tables is not None:
                for slice_index, a_i in enumerate(lora_a[:n_slices]):
                    rank_tables[slice_index][index] = (
                        0 if a_i is None else int(a_i.shape[0])
                    )
        else:
            # Single-slice layer.
            rank_tables = getattr(
                self, "_diffusion_lora_slot_ranks_by_slice", None
            )
            if rank_tables is not None:
                rank_tables[0][index] = int(lora_a.shape[0])
        self._refresh_diffusion_lora_active_slices()

    def set_batch_slot_indices(self, slot_indices: tuple[int | None, ...]) -> None:
        """Set one active LoRA slot per logical request in the next forward."""
        self._diffusion_lora_batch_slot_indices = tuple(slot_indices)
        self._diffusion_lora_batch_composition = tuple(
            () if slot is None else ((int(slot), 1.0),)
            for slot in slot_indices
        )
        self._diffusion_lora_batch_row_layout = None
        self._diffusion_lora_batch_usage_emitted = False

    def set_batch_adapter_composition(
        self,
        composition: tuple[tuple[tuple[int, float], ...], ...],
        row_layout: tuple[int, ...] | None = None,
    ) -> None:
        """Set ordered resident adapter slots for every request in a batch."""
        if row_layout is not None:
            row_layout = tuple(int(rows) for rows in row_layout)
            if len(row_layout) != len(composition):
                raise ValueError(
                    "Multi-LoRA row layout and composition must have equal "
                    f"request counts: {len(row_layout)} != {len(composition)}"
                )
            if any(rows < 0 for rows in row_layout):
                raise ValueError("Multi-LoRA row counts must be non-negative")
        normalized: list[tuple[tuple[int, float], ...]] = []
        for request_index, entries in enumerate(composition):
            request_entries: list[tuple[int, float]] = []
            seen_slots: set[int] = set()
            for slot, scale in entries:
                slot = int(slot)
                scale = float(scale)
                if slot < 0:
                    raise ValueError(
                        f"Multi-LoRA slot must be non-negative: request={request_index}"
                    )
                if slot in seen_slots:
                    raise ValueError(
                        f"duplicate Multi-LoRA slot {slot} in request "
                        f"{request_index}"
                    )
                seen_slots.add(slot)
                request_entries.append((slot, scale))
            normalized.append(tuple(request_entries))
        self._diffusion_lora_batch_composition = tuple(normalized)
        self._diffusion_lora_batch_slot_indices = tuple(
            entries[0][0] if len(entries) == 1 else None
            for entries in self._diffusion_lora_batch_composition
        )
        self._diffusion_lora_batch_row_layout = row_layout
        self._diffusion_lora_batch_usage_emitted = False

    def set_batch_request_layout(self, row_layout: tuple[int, ...]) -> None:
        """Set flattened activation rows owned by each logical request."""
        normalized = tuple(int(rows) for rows in row_layout)
        composition = getattr(self, "_diffusion_lora_batch_composition", None)
        if composition is None or len(normalized) != len(composition):
            raise ValueError(
                "Multi-LoRA row layout must match the active request composition"
            )
        if any(rows < 0 for rows in normalized):
            raise ValueError("Multi-LoRA row counts must be non-negative")
        self._diffusion_lora_batch_row_layout = normalized

    def _filter_composition_for_slice(
        self,
        composition: tuple[tuple[tuple[int, float], ...], ...],
        slice_index: int,
    ) -> tuple[tuple[tuple[int, float], ...], ...]:
        """Drop adapters that do not target the current packed slice."""
        rank_tables = getattr(self, "_diffusion_lora_slot_ranks_by_slice", None)
        if rank_tables is None or slice_index >= len(rank_tables):
            return composition
        rank_table = rank_tables[slice_index]
        filtered: list[tuple[tuple[int, float], ...]] = []
        for request_index, entries in enumerate(composition):
            request_entries: list[tuple[int, float]] = []
            for slot, scale in entries:
                slot = int(slot)
                if slot < 0 or slot >= len(rank_table):
                    raise ValueError(
                        "Multi-LoRA slot exceeds slice rank metadata: "
                        f"request={request_index}, slice={slice_index}, slot={slot}"
                    )
                if int(rank_table[slot]) > 0:
                    request_entries.append((slot, float(scale)))
            filtered.append(tuple(request_entries))
        return tuple(filtered)

    def _get_batch_row_layout(
        self,
        x_flat: torch.Tensor,
        composition: tuple[tuple[tuple[int, float], ...], ...],
    ) -> tuple[int, ...]:
        if not composition:
            return (int(x_flat.shape[0]),)
        configured = getattr(self, "_diffusion_lora_batch_row_layout", None)
        if configured is None:
            if x_flat.shape[0] % len(composition):
                raise ValueError(
                    "Multi-LoRA composition requires an explicit ragged row "
                    "layout when flattened activations are not uniform: "
                    f"rows={x_flat.shape[0]}, requests={len(composition)}"
                )
            rows = x_flat.shape[0] // len(composition)
            return (rows,) * len(composition)
        if len(configured) == len(composition) and sum(configured) == x_flat.shape[0]:
            return configured
        raise ValueError(
            "Multi-LoRA row layout does not match flattened activations: "
            f"layout={configured}, rows={x_flat.shape[0]}"
        )

    def _apply_multi_lora_operator(
        self,
        module_name: str,
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        output_slices: tuple[int, ...],
        composition: tuple[tuple[tuple[int, float], ...], ...],
    ) -> torch.Tensor:
        if not composition or not any(composition):
            return y_flat
        row_layout = self._get_batch_row_layout(x_flat, composition)
        applied_slot_indices: set[int] = set()
        for slice_index, slice_size in enumerate(output_slices):
            if (
                slice_index < len(self._diffusion_lora_active_slices)
                and not self._diffusion_lora_active_slices[slice_index]
            ):
                continue
            slice_composition = self._filter_composition_for_slice(
                composition, slice_index
            )
            if not any(slice_composition):
                continue
            a_stack = self.lora_a_stacked[slice_index]
            b_stack = self.lora_b_stacked[slice_index]
            registry: dict[int, MultiLoRAAdapter] = {}
            relation: list[list[int]] = []
            for entries in slice_composition:
                request_relation: list[int] = []
                for slot, _scale in entries:
                    if slot >= a_stack.shape[0]:
                        raise ValueError(
                            f"Multi-LoRA slot {slot} exceeds layer capacity "
                            f"{a_stack.shape[0]}"
                        )
                    request_relation.append(slot)
                    if slot not in registry:
                        rank_table = getattr(
                            self, "_diffusion_lora_slot_ranks_by_slice", ()
                        )
                        rank = (
                            int(rank_table[slice_index][slot])
                            if slice_index < len(rank_table)
                            and slot < len(rank_table[slice_index])
                            else int(a_stack.shape[-2])
                        )
                        rank = max(0, rank)
                        registry[slot] = MultiLoRAAdapter(
                            adapter_id=slot,
                            lora_a=a_stack[slot, 0, :rank, :],
                            lora_b=b_stack[slot, 0, :, :rank],
                        )
                relation.append(request_relation)
            request_scales = [
                [scale for _slot, scale in entries]
                for entries in slice_composition
            ]
            plan = prepare_lora_batch(
                module_identity=f"{module_name}[slice={slice_index}]",
                input_row_layout=row_layout,
                request_to_adapter_relation=relation,
                adapter_registry_snapshot=registry,
                request_scales=request_scales,
                execution_ownership=(
                    f"tp={getattr(self, 'tp_rank', 0)}/"
                    f"{getattr(self, 'tp_size', 1)}"
                ),
            )
            output_offset = sum(output_slices[:slice_index])

            def time_gemm(
                gemm_name: str, operation: Callable[[], torch.Tensor]
            ) -> torch.Tensor:
                interval = start_interval(
                    "pure_lora_gemm",
                    module_name=module_name,
                    execution_location="coupled",
                    executor="execute_lora",
                    gemm=gemm_name,
                )
                result = operation()
                finish_interval(interval)
                return result

            execute_lora(
                x_flat,
                y_flat[:, output_offset : output_offset + slice_size],
                plan,
                gemm_timer=time_gemm,
            )
            applied_slot_indices.update(
                slot for entries in slice_composition for slot, _scale in entries
            )
        if applied_slot_indices:
            # #region agent log
            if (
                __import__("os").environ.get("CURSOR_AGENT_DEBUG") == "68c61b"
                and not getattr(self, "_dbg_torch_apply_logged", False)
            ):
                try:
                    import json as _json
                    import time as _time

                    with open(
                        "/workspace/.cursor/debug-68c61b.log",
                        "a",
                        encoding="utf-8",
                    ) as _dbg:
                        _dbg.write(
                            _json.dumps(
                                {
                                    "sessionId": "68c61b",
                                    "hypothesisId": "H3",
                                    "location": "base_linear.py:_apply_multi_lora_operator",
                                    "message": "torch multi_lora apply once",
                                    "data": {
                                        "module": module_name,
                                        "x_device": str(x_flat.device),
                                        "x_is_cuda": bool(x_flat.is_cuda),
                                        "x_rows": int(x_flat.shape[0]),
                                        "row_layout": list(row_layout),
                                        "composition_len": len(composition),
                                        "applied_slots": sorted(applied_slot_indices),
                                        "suppress": __import__("os").environ.get(
                                            "VLLM_OMNI_SUPPRESS_LORA_EVENTS"
                                        ),
                                    },
                                    "timestamp": int(_time.time() * 1000),
                                    "runId": "diag-pre",
                                }
                            )
                            + "\n"
                        )
                    self._dbg_torch_apply_logged = True
                except Exception:
                    pass
            # #endregion
            # Fair/suppress: one compact aggregate proof (mirrors MixFusion
            # fair_compact_once). Per-module JSONL is dropped under suppress.
            import os as _os

            rank_tables = getattr(
                self, "_diffusion_lora_slot_ranks_by_slice", None
            )
            suppress = _os.environ.get(
                "VLLM_OMNI_SUPPRESS_LORA_EVENTS", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            emit_kwargs = dict(
                module_name=module_name,
                implementation="torch",
                requested_slot_indices=sorted(
                    {
                        slot
                        for entries in composition
                        for slot, _scale in entries
                    }
                ),
                applied_slot_indices=sorted(applied_slot_indices),
                active_slot_indices=sorted(applied_slot_indices),
                rows_per_request=list(row_layout),
                input_rows=int(x_flat.shape[0]),
                input_tensor=tensor_metadata(
                    x_flat,
                    role="lora_activation_input",
                ),
                output_tensor=tensor_metadata(
                    y_flat,
                    role="lora_activation_output",
                ),
                rank_signatures=(
                    [
                        [int(rank) for rank in rank_table]
                        for rank_table in rank_tables
                    ]
                    if rank_tables is not None
                    else []
                ),
            )
            if suppress:
                from vllm_omni.diffusion.experiment_telemetry import (
                    _telemetry_dir,
                )

                # Always use the base class flag so denoise_step can reset it
                # even when the live module is a subclass.
                flag_owner = BaseLinearLayerWithLoRA
                if (
                    not getattr(flag_owner, "_torch_fair_verify_emitted", False)
                    and _telemetry_dir() is not None
                ):
                    flag_owner._torch_fair_verify_emitted = True
                    emit_event(
                        "lora_events",
                        "lora_batch_slot_applied",
                        aggregate=True,
                        fair_compact_once=True,
                        **emit_kwargs,
                    )
            else:
                emit_event(
                    "lora_events",
                    "lora_batch_slot_applied",
                    **emit_kwargs,
                )
        return y_flat

    @torch.compiler.disable
    def _apply_sequential_multi_lora(
        self,
        module_name: str,
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        output_slices: tuple[int, ...],
        composition: tuple[tuple[tuple[int, float], ...], ...],
    ) -> torch.Tensor:
        row_layout = self._get_batch_row_layout(x_flat, composition)
        row_start = 0
        request_adapter_counts: list[int] = []
        for request_index, (row_count, entries) in enumerate(
            zip(row_layout, composition)
        ):
            row_end = row_start + row_count
            request_input = x_flat[row_start:row_end]
            request_output = y_flat[row_start:row_end]
            applied_count = 0
            for slot_index, scale in entries:
                output_offset = 0
                for slice_index, slice_size in enumerate(output_slices):
                    if (
                        slice_index < len(self._diffusion_lora_active_slices)
                        and not self._diffusion_lora_active_slices[slice_index]
                    ):
                        output_offset += slice_size
                        continue
                    rank_tables = getattr(
                        self, "_diffusion_lora_slot_ranks_by_slice", None
                    )
                    if (
                        rank_tables is not None
                        and int(rank_tables[slice_index][slot_index]) <= 0
                    ):
                        output_offset += slice_size
                        continue
                    lora_a = self.lora_a_stacked[slice_index][slot_index, 0]
                    lora_b = self.lora_b_stacked[slice_index][slot_index, 0]
                    if lora_a.shape[0] <= 0 or lora_b.shape[1] <= 0:
                        raise ValueError(
                            "referenced LoRA adapter must have positive rank"
                        )
                    hidden = request_input @ lora_a.transpose(0, 1)
                    delta = hidden @ lora_b.transpose(0, 1)
                    current = request_output[:, output_offset : output_offset + slice_size]
                    request_output[:, output_offset : output_offset + slice_size] = (
                        current + delta.to(dtype=current.dtype) * float(scale)
                    )
                    output_offset += slice_size
                applied_count += 1
                emit_event(
                    "lora_events",
                    "sequential_adapter_applied",
                    module_name=module_name,
                    request_index=request_index,
                    slot_index=int(slot_index),
                    scale=float(scale),
                )
            request_adapter_counts.append(applied_count)
            row_start = row_end
        emit_event(
            "lora_events",
            "sequential_summary",
            module_name=module_name,
            request_adapter_counts=request_adapter_counts,
            adapters_applied_count=sum(request_adapter_counts),
        )
        return y_flat

    @torch.compiler.disable
    def _apply_punica_segmented(
        self,
        module_name: str,
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        output_slices: tuple[int, ...],
        composition: tuple[tuple[tuple[int, float], ...], ...],
    ) -> torch.Tensor:
        row_layout = self._get_batch_row_layout(x_flat, composition)
        total_rows = int(x_flat.shape[0])
        x_work = x_flat if x_flat.is_contiguous() else x_flat.contiguous()
        request_indices = torch.repeat_interleave(
            torch.arange(len(row_layout), device=x_flat.device),
            torch.tensor(row_layout, device=x_flat.device, dtype=torch.long),
        )
        shrink_calls = 0
        expand_calls = 0
        segment_count = 0
        for slice_index, slice_size in enumerate(output_slices):
            if (
                slice_index < len(self._diffusion_lora_active_slices)
                and not self._diffusion_lora_active_slices[slice_index]
            ):
                continue
            slice_composition = self._filter_composition_for_slice(
                composition, slice_index
            )
            if not any(slice_composition):
                continue
            segment_rows = tuple(
                (request_index, int(slot), float(scale))
                for request_index, entries in enumerate(slice_composition)
                for slot, scale in entries
            )
            a_stack = self.lora_a_stacked[slice_index]
            b_stack = self.lora_b_stacked[slice_index]
            if not a_stack.is_contiguous():
                a_stack = a_stack.contiguous()
            if not b_stack.is_contiguous():
                b_stack = b_stack.contiguous()
            rank = int(a_stack.shape[2])
            if rank <= 0:
                raise ValueError("referenced LoRA adapter must have positive rank")
            output_offset = sum(output_slices[:slice_index])
            for request_index, slot_index, scale in segment_rows:
                rows = request_indices == request_index
                if not torch.any(rows):
                    continue
                token_lora_mapping = torch.full(
                    (total_rows,),
                    -1,
                    dtype=torch.int32,
                    device=x_flat.device,
                )
                row_index_tensor = rows.nonzero(as_tuple=False).flatten()
                token_lora_mapping.index_fill_(0, row_index_tensor, slot_index)
                metadata = LoRAKernelMeta.make(
                    max_loras=int(a_stack.shape[0]),
                    max_num_tokens=total_rows,
                    device=x_flat.device,
                )
                metadata.prepare_tensors(token_lora_mapping)
                shrink_buffer = torch.empty(
                    (1, total_rows, rank),
                    dtype=torch.float32,
                    device=x_flat.device,
                )
                lora_shrink(
                    x_work,
                    [a_stack],
                    shrink_buffer,
                    *metadata.meta_args(total_rows, False),
                    float(scale),
                )
                shrink_calls += 1
                lora_expand(
                    shrink_buffer,
                    [b_stack],
                    y_flat,
                    *metadata.meta_args(total_rows, False),
                    offset_start=output_offset,
                    add_inputs=True,
                )
                expand_calls += 1
                segment_count += 1
                emit_event(
                    "lora_events",
                    "punica_segmented_call",
                    module_name=module_name,
                    slice_index=slice_index,
                    slice_size=int(slice_size),
                    slot_index=slot_index,
                    scale=scale,
                    token_count=int(row_index_tensor.numel()),
                    punica_shrink_calls=1,
                    punica_expand_calls=1,
                )
        emit_event(
            "lora_events",
            "punica_segmented_summary",
            module_name=module_name,
            punica_shrink_calls=shrink_calls,
            punica_expand_calls=expand_calls,
            segment_count=segment_count,
        )
        return y_flat

    def _apply_punica_style_reference(
        self,
        module_name: str,
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        output_slices: tuple[int, ...],
        composition: tuple[tuple[tuple[int, float], ...], ...],
    ) -> torch.Tensor:
        return self._apply_multi_lora_operator(
            module_name, x_flat, y_flat, output_slices, composition
        )

    def _apply_fused_multi_lora_operator(
        self,
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        output_slices: tuple[int, ...],
        composition: tuple[tuple[tuple[int, float], ...], ...],
        implementation: str,
        base_layer: object | None = None,
        base_bias: torch.Tensor | None = None,
        row_spans: tuple[tuple[int, int, int], ...] | None = None,
    ) -> torch.Tensor:
        if implementation == "mixfusion":
            if not mixfusion_runtime_enabled():
                raise RuntimeError(
                    "MixFusion requires VLLM_OMNI_MIXFUSION_RUNTIME=1 or "
                    "VLLM_OMNI_RUNTIME_VARIANT=system_b_mixfusion_patch"
                )
            return self._apply_mixfusion_runtime(
                x_flat,
                y_flat,
                output_slices,
                composition,
                base_layer,
                base_bias,
            )
        row_layout = self._get_batch_row_layout(x_flat, composition)
        rank_tables = getattr(
            self, "_diffusion_lora_slot_ranks_by_slice", None
        )
        if rank_tables is None:
            raise RuntimeError("LoRA rank metadata was not initialized")
        rank_policy = multi_lora_rank_policy()
        tile_plan_cache = getattr(self, "_diffusion_lora_tile_plan_cache", None)
        if tile_plan_cache is None:
            tile_plan_cache = {}
            self._diffusion_lora_tile_plan_cache = tile_plan_cache
        applied_slots: set[int] = set()
        rank_bucket_blocks_by_slice: list[list[int]] = []
        for slice_index, slice_size in enumerate(output_slices):
            if (
                slice_index < len(self._diffusion_lora_active_slices)
                and not self._diffusion_lora_active_slices[slice_index]
            ):
                continue
            slice_composition = self._filter_composition_for_slice(
                composition, slice_index
            )
            if not any(slice_composition):
                continue
            relations = tuple(
                tuple(slot for slot, _ in entries)
                for entries in slice_composition
            )
            scales = tuple(
                tuple(scale for _, scale in entries)
                for entries in slice_composition
            )
            output_start = sum(output_slices[:slice_index])
            output_view = y_flat[:, output_start : output_start + slice_size]
            if implementation in {"dense_fused", "shared_input"}:
                from vllm_omni.diffusion.lora.ops import get_dense_fused_ops

                dense_mod = get_dense_fused_ops()
                metadata_cache = getattr(
                    self, "_diffusion_dense_fused_metadata_cache", None
                )
                if metadata_cache is None:
                    metadata_cache = {}
                    self._diffusion_dense_fused_metadata_cache = metadata_cache
                operator = (
                    dense_mod.dense_fused_lora_add
                    if implementation == "dense_fused"
                    else dense_mod.shared_input_lora_add
                )
                bucket_blocks = operator(
                    x_flat,
                    output_view,
                    self.lora_a_stacked[slice_index],
                    self.lora_b_stacked[slice_index],
                    row_layout,
                    relations,
                    scales,
                    rank_tables[slice_index],
                    metadata_cache=metadata_cache,
                    row_spans=row_spans,
                )
                if implementation == "dense_fused":
                    rank_bucket_blocks_by_slice.append(
                        list(bucket_blocks or ())
                    )
                for entries in slice_composition:
                    applied_slots.update(int(slot) for slot, _ in entries)
                continue
            if implementation in {
                "tile_fused",
                "mixfusion",
                "fused_pack_input",
                "fused_pack_lora",
            }:
                rank_table = rank_tables[slice_index]
                rank_signature = tuple(int(rank) for rank in rank_table)
                cache_key = (
                    slice_index,
                    row_layout,
                    relations,
                    scales,
                    int(slice_size),
                    x_flat.device,
                    rank_signature,
                )
                disable_plan_cache = os.environ.get(
                    "VLLM_OMNI_DISABLE_TILE_PLAN_CACHE", ""
                ).strip().lower() in {"1", "true", "yes", "on"}
                tile_plan = None if disable_plan_cache else tile_plan_cache.get(cache_key)
                if tile_plan is None:
                    if not disable_plan_cache and len(tile_plan_cache) >= 64:
                        tile_plan_cache.clear()
                    tile_plan = build_multi_lora_tile_plan(
                        row_layout,
                        relations,
                        scales,
                        rank_signature,
                        int(slice_size),
                        x_flat.device,
                        tile_size=int(
                            os.environ.get("VLLM_OMNI_MULTI_LORA_TILE_SIZE", "64")
                        ),
                    )
                    if not disable_plan_cache:
                        tile_plan_cache[cache_key] = tile_plan
                rank_groups = (None,)
                if rank_policy == "bucket":
                    rank_groups = tuple(
                        sorted(
                            {
                                int(rank_table[slot])
                                for entries in relations
                                for slot in entries
                                if int(rank_table[slot]) > 0
                            }
                        )
                    )
                for rank_group in rank_groups:
                    grouped_relations = relations
                    grouped_scales = scales
                    if rank_group is not None:
                        grouped_relations = tuple(
                            tuple(
                                slot
                                for slot in entries
                                if int(rank_table[slot]) == rank_group
                            )
                            for entries in relations
                        )
                        grouped_scales = tuple(
                            tuple(
                                scale
                                for slot, scale in zip(entries, row_scales)
                                if int(rank_table[slot]) == rank_group
                            )
                            for entries, row_scales in zip(relations, scales)
                        )
                    if not any(grouped_relations):
                        continue
                    grouped_tile_plan = tile_plan
                    if rank_group is not None:
                        grouped_tile_plan = build_multi_lora_tile_plan(
                            row_layout,
                            grouped_relations,
                            grouped_scales,
                            tuple(int(rank) for rank in rank_table),
                            int(slice_size),
                            x_flat.device,
                            tile_size=multi_lora_tile_size(),
                        )
                    execute_tiled_fused_lora(
                        x_flat,
                        output_view,
                        self.lora_a_stacked[slice_index],
                        self.lora_b_stacked[slice_index],
                        row_layout,
                        grouped_relations,
                        grouped_scales,
                        rank_table,
                        tile_plan=grouped_tile_plan,
                    )
            else:
                execute_fused_lora(
                    x_flat,
                    output_view,
                    self.lora_a_stacked[slice_index],
                    self.lora_b_stacked[slice_index],
                    row_layout,
                    relations,
                    scales,
                    rank_tables[slice_index],
                )
            for entries in slice_composition:
                applied_slots.update(int(slot) for slot, _ in entries)
        # The fused FLUX path uses composition mappings rather than the legacy
        # one-slot mapping. Emit the same request-level proof used by the
        # strict experiment gate so each measured timestep can be audited.
        if applied_slots and not getattr(
            self, "_diffusion_lora_batch_usage_emitted", False
        ):
            emit_event(
                "lora_events",
                "lora_batch_slot_applied",
                module_name=getattr(self, "_edge_dit_lora_module_name", None),
                requested_slot_indices=sorted(
                    {
                        int(slot)
                        for entries in composition
                        for slot, _ in entries
                    }
                ),
                applied_slot_indices=sorted(applied_slots),
                rows_per_request=list(row_layout),
                input_rows=int(x_flat.shape[0]),
                input_tensor=tensor_metadata(x_flat, role="lora_activation_input"),
                output_tensor=tensor_metadata(y_flat, role="lora_activation_output"),
                rank_policy=multi_lora_rank_policy(),
                rank_signatures=[
                    [int(rank) for rank in rank_table]
                    for rank_table in rank_tables
                ],
                rank_grouping_applied=rank_policy == "bucket",
                rank_packing_applied=False,
                rank_policy_fallback=(
                    "exact_rank_views" if rank_policy == "packed" else None
                ),
            )
            self._diffusion_lora_batch_usage_emitted = True
            if implementation in {"dense_fused", "shared_input"}:
                emit_event(
                    "lora_events",
                    f"{implementation}_lora_dispatch",
                    module_name=getattr(self, "_edge_dit_lora_module_name", None),
                    requested_slot_indices=sorted(
                        {
                            int(slot)
                            for entries in composition
                            for slot, _ in entries
                        }
                    ),
                    applied_slot_indices=sorted(applied_slots),
                    row_layout=list(row_layout),
                    rank_signatures=[
                        [int(rank) for rank in rank_table]
                        for rank_table in rank_tables
                    ],
                    base_output_precomputed=True,
                    base_lora_same_runtime=False,
                    rank_bucket_blocks_by_slice=(
                        rank_bucket_blocks_by_slice
                        if implementation == "dense_fused"
                        else None
                    ),
                )
        return y_flat

    @torch.compiler.disable
    def _apply_mixfusion_runtime(
        self,
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        output_slices: tuple[int, ...],
        composition: tuple[tuple[tuple[int, float], ...], ...],
        base_layer: object | None,
        base_bias: torch.Tensor | None,
        token_tiles: tuple[tuple[int, int], ...] | None = None,
        token_tile_segments: tuple[tuple[int, int, int], ...] = (),
    ) -> torch.Tensor:
        """Execute Base and LoRA through one descriptor and output owner."""
        implementation = "mixfusion"
        # base-only: skip descriptor / plan / runtime dispatch overhead so
        # Fair M* base_only is not inflated by empty MixFusion bookkeeping.
        if not composition:
            if base_layer is None:
                raise RuntimeError(
                    "MixFusion base-only path requires a Base layer"
                )
            base_out = base_layer.quant_method.apply(base_layer, x_flat, base_bias)
            if base_out.data_ptr() != y_flat.data_ptr():
                y_flat.copy_(base_out)
            return y_flat
        row_layout = self._get_batch_row_layout(x_flat, composition)
        native_patch_schedule = active_native_patch_schedule()
        mixfusion_lora_plan = build_mixfusion_multilora_plan(
            row_layout, native_patch_schedule
        )
        rank_tables = getattr(self, "_diffusion_lora_slot_ranks_by_slice", None)
        if composition and rank_tables is None:
            raise RuntimeError("LoRA rank metadata was not initialized")
        request_count = len(row_layout)
        normalized_composition = composition or tuple(
            () for _ in range(request_count)
        )
        relations = tuple(
            tuple(int(slot) for slot, _ in entries)
            for entries in normalized_composition
        )
        scales = tuple(
            tuple(float(scale) for _, scale in entries)
            for entries in normalized_composition
        )
        ranks = tuple(
            tuple(int(rank_tables[0][slot]) for slot in request_relations)
            for request_relations in relations
        ) if rank_tables else tuple(() for _ in row_layout)
        slice_ranks = tuple(
            tuple(
                tuple(int(rank_table[slot]) for slot in request_relations)
                for request_relations in relations
            )
            for rank_table in rank_tables
        ) if rank_tables else ()
        source_patch_contract = build_source_patch_contract(
            patch_plan=None,
            input_rows=int(x_flat.shape[0]),
            row_layout=tuple(int(rows) for rows in row_layout),
            adapter_ids=relations,
            rank_signatures=slice_ranks,
            module_name=getattr(self, "_edge_dit_lora_module_name", None),
            native_patch_schedule=native_patch_schedule,
        )
        descriptor = MixFusionDescriptor(
            request_id=getattr(self, "_edge_dit_lora_request_id", None),
            step_id=None,
            block_name=getattr(self, "_edge_dit_lora_block_name", None),
            module_name=getattr(self, "_edge_dit_lora_module_name", None),
            input_shape=tuple(int(value) for value in x_flat.shape),
            output_shape=tuple(int(value) for value in y_flat.shape),
            row_layout=tuple(int(value) for value in row_layout),
            base_operation="quant_method.apply",
            adapter_ids=relations,
            ranks=ranks,
            scales=scales,
            token_tiles=token_tiles or ((0, int(x_flat.shape[0])),),
            dependency_events=("input_ready", "base_ready", "lora_ready"),
            stream=str(torch.cuda.current_stream(x_flat.device).cuda_stream)
            if x_flat.is_cuda
            else None,
            output_add_owner="mixfusion_runtime",
            slice_ranks=slice_ranks,
            variant=runtime_variant(),
            token_tile_segments=token_tile_segments,
            source_patch_contract=source_patch_contract,
        )
        runtime = MixFusionRuntime()

        def base_forward(
            inputs: torch.Tensor, bias: torch.Tensor | None
        ) -> torch.Tensor:
            if base_layer is None:
                raise RuntimeError("Base callback was requested without a Base layer")
            base_interval = start_interval(
                "base_gemm",
                module_name=getattr(self, "_edge_dit_lora_module_name", None),
                input_shape=list(inputs.shape),
                input_dtype=str(inputs.dtype),
            )
            output = base_layer.quant_method.apply(base_layer, inputs, bias)
            finish_interval(
                base_interval,
                output_shape=list(output.shape),
                output_dtype=str(output.dtype),
            )
            return output

        def lora_forward(
            inputs: torch.Tensor,
            output: torch.Tensor,
            slice_index: int,
            slice_relations: tuple[tuple[int, ...], ...],
            slice_scales: tuple[tuple[float, ...], ...],
        ) -> None:
            if (
                slice_index < len(self._diffusion_lora_active_slices)
                and not self._diffusion_lora_active_slices[slice_index]
            ):
                return
            if rank_tables is None:
                raise RuntimeError("LoRA rank metadata was not initialized")
            if mixfusion_lora_plan is None:
                raise RuntimeError(
                    "MixFusion Multi-LoRA dispatch has no source ownership plan"
                )
            rank_table = rank_tables[slice_index]
            if implementation == "mixfusion":
                metadata_cache = getattr(
                    self, "_diffusion_mixfusion_metadata_cache", None
                )
                if metadata_cache is None:
                    metadata_cache = {}
                    self._diffusion_mixfusion_metadata_cache = metadata_cache
                dispatch = apply_mixfusion_multilora(
                    inputs,
                    output,
                    self.lora_a_stacked[slice_index],
                    self.lora_b_stacked[slice_index],
                    mixfusion_lora_plan,
                    slice_relations,
                    slice_scales,
                    rank_table,
                    metadata_cache=metadata_cache,
                )
                # Per-dispatch JSONL is opt-in: default off so Fair M* measured
                # steps are not dominated by ~1k host event writes.
                if os.environ.get(
                    "VLLM_OMNI_LORA_DISPATCH_TELEMETRY", ""
                ).strip().lower() in {"1", "true", "yes", "on"}:
                    emit_event(
                        "lora_events",
                        "mixfusion_lora_kernel_dispatch",
                        module_name=getattr(
                            self, "_edge_dit_lora_module_name", None
                        ),
                        slice_index=slice_index,
                        kernel_backend=dispatch.kernel_backend,
                        tetris_two_stage_tc=(
                            dispatch.kernel_backend == "tetris_two_stage_tc"
                        ),
                        mixfusion_lora_semantics=(
                            "patch_level_multi_lora"
                            if dispatch.ownership_mode == "patch_row_spans"
                            else "request_row_multi_lora_under_patch_schedule"
                        ),
                        mixfusion_lora_ownership_mode=dispatch.ownership_mode,
                        mixfusion_lora_patch_count=dispatch.patch_count,
                        mixfusion_lora_patch_row_span_count=(
                            dispatch.patch_row_span_count
                        ),
                        mixfusion_lora_rank_bucket_blocks=list(
                            dispatch.rank_bucket_blocks
                        ),
                        mixfusion_lora_used_execute_tiled_fused_lora=(
                            dispatch.used_execute_tiled_fused_lora
                        ),
                        row_layout=list(row_layout),
                        source_patch_scheduler_called=bool(
                            getattr(
                                active_native_patch_schedule(),
                                "source_split_called",
                                False,
                            )
                        ),
                        source_split_called=bool(
                            getattr(
                                active_native_patch_schedule(),
                                "source_split_called",
                                False,
                            )
                        ),
                        source_split_input_shape=(
                            list(source_patch_contract.native_split_input_shape)
                            if source_patch_contract.native_split_input_shape
                            is not None
                            else None
                        ),
                        source_split_output_shape=(
                            list(source_patch_contract.native_split_output_shape)
                            if source_patch_contract.native_split_output_shape
                            is not None
                            else None
                        ),
                        source_split_patch_count=(
                            source_patch_contract.native_split_patch_count
                        ),
                    )
                # One summary flag per layer object so backend is still
                # observable without per-call JSON.
                self._diffusion_last_mixfusion_backend = dispatch.kernel_backend
                return
            rank_groups: tuple[int | None, ...] = (None,)
            if multi_lora_rank_policy() == "bucket":
                rank_groups = tuple(
                    sorted(
                        {
                            int(rank_table[slot])
                            for entries in slice_relations
                            for slot in entries
                            if int(rank_table[slot]) > 0
                        }
                    )
                )
            for rank_group in rank_groups:
                grouped_relations = slice_relations
                grouped_scales = slice_scales
                if rank_group is not None:
                    grouped_relations = tuple(
                        tuple(
                            slot
                            for slot in entries
                            if int(rank_table[slot]) == rank_group
                        )
                        for entries in slice_relations
                    )
                    grouped_scales = tuple(
                        tuple(
                            scale
                            for slot, scale in zip(entries, row_scales)
                            if int(rank_table[slot]) == rank_group
                        )
                        for entries, row_scales in zip(
                            slice_relations, slice_scales
                        )
                    )
                if not any(grouped_relations):
                    continue
                tile_plan = build_multi_lora_tile_plan(
                    row_layout,
                    grouped_relations,
                    grouped_scales,
                    tuple(int(rank) for rank in rank_table),
                    int(output_slices[slice_index]),
                    inputs.device,
                    tile_size=multi_lora_tile_size(),
                )
                execute_tiled_fused_lora(
                    inputs,
                    output,
                    self.lora_a_stacked[slice_index],
                    self.lora_b_stacked[slice_index],
                    row_layout,
                    grouped_relations,
                    grouped_scales,
                    rank_table,
                    tile_plan=tile_plan,
                )

        runtime.execute(
            descriptor,
            x_flat,
            y_flat,
            base_forward=base_forward,
            bias=base_bias,
            lora_forward=lora_forward,
            lora_a_stacked=(
                tuple(self.lora_a_stacked) if composition else ()
            ),
            lora_b_stacked=(
                tuple(self.lora_b_stacked) if composition else ()
            ),
        )
        return y_flat

    def _apply_split_lora(
        self,
        module_name: str,
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        output_slices: tuple[int, ...],
        batch_slot_indices: tuple[int | None, ...],
        row_layout: tuple[int, ...] | None = None,
    ) -> torch.Tensor | None:
        try:
            from vllm_omni.diffusion.lora.edge_dit_lora_runtime import (
                maybe_edge_dit_lora_a_projection,
            )
        except Exception:
            return None

        projection_features = sum(
            int(a.shape[2]) for a in self.lora_a_stacked
        )
        split_records = maybe_edge_dit_lora_a_projection(
            module_name,
            x_flat,
            batch_slot_indices,
            response_features=projection_features,
            module_out_features=int(y_flat.shape[-1]),
            route_on_edge=True,
        )
        if not split_records:
            return None

        for slot_index in batch_slot_indices:
            if slot_index is None:
                continue
            if not 0 <= int(slot_index) < self.lora_a_stacked[0].shape[0]:
                raise ValueError(f"LoRA slot {slot_index} is outside resident storage")
            for a_stack, b_stack in zip(
                self.lora_a_stacked, self.lora_b_stacked
            ):
                if a_stack[slot_index, 0].shape[0] <= 0 or b_stack[slot_index, 0].shape[1] <= 0:
                    raise ValueError(
                        "referenced LoRA adapter must have positive rank"
                    )

        row_counts = tuple(
            row_layout
            if row_layout is not None
            else (x_flat.shape[0] // len(batch_slot_indices),)
            * len(batch_slot_indices)
        )
        if sum(row_counts) != x_flat.shape[0]:
            raise ValueError(
                "legacy LoRA row layout does not cover flattened activations: "
                f"layout={row_counts}, rows={x_flat.shape[0]}"
            )
        row_slot_indices = torch.repeat_interleave(
            torch.tensor(
                [slot if slot is not None else -1 for slot in batch_slot_indices],
                device=x_flat.device,
                dtype=torch.long,
            ),
            torch.tensor(
                row_counts,
                device=x_flat.device,
                dtype=torch.long,
            ),
        )
        hidden_offset = 0
        output_offset = 0
        applied_slots: set[int] = set()
        lora_a_flops = 0
        lora_b_flops = 0
        residual_add_flops = 0
        total_interval = start_interval(
            "lora_total",
            module_name=module_name,
            split_mode="split01_or_split02",
            input_shape=list(x_flat.shape),
            output_shape=list(y_flat.shape),
        )
        for slice_idx, slice_size in enumerate(output_slices):
            a = self.lora_a_stacked[slice_idx][0, 0]
            rank = int(a.shape[0])
            b_stack = self.lora_b_stacked[slice_idx]
            for record in split_records:
                group_mask = record["row_mask"]
                group_rows = group_mask.nonzero(as_tuple=False).flatten()
                group_hidden = record["hidden"]
                for slot_index in record["slot_indices"]:
                    rows = group_mask & (row_slot_indices == slot_index)
                    row_positions = (rows[group_rows]).nonzero(as_tuple=False).flatten()
                    if row_positions.numel() == 0:
                        continue
                    selected_hidden = group_hidden[row_positions, hidden_offset : hidden_offset + rank]
                    common = {
                        "module_name": module_name,
                        "slice_index": slice_idx,
                        "slot_index": slot_index,
                        "split_mode": "split01_or_split02",
                    }
                    b = b_stack[slot_index, 0]
                    b_interval = start_interval(
                        "lora_b_gemm", **common, input_shape=list(selected_hidden.shape)
                    )
                    delta = selected_hidden @ b.t()
                    lora_b_flops += 2 * int(selected_hidden.shape[0]) * int(b.shape[1]) * int(b.shape[0])
                    finish_interval(b_interval, output_shape=list(delta.shape))
                    gather_interval = start_interval("lora_output_gather", **common)
                    current_output = y_flat[rows, output_offset : output_offset + slice_size]
                    finish_interval(gather_interval, output_slice_shape=list(current_output.shape))
                    residual_interval = start_interval("lora_residual_add", **common)
                    updated_output = current_output + delta
                    residual_add_flops += int(current_output.numel())
                    finish_interval(residual_interval, residual_shape=list(updated_output.shape))
                    writeback_interval = start_interval("lora_writeback", **common)
                    y_flat[rows, output_offset : output_offset + slice_size] = updated_output
                    finish_interval(writeback_interval, output_offset=output_offset)
                    applied_slots.add(int(slot_index))
            hidden_offset += rank
            output_offset += slice_size

        finish_interval(total_interval, applied_slot_indices=sorted(applied_slots))
        if applied_slots:
            record_lora_flops(
                lora_a_flops=lora_a_flops,
                lora_b_flops=lora_b_flops,
                residual_add_flops=residual_add_flops,
            )
            emit_event(
                "lora_events",
                "lora_split_path_applied",
                module_name=module_name,
                split_mode="split01_or_split02",
                applied_slot_indices=sorted(applied_slots),
                edge_records=len(split_records),
            )
        return y_flat

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        override: Use simple matmul instead of punica_wrapper.add_lora_linear().

        This matches the exact computation in PunicaWrapperGPU.add_lora_linear()
        for the single-LoRA case. For packed projections (e.g. fused QKV), we
        apply LoRA per-slice using `output_slices`.
        """
        module_name = getattr(self, "_edge_dit_lora_module_name", None)
        _reject_removed_token_tile_env()
        served_lora_only = os.environ.get(
            "VLLM_OMNI_SERVED_LORA_ONLY", ""
        ).lower() in {"1", "true", "yes", "on"}
        composition = getattr(self, "_diffusion_lora_batch_composition", None)
        batch_slot_indices = getattr(
            self, "_diffusion_lora_batch_slot_indices", None
        )
        tetris_enabled = os.environ.get("VLLM_OMNI_TETRIS_LORA", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if (
            tetris_enabled
            and composition is None
            and batch_slot_indices
            and any(slot is not None for slot in batch_slot_indices)
        ):
            composition = tuple(
                () if slot is None else ((int(slot), 1.0),)
                for slot in batch_slot_indices
            )
        operator_enabled = _multi_lora_operator_enabled()
        implementation = _selected_multi_lora_implementation()
        output_slices = getattr(self, "output_slices", None)
        if output_slices is None and hasattr(self, "lora_b_stacked"):
            output_slices = tuple(
                lora_b.shape[2] for lora_b in self.lora_b_stacked
            )
        if output_slices is None:
            output_slices = ()
        lora_active = self._lora_activation_enabled(module_name)
        if (
            tetris_enabled
            and module_name is not None
            and os.environ.get("TETRIS_LORA_VERBOSE", "").lower()
            in {"1", "true", "yes", "on"}
        ):
            print(
                "[TetrisLoRA] gate module=%s ndim=%d composition=%s slots=%s "
                "active=%s a=%s b=%s",
                module_name,
                x.ndim,
                composition,
                batch_slot_indices,
                lora_active,
                hasattr(self, "lora_a_stacked"),
                hasattr(self, "lora_b_stacked"),
                flush=True,
            )
        if (
            not lora_active
            or not hasattr(self, "lora_a_stacked")
            or not hasattr(self, "lora_b_stacked")
            or not self.lora_a_stacked
            or not self.lora_b_stacked
        ):
            composition = None
        if composition is not None and (not composition or not any(composition)):
            composition = None
        if composition is None and (
            not batch_slot_indices or all(slot is None for slot in batch_slot_indices)
        ):
            composition = None
        has_lora_request = bool(composition and any(composition))
        if composition is not None and any(len(entries) > 1 for entries in composition):
            if not operator_enabled:
                raise RuntimeError(
                    "Multi-LoRA composition with K>1 requires "
                    "VLLM_OMNI_ENABLE_MULTI_LORA_OPERATOR=1"
                )
        full_mixfusion_runtime = (
            implementation == "mixfusion"
            and operator_enabled
            and uses_mixfusion_runtime()
            and has_lora_request
        )
        # B3: refuse operator+edge-split coexistence (would skip edge-A then
        # run full local LoRA, double-counting or dropping the split path).
        edge_split_env = os.environ.get(
            "VLLM_OMNI_EDGE_DIT_SPLIT_MODE", ""
        ).strip().lower()
        if (
            operator_enabled
            and has_lora_request
            and edge_split_env in {"split01", "split02"}
            and not full_mixfusion_runtime
        ):
            raise RuntimeError(
                "VLLM_OMNI_ENABLE_MULTI_LORA_OPERATOR with "
                f"VLLM_OMNI_EDGE_DIT_SPLIT_MODE={edge_split_env!r} is refused: "
                "edge split-A and the full multi-LoRA operator cannot both own "
                "the same residual; disable one path"
            )
        tetris_operation = None
        if (
            x.ndim in {2, 3}
            and not served_lora_only
            and not full_mixfusion_runtime
            and has_lora_request
            and tetris_enabled
        ):
            from vllm_omni.diffusion.lora.tetris_lora_runtime import (
                begin_tetris_lora,
            )

            tetris_operation = begin_tetris_lora(
                str(module_name),
                x.reshape(-1, x.shape[-1]),
                composition,
                getattr(self, "_tetris_lora_slot_paths", None),
                rank_tables=tuple(
                    tuple(int(rank) for rank in table)
                    for table in getattr(
                        self, "_diffusion_lora_slot_ranks_by_slice", ()
                    )
                ),
                output_slices=tuple(int(size) for size in output_slices),
                row_layout=tuple(
                    int(rows)
                    for rows in self._get_batch_row_layout(
                        x.reshape(-1, x.shape[-1]), composition
                    )
                ),
            )
        # D2: begin holds client lock until finish; Base exceptions must cancel.
        try:
            shared_tile_result = None
            if not full_mixfusion_runtime:
                base_interval = start_interval(
                    "base_gemm",
                    module_name=module_name,
                    input_shape=list(x.shape),
                    input_dtype=str(x.dtype),
                )
                if served_lora_only:
                    output = torch.zeros(
                        (*x.shape[:-1], int(self.output_size)),
                        device=x.device,
                        dtype=x.dtype,
                    )
                else:
                    output = self.base_layer.quant_method.apply(
                        self.base_layer, x, bias
                    )
                shared_lora_applied = False
            elif full_mixfusion_runtime:
                output = torch.empty(
                    (*x.shape[:-1], int(self.output_size)),
                    device=x.device,
                    dtype=x.dtype,
                )
                shared_lora_applied = False
            else:
                output, shared_lora_applied = shared_tile_result
            if shared_tile_result is None and not full_mixfusion_runtime:
                finish_interval(
                    base_interval,
                    execution_location=(
                        "skipped_served_lora_only" if served_lora_only else None
                    ),
                    output_shape=list(output.shape),
                    output_dtype=str(output.dtype),
                )

            original_shape = output.shape
            x_flat = x.reshape(-1, x.shape[-1])
            y_flat = output.reshape(-1, output.shape[-1])

            if not lora_active and not full_mixfusion_runtime:
                return output

            if shared_lora_applied:
                return output

            if tetris_operation is not None:
                from vllm_omni.diffusion.lora.tetris_lora_runtime import (
                    finish_tetris_lora,
                )

                result = finish_tetris_lora(tetris_operation, y_flat).view(
                    original_shape
                )
                tetris_operation = None
                return result
        except BaseException:
            if tetris_operation is not None:
                from vllm_omni.diffusion.lora.tetris_lora_runtime import (
                    cancel_tetris_lora,
                )

                cancel_tetris_lora(tetris_operation)
                tetris_operation = None
            raise

        lora_path_interval = None

        # Edge-owned path: prefer remote residual when runtime metadata targets
        # this module. Checked before local LoRA so edge-only modules work
        # without cloud A/B weights.
        if module_name:
            try:
                from vllm_omni.diffusion.lora.edge_dit_lora_runtime import (
                    maybe_edge_dit_lora_residual,
                )
            except Exception:  # pragma: no cover
                maybe_edge_dit_lora_residual = None  # type: ignore[assignment]
            if maybe_edge_dit_lora_residual is not None:
                lora_path_interval = start_interval(
                    "lora_path_e2e",
                    module_name=module_name,
                    execution_location="edge",
                    input_shape=list(x_flat.shape),
                    output_shape=list(y_flat.shape),
                )
                edge_delta = maybe_edge_dit_lora_residual(
                    str(module_name), x_flat, int(y_flat.shape[-1])
                )
                if edge_delta is not None:
                    y_flat = y_flat + edge_delta.to(
                        device=y_flat.device, dtype=y_flat.dtype
                    )
                    finish_interval(
                        lora_path_interval,
                        execution_location="edge",
                        output_shape=list(y_flat.shape),
                    )
                    return y_flat.view(original_shape)
                finish_interval(lora_path_interval, execution_location="none")

        if not hasattr(self, "lora_a_stacked") or not hasattr(self, "lora_b_stacked"):
            if full_mixfusion_runtime:
                return self._apply_mixfusion_runtime(
                    x_flat,
                    y_flat,
                    (),
                    (),
                    self.base_layer,
                    bias,
                ).view(original_shape)
            return output
        if not self.lora_a_stacked or not self.lora_b_stacked:
            if full_mixfusion_runtime:
                return self._apply_mixfusion_runtime(
                    x_flat,
                    y_flat,
                    (),
                    (),
                    self.base_layer,
                    bias,
                ).view(original_shape)
            return output
        # Fast path: if no LoRA is active for this layer, skip matmuls.
        active_slices = getattr(self, "_diffusion_lora_active_slices", None)
        if active_slices is not None and not any(active_slices):
            if full_mixfusion_runtime:
                return self._apply_mixfusion_runtime(
                    x_flat,
                    y_flat,
                    (),
                    (),
                    self.base_layer,
                    bias,
                ).view(original_shape)
            return output

        # In fully-sharded LoRA mode, vLLM uses an all-gather between shrink and
        # expand for ColumnParallelLinear variants. This diffusion path doesn't
        # implement that communication yet.
        if getattr(self, "lora_config", None) is not None:
            if self.lora_config.fully_sharded_loras and self.tp_size > 1:
                raise NotImplementedError(
                    "Diffusion LoRA apply() does not support fully_sharded_loras with tensor parallelism yet."
                )

        output_slices = getattr(self, "output_slices", None)
        if output_slices is None:
            output_slices = tuple(
                lora_b.shape[2] for lora_b in self.lora_b_stacked
            )

        if len(output_slices) != len(self.lora_a_stacked) or len(output_slices) != len(self.lora_b_stacked):
            raise RuntimeError(
                "LoRA slice metadata mismatch: "
                f"output_slices={len(output_slices)}, "
                f"lora_a_stacked={len(self.lora_a_stacked)}, "
                f"lora_b_stacked={len(self.lora_b_stacked)}"
            )

        composition = getattr(self, "_diffusion_lora_batch_composition", None)
        batch_slot_indices = getattr(
            self, "_diffusion_lora_batch_slot_indices", None
        )
        if batch_slot_indices is None:
            batch_slot_indices = (None,)
        if (
            composition is None
            and operator_enabled
            and batch_slot_indices
            and any(slot is not None for slot in batch_slot_indices)
        ):
            composition = tuple(
                () if slot is None else ((int(slot), 1.0),)
                for slot in batch_slot_indices
            )
        if composition is None and (
            not batch_slot_indices or all(slot is None for slot in batch_slot_indices)
        ):
            if full_mixfusion_runtime:
                return self._apply_mixfusion_runtime(
                    x_flat,
                    y_flat,
                    (),
                    (),
                    self.base_layer,
                    bias,
                ).view(original_shape)
            return output
        if composition is not None and (not composition or not any(composition)):
            return output
        operator_enabled = _multi_lora_operator_enabled()
        if composition is not None and any(len(entries) > 1 for entries in composition):
            if not operator_enabled:
                raise RuntimeError(
                    "Multi-LoRA composition with K>1 requires "
                    "VLLM_OMNI_ENABLE_MULTI_LORA_OPERATOR=1"
                )
        if composition is not None and not operator_enabled:
            batch_slot_indices = tuple(
                entries[0][0] if entries else None for entries in composition
            )
        if served_lora_only and composition is not None and not operator_enabled:
            lora_path_interval = start_interval(
                "lora_path_e2e",
                module_name=module_name,
                execution_location="served_lora_only",
                input_shape=list(x_flat.shape),
                output_shape=list(y_flat.shape),
            )
            y_flat = self._apply_sequential_multi_lora(
                str(module_name),
                x_flat,
                y_flat,
                tuple(int(size) for size in output_slices),
                tuple(composition),
            )
            finish_interval(
                lora_path_interval,
                execution_location="served_lora_only",
                output_shape=list(y_flat.shape),
            )
            return y_flat.view(original_shape)
        if operator_enabled and composition is not None:
            lora_path_interval = start_interval(
                "lora_path_e2e",
                module_name=module_name,
                execution_location="multi_lora_operator",
                input_shape=list(x_flat.shape),
                output_shape=list(y_flat.shape),
            )
            implementation = multi_lora_implementation()
            if implementation == "sequential":
                raise ValueError(
                    "VLLM_OMNI_MULTI_LORA_IMPL=sequential is reserved for "
                    "request-level serial transport; use torch for the "
                    "request-internal operator"
                )
            elif implementation == "punica_segmented":
                y_flat = self._apply_punica_segmented(
                    str(module_name),
                    x_flat,
                    y_flat,
                    tuple(int(size) for size in output_slices),
                    tuple(composition),
                )
            elif implementation == "punica_ref":
                y_flat = self._apply_punica_style_reference(
                    str(module_name),
                    x_flat,
                    y_flat,
                    tuple(int(size) for size in output_slices),
                    tuple(composition),
                )
            elif implementation == "torch":
                y_flat = self._apply_multi_lora_operator(
                    str(module_name),
                    x_flat,
                    y_flat,
                    tuple(int(size) for size in output_slices),
                    tuple(composition),
                )
            elif implementation in {
                "fused",
                "tile_fused",
                "mixfusion",
                "fused_pack_input",
                "fused_pack_lora",
                "dense_fused",
                "shared_input",
            }:
                y_flat = self._apply_fused_multi_lora_operator(
                    x_flat,
                    y_flat,
                    tuple(int(size) for size in output_slices),
                    tuple(composition),
                    implementation,
                    base_layer=(
                        self.base_layer if implementation == "mixfusion" else None
                    ),
                    base_bias=bias if implementation == "mixfusion" else None,
                )
            else:
                raise ValueError(
                    "unsupported VLLM_OMNI_MULTI_LORA_IMPL: "
                    f"{implementation}"
                )
            finish_interval(
                lora_path_interval,
                execution_location="multi_lora_operator",
                output_shape=list(y_flat.shape),
            )
            return y_flat.view(original_shape)

        lora_path_interval = start_interval(
            "lora_path_e2e",
            module_name=module_name,
            execution_location="cloud",
            input_shape=list(x_flat.shape),
            output_shape=list(y_flat.shape),
        )

        legacy_row_layout = (
            self._get_batch_row_layout(x_flat, composition)
            if composition is not None
            else None
        )
        split_output = self._apply_split_lora(
            str(module_name),
            x_flat,
            y_flat,
            tuple(int(size) for size in output_slices),
            tuple(batch_slot_indices),
            legacy_row_layout,
        )
        if split_output is not None:
            finish_interval(
                lora_path_interval,
                execution_location="split",
                output_shape=list(split_output.shape),
            )
            return split_output.view(original_shape)
        diagnostic_mode = _lora_diagnostic_mode()
        if diagnostic_mode not in {
            "normal",
            "route_only",
            "route_gather",
            "compute_no_writeback",
        }:
            raise ValueError(
                "unsupported VLLM_OMNI_LORA_DIAGNOSTIC_MODE: "
                f"{diagnostic_mode}"
            )
        route_unique_mode = os.environ.get(
            "VLLM_OMNI_LORA_ROUTE_UNIQUE_MODE", "device_unique"
        ).strip().lower()
        if route_unique_mode not in {"device_unique", "host_cached"}:
            raise ValueError(
                "unsupported VLLM_OMNI_LORA_ROUTE_UNIQUE_MODE: "
                f"{route_unique_mode}"
            )
        total_interval = start_interval(
            "lora_total",
            module_name=module_name,
            input_shape=list(x_flat.shape),
            output_shape=list(y_flat.shape),
            diagnostic_mode=diagnostic_mode,
            route_unique_mode=route_unique_mode,
        )
        prepare_interval = start_interval(
            "lora_routing_prepare",
            module_name=module_name,
        )
        if x_flat.shape[0] % len(batch_slot_indices):
            raise RuntimeError(
                "LoRA batch slot mapping does not divide flattened activations: "
                f"rows={x_flat.shape[0]}, slots={len(batch_slot_indices)}"
            )
        rows_per_request = x_flat.shape[0] // len(batch_slot_indices)
        row_slot_indices = torch.repeat_interleave(
            torch.tensor(
                [slot if slot is not None else -1 for slot in batch_slot_indices],
                device=x_flat.device,
                dtype=torch.long,
            ),
            rows_per_request,
        )
        finish_interval(
            prepare_interval,
            batch_slot_indices=[
                slot if slot is not None else -1
                for slot in batch_slot_indices
            ],
            rows_per_request=rows_per_request,
        )

        unique_interval = start_interval(
            "lora_slot_unique",
            module_name=module_name,
            route_unique_mode=route_unique_mode,
        )
        if route_unique_mode == "host_cached":
            unique_slot_indices = sorted(
                {int(slot) for slot in batch_slot_indices if slot is not None}
            )
        else:
            unique_slot_indices = torch.unique(row_slot_indices).tolist()
        finish_interval(
            unique_interval,
            unique_slot_indices=unique_slot_indices,
        )

        if diagnostic_mode == "route_only":
            finish_interval(
                total_interval,
                applied_slot_indices=[],
                diagnostic_mode=diagnostic_mode,
            )
            finish_interval(
                lora_path_interval,
                execution_location="cloud",
                diagnostic_mode=diagnostic_mode,
            )
            return output

        applied_slot_indices: set[int] = set()
        offset = 0
        lora_a_flops = 0
        lora_b_flops = 0
        residual_add_flops = 0
        for slice_idx, slice_size in enumerate(output_slices):
            if active_slices is not None and slice_idx < len(active_slices) and not active_slices[slice_idx]:
                offset += slice_size
                continue

            for slot_index in unique_slot_indices:
                if slot_index < 0:
                    continue
                A = self.lora_a_stacked[slice_idx][slot_index, 0, :, :]
                B = self.lora_b_stacked[slice_idx][slot_index, 0, :, :]

                if A.numel() == 0 or B.numel() == 0:
                    continue

                common_metadata = {
                    "module_name": module_name,
                    "slice_index": slice_idx,
                    "slot_index": slot_index,
                    "lora_a_shape": list(A.shape),
                    "lora_b_shape": list(B.shape),
                }
                mask_interval = start_interval(
                    "lora_row_mask",
                    **common_metadata,
                )
                rows = row_slot_indices == slot_index
                finish_interval(mask_interval)

                input_gather_interval = start_interval(
                    "lora_input_gather",
                    **common_metadata,
                )
                selected_x = x_flat[rows]
                finish_interval(
                    input_gather_interval,
                    selected_input_shape=list(selected_x.shape),
                )

                if diagnostic_mode == "route_gather":
                    output_gather_interval = start_interval(
                        "lora_output_gather",
                        **common_metadata,
                    )
                    current_output = y_flat[
                        rows, offset : offset + slice_size
                    ]
                    finish_interval(
                        output_gather_interval,
                        output_slice_shape=list(current_output.shape),
                    )
                    continue

                lora_a_interval = start_interval(
                    "lora_a_gemm",
                    **common_metadata,
                    input_shape=list(selected_x.shape),
                )
                lora_hidden = selected_x @ A.t()
                lora_a_flops += 2 * int(selected_x.shape[0]) * int(A.shape[1]) * int(A.shape[0])
                finish_interval(
                    lora_a_interval,
                    output_shape=list(lora_hidden.shape),
                )

                lora_b_interval = start_interval(
                    "lora_b_gemm",
                    **common_metadata,
                    input_shape=list(lora_hidden.shape),
                )
                delta = lora_hidden @ B.t()
                lora_b_flops += 2 * int(lora_hidden.shape[0]) * int(B.shape[1]) * int(B.shape[0])
                finish_interval(
                    lora_b_interval,
                    output_shape=list(delta.shape),
                )

                output_gather_interval = start_interval(
                    "lora_output_gather",
                    **common_metadata,
                )
                current_output = y_flat[
                    rows, offset : offset + slice_size
                ]
                finish_interval(
                    output_gather_interval,
                    output_slice_shape=list(current_output.shape),
                )

                residual_interval = start_interval(
                    "lora_residual_add",
                    **common_metadata,
                )
                updated_output = current_output + delta
                residual_add_flops += int(current_output.numel())
                finish_interval(
                    residual_interval,
                    residual_shape=list(updated_output.shape),
                )

                if diagnostic_mode == "normal":
                    writeback_interval = start_interval(
                        "lora_writeback",
                        **common_metadata,
                    )
                    y_flat[rows, offset : offset + slice_size] = updated_output
                    finish_interval(
                        writeback_interval,
                        output_offset=offset,
                        output_slice_size=slice_size,
                    )
                applied_slot_indices.add(slot_index)
            offset += slice_size

        finish_interval(
            total_interval,
            applied_slot_indices=sorted(applied_slot_indices),
        )
        finish_interval(
            lora_path_interval,
            execution_location="cloud",
            applied_slot_indices=sorted(applied_slot_indices),
        )

        if applied_slot_indices:
            record_lora_flops(
                lora_a_flops=lora_a_flops,
                lora_b_flops=lora_b_flops,
                residual_add_flops=residual_add_flops,
            )

        if applied_slot_indices and not getattr(
            self, "_diffusion_lora_batch_usage_emitted", False
        ):
            emit_event(
                "lora_events",
                "lora_batch_slot_applied",
                module_name=getattr(self, "_edge_dit_lora_module_name", None),
                requested_slot_indices=sorted(
                    {slot for slot in batch_slot_indices if slot is not None}
                ),
                applied_slot_indices=sorted(applied_slot_indices),
                rows_per_request=rows_per_request,
                input_rows=int(x_flat.shape[0]),
                input_tensor=tensor_metadata(
                    x_flat,
                    role="lora_activation_input",
                ),
                output_tensor=tensor_metadata(
                    y_flat,
                    role="lora_activation_output",
                ),
            )
            self._diffusion_lora_batch_usage_emitted = True

        return y_flat.view(original_shape)

    def __getattr__(self, name: str):
        # The diffusion model implementations may access attributes directly
        # from linear layers (e.g. QKVParallelLinear.num_heads). vLLM's LoRA
        # wrappers don't forward these attributes by default, so we delegate
        # missing attribute lookups to the underlying base_layer.
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            base_layer = object.__getattribute__(self, "__dict__").get("_diffusion_base_layer_ref")
            if base_layer is None:
                base_layer = object.__getattribute__(self, "_modules").get("base_layer")
            if base_layer is None:
                raise exc
            try:
                return getattr(base_layer, name)
            except AttributeError:
                raise exc
