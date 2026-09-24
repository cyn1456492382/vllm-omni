# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re

import torch
from vllm.lora.multi_lora import (
    MultiLoRAAdapter,
    execute_fused_lora,
    execute_tiled_fused_lora,
    execute_lora,
    prepare_lora_batch,
)
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.ops.triton_ops import (
    LoRAKernelMeta,
    lora_expand,
    lora_shrink,
)
from vllm.lora.ops.triton_ops.multi_lora_tile_plan import (
    build_multi_lora_tile_plan,
)
from vllm_omni.diffusion.experiment_telemetry import emit_event, tensor_metadata
from vllm_omni.diffusion.lora.lora_compute_breakdown import (
    finish_interval,
    record_lora_flops,
    start_interval,
)


def _multi_lora_operator_enabled() -> bool:
    value = os.environ.get("VLLM_OMNI_ENABLE_MULTI_LORA_OPERATOR", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _lora_diagnostic_mode() -> str:
    enabled = os.environ.get("VLLM_OMNI_ENABLE_LORA_DIAGNOSTIC", "")
    if enabled.strip().lower() not in {"1", "true", "yes", "on"}:
        return "normal"
    return os.environ.get("VLLM_OMNI_LORA_DIAGNOSTIC_MODE", "normal").strip().lower()


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
        match = re.search(r"(?:layers|noise_refiner|context_refiner)\.(\d+)", str(module_name))
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
                        registry[slot] = MultiLoRAAdapter(
                            adapter_id=slot,
                            lora_a=a_stack[slot, 0],
                            lora_b=b_stack[slot, 0],
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
            execute_lora(
                x_flat,
                y_flat[:, output_offset : output_offset + slice_size],
                plan,
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
    ) -> torch.Tensor:
        row_layout = self._get_batch_row_layout(x_flat, composition)
        rank_tables = getattr(
            self, "_diffusion_lora_slot_ranks_by_slice", None
        )
        if rank_tables is None:
            raise RuntimeError("LoRA rank metadata was not initialized")
        tile_plan_cache = getattr(self, "_diffusion_lora_tile_plan_cache", None)
        if tile_plan_cache is None:
            tile_plan_cache = {}
            self._diffusion_lora_tile_plan_cache = tile_plan_cache
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
            if implementation == "tile_fused":
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
                execute_tiled_fused_lora(
                    x_flat,
                    output_view,
                    self.lora_a_stacked[slice_index],
                    self.lora_b_stacked[slice_index],
                    row_layout,
                    relations,
                    scales,
                    rank_table,
                    tile_plan=tile_plan,
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
        base_interval = start_interval(
            "base_gemm",
            module_name=module_name,
            input_shape=list(x.shape),
            input_dtype=str(x.dtype),
        )
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        finish_interval(
            base_interval,
            output_shape=list(output.shape),
            output_dtype=str(output.dtype),
        )

        original_shape = output.shape
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = output.reshape(-1, output.shape[-1])

        if not self._lora_activation_enabled(module_name):
            return output

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
            return output
        if not self.lora_a_stacked or not self.lora_b_stacked:
            return output
        # Fast path: if no LoRA is active for this layer, skip matmuls.
        active_slices = getattr(self, "_diffusion_lora_active_slices", None)
        if active_slices is not None and not any(active_slices):
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
            # Fallback: infer slice sizes from the allocated tensors.
            output_slices = tuple(lora_b.shape[2] for lora_b in self.lora_b_stacked)

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
            batch_slot_indices = (0,)
        if composition is None and (
            not batch_slot_indices or all(slot is None for slot in batch_slot_indices)
        ):
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
        if operator_enabled and composition is not None:
            lora_path_interval = start_interval(
                "lora_path_e2e",
                module_name=module_name,
                execution_location="multi_lora_operator",
                input_shape=list(x_flat.shape),
                output_shape=list(y_flat.shape),
            )
            implementation = os.environ.get(
                "VLLM_OMNI_MULTI_LORA_IMPL", "fused"
            ).strip().lower()
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
            elif implementation in {"fused", "tile_fused"}:
                y_flat = self._apply_fused_multi_lora_operator(
                    x_flat,
                    y_flat,
                    tuple(int(size) for size in output_slices),
                    tuple(composition),
                    implementation,
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
