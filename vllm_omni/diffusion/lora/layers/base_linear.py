# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import re

import torch
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm_omni.diffusion.experiment_telemetry import emit_event, tensor_metadata
from vllm_omni.diffusion.lora.lora_compute_breakdown import (
    finish_interval,
    record_lora_flops,
    start_interval,
)

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

    def reset_lora(self, index: int):
        super().reset_lora(index)
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
            active_slices = []
            for a_i, b_i in zip(lora_a[:n_slices], lora_b[:n_slices]):
                active_slices.append(a_i is not None and b_i is not None)
            if len(active_slices) < n_slices:
                active_slices.extend([False] * (n_slices - len(active_slices)))
            self._diffusion_lora_active_slices = tuple(active_slices)
        else:
            # Single-slice layer.
            self._diffusion_lora_active_slices = (True,)

    def set_batch_slot_indices(self, slot_indices: tuple[int | None, ...]) -> None:
        """Set one active LoRA slot per logical request in the next forward."""
        self._diffusion_lora_batch_slot_indices = tuple(slot_indices)
        self._diffusion_lora_batch_usage_emitted = False

    def _apply_split_lora(
        self,
        module_name: str,
        x_flat: torch.Tensor,
        y_flat: torch.Tensor,
        output_slices: tuple[int, ...],
        batch_slot_indices: tuple[int | None, ...],
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

        row_slot_indices = torch.repeat_interleave(
            torch.tensor(
                [slot if slot is not None else -1 for slot in batch_slot_indices],
                device=x_flat.device,
                dtype=torch.long,
            ),
            x_flat.shape[0] // len(batch_slot_indices),
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

        batch_slot_indices = getattr(
            self, "_diffusion_lora_batch_slot_indices", None
        )
        if batch_slot_indices is None:
            batch_slot_indices = (None,)
        if not batch_slot_indices or all(
            slot is None for slot in batch_slot_indices
        ):
            return output

        lora_path_interval = start_interval(
            "lora_path_e2e",
            module_name=module_name,
            execution_location="cloud",
            input_shape=list(x_flat.shape),
            output_shape=list(y_flat.shape),
        )

        split_output = self._apply_split_lora(
            str(module_name),
            x_flat,
            y_flat,
            tuple(int(size) for size in output_slices),
            tuple(batch_slot_indices),
        )
        if split_output is not None:
            finish_interval(
                lora_path_interval,
                execution_location="split",
                output_shape=list(split_output.shape),
            )
            return split_output.view(original_shape)
        diagnostic_mode = os.environ.get(
            "VLLM_OMNI_LORA_DIAGNOSTIC_MODE", "normal"
        ).strip().lower()
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
