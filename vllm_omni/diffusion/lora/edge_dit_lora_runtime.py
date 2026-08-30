# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped runtime for edge-owned DiT LoRA residual injection."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_omni.diffusion.experiment_telemetry import emit_event, tensor_metadata
from vllm_omni.diffusion.lora.lora_compute_breakdown import (
    finish_interval,
    start_interval,
)
from vllm_omni.diffusion.lora.dit_lora_overlap import runtime as lora_runtime

try:
    from vllm_omni.diffusion.lora.edge_dit_lora_provider import (
        EdgeDiTLoRAResidualProvider,
    )
except Exception:  # pragma: no cover
    EdgeDiTLoRAResidualProvider = Any  # type: ignore[misc, assignment]


@dataclass
class EdgeDiTLoRARuntimeState:
    metadata: dict[str, dict[str, Any]] | None = None
    providers: dict[str, Any] = field(default_factory=dict)
    call_counts: dict[str, Any] = field(
        default_factory=lambda: {"calls": 0, "module_calls": {}}
    )
    last_provider_meta: dict[str, Any] | None = None
    provider_meta_log: list[dict[str, Any]] = field(default_factory=list)


_STATE = EdgeDiTLoRARuntimeState()
_DEBUG_PATH = Path("/tmp/edge_dit_runtime_debug.jsonl")


def reset_edge_dit_lora_runtime_state() -> None:
    for provider in _STATE.providers.values():
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    _STATE.metadata = None
    _STATE.providers.clear()
    _STATE.call_counts = {"calls": 0, "module_calls": {}}
    _STATE.last_provider_meta = None
    _STATE.provider_meta_log.clear()


def remove_edge_dit_lora_runtime_request(request_id: str) -> None:
    runtime_request_id = str(request_id)
    if _STATE.metadata is not None:
        _STATE.metadata.pop(runtime_request_id, None)
    provider = _STATE.providers.pop(runtime_request_id, None)
    close = getattr(provider, "close", None)
    if callable(close):
        close()


def consume_provider_meta_log() -> list[dict[str, Any]]:
    log = list(_STATE.provider_meta_log)
    _STATE.provider_meta_log.clear()
    return log


def _debug_event(event: str, **payload: Any) -> None:
    try:
        with _DEBUG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, **payload}, sort_keys=True) + "\n")
    except Exception:
        pass


def _normalize_edge_module_name(module_name: str) -> str:
    if module_name.startswith("transformer."):
        return module_name[len("transformer.") :]
    return module_name


def _provider_for_decision(
    decision: dict[str, Any],
    existing: Any | None,
) -> Any:
    from vllm_omni.diffusion.lora.edge_dit_lora_provider import (
        EdgeDiTLoRAResidualProvider as _Prov,
    )

    host = str(decision.get("edge_worker_host", "127.0.0.1"))
    port = int(decision.get("edge_worker_port", 0))
    adapter_id = str(decision.get("adapter_id", ""))
    precision = str(decision.get("wire_precision", "fp16"))
    fail_closed = bool(decision.get("fail_closed", True))
    timeout_s = float(decision.get("timeout_s", 30.0))
    profile_residual = bool(decision.get("profile_residual", False))
    if existing is not None and isinstance(existing, _Prov):
        if (
            existing.host == host
            and existing.port == port
            and existing.adapter_id == adapter_id
            and existing.precision == precision
            and existing.fail_closed == fail_closed
            and existing.timeout_s == timeout_s
            and existing.profile_residual == profile_residual
        ):
            return existing
        existing.close()
    return _Prov(
        host=host,
        port=port,
        adapter_id=adapter_id,
        precision=precision,
        fail_closed=fail_closed,
        timeout_s=timeout_s,
        profile_residual=profile_residual,
    )


def install_edge_dit_lora_runtime_from_sampling(
    sampling_params: Any,
    request_id: str,
) -> None:
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    if not isinstance(extra_args, dict):
        _debug_event("install_skip_extra_args", request_id=str(request_id))
        return
    metadata = extra_args.get("edge_dit_lora_metadata")
    if not isinstance(metadata, dict) or not metadata:
        _debug_event("install_skip_metadata", request_id=str(request_id))
        return
    decision = metadata.get(str(request_id))
    metadata_request_id = str(request_id)
    if not isinstance(decision, dict):
        valid = [
            (str(meta_request_id), item)
            for meta_request_id, item in metadata.items()
            if isinstance(item, dict)
        ]
        if len(valid) != 1:
            raise ValueError(
                "edge DiT LoRA metadata must contain the runtime request id or "
                "exactly one reusable decision"
            )
        metadata_request_id, decision = valid[0]

    runtime_request_id = str(request_id)
    if _STATE.metadata is None:
        _STATE.metadata = {}
    _STATE.metadata[runtime_request_id] = decision
    existing_provider = _STATE.providers.get(runtime_request_id)
    provider = _provider_for_decision(decision, existing_provider)
    _STATE.providers[runtime_request_id] = provider
    _debug_event(
        "install_ok",
        request_id=runtime_request_id,
        metadata_request_id=metadata_request_id,
        provider_host=getattr(provider, "host", ""),
        provider_port=getattr(provider, "port", 0),
        adapter_id=getattr(provider, "adapter_id", ""),
    )
    emit_event(
        "lora_events",
        "edge_lora_provider_install",
        request_id=runtime_request_id,
        metadata_request_id=metadata_request_id,
        adapter_id=getattr(provider, "adapter_id", ""),
        provider_host=getattr(provider, "host", ""),
        provider_port=getattr(provider, "port", 0),
        provider_reused=existing_provider is provider,
    )


def _remote_decision_for_module(
    request_id: str,
    module_name: str,
) -> tuple[dict[str, Any], Any] | None:
    if not _STATE.metadata:
        return None
    decision = _STATE.metadata.get(request_id)
    provider = _STATE.providers.get(request_id)
    if not isinstance(decision, dict) or provider is None:
        return None
    runtime = lora_runtime()
    step_start = decision.get("active_step_start")
    step_end = decision.get("active_step_end")
    if step_start is not None and step_end is not None:
        if not int(step_start) <= int(runtime.step_index) <= int(step_end):
            return None
    block_start = decision.get("active_block_start")
    block_end = decision.get("active_block_end")
    if block_start is not None and block_end is not None:
        match = re.search(r"(?:layers|noise_refiner|context_refiner)\.(\d+)", module_name)
        if match is None or not int(block_start) <= int(match.group(1)) <= int(block_end):
            return None
    placement_by_module = decision.get("placement_by_module", {}) or {}
    target_module_names = {
        str(name) for name in list(decision.get("target_module_names") or [])
    }
    if module_name not in placement_by_module and module_name not in target_module_names:
        return None
    placement = str(
        placement_by_module.get(module_name, decision.get("placement", "edge_remote"))
    )
    if placement not in {"edge_remote", "edge_prefetch", "edge_async"}:
        return None
    return decision, provider


def _provider_group_key(decision: dict[str, Any], provider: Any) -> tuple[Any, ...]:
    return (
        getattr(provider, "host", decision.get("edge_worker_host")),
        int(getattr(provider, "port", decision.get("edge_worker_port", 0))),
        str(getattr(provider, "adapter_id", decision.get("adapter_id", ""))),
        str(decision.get("wire_precision", "fp16")),
        float(decision.get("scale", 1.0)),
    )


def maybe_edge_dit_lora_a_projection(
    module_name: str,
    x: torch.Tensor,
    batch_slot_indices: tuple[int | None, ...],
    response_features: int,
    module_out_features: int,
    route_on_edge: bool,
) -> list[dict[str, Any]] | None:
    """Run the LoRA-A projection remotely for an explicit split mode.

    ``split01`` sends the complete activation and row indices to the edge, so
    row selection happens there. ``split02`` selects rows on the cloud first
    and sends only the selected activation rows. In both cases the returned
    records retain the cloud row mask needed for the local B projection and
    writeback.
    """
    if not _STATE.metadata or not batch_slot_indices:
        return None
    routing_interval = start_interval(
        "lora_routing_prepare",
        module_name=module_name,
        split_mode="split01_or_split02",
        execution_location="cloud",
    )
    runtime = lora_runtime()
    request_ids = tuple(
        str(request_id)
        for request_id in getattr(runtime, "batch_request_ids", ())
    )
    if not request_ids:
        if runtime.request_id is not None:
            request_ids = (str(runtime.request_id),)
        elif len(_STATE.metadata) == 1:
            request_ids = (str(next(iter(_STATE.metadata))),)
    repeat = int(getattr(runtime, "batch_repeat", 1))
    expanded_request_ids = request_ids * repeat
    if not expanded_request_ids or x.shape[0] % len(expanded_request_ids):
        return None
    rows_per_request = x.shape[0] // len(expanded_request_ids)
    row_request_ids = [
        request_id
        for request_id in expanded_request_ids
        for _ in range(rows_per_request)
    ]
    slot_by_request = {
        request_id: batch_slot_indices[index]
        for index, request_id in enumerate(request_ids)
        if index < len(batch_slot_indices)
    }
    resolved = {
        request_id: _remote_decision_for_module(request_id, module_name)
        for request_id in set(request_ids)
    }
    finish_interval(
        routing_interval,
        request_count=len(request_ids),
        resolved_request_count=len(resolved),
    )
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for request_id, item in resolved.items():
        if item is None:
            continue
        decision, provider = item
        if str(decision.get("split_mode", "none")) not in {
            "split01",
            "split02",
        }:
            continue
        group = groups.setdefault(
            _provider_group_key(decision, provider),
            {"decision": decision, "provider": provider, "request_ids": []},
        )
        group["request_ids"].append(request_id)
    if not groups:
        return None

    slot_interval = start_interval(
        "lora_slot_unique",
        module_name=module_name,
        split_mode="split01_or_split02",
        execution_location="cloud",
    )
    row_slot_indices = [slot_by_request.get(request_id) for request_id in row_request_ids]
    unique_slot_indices = sorted({int(slot) for slot in row_slot_indices if slot is not None})
    finish_interval(slot_interval, unique_slot_indices=unique_slot_indices)
    remote_module_name = _normalize_edge_module_name(str(module_name))
    results: list[dict[str, Any]] = []
    for group in groups.values():
        decision = group["decision"]
        provider = group["provider"]
        group_request_ids = tuple(group["request_ids"])
        precision = str(decision.get("wire_precision", "fp16"))
        split_mode = str(decision.get("split_mode", "none"))
        row_interval = start_interval(
            "lora_row_mask",
            module_name=module_name,
            split_mode=split_mode,
            execution_location="cloud",
        )
        selected_rows = [
            index for index, request_id in enumerate(row_request_ids)
            if request_id in group_request_ids
        ]
        row_mask = torch.zeros(x.shape[0], device=x.device, dtype=torch.bool)
        if selected_rows:
            row_mask[torch.tensor(selected_rows, device=x.device)] = True
        finish_interval(row_interval, selected_rows=len(selected_rows))
        input_interval = start_interval(
            "lora_input_gather",
            module_name=module_name,
            split_mode=split_mode,
            execution_location="cloud",
        )
        if split_mode == "split01" and route_on_edge:
            edge_input = x
            edge_rows = selected_rows
        else:
            edge_input = x[row_mask]
            edge_rows = None
        finish_interval(input_interval, selected_input_shape=list(edge_input.shape))
        call_start_ns = time.monotonic_ns()
        a_interval = start_interval(
            "lora_a_gemm",
            module_name=module_name,
            split_mode=split_mode,
            execution_location="edge",
            input_shape=list(edge_input.shape),
        )
        hidden = provider.compute_a_projection(
            remote_module_name,
            edge_input,
            response_features=response_features,
            module_out_features=module_out_features,
            precision=precision,
            row_indices=edge_rows,
        )
        finish_interval(
            a_interval,
            execution_location="edge",
            output_shape=list(hidden.shape),
        )
        _STATE.call_counts["calls"] = int(
            _STATE.call_counts.get("calls", 0)
        ) + 1
        module_calls = _STATE.call_counts.setdefault("module_calls", {})
        module_calls[module_name] = int(module_calls.get(module_name, 0)) + 1
        call_index = int(_STATE.call_counts["calls"])
        module_call_index = int(module_calls[module_name])
        provider_meta = getattr(provider, "last_meta", None)
        if isinstance(provider_meta, dict):
            _STATE.last_provider_meta = provider_meta
            _STATE.provider_meta_log.append(
                {
                    "call_index": call_index,
                    "module_call_index": module_call_index,
                    "request_id": group_request_ids[0],
                    "runtime_request_ids": list(group_request_ids),
                    "denoise_step_index": int(runtime.step_index),
                    "module_name": str(module_name),
                    "operation": "lora_a_projection",
                    "split_mode": split_mode,
                    **provider_meta,
                }
            )
        emit_event(
            "lora_events",
            "edge_lora_residual",
            request_id=group_request_ids[0],
            runtime_request_ids=list(group_request_ids),
            denoise_step_index=int(runtime.step_index),
            call_index=call_index,
            module_call_index=module_call_index,
            module_name=str(module_name),
            remote_module_name=remote_module_name,
            operation="lora_a_projection",
            split_mode=split_mode,
            input_tensor=tensor_metadata(
                edge_input, role="lora_activation_input"
            ),
            residual_tensor=tensor_metadata(
                hidden, role="lora_a_projection_output"
            ),
            invocation_elapsed_ms=(time.monotonic_ns() - call_start_ns)
            / 1_000_000,
            provider_meta=(
                provider_meta if isinstance(provider_meta, dict) else None
            ),
        )
        results.append(
            {
                "row_mask": row_mask,
                "hidden": hidden,
                "slot_indices": tuple(
                    sorted(
                        {
                            int(slot_by_request[request_id])
                            for request_id in group_request_ids
                            if slot_by_request.get(request_id) is not None
                        }
                    )
                ),
                "request_ids": group_request_ids,
                "decision": decision,
                "provider_meta": provider_meta,
            }
        )
    return results


def maybe_edge_dit_lora_residual(
    module_name: str,
    x: torch.Tensor,
    out_features: int,
) -> torch.Tensor | None:
    if not _STATE.metadata:
        _debug_event("call_skip_no_metadata", module_name=str(module_name))
        return None
    runtime = lora_runtime()
    request_ids = tuple(
        str(request_id)
        for request_id in getattr(runtime, "batch_request_ids", ())
    )
    if not request_ids:
        if runtime.request_id is not None:
            request_ids = (str(runtime.request_id),)
        elif len(_STATE.metadata) == 1:
            request_ids = (str(next(iter(_STATE.metadata))),)
    repeat = int(getattr(runtime, "batch_repeat", 1))
    if repeat < 1:
        raise RuntimeError("edge DiT LoRA batch repeat must be positive")
    expanded_request_ids = request_ids * repeat
    if not expanded_request_ids:
        return None
    if x.shape[0] % len(expanded_request_ids):
        raise RuntimeError(
            "edge DiT LoRA request mapping does not divide flattened activations: "
            f"rows={x.shape[0]}, batch_entries={len(expanded_request_ids)}"
        )

    rows_per_batch_entry = x.shape[0] // len(expanded_request_ids)
    row_request_ids = [
        request_id
        for request_id in expanded_request_ids
        for _ in range(rows_per_batch_entry)
    ]
    resolved = {
        request_id: _remote_decision_for_module(request_id, module_name)
        for request_id in set(request_ids)
    }
    remote_request_ids = {
        request_id for request_id, item in resolved.items() if item is not None
    }
    if not remote_request_ids:
        return None
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for request_id in sorted(remote_request_ids):
        decision, provider = resolved[request_id]  # type: ignore[misc]
        if str(decision.get("split_mode", "none")) in {"split01", "split02"}:
            continue
        group = groups.setdefault(
            _provider_group_key(decision, provider),
            {
                "decision": decision,
                "provider": provider,
                "request_ids": [],
            },
        )
        group["request_ids"].append(request_id)

    if not groups:
        return None

    remote_module_name = _normalize_edge_module_name(str(module_name))
    denoise_step_index = int(runtime.step_index)
    delta = torch.zeros(
        (x.shape[0], int(out_features)),
        device=x.device,
        dtype=x.dtype,
    )
    for group in groups.values():
        decision = group["decision"]
        provider = group["provider"]
        group_request_ids = tuple(group["request_ids"])
        row_mask = torch.tensor(
            [request_id in group_request_ids for request_id in row_request_ids],
            device=x.device,
            dtype=torch.bool,
        )
        precision = str(decision.get("wire_precision", "fp16"))
        scale = float(decision.get("scale", 1.0))
        placement = str(
            (decision.get("placement_by_module", {}) or {}).get(
                module_name, decision.get("placement", "edge_remote")
            )
        )
        call_start_ns = time.monotonic_ns()
        try:
            group_delta = provider.compute_residual(
                remote_module_name,
                x[row_mask],
                out_features,
                precision=precision,
            )
        except Exception as exc:
            emit_event(
                "lora_events",
                "edge_lora_residual_error",
                request_id=group_request_ids[0],
                runtime_request_ids=list(group_request_ids),
                denoise_step_index=denoise_step_index,
                module_name=str(module_name),
                remote_module_name=remote_module_name,
                placement=placement,
                input_tensor=tensor_metadata(
                    x[row_mask], role="lora_activation_input"
                ),
                invocation_elapsed_ms=(time.monotonic_ns() - call_start_ns)
                / 1_000_000,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        _STATE.call_counts["calls"] = int(_STATE.call_counts.get("calls", 0)) + 1
        module_calls = _STATE.call_counts.setdefault("module_calls", {})
        module_calls[module_name] = int(module_calls.get(module_name, 0)) + 1
        call_index = int(_STATE.call_counts["calls"])
        module_call_index = int(module_calls[module_name])
        provider_meta = getattr(provider, "last_meta", None)
        if isinstance(provider_meta, dict):
            _STATE.last_provider_meta = provider_meta
            _STATE.provider_meta_log.append(
                {
                    "call_index": call_index,
                    "module_call_index": module_call_index,
                    "request_id": group_request_ids[0],
                    "runtime_request_ids": list(group_request_ids),
                    "denoise_step_index": denoise_step_index,
                    "module_name": str(module_name),
                    **provider_meta,
                }
            )
        emit_event(
            "lora_events",
            "edge_lora_residual",
            request_id=group_request_ids[0],
            runtime_request_ids=list(group_request_ids),
            denoise_step_index=denoise_step_index,
            call_index=call_index,
            module_call_index=module_call_index,
            module_name=str(module_name),
            remote_module_name=remote_module_name,
            placement=placement,
            input_tensor=tensor_metadata(x[row_mask], role="lora_activation_input"),
            residual_tensor=tensor_metadata(
                group_delta, role="lora_residual_output"
            ),
            invocation_elapsed_ms=(time.monotonic_ns() - call_start_ns) / 1_000_000,
            provider_meta=provider_meta if isinstance(provider_meta, dict) else None,
        )
        _debug_event(
            "call_ok",
            request_ids=list(group_request_ids),
            module_name=str(module_name),
            remote_module_name=remote_module_name,
            call_index=call_index,
            module_call_index=module_call_index,
            denoise_step_index=denoise_step_index,
        )
        delta[row_mask] = group_delta * scale
    return delta


def edge_dit_lora_runtime_state() -> EdgeDiTLoRARuntimeState:
    return _STATE
