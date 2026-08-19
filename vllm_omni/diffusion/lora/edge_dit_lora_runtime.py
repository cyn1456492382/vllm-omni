# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped runtime for edge-owned DiT LoRA residual injection."""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_omni.diffusion.experiment_telemetry import emit_event, tensor_metadata
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
    if existing is not None and isinstance(existing, _Prov):
        if (
            existing.host == host
            and existing.port == port
            and existing.adapter_id == adapter_id
            and existing.precision == precision
            and existing.fail_closed == fail_closed
            and existing.timeout_s == timeout_s
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
    if remote_request_ids != set(request_ids):
        missing = sorted(set(request_ids) - remote_request_ids)
        raise RuntimeError(
            "partial edge-remote mixed batch is unsupported for a shared linear "
            f"output; missing remote policy for requests={missing} module={module_name}"
        )

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for request_id in request_ids:
        decision, provider = resolved[request_id]  # type: ignore[misc]
        group = groups.setdefault(
            _provider_group_key(decision, provider),
            {
                "decision": decision,
                "provider": provider,
                "request_ids": [],
            },
        )
        group["request_ids"].append(request_id)

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
