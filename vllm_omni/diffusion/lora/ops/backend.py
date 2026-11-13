# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit LoRA device-backend registry for Omni MixFusion / TetrisLoRA.

Performance backends must be named. Missing performance kernels fail closed.
Reference/debug math is only used when the caller explicitly selects
``reference`` (or legacy ``TETRIS_LORA_REFERENCE``). Silent FP32 fallback from
a performance backend is forbidden.

Canonical selector: ``VLLM_OMNI_LORA_BACKEND`` in
``{dense_fused, two_stage_tc, reference}``.
Legacy env mapping is documented in docs/tetris_lora/COMPATIBILITY_MATRIX.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

_TRUE = frozenset({"1", "true", "yes", "on"})

BACKEND_DENSE_FUSED = "dense_fused"
BACKEND_TWO_STAGE_TC = "two_stage_tc"
BACKEND_REFERENCE = "reference"
PERFORMANCE_BACKENDS = frozenset({BACKEND_DENSE_FUSED, BACKEND_TWO_STAGE_TC})
ALL_BACKENDS = frozenset(
    {BACKEND_DENSE_FUSED, BACKEND_TWO_STAGE_TC, BACKEND_REFERENCE}
)


class LoRABackendError(RuntimeError):
    """Raised when a requested LoRA backend cannot be used."""


@dataclass(frozen=True)
class LoRABackend:
    """One registered device backend."""

    name: str
    is_performance: bool
    loader: Callable[[], Any]
    description: str


_REGISTRY: dict[str, LoRABackend] = {}
_LOADED: dict[str, Any] = {}


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE


def register_backend(backend: LoRABackend) -> None:
    """Register or replace a named backend."""
    if backend.name not in ALL_BACKENDS:
        raise LoRABackendError(
            f"unsupported LoRA backend name {backend.name!r}; "
            f"expected one of {sorted(ALL_BACKENDS)}"
        )
    _REGISTRY[backend.name] = backend
    _LOADED.pop(backend.name, None)


def _load_dense_fused() -> Any:
    from . import dense_fused_lora_op

    return dense_fused_lora_op


def _load_two_stage() -> Any:
    from vllm_omni.diffusion.lora.tetris_lora_twostage import (
        execute_two_stage_lora,
    )

    return execute_two_stage_lora


def _load_reference() -> Any:
    """Reference marker object; callers implement explicit torch math."""

    return {"kind": BACKEND_REFERENCE}


def _ensure_defaults() -> None:
    if BACKEND_DENSE_FUSED not in _REGISTRY:
        register_backend(
            LoRABackend(
                name=BACKEND_DENSE_FUSED,
                is_performance=True,
                loader=_load_dense_fused,
                description=(
                    "Omni-vendored dense fused Triton LoRA "
                    "(copied from fork vllm/lora/ops/triton_ops/"
                    "dense_fused_lora_op.py @ bf69e4612)"
                ),
            )
        )
    if BACKEND_TWO_STAGE_TC not in _REGISTRY:
        register_backend(
            LoRABackend(
                name=BACKEND_TWO_STAGE_TC,
                is_performance=True,
                loader=_load_two_stage,
                description="Omni two-stage Tensor-Core LoRA",
            )
        )
    if BACKEND_REFERENCE not in _REGISTRY:
        register_backend(
            LoRABackend(
                name=BACKEND_REFERENCE,
                is_performance=False,
                loader=_load_reference,
                description="Explicit reference/debug path only",
            )
        )


def resolve_backend_name(
    *,
    explicit: str | None = None,
    for_mixfusion: bool = False,
) -> str:
    """Resolve the backend name from explicit arg and environment.

    Args:
      explicit: Caller-forced backend name.
      for_mixfusion: If true, honor ``VLLM_OMNI_TETRIS_TWO_STAGE_TC`` when
        ``VLLM_OMNI_LORA_BACKEND`` is unset.

    Returns:
      Canonical backend name.
    """
    # Keep registry mutation / env reads outside Dynamo to avoid graph breaks
    # on ``_LOADED`` / ``_REGISTRY`` during MixFusion forwards.
    try:
        import torch

        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            if torch.compiler.is_compiling():
                # Fall through; callers should resolve outside compile.
                pass
    except Exception:
        pass
    _ensure_defaults()
    if explicit:
        name = explicit.strip().lower()
    else:
        name = (os.environ.get("VLLM_OMNI_LORA_BACKEND") or "").strip().lower()
        if not name:
            if _is_true(os.environ.get("TETRIS_LORA_REFERENCE")):
                name = BACKEND_REFERENCE
            elif for_mixfusion and _is_true(
                os.environ.get("VLLM_OMNI_TETRIS_TWO_STAGE_TC")
            ):
                name = BACKEND_TWO_STAGE_TC
            elif for_mixfusion and _is_true(
                os.environ.get("VLLM_OMNI_FORCE_DENSE_FUSED")
            ):
                # Explicit dense_fused contrast path (repeated Shrink measured).
                name = BACKEND_DENSE_FUSED
            elif for_mixfusion:
                # Prefer two_stage_tc for MixFusion / Fair M* (no per-N Shrink).
                name = BACKEND_TWO_STAGE_TC
            else:
                name = BACKEND_DENSE_FUSED
    if name not in ALL_BACKENDS:
        raise LoRABackendError(
            f"unknown VLLM_OMNI_LORA_BACKEND={name!r}; "
            f"expected one of {sorted(ALL_BACKENDS)}"
        )
    return name


def require_backend(name: str) -> Any:
    """Load a backend or raise LoRABackendError (fail-closed).

    Args:
      name: Canonical backend name.

    Returns:
      Loaded backend module or callable payload.
    """
    _ensure_defaults()
    if name not in _REGISTRY:
        raise LoRABackendError(
            f"LoRA backend {name!r} is not registered"
        )
    backend = _REGISTRY[name]
    if name in _LOADED:
        return _LOADED[name]
    try:
        loaded = backend.loader()
    except Exception as exc:  # noqa: BLE001 - surface any load failure
        raise LoRABackendError(
            f"LoRA performance backend {name!r} failed to load: {exc}. "
            f"Description: {backend.description}. "
            "Silent fallback to reference/FP32 is disabled."
        ) from exc
    if loaded is None:
        raise LoRABackendError(
            f"LoRA backend {name!r} loaded as None ({backend.description})"
        )
    _LOADED[name] = loaded
    return loaded


def get_dense_fused_ops():
    """Return the Omni-vendored dense_fused module (fail-closed)."""
    return require_backend(BACKEND_DENSE_FUSED)


def probe_backends() -> dict[str, dict[str, Any]]:
    """Probe registration/load status for diagnostics (does not fallback)."""
    _ensure_defaults()
    report: dict[str, dict[str, Any]] = {}
    for name, backend in _REGISTRY.items():
        entry: dict[str, Any] = {
            "registered": True,
            "is_performance": backend.is_performance,
            "description": backend.description,
            "loaded": name in _LOADED,
            "error": None,
        }
        try:
            require_backend(name)
            entry["loaded"] = True
        except LoRABackendError as exc:
            entry["error"] = str(exc)
        report[name] = entry
    return report


def backend_capability_matrix() -> dict[str, dict[str, Any]]:
    """K6/K7: explicit output contract and rank limits (no silent widen)."""
    return {
        BACKEND_DENSE_FUSED: {
            "output_contract": "y_plus_equals_delta",
            "max_stored_rank": 32,
            "shrink_repeats_per_n_tile": True,
            "is_performance": True,
            "no_tile_supported": True,
        },
        BACKEND_TWO_STAGE_TC: {
            "output_contract": "pure_delta_then_caller_add",
            "max_stored_rank": 128,
            "shrink_repeats_per_n_tile": False,
            "is_performance": True,
            "no_tile_supported": True,
        },
        BACKEND_REFERENCE: {
            "output_contract": "y_plus_equals_delta_torch",
            "max_stored_rank": None,
            "shrink_repeats_per_n_tile": False,
            "is_performance": False,
            "no_tile_supported": True,
            "note": "multi_lora.execute_* torch matmul; not a fused kernel",
        },
    }


__all__ = [
    "ALL_BACKENDS",
    "BACKEND_DENSE_FUSED",
    "BACKEND_REFERENCE",
    "BACKEND_TWO_STAGE_TC",
    "LoRABackend",
    "LoRABackendError",
    "PERFORMANCE_BACKENDS",
    "backend_capability_matrix",
    "get_dense_fused_ops",
    "probe_backends",
    "register_backend",
    "require_backend",
    "resolve_backend_name",
]
