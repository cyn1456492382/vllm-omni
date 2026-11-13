"""Unit tests for Omni LoRA backend registry (fail-closed)."""

from __future__ import annotations

import os

import pytest

from vllm_omni.diffusion.lora.ops import (
    BACKEND_DENSE_FUSED,
    BACKEND_REFERENCE,
    BACKEND_TWO_STAGE_TC,
    LoRABackendError,
    get_dense_fused_ops,
    probe_backends,
    require_backend,
    resolve_backend_name,
)
from vllm_omni.diffusion.lora.ops.backend import backend_capability_matrix


def test_resolve_mixfusion_default_is_two_stage(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_LORA_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_OMNI_TETRIS_TWO_STAGE_TC", raising=False)
    monkeypatch.delenv("VLLM_OMNI_FORCE_DENSE_FUSED", raising=False)
    monkeypatch.delenv("TETRIS_LORA_REFERENCE", raising=False)
    assert resolve_backend_name(for_mixfusion=True) == BACKEND_TWO_STAGE_TC
    assert resolve_backend_name(for_mixfusion=False) == BACKEND_DENSE_FUSED


def test_force_dense_fused_selects_dense(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_LORA_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_OMNI_TETRIS_TWO_STAGE_TC", raising=False)
    monkeypatch.delenv("TETRIS_LORA_REFERENCE", raising=False)
    monkeypatch.setenv("VLLM_OMNI_FORCE_DENSE_FUSED", "1")
    assert resolve_backend_name(for_mixfusion=True) == BACKEND_DENSE_FUSED


def test_resolve_two_stage_legacy_env(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_LORA_BACKEND", raising=False)
    monkeypatch.setenv("VLLM_OMNI_TETRIS_TWO_STAGE_TC", "1")
    assert resolve_backend_name(for_mixfusion=True) == BACKEND_TWO_STAGE_TC


def test_resolve_explicit_backend_overrides_env(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_LORA_BACKEND", BACKEND_DENSE_FUSED)
    assert (
        resolve_backend_name(explicit=BACKEND_TWO_STAGE_TC, for_mixfusion=True)
        == BACKEND_TWO_STAGE_TC
    )


def test_unknown_backend_fails_closed(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_LORA_BACKEND", "not_a_backend")
    with pytest.raises(LoRABackendError, match="unknown"):
        resolve_backend_name(for_mixfusion=True)


def test_dense_fused_ops_load_from_omni_package():
    mod = get_dense_fused_ops()
    assert mod.__name__.endswith("dense_fused_lora_op")
    assert "vllm_omni" in mod.__file__
    assert hasattr(mod, "dense_fused_lora_add")
    assert hasattr(mod, "shared_input_lora_add")


def test_probe_backends_reports_all():
    report = probe_backends()
    assert set(report) >= {
        BACKEND_DENSE_FUSED,
        BACKEND_TWO_STAGE_TC,
        BACKEND_REFERENCE,
    }
    assert report[BACKEND_DENSE_FUSED]["error"] is None
    assert require_backend(BACKEND_REFERENCE)["kind"] == BACKEND_REFERENCE


def test_backend_capability_matrix_contracts():
    matrix = backend_capability_matrix()
    assert matrix[BACKEND_DENSE_FUSED]["max_stored_rank"] == 32
    assert matrix[BACKEND_TWO_STAGE_TC]["max_stored_rank"] == 128
    assert matrix[BACKEND_DENSE_FUSED]["output_contract"] == "y_plus_equals_delta"
    assert (
        matrix[BACKEND_TWO_STAGE_TC]["output_contract"]
        == "pure_delta_then_caller_add"
    )
    assert matrix[BACKEND_REFERENCE]["is_performance"] is False
    assert all(entry["no_tile_supported"] for entry in matrix.values())


def test_mixfusion_module_has_no_parent_dense_import():
    import vllm_omni.diffusion.lora.mixfusion_multi_lora as mm

    source = open(mm.__file__, encoding="utf-8").read()
    assert "vllm.lora.ops.triton_ops.dense_fused_lora_op" not in source
