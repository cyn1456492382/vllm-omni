from __future__ import annotations

import pytest

from vllm_omni.diffusion.lora.mixfusion_upstream_scheduler import (
    NativeMixFusionScheduler,
    SchedulerRequest,
    is_heterogeneous_mode,
)


def test_homogeneous_scheduler_keeps_matching_signature(monkeypatch):
    monkeypatch.delenv("MIXFUSION_MULTILORA_ALLOW_HETEROGENEOUS", raising=False)
    monkeypatch.delenv("MIXFUSION_MULTILORA_TREATMENT_B", raising=False)
    scheduler = NativeMixFusionScheduler(
        (
            SchedulerRequest(0, (1,)),
            SchedulerRequest(1, (2,)),
            SchedulerRequest(2, (1,)),
        )
    )
    output = scheduler.schedule()
    assert output.request_indices == (0, 2)
    assert output.adapter_signature == (1,)


def test_heterogeneous_scheduler_keeps_all_and_marks_mixed(monkeypatch):
    monkeypatch.setenv("MIXFUSION_MULTILORA_ALLOW_HETEROGENEOUS", "1")
    monkeypatch.setenv("MIXFUSION_MULTILORA_TREATMENT_B", "1")
    scheduler = NativeMixFusionScheduler(
        (
            SchedulerRequest(0, (1,)),
            SchedulerRequest(1, (2,)),
            SchedulerRequest(2, (1,)),
        )
    )
    output = scheduler.schedule()
    assert output.request_indices == (0, 1, 2)
    assert output.adapter_signature is None


def _aba_requests() -> tuple[SchedulerRequest, ...]:
    return (
        SchedulerRequest(0, (1,)),
        SchedulerRequest(1, (2,)),
        SchedulerRequest(2, (1,)),
    )


def test_homogeneous_schedule_all_cycles_splits_aba(monkeypatch):
    monkeypatch.delenv("MIXFUSION_MULTILORA_ALLOW_HETEROGENEOUS", raising=False)
    monkeypatch.delenv("MIXFUSION_MULTILORA_TREATMENT_B", raising=False)
    cycles = NativeMixFusionScheduler(_aba_requests()).schedule_all_cycles()
    assert [cycle.request_indices for cycle in cycles] == [(0, 2), (1,)]
    assert cycles[0].adapter_signature == (1,)
    assert cycles[1].adapter_signature == (2,)


def test_heterogeneous_schedule_all_cycles_covers_all_once(monkeypatch):
    monkeypatch.setenv("MIXFUSION_MULTILORA_ALLOW_HETEROGENEOUS", "1")
    monkeypatch.setenv("MIXFUSION_MULTILORA_TREATMENT_B", "1")
    assert is_heterogeneous_mode()
    cycles = NativeMixFusionScheduler(_aba_requests()).schedule_all_cycles()
    assert len(cycles) == 1
    assert cycles[0].request_indices == (0, 1, 2)
    assert cycles[0].adapter_signature is None


def test_duplicate_request_index_raises():
    with pytest.raises(ValueError, match="duplicate MixFusion request_index=0"):
        NativeMixFusionScheduler(
            (
                SchedulerRequest(0, (1,)),
                SchedulerRequest(0, (2,)),
            )
        )


def test_treatment_b_alone_stays_homogeneous(monkeypatch):
    monkeypatch.delenv("MIXFUSION_MULTILORA_ALLOW_HETEROGENEOUS", raising=False)
    monkeypatch.setenv("MIXFUSION_MULTILORA_TREATMENT_B", "1")
    assert not is_heterogeneous_mode()
    cycles = NativeMixFusionScheduler(_aba_requests()).schedule_all_cycles()
    assert [cycle.request_indices for cycle in cycles] == [(0, 2), (1,)]
    assert cycles[0].adapter_signature == (1,)
    assert cycles[1].adapter_signature == (2,)
