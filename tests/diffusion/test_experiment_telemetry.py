# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json

from vllm_omni.diffusion.experiment_telemetry import emit_event


def test_suppress_lora_events_keeps_contract_mapping(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_EXPERIMENT_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_OMNI_SUPPRESS_LORA_EVENTS", "1")

    emit_event("lora_events", "sequential_adapter_applied", module_name="layer")
    emit_event(
        "lora_events",
        "lora_composition_mapping",
        unique_adapter_ids=[1],
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "lora_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["lora_composition_mapping"]
