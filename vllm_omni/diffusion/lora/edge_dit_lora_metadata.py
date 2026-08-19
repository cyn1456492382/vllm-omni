# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize edge DiT LoRA scheduler policies into runtime metadata."""

from __future__ import annotations

from typing import Any


def materialize_edge_dit_lora_runtime_metadata(
    decisions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Expand per-request policies into hook-consumable metadata.

    Args:
        decisions: Mapping of request id → raw scheduler policy dict.

    Returns:
        Mapping of request id → materialized runtime metadata.
    """
    materialized: dict[str, dict[str, Any]] = {}
    for req_id, raw in decisions.items():
        modules = [str(x) for x in list(raw.get("modules") or [])]
        placement = str(raw.get("placement", "edge_remote"))
        placement_by_module = dict(raw.get("placement_by_module") or {})
        for name in modules:
            placement_by_module.setdefault(name, placement)
        materialized[req_id] = {
            "target_module_names": modules,
            "placement_by_module": placement_by_module,
            "wire_precision": str(raw.get("wire_precision", "fp16")),
            "edge_worker_host": str(raw.get("edge_worker_host", "127.0.0.1")),
            "edge_worker_port": int(raw.get("edge_worker_port", 0)),
            "adapter_id": str(raw.get("adapter_id", "")),
            "scale": float(raw.get("scale", 1.0)),
            "fail_closed": bool(raw.get("fail_closed", True)),
            "timeout_s": float(raw.get("timeout_s", 30.0)),
            "block_coalesce": bool(raw.get("block_coalesce", False)),
        }
    return materialized
