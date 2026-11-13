"""Validate a strict reference/MixFusion experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from mixfusion_contract import (
    CellManifest,
    ContractError,
    compare_manifests,
    evaluate_speedup,
    load_json,
    load_jsonl,
    require_correctness_gate,
    require_runtime_coverage,
    summarize_timing,
)


def _cell_paths(prefix: Path) -> dict[str, Path]:
    """Resolve the required files below one cell directory."""
    return {
        "manifest": prefix / "run_manifest.json",
        "timing": prefix / "raw" / "timing.jsonl",
        "correctness": prefix / "raw" / "correctness.json",
        "telemetry": prefix / "raw" / "telemetry.jsonl",
    }


def _validate_cell(prefix: Path) -> tuple[CellManifest, Any]:
    """Validate one cell and return its manifest and timing summary."""
    paths = _cell_paths(prefix)
    manifest = CellManifest.from_mapping(load_json(paths["manifest"]))
    timing = summarize_timing(
        load_jsonl(paths["timing"]),
        manifest.measured,
        expected_batch_size=manifest.actual_batch_size,
        expected_active_slots=manifest.active_lora_slots,
        expected_mode=manifest.mode,
    )
    require_correctness_gate(load_json(paths["correctness"]))
    if manifest.variant != "reference_both":
        require_runtime_coverage(
            load_jsonl(paths["telemetry"]), manifest.variant
        )
    return manifest, timing


def main() -> int:
    """Validate reference and System B MixFusion cells."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mixfusion", required=True, type=Path)
    args = parser.parse_args()

    try:
        cells = {
            "reference_both": _validate_cell(args.reference),
            "mixfusion_both": _validate_cell(args.mixfusion),
        }
        reference_manifest, reference_timing = cells["reference_both"]
        mismatches: dict[str, list[str]] = {}
        for variant, (manifest, _) in cells.items():
            if variant == "reference_both":
                continue
            differences = compare_manifests(reference_manifest, manifest)
            if differences:
                mismatches[variant] = differences
        if mismatches:
            raise ContractError(f"manifest mismatch: {mismatches}")
        result: dict[str, Any] = {
            "valid": True,
            "code_sha": reference_manifest.code_sha,
            "resolution": reference_manifest.resolution,
            "batch_size": reference_manifest.actual_batch_size,
            "timing": {},
            "speedups": {},
        }
        for variant, (manifest, timing) in cells.items():
            result["timing"][variant] = timing.__dict__
            if variant != "reference_both":
                result["speedups"][variant] = evaluate_speedup(
                    reference_timing, timing
                )
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
