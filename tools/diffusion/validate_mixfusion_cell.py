"""Validate one MixFusion experiment cell without importing vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mixfusion_contract import (
    CellManifest,
    ContractError,
    compare_manifests,
    load_json,
    load_jsonl,
    require_correctness_gate,
    require_runtime_coverage,
    summarize_timing,
    validate_batch_and_slots,
)


def main() -> int:
    """Validate manifest, timing, telemetry, and correctness inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--timing", required=True, type=Path)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--actual-batch-size", action="append", type=int)
    parser.add_argument("--active-slot-count", action="append", type=int)
    args = parser.parse_args()

    try:
        manifest = CellManifest.from_mapping(load_json(args.manifest))
        timing_rows = load_jsonl(args.timing)
        timing = summarize_timing(
            timing_rows,
            manifest.measured,
            expected_batch_size=manifest.actual_batch_size,
            expected_active_slots=manifest.active_lora_slots,
            expected_mode=manifest.mode,
        )
        if args.telemetry is not None:
            require_runtime_coverage(
                load_jsonl(args.telemetry), manifest.variant
            )
        validate_batch_and_slots(
            args.actual_batch_size or [manifest.actual_batch_size],
            args.active_slot_count or [manifest.active_lora_slots],
            manifest.batch_size,
        )
        require_correctness_gate(load_json(args.correctness))
        mismatches: list[str] = []
        if args.reference_manifest is not None:
            reference = CellManifest.from_mapping(
                load_json(args.reference_manifest)
            )
            mismatches = compare_manifests(reference, manifest)
            if mismatches:
                raise ContractError("manifest mismatch: " + "; ".join(mismatches))
        result = {
            "valid": True,
            "cell": manifest.cell,
            "variant": manifest.variant,
            "timing": timing.__dict__,
            "manifest_mismatches": mismatches,
        }
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"valid": False, "error": str(exc)}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
