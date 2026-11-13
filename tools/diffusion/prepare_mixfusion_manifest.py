"""Create a strict, reproducible MixFusion experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from mixfusion_contract import CellManifest, ContractError, CONTAINER_NAME


def _sha256(path: Path) -> str:
    """Hash an adapter manifest without loading it as executable data."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git_sha(repo: Path) -> str:
    """Return the exact source revision used for the cell."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError("unable to resolve code_sha") from exc


def _rank_mapping(value: str) -> list[int]:
    """Parse a non-empty comma-separated rank mapping."""
    try:
        ranks = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ContractError("rank mapping must contain integers") from exc
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ContractError("rank mapping must contain positive ranks")
    return ranks


def main() -> int:
    """Validate arguments and write one fail-closed cell manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--cell", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--active-lora-slots", required=True, type=int)
    parser.add_argument("--adapter-manifest", required=True, type=Path)
    parser.add_argument("--rank-mapping", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--measured", required=True, type=int)
    parser.add_argument("--mode", default="timing")
    parser.add_argument(
        "--profiler-in-timing", action="store_true", default=False
    )
    parser.add_argument("--denoise-steps", type=int, default=1)
    parser.add_argument(
        "--timing-window", default="denoise_forward_only"
    )
    parser.add_argument(
        "--attention-semantics", default="reference_unchanged"
    )
    parser.add_argument(
        "--request-semantics", default="real_flux_imagegem"
    )
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument(
        "--adapter-loaded-before-warmup", action="store_true"
    )
    parser.add_argument("--code-sha")
    args = parser.parse_args()

    try:
        adapter_manifest = args.adapter_manifest.resolve()
        if not adapter_manifest.is_file():
            raise ContractError(f"adapter manifest does not exist: {adapter_manifest}")
        if args.container != CONTAINER_NAME:
            raise ContractError(
                f"wrong container: {args.container!r}; expected {CONTAINER_NAME!r}"
            )
        raw: dict[str, Any] = {
            "cell": args.cell,
            "variant": args.variant,
            "container": args.container,
            "code_sha": args.code_sha or _git_sha(args.repo),
            "model": args.model,
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "actual_batch_size": args.batch_size,
            "active_lora_slots": args.active_lora_slots,
            "adapter_manifest": str(adapter_manifest),
            "adapter_manifest_digest": _sha256(adapter_manifest),
            "adapter_loaded_before_warmup": args.adapter_loaded_before_warmup,
            "rank_mapping": _rank_mapping(args.rank_mapping),
            "seed": args.seed,
            "precision": args.precision,
            "warmup": args.warmup,
            "measured": args.measured,
            "denoise_steps": args.denoise_steps,
            "timing_window": args.timing_window,
            "attention_semantics": args.attention_semantics,
            "request_semantics": args.request_semantics,
            "full_model": args.full_model,
            "mode": args.mode,
            "profiler_in_timing": args.profiler_in_timing,
        }
        manifest = CellManifest.from_mapping(raw)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (ContractError, OSError, ValueError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "valid": True,
                "output": str(args.output),
                "code_sha": manifest.code_sha,
                "adapter_manifest_digest": manifest.adapter_manifest_digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
