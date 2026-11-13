"""Fail-closed contracts and statistics for MixFusion experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


CONTAINER_NAME = "vllm-adapter-a800-tp2-cu124-20261027"
VARIANTS = frozenset({"reference_both", "mixfusion_both"})
TIMING_MODES = frozenset({"timing", "diagnostic", "profile"})
REQUIRED_SIGNATURE_FIELDS = (
    "model",
    "resolution",
    "batch_size",
    "actual_batch_size",
    "active_lora_slots",
    "adapter_manifest",
    "adapter_manifest_digest",
    "adapter_loaded_before_warmup",
    "rank_mapping",
    "seed",
    "precision",
    "warmup",
    "measured",
    "denoise_steps",
    "timing_window",
    "attention_semantics",
    "request_semantics",
    "full_model",
    "code_sha",
)


class ContractError(ValueError):
    """Raised when an experiment cell violates the measurement contract."""


@dataclass(frozen=True)
class CellManifest:
    """Normalized manifest fields shared by reference and variant cells."""

    cell: str
    variant: str
    container: str
    code_sha: str
    model: str
    resolution: str
    batch_size: int
    actual_batch_size: int
    active_lora_slots: int
    adapter_manifest: str
    adapter_manifest_digest: str
    adapter_loaded_before_warmup: bool
    rank_mapping: tuple[int, ...]
    seed: int
    precision: str
    warmup: int
    measured: int
    denoise_steps: int
    timing_window: str
    attention_semantics: str
    request_semantics: str
    full_model: bool
    mode: str
    profiler_in_timing: bool

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CellManifest":
        """Build and validate a manifest from JSON-compatible data."""
        missing = [field for field in REQUIRED_SIGNATURE_FIELDS if field not in raw]
        if missing:
            raise ContractError(f"manifest missing required fields: {missing}")
        manifest = cls(
            cell=str(raw.get("cell", "")),
            variant=str(raw.get("variant", "")),
            container=str(raw.get("container", "")),
            code_sha=str(raw.get("code_sha", "")),
            model=str(raw["model"]),
            resolution=str(raw["resolution"]),
            batch_size=int(raw["batch_size"]),
            actual_batch_size=int(raw["actual_batch_size"]),
            active_lora_slots=int(raw["active_lora_slots"]),
            adapter_manifest=str(raw["adapter_manifest"]),
            adapter_manifest_digest=str(raw["adapter_manifest_digest"]),
            adapter_loaded_before_warmup=bool(
                raw["adapter_loaded_before_warmup"]
            ),
            rank_mapping=tuple(int(rank) for rank in raw["rank_mapping"]),
            seed=int(raw["seed"]),
            precision=str(raw["precision"]),
            warmup=int(raw["warmup"]),
            measured=int(raw["measured"]),
            denoise_steps=int(raw["denoise_steps"]),
            timing_window=str(raw["timing_window"]),
            attention_semantics=str(raw["attention_semantics"]),
            request_semantics=str(raw["request_semantics"]),
            full_model=bool(raw["full_model"]),
            mode=str(raw.get("mode", "timing")),
            profiler_in_timing=bool(raw.get("profiler_in_timing", False)),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Reject manifest values that can invalidate a comparison."""
        errors: list[str] = []
        if not self.cell:
            errors.append("cell is empty")
        if self.variant not in VARIANTS:
            errors.append(f"unsupported variant: {self.variant!r}")
        if self.container != CONTAINER_NAME:
            errors.append(f"wrong container: {self.container!r}")
        if not self.code_sha:
            errors.append("code_sha is empty")
        if not self.model.startswith("FLUX/ImageGem"):
            errors.append("model is not a FLUX/ImageGem full pipeline")
        if self.batch_size < 1:
            errors.append("batch_size must be positive")
        if self.actual_batch_size != self.batch_size:
            errors.append("actual_batch_size must equal batch_size")
        if self.active_lora_slots < 2:
            errors.append("active_lora_slots must prove K>=2")
        if not self.adapter_manifest:
            errors.append("adapter_manifest is empty")
        if not self.adapter_manifest_digest:
            errors.append("adapter_manifest_digest is empty")
        if not self.adapter_loaded_before_warmup:
            errors.append("adapter_loaded_before_warmup must be true")
        if not self.rank_mapping or any(rank <= 0 for rank in self.rank_mapping):
            errors.append("rank_mapping must contain positive ranks")
        if self.precision.lower() not in {"bf16", "bfloat16"}:
            errors.append("precision must be BF16")
        if self.warmup < 2 or self.measured < 5:
            errors.append("warmup/measured must be at least 2/5")
        if self.denoise_steps != 1:
            errors.append("denoise_steps must be exactly one")
        if self.timing_window != "denoise_forward_only":
            errors.append("timing_window must be denoise_forward_only")
        if self.attention_semantics != "reference_unchanged":
            errors.append("attention_semantics must be reference_unchanged")
        if self.request_semantics != "real_flux_imagegem":
            errors.append("request_semantics must be real_flux_imagegem")
        if not self.full_model:
            errors.append("full_model must be true")
        if self.mode not in TIMING_MODES:
            errors.append(f"unsupported mode: {self.mode!r}")
        if self.mode == "timing" and self.profiler_in_timing:
            errors.append("profiler_in_timing must be false")
        if errors:
            raise ContractError("; ".join(errors))

    def comparison_signature(self) -> tuple[Any, ...]:
        """Return fields that must match across reference and variants."""
        return tuple(getattr(self, field) for field in REQUIRED_SIGNATURE_FIELDS)


@dataclass(frozen=True)
class TimingStats:
    """Robust summary of complete-step timing samples."""

    count: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    std_ms: float


def load_json(path: str | Path) -> Any:
    """Load a JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContractError(f"JSONL row {line_number} is not an object")
        rows.append(value)
    return rows


def compare_manifests(
    reference: CellManifest, variant: CellManifest
) -> list[str]:
    """Return invariant mismatches between a reference and variant cell."""
    mismatches: list[str] = []
    for field, ref_value, variant_value in zip(
        REQUIRED_SIGNATURE_FIELDS,
        reference.comparison_signature(),
        variant.comparison_signature(),
    ):
        if ref_value != variant_value:
            mismatches.append(
                f"{field}: reference={ref_value!r}, variant={variant_value!r}"
            )
    return mismatches


def summarize_timing(
    rows: Iterable[dict[str, Any]],
    expected_count: int,
    *,
    expected_batch_size: int | None = None,
    expected_active_slots: int | None = None,
    expected_mode: str = "timing",
) -> TimingStats:
    """Validate and summarize full-model timing rows."""
    rows = list(rows)
    if len(rows) != expected_count:
        raise ContractError(
            f"expected {expected_count} timing rows, received {len(rows)}"
        )
    samples: list[float] = []
    for index, row in enumerate(rows):
        if row.get("success") is not True:
            raise ContractError(f"timing row {index} is not successful")
        if row.get("full_model") is not True:
            raise ContractError(f"timing row {index} is not full_model")
        if row.get("mode", expected_mode) != expected_mode:
            raise ContractError(f"timing row {index} has a different mode")
        if int(row.get("denoise_steps", 0)) != 1:
            raise ContractError(f"timing row {index} is not one denoise step")
        if row.get("profiler_in_timing") is True:
            raise ContractError(f"timing row {index} contains profiler timing")
        if expected_batch_size is not None and int(
            row.get("actual_batch_size", 0)
        ) != expected_batch_size:
            raise ContractError(f"timing row {index} does not prove actual batch")
        if expected_active_slots is not None and int(
            row.get("active_lora_slots", 0)
        ) < expected_active_slots:
            raise ContractError(f"timing row {index} does not prove active slots")
        value = float(row.get("step_ms", 0.0))
        if value <= 0:
            raise ContractError(f"timing row {index} has invalid step_ms")
        samples.append(value)
    ordered = sorted(samples)
    return TimingStats(
        count=len(samples),
        median_ms=float(statistics.median(samples)),
        p10_ms=float(ordered[max(0, int(len(samples) * 0.1))]),
        p90_ms=float(ordered[min(len(samples) - 1, int(len(samples) * 0.9))]),
        std_ms=float(statistics.pstdev(samples)),
    )


def validate_batch_and_slots(
    actual_batch_sizes: Iterable[int],
    active_slot_counts: Iterable[int],
    expected_batch_size: int,
) -> None:
    """Require true batching and the requested number of active LoRA slots."""
    batches = [int(value) for value in actual_batch_sizes]
    slots = [int(value) for value in active_slot_counts]
    if not batches or any(value != expected_batch_size for value in batches):
        raise ContractError("actual_batch_size does not prove true co-batching")
    if not slots or any(value < 2 for value in slots):
        raise ContractError("K>=2 active LoRA slots were not proven")


def require_runtime_coverage(
    rows: Iterable[dict[str, Any]], variant: str
) -> None:
    """Require one runtime event proving Base and LoRA share a dispatch."""
    if variant == "reference_both":
        return
    covered = False
    for row in rows:
        if row.get("event") != "mixfusion_runtime_dispatch":
            continue
        if row.get("base_executed") is True and row.get("lora_executed") is True:
            covered = True
            break
    if not covered:
        raise ContractError(
            "variant has no dispatch proving Base and LoRA share MixFusion runtime"
        )


def evaluate_speedup(
    reference: TimingStats,
    variant: TimingStats,
) -> dict[str, Any]:
    """Evaluate speedup using the strict noise threshold."""
    delta_ms = reference.median_ms - variant.median_ms
    threshold_ms = max(
        0.05 * reference.median_ms,
        2.0 * max(reference.std_ms, variant.std_ms),
    )
    return {
        "reference_median_ms": reference.median_ms,
        "variant_median_ms": variant.median_ms,
        "delta_ms": delta_ms,
        "speedup": reference.median_ms / variant.median_ms,
        "noise_threshold_ms": threshold_ms,
        "above_noise": delta_ms > threshold_ms,
    }


def require_correctness_gate(correctness: dict[str, Any]) -> None:
    """Reject a cell unless every required correctness gate is explicit."""
    required = {
        "base_only": True,
        "lora_only": True,
        "base_plus_lora": True,
        "mixed_adapter": True,
        "mixed_rank": True,
        "attention_output_vs_reference": True,
    }
    for field, expected in required.items():
        value = correctness.get(field)
        if field == "attention_output_vs_reference":
            passed = isinstance(value, dict) and value.get("pass") is True
        elif field == "mixed_rank":
            passed = bool(value)
        else:
            passed = value in {True, "pass", "passed"}
        if expected and not passed:
            raise ContractError(f"correctness gate failed: {field}")
    if not isinstance(correctness.get("seed"), int):
        raise ContractError("correctness seed is missing")
    attention = correctness["attention_output_vs_reference"]
    if not attention.get("numeric_digest"):
        raise ContractError("attention numeric digest is missing")


__all__ = [
    "CONTAINER_NAME",
    "CellManifest",
    "ContractError",
    "TimingStats",
    "compare_manifests",
    "evaluate_speedup",
    "load_json",
    "load_jsonl",
    "require_correctness_gate",
    "summarize_timing",
    "validate_batch_and_slots",
]
