from __future__ import annotations

import unittest

from tools.diffusion.mixfusion_contract import (
    CellManifest,
    ContractError,
    require_correctness_gate,
    require_runtime_coverage,
    summarize_timing,
)


_VALID_MANIFEST = {
    "cell": "R0-512",
    "variant": "reference_both",
    "container": "vllm-adapter-a800-tp2-cu124-20261027",
    "code_sha": "abc123",
    "model": "FLUX/ImageGem",
    "resolution": "512",
    "batch_size": 4,
    "actual_batch_size": 4,
    "active_lora_slots": 4,
    "adapter_manifest": "adapters.json",
    "adapter_manifest_digest": "sha256:abc",
    "adapter_loaded_before_warmup": True,
    "rank_mapping": [2, 8, 32],
    "seed": 7,
    "precision": "bf16",
    "warmup": 2,
    "measured": 5,
    "denoise_steps": 1,
    "timing_window": "denoise_forward_only",
    "attention_semantics": "reference_unchanged",
    "request_semantics": "real_flux_imagegem",
    "full_model": True,
    "mode": "timing",
    "profiler_in_timing": False,
}


class MixFusionContractTest(unittest.TestCase):
    def test_manifest_requires_true_batch_and_attention_contract(self) -> None:
        manifest = CellManifest.from_mapping(_VALID_MANIFEST)
        self.assertEqual(manifest.actual_batch_size, 4)
        invalid = dict(_VALID_MANIFEST, actual_batch_size=1)
        with self.assertRaisesRegex(ContractError, "actual_batch_size"):
            CellManifest.from_mapping(invalid)

    def test_timing_requires_batch_and_slots_in_each_row(self) -> None:
        rows = [
            {
                "success": True,
                "full_model": True,
                "mode": "timing",
                "denoise_steps": 1,
                "profiler_in_timing": False,
                "actual_batch_size": 4,
                "active_lora_slots": 4,
                "step_ms": 10.0,
            }
        ] * 5
        summary = summarize_timing(
            rows,
            5,
            expected_batch_size=4,
            expected_active_slots=4,
        )
        self.assertEqual(summary.count, 5)
        with self.assertRaisesRegex(ContractError, "active slots"):
            summarize_timing(
                [dict(rows[0], active_lora_slots=1)] * 5,
                5,
                expected_batch_size=4,
                expected_active_slots=4,
            )

    def test_variant_requires_one_shared_runtime_dispatch(self) -> None:
        with self.assertRaises(ContractError):
            require_runtime_coverage([], "mixfusion_both")
        require_runtime_coverage(
            [
                {
                    "event": "mixfusion_runtime_dispatch",
                    "base_executed": True,
                    "lora_executed": True,
                }
            ],
            "mixfusion_both",
        )

    def test_correctness_requires_numeric_attention_digest(self) -> None:
        correctness = {
            "seed": 7,
            "base_only": True,
            "lora_only": True,
            "base_plus_lora": True,
            "mixed_adapter": True,
            "mixed_rank": True,
            "attention_output_vs_reference": {
                "pass": True,
                "numeric_digest": {"sha256_float32": "abc"},
            },
        }
        require_correctness_gate(correctness)
        with self.assertRaisesRegex(ContractError, "numeric digest"):
            require_correctness_gate(
                dict(
                    correctness,
                    attention_output_vs_reference={"pass": True},
                )
            )
