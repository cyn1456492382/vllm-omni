"""Fail-closed preflight for the controlled MixFusion environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from mixfusion_contract import CONTAINER_NAME


def main() -> int:
    """Check CUDA, vLLM-Omni import, and the configured container identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result: dict[str, Any] = {
        "container": args.container,
        "expected_container": CONTAINER_NAME,
        "python": sys.executable,
        "cuda_available": False,
        "cuda_device_count": 0,
        "vllm_omni_import": False,
        "pytest_import": False,
    }
    errors: list[str] = []
    if args.container != CONTAINER_NAME:
        errors.append("wrong container")
    try:
        import torch

        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count())
        if not result["cuda_available"] or result["cuda_device_count"] < 1:
            errors.append("CUDA device is not visible")
    except Exception as exc:  # pragma: no cover - environment dependent
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
        errors.append("torch import failed")
    for module_name, result_key in (
        ("vllm_omni", "vllm_omni_import"),
        ("pytest", "pytest_import"),
    ):
        available = importlib.util.find_spec(module_name) is not None
        result[result_key] = available
        if not available:
            errors.append(f"{module_name} import is unavailable")
    result["valid"] = not errors
    if errors:
        result["errors"] = errors
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
