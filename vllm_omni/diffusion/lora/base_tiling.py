from __future__ import annotations

import os


_INSTALLED = False


def _token_tile_removed_error() -> ValueError:
    return ValueError(
        "token-tile path removed; only System B MixFusion "
        "(mixfusion_both / system_b_mixfusion_patch) remains"
    )


def base_tile_size() -> int | None:
    """Fail closed: VLLM_OMNI_BASE_TILING no longer selects a token-tile path."""
    enabled = os.environ.get("VLLM_OMNI_BASE_TILING", "")
    if enabled.strip().lower() in {"1", "true", "yes", "on"}:
        raise _token_tile_removed_error()
    return None


def install_base_tiling_linear_forward() -> None:
    """No-op installer retained for call-site compatibility.

    Token-tile Base forwarding was removed. Enabling VLLM_OMNI_BASE_TILING
    fails closed via base_tile_size() / runner checks.
    """
    global _INSTALLED
    _INSTALLED = True


__all__ = ["base_tile_size", "install_base_tiling_linear_forward"]
