from __future__ import annotations

import pytest

from vllm_omni.diffusion.lora.tetris_lora_runtime import (
    TransportError,
    _qkv_slice_index,
    _resolve_target_layout,
    _localize_x_mode,
)


@pytest.mark.parametrize("value", ["auto", "1", "0", "AUTO"])
def test_localize_x_mode_accepts_supported_values(monkeypatch, value):
    monkeypatch.setenv("TETRIS_LORA_LOCALIZE_X", value)
    assert _localize_x_mode() == value.lower()


def test_localize_x_mode_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("TETRIS_LORA_LOCALIZE_X", raising=False)
    assert _localize_x_mode() == "auto"


def test_localize_x_mode_rejects_implicit_fallback_values(monkeypatch):
    monkeypatch.setenv("TETRIS_LORA_LOCALIZE_X", "fallback")
    with pytest.raises(TransportError, match="one of auto, 1, or 0"):
        _localize_x_mode()


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("transformer_blocks.0.attn.to_q", 0),
        ("transformer_blocks.0.attn.to_k", 1),
        ("transformer_blocks.0.attn.to_v", 2),
        ("transformer_blocks.0.attn.to_qkv", None),
    ],
)
def test_qkv_slice_mapping_is_explicit(target, expected):
    assert _qkv_slice_index("transformer_blocks.0.attn.to_qkv", target) == expected


def test_qkv_layout_rejects_fused_and_split_mixture():
    weights = {
        (0, "transformer_blocks.0.attn.to_qkv"): object(),
        (1, "transformer_blocks.0.attn.to_q"): object(),
    }
    with pytest.raises(TransportError, match="mixed fused and split"):
        _resolve_target_layout(
            "transformer_blocks.0.attn.to_qkv", (0, 1), weights
        )


def test_qkv_layout_keeps_split_projection_order():
    weights = {
        (0, "transformer_blocks.0.attn.to_v"): object(),
        (1, "transformer_blocks.0.attn.to_q"): object(),
    }
    assert _resolve_target_layout(
        "transformer_blocks.0.attn.to_qkv", (0, 1), weights
    ) == (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_v",
    )


def test_qkv_slice_index_stays_full_layout_for_partial_targets():
    # Active targets may be K/V only, but packed indices remain 1/2.
    assert _qkv_slice_index(
        "transformer_blocks.0.attn.to_qkv",
        "transformer_blocks.0.attn.to_k",
    ) == 1
    assert _qkv_slice_index(
        "transformer_blocks.0.attn.to_qkv",
        "transformer_blocks.0.attn.to_v",
    ) == 2


def test_adapter_version_cache_avoids_repeat_sha256(tmp_path, monkeypatch):
    from vllm_omni.diffusion.lora import tetris_lora_runtime as runtime

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    weight = adapter_dir / "adapter_model.safetensors"
    weight.write_bytes(b"rank32-weight-bytes" * 4096)
    monkeypatch.setenv("TETRIS_LORA_SHA256_STATS", "1")
    runtime._ADAPTER_VERSION_CACHE.clear()
    runtime._SHA256_STATS.update({"calls": 0.0, "bytes": 0.0, "ms": 0.0})

    first = runtime.register_adapter_version(adapter_dir)
    mid_calls = runtime._SHA256_STATS["calls"]
    second = runtime.register_adapter_version(adapter_dir)
    assert first == second
    assert runtime._SHA256_STATS["calls"] == mid_calls == 1.0


def test_invalidate_adapter_version_clears_cache(tmp_path, monkeypatch):
    from vllm_omni.diffusion.lora import tetris_lora_runtime as runtime

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"v1")
    resolved = str(adapter_dir.resolve())
    other = tmp_path / "other"
    other.mkdir()
    other_resolved = str(other.resolve())

    class _Client:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    hit = _Client()
    miss = _Client()
    key_hit = (((0, resolved),), (), 0, 1)
    key_miss = (((1, other_resolved),), (), 0, 1)
    hashes: list[str] = []

    def _fake_hash(_path: object) -> str:
        token = f"hash-{len(hashes) + 1}"
        hashes.append(token)
        return token

    runtime._ADAPTER_VERSION_CACHE.clear()
    runtime._CLIENTS[key_hit] = hit
    runtime._CLIENTS[key_miss] = miss
    monkeypatch.setattr(runtime, "_sha256_file", _fake_hash)
    try:
        first = runtime.register_adapter_version(adapter_dir)
        assert first == "hash-1"
        assert runtime._ADAPTER_VERSION_CACHE[resolved] == "hash-1"
        runtime.invalidate_adapter_version(adapter_dir)
        assert resolved not in runtime._ADAPTER_VERSION_CACHE
        assert key_hit not in runtime._CLIENTS
        assert hit.closed
        assert runtime._CLIENTS[key_miss] is miss
        assert not miss.closed
        second = runtime.register_adapter_version(adapter_dir)
        assert second == "hash-2"
        assert hashes == ["hash-1", "hash-2"]
    finally:
        runtime._ADAPTER_VERSION_CACHE.pop(resolved, None)
        runtime._CLIENTS.pop(key_hit, None)
        runtime._CLIENTS.pop(key_miss, None)
