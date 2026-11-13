"""Phase-1 correctness gates: cycle composition remap, cancel, scale keys."""

from __future__ import annotations

import threading

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.lora.mixfusion_native_bridge import (
    adapter_scale_signature_key,
    restore_lora_composition,
    slice_lora_composition_for_cycle,
)
from vllm_omni.diffusion.lora.tetris_lora_runtime import (
    TetrisLoRAWorkerClient,
    cancel_tetris_lora,
)
from vllm_omni.diffusion.lora.tetris_lora_transport import TransportError


class _CompLayer(nn.Module):
    def set_batch_adapter_composition(self, composition, row_layout=None):
        self._diffusion_lora_batch_composition = tuple(composition)
        self._diffusion_lora_batch_row_layout = row_layout


def test_slice_and_restore_lora_composition_for_aba_cycle():
    root = nn.Sequential(_CompLayer())
    layer = root[0]
    layer.set_batch_adapter_composition(
        (((0, 1.0),), ((1, 0.5),), ((0, 1.0),)),
        row_layout=(10, 10, 10),
    )
    saved = slice_lora_composition_for_cycle(root, (0, 2))
    assert layer._diffusion_lora_batch_composition == (((0, 1.0),), ((0, 1.0),))
    assert layer._diffusion_lora_batch_row_layout == (10, 10)
    restore_lora_composition(saved)
    assert layer._diffusion_lora_batch_composition == (
        ((0, 1.0),),
        ((1, 0.5),),
        ((0, 1.0),),
    )
    assert layer._diffusion_lora_batch_row_layout == (10, 10, 10)


def test_adapter_scale_signature_avoids_milli_collision():
    a = adapter_scale_signature_key(1.0004)
    b = adapter_scale_signature_key(1.0006)
    # Old round(scale*1000) collided at 1000 for both; float bits must differ.
    assert a != b
    assert adapter_scale_signature_key(1.0) == adapter_scale_signature_key(1.0)


def test_cancel_tetris_releases_lock_and_blocks_finish():
    class _Client:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._lock.acquire()

    client = _Client()
    handle = {"active": True, "descriptor": object()}
    cancel_tetris_lora((client, handle))
    assert handle["active"] is False
    assert handle["cancelled"] is True
    assert not client._lock.locked()

    class _MultiSlotClient:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._inflight_cond = threading.Condition(self._lock)
            self._op_slot_free = [False, True]
            self._inflight_ops = {"op-1": 0}

    class _Desc:
        operation_id = "op-1"

    multi = _MultiSlotClient()
    multi_handle = {
        "active": True,
        "op_slot": 0,
        "descriptor": _Desc(),
    }
    cancel_tetris_lora((multi, multi_handle))
    assert multi._op_slot_free[0] is True
    assert "op-1" not in multi._inflight_ops

    mini = object.__new__(TetrisLoRAWorkerClient)
    cancelled = {"active": False, "cancelled": True, "descriptor": None}
    with pytest.raises(TransportError, match="cancelled"):
        TetrisLoRAWorkerClient.finish_delta(mini, cancelled, torch.zeros(1, 1))
