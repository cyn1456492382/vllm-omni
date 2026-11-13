# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass

# for now, it suffices to use vLLM's implementation directly
# as this is a user-facing variable, defined here to so that user can directly import LoRARequest from vllm_omni
from vllm.lora.request import LoRARequest


@dataclass(frozen=True)
class LoRACompositionItem:
    request: LoRARequest
    scale: float = 1.0

    def __post_init__(self) -> None:
        scale = float(self.scale)
        if not math.isfinite(scale):
            raise ValueError(f"LoRA scale must be finite, got {self.scale!r}")
        if scale == 0.0:
            raise ValueError("LoRA composition entries cannot have zero scale")
        object.__setattr__(self, "scale", scale)

    @property
    def adapter_id(self) -> int:
        return self.request.lora_int_id


@dataclass(frozen=True)
class LoRAComposition:
    items: tuple[LoRACompositionItem, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("LoRA composition must contain at least one adapter")
        adapter_ids = [item.adapter_id for item in self.items]
        if len(adapter_ids) != len(set(adapter_ids)):
            raise ValueError("LoRA composition cannot repeat an adapter id")

    @classmethod
    def from_items(
        cls,
        items: list[LoRACompositionItem] | tuple[LoRACompositionItem, ...],
    ) -> "LoRAComposition":
        return cls(tuple(items))

    @property
    def adapter_ids(self) -> tuple[int, ...]:
        return tuple(item.adapter_id for item in self.items)


__all__ = ["LoRAComposition", "LoRACompositionItem", "LoRARequest"]
