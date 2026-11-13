"""Pinned MixFusion ``split_sample`` source used by the Omni adapter.

Source: MixFusion e9aa899a1bd0885c100cc9b16011d987d3be4895,
``sduss/model_executor/modules/utils.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def split_sample(
    samples: dict[int, torch.Tensor],
    patch_size: int,
    input_indices: dict[str, list[str]],
):
    latent_offset = [0]
    resolution_offset = [0]
    padding_idx = []
    new_sample = []
    patch_map = []
    indices = []
    for resolution, res_sample in samples.items():
        resolution = int(resolution)
        patch_on_height = resolution // patch_size
        patch_on_width = resolution // patch_size
        latent_patch_size = int(patch_size // 8)
        if res_sample is None or res_sample.shape[0] == 0:
            continue
        index = 0
        for sample in res_sample:
            latent_offset.append(latent_offset[-1] + patch_on_width**2)
            sample = F.pad(sample, (1, 1, 1, 1), "constant", 0).unsqueeze(0)
            for height_index in range(patch_on_height):
                for width_index in range(patch_on_width):
                    paddings = torch.empty(
                        4, device=sample.device, dtype=torch.int32
                    )
                    if patch_on_height == 1:
                        paddings.fill_(-1)
                        new_sample.append(
                            sample[
                                :,
                                :,
                                height_index * latent_patch_size :
                                (height_index + 1) * latent_patch_size + 2,
                                width_index * latent_patch_size :
                                (width_index + 1) * latent_patch_size + 2,
                            ]
                        )
                        patch_map.append(len(latent_offset) - 1)
                        padding_idx.append(paddings)
                        indices.append(
                            input_indices[str(resolution)][index]
                            + f"-{height_index}-{width_index}"
                        )
                        continue
                    if width_index == 0:
                        paddings[1] = -1
                        paddings[3] = len(new_sample) + 1
                    elif width_index == patch_on_width - 1:
                        paddings[1] = len(new_sample) - 1
                        paddings[3] = -1
                    else:
                        paddings[1] = len(new_sample) - 1
                        paddings[3] = len(new_sample) + 1
                    if height_index == 0:
                        paddings[0] = -1
                        paddings[2] = len(new_sample) + patch_on_height
                    elif height_index == patch_on_height - 1:
                        paddings[0] = len(new_sample) - patch_on_height
                        paddings[2] = -1
                    else:
                        paddings[0] = len(new_sample) - patch_on_height
                        paddings[2] = len(new_sample) + patch_on_height
                    new_sample.append(
                        sample[
                            :,
                            :,
                            height_index * latent_patch_size :
                            (height_index + 1) * latent_patch_size + 2,
                            width_index * latent_patch_size :
                            (width_index + 1) * latent_patch_size + 2,
                        ]
                    )
                    patch_map.append(len(latent_offset) - 1)
                    padding_idx.append(paddings)
                    indices.append(
                        input_indices[str(resolution)][index]
                        + f"-{height_index}-{width_index}"
                    )
            index += 1
        resolution_offset.append(len(latent_offset) - 1)
    if not new_sample:
        raise ValueError("MixFusion split_sample produced no patches")
    return (
        indices,
        {
            "cuda": torch.cat(padding_idx, dim=0).cuda(),
            "cpu": padding_idx,
        },
        {
            "cuda": torch.tensor(latent_offset, device="cuda", dtype=torch.int32),
            "cpu": latent_offset,
        },
        {
            "cuda": torch.tensor(
                resolution_offset, device="cuda", dtype=torch.int32
            ),
            "cpu": resolution_offset,
        },
        torch.cat(new_sample, dim=0),
        {
            "cuda": torch.tensor(patch_map, device="cuda", dtype=torch.int32),
            "cpu": patch_map,
        },
    )


__all__ = ["split_sample"]
