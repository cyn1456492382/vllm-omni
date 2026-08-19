# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect
import json
import os
from pprint import pformat
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import PIL.Image
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.lora.dit_lora_overlap import runtime as lora_runtime, set_overlap_granularity
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.utils import create_transformers_model
from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    ZImageTransformer2DModel,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
_DEBUG_ZIMAGE_INPUTS = os.environ.get("ZIMAGE_DEBUG_INPUTS", "0").lower() in {"1", "true", "yes", "on"}


def _unwrap_single_item_list(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _pad_embeddings_to_length(
    embeds: torch.Tensor | None,
    mask: torch.Tensor | None,
    target_len: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if embeds is None:
        return None, mask
    if embeds.ndim == 2:
        embeds = embeds.unsqueeze(0)
    if mask is not None and mask.ndim == 1:
        mask = mask.unsqueeze(0)
    seq_len = int(embeds.shape[1])
    if seq_len == target_len:
        return embeds, mask
    if seq_len > target_len:
        embeds = embeds[:, :target_len]
        if mask is not None:
            mask = mask[:, :target_len]
        return embeds, mask
    pad_len = target_len - seq_len
    embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad_len), value=0.0)
    if mask is not None:
        mask = torch.nn.functional.pad(mask, (0, pad_len), value=False)
    return embeds, mask


def _batch_request_embeddings(
    states: list[Any],
    embeds_attr: str,
    mask_attr: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pad and concatenate one embedding tensor from each request state."""
    embeddings = [getattr(state, embeds_attr, None) for state in states]
    masks = [getattr(state, mask_attr, None) for state in states]
    if not embeddings or any(embedding is None for embedding in embeddings):
        return None, None

    target_len = max(int(embedding.shape[1]) for embedding in embeddings)
    padded = [
        _pad_embeddings_to_length(embedding, mask, target_len)
        for embedding, mask in zip(embeddings, masks)
    ]
    return (
        torch.cat([embedding for embedding, _ in padded], dim=0),
        torch.cat([mask for _, mask in padded if mask is not None], dim=0)
        if all(mask is not None for _, mask in padded)
        else None,
    )


def _dbg_type(value: Any) -> str:
    if value is None:
        return "None"
    return f"{type(value).__name__}"


def _dbg_value(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, list):
        sample = [_dbg_type(v) for v in value[:3]]
        return f"list(len={len(value)}, sample_types={sample})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())[:8]})"
    return type(value).__name__


def _zimage_debug(label: str, **items: Any) -> None:
    if not _DEBUG_ZIMAGE_INPUTS:
        return
    payload = {k: _dbg_value(v) for k, v in items.items()}
    logger.info("[ZImageDebug] %s %s", label, pformat(payload))


def get_post_process_func(
    od_config: OmniDiffusionConfig,
):
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers
def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class ZImagePipeline(nn.Module, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery):
    supports_request_batch = False
    supports_step_execution = True

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="text_encoder",
                revision=od_config.revision,
                prefix="text_encoder.",
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="vae",
                revision=od_config.revision,
                prefix="vae.",
            ),
        ]
        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        prefetch_subfolders(
            model,
            ["scheduler", "text_encoder", "vae", "tokenizer"],
            local_files_only=local_files_only,
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        text_encoder_config = AutoConfig.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )
        self.text_encoder = create_transformers_model(
            AutoModelForCausalLM,
            od_config,
            hf_config=text_encoder_config,
        ).to(self._execution_device)
        if text_encoder_config.tie_word_embeddings:
            self.text_encoder.lm_head.weight = self.text_encoder.get_input_embeddings().weight

        vae_config = DistributedAutoencoderKL.load_config(model, subfolder="vae", local_files_only=local_files_only)
        self.vae = DistributedAutoencoderKL.from_config(vae_config).to(self._execution_device)
        self.transformer = ZImageTransformer2DModel(quant_config=od_config.quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        # Note: Context parallelism is applied centrally in registry.initialize_model()
        # following diffusers' pattern of enable_parallelism() at model loading time

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        _zimage_debug(
            "encode_prompt",
            prompt=prompt,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds, prompt_embeds_mask = self._encode_prompt(
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ["" for _ in prompt]
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            assert len(prompt) == len(negative_prompt)
            negative_prompt_embeds, negative_prompt_embeds_mask = self._encode_prompt(
                prompt=negative_prompt,
                device=device,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None
        return prompt_embeds, negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask

    def _encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        _zimage_debug(
            "_encode_prompt_entry",
            prompt=prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        if prompt_embeds is not None:
            if not isinstance(prompt_embeds, torch.Tensor):
                raise TypeError(f"prompt_embeds must be a Tensor, got {type(prompt_embeds).__name__}")
            if prompt_embeds.ndim == 2:
                prompt_embeds = prompt_embeds.unsqueeze(0)
            prompt_masks = torch.ones(
                (prompt_embeds.shape[0], prompt_embeds.shape[1]),
                dtype=torch.bool,
                device=prompt_embeds.device,
            )
            return prompt_embeds, prompt_masks

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        return prompt_embeds, prompt_masks

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        scheduler=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if image is not None:
            if latents is not None:
                return latents.to(device=device, dtype=dtype)

            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != num_channels_latents:
                if isinstance(generator, list):
                    image_latents = [
                        retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                        for i in range(image.shape[0])
                    ]
                    image_latents = torch.cat(image_latents, dim=0)
                else:
                    image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

                image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            else:
                image_latents = image

            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )

            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            scheduler = scheduler or self.scheduler
            latents = scheduler.scale_noise(image_latents, timestep, noise)
            return latents

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    def get_timesteps(self, scheduler, num_inference_steps, strength, device):
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = scheduler.timesteps[t_start * scheduler.order :]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(t_start * scheduler.order)
        return timesteps, num_inference_steps - t_start

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_encode(self, state: "StepRequestState", **kwargs: Any) -> "StepRequestState":
        del kwargs
        prompt = state.prompt if isinstance(state.prompt, str) else (state.prompt.get("prompt") or "") if state.prompt is not None else ""
        negative_prompt = None
        if isinstance(state.prompt, dict) and state.prompt.get("negative_prompt") is not None:
            negative_prompt = state.prompt.get("negative_prompt") or ""
        device = self._execution_device
        sampling = state.sampling
        scheduler = copy.deepcopy(self.scheduler)
        _zimage_debug(
            "prepare_encode_entry",
            state_prompt=state.prompt,
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            sampling=sampling,
        )
        self._guidance_scale = float(sampling.guidance_scale or 0.0)
        self._interrupt = False
        self._cfg_normalization = sampling.cfg_normalize
        self._cfg_truncation = sampling.extra_args.get("cfg_truncation", 1.0)
        prompt_embeds, negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask = self.encode_prompt(
            prompt=[prompt],
            negative_prompt=None if negative_prompt is None else [negative_prompt],
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            device=device,
            max_sequence_length=sampling.max_sequence_length or 512,
        )
        height = sampling.height or 1024
        width = sampling.width or 1024
        num_inference_steps = sampling.num_inference_steps or 50
        generator = sampling.generator
        sigmas = sampling.sigmas
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        image = None
        multi_modal_data = state.prompt.get("multi_modal_data", {}) if isinstance(state.prompt, dict) else {}
        raw_image = multi_modal_data.get("image") if isinstance(multi_modal_data, dict) else None
        _zimage_debug(
            "prepare_encode_image",
            state_prompt=state.prompt,
            multi_modal_data=multi_modal_data,
            raw_image=raw_image,
        )
        if isinstance(raw_image, list):
            raw_image = raw_image[0] if raw_image else None
        if raw_image is not None:
            image = PIL.Image.open(raw_image) if isinstance(raw_image, str) else raw_image
        num_channels_latents = self.transformer.in_channels
        if image is not None and not isinstance(image, torch.Tensor):
            image = self.image_processor.preprocess(image, height, width).to(dtype=torch.float32, device=device)
        if image is not None:
            mu = calculate_shift(
                (height // self.vae_scale_factor // 2) * (width // self.vae_scale_factor // 2),
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )
            scheduler.sigma_min = 0.0
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
            timesteps, num_inference_steps = self.get_timesteps(scheduler, num_inference_steps, sampling.strength or 0.6, device)
            latent_timestep = timesteps[:1].repeat(num_images_per_prompt)
            latents = self.prepare_latents(num_images_per_prompt, num_channels_latents, height, width, prompt_embeds[0].dtype, device, generator, sampling.latents, image, latent_timestep, scheduler)
        else:
            mu = calculate_shift((height // self.vae_scale_factor // 2) * (width // self.vae_scale_factor // 2), scheduler.config.get("base_image_seq_len", 256), scheduler.config.get("max_image_seq_len", 4096), scheduler.config.get("base_shift", 0.5), scheduler.config.get("max_shift", 1.15))
            scheduler.sigma_min = 0.0
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
            latents = self.prepare_latents(num_images_per_prompt, num_channels_latents, height, width, torch.float32, device, generator, sampling.latents)
        state.extra.update({"image": image, "num_images_per_prompt": num_images_per_prompt, "timesteps": timesteps})
        state.prompt_embeds = prompt_embeds
        state.prompt_embeds_mask = prompt_embeds_mask
        state.negative_prompt_embeds = negative_prompt_embeds
        state.negative_prompt_embeds_mask = negative_prompt_embeds_mask
        state.latents = latents
        state.scheduler = scheduler
        state.timesteps = timesteps
        state.step_index = 0
        return state

    def denoise_step(self, input_batch: "InputBatch", **kwargs: Any) -> torch.Tensor | None:
        del kwargs
        req_states = getattr(input_batch, "states", None) or []
        if not req_states:
            return None
        state = req_states[0]
        prompt_embeds, prompt_embeds_mask = _batch_request_embeddings(
            req_states,
            "prompt_embeds",
            "prompt_embeds_mask",
        )
        if prompt_embeds is None:
            return None
        negative_prompt_embeds, negative_prompt_embeds_mask = _batch_request_embeddings(
            req_states,
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
        )
        t = input_batch.timesteps
        latents = input_batch.latents.to(self.od_config.dtype)
        runtime = lora_runtime()
        runtime.request_id = state.request_id
        runtime.batch_request_ids = tuple(
            item.request_id
            for item in req_states
            for _ in range(int(item.latents.shape[0]))
        )
        runtime.step_index = int(state.step_index)
        apply_cfg = (
            self.do_classifier_free_guidance
            and self.guidance_scale > 0
            and negative_prompt_embeds is not None
        )
        runtime.batch_repeat = 2 if apply_cfg else 1
        _zimage_debug(
            "denoise_step_entry",
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            timesteps=t,
            latents=latents,
            apply_cfg=apply_cfg,
        )
        if apply_cfg:
            prompt_embeds, prompt_embeds_mask = _pad_embeddings_to_length(
                prompt_embeds,
                prompt_embeds_mask,
                max(int(prompt_embeds.shape[1]), int(negative_prompt_embeds.shape[1])),
            )
            negative_prompt_embeds, negative_prompt_embeds_mask = _pad_embeddings_to_length(
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                int(prompt_embeds.shape[1]),
            )
        latent_model_input = latents.repeat(2, 1, 1, 1) if apply_cfg else latents
        prompt_embeds_model_input = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0) if apply_cfg else prompt_embeds
        timestep_model_input = t.repeat(2) if apply_cfg else t
        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))
        model_out_list = self.transformer(latent_model_input_list, timestep_model_input, prompt_embeds_model_input)[0]
        if apply_cfg:
            pos_out = model_out_list[: latents.shape[0]]
            neg_out = model_out_list[latents.shape[0] :]
            noise_pred = torch.stack([p.float() + self.guidance_scale * (p.float() - n.float()) for p, n in zip(pos_out, neg_out)], dim=0)
        else:
            noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)
        return -noise_pred.squeeze(2)

    def step_scheduler(self, state: "StepRequestState", noise_pred: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        t = state.current_timestep
        scheduler = state.scheduler or self.scheduler
        state.latents = scheduler.step(noise_pred.to(torch.float32), t, state.latents, return_dict=False)[0]
        state.step_index += 1

    def post_decode(self, state: "StepRequestState", **kwargs: Any) -> DiffusionOutput:
        del kwargs
        latents = state.latents
        if latents is None:
            raise ValueError("post_decode requires latents")
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return DiffusionOutput(output=image, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None)

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        # TODO: In online mode, sometimes it receives [{"negative_prompt": None}, {...}], so cannot use .get("...", "")
        # TODO: May be some data formatting operations on the API side. Hack for now.
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts]

        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        prompt_embeds = None
        negative_prompt_embeds = None

        image = None
        if req.prompts:
            if len(req.prompts) > 1:
                logger.warning(
                    "This model only supports a single prompt for img2img, not a batched request. "
                    "Taking only the first image for now."
                )
            first_prompt = req.prompts[0]
            if not isinstance(first_prompt, str):
                raw_image = first_prompt.get("multi_modal_data", {}).get("image")
                if raw_image is not None:
                    if isinstance(raw_image, list):
                        raw_image = raw_image[0] if raw_image else None
                    if raw_image is not None:
                        image = PIL.Image.open(raw_image) if isinstance(raw_image, str) else raw_image

        explicit_strength = req.sampling_params.strength is not None
        strength = req.sampling_params.strength if explicit_strength else 0.6
        if explicit_strength and image is None:
            logger.warning(
                "strength parameter (%.2f) is only applicable for image-to-image (I2I) generation. "
                "It will be ignored for text-to-image (T2I) generation.",
                strength,
            )
            strength = None
        if image is not None and strength is not None and (strength < 0 or strength > 1):
            raise ValueError(f"The value of strength should be in [0.0, 1.0] but is {strength}")

        height = req.sampling_params.height or 1024
        width = req.sampling_params.width or 1024
        num_inference_steps = req.sampling_params.num_inference_steps or 50
        generator = req.sampling_params.generator
        sigmas = req.sampling_params.sigmas
        max_sequence_length = req.sampling_params.max_sequence_length or 512
        guidance_scale = req.sampling_params.guidance_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt if req.sampling_params.num_outputs_per_prompt > 0 else 1
        )
        latents = req.sampling_params.latents

        cfg_normalization = req.sampling_params.cfg_normalize
        cfg_truncation = req.sampling_params.extra_args.get("cfg_truncation", 1.0)
        joint_attention_kwargs: dict[str, Any] | None = None
        callback_on_step_end: Callable[[int, int, dict], None] | None = None
        callback_on_step_end_tensor_inputs = ["latents"]
        output_type = req.sampling_params.output_type or "pil"

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        device = self._execution_device

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        # 2. Define call parameters
        batch_size = len(prompt)

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels

        # img2img mode: prepare latents from input image
        if image is not None:
            # Prepare image for VAE encoding using image_processor
            if not isinstance(image, torch.Tensor):
                init_image = self.image_processor.preprocess(image, height, width)
                image = init_image.to(dtype=torch.float32, device=device)

            # Initialize scheduler kwargs for img2img
            mu = calculate_shift(
                (height // self.vae_scale_factor // 2) * (width // self.vae_scale_factor // 2),
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.sigma_min = 0.0
            scheduler_kwargs = {"mu": mu}

            # First initialize timesteps in scheduler
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )

            # Then adjust timesteps based on strength
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

            if num_inference_steps < 1:
                raise ValueError(
                    f"After adjusting the num_inference_steps by strength parameter: "
                    f"{strength}, the number of pipeline steps is {num_inference_steps} "
                    f"which is < 1 and not appropriate for this pipeline."
                )
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds[0].dtype,
                device,
                generator,
                latents,
                image,
                latent_timestep,
            )
        else:
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                torch.float32,
                device,
                generator,
                latents,
            )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)
            if self.do_classifier_free_guidance and negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
                if negative_prompt_embeds_mask is not None:
                    negative_prompt_embeds_mask = negative_prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            target_embed_len = max(int(prompt_embeds.shape[1]), int(negative_prompt_embeds.shape[1]))
            prompt_embeds, prompt_embeds_mask = _pad_embeddings_to_length(prompt_embeds, prompt_embeds_mask, target_embed_len)
            negative_prompt_embeds, negative_prompt_embeds_mask = _pad_embeddings_to_length(
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
                target_embed_len,
            )

        actual_batch_size = batch_size * num_images_per_prompt

        # 5. Prepare timesteps
        if image is None:
            image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.sigma_min = 0.0
            scheduler_kwargs = {"mu": mu}

            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Precompute normalized timesteps once to avoid per-step GPU->CPU sync (.item() causes cudaStreamSynchronize)
        if isinstance(timesteps, torch.Tensor):
            timesteps_tensor = timesteps.to(device=device, dtype=torch.float32)
        else:
            timesteps_tensor = torch.as_tensor(timesteps, device=device, dtype=torch.float32)
        norm_timesteps = (1000 - timesteps_tensor) / 1000
        t_norm_list = norm_timesteps.cpu().tolist()
        if not isinstance(t_norm_list, list):
            t_norm_list = [t_norm_list]

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000
            # Normalized time for time-aware config (0 at start, 1 at end);
            # use precomputed to avoid .item() sync per step
            t_norm = t_norm_list[i]

            # Handle cfg truncation
            current_guidance_scale = self.guidance_scale
            if (
                self.do_classifier_free_guidance
                and self._cfg_truncation is not None
                and float(self._cfg_truncation) <= 1
            ):
                if t_norm > self._cfg_truncation:
                    current_guidance_scale = 0.0

            # Run CFG only if configured AND scale is non-zero
            apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0
            latents_typed = latents.to(self.od_config.dtype)

            if apply_cfg:
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents_typed
                prompt_embeds_model_input = prompt_embeds
                timestep_model_input = timestep

            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            model_out_list = self.transformer(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
            )[0]

            if apply_cfg:
                # Perform CFG
                pos_out = model_out_list[:actual_batch_size]
                neg_out = model_out_list[actual_batch_size:]

                noise_pred = []
                for j in range(actual_batch_size):
                    pos = pos_out[j].float()
                    neg = neg_out[j].float()

                    pred = pos + current_guidance_scale * (pos - neg)

                    # Renormalization (torch.where avoids GPU->CPU sync from Python if/scalar comparison)
                    if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                        ori_pos_norm = torch.linalg.vector_norm(pos)
                        new_pos_norm = torch.linalg.vector_norm(pred)
                        max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                        scale = torch.where(
                            new_pos_norm > max_new_norm,
                            (max_new_norm / new_pos_norm.clamp(min=1e-12)).to(pred.dtype),
                            pred.new_tensor(1.0),
                        )
                        pred = pred * scale

                    noise_pred.append(pred)

                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
            assert latents.dtype == torch.float32

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            # image = self.image_processor.postprocess(image, output_type=output_type)

        stage_durations = self.stage_durations if hasattr(self, "stage_durations") else None
        return DiffusionOutput(output=image, stage_durations=stage_durations)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights)
        # Record components loaded by diffusers submodules to satisfy strict checks.
        loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        # downstream pipelines (e.g. MingImagePipeline) may set ``self.text_encoder = None`` when they
        # bring their own conditioning path.
        if self.text_encoder is not None:
            loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
        return loaded_weights
