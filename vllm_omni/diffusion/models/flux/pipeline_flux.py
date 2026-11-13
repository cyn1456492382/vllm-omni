# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoConfig, CLIPTextModel, CLIPTokenizer, T5TokenizerFast
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.experiment_telemetry import emit_event, tensor_metadata
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.flux import FluxTransformer2DModel
from vllm_omni.diffusion.models.flux.flux_pipeline_mixin import FluxPipelineMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.t5_encoder import T5EncoderModel
from vllm_omni.diffusion.lora.base_tiling import install_base_tiling_linear_forward
from vllm_omni.diffusion.lora.mixfusion_native_bridge import (
    NativePatchSchedule,
    NativePatchWork,
    adapter_scale_signature_key,
    restore_lora_composition,
    schedule_patch_work,
    slice_lora_composition_for_cycle,
    use_native_patch_schedule,
)
from vllm_omni.diffusion.lora.mixfusion_runtime import (
    uses_mixfusion_runtime,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

logger = logging.getLogger(__name__)


def get_flux_post_process_func(
    od_config: OmniDiffusionConfig,
):
    if od_config.output_type == "latent":
        return lambda x: x
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    if not os.path.exists(vae_config_path):
        raise FileNotFoundError(
            f"VAE config not found at {vae_config_path}. "
            "Please ensure the model path contains a valid VAE configuration."
        )
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


class FluxPipeline(
    nn.Module, FluxPipelineMixin, CFGParallelMixin, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery
):
    supports_request_batch = True
    supports_step_execution: ClassVar[bool] = True

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "text_encoder_2"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        install_base_tiling_linear_forward()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="text_encoder_2",
                revision=None,
                prefix="text_encoder_2.",
                fall_back_to_pt=True,
            ),
        ]

        self.device = get_local_device()
        model = od_config.model
        # Check if model is a local path
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        ).to(self.device)
        t5_config = AutoConfig.from_pretrained(model, subfolder="text_encoder_2", local_files_only=local_files_only)
        self.text_encoder_2 = T5EncoderModel(t5_config, prefix="text_encoder_2").to(self.device)
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self.device
        )

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, FluxTransformer2DModel)
        self.transformer = FluxTransformer2DModel(
            **transformer_kwargs, od_config=od_config, quant_config=od_config.quantization_config
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only
        )

        self.stage = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed.
        # This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        # self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. "
                "Make sure to generate `pooled_prompt_embeds` from the same text encoder "
                "that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. "
                "Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder "
                "that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        dtype: torch.dtype | None = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        emit_event(
            "dit_events",
            "flux_tokenizer_executed",
            tokenizer="T5TokenizerFast",
            stream="text_encoder_2",
            input_ids=tensor_metadata(text_input_ids, role="t5_input_ids"),
            max_sequence_length=int(max_sequence_length),
        )
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(self.device))[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        emit_event(
            "dit_events",
            "flux_tokenizer_executed",
            tokenizer="CLIPTokenizer",
            stream="text_encoder",
            input_ids=tensor_metadata(text_input_ids, role="clip_input_ids"),
            max_sequence_length=int(self.tokenizer_max_length),
        )
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str],
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        max_sequence_length: int = 512,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=self.device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

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
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        if batch_size < 1 or num_channels_latents < 1 or height < 1 or width < 1:
            raise ValueError(
                "FLUX latent geometry must be positive: "
                f"batch={batch_size}, channels={num_channels_latents}, "
                f"height={height}, width={width}"
            )
        emit_event(
            "dit_events",
            "flux_resolution_preprocess",
            requested_height=int(height * self.vae_scale_factor),
            requested_width=int(width * self.vae_scale_factor),
            latent_height=int(height),
            latent_width=int(width),
            vae_scale_factor=int(self.vae_scale_factor),
            patch_size=2,
            batch_size=int(batch_size),
        )

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        if tuple(latents.shape) != tuple(shape):
            raise RuntimeError(
                f"FLUX noise shape mismatch: expected {shape}, got {tuple(latents.shape)}"
            )
        if latents.device != torch.device(device):
            raise RuntimeError(
                f"FLUX noise device mismatch: expected {device}, got {latents.device}"
            )
        if isinstance(generator, list):
            generator_seeds = [
                int(item.initial_seed())
                for item in generator
                if hasattr(item, "initial_seed")
            ]
        elif generator is not None and hasattr(generator, "initial_seed"):
            generator_seeds = [int(generator.initial_seed())]
        else:
            generator_seeds = []
        emit_event(
            "dit_events",
            "flux_noise_initialized",
            noise=tensor_metadata(latents, role="flux_bchw_noise"),
            generator_kind=type(generator).__name__ if generator is not None else None,
            seeds=generator_seeds,
        )
        # Keep both representations observable.  ``latents`` is the random
        # convolution-layout noise tensor (the same BCHW geometry as a VAE
        # latent), while the transformer consumes the 2x2-packed sequence.
        unpacked_latents = latents
        packed_latents = self._pack_latents(
            unpacked_latents, batch_size, num_channels_latents, height, width
        )
        expected_packed_shape = (batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        if tuple(packed_latents.shape) != expected_packed_shape:
            raise RuntimeError(
                "FLUX packed latent shape mismatch: "
                f"expected {expected_packed_shape}, got {tuple(packed_latents.shape)}"
            )
        emit_event(
            "dit_events",
            "flux_latent_pack",
            image_height=int(height * self.vae_scale_factor),
            image_width=int(width * self.vae_scale_factor),
            vae_scale_factor=int(self.vae_scale_factor),
            patch_size=2,
            latent_source="randn_text_to_image",
            vae_encode_executed=False,
            unpacked_latents=tensor_metadata(
                unpacked_latents, role="flux_unpacked_latent_noise"
            ),
            packed_latents=tensor_metadata(
                packed_latents, role="flux_packed_latent_input"
            ),
        )
        latents = packed_latents

        latent_image_ids = self._prepare_latent_image_ids(height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = self.calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor,
        latents: torch.Tensor,
        latent_image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        negative_text_ids: torch.Tensor,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        guidance: torch.Tensor,
        true_cfg_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor:
        """Diffusion loop with optional image conditioning.

        MixFusion / mixed-LoRA scheduling is only supported through
        ``denoise_step`` (InputBatch path). This entry refuses MixFusion so
        callers cannot silently drop per-request composition.
        """
        if uses_mixfusion_runtime():
            raise RuntimeError(
                "FluxPipeline.diffuse does not support MixFusion / mixed-LoRA "
                "batches; use denoise_step with InputBatch"
            )
        self.scheduler.set_begin_index(0)
        self.transformer.do_true_cfg = do_true_cfg
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            # broadcast to batch dimension and place on same device/dtype as latents
            timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            positive_kwargs = {
                "hidden_states": latents,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "pooled_projections": pooled_prompt_embeds,
                "encoder_hidden_states": prompt_embeds,
                "txt_ids": text_ids,
                "img_ids": latent_image_ids,
                "joint_attention_kwargs": self.joint_attention_kwargs,
                "return_dict": False,
            }

            # Forward pass for negative prompt (CFG)
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "pooled_projections": negative_pooled_prompt_embeds,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "txt_ids": negative_text_ids,
                    "img_ids": latent_image_ids,
                    "joint_attention_kwargs": self.joint_attention_kwargs,
                    "return_dict": False,
                }
            else:
                negative_kwargs = None

            # Predict noise with automatic CFG parallel handling
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg,
                true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize,
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

        return latents

    def check_cfg_parallel_validity(self, true_cfg_scale: float, has_neg_prompt: bool):
        if get_classifier_free_guidance_world_size() == 1:
            return True

        if true_cfg_scale <= 1:
            logger.warning("CFG parallel is NOT working correctly when true_cfg_scale <= 1.")
            return False

        if not has_neg_prompt:
            logger.warning(
                "CFG parallel is NOT working correctly when there is no negative prompt or negative prompt embeddings."
            )
            return False
        return True

    def prepare_encode(
        self,
        state: "StepRequestState",
        **kwargs: Any,
    ) -> "StepRequestState":
        """Prepare one Flux request for stateful denoising."""
        del kwargs
        sampling = state.sampling
        prompt = state.prompt if isinstance(state.prompt, str) else (state.prompt or {}).get("prompt", "")
        negative_prompt = None
        if isinstance(state.prompt, dict):
            negative_prompt = state.prompt.get("negative_prompt")

        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        if int(height) < self.vae_scale_factor * 2 or int(width) < self.vae_scale_factor * 2:
            raise ValueError(
                "FLUX requested resolution is below the minimum aligned size: "
                f"height={height}, width={width}, alignment={self.vae_scale_factor * 2}"
            )
        emit_event(
            "dit_events",
            "flux_sampling_geometry",
            request_id=str(state.request_id),
            sampling_height=int(height),
            sampling_width=int(width),
            sampling_height_provided=sampling.height is not None,
            sampling_width_provided=sampling.width is not None,
            default_sample_size=int(self.default_sample_size),
            vae_scale_factor=int(self.vae_scale_factor),
        )
        num_inference_steps = sampling.num_inference_steps or 28
        guidance_scale = sampling.guidance_scale or 0.0
        true_cfg_scale = sampling.true_cfg_scale or 1.0
        num_images_per_prompt = sampling.num_outputs_per_prompt or 1
        max_sequence_length = sampling.max_sequence_length or 512
        prompt_embeds = None
        pooled_prompt_embeds = None
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = {}
        self._current_timestep = None
        self._interrupt = False

        has_neg_prompt = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        emit_event(
            "dit_events",
            "flux_component_executed",
            component="text_encoders",
            operations=["CLIPTextModel.forward", "T5EncoderModel.forward"],
            request_id=str(state.request_id),
            prompt_embedding_shape=[int(value) for value in prompt_embeds.shape],
            pooled_embedding_shape=[int(value) for value in pooled_prompt_embeds.shape],
        )
        negative_text_ids = None
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=None,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            sampling.generator,
            sampling.latents,
        )
        timesteps, _ = self.prepare_timesteps(
            num_inference_steps,
            sampling.sigmas,
            latents.shape[1],
        )
        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)

        guidance = None
        if self.transformer.guidance_embeds:
            guidance = torch.full([latents.shape[0]], guidance_scale, dtype=torch.float32, device=self.device)

        state.prompt_embeds = prompt_embeds
        state.negative_prompt_embeds = negative_prompt_embeds
        state.latents = latents
        state.timesteps = timesteps
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = do_true_cfg
        state.guidance = guidance
        state.sampling.cfg_normalize = False
        state.extra.update(
            {
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                "text_ids": text_ids,
                "negative_text_ids": negative_text_ids,
                "latent_image_ids": latent_image_ids,
                "true_cfg_scale": true_cfg_scale,
                "height": height,
                "width": width,
                "latent_seq_len": int(latents.shape[1]),
            }
        )
        emit_event(
            "dit_events",
            "flux_latent_sequence_geometry",
            request_id=str(state.request_id),
            image_height=int(height),
            image_width=int(width),
            latent_seq_len=int(latents.shape[1]),
        )
        return state

    def denoise_step(
        self,
        input_batch: "InputBatch",
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """Run one Flux transformer denoising step."""
        del kwargs
        if self.interrupt:
            return None
        states = input_batch.states
        if not states:
            raise ValueError("Flux step execution requires at least one request state.")
        first = states[0]
        batch_size = len(states)
        if input_batch.latents.shape[0] != batch_size:
            raise RuntimeError(
                "FLUX step batch mismatch: "
                f"states={batch_size}, latent_batch={input_batch.latents.shape[0]}"
            )
        if input_batch.latents.ndim != 3:
            raise RuntimeError(
                "FLUX DiT latent batch must be [B,S,D], "
                f"got {tuple(input_batch.latents.shape)}"
            )
        latent_seq_lens = input_batch.latent_seq_lens
        if len(latent_seq_lens) != batch_size:
            raise RuntimeError("FLUX latent sequence metadata does not match batch size")
        if any(seq_len > input_batch.latents.shape[1] for seq_len in latent_seq_lens):
            raise RuntimeError("FLUX latent sequence metadata exceeds padded batch length")
        self._current_timestep = input_batch.timesteps[0]
        self.transformer.do_true_cfg = input_batch.do_true_cfg
        pooled_prompt_embeds = torch.cat([state.extra["pooled_prompt_embeds"] for state in states], dim=0)
        negative_pooled_prompt_embeds = None
        negative_text_ids = None
        if input_batch.do_true_cfg:
            negative_pooled_prompt_embeds = torch.cat(
                [state.extra["negative_pooled_prompt_embeds"] for state in states], dim=0
            )
            negative_text_ids = first.extra["negative_text_ids"]
        timestep = input_batch.timesteps
        image_id_values = [state.extra["latent_image_ids"] for state in states]
        # Same token count does not imply same RoPE geometry (e.g. 512x1024 vs
        # 1024x512). Share IDs only when (height, width) grids match; never use
        # GPU tensor.equal() on the hot path.
        image_geometries = tuple(
            (int(state.extra["height"]), int(state.extra["width"])) for state in states
        )
        share_image_ids = (
            len(set(image_geometries)) == 1
            and all(
                value.shape == image_id_values[0].shape for value in image_id_values
            )
        )
        if share_image_ids:
            latent_image_ids = image_id_values[0]
        else:
            latent_image_ids = torch.zeros(
                (batch_size, input_batch.latents.shape[1], image_id_values[0].shape[-1]),
                device=input_batch.latents.device,
                dtype=image_id_values[0].dtype,
            )
            for index, image_ids in enumerate(image_id_values):
                latent_image_ids[index, : image_ids.shape[0]] = image_ids
        text_mask = input_batch.prompt_embeds_mask
        text_len = input_batch.prompt_embeds.shape[1]
        joint_attention_kwargs = dict(self.joint_attention_kwargs or {})
        patch_tiling_enabled = os.environ.get(
            "VLLM_OMNI_PATCH_TILING", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if patch_tiling_enabled:
            raise ValueError(
                "token-tile path removed; only System B MixFusion "
                "(mixfusion_both / system_b_mixfusion_patch) remains"
            )
        patch_latents = input_batch.latents
        patch_prompt_embeds = input_batch.prompt_embeds
        patch_pooled_projections = pooled_prompt_embeds
        patch_timestep = timestep
        patch_guidance = input_batch.guidance
        patch_img_ids = latent_image_ids
        grid_shapes: list[tuple[int, int]] = []
        request_ids: list[str] = []
        adapter_ids: list[int | tuple[tuple[int, float], ...] | None] = []
        for state, sequence_length in zip(states, latent_seq_lens):
            height = int(state.extra["height"])
            width = int(state.extra["width"])
            if height % 16 or width % 16:
                raise ValueError(
                    "FLUX patch grid requires image dimensions divisible by 16: "
                    f"height={height} width={width}"
                )
            grid_shape = (height // 16, width // 16)
            if grid_shape[0] * grid_shape[1] != int(sequence_length):
                raise RuntimeError(
                    "FLUX patch grid does not match latent sequence length: "
                    f"grid={grid_shape} sequence_length={sequence_length}"
                )
            grid_shapes.append(grid_shape)
            request_ids.append(str(state.request_id))
            lora_request = getattr(state.sampling, "lora_request", None)
            # Prefer composition metadata when present; fall back to single id.
            composition = getattr(state, "lora_composition", None)
            if composition is None:
                composition = getattr(state.sampling, "lora_composition", None)
            if composition:
                items = getattr(composition, "items", None)
                if items is None:
                    raise TypeError(
                        "lora_composition must expose .items "
                        f"(got {type(composition)!r})"
                    )
                adapter_ids.append(
                    tuple(
                        (int(item.adapter_id), float(item.scale))
                        for item in items
                    )
                )
            else:
                adapter_ids.append(getattr(lora_request, "lora_int_id", None))
        native_patch_schedule = None
        # Allow one MixFusion / torch verification telemetry event per denoise
        # step so Fair M* measured windows still see adapter proof without
        # ~700 JSONL writes from every Linear.
        try:
            from vllm_omni.diffusion.lora.mixfusion_runtime import MixFusionRuntime

            MixFusionRuntime._fair_verify_emitted = False
        except Exception:
            pass
        try:
            from vllm_omni.diffusion.lora.layers.base_linear import (
                BaseLinearLayerWithLoRA,
            )

            BaseLinearLayerWithLoRA._torch_fair_verify_emitted = False
        except Exception:
            pass
        # Fair M* / coupled timing: VLLM_OMNI_MIXFUSION_SCHEDULE=0 skips host
        # schedule_patch_work + native split entirely (full-sequence GPU path).
        mixfusion_schedule_enabled = os.environ.get(
            "VLLM_OMNI_MIXFUSION_SCHEDULE", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        if uses_mixfusion_runtime() and mixfusion_schedule_enabled:
            batch_size = int(input_batch.latents.shape[0])
            if len(grid_shapes) == 1 and batch_size > 1:
                grid_shapes = grid_shapes * batch_size
            source_samples: dict[tuple[int, int], list[torch.Tensor]] = {}
            source_indices: dict[str, list[str]] = {}
            for request_index, (state, sequence_length) in enumerate(
                zip(states, latent_seq_lens)
            ):
                height = int(state.extra["height"])
                width = int(state.extra["width"])
                geometry = (height, width)
                source_sample = self._unpack_latents(
                    input_batch.latents[request_index : request_index + 1, :sequence_length],
                    height,
                    width,
                    self.vae_scale_factor,
                )
                source_samples.setdefault(geometry, []).append(source_sample)
                source_indices.setdefault(f"{height}x{width}", []).append(
                    str(request_index)
                )
            source_sample_batches = {
                # Upstream split keys by resolution int; square geometry only.
                height: torch.cat(samples, dim=0)
                for (height, width), samples in source_samples.items()
                if height == width
            }
            if len(source_sample_batches) != len(source_samples):
                raise ValueError(
                    "MixFusion source split requires square (height==width) "
                    "geometry; mixed aspect ratios must be grouped separately"
                )
            # Default: one patch covering the full token grid (historical M*
            # latency contract). Multi-patch experiments must set
            # VLLM_OMNI_MIXFUSION_PATCH_GRID explicitly (e.g. 16).
            default_patch_h = max(int(h) for h, _ in grid_shapes)
            default_patch_w = max(int(w) for _, w in grid_shapes)
            patch_grid_env = os.environ.get("VLLM_OMNI_MIXFUSION_PATCH_GRID")
            native_patch_schedule = schedule_patch_work(
                grid_shapes,
                patch_height=int(
                    patch_grid_env if patch_grid_env else default_patch_h
                ),
                patch_width=int(
                    patch_grid_env if patch_grid_env else default_patch_w
                ),
                overlap=0,
                source_samples=source_sample_batches,
                source_indices={
                    str(height): source_indices[f"{height}x{width}"]
                    for (height, width) in source_samples
                },
                adapter_signatures=tuple(
                    tuple(
                        value
                        for slot, scale in adapter_id
                        for value in (
                            int(slot),
                            adapter_scale_signature_key(float(scale)),
                        )
                    )
                    if isinstance(adapter_id, tuple)
                    else ((int(adapter_id),) if adapter_id is not None else ())
                    for adapter_id in adapter_ids
                ),
            )
            emit_event(
                "dit_events",
                "mixfusion_native_patch_schedule",
                **native_patch_schedule.event_payload(),
                execution_contract=(
                    "source_split_schedule_plus_full_sequence_attention"
                ),
            )
        elif uses_mixfusion_runtime():
            emit_event(
                "dit_events",
                "mixfusion_schedule_skipped",
                execution_contract="full_sequence_gpu_no_host_schedule",
                reason="VLLM_OMNI_MIXFUSION_SCHEDULE=0",
            )
        if text_mask is not None or len(set(latent_seq_lens)) > 1:
            attention_mask = torch.ones(
                (batch_size, text_len + input_batch.latents.shape[1]),
                dtype=torch.bool,
                device=input_batch.latents.device,
            )
            if text_mask is not None:
                attention_mask[:, :text_mask.shape[1]] = text_mask.to(torch.bool)
            for index, seq_len in enumerate(latent_seq_lens):
                attention_mask[index, text_len + seq_len :] = False
            joint_attention_kwargs["attention_mask"] = attention_mask
        positive_kwargs = {
            "hidden_states": patch_latents,
            "timestep": patch_timestep / 1000,
            "guidance": patch_guidance,
            "pooled_projections": patch_pooled_projections,
            "encoder_hidden_states": patch_prompt_embeds,
            "txt_ids": first.extra["text_ids"],
            "img_ids": patch_img_ids,
            "joint_attention_kwargs": joint_attention_kwargs,
            "return_dict": False,
        }
        negative_kwargs = None
        if input_batch.do_true_cfg:
            negative_kwargs = {
                "hidden_states": patch_latents,
                "timestep": patch_timestep / 1000,
                "guidance": patch_guidance,
                "pooled_projections": negative_pooled_prompt_embeds,
                "encoder_hidden_states": input_batch.negative_prompt_embeds,
                "txt_ids": negative_text_ids,
                "img_ids": patch_img_ids,
                "joint_attention_kwargs": joint_attention_kwargs,
                "return_dict": False,
            }
        # Homogeneous multi-cycle: execute each sub-batch separately with the
        # same request permutation for latents / embeddings / IDs / mask / LoRA
        # / outputs. Never concatenate cycles into one heterogeneous forward
        # while reporting homogeneous scheduling.
        cycles = (
            getattr(native_patch_schedule, "cycle_request_indices", ())
            if native_patch_schedule is not None
            else ()
        )
        homogeneous = bool(
            getattr(native_patch_schedule, "homogeneous", True)
        ) if native_patch_schedule is not None else True
        if (
            native_patch_schedule is not None
            and homogeneous
            and len(cycles) > 1
        ):
            noise_parts: list[torch.Tensor] = []
            restore: list[int] = []
            for cycle in cycles:
                cycle_idx = tuple(int(i) for i in cycle)
                restore.extend(cycle_idx)
                cycle_schedule = NativePatchSchedule(
                    work_items=tuple(
                        item
                        for item in native_patch_schedule.work_items
                        if int(item.request_index) in set(cycle_idx)
                    ),
                    patch_height=native_patch_schedule.patch_height,
                    patch_width=native_patch_schedule.patch_width,
                    overlap=native_patch_schedule.overlap,
                    request_grid_shapes=tuple(
                        native_patch_schedule.request_grid_shapes[i]
                        for i in cycle_idx
                    ),
                    source_scheduler_called=(
                        native_patch_schedule.source_scheduler_called
                    ),
                    source_split_called=native_patch_schedule.source_split_called,
                    source_split_input_shape=(
                        native_patch_schedule.source_split_input_shape
                    ),
                    source_split_output_shape=(
                        native_patch_schedule.source_split_output_shape
                    ),
                    source_split_patch_count=(
                        native_patch_schedule.source_split_patch_count
                    ),
                    cycle_request_indices=(cycle_idx,),
                    homogeneous=True,
                )
                # Remap work_items request_index into the packed sub-batch.
                index_map = {src: dst for dst, src in enumerate(cycle_idx)}
                remapped_items = tuple(
                    NativePatchWork(
                        request_index=index_map[int(item.request_index)],
                        patch_index=int(item.patch_index),
                        row_start=int(item.row_start),
                        column_start=int(item.column_start),
                        valid_height=int(item.valid_height),
                        valid_width=int(item.valid_width),
                        grid_height=int(item.grid_height),
                        grid_width=int(item.grid_width),
                    )
                    for item in cycle_schedule.work_items
                )
                cycle_schedule = NativePatchSchedule(
                    work_items=remapped_items,
                    patch_height=cycle_schedule.patch_height,
                    patch_width=cycle_schedule.patch_width,
                    overlap=cycle_schedule.overlap,
                    request_grid_shapes=cycle_schedule.request_grid_shapes,
                    source_scheduler_called=cycle_schedule.source_scheduler_called,
                    source_split_called=cycle_schedule.source_split_called,
                    source_split_input_shape=cycle_schedule.source_split_input_shape,
                    source_split_output_shape=cycle_schedule.source_split_output_shape,
                    source_split_patch_count=cycle_schedule.source_split_patch_count,
                    cycle_request_indices=(
                        tuple(range(len(cycle_idx))),
                    ),
                    homogeneous=True,
                )
                sub_mask = None
                if joint_attention_kwargs.get("attention_mask") is not None:
                    sub_mask = joint_attention_kwargs["attention_mask"][
                        list(cycle_idx)
                    ]
                # M2: slice per-request timestep/guidance with the same cycle.
                full_ts = positive_kwargs["timestep"]
                full_gs = positive_kwargs["guidance"]
                sub_ts = (
                    full_ts[list(cycle_idx)]
                    if isinstance(full_ts, torch.Tensor) and full_ts.dim() >= 1
                    and full_ts.shape[0] == patch_latents.shape[0]
                    else full_ts
                )
                sub_gs = (
                    full_gs[list(cycle_idx)]
                    if isinstance(full_gs, torch.Tensor)
                    and full_gs.dim() >= 1
                    and full_gs.shape[0] == patch_latents.shape[0]
                    else full_gs
                )
                sub_positive = {
                    **positive_kwargs,
                    "hidden_states": patch_latents[list(cycle_idx)],
                    "timestep": sub_ts,
                    "guidance": sub_gs,
                    "pooled_projections": patch_pooled_projections[
                        list(cycle_idx)
                    ],
                    "encoder_hidden_states": patch_prompt_embeds[
                        list(cycle_idx)
                    ],
                    "img_ids": (
                        patch_img_ids
                        if patch_img_ids.dim() == 2
                        else patch_img_ids[list(cycle_idx)]
                    ),
                    "joint_attention_kwargs": {
                        **joint_attention_kwargs,
                        **(
                            {"attention_mask": sub_mask}
                            if sub_mask is not None
                            else {}
                        ),
                    },
                }
                sub_negative = None
                if negative_kwargs is not None:
                    sub_negative = {
                        **negative_kwargs,
                        "hidden_states": patch_latents[list(cycle_idx)],
                        "timestep": sub_ts,
                        "guidance": sub_gs,
                        "pooled_projections": negative_pooled_prompt_embeds[
                            list(cycle_idx)
                        ],
                        "encoder_hidden_states": (
                            input_batch.negative_prompt_embeds[list(cycle_idx)]
                        ),
                        "img_ids": sub_positive["img_ids"],
                        "joint_attention_kwargs": sub_positive[
                            "joint_attention_kwargs"
                        ],
                    }
                # M3: rebind LoRA composition/row_layout to the packed sub-batch.
                saved_comp = slice_lora_composition_for_cycle(
                    self.transformer, cycle_idx
                )
                try:
                    with use_native_patch_schedule(cycle_schedule):
                        part = self.predict_noise_maybe_with_cfg(
                            input_batch.do_true_cfg,
                            input_batch.true_cfg_scale,
                            sub_positive,
                            sub_negative,
                            cfg_normalize=False,
                        )
                finally:
                    restore_lora_composition(saved_comp)
                noise_parts.append(part)
            packed = torch.cat(noise_parts, dim=0)
            # Restore original request order.
            inverse = [0] * len(restore)
            for new_pos, old_pos in enumerate(restore):
                inverse[old_pos] = new_pos
            return packed[inverse]

        with use_native_patch_schedule(native_patch_schedule):
            noise_pred = self.predict_noise_maybe_with_cfg(
                input_batch.do_true_cfg,
                input_batch.true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize=False,
            )
        return noise_pred

    def step_scheduler(
        self,
        state: "StepRequestState",
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Advance one Flux scheduler step."""
        del kwargs
        if self.interrupt:
            return
        if noise_pred.ndim == 3 and noise_pred.shape[1] != state.latents.shape[1]:
            if noise_pred.shape[1] < state.latents.shape[1]:
                raise RuntimeError(
                    "FLUX noise prediction is shorter than request latents: "
                    f"prediction={tuple(noise_pred.shape)}, "
                    f"latents={tuple(state.latents.shape)}"
                )
            noise_pred = noise_pred[:, : state.latents.shape[1]]
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            state.current_timestep,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        )
        state.step_index += 1

    def post_decode(
        self,
        state: "StepRequestState",
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final Flux latents."""
        del kwargs
        self._current_timestep = None
        output_type = state.sampling.output_type or "pil"
        if output_type == "latent":
            return DiffusionOutput(output=state.latents)
        height = state.extra["height"]
        width = state.extra["width"]
        latents = self._unpack_latents(state.latents, height, width, self.vae_scale_factor)
        expected_latent_shape = (
            1,
            self.transformer.in_channels // 4,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if tuple(latents.shape) != expected_latent_shape:
            raise RuntimeError(
                "FLUX unpacked latent shape mismatch: "
                f"expected {expected_latent_shape}, got {tuple(latents.shape)}"
            )
        if latents.device != self.device:
            raise RuntimeError(
                f"FLUX unpacked latent device mismatch: expected {self.device}, got {latents.device}"
            )
        emit_event(
            "dit_events",
            "flux_latent_unpacked",
            packed_latents=tensor_metadata(state.latents, role="flux_packed_latent_final"),
            unpacked_latents=tensor_metadata(latents, role="flux_bchw_latent_before_vae"),
            image_height=int(height),
            image_width=int(width),
        )
        emit_event(
            "dit_events",
            "flux_latent_before_vae_decode",
            image_height=int(height),
            image_width=int(width),
            vae_scale_factor=int(self.vae_scale_factor),
            patch_size=2,
            packed_latents=tensor_metadata(
                state.latents, role="flux_packed_latent_state"
            ),
            unpacked_latents=tensor_metadata(
                latents, role="flux_unpacked_latent_before_vae_decode"
            ),
        )
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        emit_event(
            "dit_events",
            "flux_component_executed",
            component="vae",
            operations=["AutoencoderKL.decode"],
            request_id=str(state.request_id),
            input_latent_shape=[int(value) for value in latents.shape],
            output_image_shape=[int(value) for value in image.shape],
        )
        return DiffusionOutput(output=image)

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Forward pass for flux."""
        # TODO: In online mode, sometimes it receives [{"negative_prompt": None}, {...}], so cannot use .get("...", "")
        # TODO: May be some data formatting operations on the API side. Hack for now.
        sampling_params_list = req.sampling_params_list
        common_sampling_params = sampling_params_list[0]
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts]

        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        height = common_sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = common_sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = common_sampling_params.num_inference_steps or 28
        sigmas = common_sampling_params.sigmas
        guidance_scale = common_sampling_params.guidance_scale
        true_cfg_scale = common_sampling_params.true_cfg_scale or 1.0
        num_images_per_prompt = (
            common_sampling_params.num_outputs_per_prompt if common_sampling_params.num_outputs_per_prompt > 0 else 1
        )
        max_sequence_length = common_sampling_params.max_sequence_length or 512
        generator = req.collate_request_generators(num_images_per_prompt, None)
        latents = req.collate_request_tensors("latents", None)
        prompt_fields = DiffusionRequestBatch.collate_prompt_field_map(
            req.prompts,
            {
                "prompt_embeds": None,
                "negative_prompt_embeds": None,
                "pooled_prompt_embeds": None,
                "negative_pooled_prompt_embeds": None,
            },
        )
        prompt_embeds = prompt_fields["prompt_embeds"]
        negative_prompt_embeds = prompt_fields["negative_prompt_embeds"]
        pooled_prompt_embeds = prompt_fields["pooled_prompt_embeds"]
        negative_pooled_prompt_embeds = prompt_fields["negative_pooled_prompt_embeds"]
        prompt_2 = None
        negative_prompt_2 = None
        if prompt_embeds is not None:
            prompt = None
        if negative_prompt_embeds is not None:
            negative_prompt = None

        output_type = common_sampling_params.output_type or "pil"
        joint_attention_kwargs: dict[str, Any] | None = None
        callback_on_step_end_tensor_inputs = ["latents"]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        negative_text_ids = None
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        latents = self.diffuse(
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            latents,
            latent_image_ids,
            text_ids,
            negative_text_ids,
            timesteps,
            do_true_cfg,
            guidance,
            true_cfg_scale,
            cfg_normalize=False,
        )

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]

        stage_durations = self.stage_durations if hasattr(self, "stage_durations") else None
        n = num_images_per_prompt
        if req.num_reqs == 1:
            return [DiffusionOutput(output=image, stage_durations=stage_durations)]
        return [
            DiffusionOutput(
                output=image[i * n : (i + 1) * n] if isinstance(image, list) else image[i * n : (i + 1) * n],
                stage_durations=stage_durations,
            )
            for i in range(req.num_reqs)
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class FluxDMD2Pipeline(DMD2PipelineMixin, FluxPipeline):
    """Flux pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> list[DiffusionOutput]:
        return super().forward(req, **kwargs)
