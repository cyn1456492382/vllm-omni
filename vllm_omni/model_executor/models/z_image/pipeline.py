# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Z-Image pipeline topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

Z_IMAGE_PIPELINE = PipelineConfig(
    model_type="z_image",
    default_deploy_config_name="dit_lora_async.yaml",
    model_arch="ZImagePipeline",
    hf_architectures=("ZImagePipeline",),
    diffusers_class_name="ZImagePipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch="ZImagePipeline",
            requires_multimodal_data=False,
        ),
    ),
)
