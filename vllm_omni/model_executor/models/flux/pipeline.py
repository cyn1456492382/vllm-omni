# SPDX-License-Identifier: Apache-2.0
"""Pipeline topology for diffusers FLUX.1 models."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)


FLUX_PIPELINE = PipelineConfig(
    model_type="flux",
    default_deploy_config_name="dit_lora_async.yaml",
    model_arch="FluxPipeline",
    hf_architectures=("FluxPipeline", "FluxDMD2Pipeline"),
    diffusers_class_name="FluxPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
            model_arch="FluxPipeline",
            requires_multimodal_data=False,
        ),
    ),
)
