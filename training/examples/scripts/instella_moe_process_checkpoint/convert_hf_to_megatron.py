#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Instella MoE HF -> Megatron torch_dist.

Runs Miles' converter (rl/miles/tools/convert_hf_to_torch_dist.py: mbridge loads the HF weights
into the Megatron model built from ${MODEL_ARGS[@]}, then a release torch_dist checkpoint is
saved) after registering Instella's gated-attention weight on mbridge's DeepSeek-V3 bridge:
Megatron's self_attention.linear_gate_proj is the HF checkpoint's self_attn.gate_proj.

Usage:
  torchrun --nproc-per-node 1 convert_hf_to_megatron.py \\
      ${MODEL_ARGS[@]} --hf-checkpoint <hf_dir> --save <megatron_out>
"""

import os
import runpy

from mbridge.models import DeepseekV3Bridge

DeepseekV3Bridge._ATTENTION_MAPPING = {
    **DeepseekV3Bridge._ATTENTION_MAPPING,
    "self_attention.linear_gate_proj.weight": ["model.layers.{layer_number}.self_attn.gate_proj.weight"],
}

runpy.run_path(
    os.path.join(os.environ["MILES_ROOT"], "tools", "convert_hf_to_torch_dist.py"), run_name="__main__"
)
