###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Version-compatibility layer for Instella-MoE, transformers 4.57 -> 5.15.

Instella-MoE imports the numerically-unchanged building blocks from the installed
`transformers.models.deepseek_v3`, whose internals changed several times across that range.
Every one of those differences is resolved ONCE here, at import time, and is then re-exported under a
stable name, so that `modeling_instella_moe.py` remains stable

Resolution is by attribute checking and signature instead of version number, except where there is no distinct signature

For the MoE layer we have
  * 4.57.0 - 4.57.6 - `DeepseekV3MoE.moe` is a method; the experts are an `nn.ModuleList`.
                      4.57.6 is the final 4.x release
  * 5.0.0 - 5.12.1  - `DeepseekV3MoE.route_tokens_to_experts` exists; the router returns raw
                      logits and the expert weights become fused `nn.Parameter`s on
                      `DeepseekV3NaiveMoe`
  * 5.13.0+         - `DeepseekV3Experts` exists (a rename of `DeepseekV3NaiveMoe`); the
                      router returns `(logits, weights, indices)` and there is no `route_tokens_to_experts`
"""

import inspect
import warnings

import transformers
from packaging.version import parse
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from transformers.models.deepseek_v3 import modeling_deepseek_v3 as _dsv3
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    eager_attention_forward,
)


###############################################################################
# Causal mask
###############################################################################

# 5.2 renamed the `input_embeds` argument to `inputs_embeds`, and 5.9 dropped `cache_position` Passing either one to a
# version that does not take it is a TypeError, so the call is assembled from the signature.
_MASK_PARAMS = inspect.signature(create_causal_mask).parameters
_MASK_EMBEDS_KWARG = "inputs_embeds" if "inputs_embeds" in _MASK_PARAMS else "input_embeds"
_MASK_TAKES_CACHE_POSITION = "cache_position" in _MASK_PARAMS


def build_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids, cache_position):
    kwargs = {
        "config": config,
        _MASK_EMBEDS_KWARG: inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    if _MASK_TAKES_CACHE_POSITION:
        kwargs["cache_position"] = cache_position
    return create_causal_mask(**kwargs)


###############################################################################
# Model-forward decorator
###############################################################################

# 5.2 split `check_model_inputs` into `merge_with_config_defaults` + `capture_outputs`. It was
# re-added in 5.4 for backwards compatibility, but upstream deepseek_v3 has used the split pair
# ever since, so prefer that wherever it exists and fall back to the old decorator on 4.57 - 5.1.
try:
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs
except ImportError:
    from transformers.utils.generic import check_model_inputs

    model_forward_decorator = check_model_inputs
else:

    def model_forward_decorator(func):
        return merge_with_config_defaults(capture_outputs(func))


###############################################################################
# MoE routing
###############################################################################

# The router and expert calling conventions changed twice. Each variant takes the 3D hidden
# states, runs the router on them and the experts on their flattened view

def _routed_output_moe_method(mlp, hidden_states):
    """4.57: the router returns the top-k selection directly, the experts hang off `.moe`."""
    topk_indices, topk_weights = mlp.gate(hidden_states)
    return mlp.moe(hidden_states.view(-1, hidden_states.shape[-1]), topk_indices, topk_weights)


def _routed_output_route_tokens(mlp, hidden_states):
    """5.0 - 5.12: the router returns raw logits, top-k moved to `route_tokens_to_experts`."""
    topk_indices, topk_weights = mlp.route_tokens_to_experts(mlp.gate(hidden_states))
    return mlp.experts(hidden_states.view(-1, hidden_states.shape[-1]), topk_indices, topk_weights)


def _routed_output_gate_triple(mlp, hidden_states):
    """5.13+: the router returns `(logits, weights, indices)` and computes the top-k internally."""
    _, topk_weights, topk_indices = mlp.gate(hidden_states)
    return mlp.experts(hidden_states.view(-1, hidden_states.shape[-1]), topk_indices, topk_weights)


if hasattr(_dsv3, "DeepseekV3Experts"):
    routed_expert_output = _routed_output_gate_triple
elif hasattr(DeepseekV3MoE, "route_tokens_to_experts"):
    routed_expert_output = _routed_output_route_tokens
elif hasattr(DeepseekV3MoE, "moe"):
    routed_expert_output = _routed_output_moe_method
else:
    raise ImportError(
        "Instella-MoE cannot determine the DeepseekV3MoE routing convention of transformers "
        f"{transformers.__version__}: none of `DeepseekV3Experts`, "
        "`DeepseekV3MoE.route_tokens_to_experts` or `DeepseekV3MoE.moe` is present. "
        "Supported range is 4.57 - 5.15."
    )


# 5.0 replaced the per-expert `nn.ModuleList` with a single module holding fused `gate_up_proj` /
# `down_proj` `nn.Parameter`s, renamed from `DeepseekV3NaiveMoe` to `DeepseekV3Experts` in 5.13.
# `PreTrainedModel._init_weights` dispatches on module type and so never reaches bare Parameters,
# which is why `_init_weights` has to name this class explicitly. Empty on 4.57, where the experts
# are ordinary `nn.Linear` submodules that the default initializer already handles.
FUSED_EXPERT_CLASSES = tuple(
    cls
    for cls in (getattr(_dsv3, name, None) for name in ("DeepseekV3Experts", "DeepseekV3NaiveMoe"))
    if cls is not None
)


###############################################################################
# Weight initialization
###############################################################################

# 5.0 moved weight init to `transformers.initialization`; 4.57 still initializes in place through `.data`
try:
    from transformers import initialization as _init
except ImportError:

    def normal_(tensor, std):
        tensor.data.normal_(mean=0.0, std=std)

    def zeros_(tensor):
        tensor.data.zero_()

else:

    def normal_(tensor, std):
        _init.normal_(tensor, mean=0.0, std=std)

    def zeros_(tensor):
        _init.zeros_(tensor)


###############################################################################
# fp32 module list
###############################################################################

KEEP_IN_FP32_MODULES_STRICT = getattr(DeepseekV3PreTrainedModel, "_keep_in_fp32_modules_strict", None)


###############################################################################
# Attention interface
###############################################################################

# 5.1 added `get_interface`, which also resolves kernel specs such as
# "kernels-community/flash-attn"; plain subscripting raises KeyError on those.
if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):

    def get_attention_interface(config):
        return ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)

else:

    def get_attention_interface(config):
        if config._attn_implementation == "eager":
            return eager_attention_forward
        return ALL_ATTENTION_FUNCTIONS[config._attn_implementation]


###############################################################################
# MLA KV cache layout
###############################################################################

# 5.15 moved the MLA cache read/write ahead of the `kv_b_proj` expansion, so the cache now holds
# the 512-wide latent instead of materialized K/V (~7.5x less KV memory for this config) and
# `DeepseekV3Attention` grew an `expand_kv` helper to reconstruct them. This is a key place
# where the two code paths differ, we monkey patch `MLAGatedAttention` to one of the two
# `_kv_and_cache` implementations based on this flag
HAS_LATENT_KV_CACHE = hasattr(DeepseekV3Attention, "expand_kv")


###############################################################################
# Checkpoint conversion registry
###############################################################################

try:
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping as _get_conversion_mapping,
        register_checkpoint_conversion_mapping as _register_conversion_mapping,
    )
except ImportError:  # 4.57 loads the per-expert weights as they are stored
    _get_conversion_mapping = _register_conversion_mapping = None


def register_conversion_mapping():
    """Make the stock deepseek_v3 checkpoint conversion apply for Instella-MoE. (No-op before 5.0)
    """
    if _register_conversion_mapping is None:
        return

    # `extract_weight_conversions_for_model` will skip remote-code models unless their model_type
    # was registered explicitly, so without this the per-expert -> fused `gate_up_proj`/`down_proj`
    # merge never runs and every routed expert silently loads as random weights.
    #
    # Reading the mapping back out of the registry is only correct because importing
    # `modeling_deepseek_v3` is what populates it, and this module imports it at the top.
    # Registering ahead of that import would store an empty mapping and reintroduce exactly the
    # silent random-expert failure this call exists to prevent, so the result is checked.
    #
    # We keep our own model_type as "deepseek_v3", so this also rewrites the process-global entry
    # that a stock DeepSeek-V3 loaded alongside us would use.
    try:
        mapping = _get_conversion_mapping("deepseek_v3")
    except Exception as exc:  # a renamed registry should not take the whole import down
        warnings.warn(
            f"Instella-MoE could not read the deepseek_v3 checkpoint conversion mapping ({exc!r}); "
            "routed expert weights may load uninitialized.",
            stacklevel=2,
        )
        return

    if not mapping:
        warnings.warn(
            "Instella-MoE found an empty deepseek_v3 checkpoint conversion mapping; routed expert "
            "weights may load uninitialized.",
            stacklevel=2,
        )
        return

    _register_conversion_mapping("deepseek_v3", mapping, overwrite=True)


###############################################################################
# Class-attribute colwise
###############################################################################

# 5.1 renamed the tensor-parallel style "colwise_rep" -> "colwise_gather_output" unlike everything
# else in this file there is nothing importable to probe and the version has to be compared.
TP_PLAN_COLWISE_GATHER = (
    "colwise_gather_output" if parse(transformers.__version__) >= parse("5.1") else "colwise_rep"
)
