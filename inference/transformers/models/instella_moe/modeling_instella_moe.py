###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Modular Instella-MoE for transformers 4.57 - 5.15.

This file imports the numerically-unchanged building blocks from the installed
`transformers.models.deepseek_v3` package and defines ONLY the classes that carry
a FarSkip or gated-attention delta:

  * InstellaMoEForCausalLM
  * InstellaMoEPreTrainedModel
  * InstellaMoEModel      - unwraps the residual tuple before the final norm
  * FarSkipDecoderLayer - tuple-residual (residual, residual_no_routed) dataflow
  * FarSkipMoE        - returns (routed, shared) separately (FarSkip needs both)
  * MLAGatedAttention  - adds sigmoid `gate_proj` before `o_proj`

Everything upstream changed between 4.57 and 5.15 is resolved once in
`_transformers_compat.py` and imported here under a stable name, so there are no version
checks below.
"""

from collections.abc import Callable
# `X | None` annotations are what 5.x uses, but 4.57's `auto_docstring` inspects them with
# `annotation.__name__`, which a `types.UnionType` does not have. `Optional` works on both.
from typing import Optional, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3TopkRouter,
    DeepseekV3Attention,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
)

from ._transformers_compat import (
    FUSED_EXPERT_CLASSES,
    HAS_LATENT_KV_CACHE,
    KEEP_IN_FP32_MODULES_STRICT,
    TP_PLAN_COLWISE_GATHER,
    build_causal_mask,
    get_attention_interface,
    model_forward_decorator,
    normal_,
    register_conversion_mapping,
    routed_expert_output,
    zeros_,
)
from .configuration_instella_moe import InstellaMoEConfig

register_conversion_mapping()


def _kv_and_cache_materialized(self, kv_latent, k_rot, cos, sin, past_key_values, cache_position):
    """Expand the latent to full K/V first, then cache the materialized tensors (4.57 - 5.14)."""
    batch_size, _, seq_length, _ = k_rot.shape
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    k_pass = self.kv_b_proj(kv_latent).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    key_states = torch.cat((k_pass, k_rot.expand(*k_pass.shape[:-1], -1)), dim=-1)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    return key_states, value_states


def _kv_and_cache_latent(self, kv_latent, k_rot, cos, sin, past_key_values, cache_position):
    """Cache the compressed latent and reconstruct K/V from it after the read (5.15+)."""
    batch_size, _, seq_length, _ = k_rot.shape
    # Both latents are viewed as single-head, 4D tensors so all cache layers handle them correctly
    kv_latent = kv_latent.view(batch_size, 1, seq_length, self.kv_lora_rank)

    # Cache read / write is performed while latent KV is still compressed
    if past_key_values is not None:
        kv_latent, k_rot = past_key_values.update(kv_latent, k_rot, self.layer_idx)

    return self.expand_kv(kv_latent, k_rot)


class MLAGatedAttention(DeepseekV3Attention):
    """DeepSeek-V3 MLA with optional gated attention (attn_output * sigmoid(gate_proj(x)))."""

    # The only place the supported transformers range needs two genuinely different code paths.
    _kv_and_cache = _kv_and_cache_latent if HAS_LATENT_KV_CACHE else _kv_and_cache_materialized

    def __init__(self, config: InstellaMoEConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.gated_attention = getattr(config, "gated_attention", False)
        if self.gated_attention:
            self.gate_proj = nn.Linear(
                config.hidden_size, self.num_heads * self.v_head_dim, bias=False
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states, value_states = self._kv_and_cache(
            kv_latent, k_rot, cos, sin, past_key_values, cache_position
        )

        attention_interface: Callable = get_attention_interface(self.config)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()

        if self.gated_attention:
            attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class FarSkipMoE(DeepseekV3MoE):
    """FarSkip-Collective MoE that returns routed and shared outputs separately so FarSkip can route them
    into the two residual streams independently."""

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        routed = routed_expert_output(self, hidden_states).view(*orig_shape)
        shared = self.shared_experts(residuals)
        # First element is the full MoE output (routed + shared) so the main residual
        # stream stays numerically identical to stock DeepSeek-V3; `shared` is returned
        # separately so FarSkip can build the routed-free residual stream.
        return routed + shared, shared


class FarSkipDecoderLayer(GradientCheckpointingLayer):
    """
    FarSkip-Collective connectivity decoder layer
    """
    def __init__(self, config: InstellaMoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MLAGatedAttention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = FarSkipMoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.config = config
        self.layer_idx = layer_idx
        self.farskip = (
            config.farskip
            and layer_idx >= config.farskip_start_idx
            and layer_idx <= min(config.farskip_end_idx, config.num_hidden_layers - 1)
        )

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if self.farskip:
            if not isinstance(hidden_states, tuple):  # first farskip layer
                residual = hidden_states
                input_to_attn = hidden_states
                input_to_mlp = hidden_states
            else:
                residual = hidden_states[0]
                input_to_attn = hidden_states[1]
                input_to_mlp = residual
            if self.config.attn_only_farskip:
                input_to_mlp = None
            if self.config.mlp_only_farskip:
                input_to_attn = residual
        else:
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            residual = hidden_states
            input_to_attn = hidden_states
            input_to_mlp = None

        input_to_attn = self.input_layernorm(input_to_attn)

        attn_output, _ = self.self_attn(
            hidden_states=input_to_attn,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        residual = residual + attn_output

        if input_to_mlp is None:
            input_to_mlp = residual
        input_to_mlp = self.post_attention_layernorm(input_to_mlp)

        if isinstance(self.mlp, FarSkipMoE):
            mlp_output, mlp_shared_output = self.mlp(input_to_mlp)
            residual_no_routed = residual + mlp_shared_output
            residual = residual + mlp_output
            # residual_no_routed is combine-free and feeds the next block's attention
            hidden_states = (residual, residual_no_routed)
        else:
            hidden_states = residual + self.mlp(input_to_mlp)

        return hidden_states


@auto_docstring
class InstellaMoEPreTrainedModel(PreTrainedModel):
    config: InstellaMoEConfig
    config_class = InstellaMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FarSkipDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": FarSkipDecoderLayer,
        "attentions": MLAGatedAttention,
    }
    _keep_in_fp32_modules_strict = KEEP_IN_FP32_MODULES_STRICT

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, DeepseekV3TopkRouter):
            normal_(module.weight, self.config.initializer_range)
            zeros_(module.e_score_correction_bias)
        elif isinstance(module, FUSED_EXPERT_CLASSES):
            normal_(module.gate_up_proj, self.config.initializer_range)
            normal_(module.down_proj, self.config.initializer_range)


@auto_docstring
class InstellaMoEModel(InstellaMoEPreTrainedModel):
    def __init__(self, config: InstellaMoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [FarSkipDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @model_forward_decorator
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        cache_position (`torch.LongTensor` of shape `(sequence_length,)`, *optional*):
            Indices of the input tokens within the full sequence. Documented here because 5.9
            removed it and `auto_docstring` will complain for any
            undocumented argument.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # `cache_position` is still needed for the pre-5.15 materialized KV cache write even
        # though the 5.9+ mask builder derives it from `position_ids` itself.
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = build_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            cache_position=cache_position,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]  # routed-inclusive residual stream
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class InstellaMoEForCausalLM(InstellaMoEPreTrainedModel, GenerationMixin):
    # The mapping form is read directly on 5.x; 4.57 iterates it, which yields the keys and so
    # matches the list of tied-weight patterns it expects.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": TP_PLAN_COLWISE_GATHER}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = InstellaMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        cache_position (`torch.LongTensor` of shape `(sequence_length,)`, *optional*):
            Indices of the input tokens within the full sequence. Documented here because 5.9
            removed it and `auto_docstring` will complain for any
            undocumented argument.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "InstellaMoEPreTrainedModel",
    "InstellaMoEModel",
    "InstellaMoEForCausalLM",
]
