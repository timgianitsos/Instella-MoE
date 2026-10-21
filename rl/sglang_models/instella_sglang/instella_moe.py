###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Instella-MoE for SGLang, as an external model package (no SGLang overlay).

Instella-MoE is DeepSeek-V3's MoE with two changes (HF config: ``gated_attention``,
``farskip``):
- gated MLA: the attention output is multiplied by sigmoid(gate_proj(attention input)) before
  o_proj;
- FarSkip-Collective connectivity: besides the residual stream h, each layer passes on h', its
  output without the routed experts' contribution (h' = h for the first layer and after dense
  layers). With r = h + attention(norm(h')), a layer outputs r + mlp(norm(h)) and
  h' = r + shared_experts(norm(h)).

This is the reference (non-overlapped) FarSkip forward, built from SGLang's DeepSeek-V2 modules.
It supports one attention/MoE tensor-parallel rank per engine (TP=1, no DP attention or PP),
which is how Miles runs the Instella student.

    SGLANG_EXTERNAL_MODEL_PACKAGE=instella_sglang PYTHONPATH=<this dir's parent>:...
"""

from typing import Optional

import torch

from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_v2 import (
    BumpAllocator,
    DeepseekV2ForCausalLM,
    DeepseekV2MoE,
    get_attn_tp_context,
    get_parallel,
)


class InstellaMoEForCausalLM(DeepseekV2ForCausalLM):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        parallel = get_parallel()
        assert parallel.tp_size == 1 and parallel.attn_tp_size == 1, "Instella FarSkip needs TP=1"
        assert self.pp_group.world_size == 1, "Instella FarSkip does not support PP"
        self.farskip = getattr(config, "farskip", False)
        self.gated_attention = getattr(config, "gated_attention", False)
        if self.gated_attention:
            for layer in self.model.layers:
                _add_attention_gate(layer.self_attn, config, quant_config)

    def load_weights(self, weights, is_nextn=False):
        weights = (
            (name.replace(".self_attn.gate_proj.", ".self_attn.o_gate."), tensor)
            for name, tensor in weights
        )
        return super().load_weights(weights, is_nextn=is_nextn)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self._model_forward(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def _model_forward(self, input_ids, positions, forward_batch, input_embeds):
        model = self.model
        hidden_states = model.embed_tokens(input_ids) if input_embeds is None else input_embeds
        # h' (see module docstring); equals h unless FarSkip is on.
        hidden_states_no_routed = hidden_states
        zero_allocator = BumpAllocator(
            buffer_size=len(model.layers) * 2, dtype=torch.float32, device=positions.device
        )
        for layer in model.layers:
            attn_input = layer.input_layernorm(hidden_states_no_routed)
            attn = layer.self_attn
            if self.gated_attention:
                attn.instella_gate = torch.sigmoid(attn.o_gate(attn_input)[0])
            with attn.maybe_use_decode_attn_tp(forward_batch):
                attn_output = attn(
                    positions=positions,
                    hidden_states=attn_input,
                    forward_batch=forward_batch,
                    zero_allocator=zero_allocator,
                )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            get_attn_tp_context().clear_attn_inputs()
            residual = hidden_states + attn_output

            mlp_input = layer.post_attention_layernorm(hidden_states if self.farskip else residual)
            if isinstance(layer.mlp, DeepseekV2MoE):
                moe_output = layer.mlp(mlp_input, forward_batch, None, return_moe_output=True)
                hidden_states = residual + moe_output.get_merged()
                shared = moe_output.shared
                hidden_states_no_routed = residual if shared is None else residual + shared
            else:
                hidden_states = residual + layer.mlp(mlp_input, forward_batch, None)
                hidden_states_no_routed = hidden_states
            if not self.farskip:
                hidden_states_no_routed = hidden_states
        return model.norm(hidden_states)


def _add_attention_gate(attn, config, quant_config):
    """The attention gate; o_proj's input is multiplied by sigmoid(gate).

    Registered as ``o_gate`` (the checkpoint's ``self_attn.gate_proj``, see load_weights): the
    DeepSeek weight loader would rename any ``gate_proj`` to the MLP's ``gate_up_proj``.
    """
    attn.o_gate = ColumnParallelLinear(
        config.hidden_size,
        config.num_attention_heads * config.v_head_dim,
        bias=False,
        quant_config=None,
    )

    def gate_o_proj_input(module, args):
        return (args[0] * attn.instella_gate,) + tuple(args[1:])

    attn.o_proj.register_forward_pre_hook(gate_o_proj_input)


EntryClass = InstellaMoEForCausalLM
