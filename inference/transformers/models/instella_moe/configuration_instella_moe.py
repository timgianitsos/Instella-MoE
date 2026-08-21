###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}
class InstellaMoEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InstellaMoEModel`]. It is used to instantiate an
    Instella-MoE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of Instella-MoE-16B-A3B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 128896):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`InstellaMoEModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10944):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1408):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 27):
            Number of hidden layers in the Transformer decoder.
        num_nextn_predict_layers (`int`, *optional*, defaults to 0):
            Number of nextn predict layers in the InstellaMoE Model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        n_shared_experts (`int`, *optional*, defaults to 2):
            Number of shared experts, None means dense model.
        n_routed_experts (`int`, *optional*, defaults to 64):
            Number of routed experts, None means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor or routed experts.
        topk_method (`str`, *optional*, defaults to `noaux_tc`):
            Topk method used in routed gate.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to 6):
            Number of selected experts, None means dense model.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                            \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to True):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to 'sigmoid'):
            Method of computing expert weights.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary loss weight coefficient.
        seq_aux = (`bool`, *optional*, defaults to True):
            Whether to compute the auxiliary loss for each individual sample.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 8000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import InstellaMoEModel, InstellaMoEConfig

    >>> # Initializing an Instella-MoE style configuration
    >>> configuration = InstellaMoEConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # transformers 5.x `DeepseekV3TopkRouter` / `DeepseekV3Experts` read `num_local_experts`.
    attribute_map = {"num_local_experts": "n_routed_experts"}

    # Fused-expert parameter names only exist on transformers 5.x; these entries match
    # nothing on 4.x, where the routed experts are separate `DeepseekV3MLP` modules.
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    def __init__(
        self,
        vocab_size=128896,
        hidden_size=2048,
        intermediate_size=10944,
        moe_intermediate_size = 1408,
        num_hidden_layers=27,
        num_nextn_predict_layers=0,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_shared_experts = 2,
        n_routed_experts = 64,
        ep_size = 1,
        routed_scaling_factor = 2.5,
        kv_lora_rank = 512,
        q_lora_rank = None,
        qk_rope_head_dim = 32,
        qk_layernorm = True,
        v_head_dim = 128,
        qk_nope_head_dim = 96,
        qk_head_dim = None,
        rope_interleave = True,
        topk_method = 'noaux_tc',
        n_group = 1,
        topk_group = 1,
        num_experts_per_tok = 6,
        moe_layer_freq = 1,
        first_k_dense_replace = 1,
        gated_attention = True,
        norm_topk_prob = True,
        scoring_func = 'sigmoid',
        aux_loss_alpha = 0.001,
        seq_aux = True,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=8000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        farskip=True,
        tp_split=None,
        farskip_start_idx=0,
        farskip_end_idx=1e4,
        apply_all_reduce=False,
        attn_only_farskip=False,
        mlp_only_farskip=False,
        verbose=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_layernorm = qk_layernorm
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_head_dim if qk_head_dim is not None else qk_nope_head_dim + qk_rope_head_dim
        self.rope_interleave = rope_interleave
        # yarn rope init reads config.head_dim (falls back to hidden_size//num_heads
        # otherwise). MLA rope operates on qk_rope_head_dim, so pin it here to match.
        self.head_dim = qk_rope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.farskip = farskip
        self.tp_split = tp_split
        self.farskip_start_idx = farskip_start_idx
        self.farskip_end_idx = farskip_end_idx
        self.apply_all_reduce = apply_all_reduce
        self.attn_only_farskip = attn_only_farskip
        self.mlp_only_farskip = mlp_only_farskip
        self.verbose = verbose
        self.gated_attention = gated_attention

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            # 5.0 and 5.1 assemble `rope_parameters` by popping `rope_theta` out of **kwargs and
            # nowhere else, falling back to the 10000.0 default when it is absent - and declaring
            # it as a named argument above is exactly what makes it absent. 5.2 added a lookup on
            # the attribute as well, so passing it back in here is what every other version in the
            # range already does anyway. Without it 5.0/5.1 load with the wrong rope base and
            # produce fluent-looking but wrong output, with nothing missing from the load report.
            rope_theta=rope_theta,
            **kwargs,
        )


__all__ = ["InstellaMoEConfig"]
