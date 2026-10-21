"""Smoke test for the FarSkip Megatron patch: tiny Instella-like GPT (MLA + gated attention +
MoE + SimpleFarSkip), forward/backward with and without full recompute must match."""
import os, torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.transformer_layer import SimpleFarSkipTransformerLayer

torch.distributed.init_process_group("nccl")
torch.cuda.set_device(0)
parallel_state.initialize_model_parallel(1, 1)
model_parallel_cuda_manual_seed(1234)

def build(recompute):
    torch.manual_seed(0); model_parallel_cuda_manual_seed(1234)
    cfg = MLATransformerConfig(
        num_layers=3, hidden_size=256, num_attention_heads=4, ffn_hidden_size=512,
        multi_latent_attention=True, kv_lora_rank=64, qk_head_dim=32, qk_pos_emb_head_dim=16,
        v_head_dim=32, qk_layernorm=True, normalization="RMSNorm", gated_attention=True,
        use_simple_farskip_layer=True, num_moe_experts=4, moe_router_topk=2,
        moe_ffn_hidden_size=128, moe_shared_expert_intermediate_size=128, moe_layer_freq=[0, 1, 1],
        moe_token_dispatcher_type="alltoall", moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True, moe_router_dtype="fp32", moe_grouped_gemm=True,
        add_bias_linear=False, gated_linear_unit=True, bf16=True, params_dtype=torch.bfloat16,
        rope_type="rope", rotary_base=10000,
        hidden_dropout=0.0, attention_dropout=0.0,
        recompute_granularity="full" if recompute else None,
        recompute_method="uniform" if recompute else None,
        recompute_num_layers=1 if recompute else None,
    )
    spec = get_gpt_decoder_block_spec(cfg, use_transformer_engine=True)
    assert all(s.module is SimpleFarSkipTransformerLayer for s in spec.layer_specs), spec
    m = GPTModel(cfg, spec, vocab_size=512, max_sequence_length=64,
                 position_embedding_type="none").cuda()
    return m

tokens = torch.randint(0, 512, (2, 32), device="cuda")
pos = torch.arange(32, device="cuda").unsqueeze(0).expand(2, -1)
results = []
for recompute in (False, True):
    m = build(recompute); m.train()
    out = m(tokens, pos, attention_mask=None, labels=tokens)
    loss = out.float().mean(); loss.backward()
    gsum = sum(p.grad.float().abs().sum().item() for p in m.parameters() if p.grad is not None)
    gate = [n for n, _ in m.named_parameters() if "linear_gate_proj" in n]
    print(f"recompute={recompute} loss={loss.item():.6f} gradsum={gsum:.4f} gate_params={len(gate)}")
    results.append((loss.item(), gsum))
assert abs(results[0][0] - results[1][0]) < 1e-3, results
assert abs(results[0][1] - results[1][1]) / results[0][1] < 1e-2, results
print("FARSKIP_TEST_OK")
