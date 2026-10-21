#!/bin/bash
# Instella-MoE RL setup on current Miles: run inside Miles' ROCm image
# (rocm/sgl-dev:miles-rocm724-mi35x-*, which ships the Miles, Megatron-LM and SGLang versions
# rl/miles pins). Applies Instella's three small patches; idempotent.
#   - rl/patches/megatron_farskip.patch -> the image's Megatron-LM (FarSkip layer, gated MLA)
#   - rl/patches/sglang_instella.patch  -> the image's SGLang (MLA arch list, rope-32 decode)
#   - rl/patches/miles_instella.patch   -> rl/miles (gate_proj weight sync)
# SGLang loads the model from rl/sglang_models (SGLANG_EXTERNAL_MODEL_PACKAGE=instella_sglang).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"

apply_patch() {  # repo patch
    if git -C "$1" apply --reverse --check "$2" 2>/dev/null; then
        log_ok "already applied: $(basename "$2")"
    else
        git -C "$1" apply "$2"
        log_ok "applied: $(basename "$2") -> $1"
    fi
}

log_step "patches" "Instella-MoE on Miles $(git -C "${MILES_DIR}" rev-parse --short HEAD)"
apply_patch "${MEGATRON_ROOT}" "${REPO_ROOT}/rl/patches/megatron_farskip.patch"
apply_patch "${SGLANG_ROOT}" "${REPO_ROOT}/rl/patches/sglang_instella.patch"
apply_patch "${MILES_DIR}" "${REPO_ROOT}/rl/patches/miles_instella.patch"
log_step "done" "export SGLANG_EXTERNAL_MODEL_PACKAGE=instella_sglang PYTHONPATH=${REPO_ROOT}/rl/sglang_models:\$PYTHONPATH"
