#!/usr/bin/env bash
# Short-sequence smoke test for jointly training the understanding and
# generation LLM experts while freezing the vision encoding/decoding modules.
#
# Examples:
#   WP_SIZE=8 MODEL_NAME_OR_PATH=/models/u1 VOCAB_FILE=/models/qwen3-tokenizer \
#     bash shell/train_u1/8B_dual_expert_smoke.sh
#   WP_SIZE=4 NPROC_PER_NODE=4 MODEL_NAME_OR_PATH=/models/u1 VOCAB_FILE=/models/qwen3-tokenizer \
#     bash shell/train_u1/8B_dual_expert_smoke.sh

set -euo pipefail

cd "$(dirname "$0")/../.."

# ----------------------------- Distributed -----------------------------
export WP_SIZE="${WP_SIZE:-8}"
case "${WP_SIZE}" in
  4|8) ;;
  *)
    echo "WP_SIZE must be 4 or 8; got ${WP_SIZE}." >&2
    exit 1
    ;;
esac

export NPROC_PER_NODE="${NPROC_PER_NODE:-${WP_SIZE}}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
if [[ "${WORLD_SIZE}" -ne "${WP_SIZE}" ]]; then
  echo "wp=${WP_SIZE}, tp=1, pp=1 requires ${WP_SIZE} total ranks; got ${WORLD_SIZE}." >&2
  exit 1
fi

# ------------------------------ Model/data ------------------------------
export CONFIG_NAME="configs/sensenovavl_qwen3_gen/sensenovau1_8b_mot_sft.py"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/path/to/SenseNova-U1-8B-MoT-SFT}"
export VOCAB_FILE="${VOCAB_FILE:-/path/to/qwen3/tokenizer}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${VOCAB_FILE}}"
export mm_data_path="${MM_DATA_PATH:-data/sample/sample_data_meta.json}"
export load_optimizer="${LOAD_CONTENT:-model}"

if [[ ! -f "${mm_data_path}" ]]; then
  echo "Dataset meta not found: ${mm_data_path}" >&2
  exit 1
fi
if [[ "${MODEL_NAME_OR_PATH}" == /path/to/* ]]; then
  echo "Set MODEL_NAME_OR_PATH to the local SenseNova-U1-8B-MoT-SFT checkpoint." >&2
  exit 1
fi
if [[ "${VOCAB_FILE}" == /path/to/* ]]; then
  echo "Set VOCAB_FILE (or TOKENIZER_PATH) to the local Qwen3 tokenizer." >&2
  exit 1
fi

# ----------------------------- Parallelism -----------------------------
export zero1_size="${ZERO1_SIZE:--1}"
export wp_size="${WP_SIZE}"
export tp_size=1
export pp_size=1

# ----------------------------- Optimization ----------------------------
export SEED="${SEED:-42}"
export lr="${LR:-2e-5}"
export lr_scheduler_type="${LR_SCHEDULER_TYPE:-constant}"
export min_lr_ratio="${MIN_LR_RATIO:-0.5}"
export mlp_lr_scale=1.0
export mot_gen_lr_scale="${MOT_GEN_LR_SCALE:-1.0}"
export fm_modules_lr_scale="${FM_MODULES_LR_SCALE:-1.0}"
export weight_decay="${WEIGHT_DECAY:-0}"
export grad_accm="${GRAD_ACCM:-1}"
export total_steps="${TOTAL_STEPS:-20}"
export init_steps="${WARMUP_STEPS:-1}"

# -------------------------- Small packed samples ------------------------
export seq_len="${SEQ_LEN:-1024}"
export max_sample_tokens="${MAX_SAMPLE_TOKENS:-${seq_len}}"
export num_imgs="${NUM_IMGS:-4}"
export dataset_replacement=true
export min_num_frame=1
export max_num_frame="${MAX_NUM_FRAME:-1}"
export dynamic_image_version=native_resolution
export CONV_STYLE=sensenovalm2-chat-v3
export down_sample_ratio=0.5

# Keep understanding inputs and generation targets at exactly 256x256.
export max_pixels="${MAX_PIXELS:-65536}"
export min_pixels="${MIN_PIXELS:-65536}"
export max_pixels_gen="${MAX_PIXELS_GEN:-65536}"
export min_pixels_gen="${MIN_PIXELS_GEN:-65536}"
export LLM_DATA_WEIGHTS=0
export MM_CC_DATA_WEIGHTS=0

# ------------------------- Dual-expert training -------------------------
# Train both language experts, but freeze vision encoders, output head, and
# timestep/noise embedders. Gradients still flow through the frozen fm_head
# into the generation language expert.
export freeze_llm=false
export freeze_backbone=true
export freeze_mlp=true
export unfreeze_mot_gen=true
export freeze_vision_io=true
export train_buffer=false
export unfreeze_post_buffer=false
export enable_und_loss=true
export ce_loss_weight="${CE_LOSS_WEIGHT:-0.1}"

# The checkpoint already contains both MoT branches. EMA is intentionally off
# for this memory-focused smoke test.
export mot_random_init=false
export enable_ema=false
export ema_decay="${EMA_DECAY:-0.9999}"

# ----------------------- Flow-matching defaults -------------------------
export time_schedule=standard
export time_shift_type=exponential
export time_base_dist=logit_normal
export base_shift=0.5
export max_shift=1.15
export base_image_seq_len=64
export max_image_seq_len=4096
export noise_scale_mode=resolution
export noise_scale_base_image_seq_len=64
export add_noise_scale_embedding=true
export noise_scale_max_value=8
export P_mean=-0.8
export P_std=0.8
export cfg_txt_uncond_drop_prob="${CFG_TXT_DROP:-0.1}"
export cfg_img_uncond_drop_prob="${CFG_IMG_DROP:-0}"
export cfg_txtimg_uncond_drop_prob="${CFG_TXTIMG_DROP:-0.1}"
export cfg_is_uncond_drop_independent=false
export pad_dummy_image_gen=true
export thinking_method=tag

export JOB_NAME="${JOB_NAME:-sensenovau1_8b_dual_expert_smoke_wp${WP_SIZE}}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "Launching ${JOB_NAME}: wp=${wp_size}, seq_len=${seq_len}, num_imgs=${num_imgs}, pixels=${max_pixels}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train_sensenovau1.py \
  --config "${CONFIG_NAME}" \
  --launcher torch \
  --seed "${SEED}" \
  --backend nccl
