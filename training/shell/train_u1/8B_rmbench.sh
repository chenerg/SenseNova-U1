#!/usr/bin/env bash
# SenseNova-U1 8B fine-tuning launcher for RMBench mm_video_gen data.
#
# Single-node example:
#   MODEL_NAME_OR_PATH=/models/SenseNova-U1-8B-MoT-SFT \
#   VOCAB_FILE=/models/SenseNova-U1-8B-MoT-SFT \
#   MM_DATA_PATH=/datasets/RMBench/generated/rmbench_mm_video_gen_meta.json \
#   bash shell/train_u1/8B_rmbench.sh
#
set -euo pipefail

cd "$(dirname "$0")/../.."

# ----------------------------- Distributed -----------------------------
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# ------------------------------ Model/data ------------------------------
export CONFIG_NAME="configs/sensenovavl_qwen3_gen/sensenovau1_8b_mot_sft.py"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/path/to/SenseNova-U1-8B-MoT-SFT}"
export VOCAB_FILE="${VOCAB_FILE:-/path/to/SenseNova-U1-8B-MoT-SFT}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${VOCAB_FILE}}"
export mm_data_path="${MM_DATA_PATH:-}"
export load_optimizer="${LOAD_CONTENT:-model}"

if [[ "${MODEL_NAME_OR_PATH}" == /path/to/* || ! -d "${MODEL_NAME_OR_PATH}" ]]; then
  echo "Set MODEL_NAME_OR_PATH to the local SenseNova-U1-8B-MoT-SFT directory." >&2
  exit 1
fi
if [[ "${VOCAB_FILE}" == /path/to/* || ! -d "${VOCAB_FILE}" ]]; then
  echo "Set VOCAB_FILE (or TOKENIZER_PATH) to the local tokenizer directory." >&2
  exit 1
fi
if [[ -z "${mm_data_path}" || ! -f "${mm_data_path}" ]]; then
  echo "Set MM_DATA_PATH to rmbench_mm_video_gen_meta.json." >&2
  exit 1
fi

# To resume an InternEvo checkpoint, set MODEL_ONLY_FOLDER and use:
#   LOAD_CONTENT=all AUTO_RESUME=true RESUME_DS=true
export auto_resume="${AUTO_RESUME:-false}"
export resume_ds="${RESUME_DS:-false}"

# ----------------------------- Parallelism -----------------------------
export zero1_size=-1
export wp_size="${WP_SIZE:-8}"
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
export total_steps="${TOTAL_STEPS:-2000}"
export init_steps="${WARMUP_STEPS:-100}"

# -------------------------- RMBench video data --------------------------
export seq_len="${SEQ_LEN:-28672}"
export max_sample_tokens="${MAX_SAMPLE_TOKENS:-${seq_len}}"
export num_imgs="${NUM_IMGS:-144}"
export dataset_replacement=true
export min_num_frame=1
export max_num_frame="${MAX_NUM_FRAME:-128}"
# Limit mm_video_gen to eight logical frames. Intermediate teacher-forcing
# duplicates are added later by the dataset.
export max_num_frame_gen="${MAX_NUM_FRAME_GEN:-8}"
export dynamic_image_version=native_resolution
export CONV_STYLE=sensenovalm2-chat-v3
export down_sample_ratio=0.5

# Understanding and generation inputs both use native resolution in
# [256x256, 512x512].
export max_pixels="${MAX_PIXELS:-262144}"
export min_pixels="${MIN_PIXELS:-65536}"
export max_pixels_gen="${MAX_PIXELS_GEN:-262144}"
export min_pixels_gen="${MIN_PIXELS_GEN:-65536}"
export LLM_DATA_WEIGHTS=0
export MM_CC_DATA_WEIGHTS=0

# ------------------------- Trainable/frozen modules ---------------------
# Train everything except the understanding vision model and the
# language-model output head.
export freeze_llm=false
export freeze_backbone=true
export freeze_mlp=false
export unfreeze_mot_gen=true
export freeze_vision_io=false
export freeze_lm_head=true
export train_buffer=false
export unfreeze_post_buffer=false

# The SFT checkpoint already contains the MoT generation branch.
export mot_random_init=false

# EMA is disabled by default for RMBench fine-tuning.
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

# ---------------------------- Understanding ----------------------------
export pad_dummy_image_gen=true
export ce_loss_weight="${CE_LOSS_WEIGHT:-0.1}"
export enable_und_loss=true
export thinking_method=tag

# ----------------------------- Job/logging ------------------------------
export JOB_NAME="${JOB_NAME:-sensenovau1_8b_rmbench_video_sft}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "Launching ${JOB_NAME}: wp=${wp_size}, seq_len=${seq_len}, num_imgs=${num_imgs}, max_num_frame_gen=${max_num_frame_gen}"
echo "RMBench meta: ${mm_data_path}"

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
