#!/usr/bin/env bash
set -eu

###############################################
# 汎用グリッドサーチラッパー for train.py (1GPU / 多GPU 対応版)
#
# 使い方例 (1GPU):
#   ./grid_search_train_wrapper.sh \
#     --train_script /path/to/train.py \
#     --data_path    /path/to/dataset_subset20 \
#     --model_name_or_path zhihan1996/DNABERT-2-117M \
#     --output_root  /path/to/output_grid_subset20 \
#     --run_name_prefix DNABERT2_fd1_subset20 \
#     --nproc_per_node 1 \
#     --extra_args "--kmer -1 --model_max_length 10 --fp16 --find_unused_parameters False"
#
# 使い方例 (4GPU):
#   ./grid_search_train_wrapper.sh \
#     --train_script /path/to/train.py \
#     --data_path    /path/to/dataset_subset20 \
#     --model_name_or_path zhihan1996/DNABERT-2-117M \
#     --output_root  /path/to/output_grid_subset20 \
#     --run_name_prefix DNABERT2_fd1_subset20_4gpu \
#     --nproc_per_node 4 \
#     --extra_args "--kmer -1 --model_max_length 10 --fp16 --find_unused_parameters False"
###############################################

# ========= デフォルト値 =========
NPROC_PER_NODE=1
RUN_NAME_PREFIX="gridrun"
OUTPUT_ROOT="./output"
DATA_PATH=""
MODEL_NAME_OR_PATH=""
TRAIN_SCRIPT="train.py"
LOG_ROOT="./grid_logs"

EXTRA_ARGS=""

# ========= 引数パース =========
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_script)
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --model_name_or_path)
      MODEL_NAME_OR_PATH="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run_name_prefix)
      RUN_NAME_PREFIX="$2"
      shift 2
      ;;
    --nproc_per_node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --extra_args)
      EXTRA_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "  (ヒント: train.py に渡したい引数は --extra_args \"...\" の中に書いてください)"
      exit 1
      ;;
  esac
done

if [[ -z "${DATA_PATH}" ]]; then
  echo "ERROR: --data_path is required."
  exit 1
fi
if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "ERROR: --model_name_or_path is required."
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${LOG_ROOT}"

WORLD_SIZE=${NPROC_PER_NODE}

# ========= train.csv 行数から TOTAL_TRAIN_EXAMPLES を算出 =========
if [[ -f "${DATA_PATH}/train.csv" ]]; then
  echo "Detecting train size from ${DATA_PATH}/train.csv ..."
  TOTAL_TRAIN_EXAMPLES=$(python - <<EOF
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("${DATA_PATH}") / "train.csv")
print(len(df))
EOF
)
else
  echo "ERROR: ${DATA_PATH}/train.csv not found. This wrapper assumes CSV (train/dev/test.csv) dataset."
  exit 1
fi

echo "TOTAL_TRAIN_EXAMPLES: ${TOTAL_TRAIN_EXAMPLES}"
echo "WORLD_SIZE (GPUs per node): ${WORLD_SIZE}"

# ========= グリッド候補 =========
# BATCH_CONFIGS: "<per_device_train_batch_size>x<gradient_accumulation_steps>"
# 有効バッチサイズ eff_batch = BS * GAS * WORLD_SIZE
BATCH_CONFIGS=(
  "8x8"    # eff_batch=64 x WORLD_SIZE
  "16x8"    # eff_batch=128 x WORLD_SIZE
  "32x8"   # eff_batch=256 x WORLD_SIZE
)

LRS=(
  "1e-5"
  "2e-5"
  "3e-5"
)

EPOCHS=(
  "2"
)

WARMUP_RATIOS=(
  "0.1"
)

WEIGHT_DECAYS=(
  "0.01"
  "0.03"
)

echo "Grid search settings:"
echo "  BATCH_CONFIGS  : ${BATCH_CONFIGS[*]}"
echo "  LRS            : ${LRS[*]}"
echo "  EPOCHS         : ${EPOCHS[*]}"
echo "  WARMUP_RATIOS  : ${WARMUP_RATIOS[*]}"
echo "  WEIGHT_DECAYS  : ${WEIGHT_DECAYS[*]}"
echo "  NPROC_PER_NODE : ${NPROC_PER_NODE}"
echo "  EXTRA_ARGS     : ${EXTRA_ARGS}"

# ========= ループ開始 =========
for cfg in "${BATCH_CONFIGS[@]}"; do
  IFS="x" read -r BS GAS <<< "${cfg}"

  eff_batch=$(( BS * GAS * WORLD_SIZE ))

  # 1 epoch あたりのステップ数（global eff_batch 基準）
  steps_per_epoch=$(( (TOTAL_TRAIN_EXAMPLES + eff_batch - 1) / eff_batch ))

  for lr in "${LRS[@]}"; do
    for epochs in "${EPOCHS[@]}"; do
      total_steps=$(( steps_per_epoch * epochs ))

      for warmup_ratio in "${WARMUP_RATIOS[@]}"; do
        # warmup_steps = total_steps * warmup_ratio
        warmup_steps=$(python - <<EOF
total_steps = ${total_steps}
warmup_ratio = ${warmup_ratio}
print(max(1, int(total_steps * warmup_ratio)))
EOF
)
        for wd in "${WEIGHT_DECAYS[@]}"; do

          RUN_NAME="${RUN_NAME_PREFIX}_bs${BS}_gas${GAS}_lr${lr}_ep${epochs}_wr${warmup_ratio}_wd${wd}_g${WORLD_SIZE}"
          OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

          echo "==============================================="
          echo "Running: ${RUN_NAME}"
          echo "  data_path        : ${DATA_PATH}"
          echo "  model_name       : ${MODEL_NAME_OR_PATH}"
          echo "  WORLD_SIZE       : ${WORLD_SIZE}"
          echo "  per_device_bs    : ${BS}"
          echo "  grad_accum       : ${GAS}"
          echo "  eff_batch        : ${eff_batch}"
          echo "  steps_per_epoch  : ${steps_per_epoch}"
          echo "  total_steps      : ${total_steps}"
          echo "  warmup_steps     : ${warmup_steps}"
          echo "==============================================="

          mkdir -p "${OUT_DIR}"

          torchrun --nproc_per_node=${NPROC_PER_NODE} "${TRAIN_SCRIPT}" \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --data_path  "${DATA_PATH}" \
            --run_name "${RUN_NAME}" \
            --per_device_train_batch_size ${BS} \
            --per_device_eval_batch_size $(( BS * 2 )) \
            --gradient_accumulation_steps ${GAS} \
            --learning_rate ${lr} \
            --num_train_epochs ${epochs} \
            --warmup_steps ${warmup_steps} \
            --weight_decay ${wd} \
            --output_dir "${OUT_DIR}" \
            --save_steps 0 \
            --evaluation_strategy steps \
            --eval_steps ${steps_per_epoch} \
            --logging_steps ${steps_per_epoch} \
            --overwrite_output_dir True \
            --log_level info \
            ${EXTRA_ARGS} \
            > "${LOG_ROOT}/${RUN_NAME}.log" 2>&1

        done
      done
    done
  done
done
