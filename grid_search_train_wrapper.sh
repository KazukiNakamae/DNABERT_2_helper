#!/usr/bin/env bash
set -eu

###############################################
# Generic grid search wrapper for train.py (1GPU / multi-GPU)
#
# Example (1 GPU, subset dataset):
#
#   ./grid_search_train_wrapper.sh \
#     --train_script /path/to/train.py \
#     --data_path    /path/to/dataset_subset20 \
#     --model_name_or_path zhihan1996/DNABERT-2-117M \
#     --output_root  /path/to/output_grid_subset20 \
#     --run_name_prefix DNABERT2_fd1_subset20 \
#     --nproc_per_node 1 \
#     --lrs "1e-5,2e-5,3e-5" \
#     --weight_decays "0.01,0.03" \
#     --extra_args "--kmer -1 --model_max_length 10 --fp16 --find_unused_parameters False"
#
# Example (4 GPUs, same grid):
#
#   ./grid_search_train_wrapper.sh \
#     --train_script /path/to/train.py \
#     --data_path    /path/to/dataset_subset20 \
#     --model_name_or_path zhihan1996/DNABERT-2-117M \
#     --output_root  /path/to/output_grid_subset20_4gpu \
#     --run_name_prefix DNABERT2_fd1_subset20_4gpu \
#     --nproc_per_node 4 \
#     --lrs "1e-5,2e-5,3e-5" \
#     --weight_decays "0.01,0.03" \
#     --extra_args "--kmer -1 --model_max_length 10 --fp16 --find_unused_parameters False"
###############################################

# ========= Default values =========
NPROC_PER_NODE=1
RUN_NAME_PREFIX="gridrun"
OUTPUT_ROOT="./output"
DATA_PATH=""
MODEL_NAME_OR_PATH=""
TRAIN_SCRIPT="train.py"
LOG_ROOT="./grid_logs"
EXTRA_ARGS=""

# Defaults for grid (can be overridden via CLI)
BATCH_CONFIGS_DEFAULT="64x2,128x2"  # "<per_device_bs>x<grad_accum>" (global eff_batch = bs * accum * WORLD_SIZE)
LRS_DEFAULT="1e-5,2e-5,3e-5"
EPOCHS_DEFAULT="2"
WARMUP_RATIOS_DEFAULT="0.1"
WEIGHT_DECAYS_DEFAULT="0.01,0.03"

# Placeholders for CLI overrides (comma-separated strings)
BATCH_CONFIGS_STR=""
LRS_STR=""
EPOCHS_STR=""
WARMUP_RATIOS_STR=""
WEIGHT_DECAYS_STR=""

# ========= Parse arguments =========
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
    --batch_configs)
      # e.g. --batch_configs "64x2,128x2,128x4"
      BATCH_CONFIGS_STR="$2"
      shift 2
      ;;
    --lrs)
      # e.g. --lrs "1e-5,2e-5,3e-5"
      LRS_STR="$2"
      shift 2
      ;;
    --epochs)
      # e.g. --epochs "2,4"
      EPOCHS_STR="$2"
      shift 2
      ;;
    --warmup_ratios)
      # e.g. --warmup_ratios "0.1,0.05"
      WARMUP_RATIOS_STR="$2"
      shift 2
      ;;
    --weight_decays)
      # e.g. --weight_decays "0.01,0.03"
      WEIGHT_DECAYS_STR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "  (Hint: arguments for train.py should go inside --extra_args \"...\" )"
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

# ========= Build arrays from defaults or CLI overrides =========

# batch configs
if [[ -n "${BATCH_CONFIGS_STR}" ]]; then
  IFS=',' read -r -a BATCH_CONFIGS <<< "${BATCH_CONFIGS_STR}"
else
  IFS=',' read -r -a BATCH_CONFIGS <<< "${BATCH_CONFIGS_DEFAULT}"
fi

# learning rates
if [[ -n "${LRS_STR}" ]]; then
  IFS=',' read -r -a LRS <<< "${LRS_STR}"
else
  IFS=',' read -r -a LRS <<< "${LRS_DEFAULT}"
fi

# epochs
if [[ -n "${EPOCHS_STR}" ]]; then
  IFS=',' read -r -a EPOCHS <<< "${EPOCHS_STR}"
else
  IFS=',' read -r -a EPOCHS <<< "${EPOCHS_DEFAULT}"
fi

# warmup ratios
if [[ -n "${WARMUP_RATIOS_STR}" ]]; then
  IFS=',' read -r -a WARMUP_RATIOS <<< "${WARMUP_RATIOS_STR}"
else
  IFS=',' read -r -a WARMUP_RATIOS <<< "${WARMUP_RATIOS_DEFAULT}"
fi

# weight decays
if [[ -n "${WEIGHT_DECAYS_STR}" ]]; then
  IFS=',' read -r -a WEIGHT_DECAYS <<< "${WEIGHT_DECAYS_STR}"
else
  IFS=',' read -r -a WEIGHT_DECAYS <<< "${WEIGHT_DECAYS_DEFAULT}"
fi

# ========= Detect train size from train.csv =========
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
  echo "ERROR: ${DATA_PATH}/train.csv not found. This wrapper assumes CSV dataset (train/dev/test.csv)."
  exit 1
fi

echo "TOTAL_TRAIN_EXAMPLES: ${TOTAL_TRAIN_EXAMPLES}"
echo "WORLD_SIZE (GPUs per node): ${WORLD_SIZE}"

echo "Grid search settings:"
echo "  BATCH_CONFIGS  : ${BATCH_CONFIGS[*]}"
echo "  LRS            : ${LRS[*]}"
echo "  EPOCHS         : ${EPOCHS[*]}"
echo "  WARMUP_RATIOS  : ${WARMUP_RATIOS[*]}"
echo "  WEIGHT_DECAYS  : ${WEIGHT_DECAYS[*]}"
echo "  NPROC_PER_NODE : ${NPROC_PER_NODE}"
echo "  EXTRA_ARGS     : ${EXTRA_ARGS}"

# ========= Main grid loop =========
for cfg in "${BATCH_CONFIGS[@]}"; do
  IFS="x" read -r BS GAS <<< "${cfg}"

  eff_batch=$(( BS * GAS * WORLD_SIZE ))

  # steps per epoch based on global effective batch size
  steps_per_epoch=$(( (TOTAL_TRAIN_EXAMPLES + eff_batch - 1) / eff_batch ))

  for lr in "${LRS[@]}"; do
    for epochs in "${EPOCHS[@]}"; do
      total_steps=$(( steps_per_epoch * epochs ))

      for warmup_ratio in "${WARMUP_RATIOS[@]}"; do
        # compute warmup steps = total_steps * warmup_ratio
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
