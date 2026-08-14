#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

CONDA_ENV="${CONDA_ENV:-0320}"
QUEUE_TAG="${QUEUE_TAG:-20260329}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export RECORDING_EVAL_EPISODES="${RECORDING_EVAL_EPISODES:-1}"
export CASE_RETRIES="${CASE_RETRIES:-2}"

echo "=== Supplement queue start ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DEVICE=${DEVICE}"

echo
echo "=== 1/3: single-step seed4 video eval from existing checkpoints ==="
CONDA_ENV="${CONDA_ENV}" \
SEED=4 \
DEVICE="${DEVICE}" \
SOURCE_SAVE_DIR="${PROJECT_ROOT}/logs/experiment_test" \
SOURCE_EXPERIMENT_NAME="task_dyn_true" \
TARGET_SAVE_DIR="${PROJECT_ROOT}/logs/task_dyn_true_video_seed4_${QUEUE_TAG}" \
TARGET_EXPERIMENT_NAME="task_dyn_true_video_seed4_${QUEUE_TAG}" \
TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_4/representation_cls_transformer_checkpoint.pth" \
"${PROJECT_ROOT}/scripts/run_existing_model_video_eval.sh"

echo
echo "=== 2/3: H=3 seed3 full run ==="
CONDA_ENV="${CONDA_ENV}" \
SEED=3 \
DEVICE="${DEVICE}" \
RUN_NAME="task_dyn_true_multistep_h3_seed3_${QUEUE_TAG}" \
EXPERIMENT_NAME="task_dyn_true_multistep_h3_seed3_${QUEUE_TAG}" \
SAVE_DIR="${PROJECT_ROOT}/logs/task_dyn_true_multistep_h3_seed3_${QUEUE_TAG}" \
ROLLOUT_HORIZON=3 \
ROLLOUT_DISCOUNT=1.0 \
SAVE_VIDEO=True \
TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_3/representation_cls_transformer_checkpoint.pth" \
"${PROJECT_ROOT}/scripts/run_existing_buffer_variant.sh"

echo
echo "=== 3/3: H=3 seed4 full run ==="
CONDA_ENV="${CONDA_ENV}" \
SEED=4 \
DEVICE="${DEVICE}" \
RUN_NAME="task_dyn_true_multistep_h3_seed4_${QUEUE_TAG}" \
EXPERIMENT_NAME="task_dyn_true_multistep_h3_seed4_${QUEUE_TAG}" \
SAVE_DIR="${PROJECT_ROOT}/logs/task_dyn_true_multistep_h3_seed4_${QUEUE_TAG}" \
ROLLOUT_HORIZON=3 \
ROLLOUT_DISCOUNT=1.0 \
SAVE_VIDEO=True \
TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_4/representation_cls_transformer_checkpoint.pth" \
"${PROJECT_ROOT}/scripts/run_existing_buffer_variant.sh"

echo
echo "Supplement queue finished."
