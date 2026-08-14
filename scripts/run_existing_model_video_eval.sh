#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

CONDA_ENV="${CONDA_ENV:-0320}"
SOURCE_SAVE_DIR="${SOURCE_SAVE_DIR:-${PROJECT_ROOT}/logs/experiment_test}"
SOURCE_EXPERIMENT_NAME="${SOURCE_EXPERIMENT_NAME:-task_dyn_true}"
TARGET_SAVE_DIR="${TARGET_SAVE_DIR:-${PROJECT_ROOT}/logs/${SOURCE_EXPERIMENT_NAME}_video_seed${SEED:-3}}"
TARGET_EXPERIMENT_NAME="${TARGET_EXPERIMENT_NAME:-${SOURCE_EXPERIMENT_NAME}_video_seed${SEED:-3}}"
SEED="${SEED:-3}"
DEVICE="${DEVICE:-cuda:0}"
RECORDING_EVAL_EPISODES="${RECORDING_EVAL_EPISODES:-1}"
CASE_RETRIES="${CASE_RETRIES:-2}"

DEFAULT_TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_${SEED}/representation_cls_transformer_checkpoint.pth"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${DEFAULT_TRANSFORMER_PATH}}"

if [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "${TARGET_SAVE_DIR}/model_dir"

if [[ ! -e "${TARGET_SAVE_DIR}/model_dir/${TARGET_EXPERIMENT_NAME}" ]]; then
  ln -s "${SOURCE_SAVE_DIR}/model_dir/${SOURCE_EXPERIMENT_NAME}" \
    "${TARGET_SAVE_DIR}/model_dir/${TARGET_EXPERIMENT_NAME}"
fi

(
  cd "${PROJECT_ROOT}"
  PROJECT_ROOT="${PROJECT_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  EXPERIMENT_NAME="${TARGET_EXPERIMENT_NAME}" \
  SEED="${SEED}" \
  TRANSFORMER_PATH="${TRANSFORMER_PATH}" \
  EXPECT_VIDEO=1 \
  MUJOCO_GL="${MUJOCO_GL}" \
  PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}" \
  SAVE_DIR="${TARGET_SAVE_DIR}" \
  CASE_RETRIES="${CASE_RETRIES}" \
  bash scripts/eval_collective_resumable.sh \
    setup.save_dir="${TARGET_SAVE_DIR}" \
    setup.base_path="${PROJECT_ROOT}" \
    setup.device="${DEVICE}" \
    experiment.save_video=True \
    "experiment.recording_eval_episodes=${RECORDING_EVAL_EPISODES}" \
    transformer_collective_network.predictive_adapter.rollout_horizon=1 \
    transformer_collective_network.predictive_adapter.rollout_discount=1.0 \
    "transformer_collective_network.predictive_adapter.pretrained_dir=${TARGET_SAVE_DIR}/model_dir"
)

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_collective_results.py" \
  --experiment "${TARGET_EXPERIMENT_NAME}" \
  --seed "${SEED}" \
  --output "${TARGET_SAVE_DIR}/evaluation_summary.json"

echo "Finished video eval."
echo "TARGET_SAVE_DIR=${TARGET_SAVE_DIR}"
echo "TARGET_EXPERIMENT_NAME=${TARGET_EXPERIMENT_NAME}"
