#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

CONDA_ENV="${CONDA_ENV:-0320}"
RUN_NAME="${RUN_NAME:-multistep_h5_seed5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-task_dyn_true_multistep_h5_seed5}"
SEED="${SEED:-5}"
DEVICE="${DEVICE:-cuda:0}"

SOURCE_SAVE_DIR="${SOURCE_SAVE_DIR:-${PROJECT_ROOT}/logs/experiment_test}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/logs/${RUN_NAME}}"
STAGE_LOG_DIR="${STAGE_LOG_DIR:-${SAVE_DIR}/stage_logs}"
BUFFER_SOURCE_DIR="${BUFFER_SOURCE_DIR:-${SOURCE_SAVE_DIR}/buffer/collective_buffer}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_5/representation_cls_transformer_checkpoint.pth}"

PA_PARENT_DIR="${SAVE_DIR}/model_dir/${EXPERIMENT_NAME}"
PA_SAVE_DIR="${PA_PARENT_DIR}/model_predictive_adapter_seed_${SEED}"
PA_ALIAS_DIR="${PA_PARENT_DIR}/predictive_adapter_seed_${SEED}"

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

mkdir -p "${SAVE_DIR}" "${STAGE_LOG_DIR}" "${SAVE_DIR}/buffer"

if [[ ! -e "${SAVE_DIR}/buffer/collective_buffer" ]]; then
  ln -s "${BUFFER_SOURCE_DIR}" "${SAVE_DIR}/buffer/collective_buffer"
fi

run_main() {
  local log_file="$1"
  shift
  echo
  echo "[$(date '+%F %T')] $*" | tee -a "${log_file}"
  "${PYTHON_BIN}" -u "${PROJECT_ROOT}/main.py" \
    setup=metaworld \
    env=metaworld-mt1 \
    worker.multitask.num_envs=1 \
    setup.base_path="${PROJECT_ROOT}" \
    setup.save_dir="${SAVE_DIR}" \
    setup.device="${DEVICE}" \
    setup.seed="${SEED}" \
    experiment.experiment="${EXPERIMENT_NAME}" \
    logger.use_tb=True \
    transformer_collective_network.transformer_encoder.representation_transformer.model_path="${TRANSFORMER_PATH}" \
    transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="${TRANSFORMER_PATH}" \
    "$@" 2>&1 | tee -a "${log_file}"
}

adapter_log="${STAGE_LOG_DIR}/01_predictive_adapter.log"
collective_log="${STAGE_LOG_DIR}/02_collective.log"
eval_log="${STAGE_LOG_DIR}/03_eval.log"

: > "${adapter_log}"
: > "${collective_log}"
: > "${eval_log}"

echo "PROJECT_ROOT=${PROJECT_ROOT}" | tee -a "${adapter_log}"
echo "SAVE_DIR=${SAVE_DIR}" | tee -a "${adapter_log}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}" | tee -a "${adapter_log}"
echo "SEED=${SEED}" | tee -a "${adapter_log}"
echo "BUFFER_SOURCE_DIR=${BUFFER_SOURCE_DIR}" | tee -a "${adapter_log}"
echo "TRANSFORMER_PATH=${TRANSFORMER_PATH}" | tee -a "${adapter_log}"
echo "MUJOCO_GL=${MUJOCO_GL}" | tee -a "${adapter_log}"
echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}" | tee -a "${adapter_log}"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${adapter_log}"

run_main "${adapter_log}" \
  experiment.mode=train_predictive_adapter \
  transformer_collective_network.predictive_adapter.load_on_init=False

if [[ -d "${PA_SAVE_DIR}" && ! -e "${PA_ALIAS_DIR}" ]]; then
  ln -s "model_predictive_adapter_seed_${SEED}" "${PA_ALIAS_DIR}"
fi

run_main "${collective_log}" \
  experiment.mode=distill_collective_transformer \
  transformer_collective_network.predictive_adapter.pretrained_dir="${SAVE_DIR}/model_dir"

(
  cd "${PROJECT_ROOT}"
  PROJECT_ROOT="${PROJECT_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  SEED="${SEED}" \
  TRANSFORMER_PATH="${TRANSFORMER_PATH}" \
  MUJOCO_GL="${MUJOCO_GL}" \
  PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}" \
  bash eval.sh \
    setup.save_dir="${SAVE_DIR}" \
    setup.base_path="${PROJECT_ROOT}" \
    setup.device="${DEVICE}" \
    "transformer_collective_network.predictive_adapter.pretrained_dir=${SAVE_DIR}/model_dir"
) 2>&1 | tee -a "${eval_log}"

echo
echo "Finished multistep pipeline."
echo "SAVE_DIR=${SAVE_DIR}"
echo "PA_SAVE_DIR=${PA_SAVE_DIR}"
echo "PA_ALIAS_DIR=${PA_ALIAS_DIR}"
