#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

if [[ -d "${PROJECT_ROOT}/mtenv_repo" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}/mtenv_repo${PYTHONPATH:+:${PYTHONPATH}}"
fi

CONDA_ENV="${CONDA_ENV:-0320}"
RUN_NAME="${RUN_NAME:-pace_variant}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-task_dyn_true_variant}"
SEED="${SEED:-5}"
DEVICE="${DEVICE:-cuda:0}"

SOURCE_SAVE_DIR="${SOURCE_SAVE_DIR:-${PROJECT_ROOT}/logs/experiment_test}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/logs/${RUN_NAME}}"
STAGE_LOG_DIR="${STAGE_LOG_DIR:-${SAVE_DIR}/stage_logs}"
BUFFER_SOURCE_DIR="${BUFFER_SOURCE_DIR:-${SOURCE_SAVE_DIR}/buffer/collective_buffer}"
DEFAULT_TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_${SEED}/representation_cls_transformer_checkpoint.pth"
if [[ ! -f "${DEFAULT_TRANSFORMER_PATH}" ]]; then
  DEFAULT_TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_5/representation_cls_transformer_checkpoint.pth"
fi
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${DEFAULT_TRANSFORMER_PATH}}"

ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-1}"
ROLLOUT_DISCOUNT="${ROLLOUT_DISCOUNT:-1.0}"
SAVE_VIDEO="${SAVE_VIDEO:-True}"
RECORDING_EVAL_EPISODES="${RECORDING_EVAL_EPISODES:-1}"
CASE_RETRIES="${CASE_RETRIES:-2}"

PA_PARENT_DIR="${SAVE_DIR}/model_dir/${EXPERIMENT_NAME}"
PA_SAVE_DIR="${PA_PARENT_DIR}/model_predictive_adapter_seed_${SEED}"
PA_ALIAS_DIR="${PA_PARENT_DIR}/predictive_adapter_seed_${SEED}"
COL_SAVE_DIR="${PA_PARENT_DIR}/model_col_seed_${SEED}"

if [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

EXTRA_MAIN_ARGS_STR="${EXTRA_MAIN_ARGS:-}"
EXTRA_MAIN_ARGS_ARRAY=()
if [[ -n "${EXTRA_MAIN_ARGS_STR}" ]]; then
  # Intentionally split on shell whitespace so callers can pass Hydra overrides.
  # Example:
  # EXTRA_MAIN_ARGS="transformer_collective_network.predictive_adapter.delta_state_loss_weight=0.0 experiment.early_stopping=False"
  read -r -a EXTRA_MAIN_ARGS_ARRAY <<< "${EXTRA_MAIN_ARGS_STR}"
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
    transformer_collective_network.predictive_adapter.rollout_horizon="${ROLLOUT_HORIZON}" \
    transformer_collective_network.predictive_adapter.rollout_discount="${ROLLOUT_DISCOUNT}" \
    transformer_collective_network.transformer_encoder.representation_transformer.model_path="${TRANSFORMER_PATH}" \
    transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="${TRANSFORMER_PATH}" \
    "${EXTRA_MAIN_ARGS_ARRAY[@]}" \
    "$@" 2>&1 | tee -a "${log_file}"
}

adapter_log="${STAGE_LOG_DIR}/01_predictive_adapter.log"
collective_log="${STAGE_LOG_DIR}/02_collective.log"
eval_log="${STAGE_LOG_DIR}/03_eval.log"

touch "${adapter_log}" "${collective_log}" "${eval_log}"

echo >> "${adapter_log}"
echo "===== $(date '+%F %T') variant run start =====" | tee -a "${adapter_log}"

echo "PROJECT_ROOT=${PROJECT_ROOT}" | tee -a "${adapter_log}"
echo "SAVE_DIR=${SAVE_DIR}" | tee -a "${adapter_log}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}" | tee -a "${adapter_log}"
echo "SEED=${SEED}" | tee -a "${adapter_log}"
echo "BUFFER_SOURCE_DIR=${BUFFER_SOURCE_DIR}" | tee -a "${adapter_log}"
echo "TRANSFORMER_PATH=${TRANSFORMER_PATH}" | tee -a "${adapter_log}"
echo "ROLLOUT_HORIZON=${ROLLOUT_HORIZON}" | tee -a "${adapter_log}"
echo "ROLLOUT_DISCOUNT=${ROLLOUT_DISCOUNT}" | tee -a "${adapter_log}"
echo "SAVE_VIDEO=${SAVE_VIDEO}" | tee -a "${adapter_log}"
echo "RECORDING_EVAL_EPISODES=${RECORDING_EVAL_EPISODES}" | tee -a "${adapter_log}"
echo "CASE_RETRIES=${CASE_RETRIES}" | tee -a "${adapter_log}"
echo "MUJOCO_GL=${MUJOCO_GL}" | tee -a "${adapter_log}"
echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}" | tee -a "${adapter_log}"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${adapter_log}"

pa_done=false
if [[ -f "${PA_SAVE_DIR}/metadata.pt" ]] && compgen -G "${PA_SAVE_DIR}/predictive_adapter_*.pt" > /dev/null; then
  pa_done=true
fi

if [[ "${pa_done}" == "true" ]]; then
  echo "[$(date '+%F %T')] Skipping predictive adapter training; checkpoint already present at ${PA_SAVE_DIR}" | tee -a "${adapter_log}"
else
  run_main "${adapter_log}" \
    experiment.mode=train_predictive_adapter \
    transformer_collective_network.predictive_adapter.load_on_init=False
fi

if [[ -d "${PA_SAVE_DIR}" && ! -e "${PA_ALIAS_DIR}" ]]; then
  ln -s "model_predictive_adapter_seed_${SEED}" "${PA_ALIAS_DIR}"
fi

col_done=false
if [[ -f "${COL_SAVE_DIR}/metadata.pt" ]] && compgen -G "${COL_SAVE_DIR}/actor_*.pt" > /dev/null; then
  col_done=true
fi

if [[ "${col_done}" == "true" ]]; then
  echo "[$(date '+%F %T')] Skipping collective distillation; checkpoint already present at ${COL_SAVE_DIR}" | tee -a "${collective_log}"
else
  run_main "${collective_log}" \
    experiment.mode=distill_collective_transformer \
    transformer_collective_network.predictive_adapter.pretrained_dir="${SAVE_DIR}/model_dir"
fi

(
  cd "${PROJECT_ROOT}"
  PROJECT_ROOT="${PROJECT_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  SEED="${SEED}" \
  TRANSFORMER_PATH="${TRANSFORMER_PATH}" \
  EXPECT_VIDEO="$([[ "${SAVE_VIDEO}" == "True" || "${SAVE_VIDEO}" == "true" ]] && echo 1 || echo 0)" \
  MUJOCO_GL="${MUJOCO_GL}" \
  PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}" \
  SAVE_DIR="${SAVE_DIR}" \
  CASE_RETRIES="${CASE_RETRIES}" \
  bash scripts/eval_collective_resumable.sh \
    setup.save_dir="${SAVE_DIR}" \
    setup.base_path="${PROJECT_ROOT}" \
    setup.device="${DEVICE}" \
    "experiment.save_video=${SAVE_VIDEO}" \
    "experiment.recording_eval_episodes=${RECORDING_EVAL_EPISODES}" \
    "transformer_collective_network.predictive_adapter.rollout_horizon=${ROLLOUT_HORIZON}" \
    "transformer_collective_network.predictive_adapter.rollout_discount=${ROLLOUT_DISCOUNT}" \
    "transformer_collective_network.predictive_adapter.pretrained_dir=${SAVE_DIR}/model_dir"
) 2>&1 | tee -a "${eval_log}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_collective_results.py" \
  --experiment "${EXPERIMENT_NAME}" \
  --seed "${SEED}" \
  --output "${SAVE_DIR}/evaluation_summary.json" \
  2>&1 | tee -a "${eval_log}"

echo
echo "Finished variant pipeline."
echo "SAVE_DIR=${SAVE_DIR}"
echo "PA_SAVE_DIR=${PA_SAVE_DIR}"
echo "PA_ALIAS_DIR=${PA_ALIAS_DIR}"
