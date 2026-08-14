#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

RUN_SAVE_DIR="${RUN_SAVE_DIR:-${PROJECT_ROOT}/logs/task_dyn_true_multistep_h3_seed5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "${RUN_SAVE_DIR}")}"
SEED="${SEED:-5}"
MODE="${MODE:-obs_noise}"
VALUES="${VALUES:-0.0,0.01,0.02,0.05}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RUN_SAVE_DIR}/analysis/${MODE}_sweep}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python"
fi

DEFAULT_TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_${SEED}/representation_cls_transformer_checkpoint.pth"
if [[ ! -f "${DEFAULT_TRANSFORMER_PATH}" ]]; then
  DEFAULT_TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_5/representation_cls_transformer_checkpoint.pth"
fi
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${DEFAULT_TRANSFORMER_PATH}}"

mkdir -p "${ANALYSIS_DIR}"

IFS=',' read -r -a SWEEP_VALUES <<< "${VALUES}"

prepare_point_workspace() {
  local point_save_dir="$1"
  local point_experiment_name="$2"

  mkdir -p "${point_save_dir}" "${point_save_dir}/model_dir" "${point_save_dir}/buffer"
  ln -sfn "${RUN_SAVE_DIR}/model_dir/${EXPERIMENT_NAME}" \
    "${point_save_dir}/model_dir/${point_experiment_name}"
  ln -sfn "${RUN_SAVE_DIR}/buffer/collective_buffer" \
    "${point_save_dir}/buffer/collective_buffer"
}

case "${MODE}" in
  obs_noise)
    override_key="experiment.obs_noise_std"
    prefix="obs_noise"
    ;;
  action_delay)
    override_key="experiment.action_delay_steps"
    prefix="action_delay"
    ;;
  action_noise)
    override_key="experiment.action_noise_std"
    prefix="action_noise"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use obs_noise, action_delay, or action_noise."
    exit 1
    ;;
esac

echo "RUN_SAVE_DIR=${RUN_SAVE_DIR}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "SEED=${SEED}"
echo "TRANSFORMER_PATH=${TRANSFORMER_PATH}"
echo "MODE=${MODE}"
echo "VALUES=${VALUES}"
echo "ANALYSIS_DIR=${ANALYSIS_DIR}"

for value in "${SWEEP_VALUES[@]}"; do
  echo
  echo "=== Evaluating ${MODE}=${value} ==="
  safe_value="${value//./p}"
  point_experiment_name="${EXPERIMENT_NAME}_${prefix}_${safe_value}"
  point_save_dir="${RUN_SAVE_DIR}/eval_runs/${prefix}_${safe_value}"
  prepare_point_workspace "${point_save_dir}" "${point_experiment_name}"

  (
    cd "${PROJECT_ROOT}"
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    EXPERIMENT_NAME="${point_experiment_name}" \
    SEED="${SEED}" \
    TRANSFORMER_PATH="${TRANSFORMER_PATH}" \
    SAVE_DIR="${point_save_dir}" \
    EXPECT_VIDEO=0 \
    bash scripts/eval_collective_resumable.sh \
      setup.save_dir="${point_save_dir}" \
      setup.base_path="${PROJECT_ROOT}" \
      "experiment.save_video=${SAVE_VIDEO}" \
      "${override_key}=${value}" \
      "transformer_collective_network.predictive_adapter.pretrained_dir=${point_save_dir}/model_dir"
  )

  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_collective_results.py" \
    --experiment "${point_experiment_name}" \
    --seed "${SEED}" \
    --output "${ANALYSIS_DIR}/${prefix}_${safe_value}.json"
done

echo
echo "Robustness sweep finished."
echo "Summaries saved under: ${ANALYSIS_DIR}"
