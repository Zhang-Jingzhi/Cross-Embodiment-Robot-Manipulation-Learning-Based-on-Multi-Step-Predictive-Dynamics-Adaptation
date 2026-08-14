#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

RUN_SAVE_DIR="${RUN_SAVE_DIR:-${PROJECT_ROOT}/logs/task_dyn_true_multistep_h3_seed5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "${RUN_SAVE_DIR}")}"
SEED="${SEED:-5}"
CONTEXT_STEPS="${CONTEXT_STEPS:-0,1,3,5,10,20}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RUN_SAVE_DIR}/analysis/context_sweep}"
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

IFS=',' read -r -a CONTEXT_VALUES <<< "${CONTEXT_STEPS}"

prepare_point_workspace() {
  local point_save_dir="$1"
  local point_experiment_name="$2"

  mkdir -p "${point_save_dir}" "${point_save_dir}/model_dir" "${point_save_dir}/buffer"
  ln -sfn "${RUN_SAVE_DIR}/model_dir/${EXPERIMENT_NAME}" \
    "${point_save_dir}/model_dir/${point_experiment_name}"
  ln -sfn "${RUN_SAVE_DIR}/buffer/collective_buffer" \
    "${point_save_dir}/buffer/collective_buffer"
}

echo "RUN_SAVE_DIR=${RUN_SAVE_DIR}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "SEED=${SEED}"
echo "TRANSFORMER_PATH=${TRANSFORMER_PATH}"
echo "CONTEXT_STEPS=${CONTEXT_STEPS}"
echo "ANALYSIS_DIR=${ANALYSIS_DIR}"

for context_steps in "${CONTEXT_VALUES[@]}"; do
  echo
  echo "=== Evaluating context_steps=${context_steps} ==="
  point_experiment_name="${EXPERIMENT_NAME}_context_${context_steps}"
  point_save_dir="${RUN_SAVE_DIR}/eval_runs/context_${context_steps}"
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
      "experiment.eval_history_steps=${context_steps}" \
      "transformer_collective_network.predictive_adapter.pretrained_dir=${point_save_dir}/model_dir"
  )

  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_collective_results.py" \
    --experiment "${point_experiment_name}" \
    --seed "${SEED}" \
    --output "${ANALYSIS_DIR}/context_${context_steps}.json"
done

echo
echo "Context sweep finished."
echo "Summaries saved under: ${ANALYSIS_DIR}"
