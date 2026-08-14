#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-0320}"
SEED="${SEED:-5}"
DEVICE="${DEVICE:-cuda:0}"
SAVE_VIDEO="${SAVE_VIDEO:-True}"
RECORDING_EVAL_EPISODES="${RECORDING_EVAL_EPISODES:-1}"
ROLLOUT_DISCOUNT="${ROLLOUT_DISCOUNT:-1.0}"

run_variant() {
  local run_name="$1"
  local exp_name="$2"
  local horizon="$3"

  RUN_NAME="${run_name}" \
  EXPERIMENT_NAME="${exp_name}" \
  SAVE_DIR="${PROJECT_ROOT}/logs/${run_name}" \
  CONDA_ENV="${CONDA_ENV}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
  SAVE_VIDEO="${SAVE_VIDEO}" \
  RECORDING_EVAL_EPISODES="${RECORDING_EVAL_EPISODES}" \
  ROLLOUT_HORIZON="${horizon}" \
  ROLLOUT_DISCOUNT="${ROLLOUT_DISCOUNT}" \
  "${PROJECT_ROOT}/scripts/run_existing_buffer_variant.sh"
}

baseline_run="task_dyn_true_singlestep_seed${SEED}_${STAMP}"
baseline_exp="${baseline_run}"
h3_run="task_dyn_true_multistep_h3_seed${SEED}_${STAMP}"
h3_exp="${h3_run}"

echo "=== Baseline run ==="
echo "RUN_NAME=${baseline_run}"
echo "EXPERIMENT_NAME=${baseline_exp}"
run_variant "${baseline_run}" "${baseline_exp}" "1"

echo
echo "=== H=3 run ==="
echo "RUN_NAME=${h3_run}"
echo "EXPERIMENT_NAME=${h3_exp}"
run_variant "${h3_run}" "${h3_exp}" "3"

echo
echo "Both runs finished."
echo "Baseline save dir: ${PROJECT_ROOT}/logs/${baseline_run}"
echo "H3 save dir: ${PROJECT_ROOT}/logs/${h3_run}"
