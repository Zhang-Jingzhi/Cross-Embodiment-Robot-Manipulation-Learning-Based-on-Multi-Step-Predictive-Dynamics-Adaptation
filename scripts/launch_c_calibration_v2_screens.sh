#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

CONDA_ENV="${CONDA_ENV:-jingzhizhang0320}"
CALIBRATION_TAG="${CALIBRATION_TAG:-$(date +%Y%m%d_%H%M)}"
MODE="${MODE:-robot_joint_damping}"
BASE_GPU="${BASE_GPU:-0}"
DELTA_GPU="${DELTA_GPU:-1}"
TARGET_ROBOTS="${TARGET_ROBOTS:-panda,kuka,ur5e,viperx}"
VALUES="${VALUES:-}"
CASE_RETRIES="${CASE_RETRIES:-2}"

BASE_RUN_SAVE_DIR="${BASE_RUN_SAVE_DIR:-${PROJECT_ROOT}/logs/task_dyn_true_singlestep_seed5_20260325_153611}"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-task_dyn_true_singlestep_seed5_20260325_153611}"
DELTA_RUN_SAVE_DIR="${DELTA_RUN_SAVE_DIR:-${PROJECT_ROOT}/logs/task_dyn_true_multistep_h3_deltafix_seed5_20260402}"
DELTA_EXPERIMENT_NAME="${DELTA_EXPERIMENT_NAME:-task_dyn_true_multistep_h3_deltafix_seed5_20260402}"

base_screen="c2_base_${MODE}_${CALIBRATION_TAG}"
delta_screen="c2_delta_${MODE}_${CALIBRATION_TAG}"
base_log="${PROJECT_ROOT}/logs/${base_screen}.log"
delta_log="${PROJECT_ROOT}/logs/${delta_screen}.log"

base_cmd=$(printf '%s\n' \
"cd ${PROJECT_ROOT}" \
"source /opt/conda/etc/profile.d/conda.sh" \
"conda activate ${CONDA_ENV}" \
"export PROJECT_ROOT=${PROJECT_ROOT}" \
"export PYTHONPATH=\"${PROJECT_ROOT}/mtenv_repo:\${PYTHONPATH:-}\"" \
"export MUJOCO_GL=egl" \
"export PYOPENGL_PLATFORM=egl" \
"export CUDA_VISIBLE_DEVICES=${BASE_GPU}" \
"export RUN_SAVE_DIR=\"${BASE_RUN_SAVE_DIR}\"" \
"export EXPERIMENT_NAME=${BASE_EXPERIMENT_NAME}" \
"export SOURCE_EXPERIMENT_NAME=${BASE_EXPERIMENT_NAME}" \
"export MODE=${MODE}" \
"export CALIBRATION_TAG=${CALIBRATION_TAG}" \
"export TARGET_ROBOTS=${TARGET_ROBOTS}" \
"export CASE_RETRIES=${CASE_RETRIES}" \
"export SAVE_VIDEO=False" \
"export ANALYSIS_DIR=\"${BASE_RUN_SAVE_DIR}/analysis/${MODE}_calibv2_${CALIBRATION_TAG}\"" \
"$( [[ -n "${VALUES}" ]] && printf 'export VALUES=%s' "${VALUES}" )" \
"bash scripts/run_c_calibration_v2.sh > \"${base_log}\" 2>&1")

delta_cmd=$(printf '%s\n' \
"cd ${PROJECT_ROOT}" \
"source /opt/conda/etc/profile.d/conda.sh" \
"conda activate ${CONDA_ENV}" \
"export PROJECT_ROOT=${PROJECT_ROOT}" \
"export PYTHONPATH=\"${PROJECT_ROOT}/mtenv_repo:\${PYTHONPATH:-}\"" \
"export MUJOCO_GL=egl" \
"export PYOPENGL_PLATFORM=egl" \
"export CUDA_VISIBLE_DEVICES=${DELTA_GPU}" \
"export RUN_SAVE_DIR=\"${DELTA_RUN_SAVE_DIR}\"" \
"export EXPERIMENT_NAME=${DELTA_EXPERIMENT_NAME}" \
"export SOURCE_EXPERIMENT_NAME=${DELTA_EXPERIMENT_NAME}" \
"export MODE=${MODE}" \
"export CALIBRATION_TAG=${CALIBRATION_TAG}" \
"export TARGET_ROBOTS=${TARGET_ROBOTS}" \
"export CASE_RETRIES=${CASE_RETRIES}" \
"export SAVE_VIDEO=False" \
"export ANALYSIS_DIR=\"${DELTA_RUN_SAVE_DIR}/analysis/${MODE}_calibv2_${CALIBRATION_TAG}\"" \
"$( [[ -n "${VALUES}" ]] && printf 'export VALUES=%s' "${VALUES}" )" \
"bash scripts/run_c_calibration_v2.sh > \"${delta_log}\" 2>&1")

screen -dmS "${base_screen}" bash -lc "${base_cmd}"
screen -dmS "${delta_screen}" bash -lc "${delta_cmd}"

echo "CALIBRATION_TAG=${CALIBRATION_TAG}"
echo "BASE_SCREEN=${base_screen}"
echo "DELTA_SCREEN=${delta_screen}"
echo "BASE_LOG=${base_log}"
echo "DELTA_LOG=${delta_log}"
