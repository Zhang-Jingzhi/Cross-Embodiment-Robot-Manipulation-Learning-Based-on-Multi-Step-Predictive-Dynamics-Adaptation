#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-0320}"
BASELINE_SOURCE_DIR="${BASELINE_SOURCE_DIR:-${PROJECT_ROOT}/logs/task_dyn_true_singlestep_seed5_20260325_153611}"
TARGET_SOURCE_DIR="${TARGET_SOURCE_DIR:-${PROJECT_ROOT}/logs/task_dyn_true_multistep_h3_deltafix_seed5_20260402}"
BASELINE_SOURCE_EXPERIMENT="${BASELINE_SOURCE_EXPERIMENT:-$(basename "${BASELINE_SOURCE_DIR}")}"
TARGET_SOURCE_EXPERIMENT="${TARGET_SOURCE_EXPERIMENT:-$(basename "${TARGET_SOURCE_DIR}")}"
SEED="${SEED:-5}"
SUITE_TAG="${SUITE_TAG:-paper_eval_$(date +%Y%m%d_%H%M%S)}"
SUITE_ROOT="${SUITE_ROOT:-${PROJECT_ROOT}/logs/${SUITE_TAG}}"
BASELINE_VIDEO_DIR="${BASELINE_VIDEO_DIR:-${BASELINE_SOURCE_DIR}/video}"

mkdir -p "${SUITE_ROOT}"

prepare_eval_workspace() {
  local source_dir="$1"
  local source_experiment="$2"
  local target_dir="$3"
  local target_experiment="$4"

  mkdir -p "${target_dir}" "${target_dir}/model_dir" "${target_dir}/buffer"

  ln -sfn "${source_dir}/model_dir/${source_experiment}" \
    "${target_dir}/model_dir/${target_experiment}"
  ln -sfn "${source_dir}/buffer/collective_buffer" \
    "${target_dir}/buffer/collective_buffer"
}

launch_screen() {
  local name="$1"
  local command="$2"
  local log_file="${SUITE_ROOT}/${name}.screen.log"

  screen -dmS "${name}" bash -lc "${command} |& tee '${log_file}'"
  echo "${name} -> ${log_file}"
}

common_shell_prefix() {
  cat <<EOF
cd '${PROJECT_ROOT}'
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate '${CONDA_ENV}'
fi
export PROJECT_ROOT='${PROJECT_ROOT}'
export PYTHONPATH='${PROJECT_ROOT}/mtenv_repo'"${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL='egl'
export PYOPENGL_PLATFORM='egl'
EOF
}

BASELINE_CORE_DIR="${SUITE_ROOT}/baseline_core"
BASELINE_ROBUST_DIR="${SUITE_ROOT}/baseline_robust"
TARGET_CORE_DIR="${SUITE_ROOT}/deltafix_core"
TARGET_ROBUST_DIR="${SUITE_ROOT}/deltafix_robust"
TARGET_VIDEO_DIR="${SUITE_ROOT}/deltafix_video"
BASELINE_CORE_EXPERIMENT="${SUITE_TAG}_baseline_core_eval"
BASELINE_ROBUST_EXPERIMENT="${SUITE_TAG}_baseline_robust_eval"
TARGET_CORE_EXPERIMENT="${SUITE_TAG}_deltafix_core_eval"
TARGET_ROBUST_EXPERIMENT="${SUITE_TAG}_deltafix_robust_eval"
TARGET_VIDEO_EXPERIMENT="${SUITE_TAG}_deltafix_video_eval"

prepare_eval_workspace "${BASELINE_SOURCE_DIR}" "${BASELINE_SOURCE_EXPERIMENT}" "${BASELINE_CORE_DIR}" "${BASELINE_CORE_EXPERIMENT}"
prepare_eval_workspace "${BASELINE_SOURCE_DIR}" "${BASELINE_SOURCE_EXPERIMENT}" "${BASELINE_ROBUST_DIR}" "${BASELINE_ROBUST_EXPERIMENT}"
prepare_eval_workspace "${TARGET_SOURCE_DIR}" "${TARGET_SOURCE_EXPERIMENT}" "${TARGET_CORE_DIR}" "${TARGET_CORE_EXPERIMENT}"
prepare_eval_workspace "${TARGET_SOURCE_DIR}" "${TARGET_SOURCE_EXPERIMENT}" "${TARGET_ROBUST_DIR}" "${TARGET_ROBUST_EXPERIMENT}"

BASELINE_CORE_CMD="$(common_shell_prefix)
export CUDA_VISIBLE_DEVICES=0
RUN_SAVE_DIR='${BASELINE_CORE_DIR}' EXPERIMENT_NAME='${BASELINE_CORE_EXPERIMENT}' SEED='${SEED}' \\
  ANALYSIS_DIR='${BASELINE_CORE_DIR}/analysis/context_sweep' \\
  bash scripts/run_eval_context_sweep.sh
RUN_SAVE_DIR='${BASELINE_CORE_DIR}' EXPERIMENT_NAME='${BASELINE_CORE_EXPERIMENT}' SEED='${SEED}' \\
  ANALYSIS_DIR='${BASELINE_CORE_DIR}/analysis/step_budget_sweep' \\
  bash scripts/run_eval_step_budget_sweep.sh
python scripts/evaluate_pa_k_step_error.py \\
  --project-root '${PROJECT_ROOT}' \\
  --device cuda \\
  --eval-horizon 5 \\
  --num-samples 32768 \\
  --run baseline='${BASELINE_SOURCE_DIR}':'${SEED}' \\
  --run deltafix='${TARGET_SOURCE_DIR}':'${SEED}' \\
  --output '${SUITE_ROOT}/analysis/pa_k_step_error.json'"

BASELINE_ROBUST_CMD="$(common_shell_prefix)
export CUDA_VISIBLE_DEVICES=0
RUN_SAVE_DIR='${BASELINE_ROBUST_DIR}' EXPERIMENT_NAME='${BASELINE_ROBUST_EXPERIMENT}' SEED='${SEED}' \\
  MODE=obs_noise VALUES='0.0,0.01,0.02,0.05' \\
  ANALYSIS_DIR='${BASELINE_ROBUST_DIR}/analysis/obs_noise_sweep' \\
  bash scripts/run_eval_robustness_sweep.sh
RUN_SAVE_DIR='${BASELINE_ROBUST_DIR}' EXPERIMENT_NAME='${BASELINE_ROBUST_EXPERIMENT}' SEED='${SEED}' \\
  MODE=action_delay VALUES='0,1,2,4' \\
  ANALYSIS_DIR='${BASELINE_ROBUST_DIR}/analysis/action_delay_sweep' \\
  bash scripts/run_eval_robustness_sweep.sh
RUN_SAVE_DIR='${BASELINE_ROBUST_DIR}' EXPERIMENT_NAME='${BASELINE_ROBUST_EXPERIMENT}' SEED='${SEED}' \\
  MODE=action_noise VALUES='0.0,0.01,0.02,0.05' \\
  ANALYSIS_DIR='${BASELINE_ROBUST_DIR}/analysis/action_noise_sweep' \\
  bash scripts/run_eval_robustness_sweep.sh"

TARGET_CORE_CMD="$(common_shell_prefix)
export CUDA_VISIBLE_DEVICES=1
RUN_SAVE_DIR='${TARGET_CORE_DIR}' EXPERIMENT_NAME='${TARGET_CORE_EXPERIMENT}' SEED='${SEED}' \\
  ANALYSIS_DIR='${TARGET_CORE_DIR}/analysis/context_sweep' \\
  bash scripts/run_eval_context_sweep.sh
RUN_SAVE_DIR='${TARGET_CORE_DIR}' EXPERIMENT_NAME='${TARGET_CORE_EXPERIMENT}' SEED='${SEED}' \\
  ANALYSIS_DIR='${TARGET_CORE_DIR}/analysis/step_budget_sweep' \\
  bash scripts/run_eval_step_budget_sweep.sh"

TARGET_ROBUST_CMD="$(common_shell_prefix)
export CUDA_VISIBLE_DEVICES=1
RUN_SAVE_DIR='${TARGET_ROBUST_DIR}' EXPERIMENT_NAME='${TARGET_ROBUST_EXPERIMENT}' SEED='${SEED}' \\
  MODE=obs_noise VALUES='0.0,0.01,0.02,0.05' \\
  ANALYSIS_DIR='${TARGET_ROBUST_DIR}/analysis/obs_noise_sweep' \\
  bash scripts/run_eval_robustness_sweep.sh
RUN_SAVE_DIR='${TARGET_ROBUST_DIR}' EXPERIMENT_NAME='${TARGET_ROBUST_EXPERIMENT}' SEED='${SEED}' \\
  MODE=action_delay VALUES='0,1,2,4' \\
  ANALYSIS_DIR='${TARGET_ROBUST_DIR}/analysis/action_delay_sweep' \\
  bash scripts/run_eval_robustness_sweep.sh
RUN_SAVE_DIR='${TARGET_ROBUST_DIR}' EXPERIMENT_NAME='${TARGET_ROBUST_EXPERIMENT}' SEED='${SEED}' \\
  MODE=action_noise VALUES='0.0,0.01,0.02,0.05' \\
  ANALYSIS_DIR='${TARGET_ROBUST_DIR}/analysis/action_noise_sweep' \\
  bash scripts/run_eval_robustness_sweep.sh"

TARGET_VIDEO_CMD="$(common_shell_prefix)
export CUDA_VISIBLE_DEVICES=1
SOURCE_SAVE_DIR='${TARGET_SOURCE_DIR}' \\
SOURCE_EXPERIMENT_NAME='${TARGET_SOURCE_EXPERIMENT}' \\
TARGET_SAVE_DIR='${TARGET_VIDEO_DIR}' \\
TARGET_EXPERIMENT_NAME='${TARGET_VIDEO_EXPERIMENT}' \\
SEED='${SEED}' \\
DEVICE='cuda:0' \\
RECORDING_EVAL_EPISODES=1 \\
  bash scripts/run_existing_model_video_eval.sh
python scripts/evaluate_video_motion_proxy.py \\
  --run baseline='${BASELINE_VIDEO_DIR}' \\
  --run deltafix='${TARGET_VIDEO_DIR}/video' \\
  --output '${SUITE_ROOT}/analysis/motion_proxy.json'"

PLOT_CMD="$(common_shell_prefix)
mkdir -p '${SUITE_ROOT}/plots' '${SUITE_ROOT}/analysis'
while [[ ! -f '${BASELINE_CORE_DIR}/analysis/context_sweep/context_0.json' || ! -f '${BASELINE_CORE_DIR}/analysis/context_sweep/context_20.json' || ! -f '${TARGET_CORE_DIR}/analysis/context_sweep/context_0.json' || ! -f '${TARGET_CORE_DIR}/analysis/context_sweep/context_20.json' ]]; do sleep 60; done
python scripts/plot_eval_sweep.py \\
  --metric unseen \\
  --title 'Unseen Success vs Context Size' \\
  --xlabel 'Reference Trajectories / History Steps' \\
  --ylabel 'Success Rate (%)' \\
  --point baseline:0='${BASELINE_CORE_DIR}/analysis/context_sweep/context_0.json' \\
  --point baseline:1='${BASELINE_CORE_DIR}/analysis/context_sweep/context_1.json' \\
  --point baseline:3='${BASELINE_CORE_DIR}/analysis/context_sweep/context_3.json' \\
  --point baseline:5='${BASELINE_CORE_DIR}/analysis/context_sweep/context_5.json' \\
  --point baseline:10='${BASELINE_CORE_DIR}/analysis/context_sweep/context_10.json' \\
  --point baseline:20='${BASELINE_CORE_DIR}/analysis/context_sweep/context_20.json' \\
  --point deltafix:0='${TARGET_CORE_DIR}/analysis/context_sweep/context_0.json' \\
  --point deltafix:1='${TARGET_CORE_DIR}/analysis/context_sweep/context_1.json' \\
  --point deltafix:3='${TARGET_CORE_DIR}/analysis/context_sweep/context_3.json' \\
  --point deltafix:5='${TARGET_CORE_DIR}/analysis/context_sweep/context_5.json' \\
  --point deltafix:10='${TARGET_CORE_DIR}/analysis/context_sweep/context_10.json' \\
  --point deltafix:20='${TARGET_CORE_DIR}/analysis/context_sweep/context_20.json' \\
  --output '${SUITE_ROOT}/plots/context_unseen.png'
while [[ ! -f '${BASELINE_CORE_DIR}/analysis/step_budget_sweep/budget_50.json' || ! -f '${BASELINE_CORE_DIR}/analysis/step_budget_sweep/budget_400.json' || ! -f '${TARGET_CORE_DIR}/analysis/step_budget_sweep/budget_50.json' || ! -f '${TARGET_CORE_DIR}/analysis/step_budget_sweep/budget_400.json' ]]; do sleep 60; done
python scripts/plot_eval_sweep.py \\
  --metric unseen \\
  --title 'Unseen Success vs Step Budget' \\
  --xlabel 'Episode Step Limit' \\
  --ylabel 'Success Rate (%)' \\
  --point baseline:50='${BASELINE_CORE_DIR}/analysis/step_budget_sweep/budget_50.json' \\
  --point baseline:100='${BASELINE_CORE_DIR}/analysis/step_budget_sweep/budget_100.json' \\
  --point baseline:200='${BASELINE_CORE_DIR}/analysis/step_budget_sweep/budget_200.json' \\
  --point baseline:300='${BASELINE_CORE_DIR}/analysis/step_budget_sweep/budget_300.json' \\
  --point baseline:400='${BASELINE_CORE_DIR}/analysis/step_budget_sweep/budget_400.json' \\
  --point deltafix:50='${TARGET_CORE_DIR}/analysis/step_budget_sweep/budget_50.json' \\
  --point deltafix:100='${TARGET_CORE_DIR}/analysis/step_budget_sweep/budget_100.json' \\
  --point deltafix:200='${TARGET_CORE_DIR}/analysis/step_budget_sweep/budget_200.json' \\
  --point deltafix:300='${TARGET_CORE_DIR}/analysis/step_budget_sweep/budget_300.json' \\
  --point deltafix:400='${TARGET_CORE_DIR}/analysis/step_budget_sweep/budget_400.json' \\
  --output '${SUITE_ROOT}/plots/step_budget_unseen.png'
while [[ ! -f '${BASELINE_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p0.json' || ! -f '${BASELINE_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p05.json' || ! -f '${TARGET_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p0.json' || ! -f '${TARGET_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p05.json' ]]; do sleep 60; done
python scripts/plot_eval_sweep.py \\
  --metric unseen \\
  --title 'Unseen Success vs Observation Noise' \\
  --xlabel 'Observation Noise Std' \\
  --ylabel 'Success Rate (%)' \\
  --point baseline:0.0='${BASELINE_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p0.json' \\
  --point baseline:0.01='${BASELINE_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p01.json' \\
  --point baseline:0.02='${BASELINE_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p02.json' \\
  --point baseline:0.05='${BASELINE_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p05.json' \\
  --point deltafix:0.0='${TARGET_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p0.json' \\
  --point deltafix:0.01='${TARGET_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p01.json' \\
  --point deltafix:0.02='${TARGET_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p02.json' \\
  --point deltafix:0.05='${TARGET_ROBUST_DIR}/analysis/obs_noise_sweep/obs_noise_0p05.json' \\
  --output '${SUITE_ROOT}/plots/obs_noise_unseen.png'
while [[ ! -f '${BASELINE_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_0.json' || ! -f '${BASELINE_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_4.json' || ! -f '${TARGET_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_0.json' || ! -f '${TARGET_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_4.json' ]]; do sleep 60; done
python scripts/plot_eval_sweep.py \\
  --metric unseen \\
  --title 'Unseen Success vs Action Delay' \\
  --xlabel 'Action Delay Steps' \\
  --ylabel 'Success Rate (%)' \\
  --point baseline:0='${BASELINE_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_0.json' \\
  --point baseline:1='${BASELINE_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_1.json' \\
  --point baseline:2='${BASELINE_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_2.json' \\
  --point baseline:4='${BASELINE_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_4.json' \\
  --point deltafix:0='${TARGET_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_0.json' \\
  --point deltafix:1='${TARGET_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_1.json' \\
  --point deltafix:2='${TARGET_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_2.json' \\
  --point deltafix:4='${TARGET_ROBUST_DIR}/analysis/action_delay_sweep/action_delay_4.json' \\
  --output '${SUITE_ROOT}/plots/action_delay_unseen.png'
while [[ ! -f '${BASELINE_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p0.json' || ! -f '${BASELINE_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p05.json' || ! -f '${TARGET_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p0.json' || ! -f '${TARGET_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p05.json' ]]; do sleep 60; done
python scripts/plot_eval_sweep.py \\
  --metric unseen \\
  --title 'Unseen Success vs Action Noise' \\
  --xlabel 'Action Noise Std' \\
  --ylabel 'Success Rate (%)' \\
  --point baseline:0.0='${BASELINE_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p0.json' \\
  --point baseline:0.01='${BASELINE_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p01.json' \\
  --point baseline:0.02='${BASELINE_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p02.json' \\
  --point baseline:0.05='${BASELINE_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p05.json' \\
  --point deltafix:0.0='${TARGET_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p0.json' \\
  --point deltafix:0.01='${TARGET_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p01.json' \\
  --point deltafix:0.02='${TARGET_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p02.json' \\
  --point deltafix:0.05='${TARGET_ROBUST_DIR}/analysis/action_noise_sweep/action_noise_0p05.json' \\
  --output '${SUITE_ROOT}/plots/action_noise_unseen.png'"

echo "Launching screen sessions under ${SUITE_ROOT}"
launch_screen "${SUITE_TAG}_base_core" "${BASELINE_CORE_CMD}"
launch_screen "${SUITE_TAG}_base_robust" "${BASELINE_ROBUST_CMD}"
launch_screen "${SUITE_TAG}_delta_core" "${TARGET_CORE_CMD}"
launch_screen "${SUITE_TAG}_delta_robust" "${TARGET_ROBUST_CMD}"
launch_screen "${SUITE_TAG}_delta_video" "${TARGET_VIDEO_CMD}"
launch_screen "${SUITE_TAG}_plots" "${PLOT_CMD}"

cat <<EOF
SUITE_ROOT=${SUITE_ROOT}
Baseline source: ${BASELINE_SOURCE_DIR}
Target source: ${TARGET_SOURCE_DIR}
EOF
