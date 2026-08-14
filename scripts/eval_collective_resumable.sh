#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

if [[ -d "${PROJECT_ROOT}/mtenv_repo" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}/mtenv_repo${PYTHONPATH:+:${PYTHONPATH}}"
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-task_dyn_true}"
SEED="${SEED:-3}"
DEFAULT_TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_${SEED}/representation_cls_transformer_checkpoint.pth"
if [[ ! -f "${DEFAULT_TRANSFORMER_PATH}" ]]; then
  DEFAULT_TRANSFORMER_PATH="${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_5/representation_cls_transformer_checkpoint.pth"
fi
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${DEFAULT_TRANSFORMER_PATH}}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/logs/${EXPERIMENT_NAME}}"
CASE_RETRIES="${CASE_RETRIES:-2}"
EXPECT_VIDEO="${EXPECT_VIDEO:-1}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python"
fi

SCRIPT_EXTRA_ARGS=()
TARGET_ROBOT=""
TARGET_TASK=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    *=*|--*)
      SCRIPT_EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ -z "${TARGET_ROBOT}" ]]; then
        TARGET_ROBOT="$1"
      elif [[ -z "${TARGET_TASK}" ]]; then
        TARGET_TASK="$1"
      else
        SCRIPT_EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

ALL_ROBOTS=("sawyer" "panda" "kuka" "ur5e" "ur10e" "xarm7" "unitree_z1" "gen3" "viperx")
ALL_TASKS=("reach-v2" "push-v2" "peg-insert-side-v2" "door-open-v2" \
           "window-open-v2" "window-close-v2" "drawer-open-v2" \
           "button-press-topdown-v2" "faucet-open-v2" "pick-place-v2")

mkdir -p "${PROJECT_ROOT}/logs/results/col"
mkdir -p "${SAVE_DIR}/video"

is_case_complete() {
  local robot="$1"
  local task="$2"
  local log_file="$3"
  local video_glob="${SAVE_DIR}/video/${robot}_transformer_env0_${task}_*sample_0.gif"

  if [[ ! -f "${log_file}" ]]; then
    return 1
  fi

  if ! grep -q "Evaluation: robot ${robot} " "${log_file}"; then
    return 1
  fi

  if [[ "${EXPECT_VIDEO}" != "1" ]]; then
    return 0
  fi

  shopt -s nullglob
  local videos=( ${video_glob} )
  shopt -u nullglob
  if [[ ${#videos[@]} -eq 0 ]]; then
    return 1
  fi

  return 0
}

run_case() {
  local robot="$1"
  local task="$2"
  local log_dir="${PROJECT_ROOT}/logs/results/col/${task}_${EXPERIMENT_NAME}"
  local log_file="${log_dir}/eval_${robot}_seed_${SEED}.log"
  local attempt

  mkdir -p "${log_dir}"

  if is_case_complete "${robot}" "${task}" "${log_file}"; then
    echo "[SKIP] ${robot} ${task} already complete"
    return 0
  fi

  for attempt in $(seq 1 "${CASE_RETRIES}"); do
    echo "[RUN ] ${robot} ${task} (attempt ${attempt}/${CASE_RETRIES})"
    rm -f "${log_file}"
    if "${PYTHON_BIN}" -u "${PROJECT_ROOT}/main.py" \
      setup=metaworld \
      env=metaworld-mt1 \
      worker.multitask.num_envs=1 \
      experiment.robot_type="${robot}" \
      experiment.mode=evaluate_collective_transformer \
      env.benchmark.env_name="${task}" \
      experiment.evaluate_transformer=collective_network \
      experiment.experiment="${EXPERIMENT_NAME}" \
      setup.seed="${SEED}" \
      transformer_collective_network.transformer_encoder.representation_transformer.model_path="${TRANSFORMER_PATH}" \
      transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="${TRANSFORMER_PATH}" \
      "${SCRIPT_EXTRA_ARGS[@]}" > "${log_file}" 2>&1; then
      if is_case_complete "${robot}" "${task}" "${log_file}"; then
        echo "[DONE] ${robot} ${task}"
        return 0
      fi
      echo "[WARN] ${robot} ${task} finished without complete markers; retrying"
    else
      echo "[WARN] ${robot} ${task} failed on attempt ${attempt}; retrying"
    fi
    sleep 2
  done

  echo "[FAIL] ${robot} ${task} could not be completed after ${CASE_RETRIES} attempts"
  return 1
}

failures=0
completed=0
total=$(( ${#ALL_ROBOTS[@]} * ${#ALL_TASKS[@]} ))

ROBOTS_TO_RUN=("${ALL_ROBOTS[@]}")
TASKS_TO_RUN=("${ALL_TASKS[@]}")
if [[ -n "${TARGET_ROBOT}" ]]; then
  ROBOTS_TO_RUN=("${TARGET_ROBOT}")
  if [[ -n "${TARGET_TASK}" ]]; then
    TASKS_TO_RUN=("${TARGET_TASK}")
    total=1
  else
    total=${#ALL_TASKS[@]}
  fi
fi

echo "Using python interpreter: ${PYTHON_BIN}"
echo "Evaluating experiment=${EXPERIMENT_NAME}, seed=${SEED}, save_dir=${SAVE_DIR}"
echo "Video backend: MUJOCO_GL=${MUJOCO_GL:-unset}, PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-unset}"

for robot in "${ROBOTS_TO_RUN[@]}"; do
  for task in "${TASKS_TO_RUN[@]}"; do
    if run_case "${robot}" "${task}"; then
      completed=$((completed + 1))
    else
      failures=$((failures + 1))
    fi
    echo "[PROGRESS] completed=${completed}/${total}, failures=${failures}"
  done
done

if [[ "${failures}" -gt 0 ]]; then
  echo "[ERROR] ${failures} case(s) failed during resumable eval"
  exit 1
fi

echo "All evaluation cases completed successfully."
