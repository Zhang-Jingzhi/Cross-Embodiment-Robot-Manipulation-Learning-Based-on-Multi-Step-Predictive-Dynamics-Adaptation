#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

if [[ -d "${PROJECT_ROOT}/mtenv_repo" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}/mtenv_repo${PYTHONPATH:+:${PYTHONPATH}}"
fi

CONDA_ENV="${CONDA_ENV:-0320}"
SEED="${SEED:-5}"
DEVICE="${DEVICE:-cuda:0}"
SUPPORT_ROBOTS="${SUPPORT_ROBOTS:-panda,kuka,ur5e,viperx}"
TASK_LIST="${TASK_LIST:-reach-v2,push-v2,peg-insert-side-v2,door-open-v2,window-open-v2,window-close-v2,drawer-open-v2,button-press-topdown-v2,faucet-open-v2,pick-place-v2}"
EPISODE_LEN="${EPISODE_LEN:-400}"
MAX_SUPPORT_SHOTS="${MAX_SUPPORT_SHOTS:-20}"
SPLIT_RATIO="${SPLIT_RATIO:-1.0}"
RUN_NAME="${RUN_NAME:-fewshot_support_seed${SEED}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-fewshot_support_seed${SEED}}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/logs/${RUN_NAME}}"
STAGE_LOG_DIR="${STAGE_LOG_DIR:-${SAVE_DIR}/stage_logs}"
COLLECTIVE_BUFFER_DIR="${COLLECTIVE_BUFFER_DIR:-${SAVE_DIR}/buffer/collective_buffer}"

SUPPORT_STEPS="${SUPPORT_STEPS:-$(( MAX_SUPPORT_SHOTS * EPISODE_LEN + 1 ))}"

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

IFS=',' read -r -a ROBOT_LIST <<< "${SUPPORT_ROBOTS}"
IFS=',' read -r -a TASKS <<< "${TASK_LIST}"

mkdir -p "${SAVE_DIR}" "${STAGE_LOG_DIR}" "${COLLECTIVE_BUFFER_DIR}/train" "${COLLECTIVE_BUFFER_DIR}/validation"

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
    "$@" 2>&1 | tee -a "${log_file}"
}

support_complete() {
  local train_dir="$1"
  if [[ ! -d "${train_dir}" ]]; then
    return 1
  fi
  if compgen -G "${train_dir}/*.pt" > /dev/null; then
    return 0
  fi
  return 1
}

sanitize_buffer_dir() {
  local source_dir="$1"
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
import torch

source_dir = Path(${source_dir@Q})
episode_len = int(${EPISODE_LEN})

for path in sorted(source_dir.glob("*.pt")):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = int(payload[0].shape[0])
    if rows < episode_len or rows % episode_len != 0:
        path.unlink()
        print(f"[SANITIZE] removed short/non-episode chunk: {path.name} ({rows} rows)")
PY
}

collect_log="${STAGE_LOG_DIR}/01_collect_support.log"
split_log="${STAGE_LOG_DIR}/02_split_support.log"
: > "${collect_log}"
: > "${split_log}"

echo "PROJECT_ROOT=${PROJECT_ROOT}" | tee -a "${collect_log}"
echo "SAVE_DIR=${SAVE_DIR}" | tee -a "${collect_log}"
echo "SUPPORT_ROBOTS=${SUPPORT_ROBOTS}" | tee -a "${collect_log}"
echo "TASK_LIST=${TASK_LIST}" | tee -a "${collect_log}"
echo "SUPPORT_STEPS=${SUPPORT_STEPS}" | tee -a "${collect_log}"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${collect_log}"

for robot in "${ROBOT_LIST[@]}"; do
  for task in "${TASKS[@]}"; do
    source_dir="${SAVE_DIR}/buffer/buffer/buffer_${robot}_${task}_seed_${SEED}"
    train_dir="${COLLECTIVE_BUFFER_DIR}/train/online_buffer_${robot}_${task}_seed_${SEED}"
    val_dir="${COLLECTIVE_BUFFER_DIR}/validation/online_buffer_${robot}_${task}_seed_${SEED}"

    if support_complete "${train_dir}"; then
      echo "[SKIP] support already prepared for ${robot} ${task}" | tee -a "${split_log}"
      continue
    fi

    if [[ ! -d "${source_dir}" ]] || ! compgen -G "${source_dir}/*.pt" > /dev/null; then
      run_main "${collect_log}" \
        experiment.mode=train_worker \
        experiment.robot_type="${robot}" \
        env.benchmark.env_name="${task}" \
        experiment.num_train_steps="${SUPPORT_STEPS}" \
        experiment.init_steps=0 \
        experiment.follow_scripted=True \
        experiment.train_worker=False \
        experiment.num_warmup_episodes=0 \
        experiment.eval_freq=100000000 \
        experiment.save_freq=100000000 \
        experiment.save_video=False \
        experiment.col_training_warmup=100000000 \
        experiment.col_sampling_freq=100000000 \
        experiment.save.buffer.should_save=True
    else
      echo "[SKIP] raw worker buffer already exists for ${robot} ${task}" | tee -a "${collect_log}"
    fi

    sanitize_buffer_dir "${source_dir}" | tee -a "${split_log}"
    rm -rf "${train_dir}" "${val_dir}"
    mkdir -p "${train_dir}" "${val_dir}"
    echo "[$(date '+%F %T')] split ${source_dir} -> ${train_dir}" | tee -a "${split_log}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/split_buffer_files.py" \
      --source "${source_dir}" \
      --train "${train_dir}" \
      --val "${val_dir}" \
      --ratio "${SPLIT_RATIO}" 2>&1 | tee -a "${split_log}"
  done
done

echo
echo "Support collection finished."
echo "COLLECTIVE_BUFFER_DIR=${COLLECTIVE_BUFFER_DIR}"
