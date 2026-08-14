#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

CONDA_ENV="${CONDA_ENV:-0320}"
SEED="${SEED:-5}"
DEVICE="${DEVICE:-cuda:0}"
METHODS="${METHODS:-baseline,multistep,deltafix}"
K_SHOTS="${K_SHOTS:-0,1,3,5,10,20}"
FEWSHOT_ROBOTS="${FEWSHOT_ROBOTS:-panda,kuka,ur5e,viperx}"
FEWSHOT_VAL_SHOTS="${FEWSHOT_VAL_SHOTS:-0}"
SAMPLING_MODE="${SAMPLING_MODE:-random}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
RECORDING_EVAL_EPISODES="${RECORDING_EVAL_EPISODES:-0}"
CASE_RETRIES="${CASE_RETRIES:-2}"
AUTO_COLLECT_SUPPORT="${AUTO_COLLECT_SUPPORT:-True}"

BASE_BUFFER_SOURCE_DIR="${BASE_BUFFER_SOURCE_DIR:-${PROJECT_ROOT}/logs/task_dyn_true_singlestep_seed5_20260325_153611/buffer/collective_buffer}"
BUFFER_ROOT="${BUFFER_ROOT:-${PROJECT_ROOT}/logs/fewshot_buffers_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/logs/paper_fewshot_adaptation_seed${SEED}}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_5/representation_cls_transformer_checkpoint.pth}"
SUPPORT_RUN_NAME="${SUPPORT_RUN_NAME:-fewshot_support_seed${SEED}}"
SUPPORT_SAVE_DIR="${SUPPORT_SAVE_DIR:-${PROJECT_ROOT}/logs/${SUPPORT_RUN_NAME}}"
SUPPORT_COLLECTIVE_BUFFER_DIR="${SUPPORT_COLLECTIVE_BUFFER_DIR:-${SUPPORT_SAVE_DIR}/buffer/collective_buffer}"
MERGED_BUFFER_SOURCE_DIR="${MERGED_BUFFER_SOURCE_DIR:-${BUFFER_ROOT}/merged_collective_buffer}"

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

IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
IFS=',' read -r -a SHOT_LIST <<< "${K_SHOTS}"

mkdir -p "${BUFFER_ROOT}" "${OUTPUT_ROOT}"

max_shot() {
  local max_value=0
  local shot
  for shot in "${SHOT_LIST[@]}"; do
    if (( shot > max_value )); then
      max_value="${shot}"
    fi
  done
  echo "${max_value}"
}

link_buffer_split() {
  local src_root="$1"
  local dst_root="$2"
  local split="$3"
  local src_split="${src_root}/${split}"
  local dst_split="${dst_root}/${split}"
  mkdir -p "${dst_split}"
  if [[ ! -d "${src_split}" ]]; then
    return 0
  fi
  local path
  for path in "${src_split}"/*; do
    [[ -e "${path}" ]] || continue
    ln -sfn "${path}" "${dst_split}/$(basename "${path}")"
  done
}

method_rollout_horizon() {
  case "$1" in
    baseline) echo "1" ;;
    multistep) echo "5" ;;
    deltafix) echo "5" ;;
    *)
      echo "Unsupported method=$1" >&2
      return 1
      ;;
  esac
}

method_extra_args() {
  case "$1" in
    baseline)
      echo "transformer_collective_network.predictive_adapter.delta_state_loss_weight=0.0 experiment.early_stopping=False"
      ;;
    multistep)
      echo "transformer_collective_network.predictive_adapter.delta_state_loss_weight=0.0 experiment.early_stopping=False"
      ;;
    deltafix)
      echo "experiment.early_stopping=False"
      ;;
    *)
      echo "Unsupported method=$1" >&2
      return 1
      ;;
  esac
}

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "BUFFER_ROOT=${BUFFER_ROOT}"
echo "METHODS=${METHODS}"
echo "K_SHOTS=${K_SHOTS}"
echo "DEVICE=${DEVICE}"
echo "PYTHON_BIN=${PYTHON_BIN}"

if [[ "${AUTO_COLLECT_SUPPORT}" == "True" || "${AUTO_COLLECT_SUPPORT}" == "true" ]]; then
  support_max_shot="$(max_shot)"
  echo
  echo "=== Collecting unseen support pool up to K=${support_max_shot} ==="
  CONDA_ENV="${CONDA_ENV}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
  SUPPORT_ROBOTS="${FEWSHOT_ROBOTS}" \
  MAX_SUPPORT_SHOTS="${support_max_shot}" \
  SAVE_DIR="${SUPPORT_SAVE_DIR}" \
  COLLECTIVE_BUFFER_DIR="${SUPPORT_COLLECTIVE_BUFFER_DIR}" \
  bash "${PROJECT_ROOT}/scripts/collect_unseen_support_buffers.sh"
fi

echo
echo "=== Building merged collective buffer source ==="
rm -rf "${MERGED_BUFFER_SOURCE_DIR}"
mkdir -p "${MERGED_BUFFER_SOURCE_DIR}/train" "${MERGED_BUFFER_SOURCE_DIR}/validation"
link_buffer_split "${BASE_BUFFER_SOURCE_DIR}" "${MERGED_BUFFER_SOURCE_DIR}" train
link_buffer_split "${BASE_BUFFER_SOURCE_DIR}" "${MERGED_BUFFER_SOURCE_DIR}" validation
link_buffer_split "${SUPPORT_COLLECTIVE_BUFFER_DIR}" "${MERGED_BUFFER_SOURCE_DIR}" train
link_buffer_split "${SUPPORT_COLLECTIVE_BUFFER_DIR}" "${MERGED_BUFFER_SOURCE_DIR}" validation

for shot in "${SHOT_LIST[@]}"; do
  buffer_dir="${BUFFER_ROOT}/fewshot_k${shot}"
  metadata_path="${buffer_dir}/fewshot_metadata.json"
  if [[ ! -f "${metadata_path}" ]]; then
    echo
    echo "=== Building few-shot buffer K=${shot} ==="
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/make_fewshot_collective_buffer.py" \
      --src "${MERGED_BUFFER_SOURCE_DIR}" \
      --dst "${buffer_dir}" \
      --shots "${shot}" \
      --fewshot-robots "${FEWSHOT_ROBOTS}" \
      --fewshot-val-shots "${FEWSHOT_VAL_SHOTS}" \
      --sampling-mode "${SAMPLING_MODE}" \
      --seed "${SEED}"
  else
    echo
    echo "=== Reusing few-shot buffer K=${shot}: ${buffer_dir} ==="
  fi

  for method in "${METHOD_LIST[@]}"; do
    run_name="fewshot_${method}_k${shot}_seed${SEED}"
    save_dir="${OUTPUT_ROOT}/${run_name}"
    summary_path="${save_dir}/evaluation_summary.json"
    if [[ -f "${summary_path}" ]]; then
      echo "[SKIP] ${run_name} already finished"
      continue
    fi

    rollout_horizon="$(method_rollout_horizon "${method}")"
    extra_args="$(method_extra_args "${method}")"

    echo
    echo "=== Running ${run_name} ==="
    CONDA_ENV="${CONDA_ENV}" \
    RUN_NAME="${run_name}" \
    EXPERIMENT_NAME="${run_name}" \
    SEED="${SEED}" \
    DEVICE="${DEVICE}" \
    BUFFER_SOURCE_DIR="${buffer_dir}" \
    SAVE_DIR="${save_dir}" \
    TRANSFORMER_PATH="${TRANSFORMER_PATH}" \
    ROLLOUT_HORIZON="${rollout_horizon}" \
    SAVE_VIDEO="${SAVE_VIDEO}" \
    RECORDING_EVAL_EPISODES="${RECORDING_EVAL_EPISODES}" \
    CASE_RETRIES="${CASE_RETRIES}" \
    EXTRA_MAIN_ARGS="${extra_args}" \
    bash "${PROJECT_ROOT}/scripts/run_existing_buffer_variant.sh"
  done
done

echo
echo "Few-shot adaptation sweep finished."
echo "Outputs under ${OUTPUT_ROOT}"
