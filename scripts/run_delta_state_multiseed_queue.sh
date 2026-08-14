#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

CONDA_ENV="${CONDA_ENV:-0320}"
GPU_INDEX="${GPU_INDEX:-1}"
QUEUE_TAG="${QUEUE_TAG:-$(date +%Y%m%d_%H%M%S)}"
SOURCE_SAVE_DIR="${SOURCE_SAVE_DIR:-${PROJECT_ROOT}/logs/experiment_test}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-3}"
ROLLOUT_DISCOUNT="${ROLLOUT_DISCOUNT:-1.0}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
CASE_RETRIES="${CASE_RETRIES:-2}"

SEEDS=("${@:-3 4 5}")

echo "=== Delta-state multiseed queue start ==="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "GPU_INDEX=${GPU_INDEX}"
echo "QUEUE_TAG=${QUEUE_TAG}"
echo "SOURCE_SAVE_DIR=${SOURCE_SAVE_DIR}"
echo "ROLLOUT_HORIZON=${ROLLOUT_HORIZON}"
echo "ROLLOUT_DISCOUNT=${ROLLOUT_DISCOUNT}"
echo "SAVE_VIDEO=${SAVE_VIDEO}"
echo "SEEDS=${SEEDS[*]}"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

for seed in "${SEEDS[@]}"; do
  run_name="task_dyn_true_multistep_h${ROLLOUT_HORIZON}_delta_seed${seed}_${QUEUE_TAG}"
  experiment_name="${run_name}"

  echo
  echo "=== Running seed=${seed} experiment=${experiment_name} ==="

  CONDA_ENV="${CONDA_ENV}" \
  RUN_NAME="${run_name}" \
  EXPERIMENT_NAME="${experiment_name}" \
  SEED="${seed}" \
  DEVICE="cuda:0" \
  SOURCE_SAVE_DIR="${SOURCE_SAVE_DIR}" \
  SAVE_DIR="${PROJECT_ROOT}/logs/${run_name}" \
  BUFFER_SOURCE_DIR="${SOURCE_SAVE_DIR}/buffer/collective_buffer" \
  ROLLOUT_HORIZON="${ROLLOUT_HORIZON}" \
  ROLLOUT_DISCOUNT="${ROLLOUT_DISCOUNT}" \
  SAVE_VIDEO="${SAVE_VIDEO}" \
  CASE_RETRIES="${CASE_RETRIES}" \
  "${PROJECT_ROOT}/scripts/run_existing_buffer_variant.sh"
done

echo
echo "=== Delta-state multiseed queue finished ==="
