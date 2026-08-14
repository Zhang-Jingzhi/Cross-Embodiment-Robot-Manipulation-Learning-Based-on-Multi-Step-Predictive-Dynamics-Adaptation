#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

if [[ -d "${PROJECT_ROOT}/mtenv_repo" ]]; then
  export PYTHONPATH="${PROJECT_ROOT}/mtenv_repo${PYTHONPATH:+:${PYTHONPATH}}"
fi

RUN_SAVE_DIR="${RUN_SAVE_DIR:-${PROJECT_ROOT}/logs/task_dyn_true_multistep_h3_seed5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "${RUN_SAVE_DIR}")}"
SOURCE_EXPERIMENT_NAME="${SOURCE_EXPERIMENT_NAME:-${EXPERIMENT_NAME}}"
SEED="${SEED:-5}"
MODE="${MODE:-robot_joint_damping}"
VALUES="${VALUES:-1.0,1.5,2.0,3.0}"
TARGET_ROBOTS="${TARGET_ROBOTS:-}"
TARGET_ROBOT="${TARGET_ROBOT:-}"
TARGET_TASK="${TARGET_TASK:-}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RUN_SAVE_DIR}/analysis/${MODE}_sweep}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
EXPERIMENT_SUFFIX="${EXPERIMENT_SUFFIX:-}"
CASE_RETRIES="${CASE_RETRIES:-2}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

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
  ln -sfn "${RUN_SAVE_DIR}/model_dir/${SOURCE_EXPERIMENT_NAME}" \
    "${point_save_dir}/model_dir/${point_experiment_name}"
  ln -sfn "${RUN_SAVE_DIR}/buffer/collective_buffer" \
    "${point_save_dir}/buffer/collective_buffer"
}

case "${MODE}" in
  robot_body_mass)
    prefix="robot_body_mass"
    body_mass_scale_var="VALUE"
    joint_damping_scale="1.0"
    joint_armature_scale="1.0"
    ;;
  robot_joint_damping)
    prefix="robot_joint_damping"
    body_mass_scale_var="1.0"
    joint_damping_scale="VALUE"
    joint_armature_scale="1.0"
    ;;
  robot_joint_armature)
    prefix="robot_joint_armature"
    body_mass_scale_var="1.0"
    joint_damping_scale="1.0"
    joint_armature_scale="VALUE"
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use robot_body_mass, robot_joint_damping, or robot_joint_armature."
    exit 1
    ;;
esac

echo "RUN_SAVE_DIR=${RUN_SAVE_DIR}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "SOURCE_EXPERIMENT_NAME=${SOURCE_EXPERIMENT_NAME}"
echo "SEED=${SEED}"
echo "TRANSFORMER_PATH=${TRANSFORMER_PATH}"
echo "MODE=${MODE}"
echo "VALUES=${VALUES}"
echo "TARGET_ROBOTS=${TARGET_ROBOTS}"
echo "TARGET_ROBOT=${TARGET_ROBOT}"
echo "TARGET_TASK=${TARGET_TASK}"
echo "ANALYSIS_DIR=${ANALYSIS_DIR}"
echo "EXPERIMENT_SUFFIX=${EXPERIMENT_SUFFIX}"
echo "CASE_RETRIES=${CASE_RETRIES}"
echo "EXTRA_EVAL_ARGS=${EXTRA_EVAL_ARGS}"

for value in "${SWEEP_VALUES[@]}"; do
  echo
  echo "=== Evaluating ${MODE}=${value} ==="
  safe_value="${value//./p}"
  suffix_part=""
  if [[ -n "${EXPERIMENT_SUFFIX}" ]]; then
    suffix_part="_${EXPERIMENT_SUFFIX}"
  fi
  point_experiment_name="${EXPERIMENT_NAME}_${prefix}_${safe_value}${suffix_part}"
  point_save_dir="${RUN_SAVE_DIR}/eval_runs/${prefix}_${safe_value}${suffix_part}"
  prepare_point_workspace "${point_save_dir}" "${point_experiment_name}"

  body_mass_scale="${body_mass_scale_var/VALUE/${value}}"
  joint_damping="${joint_damping_scale/VALUE/${value}}"
  joint_armature="${joint_armature_scale/VALUE/${value}}"
  eval_args=()
  extra_eval_args_array=()
  if [[ -n "${EXTRA_EVAL_ARGS}" ]]; then
    read -r -a extra_eval_args_array <<< "${EXTRA_EVAL_ARGS}"
  fi
  if [[ -n "${TARGET_ROBOT}" ]]; then
    eval_args+=("${TARGET_ROBOT}")
    if [[ -n "${TARGET_TASK}" ]]; then
      eval_args+=("${TARGET_TASK}")
    fi
  fi
  eval_args+=(
    "setup.save_dir=${point_save_dir}"
    "setup.base_path=${PROJECT_ROOT}"
    "experiment.save_video=${SAVE_VIDEO}"
    "transformer_collective_network.predictive_adapter.pretrained_dir=${point_save_dir}/model_dir"
  )
  if [[ ${#extra_eval_args_array[@]} -gt 0 ]]; then
    eval_args+=("${extra_eval_args_array[@]}")
  fi

  (
    cd "${PROJECT_ROOT}"
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    EXPERIMENT_NAME="${point_experiment_name}" \
    SEED="${SEED}" \
    TRANSFORMER_PATH="${TRANSFORMER_PATH}" \
    SAVE_DIR="${point_save_dir}" \
    EXPECT_VIDEO=0 \
    CASE_RETRIES="${CASE_RETRIES}" \
    METAWORLD_DYNAMICS_VARIANT_TAG="${prefix}_${safe_value}" \
    METAWORLD_DYNAMICS_VARIANT_ROBOTS="${TARGET_ROBOTS}" \
    METAWORLD_ROBOT_BODY_MASS_SCALE="${body_mass_scale}" \
    METAWORLD_ROBOT_JOINT_DAMPING_SCALE="${joint_damping}" \
    METAWORLD_ROBOT_JOINT_ARMATURE_SCALE="${joint_armature}" \
    bash scripts/eval_collective_resumable.sh "${eval_args[@]}"
  )

  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/summarize_collective_results.py" \
    --experiment "${point_experiment_name}" \
    --seed "${SEED}" \
    --output "${ANALYSIS_DIR}/${prefix}_${safe_value}.json"
done

echo
echo "Robot dynamics sweep finished."
echo "Summaries saved under: ${ANALYSIS_DIR}"
