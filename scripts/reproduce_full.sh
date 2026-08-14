#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_ROOT

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-pace_repro}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-task_dyn_true}"
SEED="${SEED:-5}"
DEVICE="${DEVICE:-cuda:0}"
STEP_SCALE="${STEP_SCALE:-1.0}"
TRANSFORMER_EPOCHS="${TRANSFORMER_EPOCHS:-50}"
TRANSFORMER_BATCH_SIZE="${TRANSFORMER_BATCH_SIZE:-256}"
TRANSFORMER_LR="${TRANSFORMER_LR:-0.0002}"
RUN_STAGES="${RUN_STAGES:-workers,distill,split,transformer,adapter,collective,eval}"

SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/logs/${RUN_NAME}}"
STAGE_LOG_DIR="${STAGE_LOG_DIR:-${SAVE_DIR}/stage_logs}"
SPLIT_DATASET_ROOT="${SPLIT_DATASET_ROOT:-${PROJECT_ROOT}/Transformer_RNN/${RUN_NAME}/dataset}"
TRANSFORMER_DATA_ROOT="${TRANSFORMER_DATA_ROOT:-${PROJECT_ROOT}/Transformer_RNN/${RUN_NAME}/decision_tf_dataset}"
TRANSFORMER_CKPT_DIR="${TRANSFORMER_CKPT_DIR:-${PROJECT_ROOT}/Transformer_RNN/checkpoints_${RUN_NAME}_seed_${SEED}}"
TRANSFORMER_CKPT_PATH="${TRANSFORMER_CKPT_PATH:-${TRANSFORMER_CKPT_DIR}/representation_cls_transformer_checkpoint.pth}"
TRANSFORMER_TB_DIR="${TRANSFORMER_TB_DIR:-${PROJECT_ROOT}/Transformer_RNN/tensorboard_log/${RUN_NAME}_seed_${SEED}}"
TRANSFORMER_EMB_PATH="${TRANSFORMER_EMB_PATH:-${PROJECT_ROOT}/Transformer_RNN/embedding_log_${RUN_NAME}_seed_${SEED}/emb.pth}"

mkdir -p "${SAVE_DIR}" "${STAGE_LOG_DIR}" "${SPLIT_DATASET_ROOT}/train" "${SPLIT_DATASET_ROOT}/validation"
mkdir -p "${TRANSFORMER_DATA_ROOT}" "${TRANSFORMER_CKPT_DIR}" "$(dirname "${TRANSFORMER_TB_DIR}")" "$(dirname "${TRANSFORMER_EMB_PATH}")"

SEEN_ROBOT_LIST="${SEEN_ROBOT_LIST:-sawyer,ur10e,xarm7,unitree_z1,gen3}"
ALL_ROBOT_LIST="${ALL_ROBOT_LIST:-sawyer,panda,kuka,ur5e,ur10e,xarm7,unitree_z1,gen3,viperx}"
EVAL_ROBOT_LIST="${EVAL_ROBOT_LIST:-${ALL_ROBOT_LIST}}"
TASK_LIST="${TASK_LIST:-reach-v2,push-v2,pick-place-v2,door-open-v2,drawer-open-v2,button-press-topdown-v2,peg-insert-side-v2,window-open-v2,window-close-v2}"

IFS=',' read -r -a SEEN_ROBOTS <<< "${SEEN_ROBOT_LIST}"
IFS=',' read -r -a ALL_ROBOTS <<< "${ALL_ROBOT_LIST}"
IFS=',' read -r -a EVAL_ROBOTS <<< "${EVAL_ROBOT_LIST}"
IFS=',' read -r -a TASKS <<< "${TASK_LIST}"

DEFAULT_TASKS=(
  "reach-v2"
  "push-v2"
  "pick-place-v2"
  "door-open-v2"
  "drawer-open-v2"
  "button-press-topdown-v2"
  "peg-insert-side-v2"
  "window-open-v2"
  "window-close-v2"
)

declare -A TASK_STEPS=(
  ["reach-v2"]=200000
  ["push-v2"]=900000
  ["pick-place-v2"]=2400000
  ["door-open-v2"]=1000000
  ["drawer-open-v2"]=500000
  ["button-press-topdown-v2"]=500000
  ["peg-insert-side-v2"]=1300000
  ["window-open-v2"]=300000
  ["window-close-v2"]=400000
)

stage_enabled() {
  local stage="$1"
  [[ ",${RUN_STAGES}," == *",${stage},"* ]]
}

run_cmd() {
  local log_file="$1"
  shift
  echo
  echo "[$(date '+%F %T')] $*" | tee -a "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
}

run_main() {
  local log_file="$1"
  shift
  run_cmd "${log_file}" \
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
    "$@"
}

train_workers() {
  local log_file="${STAGE_LOG_DIR}/01_workers.log"
  : > "${log_file}"
  for robot in "${SEEN_ROBOTS[@]}"; do
    for task in "${TASKS[@]}"; do
      local base_steps="${TASK_STEPS[${task}]}"
      local steps
      steps="$("${PYTHON_BIN}" - <<PY
base_steps = int(${base_steps})
scaled = max(1000, int(round(base_steps * float(${STEP_SCALE}))))
print(scaled)
PY
)"
      local extra_args=()
      if [[ "${task}" == "reach-v2" ]]; then
        extra_args+=("worker.builder.actor_update_freq=1")
      fi
      run_main "${log_file}" \
        experiment.mode=train_worker \
        experiment.robot_type="${robot}" \
        env.benchmark.env_name="${task}" \
        experiment.num_train_steps="${steps}" \
        "${extra_args[@]}"
    done
  done
}

run_online_distill() {
  local log_file="${STAGE_LOG_DIR}/02_online_distill.log"
  : > "${log_file}"
  for robot in "${SEEN_ROBOTS[@]}"; do
    for task in "${TASKS[@]}"; do
      run_main "${log_file}" \
        experiment.mode=online_distill_collective_transformer \
        experiment.robot_type="${robot}" \
        env.benchmark.env_name="${task}" \
        transformer_collective_network.predictive_adapter.load_on_init=False \
        transformer_collective_network.predictive_adapter.pretrained_dir="${SAVE_DIR}/model_dir"
    done
  done
}

split_distill_buffers() {
  local log_file="${STAGE_LOG_DIR}/03_split_buffers.log"
  : > "${log_file}"
  for robot in "${SEEN_ROBOTS[@]}"; do
    for task in "${TASKS[@]}"; do
      local source_dir="${SAVE_DIR}/buffer/buffer_distill/buffer_distill_${robot}_${task}_seed_${SEED}"
      local train_dir="${SPLIT_DATASET_ROOT}/train/buffer_distill_${robot}_${task}_seed_${SEED}"
      local val_dir="${SPLIT_DATASET_ROOT}/validation/buffer_distill_${robot}_${task}_seed_${SEED}"
      if [[ ! -d "${source_dir}" ]]; then
        echo "Skipping missing distill buffer: ${source_dir}" | tee -a "${log_file}"
        continue
      fi
      run_cmd "${log_file}" \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/split_buffer_files.py" \
        --source "${source_dir}" \
        --train "${train_dir}" \
        --val "${val_dir}"
    done
  done

  for robot in "${SEEN_ROBOTS[@]}"; do
    for task in "${TASKS[@]}"; do
      local source_dir="${SAVE_DIR}/buffer/online_buffer_${robot}_${task}"
      local train_dir="${SAVE_DIR}/buffer/collective_buffer/train/online_buffer_${robot}_${task}_seed_${SEED}"
      local val_dir="${SAVE_DIR}/buffer/collective_buffer/validation/online_buffer_${robot}_${task}_seed_${SEED}"
      if [[ ! -d "${source_dir}" ]]; then
        echo "Skipping missing online buffer: ${source_dir}" | tee -a "${log_file}"
        continue
      fi
      run_cmd "${log_file}" \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/split_buffer_files.py" \
        --source "${source_dir}" \
        --train "${train_dir}" \
        --val "${val_dir}"
    done
  done
}

train_transformer() {
  local log_file="${STAGE_LOG_DIR}/04_transformer.log"
  : > "${log_file}"
  run_cmd "${log_file}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/Transformer_RNN/dataset_tf.py" \
    --input-root "${SPLIT_DATASET_ROOT}" \
    --output-root "${TRANSFORMER_DATA_ROOT}" \
    --seed "${SEED}"

  run_cmd "${log_file}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/Transformer_RNN/RepresentationTransformerWithCLS.py" \
    --epochs "${TRANSFORMER_EPOCHS}" \
    --batch-size "${TRANSFORMER_BATCH_SIZE}" \
    --learning-rate "${TRANSFORMER_LR}" \
    --seed "${SEED}" \
    --model-path "${TRANSFORMER_CKPT_PATH}" \
    --train-dataset-path "${TRANSFORMER_DATA_ROOT}/train/data" \
    --val-dataset-path "${TRANSFORMER_DATA_ROOT}/validation/data" \
    --embeddings-path "${TRANSFORMER_EMB_PATH}" \
    --log-path "${TRANSFORMER_TB_DIR}" \
    --fresh-start
}

train_predictive_adapter() {
  local log_file="${STAGE_LOG_DIR}/05_predictive_adapter.log"
  : > "${log_file}"
  run_main "${log_file}" \
    experiment.mode=train_predictive_adapter \
    transformer_collective_network.predictive_adapter.load_on_init=False \
    transformer_collective_network.transformer_encoder.representation_transformer.model_path="${TRANSFORMER_CKPT_PATH}" \
    transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="${TRANSFORMER_CKPT_PATH}"
}

train_collective_policy() {
  local log_file="${STAGE_LOG_DIR}/06_collective.log"
  : > "${log_file}"
  run_main "${log_file}" \
    experiment.mode=distill_collective_transformer \
    transformer_collective_network.predictive_adapter.pretrained_dir="${SAVE_DIR}/model_dir" \
    transformer_collective_network.transformer_encoder.representation_transformer.model_path="${TRANSFORMER_CKPT_PATH}" \
    transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="${TRANSFORMER_CKPT_PATH}"
}

run_eval() {
  local log_file="${STAGE_LOG_DIR}/07_eval.log"
  : > "${log_file}"
  for robot in "${EVAL_ROBOTS[@]}"; do
    for task in "${TASKS[@]}"; do
      run_main "${log_file}" \
        experiment.mode=evaluate_collective_transformer \
        experiment.robot_type="${robot}" \
        env.benchmark.env_name="${task}" \
        experiment.evaluate_transformer=collective_network \
        transformer_collective_network.predictive_adapter.pretrained_dir="${SAVE_DIR}/model_dir" \
        transformer_collective_network.transformer_encoder.representation_transformer.model_path="${TRANSFORMER_CKPT_PATH}" \
        transformer_collective_network.transformer_encoder.prediction_head_cls.model_path="${TRANSFORMER_CKPT_PATH}"
    done
  done
}

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "SEED=${SEED}"
echo "STEP_SCALE=${STEP_SCALE}"
echo "RUN_STAGES=${RUN_STAGES}"
echo "SEEN_ROBOTS=${SEEN_ROBOT_LIST}"
echo "EVAL_ROBOTS=${EVAL_ROBOT_LIST}"
echo "TASKS=${TASK_LIST}"

stage_enabled workers && train_workers
stage_enabled distill && run_online_distill
stage_enabled split && split_distill_buffers
stage_enabled transformer && train_transformer
stage_enabled adapter && train_predictive_adapter
stage_enabled collective && train_collective_policy
stage_enabled eval && run_eval

echo
echo "Pipeline finished. Logs are under: ${SAVE_DIR}"
echo "Stage logs are under: ${STAGE_LOG_DIR}"
