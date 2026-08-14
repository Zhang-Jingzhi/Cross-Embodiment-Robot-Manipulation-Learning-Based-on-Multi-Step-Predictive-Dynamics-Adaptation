#!/bin/bash
# ==============================================================================
# PACE: Predictive Adaptation for Collective Embodiments Learning
# Consolidated Training Script
#
# This script provides helper functions for the full training pipeline:
#   1. Train expert workers on individual tasks
#   2. Run online distillation (or generate noise-injected data)
#   3. Train the predictive adapter
#   4. Train the representation transformer (in Transformer_RNN/)
#   5. Distill collective transformer
#
# Usage:
#   source train.sh          # Load helper functions
#   bash train.sh            # Run the full example pipeline
# ==============================================================================

set -euo pipefail

### ===================== Configurable Variables ===================== ###
# Experiment name — controls which experiment model directory is used.
# Change this to switch between different experiment configurations.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-none}"

# Seed for reproducibility
SEED="${SEED:-1}"

# Transformer model path — override the default representation transformer checkpoint.
# If empty, the default path from the config YAML will be used.
TRANSFORMER_PATH="${TRANSFORMER_PATH:-}"

### ===================== Helper Functions ===================== ###

# --- Remove saved models ---
rm_model(){
    local robot_type="$1"
    local task_name="$2"
    if [[ -z "$task_name" ]]; then
        echo "Usage: rm_model <robot_type> <task-name>"
        return 1
    fi
    echo "Removing task: $task_name"
    rm -f logs/experiment_test/model_dir/model_${robot_type}_${task_name}_seed_1/*
    rm -f logs/experiment_test/buffer/buffer/buffer_${robot_type}_${task_name}_seed_1/*
    rm -f logs/experiment_test/buffer/buffer_distill/buffer_distill_${robot_type}_${task_name}_seed_1/0_*
    rm -f logs/experiment_test/buffer/buffer_distill_tmp/buffer_distill_tmp_${robot_type}_${task_name}_seed_1/*
}

rm_col_model(){
    echo "Removing col_model!"
    rm -f logs/experiment_test/model_dir/model_col/*
}

# --- Step 1: Train expert worker ---
train_task(){
    local robot_type="$1"
    local task_name="$2"
    local nr_steps="$3"
    shift 3

    if [[ -z "$task_name" || -z "$nr_steps" ]]; then
        echo "Usage: train_task <robot_type> <task-name> <nr-steps> [additional args]"
        return 1
    fi

    echo "=== Training expert on task: $task_name | robot: $robot_type | steps: $nr_steps ==="
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=train_worker \
        experiment.experiment="${EXPERIMENT_NAME}" \
        experiment.robot_type="${robot_type}" \
        env.benchmark.env_name="${task_name}" \
        experiment.num_train_steps="${nr_steps}" \
        setup.seed="${SEED}" \
        "$@"
}

# --- Step 2a: Online distillation ---
online_distill(){
    local robot_type="$1"
    local task_name="$2"
    shift 2

    if [[ -z "$task_name" ]]; then
        echo "Usage: online_distill <robot_type> <task-name> [additional args]"
        return 1
    fi

    echo "=== Online distillation: $task_name | robot: $robot_type ==="
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=online_distill_collective_transformer \
        experiment.experiment="${EXPERIMENT_NAME}" \
        env.benchmark.env_name="${task_name}" \
        experiment.robot_type="${robot_type}" \
        setup.seed="${SEED}" \
        "$@"
}

# --- Step 2b: Generate noise-injected data (alternative to online distillation) ---
generate_distill_data(){
    local robot_type=$1
    local task_name=$2
    local gen_steps=$3

    if [[ -z "$task_name" || -z "$gen_steps" ]]; then
        echo "Usage: generate_distill_data <robot_type> <task-name> <steps>"
        return 1
    fi

    echo "=== Generating noise-injected distill data ==="
    echo "  Robot: ${robot_type} | Task: ${task_name} | Steps: ${gen_steps}"
    python3 main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=generate_distill_data \
        experiment.robot_type=${robot_type} \
        env.benchmark.env_name="${task_name}" \
        experiment.num_distill_gen_steps=${gen_steps}
}

# --- Step 3a: Train predictive adapter ---
train_predictive_adapter(){
    local tf_args=()
    if [[ -n "$TRANSFORMER_PATH" ]]; then
        tf_args+=(
            "transformer_collective_network.transformer_encoder.representation_transformer.model_path=${TRANSFORMER_PATH}"
            "transformer_collective_network.transformer_encoder.prediction_head_cls.model_path=${TRANSFORMER_PATH}"
        )
    fi

    echo "=== Training predictive adapter ==="
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=train_predictive_adapter \
        experiment.experiment="${EXPERIMENT_NAME}" \
        transformer_collective_network.predictive_adapter.load_on_init=False \
        setup.seed="${SEED}" \
        ${tf_args[@]+"${tf_args[@]}"} \
        "$@"
}

# --- Step 3b: Train representation transformer (in Transformer_RNN/) ---
train_transformer(){
    echo "=== Creating transformer dataset ==="
    python3 Transformer_RNN/dataset_tf.py

    echo "=== Training representation transformer ==="
    python3 Transformer_RNN/RepresentationTransformerWithCLS.py "$@"
}

# --- Step 4: Distill collective transformer ---
distill_collective(){
    local tf_args=()
    if [[ -n "$TRANSFORMER_PATH" ]]; then
        tf_args+=(
            "transformer_collective_network.transformer_encoder.representation_transformer.model_path=${TRANSFORMER_PATH}"
            "transformer_collective_network.transformer_encoder.prediction_head_cls.model_path=${TRANSFORMER_PATH}"
        )
    fi

    echo "=== Distilling collective transformer ==="
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=distill_collective_transformer \
        experiment.experiment="${EXPERIMENT_NAME}" \
        setup.seed="${SEED}" \
        ${tf_args[@]+"${tf_args[@]}"} \
        "$@"
}

# --- Buffer splitting utilities ---
split_buffer(){
    local robot_type="$1"
    local task_name="$2"
    if [[ -z "$task_name" ]]; then
        echo "Usage: split_buffer <robot_type> <task-name>"
        return 1
    fi
    python split_buffer_files.py \
        --source ${PROJECT_ROOT}/logs/experiment_test/buffer/buffer_distill/buffer_distill_${robot_type}_${task_name}_seed_5 \
        --train  ${PROJECT_ROOT}/Transformer_RNN/dataset_3/train/buffer_distill_${robot_type}_${task_name}_seed_5 \
        --val    ${PROJECT_ROOT}/Transformer_RNN/dataset_3/validation/buffer_distill_${robot_type}_${task_name}_seed_5
}

split_online_buffer(){
    local robot_type="$1"
    local task_name="$2"
    if [[ -z "$task_name" ]]; then
        echo "Usage: split_online_buffer <robot_type> <task-name>"
        return 1
    fi
    python split_buffer_files.py \
        --source logs/experiment_test/buffer/online_buffer_${robot_type}_${task_name} \
        --train  logs/experiment_test/buffer/collective_buffer_3/train/online_buffer_${robot_type}_${task_name}_seed_5 \
        --val    logs/experiment_test/buffer/collective_buffer_3/validation/online_buffer_${robot_type}_${task_name}_seed_5
}

# --- Print results ---
print_results(){
    echo
    echo "RESULTS expert:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/worker 2>/dev/null | grep Evaluation || echo "  (no results)"
    echo "RESULTS collective network:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/col 2>/dev/null | grep Evaluation || echo "  (no results)"
    echo
}


### ===================== Source Mode Guard ===================== ###
# If this script is sourced (not executed), return here so functions
# are available in the current shell without running the pipeline.

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "train.sh sourced — helper functions loaded."
    return 0
fi


### ===================== Example Full Pipeline ===================== ###

echo "============================================"
echo " PACE Full Training Pipeline"
echo "============================================"

ROBOTS=("sawyer" "ur10e" "xarm7" "unitree_z1" "gen3")
TASKS=("reach-v2" "push-v2" "pick-place-v2" "door-open-v2" "drawer-open-v2"
       "button-press-topdown-v2" "peg-insert-side-v2" "window-open-v2" "window-close-v2")

mkdir -p ${PROJECT_ROOT}/logs/results/worker
mkdir -p ${PROJECT_ROOT}/logs/results/col

# --- Step 1: Train experts ---
echo ">>> Step 1: Training expert workers..."
train_task sawyer reach-v2 200000 worker.builder.actor_update_freq=1
train_task sawyer push-v2 900000
train_task sawyer pick-place-v2 2400000
train_task sawyer door-open-v2 1000000
train_task sawyer drawer-open-v2 500000
train_task sawyer drawer-close-v2 200000
train_task sawyer button-press-topdown-v2 500000
train_task sawyer peg-insert-side-v2 1300000
train_task sawyer window-open-v2 300000
train_task sawyer window-close-v2 400000

# --- Step 2: Online distillation ---
echo ">>> Step 2: Online distillation..."
for task in "${TASKS[@]}"; do
    online_distill sawyer "$task"
done

# --- Step 2 (cont.): Split buffers for all robots ---
echo ">>> Splitting buffers..."
for robot in "${ROBOTS[@]}"; do
    for task in "${TASKS[@]}"; do
        split_buffer "$robot" "$task"
        split_online_buffer "$robot" "$task"
    done
done

# --- Step 3a: Train representation transformer ---
echo ">>> Step 3a: Training representation transformer..."
train_transformer

# --- Step 3b: Train predictive adapter ---
echo ">>> Step 3b: Training predictive adapter..."
train_predictive_adapter

# --- Step 4: Distill collective transformer ---
echo ">>> Step 4: Distilling collective transformer..."
distill_collective

echo "============================================"
echo " Training pipeline complete!"
echo "============================================"