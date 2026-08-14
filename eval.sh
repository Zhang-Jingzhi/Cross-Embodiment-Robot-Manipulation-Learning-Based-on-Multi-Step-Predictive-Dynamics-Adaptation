#!/bin/bash
# ==============================================================================
# PACE: Predictive Adaptation for Collective Embodiments Learning
# Consolidated Evaluation Script
#
# Usage:
#   bash eval.sh                          # Evaluate all robots on all tasks
#   bash eval.sh sawyer                   # Evaluate a specific robot
#   bash eval.sh sawyer reach-v2          # Evaluate a specific robot+task
#   bash eval.sh sawyer 3 my_exp          # Evaluate with seed=3 and experiment=my_exp
#   bash eval.sh --all setup.seed=5       # Evaluate all with extra args
#
# Environment variables:
#   EXPERIMENT_NAME  — experiment identifier (default: "none")
#   SEED             — random seed (default: "1")
#   TRANSFORMER_PATH — override transformer checkpoint path (default: uses config)
# ==============================================================================

set -euo pipefail

# Project root: default to the directory containing this script.
# Can still be overridden by exporting PROJECT_ROOT manually.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PROJECT_ROOT

### ===================== Configurable Variables ===================== ###
# Experiment name — controls which experiment model directory is used.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-task_dyn_true}"

# Seed for reproducibility
SEED="${SEED:-3}"

# Transformer model path — override the default representation transformer checkpoint.
# If empty, the default path from the config YAML will be used.
TRANSFORMER_PATH="${TRANSFORMER_PATH:-"${PROJECT_ROOT}/Transformer_RNN/checkpoints_task_dyn_true_seed_3/representation_cls_transformer_checkpoint.pth"}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="python"
fi

SCRIPT_EXTRA_ARGS=()
EVAL_ROBOT=""
EVAL_TASK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            shift
            ;;
        --*)
            SCRIPT_EXTRA_ARGS+=("$1")
            shift
            ;;
        *=*)
            SCRIPT_EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            if [[ -z "$EVAL_ROBOT" ]]; then
                EVAL_ROBOT="$1"
            elif [[ -z "$EVAL_TASK" ]]; then
                EVAL_TASK="$1"
            else
                SCRIPT_EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Default robots and tasks
ALL_ROBOTS=("sawyer" "panda" "kuka" "ur5e" "ur10e" "xarm7" "unitree_z1" "gen3" "viperx")
ALL_TASKS=("reach-v2" "push-v2" "peg-insert-side-v2" "door-open-v2"
           "window-open-v2" "window-close-v2" "drawer-open-v2"
           "button-press-topdown-v2" "faucet-open-v2" "pick-place-v2")

# --- Evaluate collective network on a single robot+task ---
evaluate_col_agent(){
    local robot_type="$1"
    local task_name="$2"
    if [[ -z "$task_name" ]]; then
        echo "Usage: evaluate_col_agent <robot_type> <task-name>"
        return 1
    fi

    local tf_args=()
    if [[ -n "$TRANSFORMER_PATH" ]]; then
        tf_args+=(
            "transformer_collective_network.transformer_encoder.representation_transformer.model_path=${TRANSFORMER_PATH}"
            "transformer_collective_network.transformer_encoder.prediction_head_cls.model_path=${TRANSFORMER_PATH}"
        )
    fi

    echo "Evaluating collective network: robot=${robot_type}, task=${task_name}, experiment=${EXPERIMENT_NAME}"
    local log_dir="${PROJECT_ROOT}/logs/results/col/${task_name}_${EXPERIMENT_NAME}"
    mkdir -p "$log_dir"
    local log_file="${log_dir}/eval_${robot_type}_seed_${SEED}.log"

    "$PYTHON_BIN" -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.robot_type="${robot_type}" \
        experiment.mode=evaluate_collective_transformer \
        env.benchmark.env_name="$task_name" \
        experiment.evaluate_transformer="collective_network" \
        experiment.experiment="${EXPERIMENT_NAME}" \
        setup.seed="${SEED}" \
        ${tf_args[@]+"${tf_args[@]}"} \
        "${SCRIPT_EXTRA_ARGS[@]}" > "$log_file" 2>&1
}

# --- Evaluate expert worker on a single robot+task ---
evaluate_expert(){
    local robot_type="$1"
    local task_name="$2"
    if [[ -z "$task_name" ]]; then
        echo "Usage: evaluate_expert <robot_type> <task-name>"
        return 1
    fi

    echo "Evaluating expert: robot=${robot_type}, task=${task_name}"
    local result_path="${PROJECT_ROOT}/logs/results/worker/${task_name}"
    mkdir -p "$(dirname "$result_path")"

    "$PYTHON_BIN" -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=evaluate_collective_transformer \
        experiment.robot_type="${robot_type}" \
        env.benchmark.env_name="${task_name}" \
        experiment.evaluate_transformer="agent" \
        experiment.experiment="${EXPERIMENT_NAME}" \
        setup.seed="${SEED}" \
        "${SCRIPT_EXTRA_ARGS[@]}" | tee -a "$result_path"
}

# --- Evaluate predictive adapter ---
evaluate_predictive_adapter(){
    local tf_args=()
    if [[ -n "$TRANSFORMER_PATH" ]]; then
        tf_args+=(
            "transformer_collective_network.transformer_encoder.representation_transformer.model_path=${TRANSFORMER_PATH}"
            "transformer_collective_network.transformer_encoder.prediction_head_cls.model_path=${TRANSFORMER_PATH}"
        )
    fi

    echo "=== Evaluating predictive adapter ==="
    "$PYTHON_BIN" -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=evaluate_predictive_adapter \
        experiment.experiment="${EXPERIMENT_NAME}" \
        setup.seed="${SEED}" \
        ${tf_args[@]+"${tf_args[@]}"} \
        "${SCRIPT_EXTRA_ARGS[@]}"
}

# --- Print results summary ---
print_results(){
    echo
    echo "========== Evaluation Results =========="
    echo "RESULTS expert:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/worker 2>/dev/null | grep Evaluation || echo "  (no results)"
    echo "RESULTS collective network:"
    grep -r Evaluation ${PROJECT_ROOT}/logs/results/col 2>/dev/null | grep Evaluation || echo "  (no results)"
    echo "========================================"
    echo
}

### ===================== Main Execution ===================== ###

mkdir -p ${PROJECT_ROOT}/logs/results/worker
mkdir -p ${PROJECT_ROOT}/logs/results/col
mkdir -p "${PROJECT_ROOT}/logs/results/worker"
mkdir -p "${PROJECT_ROOT}/logs/results/col"
echo "Using python interpreter: ${PYTHON_BIN}"

if [[ -n "$EVAL_ROBOT" && -n "$EVAL_TASK" ]]; then
    # Evaluate specific robot + task
    evaluate_col_agent "$EVAL_ROBOT" "$EVAL_TASK"
elif [[ -n "$EVAL_ROBOT" ]]; then
    # Evaluate specific robot on all tasks
    for task in "${ALL_TASKS[@]}"; do
        evaluate_col_agent "$EVAL_ROBOT" "$task"
    done
else
    # Evaluate all robots on all tasks
    for robot in "${ALL_ROBOTS[@]}"; do
        for task in "${ALL_TASKS[@]}"; do
            evaluate_col_agent "$robot" "$task"
        done
    done
fi

print_results
