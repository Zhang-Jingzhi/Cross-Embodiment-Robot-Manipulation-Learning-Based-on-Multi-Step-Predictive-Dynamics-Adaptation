#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-pace_repro}"
PORT="${PORT:-6006}"

SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/logs/${RUN_NAME}}"
TRANSFORMER_TB_ROOT="${TRANSFORMER_TB_ROOT:-${PROJECT_ROOT}/Transformer_RNN/tensorboard_log}"

echo "TensorBoard save_dir: ${SAVE_DIR}"
echo "TensorBoard transformer root: ${TRANSFORMER_TB_ROOT}"
echo "TensorBoard port: ${PORT}"

tensorboard \
  --bind_all \
  --port "${PORT}" \
  --logdir_spec "mtrl:${SAVE_DIR},transformer:${TRANSFORMER_TB_ROOT}"
