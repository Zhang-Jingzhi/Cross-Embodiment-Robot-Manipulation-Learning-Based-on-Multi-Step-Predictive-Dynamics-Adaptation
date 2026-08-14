#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_NAME="${RUN_NAME:-pace_repro}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/logs/${RUN_NAME}}"
STAGE_LOG_DIR="${STAGE_LOG_DIR:-${SAVE_DIR}/stage_logs}"

echo "Watching logs in ${SAVE_DIR}"
echo "Press Ctrl+C to stop."

tail -n 30 -F \
  "${STAGE_LOG_DIR}"/*.log \
  "${SAVE_DIR}/train.log" \
  "${SAVE_DIR}/eval.log" \
  "${SAVE_DIR}/col_train.log" \
  "${SAVE_DIR}/col_eval.log" 2>/dev/null
