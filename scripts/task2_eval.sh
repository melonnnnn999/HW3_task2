#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/task2_act_calvin.yaml}
OUTPUT_DIR=${2:-outputs/task2_act_calvin}
RESULTS_DIR=${3:-results}
PYTHON_BIN=${PYTHON:-python}

"${PYTHON_BIN}" scripts/evaluate_action_error.py \
  --config "${CONFIG}" \
  --checkpoint "${OUTPUT_DIR}/act_calvin_a/checkpoints/last/pretrained_model" \
  --run-name act_calvin_a \
  --split a_val \
  --output-dir "${RESULTS_DIR}"

"${PYTHON_BIN}" scripts/evaluate_action_error.py \
  --config "${CONFIG}" \
  --checkpoint "${OUTPUT_DIR}/act_calvin_a/checkpoints/last/pretrained_model" \
  --run-name act_calvin_a \
  --split d_test \
  --output-dir "${RESULTS_DIR}"

"${PYTHON_BIN}" scripts/evaluate_action_error.py \
  --config "${CONFIG}" \
  --checkpoint "${OUTPUT_DIR}/act_calvin_abc/checkpoints/last/pretrained_model" \
  --run-name act_calvin_abc \
  --split abc_val \
  --output-dir "${RESULTS_DIR}"

"${PYTHON_BIN}" scripts/evaluate_action_error.py \
  --config "${CONFIG}" \
  --checkpoint "${OUTPUT_DIR}/act_calvin_abc/checkpoints/last/pretrained_model" \
  --run-name act_calvin_abc \
  --split d_test \
  --output-dir "${RESULTS_DIR}"

"${PYTHON_BIN}" scripts/summarize_results.py --results-dir "${RESULTS_DIR}" --output "${RESULTS_DIR}/task2_summary.md"
