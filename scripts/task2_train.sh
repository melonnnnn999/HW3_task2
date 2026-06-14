#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/task2_act_calvin.yaml}
RUN_KIND=${2:-a}
PYTHON_BIN=${PYTHON:-python}

if [[ "${RUN_KIND}" != "a" && "${RUN_KIND}" != "abc" ]]; then
  echo "Usage: bash scripts/task2_train.sh [config.yaml] [a|abc]" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/train_act_calvin.py --config "${CONFIG}" --run-kind "${RUN_KIND}"
