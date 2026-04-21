#!/usr/bin/env bash
set -euo pipefail

# Lightweight preset for laptops:
# - Reduce worker count to avoid CPU / memory saturation.
# - Reduce visualization samples to avoid long rendering freezes.
# You can increase values later if resources allow.

N_WORKER="${N_WORKER:-4}"
VOBJ_MAX_NUM="${VOBJ_MAX_NUM:-5}"
VUSD_MAX_NUM="${VUSD_MAX_NUM:-5}"

python src/main.py task=eval exp_name=example hand=shadow n_worker="${N_WORKER}"                 # Evaluation
python src/main.py task=stat exp_name=example hand=shadow n_worker=2                              # Statistics calculation
python src/main.py task=vobj exp_name=example hand=shadow n_worker=2 task.max_num="${VOBJ_MAX_NUM}"  # Visualization with OBJ files
python src/main.py task=vusd exp_name=example hand=shadow n_worker=1 task.max_num="${VUSD_MAX_NUM}"  # Visualization with OpenUSD