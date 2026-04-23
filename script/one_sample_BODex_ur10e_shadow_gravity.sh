#!/usr/bin/env bash
set -e

# BODex raw root folder (contains many objects with *_grasp.npy / *_mogen.npy)
BODEX_RAW_ROOT="${BODEX_RAW_ROOT:-../BODex/src/curobo/content/assets/output/sim_shadow/tabletop/debug/grasp_data}"

# Output folder for this exp. Clean before run so eval won't pick old npy.
OUT_DIR="output/debug_one_ur10e_shadow"
rm -rf "$OUT_DIR"

# Random pick: task.max_num=1 uses np.random.permutation; set seed per run (override: SEED=123 bash script/...)
SEED="${SEED:-$RANDOM}"
# Evaluate all converted grasps from the selected object by default.
EVAL_MAX_NUM="${EVAL_MAX_NUM:--1}"
# Convert only one raw file under the selected object by default (usually 20 grasps).
FORMAT_MAX_NUM="${FORMAT_MAX_NUM:-1}"
# Batch evaluation: viewer off by default.
DEBUG_VIEWER="${DEBUG_VIEWER:-False}"
# Sample visualization: viewer on by default.
SAMPLE_DEBUG_VIEWER="${SAMPLE_DEBUG_VIEWER:-True}"

# 0) Randomly pick one object folder, then run all grasps under this object.
SELECTED_OBJECT_DIR="$(
python - "$BODEX_RAW_ROOT" "$SEED" <<'PY'
import os
import sys
import glob
import numpy as np

root = sys.argv[1]
seed = int(sys.argv[2])
files = glob.glob(os.path.join(root, "**", "*_mogen.npy"), recursive=True)
if len(files) == 0:
    files = glob.glob(os.path.join(root, "**", "*_grasp.npy"), recursive=True)
if len(files) == 0:
    raise SystemExit(f"[ERROR] no *_mogen.npy or *_grasp.npy found under: {root}")

obj_dirs = sorted({os.path.join(root, os.path.relpath(p, root).split(os.sep)[0]) for p in files})
rng = np.random.default_rng(seed)
picked = obj_dirs[rng.integers(0, len(obj_dirs))]
print(picked)
PY
)"
echo "[INFO] Using SEED=${SEED}"
echo "[INFO] Selected object folder: ${SELECTED_OBJECT_DIR}"

# 1) Convert one raw sample to DexGraspBench format.
python src/main.py \
  seed="$SEED" \
  setting=tabletop \
  hand=ur10e_shadow \
  task=format \
  exp_name=debug_one \
  task.data_name=BODex \
  task.max_num="$FORMAT_MAX_NUM" \
  task.data_path="$SELECTED_OBJECT_DIR"

# 2) Evaluate all converted grasps under this selected object (real gravity mode).
python src/main.py \
  seed="$SEED" \
  setting=tabletop \
  hand=ur10e_shadow \
  task=eval_gravity \
  exp_name=debug_one \
  task.max_num="$EVAL_MAX_NUM" \
  task.debug_viewer="$DEBUG_VIEWER" \
  n_worker=1

# 3) Print one random success path (for logging) and visualize one random success by default.
SAMPLED_SUCCESS_PATH="$(
python - "$OUT_DIR" "$SEED" <<'PY'
import os
import sys
import glob
import numpy as np

out_dir = sys.argv[1]
seed = int(sys.argv[2])
rng = np.random.default_rng(seed)
succ = sorted(glob.glob(os.path.join(out_dir, "succgrasp", "**", "*.npy"), recursive=True))
if len(succ) > 0:
    p = succ[rng.integers(0, len(succ))]
    print(p)
else:
    print("")
PY
)"
if [ -n "$SAMPLED_SUCCESS_PATH" ]; then
  echo "[INFO] Random sampled SUCCESS: ${SAMPLED_SUCCESS_PATH}"
  python src/main.py \
    seed="$SEED" \
    setting=tabletop \
    hand=ur10e_shadow \
    task=eval_gravity \
    exp_name=debug_one \
    grasp_dir="$OUT_DIR/succgrasp" \
    eval_dir="$OUT_DIR/evaluation_view_one" \
    succ_dir="$OUT_DIR/succgrasp_view_one" \
    task.max_num=1 \
    task.debug_viewer="$SAMPLE_DEBUG_VIEWER" \
    n_worker=1
else
  echo "[WARN] No success sample found, skip viewer stage."
fi
