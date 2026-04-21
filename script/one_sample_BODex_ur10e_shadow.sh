#!/usr/bin/env bash
set -e

# BODex raw output folder (contains *_grasp.npy / *_mogen.npy)
BODEX_RAW_DIR="../BODex/src/curobo/content/assets/output/sim_shadow/tabletop/debug/grasp_data"

# Output folder for this exp (must only contain this run's graspdata so eval does not pick old npy)
GRASP_OUT="output/debug_one_ur10e_shadow/graspdata"
rm -rf "$GRASP_OUT"

# Random pick: task.max_num=1 uses np.random.permutation; set seed per run (override: SEED=123 bash script/...)
SEED="${SEED:-$RANDOM}"

# 1) Convert one raw sample to DexGraspBench format.
python src/main.py \
  seed="$SEED" \
  setting=tabletop \
  hand=ur10e_shadow \
  task=format \
  exp_name=debug_one \
  task.data_name=BODex \
  task.max_num=1 \
  task.data_path="$BODEX_RAW_DIR"

# 2) Evaluate one sample with interactive viewer (same SEED reproduces which index is chosen among *.npy).
python src/main.py \
  seed="$SEED" \
  setting=tabletop \
  hand=ur10e_shadow \
  task=eval \
  exp_name=debug_one \
  task.max_num=1 \
  task.debug_viewer=True \
  n_worker=1
