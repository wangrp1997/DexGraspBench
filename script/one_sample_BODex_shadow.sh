#!/usr/bin/env bash
set -e

# BODex raw output folder (contains *_grasp.npy / *_mogen.npy)
# Override if needed:
# BODEX_RAW_DIR=... bash script/one_sample_BODex_shadow.sh
BODEX_RAW_DIR="${BODEX_RAW_DIR:-../BODex/src/curobo/content/assets/output/sim_shadow/fc/debug/grasp_data}"

# Evaluation setting for DexGraspBench: fc or tabletop
SETTING="${SETTING:-fc}"

# Output folder for this exp (must only contain this run's graspdata)
GRASP_OUT="output/debug_one_shadow/graspdata"
rm -rf "$GRASP_OUT"

# Random pick: task.max_num=1 uses np.random.permutation
SEED="${SEED:-$RANDOM}"

# 1) Convert one raw sample to DexGraspBench format.
python src/main.py \
  seed="$SEED" \
  setting="$SETTING" \
  hand=shadow \
  task=format \
  exp_name=debug_one \
  task.data_name=BODex \
  task.max_num=1 \
  task.data_path="$BODEX_RAW_DIR"

# 2) Evaluate one sample with interactive viewer.
#    MuJoCo window stays alive until you close it manually.
python src/main.py \
  seed="$SEED" \
  setting="$SETTING" \
  hand=shadow \
  task=eval \
  exp_name=debug_one \
  task.max_num=1 \
  task.debug_viewer=True \
  n_worker=1

