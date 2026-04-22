#!/usr/bin/env bash
set -e

# BODex raw output folder (contains *_grasp.npy / *_mogen.npy)
# Override if needed:
# BODEX_RAW_DIR=... bash script/one_sample_BODex_botyard.sh
BODEX_RAW_DIR="${BODEX_RAW_DIR:-../BODex/src/curobo/content/assets/output/sim_botyard/fc/debug/grasp_data}"

# Evaluation setting for DexGraspBench: fc or tabletop
SETTING="${SETTING:-fc}"

# Leave empty by default: conversion code will auto-derive Botyard root transform
# from URDF/XML (pmbase <-> palm). Set these env vars only when manual override is needed.
BOTYARD_FIXED_BIAS_VEC="${BOTYARD_FIXED_BIAS_VEC:-}"
BOTYARD_FIXED_RPY_BIAS="${BOTYARD_FIXED_RPY_BIAS:-}"

# Keep deterministic by default; set RANDOM_EACH_RUN=True for per-run random seed.
SEED="${SEED:-3025}"
RANDOM_EACH_RUN="${RANDOM_EACH_RUN:-False}"
if [[ "${RANDOM_EACH_RUN,,}" == "true" ]]; then
  SEED="$(date +%s)"
fi
echo "[INFO] Using SEED=${SEED}"

# Number of raw files to format:
#   1  -> one object (fast, default)
#  -1  -> all objects under BODEX_RAW_DIR
FORMAT_MAX_NUM="${FORMAT_MAX_NUM:-1}"

# Evaluate all converted grasps from the sampled raw file by default.
EVAL_MAX_NUM="${EVAL_MAX_NUM:--1}"

# Toggle viewer for stability/debug convenience.
DEBUG_VIEWER="${DEBUG_VIEWER:-False}"
# Botyard external-force stages for FC simulation (shadow-like non-zero by default).
BOTYARD_FORCE_STAGES="${BOTYARD_FORCE_STAGES:-10.0}"

# NOTE:
# This script requires a DexGraspBench hand config:
#   config/hand/botyard.yaml
# and a valid hand xml referenced by that yaml.
# Example:
#   xml_path: assets/hand/botyard/right_hand.xml
#   mocap: True

if [ -f "config/hand/botyard.yaml" ]; then
  :
elif [ -f "config/hand/botyard.yml" ]; then
  :
else
  echo "[ERROR] Missing config/hand/botyard.yaml or config/hand/botyard.yml"
  echo "Please add botyard hand config first, then rerun this script."
  exit 1
fi

# Output folder for this exp (clear all logs/results for clean comparison)
EXP_OUT="output/debug_one_botyard"
rm -rf "$EXP_OUT"

# 1) Convert one raw sample to DexGraspBench format.
if [ -n "$BOTYARD_FIXED_BIAS_VEC" ] || [ -n "$BOTYARD_FIXED_RPY_BIAS" ]; then
  BOTYARD_FIXED_BIAS_VEC="${BOTYARD_FIXED_BIAS_VEC}" BOTYARD_FIXED_RPY_BIAS="${BOTYARD_FIXED_RPY_BIAS}" python src/main.py \
    seed="$SEED" \
    setting="$SETTING" \
    hand=botyard \
    task=format \
    exp_name=debug_one \
    task.data_name=BODex \
    task.max_num="$FORMAT_MAX_NUM" \
    task.data_path="$BODEX_RAW_DIR"
else
  python src/main.py \
    seed="$SEED" \
    setting="$SETTING" \
    hand=botyard \
    task=format \
    exp_name=debug_one \
    task.data_name=BODex \
    task.max_num="$FORMAT_MAX_NUM" \
    task.data_path="$BODEX_RAW_DIR"
fi

# 2) Evaluate one sample with interactive viewer.
#    MuJoCo window stays alive until you close it manually.
if [ -n "$BOTYARD_FIXED_BIAS_VEC" ] || [ -n "$BOTYARD_FIXED_RPY_BIAS" ]; then
  BOTYARD_FIXED_BIAS_VEC="${BOTYARD_FIXED_BIAS_VEC}" BOTYARD_FIXED_RPY_BIAS="${BOTYARD_FIXED_RPY_BIAS}" BOTYARD_FORCE_STAGES="${BOTYARD_FORCE_STAGES}" python src/main.py \
    seed="$SEED" \
    setting="$SETTING" \
    hand=botyard \
    task=eval \
    exp_name=debug_one \
    task.max_num="$EVAL_MAX_NUM" \
    task.simulation_metrics.max_pene=0.05 \
    task.debug_viewer="$DEBUG_VIEWER" \
    n_worker=1
else
  BOTYARD_FORCE_STAGES="${BOTYARD_FORCE_STAGES}" python src/main.py \
    seed="$SEED" \
    setting="$SETTING" \
    hand=botyard \
    task=eval \
    exp_name=debug_one \
    task.max_num="$EVAL_MAX_NUM" \
    task.simulation_metrics.max_pene=0.05 \
    task.debug_viewer="$DEBUG_VIEWER" \
    n_worker=1
fi

