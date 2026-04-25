#!/usr/bin/env bash
set -euo pipefail

# Batch wrapper around one_sample_BODex_ur10e_shadow_gravity.sh
#
# It runs the existing one-sample script N times, with visualization disabled,
# and aggregates EKF metrics across successful runs.
#
# Usage:
#   N_RUNS=20 bash script/batch_repeat_one_sample_gravity.sh
#
# Optional env vars:
#   N_RUNS=20
#   SEED_BASE=12345
#   BATCH_OUT=output/debug_batch_one_sample_ur10e_shadow
#   BATCH_VIEWER=True
#   BATCH_POST_LIFT_STEPS=2000  # auto-close viewer after fixed EKF hold steps
#   BATCH_REALTIME_PLOT=False

N_RUNS="${N_RUNS:-10}"
SEED_BASE="${SEED_BASE:-$RANDOM}"
BATCH_OUT="${BATCH_OUT:-output/debug_batch_one_sample_ur10e_shadow}"
BATCH_VIEWER="${BATCH_VIEWER:-False}"
BATCH_POST_LIFT_STEPS="${BATCH_POST_LIFT_STEPS:-2000}"
BATCH_REALTIME_PLOT="${BATCH_REALTIME_PLOT:-False}"

SINGLE_OUT="output/debug_one_ur10e_shadow"
RUN_DIR="${BATCH_OUT}/runs"
LOG_DIR="${BATCH_OUT}/logs"
META_FILE="${BATCH_OUT}/run_meta.jsonl"

rm -rf "$BATCH_OUT"
mkdir -p "$RUN_DIR" "$LOG_DIR"
: > "$META_FILE"

echo "[INFO] N_RUNS=${N_RUNS}"
echo "[INFO] SEED_BASE=${SEED_BASE}"
echo "[INFO] BATCH_OUT=${BATCH_OUT}"
echo "[INFO] BATCH_VIEWER=${BATCH_VIEWER}"
echo "[INFO] BATCH_POST_LIFT_STEPS=${BATCH_POST_LIFT_STEPS}"
echo "[INFO] BATCH_REALTIME_PLOT=${BATCH_REALTIME_PLOT}"

for i in $(seq 1 "$N_RUNS"); do
  seed=$((SEED_BASE + i))
  tag=$(printf "run_%03d" "$i")
  run_case_dir="${RUN_DIR}/${tag}"
  mkdir -p "$run_case_dir"
  echo "[INFO] ===== ${tag} seed=${seed} ====="

  if SEED="$seed" DEBUG_VIEWER=False SAMPLE_DEBUG_VIEWER=False bash script/one_sample_BODex_ur10e_shadow_gravity.sh > "${LOG_DIR}/${tag}.log" 2>&1; then
    run_ok=true
  else
    run_ok=false
  fi

  # Always run one deterministic EKF eval pass for metrics.
  # Viewer is optional and controlled only by BATCH_VIEWER.
  if [ "$run_ok" = "true" ]; then
    if [ -d "${SINGLE_OUT}/succgrasp" ]; then
      viewer_flag="False"
      autoclose_flag="False"
      if [ "$(echo "$BATCH_VIEWER" | tr '[:upper:]' '[:lower:]')" = "true" ]; then
        viewer_flag="True"
        autoclose_flag="True"
      fi
      python src/main.py \
        seed="$seed" \
        setting=tabletop \
        hand=ur10e_shadow \
        task=eval_gravity \
        exp_name=debug_one \
        grasp_dir="${SINGLE_OUT}/succgrasp" \
        eval_dir="${SINGLE_OUT}/evaluation_batch_view_one" \
        succ_dir="${SINGLE_OUT}/succgrasp_batch_view_one" \
        task.max_num=1 \
        task.debug_viewer="$viewer_flag" \
        task.debug_viewer_autoclose="$autoclose_flag" \
        task.ekf_input_post_lift_steps="$BATCH_POST_LIFT_STEPS" \
        task.ekf_pose_eval_realtime_plot="$BATCH_REALTIME_PLOT" \
        n_worker=1 >> "${LOG_DIR}/${tag}.log" 2>&1 || true
    fi
  fi

  python3 - "$SINGLE_OUT" "$run_case_dir" "$META_FILE" "$tag" "$seed" "$run_ok" <<'PY'
import glob
import json
import os
import shutil
import sys
import numpy as np

single_out = sys.argv[1]
run_case_dir = sys.argv[2]
meta_file = sys.argv[3]
tag = sys.argv[4]
seed = int(sys.argv[5])
run_ok = sys.argv[6].lower() == "true"

grasp_paths = sorted(glob.glob(os.path.join(single_out, "graspdata", "**", "*.npy"), recursive=True))
obj_name = None
if grasp_paths:
    try:
        d0 = np.load(grasp_paths[0], allow_pickle=True).item()
        obj_name = os.path.basename(os.path.normpath(str(d0.get("obj_path", "unknown_obj"))))
    except Exception:
        obj_name = None

succ_paths = sorted(glob.glob(os.path.join(single_out, "succgrasp", "**", "*.npy"), recursive=True))
has_success = len(succ_paths) > 0

metrics_npz = os.path.join(single_out, "metrics", "metrics.npz")
metrics_png = os.path.join(single_out, "metrics", "metrics.png")
has_metrics = os.path.exists(metrics_npz)
pos_mean = None
rot_mean = None
pos_final = None
rot_final = None
if has_metrics:
    shutil.copy2(metrics_npz, os.path.join(run_case_dir, "metrics.npz"))
    if os.path.exists(metrics_png):
        shutil.copy2(metrics_png, os.path.join(run_case_dir, "metrics.png"))
    try:
        d = np.load(metrics_npz)
        pos = np.asarray(d["pos_err"], dtype=float)
        rot = np.asarray(d["rot_err_deg"], dtype=float)
        pos_mean = float(np.mean(pos)) if pos.size > 0 else None
        rot_mean = float(np.mean(rot)) if rot.size > 0 else None
        pos_final = float(pos[-1]) if pos.size > 0 else None
        rot_final = float(rot[-1]) if rot.size > 0 else None
    except Exception:
        pass

row = {
    "tag": tag,
    "seed": seed,
    "run_ok": run_ok,
    "object_name": obj_name,
    "has_success": has_success,
    "has_metrics": has_metrics,
    "pos_mean": pos_mean,
    "rot_mean_deg": rot_mean,
    "pos_final": pos_final,
    "rot_final_deg": rot_final,
}
with open(meta_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[INFO] {tag}: object={obj_name}, success={has_success}, metrics={has_metrics}")
PY
done

python3 - "$META_FILE" "$BATCH_OUT" <<'PY'
import csv
import json
import os
import sys
import numpy as np

meta_file = sys.argv[1]
out_dir = sys.argv[2]

rows = []
with open(meta_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

if not rows:
    raise SystemExit("[ERROR] no run metadata collected")

csv_path = os.path.join(out_dir, "summary.csv")
fields = [
    "tag", "seed", "run_ok", "object_name",
    "has_success", "has_metrics", "pos_mean", "rot_mean_deg", "pos_final", "rot_final_deg"
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fields})

with_metrics = [r for r in rows if r.get("has_metrics")]
objs = sorted({r.get("object_name") for r in rows if r.get("object_name")})
overall = {
    "num_runs": len(rows),
    "num_run_ok": int(sum(bool(r.get("run_ok")) for r in rows)),
    "num_with_success": int(sum(bool(r.get("has_success")) for r in rows)),
    "num_with_metrics": len(with_metrics),
    "num_distinct_objects": len(objs),
    "distinct_objects": objs,
    "pos_mean_avg": float(np.mean([r["pos_mean"] for r in with_metrics])) if with_metrics else None,
    "rot_mean_deg_avg": float(np.mean([r["rot_mean_deg"] for r in with_metrics])) if with_metrics else None,
    "pos_final_avg": float(np.mean([r["pos_final"] for r in with_metrics])) if with_metrics else None,
    "rot_final_deg_avg": float(np.mean([r["rot_final_deg"] for r in with_metrics])) if with_metrics else None,
}
overall_path = os.path.join(out_dir, "overall_summary.json")
with open(overall_path, "w", encoding="utf-8") as f:
    json.dump(overall, f, indent=2, ensure_ascii=False)

print(f"[INFO] summary.csv: {csv_path}")
print(f"[INFO] overall_summary.json: {overall_path}")
print(f"[INFO] overall: {overall}")
PY

echo "[INFO] Done. Batch results: ${BATCH_OUT}"
