#!/usr/bin/env bash
set -euo pipefail

# Batch wrapper around one_sample_BODex_ur10e_shadow_gravity.sh
#
# Retries until N_RUNS successful EKF metrics are collected (default 10).
#
# Usage:
#   N_RUNS=10 bash script/batch_ekf_one_per_object.sh
#
# Optional env vars:
#   N_RUNS=10              # target count of successful metrics runs
#   MAX_ATTEMPTS=50        # max trials before giving up (default N_RUNS*5)
#   SEED_BASE=12345
#   BATCH_OUT=output/debug_batch_one_sample_ur10e_shadow
#   BATCH_VIEWER=True
#   BATCH_POST_LIFT_STEPS=2000
#   BATCH_REALTIME_PLOT=False

N_RUNS="${N_RUNS:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-$((N_RUNS * 5))}"
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

echo "[INFO] TARGET_SUCCESS=${N_RUNS}"
echo "[INFO] MAX_ATTEMPTS=${MAX_ATTEMPTS}"
echo "[INFO] SEED_BASE=${SEED_BASE}"
echo "[INFO] BATCH_OUT=${BATCH_OUT}"
echo "[INFO] BATCH_VIEWER=${BATCH_VIEWER}"
echo "[INFO] BATCH_POST_LIFT_STEPS=${BATCH_POST_LIFT_STEPS}"
echo "[INFO] BATCH_REALTIME_PLOT=${BATCH_REALTIME_PLOT}"

success_count=0
attempt=0
while [ "$success_count" -lt "$N_RUNS" ] && [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
  attempt=$((attempt + 1))
  seed=$((SEED_BASE + attempt))
  try_log="${LOG_DIR}/attempt_$(printf '%03d' "$attempt").log"
  echo "[INFO] ===== attempt=${attempt} seed=${seed} (success ${success_count}/${N_RUNS}) ====="

  if SEED="$seed" DEBUG_VIEWER=False SAMPLE_DEBUG_VIEWER=False bash script/one_sample_BODex_ur10e_shadow_gravity.sh > "${try_log}" 2>&1; then
    run_ok=true
  else
    run_ok=false
  fi

  if [ "$run_ok" = "true" ] && [ -d "${SINGLE_OUT}/succgrasp" ]; then
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
      n_worker=1 >> "${try_log}" 2>&1 || true
  fi

  next_idx=$((success_count + 1))
  tag=$(printf "run_%03d" "$next_idx")
  run_case_dir="${RUN_DIR}/${tag}"

  if python3 - "$SINGLE_OUT" "$run_case_dir" "$META_FILE" "$tag" "$seed" "$run_ok" "$attempt" <<'PY'
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
attempt = int(sys.argv[7])

repo_root = os.path.abspath(os.getcwd())
sys.path.insert(0, os.path.join(repo_root, "src"))
from ekf_inhand.pose_metrics import summarize_ekf_pose_errors

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
summary = {
    "init_pos_err": None,
    "init_rot_err_deg": None,
    "hold_pos_mean": None,
    "hold_rot_mean_deg": None,
    "pos_final": None,
    "rot_final_deg": None,
}
if not has_metrics:
    print(
        f"[WARN] attempt={attempt} skipped: object={obj_name}, "
        f"run_ok={run_ok}, has_success={has_success}, has_metrics=False"
    )
    sys.exit(1)

os.makedirs(run_case_dir, exist_ok=True)
shutil.copy2(metrics_npz, os.path.join(run_case_dir, "metrics.npz"))
if os.path.exists(metrics_png):
    shutil.copy2(metrics_png, os.path.join(run_case_dir, "metrics.png"))
try:
    d = np.load(metrics_npz)
    steps = np.asarray(d["step"], dtype=float)
    pos = np.asarray(d["pos_err"], dtype=float)
    rot = np.asarray(d["rot_err_deg"], dtype=float)
    init_pos = float(d["init_pos_err"]) if "init_pos_err" in d else None
    init_rot = float(d["init_rot_err_deg"]) if "init_rot_err_deg" in d else None
    summary = summarize_ekf_pose_errors(
        steps=steps,
        pos_err=pos,
        rot_err_deg=rot,
        init_pos_err=init_pos,
        init_rot_err_deg=init_rot,
    )
except Exception:
    print(f"[WARN] attempt={attempt} skipped: metrics parse failed")
    sys.exit(1)

row = {
    "tag": tag,
    "seed": seed,
    "attempt": attempt,
    "run_ok": run_ok,
    "object_name": obj_name,
    "has_success": has_success,
    "has_metrics": True,
    **summary,
}
with open(meta_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(
    f"[INFO] {tag}: attempt={attempt}, object={obj_name}, "
    f"hold_rot={summary.get('hold_rot_mean_deg')}, rot_final={summary.get('rot_final_deg')}"
)
PY
  then
    success_count=$next_idx
    cp "${try_log}" "${LOG_DIR}/${tag}.log"
  else
    echo "[WARN] attempt=${attempt} failed, retry..."
  fi
done

if [ "$success_count" -lt "$N_RUNS" ]; then
  echo "[ERROR] only ${success_count}/${N_RUNS} successful runs after ${attempt} attempts"
  exit 1
fi

echo "[INFO] collected ${success_count} successful runs in ${attempt} attempts"

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
    "tag", "seed", "attempt", "run_ok", "object_name",
    "has_success", "has_metrics",
    "init_pos_err", "init_rot_err_deg",
    "hold_pos_mean", "hold_rot_mean_deg",
    "pos_final", "rot_final_deg",
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fields})

with_metrics = [r for r in rows if r.get("has_metrics")]

def _agg(key, fn):
    vals = [float(r[key]) for r in with_metrics if r.get(key) is not None]
    return float(fn(vals)) if vals else None

objs = sorted({r.get("object_name") for r in rows if r.get("object_name")})
fail_thr = 30.0
failed = [
    r for r in with_metrics
    if r.get("rot_final_deg") is not None and float(r["rot_final_deg"]) > fail_thr
]
overall = {
    "target_success_runs": len(with_metrics),
    "num_attempts": max((int(r.get("attempt", 0)) for r in rows), default=0),
    "num_runs": len(with_metrics),
    "num_failed_rot_final_gt_30deg": len(failed),
    "num_distinct_objects": len(objs),
    "distinct_objects": objs,
    "init_pos_err_avg": _agg("init_pos_err", np.mean),
    "init_rot_err_deg_avg": _agg("init_rot_err_deg", np.mean),
    "hold_pos_mean_avg": _agg("hold_pos_mean", np.mean),
    "hold_rot_mean_deg_avg": _agg("hold_rot_mean_deg", np.mean),
    "hold_rot_mean_deg_median": _agg("hold_rot_mean_deg", np.median),
    "hold_rot_mean_deg_p90": _agg("hold_rot_mean_deg", lambda v: np.percentile(v, 90)),
    "pos_final_avg": _agg("pos_final", np.mean),
    "rot_final_deg_avg": _agg("rot_final_deg", np.mean),
    "rot_final_deg_median": _agg("rot_final_deg", np.median),
}
overall_path = os.path.join(out_dir, "overall_summary.json")
with open(overall_path, "w", encoding="utf-8") as f:
    json.dump(overall, f, indent=2, ensure_ascii=False)

print(f"[INFO] summary.csv: {csv_path}")
print(f"[INFO] overall_summary.json: {overall_path}")
print(f"[INFO] overall: {overall}")
PY

echo "[INFO] Done. Batch results: ${BATCH_OUT}"
