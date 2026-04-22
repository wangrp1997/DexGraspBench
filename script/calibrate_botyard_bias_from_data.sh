#!/usr/bin/env bash
set -euo pipefail

# Auto-calibrate BOTYARD_FIXED_BIAS_VEC from existing BODex data.
# It evaluates multiple candidate biases and picks the best by:
#   1) higher success_rate
#   2) lower mean_delta_pos
#   3) lower mean_delta_angle
#
# Usage examples:
#   bash script/calibrate_botyard_bias_from_data.sh
#   BODEX_RAW_DIR=../BODex/src/curobo/content/assets/output/sim_botyard/fc/debug/grasp_data \
#   MAX_NUM=3 \
#   BIAS_CANDIDATES="-0.011315,0,0.014;-0.011315,0,0.016;-0.011315,0,0.01845" \
#   bash script/calibrate_botyard_bias_from_data.sh

BODEX_RAW_DIR="${BODEX_RAW_DIR:-../BODex/src/curobo/content/assets/output/sim_botyard/fc/debug/grasp_data}"
SETTING="${SETTING:-fc}"
SEED="${SEED:-3025}"
MAX_NUM="${MAX_NUM:-1}"
FORMAT_MAX_NUM="${FORMAT_MAX_NUM:-$MAX_NUM}"
EVAL_MAX_NUM="${EVAL_MAX_NUM:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CALIB_MAX_PENE="${CALIB_MAX_PENE:-0.05}"

# Semicolon-separated list of bias vectors.
BIAS_CANDIDATES="${BIAS_CANDIDATES:--0.011315,0,0.014;-0.011315,0,0.016;-0.011315,0,0.01845;-0.011315,0,0.020;-0.011315,0,0.022}"

OUT_ROOT="output/bias_calib_botyard"
RESULT_CSV="${OUT_ROOT}/results.csv"
BEST_TXT="${OUT_ROOT}/best_bias.txt"

mkdir -p "${OUT_ROOT}"
echo "bias,succeeded,evaluated,success_rate,mean_delta_pos,mean_delta_angle,exp_name" > "${RESULT_CSV}"

IFS=';' read -r -a BIAS_ARR <<< "${BIAS_CANDIDATES}"

idx=0
for bias in "${BIAS_ARR[@]}"; do
  exp_name="biascalib_${idx}"
  exp_out="output/${exp_name}_botyard"
  rm -rf "${exp_out}"

  echo "=== [${idx}] testing BOTYARD_FIXED_BIAS_VEC=${bias} ==="

  eval_stdout="${OUT_ROOT}/${exp_name}_eval_stdout.log"
  BOTYARD_FIXED_BIAS_VEC="${bias}" "${PYTHON_BIN}" src/main.py \
    seed="${SEED}" \
    setting="${SETTING}" \
    hand=botyard \
    task=format \
    exp_name="${exp_name}" \
    task.data_name=BODex \
    task.max_num="${FORMAT_MAX_NUM}" \
    task.data_path="${BODEX_RAW_DIR}"

  BOTYARD_FIXED_BIAS_VEC="${bias}" "${PYTHON_BIN}" src/main.py \
    seed="${SEED}" \
    setting="${SETTING}" \
    hand=botyard \
    task=eval \
    exp_name="${exp_name}" \
    task.max_num="${EVAL_MAX_NUM}" \
    task.simulation_metrics.max_pene="${CALIB_MAX_PENE}" \
    n_worker=1 \
    task.debug_viewer=False \
    task.debug_render=True | tee "${eval_stdout}"

  latest_eval_log="$(python - "${exp_name}" <<'PY'
import glob
import sys
exp_name = sys.argv[1]
paths = sorted(glob.glob(f"output/{exp_name}_botyard/log/eval/*/main.log"))
print(paths[-1] if paths else "")
PY
)"

  if [[ -z "${latest_eval_log}" ]]; then
    echo "[WARN] cannot find eval log for ${exp_name}, skip."
    idx=$((idx + 1))
    continue
  fi

  parsed="$(python - "${latest_eval_log}" "${eval_stdout}" <<'PY'
import re
import sys
log_path = sys.argv[1]
stdout_path = sys.argv[2]
text = open(log_path, "r", encoding="utf-8").read()
m = re.search(r"Get\s+\d+\s+grasp data,\s+(\d+)\s+evaluated,\s+and\s+(\d+)\s+succeeded", text)
if not m:
    print("0,0,0.0,1000000000.0,1000000000.0")
    sys.exit(0)
evaluated = int(m.group(1))
succeeded = int(m.group(2))
success_rate = 0.0 if evaluated == 0 else (succeeded / evaluated)

delta_pos = []
delta_ang = []
stdout_text = open(stdout_path, "r", encoding="utf-8").read() if stdout_path else ""
for line in stdout_text.splitlines():
    parts = line.strip().split()
    if len(parts) >= 3 and parts[0] in {"True", "False"}:
        try:
            delta_pos.append(float(parts[1]))
            delta_ang.append(float(parts[2]))
        except ValueError:
            pass

if len(delta_pos) == 0:
    mean_delta_pos = 1e9
    mean_delta_ang = 1e9
else:
    mean_delta_pos = sum(delta_pos) / len(delta_pos)
    mean_delta_ang = sum(delta_ang) / len(delta_ang)

print(
    f"{succeeded},{evaluated},{success_rate:.6f},"
    f"{mean_delta_pos:.6f},{mean_delta_ang:.6f}"
)
PY
)"

  IFS=',' read -r succeeded evaluated success_rate mean_delta_pos mean_delta_angle <<< "${parsed}"
  # Quote bias because it contains commas itself.
  echo "\"${bias}\",${succeeded},${evaluated},${success_rate},${mean_delta_pos},${mean_delta_angle},${exp_name}" >> "${RESULT_CSV}"
  idx=$((idx + 1))
done

python - "${RESULT_CSV}" "${BEST_TXT}" <<'PY'
import csv
import sys
result_csv, best_txt = sys.argv[1], sys.argv[2]
rows = []
with open(result_csv, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            rows.append(
                (
                    r["bias"],
                    int(r["succeeded"]),
                    int(r["evaluated"]),
                    float(r["success_rate"]),
                    float(r["mean_delta_pos"]),
                    float(r["mean_delta_angle"]),
                    r["exp_name"],
                )
            )
        except Exception:
            pass
if not rows:
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write("No valid results.\n")
    print("No valid results.")
    sys.exit(0)
rows.sort(key=lambda x: (-x[3], x[4], x[5], -x[1]))
best = rows[0]
with open(best_txt, "w", encoding="utf-8") as f:
    f.write(
        "best_bias={}\n".format(best[0]) +
        "succeeded={}\n".format(best[1]) +
        "evaluated={}\n".format(best[2]) +
        "success_rate={:.6f}\n".format(best[3]) +
        "mean_delta_pos={:.6f}\n".format(best[4]) +
        "mean_delta_angle={:.6f}\n".format(best[5]) +
        "exp_name={}\n".format(best[6])
    )
print(
    "Best bias:", best[0],
    "succeeded:", best[1],
    "evaluated:", best[2],
    "success_rate:", f"{best[3]:.6f}",
    "mean_delta_pos:", f"{best[4]:.6f}",
    "mean_delta_angle:", f"{best[5]:.6f}",
)
PY

echo "Calibration summary: ${RESULT_CSV}"
echo "Best bias file: ${BEST_TXT}"
