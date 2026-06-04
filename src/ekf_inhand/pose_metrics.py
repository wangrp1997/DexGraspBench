import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from .pose_viz import EkfPoseViz


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quaternion must have dim 4, got {q.shape}")
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_wxyz_to_rotvec(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=float).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"q_wxyz must have dim 4, got {q.shape}")
    q = q / max(1e-12, float(np.linalg.norm(q)))
    return R.from_quat(_wxyz_to_xyzw(q)).as_rotvec()


def _rot_err_deg(rotvec_est: np.ndarray, q_gt_wxyz: np.ndarray) -> float:
    r_est = R.from_rotvec(np.asarray(rotvec_est, dtype=float).reshape(3))
    q_gt = np.asarray(q_gt_wxyz, dtype=float).reshape(-1)
    if q_gt.shape[0] != 4:
        return float("nan")
    q_gt = q_gt / max(1e-12, float(np.linalg.norm(q_gt)))
    r_gt = R.from_quat(_wxyz_to_xyzw(q_gt))
    return float(np.degrees((r_est * r_gt.inv()).magnitude()))


def pose_error_x6_vs_gt7(x_est6: np.ndarray, x_gt7: np.ndarray) -> tuple[float, float]:
    """Return (pos_err_m, rot_err_deg) for EKF x(6) vs MuJoCo object qpos(7)."""
    x_est6 = np.asarray(x_est6, dtype=float).reshape(-1)
    x_gt7 = np.asarray(x_gt7, dtype=float).reshape(-1)
    if x_est6.shape[0] < 6 or x_gt7.shape[0] < 7:
        return float("nan"), float("nan")
    pos_err = float(np.linalg.norm(x_est6[:3] - x_gt7[:3]))
    q_gt = np.array([x_gt7[3], x_gt7[4], x_gt7[5], x_gt7[6]], dtype=float)
    rot_err = _rot_err_deg(x_est6[3:6], q_gt)
    return pos_err, rot_err


def summarize_ekf_pose_errors(
    steps: np.ndarray,
    pos_err: np.ndarray,
    rot_err_deg: np.ndarray,
    init_pos_err: float | None = None,
    init_rot_err_deg: float | None = None,
    hold_burnin_fraction: float = 0.2,
) -> dict[str, float]:
    """
    Standard EKF pose error summary:
      - init_*: pre-update x0 error (scalars preferred; else step==0)
      - hold_*_mean: steady hold = last (1-burnin) of filter updates (default 20%–100%)
      - *_final: last filter update step (excludes step==0 marker)
    """
    steps = np.asarray(steps, dtype=float).reshape(-1)
    pos = np.asarray(pos_err, dtype=float).reshape(-1)
    rot = np.asarray(rot_err_deg, dtype=float).reshape(-1)
    if steps.size == 0 or pos.size != steps.size or rot.size != steps.size:
        nan = float("nan")
        return {
            "init_pos_err": nan,
            "init_rot_err_deg": nan,
            "hold_pos_mean": nan,
            "hold_rot_mean_deg": nan,
            "pos_final": nan,
            "rot_final_deg": nan,
        }

    init_pos = (
        float(init_pos_err)
        if init_pos_err is not None and np.isfinite(init_pos_err)
        else float("nan")
    )
    init_rot = (
        float(init_rot_err_deg)
        if init_rot_err_deg is not None and np.isfinite(init_rot_err_deg)
        else float("nan")
    )
    if not np.isfinite(init_pos) and steps.size > 0 and int(steps[0]) == 0:
        init_pos = float(pos[0])
        init_rot = float(rot[0])

    upd = steps > 0
    steps_u = steps[upd]
    pos_u = pos[upd]
    rot_u = rot[upd]
    if steps_u.size == 0:
        nan = float("nan")
        return {
            "init_pos_err": init_pos,
            "init_rot_err_deg": init_rot,
            "hold_pos_mean": nan,
            "hold_rot_mean_deg": nan,
            "pos_final": nan,
            "rot_final_deg": nan,
        }

    pos_final = float(pos_u[-1])
    rot_final = float(rot_u[-1])

    n = int(steps_u.size)
    burnin = float(np.clip(hold_burnin_fraction, 0.0, 0.95))
    start = int(burnin * n)
    start = min(start, max(0, n - 1))
    hold = np.zeros(n, dtype=bool)
    hold[start:] = True

    hold_pos_mean = float(np.mean(pos_u[hold]))
    hold_rot_mean = float(np.mean(rot_u[hold]))
    return {
        "init_pos_err": init_pos,
        "init_rot_err_deg": init_rot,
        "hold_pos_mean": hold_pos_mean,
        "hold_rot_mean_deg": hold_rot_mean,
        "pos_final": pos_final,
        "rot_final_deg": rot_final,
    }


class EkfPoseMetricsTracker:
    """Track EKF pose error against MuJoCo GT without bloating task code."""

    COLOR_TAG = "\033[96m"
    COLOR_KEY = "\033[93m"
    COLOR_VAL = "\033[92m"
    COLOR_RESET = "\033[0m"

    def __init__(
        self,
        print_every: int = 50,
        output_dir: str | None = None,
        realtime_plot: bool = False,
    ):
        self.print_every = max(1, int(print_every))
        self.output_dir = output_dir or os.path.abspath("output/debug_one_ur10e_shadow")
        self.viz = EkfPoseViz(
            output_dir=self.output_dir,
            realtime_enable=realtime_plot,
        )
        self._logs = {
            "step": [],
            "n_contacts": [],
            "pos_err": [],
            "rot_err_deg": [],
            "u_norm": [],
            "innov_norm": [],
            "init_pos_err": float("nan"),
            "init_rot_err_deg": float("nan"),
            "init_mode": "",
            "init_nominal_ref": "",
            "init_sim_step": -1,
        }
        self._init_recorded = False

    def record_init(
        self,
        sim_step: int,
        x0_est6: np.ndarray,
        x_gt7: np.ndarray,
        init_mode: str,
        nominal_ref: str = "",
    ) -> None:
        """Log true pre-update init error (x0 vs sim GT) and prepend step=0 to curves."""
        if self._init_recorded:
            return
        pos_err, rot_err = pose_error_x6_vs_gt7(x0_est6, x_gt7)
        self._logs["init_pos_err"] = pos_err
        self._logs["init_rot_err_deg"] = rot_err
        self._logs["init_mode"] = str(init_mode)
        self._logs["init_nominal_ref"] = str(nominal_ref)
        self._logs["init_sim_step"] = int(sim_step)
        self._init_recorded = True

        if len(self._logs["step"]) == 0:
            self._logs["step"].append(0)
            self._logs["n_contacts"].append(0)
            self._logs["pos_err"].append(pos_err)
            self._logs["rot_err_deg"].append(rot_err)
            self._logs["u_norm"].append(float("nan"))
            self._logs["innov_norm"].append(float("nan"))

        ref_txt = f" \033[94mref=\033[0m\033[92m{nominal_ref}\033[0m" if nominal_ref else ""
        print(
            f"\033[95m[EKF-INIT-ERR]\033[0m "
            f"\033[94msim_step=\033[0m\033[92m{sim_step}\033[0m "
            f"\033[94mmode=\033[0m\033[92m{init_mode}\033[0m"
            f"{ref_txt} "
            f"\033[94mpos_err(m)=\033[0m\033[92m{pos_err:.6f}\033[0m "
            f"\033[94mrot_err(deg)=\033[0m\033[92m{rot_err:.3f}\033[0m "
            f"\033[93m(pre-update x0 vs sim GT)\033[0m"
        )

    def update(
        self,
        step: int,
        x_est6: np.ndarray,
        x_gt7: np.ndarray,
        n_contacts: int,
        u_norm: float = float("nan"),
        innov_norm: float = float("nan"),
    ) -> None:
        x_est6 = np.asarray(x_est6, dtype=float).reshape(-1)
        x_gt7 = np.asarray(x_gt7, dtype=float).reshape(-1)
        if x_est6.shape[0] < 6 or x_gt7.shape[0] < 7:
            return
        pos_err, rot_err = pose_error_x6_vs_gt7(x_est6, x_gt7)
        self._logs["step"].append(int(step))
        self._logs["n_contacts"].append(int(n_contacts))
        self._logs["pos_err"].append(pos_err)
        self._logs["rot_err_deg"].append(rot_err)
        self._logs["u_norm"].append(float(u_norm))
        self._logs["innov_norm"].append(float(innov_norm))
        self.viz.update_realtime(
            steps=self._logs["step"],
            pos_err=self._logs["pos_err"],
            rot_err=self._logs["rot_err_deg"],
            u_norm=self._logs["u_norm"],
        )
        if step % self.print_every == 0:
            print(
                f"{self.COLOR_TAG}[EKF-METRIC]{self.COLOR_RESET} "
                f"{self.COLOR_KEY}step={self.COLOR_RESET}{self.COLOR_VAL}{step}{self.COLOR_RESET} "
                f"{self.COLOR_KEY}n_contacts={self.COLOR_RESET}{self.COLOR_VAL}{n_contacts}{self.COLOR_RESET} "
                f"{self.COLOR_KEY}pos_err(m)={self.COLOR_RESET}{self.COLOR_VAL}{pos_err:.6f}{self.COLOR_RESET} "
                f"{self.COLOR_KEY}rot_err(deg)={self.COLOR_RESET}{self.COLOR_VAL}{rot_err:.3f}{self.COLOR_RESET}"
            )

    def emit_summary(self) -> None:
        if len(self._logs["pos_err"]) == 0:
            return
        steps = np.asarray(self._logs["step"], dtype=float)
        pos_arr = np.asarray(self._logs["pos_err"], dtype=float)
        rot_arr = np.asarray(self._logs["rot_err_deg"], dtype=float)
        summary = summarize_ekf_pose_errors(
            steps=steps,
            pos_err=pos_arr,
            rot_err_deg=rot_arr,
            init_pos_err=float(self._logs.get("init_pos_err", float("nan"))),
            init_rot_err_deg=float(self._logs.get("init_rot_err_deg", float("nan"))),
        )
        for key, val in summary.items():
            self._logs[key] = float(val)

        init_pos = summary["init_pos_err"]
        init_rot = summary["init_rot_err_deg"]
        if np.isfinite(init_pos):
            print(
                f"{self.COLOR_TAG}[EKF-INIT-ERR-SUM]{self.COLOR_RESET} "
                f"{self.COLOR_KEY}init_pos(m)={self.COLOR_RESET}{self.COLOR_VAL}{init_pos:.6f}{self.COLOR_RESET} "
                f"{self.COLOR_KEY}init_rot(deg)={self.COLOR_RESET}{self.COLOR_VAL}{init_rot:.3f}{self.COLOR_RESET} "
                f"{self.COLOR_KEY}ref={self.COLOR_RESET}{self.COLOR_VAL}{self._logs.get('init_nominal_ref', '')}{self.COLOR_RESET}"
            )
        print(
            f"{self.COLOR_TAG}[EKF-METRIC-SUM]{self.COLOR_RESET} "
            f"{self.COLOR_KEY}hold_pos_mean(m)={self.COLOR_RESET}{self.COLOR_VAL}{summary['hold_pos_mean']:.6f}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}hold_rot_mean(deg)={self.COLOR_RESET}{self.COLOR_VAL}{summary['hold_rot_mean_deg']:.3f}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}pos_final(m)={self.COLOR_RESET}{self.COLOR_VAL}{summary['pos_final']:.6f}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}rot_final(deg)={self.COLOR_RESET}{self.COLOR_VAL}{summary['rot_final_deg']:.3f}{self.COLOR_RESET}"
        )
        save_path = self.viz.save_outputs(self._logs, tag="ekf_pose_metrics")
        print(
            f"{self.COLOR_TAG}[EKF-METRIC-SAVE]{self.COLOR_RESET} "
            f"{self.COLOR_KEY}path={self.COLOR_RESET}{self.COLOR_VAL}{save_path}{self.COLOR_RESET}"
        )
