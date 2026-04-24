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
        }

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
        pos_err = float(np.linalg.norm(x_est6[:3] - x_gt7[:3]))
        # MuJoCo free joint qpos convention: [x, y, z, qw, qx, qy, qz]
        q_gt = np.array([x_gt7[3], x_gt7[4], x_gt7[5], x_gt7[6]], dtype=float)
        rot_err = _rot_err_deg(x_est6[3:6], q_gt)
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
        pos_arr = np.asarray(self._logs["pos_err"], dtype=float)
        rot_arr = np.asarray(self._logs["rot_err_deg"], dtype=float)
        print(
            f"{self.COLOR_TAG}[EKF-METRIC-SUM]{self.COLOR_RESET} "
            f"{self.COLOR_KEY}N={self.COLOR_RESET}{self.COLOR_VAL}{pos_arr.size}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}pos_mean={self.COLOR_RESET}{self.COLOR_VAL}{np.mean(pos_arr):.6f}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}pos_p90={self.COLOR_RESET}{self.COLOR_VAL}{np.percentile(pos_arr, 90):.6f}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}pos_max={self.COLOR_RESET}{self.COLOR_VAL}{np.max(pos_arr):.6f}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}rot_mean={self.COLOR_RESET}{self.COLOR_VAL}{np.mean(rot_arr):.3f}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}rot_p90={self.COLOR_RESET}{self.COLOR_VAL}{np.percentile(rot_arr, 90):.3f}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}rot_max={self.COLOR_RESET}{self.COLOR_VAL}{np.max(rot_arr):.3f}{self.COLOR_RESET}"
        )
        save_path = self.viz.save_outputs(self._logs, tag="ekf_pose_metrics")
        print(
            f"{self.COLOR_TAG}[EKF-METRIC-SAVE]{self.COLOR_RESET} "
            f"{self.COLOR_KEY}path={self.COLOR_RESET}{self.COLOR_VAL}{save_path}{self.COLOR_RESET}"
        )
