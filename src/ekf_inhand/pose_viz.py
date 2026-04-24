import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


class EkfPoseViz:
    """Realtime/offline visualization helper for EKF pose metrics."""

    def __init__(self, output_dir: str, realtime_enable: bool = False):
        self.output_dir = output_dir
        self.realtime_enable = bool(realtime_enable and plt is not None)
        self._fig = None
        self._axes = None
        self._lines = None

    def update_realtime(self, steps, pos_err, rot_err, u_norm):
        if not self.realtime_enable:
            return
        if self._fig is None:
            self._fig, self._axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
            self._lines = [
                self._axes[0].plot([], [], color="tab:blue", label="pos_err(m)")[0],
                self._axes[1].plot([], [], color="tab:orange", label="rot_err(deg)")[0],
                self._axes[2].plot([], [], color="tab:green", label="|u|")[0],
            ]
            self._axes[0].set_ylabel("m")
            self._axes[1].set_ylabel("deg")
            self._axes[2].set_ylabel("norm")
            self._axes[2].set_xlabel("step")
            for ax in self._axes:
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right")
            plt.tight_layout()
            plt.ion()
            plt.show(block=False)
        x = np.asarray(steps, dtype=float)
        series = [np.asarray(pos_err), np.asarray(rot_err), np.asarray(u_norm)]
        for ax, line, y in zip(self._axes, self._lines, series):
            line.set_data(x, y)
            if x.size > 0:
                ax.set_xlim(float(x.min()), float(x.max()) + 1.0)
            if y.size > 0:
                y_min, y_max = float(np.min(y)), float(np.max(y))
                if abs(y_max - y_min) < 1e-12:
                    pad = 1.0 if y_max == 0.0 else abs(y_max) * 0.1
                else:
                    pad = 0.1 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def save_outputs(self, log_dict: dict, tag: str = "ekf_pose") -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        run_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(run_dir, exist_ok=True)
        npz_path = os.path.join(run_dir, "metrics.npz")
        np.savez_compressed(npz_path, **log_dict)
        plot_metrics(log_dict, os.path.join(run_dir, "metrics.png"))
        return npz_path


def plot_metrics(log_dict: dict, save_path: str | None = None):
    if plt is None:
        return
    steps = np.asarray(log_dict.get("step", []), dtype=float)
    pos_err = np.asarray(log_dict.get("pos_err", []), dtype=float)
    rot_err = np.asarray(log_dict.get("rot_err_deg", []), dtype=float)
    u_norm = np.asarray(log_dict.get("u_norm", []), dtype=float)
    innov = np.asarray(log_dict.get("innov_norm", []), dtype=float)
    n_contacts = np.asarray(log_dict.get("n_contacts", []), dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(steps, pos_err, color="tab:blue")
    axes[0].set_ylabel("pos err (m)")
    axes[1].plot(steps, rot_err, color="tab:orange")
    axes[1].set_ylabel("rot err (deg)")
    axes[2].plot(steps, u_norm, color="tab:green", label="|u|")
    axes[2].plot(steps, innov, color="tab:red", alpha=0.7, label="|innov|")
    axes[2].set_ylabel("norm")
    axes[2].legend(loc="upper right")
    axes[3].plot(steps, n_contacts, color="tab:purple")
    axes[3].set_ylabel("n_contacts")
    axes[3].set_xlabel("step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=160)
    plt.close(fig)
