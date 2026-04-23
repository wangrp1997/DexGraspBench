import numpy as np


class EkfOnlineInputLogger:
    """Online logger for EKF inputs from MuJoCo simulation steps."""

    COLOR_RESET = "\033[0m"
    COLOR_TAG = "\033[96m"   # cyan
    COLOR_KEY = "\033[93m"   # yellow
    COLOR_VAL = "\033[92m"   # green

    def __init__(self, print_every=20):
        self.print_every = max(1, int(print_every))
        self.step_idx = 0
        self._finger_qpos_idx = None
        self._finger_dof_idx = None

    def _build_finger_indices(self, mj_ho):
        # Keep only Shadow finger joints, exclude wrist joints WRJ1/WRJ2.
        # This follows the strict "m_finger" observation choice.
        finger_qpos_idx = []
        finger_dof_idx = []
        for j in range(mj_ho.model.njnt):
            jnt = mj_ho.model.joint(j)
            name = jnt.name
            if not isinstance(name, str):
                continue
            # In this project joints can be prefixed after attach (e.g. "child-hand:rh_FFJ4").
            if "hand:rh_" not in name:
                continue
            if name.endswith("hand:rh_WRJ1") or name.endswith("hand:rh_WRJ2"):
                continue
            # In this hand model joints are 1-DOF, so one qpos/dof index each.
            finger_qpos_idx.append(int(jnt.qposadr[0]))
            finger_dof_idx.append(int(jnt.dofadr[0]))

        self._finger_qpos_idx = np.array(sorted(finger_qpos_idx), dtype=np.int32)
        self._finger_dof_idx = np.array(sorted(finger_dof_idx), dtype=np.int32)

    def _fmt_vec(self, arr, head=3):
        arr = np.asarray(arr).reshape(-1)
        preview = ", ".join([f"{v:.4f}" for v in arr[:head]])
        return f"[{preview}{', ...' if arr.size > head else ''}]"

    def on_step(self, mj_ho):
        """Called at each mj_step."""
        self.step_idx += 1
        if self._finger_qpos_idx is None or self._finger_dof_idx is None:
            self._build_finger_indices(mj_ho)

        if self.step_idx % self.print_every != 0:
            return

        q, qvel, tau_meas, x_gt = self.get_finger_stream(mj_ho)

        tag = f"{self.COLOR_TAG}[EKF-IN]{self.COLOR_RESET}"
        line1 = (
            f"{tag} "
            f"{self.COLOR_KEY}step={self.COLOR_RESET}{self.COLOR_VAL}{self.step_idx}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}m={self.COLOR_RESET}{self.COLOR_VAL}{q.shape[0]}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}q.shape={self.COLOR_RESET}{self.COLOR_VAL}{q.shape}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}qvel.shape={self.COLOR_RESET}{self.COLOR_VAL}{qvel.shape}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}tau.shape={self.COLOR_RESET}{self.COLOR_VAL}{tau_meas.shape}{self.COLOR_RESET}"
        )
        line2 = (
            f"{tag} "
            f"{self.COLOR_KEY}q={self.COLOR_RESET}{self.COLOR_VAL}{self._fmt_vec(q)}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}qvel={self.COLOR_RESET}{self.COLOR_VAL}{self._fmt_vec(qvel)}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}tau={self.COLOR_RESET}{self.COLOR_VAL}{self._fmt_vec(tau_meas)}{self.COLOR_RESET} "
            f"{self.COLOR_KEY}x_gt={self.COLOR_RESET}{self.COLOR_VAL}{self._fmt_vec(x_gt)}{self.COLOR_RESET}"
        )
        print(line1)
        print(line2)

    def get_finger_stream(self, mj_ho):
        """Return finger-only (q, qvel, tau_meas, x_gt) from current MuJoCo step."""
        # Online EKF input channels:
        # u_t  : qvel (control)
        # z_t  : [q, tau_meas], tau_meas = qfrc_constraint
        # x_gt : object pose (only for evaluation, not fed to EKF update)
        q_all = mj_ho.data.qpos[:-7]
        qvel_all = mj_ho.data.qvel[:-6]
        tau_all = mj_ho.data.qfrc_constraint[:-6]
        q = q_all[self._finger_qpos_idx]
        qvel = qvel_all[self._finger_dof_idx]
        tau_meas = tau_all[self._finger_dof_idx]
        x_gt = mj_ho.get_obj_pose()
        return q, qvel, tau_meas, x_gt

