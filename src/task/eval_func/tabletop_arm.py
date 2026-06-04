import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from .base import BaseEval
from ekf_inhand.contact_manager import build_hq_inputs_with_contact_matching
from ekf_inhand.contact_manager import audit_state_cov_contact_alignment
from ekf_inhand.contact_manager import expand_state_cov_for_new_contacts
from ekf_inhand.contact_manager import shrink_state_cov_for_lost_contacts
from ekf_inhand.input_stream import EkfOnlineInputLogger
from ekf_inhand.init import EkfInitConfig
from ekf_inhand.init import build_initial_covariance
from ekf_inhand.init import build_initial_state
from ekf_inhand.mj_geometry_adapter import MjEkfGeometryAdapter
from ekf_inhand.pose_metrics import EkfPoseMetricsTracker
from ekf_inhand.pose_metrics import quat_wxyz_to_rotvec
from ekf_inhand.observation_model import h_q_eq8_11
from ekf_inhand.runner import run_one_step_smoke
from ekf_inhand.state import InhandState


class tabletopArmEval(BaseEval):
    @staticmethod
    def _physical_pre_obj_qpos(pre_obj_qpos: np.ndarray, setting: str) -> np.ndarray:
        """Strip tabletop success-metric z offset (+0.1 m) from pre_obj_qpos."""
        q = np.asarray(pre_obj_qpos, dtype=float).reshape(-1).copy()
        if setting == "tabletop":
            q[2] -= 0.1
        return q

    def _resolve_nominal_obj_qpos_ref(self, ekf_ctx: dict) -> tuple[np.ndarray, str]:
        """
        Resolve blind nominal object pose for EKF init (7D MuJoCo qpos, no sim peek).

        ekf_init_nominal_ref:
          - grasp: grasp_data["obj_pose"] from dataset/planner
          - pre_grasp: pre-squeeze sim pose (tabletop +0.1 m eval offset stripped)
          - lift_end: sim pose at end of lift (semi-oracle, debugging only)
          - squeeze_end: sim pose at end of squeeze (semi-oracle)
        """
        ref_key = str(
            getattr(self.configs.task, "ekf_init_nominal_ref", "grasp")
        ).lower()
        if ref_key == "grasp":
            if not hasattr(self, "grasp_data") or "obj_pose" not in self.grasp_data:
                raise ValueError(
                    "ekf_init_nominal_ref=grasp requires grasp_data['obj_pose']"
                )
            ref = np.asarray(self.grasp_data["obj_pose"], dtype=float).reshape(-1).copy()
            if ref.shape[0] < 7:
                raise ValueError(
                    f"grasp_data['obj_pose'] must have at least 7 dims, got {ref.shape}"
                )
            return ref, "grasp"
        if ref_key == "pre_grasp":
            ref = self._physical_pre_obj_qpos(
                ekf_ctx["pre_obj_qpos"], self.configs.setting
            )
            return ref, "pre_grasp"
        if ref_key == "lift_end":
            lift_ref = ekf_ctx.get("lift_end_obj_qpos")
            if lift_ref is None:
                raise ValueError(
                    "ekf_init_nominal_ref=lift_end but lift_end_obj_qpos is unavailable"
                )
            return np.asarray(lift_ref, dtype=float).reshape(-1).copy(), "lift_end"
        if ref_key == "squeeze_end":
            squeeze_ref = ekf_ctx.get("squeeze_end_obj_qpos")
            if squeeze_ref is None:
                raise ValueError(
                    "ekf_init_nominal_ref=squeeze_end but squeeze_end_obj_qpos is unavailable"
                )
            return np.asarray(squeeze_ref, dtype=float).reshape(-1).copy(), "squeeze_end"
        raise ValueError(
            f"Unsupported ekf_init_nominal_ref={ref_key!r}, "
            "expected grasp | pre_grasp | lift_end | squeeze_end"
        )

    @staticmethod
    def _x6_to_pose7(x_est6: np.ndarray) -> np.ndarray:
        x_est6 = np.asarray(x_est6, dtype=float).reshape(-1)
        if x_est6.shape[0] != 6:
            raise ValueError(f"x_est6 must have shape (6,), got {x_est6.shape}")
        quat_xyzw = R.from_rotvec(x_est6[3:6]).as_quat()
        # MuJoCo free joint uses [x, y, z, qw, qx, qy, qz].
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
        )
        return np.concatenate([x_est6[:3], quat_wxyz], axis=0)

    @staticmethod
    def _build_R_block_diag(m: int, r_q: float, r_tau: float) -> np.ndarray:
        R = np.zeros((2 * m, 2 * m), dtype=float)
        R[:m, :m] = np.eye(m, dtype=float) * float(r_q)
        R[m:, m:] = np.eye(m, dtype=float) * float(r_tau)
        return R

    @staticmethod
    def _build_Q_block_diag(state: InhandState, q_x: float, q_xi: float, q_f: float) -> np.ndarray:
        dim = state.dim
        Q = np.zeros((dim, dim), dtype=float)
        Q[:6, :6] = np.eye(6, dtype=float) * float(q_x)
        n = state.n_contacts
        if n > 0:
            xi_start = 6
            xi_end = 6 + 2 * n
            f_start = xi_end
            f_end = f_start + n
            Q[xi_start:xi_end, xi_start:xi_end] = np.eye(2 * n, dtype=float) * float(q_xi)
            Q[f_start:f_end, f_start:f_end] = np.eye(n, dtype=float) * float(q_f)
        return Q

    def _get_ekf_geom_adapter(self, mj_ho) -> MjEkfGeometryAdapter:
        adapter = getattr(self, "_ekf_geom_adapter", None)
        if adapter is None or adapter.mj_ho is not mj_ho:
            adapter = MjEkfGeometryAdapter(mj_ho)
            self._ekf_geom_adapter = adapter
        return adapter

    def _map_contact_to_object_xi_with_hint(self, obj_body_id, c_world, hint_face_id=-1):
        return self._get_ekf_geom_adapter(self.mj_ho).map_contact_to_object_xi_with_hint(
            obj_body_id=obj_body_id,
            c_world=c_world,
            hint_face_id=hint_face_id,
        )

    def _map_contact_to_hand_surface(self, hand_body_id, c_world, n_obj, hint_face_id=-1):
        return self._get_ekf_geom_adapter(self.mj_ho).map_contact_to_hand_surface(
            hand_body_id=hand_body_id,
            c_world=c_world,
            n_obj=n_obj,
            hint_face_id=hint_face_id,
        )

    def _build_hand_normal_fn(self, hand_body_id, face_id):
        return self._get_ekf_geom_adapter(self.mj_ho).build_hand_normal_fn(
            hand_body_id=hand_body_id,
            face_id=face_id,
        )

    def _build_motion_grasp_matrices(
        self,
        mj_ho,
        c_obj: np.ndarray,
        J_obs_contact: np.ndarray,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._get_ekf_geom_adapter(mj_ho).build_motion_grasp_matrices(
            c_obj=c_obj,
            J_obs_contact=J_obs_contact,
            m=m,
        )

    def _simulate_under_extforce_details(self, pre_obj_qpos):
        ekf_logger = None
        ekf_stage = getattr(self.configs.task, "ekf_input_stage", "post_lift")
        run_ekf_smoke = getattr(self.configs.task, "ekf_smoke_enable", False)
        run_ekf_eval = bool(getattr(self.configs.task, "ekf_pose_eval_enable", True))
        log_ekf_inputs = bool(getattr(self.configs.task, "ekf_input_debug", False))
        run_ekf_callback = bool(run_ekf_smoke or run_ekf_eval)
        ekf_ctx = {
            "state": None,
            "P": None,
            "inited": False,
            "step": 0,
            "contact_cache": None,
            "contact_stats": None,
            "no_contact_streak": 0,
            "pose_metrics": None,
            "init_logged": False,
            "pre_obj_qpos": np.asarray(pre_obj_qpos, dtype=float).reshape(-1).copy(),
            "squeeze_end_obj_qpos": None,
            "lift_end_obj_qpos": None,
            "q_prev": None,
        }
        if getattr(self.configs.task, "ekf_pose_eval_enable", True):
            pose_output_dir = str(
                getattr(
                    self.configs.task,
                    "ekf_pose_eval_output_dir",
                    os.path.abspath("output/debug_one_ur10e_shadow"),
                )
            )
            ekf_ctx["pose_metrics"] = EkfPoseMetricsTracker(
                print_every=int(getattr(self.configs.task, "ekf_pose_eval_print_every", 50)),
                output_dir=pose_output_dir,
                realtime_plot=bool(getattr(self.configs.task, "ekf_pose_eval_realtime_plot", False)),
            )
        if run_ekf_callback or log_ekf_inputs:
            ekf_logger = EkfOnlineInputLogger(
                print_every=(
                    int(getattr(self.configs.task, "ekf_input_print_every", 20))
                    if log_ekf_inputs
                    else 10**9
                )
            )
            if ekf_stage == "all" and run_ekf_callback:
                self.mj_ho.step_callback = lambda mj: self._ekf_debug_callback(
                    mj, ekf_logger, ekf_ctx
                )
            elif ekf_stage == "all" and log_ekf_inputs:
                self.mj_ho.step_callback = ekf_logger.on_step
        try:
            # 1. Set object gravity (legacy mode uses equivalent external force when gravity is disabled).
            if getattr(self.configs.task, "use_external_gravity", True):
                external_force_direction = np.array([0.0, 0, -1, 0, 0, 0])
                self.mj_ho.set_ext_force_on_obj(
                    10 * external_force_direction * self.configs.task.obj_mass
                )

            # 2. Approaching
            approach_length = self.grasp_data["approach_qpos"].shape[0]
            for i in range(approach_length - 1):
                self.mj_ho.control_hand_with_interp(
                    self.grasp_data["approach_qpos"][i],
                    self.grasp_data["approach_qpos"][i + 1],
                    step_outer=3 if (i % 5 == 4 or i == approach_length - 2) else 1,
                )

            # 3. Move hand to pre-grasp pose
            self.mj_ho.control_hand_with_interp(
                self.grasp_data["approach_qpos"][-1],
                self.grasp_data["pregrasp_qpos"],
            )

            # 4. Move hand to grasp pose
            self.mj_ho.control_hand_with_interp(
                self.grasp_data["pregrasp_qpos"],
                self.grasp_data["grasp_qpos"],
            )

            # 5. Move hand to squeeze pose.
            # NOTE step 2 and 3 are seperate because pre -> grasp -> squeeze are stage-wise linear.
            # If step 2 and 3 are merged to one linear interpolation, the performance will drop a lot.
            self.mj_ho.control_hand_with_interp(
                self.grasp_data["grasp_qpos"],
                self.grasp_data["squeeze_qpos"],
            )
            ekf_ctx["squeeze_end_obj_qpos"] = np.asarray(
                self.mj_ho.get_obj_pose(), dtype=float
            ).reshape(-1).copy()

            # 6. Lift the object
            # NOTE: For strict "post-lift" evaluation, do NOT run EKF during lift.
            # EKF callback is enabled only after this lift interpolation finishes.
            self.mj_ho.control_hand_with_interp(
                self.grasp_data["squeeze_qpos"],
                self.grasp_data["lift_qpos"],
            )
            ekf_ctx["lift_end_obj_qpos"] = np.asarray(
                self.mj_ho.get_obj_pose(), dtype=float
            ).reshape(-1).copy()

            # 7. Hold-and-observe stage after lift.
            # -1 means "keep printing while viewer is open".
            if ekf_logger is not None and ekf_stage == "post_lift" and run_ekf_callback:
                self.mj_ho.step_callback = lambda mj: self._ekf_debug_callback(
                    mj, ekf_logger, ekf_ctx
                )
            elif ekf_logger is not None and ekf_stage == "post_lift" and log_ekf_inputs:
                self.mj_ho.step_callback = ekf_logger.on_step
            if ekf_logger is not None and ekf_stage == "post_lift":
                post_lift_steps = int(
                    getattr(self.configs.task, "ekf_input_post_lift_steps", 2000)
                )
                if post_lift_steps == -1 and self.mj_ho.debug_viewer is not None:
                    while self.mj_ho.debug_viewer.is_running():
                        self.mj_ho.control_hand_step(1)
                elif post_lift_steps > 0:
                    self.mj_ho.control_hand_step(post_lift_steps)
        finally:
            if ekf_logger is not None:
                self.mj_ho.step_callback = None
            pose_metrics = ekf_ctx.get("pose_metrics")
            if pose_metrics is not None:
                pose_metrics.emit_summary()

        return

    def _ekf_debug_callback(self, mj_ho, ekf_logger, ekf_ctx):
        # Keep existing EKF input print behavior.
        ekf_logger.on_step(mj_ho)
        ekf_ctx["step"] += 1
        ekf_step_every = int(getattr(self.configs.task, "ekf_step_every", 1))
        if ekf_step_every > 1 and (ekf_ctx["step"] % ekf_step_every != 0):
            return

        # Run a minimal smoke step each simulation tick.
        q, qvel, tau, x_gt = ekf_logger.get_finger_stream(mj_ho)
        z_t = np.concatenate([q, tau], axis=0)
        u_t = qvel

        if not ekf_ctx["inited"]:
            # x0 hint for object pose block.
            x0_hint = np.zeros((6,), dtype=float)
            init_mode = str(getattr(self.configs.task, "ekf_init_mode", "gt_perturbed"))
            if init_mode in ("gt_exact", "gt_perturbed"):
                x0_hint[:3] = np.asarray(x_gt[:3], dtype=float)
                x0_hint[3:6] = quat_wxyz_to_rotvec(np.asarray(x_gt[3:7], dtype=float))
                if init_mode == "gt_perturbed":
                    seed = int(getattr(self.configs.task, "ekf_init_seed", 0))
                    sigma_pos = float(
                        getattr(self.configs.task, "ekf_init_gt_pos_sigma", 0.01)
                    )
                    sigma_rot_deg = float(
                        getattr(self.configs.task, "ekf_init_gt_rot_sigma_deg", 5.0)
                    )
                    rng = np.random.default_rng(seed)
                    x0_hint[:3] = x0_hint[:3] + rng.normal(0.0, sigma_pos, size=3)
                    rot_noise = rng.normal(0.0, np.deg2rad(sigma_rot_deg), size=3)
                    x0_hint[3:6] = x0_hint[3:6] + rot_noise
            elif init_mode == "nominal":
                nominal_ref, nominal_ref_name = self._resolve_nominal_obj_qpos_ref(
                    ekf_ctx
                )
                if nominal_ref.shape[0] < 7:
                    raise ValueError(
                        f"nominal_obj_qpos_ref must have at least 7 dims, got {nominal_ref.shape}"
                    )
                x0_hint[:3] = nominal_ref[:3]
                x0_hint[3:6] = quat_wxyz_to_rotvec(nominal_ref[3:7])
                sigma_pos = float(
                    getattr(self.configs.task, "ekf_init_nominal_pos_sigma", 0.02)
                )
                sigma_rot_deg = float(
                    getattr(self.configs.task, "ekf_init_nominal_rot_sigma_deg", 10.0)
                )
                seed = int(getattr(self.configs.task, "ekf_init_seed", 0))
                rng = np.random.default_rng(seed)
                x0_hint[:3] = x0_hint[:3] + rng.normal(0.0, sigma_pos, size=3)
                rot_noise = rng.normal(0.0, np.deg2rad(sigma_rot_deg), size=3)
                x0_hint[3:6] = x0_hint[3:6] + rot_noise
            else:
                raise ValueError(
                    f"Unsupported ekf_init_mode={init_mode}, expected gt_exact / gt_perturbed / nominal"
                )
            if not ekf_ctx["init_logged"]:
                init_extra = ""
                if init_mode == "nominal":
                    init_extra = (
                        f" \033[94mref=\033[0m\033[92m{nominal_ref_name}\033[0m"
                    )
                print(
                    f"\033[95m[EKF-INIT]\033[0m "
                    f"\033[94mmode=\033[0m\033[92m{init_mode}\033[0m"
                    f"{init_extra} "
                    f"\033[94mx0_pos=\033[0m\033[92m{x0_hint[:3]}\033[0m "
                    f"\033[94mx0_rotvec=\033[0m\033[92m{x0_hint[3:6]}\033[0m"
                )
                ekf_ctx["init_logged"] = True
            init_cfg = EkfInitConfig(init_contacts=0, x0_std=0.0, seed=0)
            state0 = build_initial_state(x0_hint=x0_hint, cfg=init_cfg)
            P0 = build_initial_covariance(state0=state0, cfg=init_cfg)
            ekf_ctx["state"] = state0
            ekf_ctx["P"] = P0
            ekf_ctx["inited"] = True
            pose_metrics = ekf_ctx.get("pose_metrics")
            if pose_metrics is not None:
                nominal_ref_tag = ""
                if init_mode == "nominal":
                    nominal_ref_tag = str(nominal_ref_name)
                pose_metrics.record_init(
                    sim_step=int(ekf_ctx["step"]),
                    x0_est6=state0.x,
                    x_gt7=x_gt,
                    init_mode=init_mode,
                    nominal_ref=nominal_ref_tag,
                )

        state_prev: InhandState = ekf_ctx["state"]
        P_prev: np.ndarray = ekf_ctx["P"]

        m = q.shape[0]
        r_q = float(getattr(self.configs.task, "ekf_r_q", 1e-3))
        r_tau = float(getattr(self.configs.task, "ekf_r_tau", 1e-3))
        q_x = float(getattr(self.configs.task, "ekf_q_x", 1e-4))
        q_xi = float(getattr(self.configs.task, "ekf_q_xi", 1e-4))
        q_f = float(getattr(self.configs.task, "ekf_q_f", 1e-4))
        R_t = self._build_R_block_diag(m=m, r_q=r_q, r_tau=r_tau)
        (
            c_obj,
            c_f_prev,
            contact_normals_obj,
            J_obs_contact,
            J_pinv,
            xi_new_init,
            f_new_init,
            obj_face_ids,
            next_cache,
            match_stats,
            match_events,
        ) = build_hq_inputs_with_contact_matching(
            mj_ho=mj_ho,
            m=m,
            tau_meas=tau,
            finger_dof_idx=ekf_logger._finger_dof_idx,
            prev_contact_cache=ekf_ctx.get("contact_cache"),
            obj_xi_mapper=self._map_contact_to_object_xi_with_hint,
            hand_surface_mapper=self._map_contact_to_hand_surface,
            hand_normal_fn_builder=self._build_hand_normal_fn,
            rho_plus=float(getattr(self.configs.task, "ekf_contact_rho_plus", 0.05)),
            rho_minus=float(getattr(self.configs.task, "ekf_contact_rho_minus", 0.02)),
            contact_score_eps=float(getattr(self.configs.task, "ekf_contact_score_eps", 1e-9)),
        )
        G_pinv, J_motion = self._build_motion_grasp_matrices(
            mj_ho=mj_ho,
            c_obj=c_obj,
            J_obs_contact=J_obs_contact,
            m=m,
        )
        state_work = state_prev
        P_work = P_prev
        Q_work = self._build_Q_block_diag(state=state_work, q_x=q_x, q_xi=q_xi, q_f=q_f)
        Q_prev_local = Q_work.copy()
        if len(match_events.lost_prev_idx) > 0:
            state_work, P_work, Q_work = shrink_state_cov_for_lost_contacts(
                state_prev=state_work,
                P_prev=P_work,
                Q_prev=Q_work,
                lost_prev_idx=match_events.lost_prev_idx,
            )
        state_after_shrink = state_work
        P_after_shrink = P_work
        Q_after_shrink = Q_work
        if len(match_events.new_curr_idx) > 0:
            state_work, P_work, Q_work = expand_state_cov_for_new_contacts(
                state_prev=state_work,
                P_prev=P_work,
                Q_prev=Q_work,
                new_count=len(match_events.new_curr_idx),
                xi_init=xi_new_init,
                f_init=f_new_init,
                q_new_var=q_xi,
            )
        # Keep process noise consistent with configured block-diagonal policy.
        Q_work = self._build_Q_block_diag(state=state_work, q_x=q_x, q_xi=q_xi, q_f=q_f)
        state_after_expand = state_work
        P_after_expand = P_work
        Q_after_expand = Q_work

        audit_enable = bool(getattr(self.configs.task, "ekf_audit_enable", False))
        audit_every = int(getattr(self.configs.task, "ekf_audit_every", 20))
        if audit_enable and audit_every > 0 and (ekf_ctx["step"] % audit_every == 0):
            audit_state_cov_contact_alignment(
                state_prev=state_prev,
                P_prev=P_prev,
                Q_prev=Q_prev_local,
                state_after_shrink=state_after_shrink,
                P_after_shrink=P_after_shrink,
                Q_after_shrink=Q_after_shrink,
                state_after_expand=state_after_expand,
                P_after_expand=P_after_expand,
                Q_after_expand=Q_after_expand,
                match_events=match_events,
                xi_new_init=xi_new_init,
                f_new_init=f_new_init,
            )

        ekf_ctx["contact_cache"] = next_cache
        ekf_ctx["contact_stats"] = match_stats
        if state_work.n_contacts == 0:
            ekf_ctx["no_contact_streak"] = int(ekf_ctx.get("no_contact_streak", 0)) + 1
        else:
            ekf_ctx["no_contact_streak"] = 0
        no_contact_steps_thr = int(
            getattr(self.configs.task, "ekf_no_contact_steps_threshold", 8)
        )
        low_u_hold = False
        if ekf_ctx["no_contact_streak"] >= max(1, no_contact_steps_thr):
            tau_r_scale = float(
                getattr(self.configs.task, "ekf_no_contact_tau_R_scale", 100.0)
            )
            R_t[m:, m:] *= tau_r_scale
        else:
            u_norm = float(np.linalg.norm(u_t))
            low_u_thr = float(
                getattr(self.configs.task, "ekf_low_u_norm_threshold", 0.0)
            )
            low_u_hold = (
                low_u_thr > 0.0
                and u_norm < low_u_thr
                and state_work.n_contacts > 0
            )
            low_u_skip_update = bool(
                getattr(self.configs.task, "ekf_low_u_skip_update", False)
            )
            if low_u_hold and not low_u_skip_update:
                hold_r_scale = float(
                    getattr(
                        self.configs.task,
                        "ekf_low_u_R_scale",
                        getattr(self.configs.task, "ekf_low_u_tau_R_scale", 50.0),
                    )
                )
                R_t *= hold_r_scale
        J_obs = np.asarray(J_obs_contact, dtype=float)
        if J_obs.shape != (3 * state_work.n_contacts, m):
            raise RuntimeError(
                f"J_obs shape mismatch, expected {(3 * state_work.n_contacts, m)}, got {J_obs.shape}"
            )
        hq_impl = None
        use_state_hq_geom = bool(
            getattr(self.configs.task, "ekf_hq_use_state_geometry", True)
        )
        if use_state_hq_geom and state_work.n_contacts > 0:
            adapter = self._get_ekf_geom_adapter(mj_ho)
            face_ids_step = np.asarray(obj_face_ids, dtype=np.int32).reshape(-1)
            c_f_step = np.asarray(c_f_prev, dtype=float)
            J_pinv_step = np.asarray(J_pinv, dtype=float)

            def _hq_from_state(state_hq: InhandState, q_prev_hq: np.ndarray) -> np.ndarray:
                n_hq = state_hq.n_contacts
                if face_ids_step.shape[0] != n_hq:
                    raise RuntimeError(
                        f"obj_face_ids length {face_ids_step.shape[0]} != state contacts {n_hq}"
                    )
                xi_hq = state_hq.xi.reshape(n_hq, 2)
                c_obj_state = adapter.object_contact_points_from_state(
                    state_hq.x, xi_hq, face_ids_step
                )
                return h_q_eq8_11(
                    state=state_hq,
                    q_prev=q_prev_hq,
                    J_pinv=J_pinv_step,
                    c_obj=c_obj_state,
                    c_f_prev=c_f_step,
                )

            hq_impl = _hq_from_state
        q_for_hq = ekf_ctx["q_prev"]
        if q_for_hq is None:
            # First EKF step: no t-1 yet, fall back to current measurement.
            q_for_hq = q
        else:
            q_for_hq = np.asarray(q_for_hq, dtype=float).reshape(-1)
            if q_for_hq.shape[0] != q.shape[0]:
                q_for_hq = q

        step_res = run_one_step_smoke(
            state_prev=state_work,
            P_prev=P_work,
            u_t=u_t,
            z_t=z_t,
            dt=float(mj_ho.model.opt.timestep),
            G_pinv=G_pinv,
            J_motion=J_motion,
            Q_t=Q_work,
            R_t=R_t,
            q_prev=q_for_hq,
            J_obs=J_obs,
            contact_normals_obj=contact_normals_obj,
            hq_impl=hq_impl,
            J_pinv=J_pinv,
            c_obj=c_obj,
            c_f_prev=c_f_prev,
            jac_eps=1e-6,
            skip_update=bool(
                low_u_hold
                and getattr(self.configs.task, "ekf_low_u_skip_update", False)
            ),
            use_analytic_h_tau=bool(
                getattr(self.configs.task, "ekf_use_analytic_h_tau", True)
            ),
        )

        ekf_ctx["state"] = InhandState.unpack(step_res.ekf_update.y_next)
        ekf_ctx["P"] = step_res.ekf_update.P_next
        ekf_ctx["q_prev"] = np.asarray(q, dtype=float).reshape(-1).copy()
        est_pose7 = self._x6_to_pose7(ekf_ctx["state"].x)
        if bool(getattr(self.configs.task, "ekf_pose_mesh_overlay_enable", False)):
            mj_ho.set_est_obj_pose(est_pose7)
        pose_metrics = ekf_ctx.get("pose_metrics")
        if pose_metrics is not None:
            innov_norm = float(np.linalg.norm(step_res.ekf_update.innovation))
            pose_metrics.update(
                step=ekf_ctx["step"],
                x_est6=ekf_ctx["state"].x,
                x_gt7=x_gt,
                n_contacts=ekf_ctx["state"].n_contacts,
                u_norm=float(np.linalg.norm(u_t)),
                innov_norm=innov_norm,
            )
        g_sanity_enable = bool(getattr(self.configs.task, "ekf_g_sanity_enable", False))

        smoke_every = int(getattr(self.configs.task, "ekf_smoke_print_every", 50))
        if smoke_every > 0 and ekf_ctx["step"] % smoke_every == 0:
            # Use magenta-like color to differ from [EKF-IN] cyan tag.
            color_tag = "\033[95m"
            color_key = "\033[94m"
            color_val = "\033[92m"
            color_warn = "\033[91m"
            color_reset = "\033[0m"
            innov_norm = float(np.linalg.norm(step_res.ekf_update.innovation))
            u_norm = float(np.linalg.norm(u_t))
            contact_stats = ekf_ctx.get("contact_stats")
            contact_stat_txt = ""
            g_sanity_txt = ""
            warn_txt = ""
            if contact_stats is not None:
                changed = contact_stats.new_count + contact_stats.lost_count
                baseline = max(1, contact_stats.prev_count + contact_stats.curr_count)
                change_ratio = float(changed) / float(baseline)
                warn_threshold = float(
                    getattr(self.configs.task, "ekf_contact_warn_ratio", 0.6)
                )
                contact_stat_txt = (
                    f" {color_key}contact(prev/curr)={color_reset}{color_val}"
                    f"{contact_stats.prev_count}/{contact_stats.curr_count}{color_reset} "
                    f"{color_key}matched/new/lost={color_reset}{color_val}"
                    f"{contact_stats.matched_count}/{contact_stats.new_count}/{contact_stats.lost_count}{color_reset}"
                )
                if change_ratio >= warn_threshold:
                    warn_txt = (
                        f" {color_warn}[WARN]{color_reset} "
                        f"{color_key}contact_change_ratio={color_reset}{color_warn}"
                        f"{change_ratio:.3f}{color_reset} "
                        f"{color_key}(thr={warn_threshold:.3f}){color_reset}"
                    )
            if g_sanity_enable:
                g_rows = int(J_motion.shape[0]) if J_motion.ndim == 2 else 0
                g_cols = int(J_motion.shape[1]) if J_motion.ndim == 2 else 0
                rank_g = 0
                cond_g = 0.0
                pred_norm = 0.0
                if (
                    G_pinv.ndim == 2
                    and J_motion.ndim == 2
                    and J_motion.shape[1] == u_t.shape[0]
                ):
                    if G_pinv.shape[1] > 0:
                        G = np.linalg.pinv(G_pinv)
                        rank_g = int(np.linalg.matrix_rank(G))
                        try:
                            cond_g = float(np.linalg.cond(G))
                        except np.linalg.LinAlgError:
                            cond_g = float("inf")
                        pred_norm = float(np.linalg.norm(G_pinv @ J_motion @ u_t))
                    g_sanity_txt = (
                        f" {color_key}Gsanity(rows,cols,rank,cond,|G+Ju|)={color_reset}"
                        f"{color_val}{g_rows},{g_cols},{rank_g},{cond_g:.3e},{pred_norm:.3e}{color_reset}"
                    )
            print(
                f"{color_tag}[EKF-SMOKE]{color_reset} "
                f"{color_key}step={color_reset}{color_val}{ekf_ctx['step']}{color_reset} "
                f"{color_key}n_contacts={color_reset}{color_val}{state_work.n_contacts}{color_reset} "
                f"{color_key}dim_x_expected={color_reset}{color_val}{6 + 3 * state_work.n_contacts}{color_reset} "
                f"{color_key}dim_x={color_reset}{color_val}{step_res.ekf_update.y_next.shape[0]}{color_reset} "
                f"{color_key}|u|={color_reset}{color_val}{u_norm:.6f}{color_reset} "
                f"{color_key}|innov|={color_reset}{color_val}{innov_norm:.6f}{color_reset}"
                f"{contact_stat_txt}"
                f"{g_sanity_txt}"
                f"{warn_txt}"
            )

