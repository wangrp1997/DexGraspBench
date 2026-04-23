import os
import sys

import numpy as np

from .base import BaseEval
from ekf_inhand.contact_manager import build_hq_inputs_with_contact_matching
from ekf_inhand.contact_manager import expand_state_cov_for_new_contacts
from ekf_inhand.contact_manager import shrink_state_cov_for_lost_contacts
from ekf_inhand.input_stream import EkfOnlineInputLogger
from ekf_inhand.init import EkfInitConfig
from ekf_inhand.init import build_initial_covariance
from ekf_inhand.init import build_initial_state
from ekf_inhand.runner import run_one_step_smoke
from ekf_inhand.state import InhandState


class tabletopArmEval(BaseEval):
    def _ensure_object_surface_cache(self, mj_ho):
        if hasattr(self, "_ekf_obj_mesh_local") and self._ekf_obj_mesh_local is not None:
            return

        # Reuse hand_util.py-style mesh extraction from compiled MuJoCo model.
        model = mj_ho.model
        verts_all = []
        faces_all = []
        v_offset = 0
        for gid in range(model.ngeom):
            g = model.geom(gid)
            gname = g.name
            if not isinstance(gname, str) or "object_collision_" not in gname:
                continue
            if int(g.dataid) < 0:
                continue
            mesh = model.mesh(int(g.dataid))
            v = np.asarray(
                model.mesh_vert[mesh.vertadr[0] : mesh.vertadr[0] + mesh.vertnum[0]],
                dtype=float,
            )
            f = np.asarray(
                model.mesh_face[mesh.faceadr[0] : mesh.faceadr[0] + mesh.facenum[0]],
                dtype=np.int32,
            )
            # mesh local -> geom local -> body(object) local
            gpos = np.asarray(model.geom_pos[gid], dtype=float)
            gquat = np.asarray(model.geom_quat[gid], dtype=float)
            rot9 = np.zeros((9,), dtype=float)
            import mujoco  # type: ignore[import-not-found]

            mujoco.mju_quat2Mat(rot9, gquat)
            grot = rot9.reshape(3, 3)
            v_body = (grot @ v.T).T + gpos
            verts_all.append(v_body)
            faces_all.append(f + v_offset)
            v_offset += v_body.shape[0]

        if len(verts_all) == 0:
            self._ekf_obj_mesh_local = None
            return
        verts = np.concatenate(verts_all, axis=0)
        faces = np.concatenate(faces_all, axis=0)
        import trimesh  # type: ignore[import-not-found]

        self._ekf_obj_mesh_local = trimesh.Trimesh(
            vertices=verts, faces=faces, process=False
        )

    @staticmethod
    def _closest_point_on_triangle(p, a, b, c):
        # Real-Time Collision Detection closest point routine.
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = np.dot(ab, ap)
        d2 = np.dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return a
        bp = p - b
        d3 = np.dot(ab, bp)
        d4 = np.dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return b
        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab
        cp = p - c
        d5 = np.dot(ab, cp)
        d6 = np.dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return c
        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac
        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + w * (c - b)
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return a + ab * v + ac * w

    @staticmethod
    def _barycentric_uv(p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = float(np.dot(v0, v0))
        d01 = float(np.dot(v0, v1))
        d11 = float(np.dot(v1, v1))
        d20 = float(np.dot(v2, v0))
        d21 = float(np.dot(v2, v1))
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            return np.array([0.0, 0.0], dtype=float)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        # store (v,w); u can be recovered as 1-v-w
        return np.array([v, w], dtype=float)

    def _ensure_hand_surface_cache(self, mj_ho):
        if hasattr(self, "_ekf_hand_mesh_by_body") and self._ekf_hand_mesh_by_body is not None:
            return
        self._ekf_hand_mesh_by_body = {}
        model = mj_ho.model
        for gid in range(model.ngeom):
            g = model.geom(gid)
            gname = g.name
            if not isinstance(gname, str) or "hand:rh_" not in gname:
                continue
            if int(g.dataid) < 0:
                continue
            body_id = int(g.bodyid)
            mesh = model.mesh(int(g.dataid))
            v = np.asarray(
                model.mesh_vert[mesh.vertadr[0] : mesh.vertadr[0] + mesh.vertnum[0]],
                dtype=float,
            )
            f = np.asarray(
                model.mesh_face[mesh.faceadr[0] : mesh.faceadr[0] + mesh.facenum[0]],
                dtype=np.int32,
            )
            gpos = np.asarray(model.geom_pos[gid], dtype=float)
            gquat = np.asarray(model.geom_quat[gid], dtype=float)
            rot9 = np.zeros((9,), dtype=float)
            import mujoco  # type: ignore[import-not-found]

            mujoco.mju_quat2Mat(rot9, gquat)
            grot = rot9.reshape(3, 3)
            v_body = (grot @ v.T).T + gpos
            if body_id not in self._ekf_hand_mesh_by_body:
                self._ekf_hand_mesh_by_body[body_id] = {"verts": [], "faces": [], "offset": 0}
            rec = self._ekf_hand_mesh_by_body[body_id]
            rec["verts"].append(v_body)
            rec["faces"].append(f + int(rec["offset"]))
            rec["offset"] = int(rec["offset"]) + v_body.shape[0]
        import trimesh  # type: ignore[import-not-found]

        for body_id, rec in list(self._ekf_hand_mesh_by_body.items()):
            vv = np.concatenate(rec["verts"], axis=0)
            ff = np.concatenate(rec["faces"], axis=0)
            mesh = trimesh.Trimesh(vertices=vv, faces=ff, process=False)
            self._ekf_hand_mesh_by_body[body_id] = {
                "mesh": mesh,
                "face_normals": np.asarray(mesh.face_normals, dtype=float),
            }

    def _map_contact_to_object_xi(self, obj_body_id, c_world):
        self._ensure_object_surface_cache(self.mj_ho)
        mesh = getattr(self, "_ekf_obj_mesh_local", None)
        if mesh is None:
            return np.zeros((2,), dtype=float), -1
        data = self.mj_ho.data
        obj_pos = np.asarray(data.xpos[obj_body_id], dtype=float)
        obj_rot = np.asarray(data.xmat[obj_body_id], dtype=float).reshape(3, 3)
        p_local = obj_rot.T @ (np.asarray(c_world, dtype=float) - obj_pos)
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        best_d2 = float("inf")
        best_face = 0
        best_q = p_local
        hint_dist2 = float("inf")
        if faces.shape[0] > 0 and hasattr(self, "_ekf_face_hint_obj"):
            pass
        hint_face_id = -1
        # backward-compatible call path may not pass hint; detect from frame locals
        # when called through mapper interface below, hint is provided by wrapper.
        # keep default full search.
        if hasattr(self, "_ekf_obj_hint_face_runtime"):
            hint_face_id = int(self._ekf_obj_hint_face_runtime)
            self._ekf_obj_hint_face_runtime = -1
        if 0 <= hint_face_id < faces.shape[0]:
            tri = faces[hint_face_id]
            a = verts[tri[0]]
            b = verts[tri[1]]
            c = verts[tri[2]]
            q_hint = self._closest_point_on_triangle(p_local, a, b, c)
            hint_dist2 = float(np.dot(p_local - q_hint, p_local - q_hint))
            if hint_dist2 < 1e-6:
                return self._barycentric_uv(q_hint, a, b, c), int(hint_face_id)
        for fi in range(faces.shape[0]):
            tri = faces[fi]
            a = verts[tri[0]]
            b = verts[tri[1]]
            c = verts[tri[2]]
            q = self._closest_point_on_triangle(p_local, a, b, c)
            d2 = float(np.dot(p_local - q, p_local - q))
            if d2 < best_d2:
                best_d2 = d2
                best_face = fi
                best_q = q
        tri = faces[best_face]
        a = verts[tri[0]]
        b = verts[tri[1]]
        c = verts[tri[2]]
        return self._barycentric_uv(best_q, a, b, c), int(best_face)

    def _map_contact_to_object_xi_with_hint(self, obj_body_id, c_world, hint_face_id=-1):
        self._ekf_obj_hint_face_runtime = int(hint_face_id)
        return self._map_contact_to_object_xi(obj_body_id, c_world)

    def _map_contact_to_hand_surface(self, hand_body_id, c_world, n_obj, hint_face_id=-1):
        self._ensure_hand_surface_cache(self.mj_ho)
        rec = getattr(self, "_ekf_hand_mesh_by_body", {}).get(int(hand_body_id), None)
        if rec is None:
            return np.zeros((2,), dtype=float), -np.asarray(n_obj, dtype=float), -1
        mesh = rec["mesh"]
        face_normals = rec["face_normals"]
        data = self.mj_ho.data
        bpos = np.asarray(data.xpos[int(hand_body_id)], dtype=float)
        brot = np.asarray(data.xmat[int(hand_body_id)], dtype=float).reshape(3, 3)
        p_local = brot.T @ (np.asarray(c_world, dtype=float) - bpos)
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        best_d2 = float("inf")
        best_face = 0
        best_q = p_local
        if 0 <= int(hint_face_id) < faces.shape[0]:
            tri = faces[int(hint_face_id)]
            a = verts[tri[0]]
            b = verts[tri[1]]
            c = verts[tri[2]]
            q_hint = self._closest_point_on_triangle(p_local, a, b, c)
            d2_hint = float(np.dot(p_local - q_hint, p_local - q_hint))
            if d2_hint < 1e-6:
                xi = self._barycentric_uv(q_hint, a, b, c)
                n_local = np.asarray(face_normals[int(hint_face_id)], dtype=float)
                n_world = brot @ n_local
                n_world = n_world / (np.linalg.norm(n_world) + 1e-12)
                return xi, n_world, int(hint_face_id)
        for fi in range(faces.shape[0]):
            tri = faces[fi]
            a = verts[tri[0]]
            b = verts[tri[1]]
            c = verts[tri[2]]
            q = self._closest_point_on_triangle(p_local, a, b, c)
            d2 = float(np.dot(p_local - q, p_local - q))
            if d2 < best_d2:
                best_d2 = d2
                best_face = fi
                best_q = q
        tri = faces[best_face]
        a = verts[tri[0]]
        b = verts[tri[1]]
        c = verts[tri[2]]
        xi = self._barycentric_uv(best_q, a, b, c)
        n_local = np.asarray(face_normals[best_face], dtype=float)
        n_world = brot @ n_local
        n_world = n_world / (np.linalg.norm(n_world) + 1e-12)
        return xi, n_world, int(best_face)

    def _build_hand_normal_fn(self, hand_body_id, face_id):
        self._ensure_hand_surface_cache(self.mj_ho)
        rec = getattr(self, "_ekf_hand_mesh_by_body", {}).get(int(hand_body_id), None)
        if rec is None or int(face_id) < 0:
            def _normal_fn(xi):
                x = np.asarray(xi, dtype=float)
                return np.zeros((x.shape[0], 3), dtype=float)
            return _normal_fn

        mesh = rec["mesh"]
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        tri = faces[int(face_id)]
        a = verts[tri[0]]
        b = verts[tri[1]]
        c = verts[tri[2]]
        data = self.mj_ho.data
        brot = np.asarray(data.xmat[int(hand_body_id)], dtype=float).reshape(3, 3)

        def _normal_fn(xi_batch):
            xi_batch = np.asarray(xi_batch, dtype=float).reshape(-1, 2)
            out = np.zeros((xi_batch.shape[0], 3), dtype=float)
            # triangle plane normal is constant; this still allows Eq.(11) interface closure.
            n_local = np.cross(b - a, c - a)
            n_world = brot @ n_local
            n_world = n_world / (np.linalg.norm(n_world) + 1e-12)
            out[:] = n_world.reshape(1, 3)
            return out

        return _normal_fn

    def _simulate_under_extforce_details(self, pre_obj_qpos):
        ekf_logger = None
        ekf_stage = getattr(self.configs.task, "ekf_input_stage", "post_lift")
        run_ekf_smoke = getattr(self.configs.task, "ekf_smoke_enable", False)
        ekf_ctx = {
            "state": None,
            "P": None,
            "inited": False,
            "step": 0,
            "contact_cache": None,
            "contact_stats": None,
        }
        if getattr(self.configs.task, "ekf_input_debug", False):
            ekf_logger = EkfOnlineInputLogger(
                print_every=getattr(self.configs.task, "ekf_input_print_every", 20)
            )
            if ekf_stage == "all":
                if run_ekf_smoke:
                    self.mj_ho.step_callback = lambda mj: self._ekf_debug_callback(
                        mj, ekf_logger, ekf_ctx
                    )
                else:
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

            # 6. Lift the object
            if ekf_logger is not None and ekf_stage == "post_lift":
                if run_ekf_smoke:
                    self.mj_ho.step_callback = lambda mj: self._ekf_debug_callback(
                        mj, ekf_logger, ekf_ctx
                    )
                else:
                    self.mj_ho.step_callback = ekf_logger.on_step
            self.mj_ho.control_hand_with_interp(
                self.grasp_data["squeeze_qpos"],
                self.grasp_data["lift_qpos"],
            )

            # 7. Hold-and-observe stage after lift.
            # -1 means "keep printing while viewer is open".
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
            # Minimal x0 hint: object position + zero orientation components.
            x0_hint = np.zeros((6,), dtype=float)
            x0_hint[:3] = np.asarray(x_gt[:3], dtype=float)
            init_cfg = EkfInitConfig(init_contacts=0, x0_std=0.0, seed=0)
            state0 = build_initial_state(x0_hint=x0_hint, cfg=init_cfg)
            P0 = build_initial_covariance(state0=state0, cfg=init_cfg)
            ekf_ctx["state"] = state0
            ekf_ctx["P"] = P0
            ekf_ctx["inited"] = True

        state_prev: InhandState = ekf_ctx["state"]
        P_prev: np.ndarray = ekf_ctx["P"]

        m = q.shape[0]
        # Placeholder matrices for smoke integration only.
        G_pinv = np.eye(6, dtype=float)
        J_motion = np.zeros((6, m), dtype=float)
        R_t = np.eye(2 * m, dtype=float) * 1e-3
        (
            c_obj,
            c_f_prev,
            contact_normals_obj,
            J_obs_contact,
            J_pinv,
            xi_new_init,
            f_new_init,
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
        )
        state_work = state_prev
        P_work = P_prev
        Q_work = np.eye(state_work.dim, dtype=float) * 1e-4
        if len(match_events.lost_prev_idx) > 0:
            state_work, P_work, Q_work = shrink_state_cov_for_lost_contacts(
                state_prev=state_work,
                P_prev=P_work,
                Q_prev=Q_work,
                lost_prev_idx=match_events.lost_prev_idx,
            )
        if len(match_events.new_curr_idx) > 0:
            state_work, P_work, Q_work = expand_state_cov_for_new_contacts(
                state_prev=state_work,
                P_prev=P_work,
                Q_prev=Q_work,
                new_count=len(match_events.new_curr_idx),
                xi_init=xi_new_init,
                f_init=f_new_init,
            )

        ekf_ctx["contact_cache"] = next_cache
        ekf_ctx["contact_stats"] = match_stats
        J_obs = np.asarray(J_obs_contact, dtype=float)
        if J_obs.shape != (3 * state_work.n_contacts, m):
            raise RuntimeError(
                f"J_obs shape mismatch, expected {(3 * state_work.n_contacts, m)}, got {J_obs.shape}"
            )
        hq_impl = None

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
            q_prev=q,
            J_obs=J_obs,
            contact_normals_obj=contact_normals_obj,
            hq_impl=hq_impl,
            J_pinv=J_pinv,
            c_obj=c_obj,
            c_f_prev=c_f_prev,
            jac_eps=1e-6,
        )

        ekf_ctx["state"] = InhandState.unpack(step_res.ekf_update.y_next)
        ekf_ctx["P"] = step_res.ekf_update.P_next

        smoke_every = int(getattr(self.configs.task, "ekf_smoke_print_every", 50))
        if smoke_every > 0 and ekf_ctx["step"] % smoke_every == 0:
            # Use magenta-like color to differ from [EKF-IN] cyan tag.
            color_tag = "\033[95m"
            color_key = "\033[94m"
            color_val = "\033[92m"
            color_warn = "\033[91m"
            color_reset = "\033[0m"
            innov_norm = float(np.linalg.norm(step_res.ekf_update.innovation))
            contact_stats = ekf_ctx.get("contact_stats")
            contact_stat_txt = ""
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
            print(
                f"{color_tag}[EKF-SMOKE]{color_reset} "
                f"{color_key}step={color_reset}{color_val}{ekf_ctx['step']}{color_reset} "
                f"{color_key}n_contacts={color_reset}{color_val}{state_work.n_contacts}{color_reset} "
                f"{color_key}dim_x_expected={color_reset}{color_val}{6 + 3 * state_work.n_contacts}{color_reset} "
                f"{color_key}dim_x={color_reset}{color_val}{step_res.ekf_update.y_next.shape[0]}{color_reset} "
                f"{color_key}|innov|={color_reset}{color_val}{innov_norm:.6f}{color_reset}"
                f"{contact_stat_txt}"
                f"{warn_txt}"
            )

