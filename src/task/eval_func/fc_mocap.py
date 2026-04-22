import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import np_get_delta_qpos
from task.eval_func.base import BaseEval


class fcMocapEval(BaseEval):
    def _simulate_under_extforce_details(self, pre_obj_qpos):
        is_botyard = self.configs.hand_name == "botyard"
        # Keep default motion cadence aligned with shadow unless explicitly overridden.
        interp_outer = int(os.environ.get("BOTYARD_INTERP_OUTER", "10")) if is_botyard else 10
        interp_inner = int(os.environ.get("BOTYARD_INTERP_INNER", "10")) if is_botyard else 10
        if is_botyard:
            # Comma-separated external-force scales, e.g. "0.0" or "0.2,0.4,0.6".
            force_stage_str = os.environ.get("BOTYARD_FORCE_STAGES", "0.0")
            force_stages = [
                float(x.strip()) for x in force_stage_str.split(",") if x.strip() != ""
            ]
            if len(force_stages) == 0:
                force_stages = [0.0]
        else:
            force_stages = [10.0]
        wait_outer = int(os.environ.get("BOTYARD_WAIT_OUTER", "10")) if is_botyard else 10
        wait_inner = int(os.environ.get("BOTYARD_WAIT_INNER", "50")) if is_botyard else 50
        hold_outer = int(os.environ.get("BOTYARD_HOLD_OUTER", "0")) if is_botyard else 0
        hold_inner = int(os.environ.get("BOTYARD_HOLD_INNER", "0")) if is_botyard else 0
        min_contact_num = int(os.environ.get("BOTYARD_MIN_CONTACT_NUM", "3"))
        min_contact_finger = int(os.environ.get("BOTYARD_MIN_CONTACT_FINGER", "2"))
        contact_th = float(os.environ.get("BOTYARD_CONTACT_TH", "0.0015"))
        enable_contact_gate = os.environ.get("BOTYARD_ENABLE_CONTACT_GATE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        verbose_contact = os.environ.get("BOTYARD_VERBOSE_CONTACT", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        external_force_direction = np.array(
            [
                [-1.0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )

        overall_grasp_passed_all_directions = True
        for i in range(len(external_force_direction)):
            succ_flag = True
            squeeze_qpos = self.grasp_data["squeeze_qpos"].copy()
            if is_botyard:
                # Soft squeeze target to reduce over-closing penetration.
                th_scale = float(os.environ.get("BOTYARD_TH_SQUEEZE_SCALE", "0.80"))
                ff_scale = float(os.environ.get("BOTYARD_FF_SQUEEZE_SCALE", "0.88"))
                mf_scale = float(os.environ.get("BOTYARD_MF_SQUEEZE_SCALE", "0.78"))
                rf_scale = float(os.environ.get("BOTYARD_RF_SQUEEZE_SCALE", "0.88"))
                lf_scale = float(os.environ.get("BOTYARD_LF_SQUEEZE_SCALE", "0.88"))
                grasp_qpos = self.grasp_data["grasp_qpos"]
                # root(7) + [TH4 TH3 TH2 TH1 FF4 FF3 FF2 FF1 MF4 MF3 MF2 MF1 RF4 RF3 RF2 RF1 LF4 LF3 LF2 LF1]
                squeeze_cfg = [
                    ([7, 8, 9, 10], th_scale),
                    ([11, 12, 13, 14], ff_scale),
                    ([15, 16, 17, 18], mf_scale),
                    ([19, 20, 21, 22], rf_scale),
                    ([23, 24, 25, 26], lf_scale),
                ]
                for qidx, scale in squeeze_cfg:
                    squeeze_qpos[qidx] = grasp_qpos[qidx] + scale * (
                        squeeze_qpos[qidx] - grasp_qpos[qidx]
                    )
            self.mj_ho.reset_pose_qpos(
                self.grasp_data["pregrasp_qpos"],
                self.grasp_data["obj_pose"],
            )

            # 2. Move hand to grasp pose
            self.mj_ho.control_hand_with_interp(
                self.grasp_data["pregrasp_qpos"],
                self.grasp_data["grasp_qpos"],
                step_outer=interp_outer,
                step_inner=interp_inner,
            )

            # 3. Move hand to squeeze pose.
            # NOTE step 2 and 3 are seperate because pre -> grasp -> squeeze are stage-wise linear.
            # If step 2 and 3 are merged to one linear interpolation, the performance will drop a lot.
            self.mj_ho.control_hand_with_interp(
                self.grasp_data["grasp_qpos"],
                squeeze_qpos,
                step_outer=interp_outer,
                step_inner=interp_inner,
            )

            # 3.5 Hold at squeeze pose for Botyard to settle contacts before loading.
            if is_botyard:
                self.mj_ho.data.ctrl[:] = self.mj_ho._qpos2ctrl(
                    squeeze_qpos
                )
                for _ in range(hold_outer):
                    self.mj_ho.control_hand_step(step_inner=hold_inner)

                # Optional Botyard contact gate before external-force test.
                # Default OFF (shadow-like behavior): do not fail early by contact count.
                ho_contact, _ = self.mj_ho.get_contact_info(
                    squeeze_qpos,
                    self.grasp_data["obj_pose"],
                    obj_margin=0.0,
                )
                contact_body = set()
                contact_finger = set()
                for c in ho_contact:
                    hand_body_name = c["body1_name"]
                    if (
                        np.abs(c["contact_dist"]) < contact_th
                        and hand_body_name in self.configs.hand.valid_body_name
                    ):
                        contact_body.add(hand_body_name)
                        for finger_prefix in self.configs.hand.finger_prefix:
                            if hand_body_name.startswith(finger_prefix):
                                contact_finger.add(finger_prefix)
                                break
                min_abs_contact_dist = (
                    min(np.abs(c["contact_dist"]) for c in ho_contact)
                    if len(ho_contact) > 0
                    else 1e9
                )
                if verbose_contact:
                    print(
                        "[botyard-contact]",
                        f"dir={i}",
                        f"body={len(contact_body)}",
                        f"finger={len(contact_finger)}",
                        f"min_abs_dist={min_abs_contact_dist:.6f}",
                        f"th={contact_th:.6f}",
                        f"need_body>={min_contact_num}",
                        f"need_finger>={min_contact_finger}",
                    )
                if enable_contact_gate:
                    if len(contact_body) < min_contact_num or len(contact_finger) < min_contact_finger:
                        if verbose_contact:
                            print(
                                "[botyard-contact-fail]",
                                f"dir={i}",
                                "insufficient_contact",
                            )
                        succ_flag = False
                        overall_grasp_passed_all_directions = False
                        break

            if not succ_flag:
                break
            # 4. Add external force on the object (staged for Botyard)
            for stage_idx, stage_force in enumerate(force_stages):
                self.mj_ho.set_ext_force_on_obj(
                    stage_force * external_force_direction[i] * self.configs.task.obj_mass
                )
                # 5. Wait and check stability at each stage
                for _ in range(wait_outer):
                    self.mj_ho.control_hand_step(step_inner=wait_inner)

                    latter_obj_qpos = self.mj_ho.get_obj_pose()
                    delta_pos, delta_angle = np_get_delta_qpos(
                        pre_obj_qpos, latter_obj_qpos
                    )
                    succ_flag = (
                        delta_pos < self.configs.task.simulation_metrics.trans_thre
                    ) & (delta_angle < self.configs.task.simulation_metrics.angle_thre)
                    if not succ_flag:
                        overall_grasp_passed_all_directions = False
                        if is_botyard and verbose_contact:
                            print(
                                "[botyard-force-fail]",
                                f"dir={i}",
                                f"stage={stage_idx}",
                                f"force_scale={stage_force:.3f}",
                                f"delta_pos={delta_pos:.6f}",
                                f"delta_angle={delta_angle:.6f}",
                            )
                        break
                if not succ_flag:
                    break
            if not succ_flag:
                break

        return overall_grasp_passed_all_directions
