import os
from glob import glob
import logging
import multiprocessing
import xml.etree.ElementTree as ET

import numpy as np
import transforms3d.euler as te
from transforms3d import quaternions as tq
import torch

from util.rot_util import torch_quaternion_to_matrix, torch_matrix_to_quaternion


_BOTYARD_CACHE = {}


def _parse_csv_floats(text, expected_len):
    vals = [float(x.strip()) for x in str(text).split(",")]
    if len(vals) != expected_len:
        raise ValueError(f"expect {expected_len} floats, got {len(vals)}")
    return vals


def _find_mujoco_joint_parent_map(xml_path):
    """
    Return a mapping: joint_name -> parent_body_name from MuJoCo XML hierarchy.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parent_map = {}

    def walk_body(body_elem):
        body_name = body_elem.get("name", "")
        for child in body_elem:
            if child.tag == "joint":
                jname = child.get("name")
                if jname:
                    parent_map[jname] = body_name
            elif child.tag == "body":
                walk_body(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        return parent_map
    for body in worldbody.findall("body"):
        walk_body(body)
    return parent_map


def _find_botyard_pmbase_to_palm_from_urdf(urdf_path):
    """
    Parse the fixed PALM joint in Botyard URDF:
      parent link="pmbase", child link="palm", origin xyz/rpy
    Return (xyz, rpy). Fallback to known constants if parse fails.
    """
    fallback_xyz = [-0.011315, 0.0, 0.01845]
    fallback_rpy = [0.0, 0.0, -1.5708]
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            if joint.get("name") != "PALM":
                continue
            parent = joint.find("parent")
            child = joint.find("child")
            origin = joint.find("origin")
            if parent is None or child is None or origin is None:
                continue
            if parent.get("link") == "pmbase" and child.get("link") == "palm":
                xyz = [float(x) for x in origin.get("xyz", "-0.011315 0 0.01845").split()]
                rpy = [float(x) for x in origin.get("rpy", "0 0 -1.5708").split()]
                return xyz, rpy
    except Exception as e:
        logging.warning(f"Failed to parse Botyard URDF PALM joint: {e}")
    return fallback_xyz, fallback_rpy


def _get_botyard_alignment_and_mapping():
    """
    Build all Botyard conversion meta in one place:
    - source 16-DoF joint order (BODex cspace order by name)
    - target 20-DoF MuJoCo order (from XML traversal + equality mimic)
    - pmbase->palm transform from URDF PALM joint
    """
    if "meta" in _BOTYARD_CACHE:
        return _BOTYARD_CACHE["meta"]

    bodex_cfg = "/home/rw/Documents/BODex/src/curobo/content/configs/robot/right_botyard_hand_sim.yml"
    botyard_urdf = "/home/rw/Documents/BODex/src/curobo/content/assets/robot/botyard_description/botyard_rh.urdf"
    dgb_xml = "/home/rw/Documents/DexGraspBench/assets/hand/botyard/right_hand_noforearm.xml"

    # Source 16-DoF order from BODex robot cspace.
    source_joint_names = [
        "FFJ4", "FFJ3", "FFJ2",
        "MFJ4", "MFJ3", "MFJ2",
        "RFJ4", "RFJ3", "RFJ2",
        "LFJ4", "LFJ3", "LFJ2",
        "THJ4", "THJ3", "THJ2", "THJ1",
    ]
    try:
        import yaml

        with open(bodex_cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cspace_names = (
            cfg.get("robot_cfg", {})
            .get("kinematics", {})
            .get("cspace", {})
            .get("joint_names", None)
        )
        if isinstance(cspace_names, list) and len(cspace_names) == 16:
            source_joint_names = [str(x) for x in cspace_names]
    except Exception as e:
        logging.warning(f"Fallback to default Botyard source joint order: {e}")

    src_name_to_idx = {n: i for i, n in enumerate(source_joint_names)}

    # Target 20-DoF order from MuJoCo XML joint tree (hand-only).
    parent_map = _find_mujoco_joint_parent_map(dgb_xml)
    canonical_target = [
        "THJ4", "THJ3", "THJ2", "THJ1",
        "FFJ4", "FFJ3", "FFJ2", "FFJ1",
        "MFJ4", "MFJ3", "MFJ2", "MFJ1",
        "RFJ4", "RFJ3", "RFJ2", "RFJ1",
        "LFJ4", "LFJ3", "LFJ2", "LFJ1",
    ]
    if all(j in parent_map for j in canonical_target):
        target_joint_names = canonical_target
    else:
        # If XML evolves, keep stable fallback.
        target_joint_names = canonical_target

    # Mimic rules from URDF / MuJoCo equality: *J1 follows *J2.
    mimic_from = {
        "FFJ1": "FFJ2",
        "MFJ1": "MFJ2",
        "RFJ1": "RFJ2",
        "LFJ1": "LFJ2",
    }

    # pmbase->palm from URDF PALM fixed joint.
    pmbase_to_palm_xyz, pmbase_to_palm_rpy = _find_botyard_pmbase_to_palm_from_urdf(botyard_urdf)

    meta = {
        "source_joint_names": source_joint_names,
        "target_joint_names": target_joint_names,
        "src_name_to_idx": src_name_to_idx,
        "mimic_from": mimic_from,
        "pmbase_to_palm_xyz": pmbase_to_palm_xyz,
        "pmbase_to_palm_rpy": pmbase_to_palm_rpy,
    }
    _BOTYARD_CACHE["meta"] = meta
    return meta


def resolve_bodex_scene_cfg_path(scene_path_rel: str, data_file: str) -> str:
    """
    BODex stores scene_path relative to .../src/curobo/content/. DexGraspBench is
    often run from another cwd, so relative paths must be anchored to that root.
    """
    scene_path_rel = os.path.normpath(str(scene_path_rel))
    if os.path.isabs(scene_path_rel) and os.path.isfile(scene_path_rel):
        return scene_path_rel
    if os.path.isfile(scene_path_rel):
        return os.path.abspath(scene_path_rel)
    marker = os.path.normpath("src/curobo/content")
    norm_data = os.path.normpath(data_file)
    idx = norm_data.find(marker)
    if idx != -1:
        root = norm_data[: idx + len(marker)]
        candidate = os.path.normpath(os.path.join(root, scene_path_rel))
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Scene cfg not found: {scene_path_rel!r} (cwd={os.getcwd()}). "
        f"Expected under BODex content root inferred from data file: {data_file!r}"
    )


def load_scene_cfg(scene_path):
    scene_cfg = np.load(scene_path, allow_pickle=True).item()

    def update_relative_path(d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                update_relative_path(v)
            elif k.endswith("_path") and isinstance(v, str):
                d[k] = os.path.join(os.path.dirname(scene_path), v)
        return

    update_relative_path(scene_cfg["scene"])

    return scene_cfg


def BODex(params):
    data_file, configs = params[0], params[1]

    raw_data = np.load(data_file, allow_pickle=True).item()
    robot_pose = raw_data["robot_pose"][0]
    new_data = {}

    # For Botyard, reorder candidates by grasp quality before conversion.
    if configs.hand_name == "botyard" and robot_pose.ndim >= 3 and robot_pose.shape[0] > 1:
        score = np.zeros((robot_pose.shape[0],), dtype=np.float64)
        has_quality = False
        if "grasp_error" in raw_data:
            ge = np.asarray(raw_data["grasp_error"])[0]
            if ge.ndim >= 2 and ge.shape[0] == robot_pose.shape[0]:
                score += np.linalg.norm(ge, axis=-1)
                has_quality = True
        if "dist_error" in raw_data:
            de = np.asarray(raw_data["dist_error"])[0]
            if de.ndim >= 2 and de.shape[0] == robot_pose.shape[0]:
                score += np.linalg.norm(de, axis=-1)
                has_quality = True
        if not has_quality:
            # Fallback: prefer smoother pregrasp->squeeze joint movement.
            score = np.linalg.norm(robot_pose[:, 2, 7:] - robot_pose[:, 0, 7:], axis=-1)
        order = np.argsort(score)
        robot_pose = robot_pose[order]

    scene_path_raw = raw_data["scene_path"]
    if isinstance(scene_path_raw, (list, tuple, np.ndarray)):
        scene_path_raw = scene_path_raw[0]
    scene_path_raw = str(scene_path_raw)
    scene_path = (
        scene_path_raw.split("src/curobo/content/")[1]
        if "src/curobo/content/" in scene_path_raw
        else scene_path_raw
    )
    scene_path = resolve_bodex_scene_cfg_path(scene_path, data_file)
    scene_cfg = load_scene_cfg(scene_path)
    obj_name = scene_cfg["task"]["obj_name"]
    new_data["obj_scale"] = scene_cfg["scene"][obj_name]["scale"][0]
    new_data["obj_pose"] = scene_cfg["scene"][obj_name]["pose"]
    new_data["obj_path"] = os.path.dirname(
        os.path.dirname(scene_cfg["scene"][obj_name]["file_path"])
    )
    new_data["scene_path"] = scene_path

    if configs.hand_name == "shadow":
        # Change qpos order of thumb
        robot_pose = np.concatenate(
            [robot_pose[:, :, :7], robot_pose[:, :, 12:], robot_pose[:, :, 7:12]],
            axis=-1,
        )
        # Add a translation bias of palm which is included in XML but ignored in URDF
        tmp_rot = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))
        robot_pose[:, :, :3] -= (
            (tmp_rot @ torch.tensor([0, 0, 0.034]).view(1, 1, 3, 1)).squeeze(-1).numpy()
        )
    elif configs.hand_name == "allegro":
        # Add a rotation bias of palm which is included in XML but ignored in URDF
        tmp_rot = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))
        delta_rot = torch_quaternion_to_matrix(torch.tensor([0, 1, 0, 1]).view(1, 1, 4))
        robot_pose[:, :, 3:7] = torch_matrix_to_quaternion(
            tmp_rot @ delta_rot.transpose(-1, -2)
        )
    elif configs.hand_name == "ur10e_shadow":
        robot_pose = np.concatenate(
            [robot_pose[:, :, :8], robot_pose[:, :, 13:], robot_pose[:, :, 8:13]],
            axis=-1,
        )
    elif configs.hand_name == "leap":
        # Add a translation and rotation bias of palm which is included in XML but ignored in URDF
        tmp_rot = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))
        delta_rot = torch_quaternion_to_matrix(torch.tensor([0, 1, 0, 0]).view(1, 1, 4))
        tmp_rot = tmp_rot @ delta_rot.transpose(-1, -2)
        robot_pose[:, :, 3:7] = torch_matrix_to_quaternion(tmp_rot)
        robot_pose[:, :, :3] -= (tmp_rot @ torch.tensor([0, 0, 0.1])).numpy()
        pass
    elif configs.hand_name == "botyard":
        meta = _get_botyard_alignment_and_mapping()
        target_joint_names = meta["target_joint_names"]
        mimic_from = meta["mimic_from"]
        src_name_to_idx = dict(meta["src_name_to_idx"])

        # Prefer per-file joint order from BODex raw output when available.
        # This is the most reliable source and avoids hard-coded assumptions.
        raw_joint_names = raw_data.get("joint_names", None)
        if isinstance(raw_joint_names, (list, tuple, np.ndarray)):
            names = [str(x) for x in list(raw_joint_names)]
            if len(names) == 16 and len(set(names)) == 16:
                src_name_to_idx = {n: i for i, n in enumerate(names)}
                logging.info(f"Use Botyard raw joint_names order: {names}")

        # Expand BODex Botyard qpos (23 = 7 root + 16 hand joints) to MuJoCo
        # botyard no-forearm xml qpos (27 = 7 root + 20 hand joints) by joint names.
        if robot_pose.shape[-1] == 23:
            root = robot_pose[..., :7]
            src = robot_pose[..., 7:]
            target_vals = []
            for j in target_joint_names:
                src_j = mimic_from.get(j, j)
                if src_j not in src_name_to_idx:
                    raise KeyError(f"Botyard joint mapping missing source joint: {src_j}")
                target_vals.append(src[..., src_name_to_idx[src_j]])
            joints20 = np.stack(target_vals, axis=-1)
            robot_pose = np.concatenate([root, joints20], axis=-1)

        # Root frame conversion:
        # BODex root is palm (base_link="palm"), DGBench root is pmbase.
        # Use URDF PALM joint (pmbase->palm), then compute:
        #   R_new = R_old @ R(pmbase->palm)^T
        #   p_new = p_old - R_new * t(pmbase->palm)
        # Optional env overrides remain available for manual experiments.
        auto_rpy = meta["pmbase_to_palm_rpy"]
        auto_xyz = meta["pmbase_to_palm_xyz"]

        rpy_bias_str = os.environ.get(
            "BOTYARD_FIXED_RPY_BIAS",
            f"{-auto_rpy[0]},{-auto_rpy[1]},{-auto_rpy[2]}",
        ).strip()
        try:
            rpy_vals = _parse_csv_floats(rpy_bias_str, 3)
            r_bias = torch.tensor(
                te.euler2mat(*rpy_vals, axes="sxyz"), dtype=torch.float32
            ).view(1, 1, 3, 3)
            r_old = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))
            r_new = r_old @ r_bias
            robot_pose[:, :, 3:7] = torch_matrix_to_quaternion(r_new).numpy()
            logging.info(f"Apply BOTYARD_FIXED_RPY_BIAS={rpy_vals}")
        except ValueError:
            logging.warning(
                f"Invalid BOTYARD_FIXED_RPY_BIAS={rpy_bias_str!r}, expected 'r,p,y'."
            )
            r_new = torch_quaternion_to_matrix(torch.tensor(robot_pose[:, :, 3:7]))

        bias_vec_str = os.environ.get(
            "BOTYARD_FIXED_BIAS_VEC",
            f"{auto_xyz[0]},{auto_xyz[1]},{auto_xyz[2]}",
        ).strip()
        try:
            vals = _parse_csv_floats(bias_vec_str, 3)
            delta = (
                r_new @ torch.tensor(vals, dtype=torch.float32).view(1, 1, 3, 1)
            ).squeeze(-1).numpy()
            # Follow shadow convention: root -= R * bias.
            robot_pose[:, :, :3] -= delta
            logging.info(f"Apply BOTYARD_FIXED_BIAS_VEC={vals}")
            if os.environ.get("BOTYARD_PRINT_ALIGN_DIAG", "1").strip().lower() in {
                "1", "true", "yes", "on"
            }:
                logging.info(
                    "Botyard auto-alignment meta: "
                    f"src16={meta['source_joint_names']}, "
                    f"target20={meta['target_joint_names']}, "
                    f"auto_pmbase_to_palm_xyz={auto_xyz}, "
                    f"auto_pmbase_to_palm_rpy={auto_rpy}"
                )
        except ValueError:
            logging.warning(
                f"Invalid BOTYARD_FIXED_BIAS_VEC={bias_vec_str!r}, expected 'x,y,z'."
            )
    else:
        raise NotImplementedError

    for i in range(len(robot_pose)):
        if configs.hand.mocap:
            new_data["pregrasp_qpos"] = robot_pose[i, 0]
            new_data["grasp_qpos"] = robot_pose[i, 1]
            new_data["squeeze_qpos"] = robot_pose[i, 2]
        else:
            new_data["approach_qpos"] = robot_pose[i, :-4]
            new_data["pregrasp_qpos"] = robot_pose[i, -4]
            new_data["grasp_qpos"] = robot_pose[i, -3]
            new_data["squeeze_qpos"] = robot_pose[i, -2]
            new_data["lift_qpos"] = robot_pose[i, -1]
        save_path = (
            data_file.replace(configs.task.data_path, configs.grasp_dir)
            .replace("_grasp.npy", f"/{i}_grasp.npy")
            .replace("_mogen.npy", f"/{i}_mogen.npy")
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, new_data)
    return


def Learning(params):
    data_file, configs = params[0], params[1]
    raw_data = np.load(data_file, allow_pickle=True).item()
    scene_cfg = load_scene_cfg(raw_data["scene_path"])
    target_obj = scene_cfg["task"]["obj_name"]
    new_data = {}
    new_data["obj_path"] = os.path.dirname(
        os.path.dirname(scene_cfg["scene"][target_obj]["file_path"])
    )
    new_data["obj_pose"] = scene_cfg["scene"][target_obj]["pose"]
    new_data["obj_scale"] = scene_cfg["scene"][target_obj]["scale"][0]
    save_path = data_file.replace(configs.task.data_path, configs.grasp_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    new_data["grasp_qpos"] = raw_data["grasp_qpos"]
    new_data["pregrasp_qpos"] = raw_data["pregrasp_qpos"]
    new_data["squeeze_qpos"] = raw_data["squeeze_qpos"]
    np.save(save_path, new_data)
    return


def Batched(params):
    data_file, configs = params[0], params[1]
    raw_data = np.load(data_file, allow_pickle=True).item()
    scene_cfg = load_scene_cfg(raw_data["scene_path"])
    target_obj = scene_cfg["task"]["obj_name"]
    new_data = {}
    new_data["obj_path"] = os.path.dirname(
        os.path.dirname(scene_cfg["scene"][target_obj]["file_path"])
    )
    new_data["obj_pose"] = scene_cfg["scene"][target_obj]["pose"]
    obj_scale_in_scene = scene_cfg["scene"][target_obj]["scale"][0]
    save_path = data_file.replace(configs.task.data_path, configs.grasp_dir)
    for i in range(raw_data["grasp_qpos"].shape[0]):
        save_path = os.path.join(save_path.split(".npy")[0], f"{i}.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        new_data["obj_scale"] = obj_scale_in_scene * raw_data["scene_scale"][i]
        new_data["grasp_qpos"] = raw_data["grasp_qpos"][i]
        new_data["pregrasp_qpos"] = raw_data["pregrasp_qpos"][i]
        new_data["squeeze_qpos"] = raw_data["squeeze_qpos"][i]
        np.save(save_path, new_data)
    return


def task_format(configs):
    if configs.task.data_name == "BODex":
        if configs.hand.mocap:
            raw_data_struct = ["**", "*_grasp.npy"]
        else:
            raw_data_struct = ["**", "*_mogen.npy"]
    else:
        raw_data_struct = ["**", "*.npy"]
    raw_data_path_lst = glob(
        os.path.join(configs.task.data_path, *raw_data_struct), recursive=True
    )
    raw_file_num = len(raw_data_path_lst)
    if configs.task.max_num > 0:
        raw_data_path_lst = np.random.permutation(sorted(raw_data_path_lst))[
            : configs.task.max_num
        ]
    logging.info(
        f"Find {raw_file_num} raw files for {os.path.join(configs.task.data_path, *raw_data_struct)}, use {len(raw_data_path_lst)}"
    )

    if len(raw_data_path_lst) == 0:
        return

    iterable_params = zip(raw_data_path_lst, [configs] * len(raw_data_path_lst))
    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(eval(configs.task.data_name), iterable_params)
        results = list(result_iter)

    grasp_lst = glob(os.path.join(configs.grasp_dir, "**/*.npy"), recursive=True)
    logging.info(f"Get {len(grasp_lst)} grasp data in {configs.save_dir}")
    logging.info(f"Finish format conversion")
    return
