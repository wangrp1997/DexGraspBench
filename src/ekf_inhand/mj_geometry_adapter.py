from __future__ import annotations

import numpy as np


class MjEkfGeometryAdapter:
    """MuJoCo geometry adapter for EKF contact parameterization and Eq.(6) motion matrices."""

    def __init__(self, mj_ho):
        self.mj_ho = mj_ho
        self._obj_mesh_local = None
        self._hand_mesh_by_body = None
        self._obj_hint_face_runtime = -1
        self._obj_body_id = None

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        return np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=float,
        )

    @staticmethod
    def _closest_point_on_triangle(p, a, b, c):
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
        return np.array([v, w], dtype=float)

    @staticmethod
    def _interp_normal_from_uv(uv, na, nb, nc):
        v = float(uv[0])
        w = float(uv[1])
        u = 1.0 - v - w
        n = (
            u * np.asarray(na, dtype=float)
            + v * np.asarray(nb, dtype=float)
            + w * np.asarray(nc, dtype=float)
        )
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            return np.zeros((3,), dtype=float)
        return n / norm

    def get_object_body_id(self) -> int | None:
        if self._obj_body_id is not None:
            return self._obj_body_id
        model = self.mj_ho.model
        for bid in range(model.nbody):
            bname = model.body(bid).name
            if isinstance(bname, str) and "object" in bname:
                self._obj_body_id = int(bid)
                break
        return self._obj_body_id

    def ensure_object_surface_cache(self):
        if self._obj_mesh_local is not None:
            return
        model = self.mj_ho.model
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
            self._obj_mesh_local = None
            return
        verts = np.concatenate(verts_all, axis=0)
        faces = np.concatenate(faces_all, axis=0)
        import trimesh  # type: ignore[import-not-found]

        self._obj_mesh_local = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    def ensure_hand_surface_cache(self):
        if self._hand_mesh_by_body is not None:
            return
        self._hand_mesh_by_body = {}
        model = self.mj_ho.model
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
            if body_id not in self._hand_mesh_by_body:
                self._hand_mesh_by_body[body_id] = {"verts": [], "faces": [], "offset": 0}
            rec = self._hand_mesh_by_body[body_id]
            rec["verts"].append(v_body)
            rec["faces"].append(f + int(rec["offset"]))
            rec["offset"] = int(rec["offset"]) + v_body.shape[0]
        import trimesh  # type: ignore[import-not-found]

        for body_id, rec in list(self._hand_mesh_by_body.items()):
            vv = np.concatenate(rec["verts"], axis=0)
            ff = np.concatenate(rec["faces"], axis=0)
            mesh = trimesh.Trimesh(vertices=vv, faces=ff, process=False)
            self._hand_mesh_by_body[body_id] = {
                "mesh": mesh,
                "face_normals": np.asarray(mesh.face_normals, dtype=float),
                "vertex_normals": np.asarray(mesh.vertex_normals, dtype=float),
            }

    def map_contact_to_object_xi_with_hint(self, obj_body_id, c_world, hint_face_id=-1):
        self.ensure_object_surface_cache()
        mesh = self._obj_mesh_local
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
        hint_face_id = int(hint_face_id)
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

    def map_contact_to_hand_surface(self, hand_body_id, c_world, n_obj, hint_face_id=-1):
        self.ensure_hand_surface_cache()
        rec = getattr(self, "_hand_mesh_by_body", {}).get(int(hand_body_id), None)
        if rec is None:
            return np.zeros((2,), dtype=float), -np.asarray(n_obj, dtype=float), -1
        mesh = rec["mesh"]
        face_normals = rec["face_normals"]
        vertex_normals = rec["vertex_normals"]
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
                n_local = self._interp_normal_from_uv(
                    xi, vertex_normals[tri[0]], vertex_normals[tri[1]], vertex_normals[tri[2]]
                )
                if float(np.linalg.norm(n_local)) < 1e-12:
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
        n_local = self._interp_normal_from_uv(
            xi, vertex_normals[tri[0]], vertex_normals[tri[1]], vertex_normals[tri[2]]
        )
        if float(np.linalg.norm(n_local)) < 1e-12:
            n_local = np.asarray(face_normals[best_face], dtype=float)
        n_world = brot @ n_local
        n_world = n_world / (np.linalg.norm(n_world) + 1e-12)
        return xi, n_world, int(best_face)

    def build_hand_normal_fn(self, hand_body_id, face_id):
        self.ensure_hand_surface_cache()
        rec = getattr(self, "_hand_mesh_by_body", {}).get(int(hand_body_id), None)
        if rec is None or int(face_id) < 0:
            def _normal_fn(xi):
                x = np.asarray(xi, dtype=float)
                return np.zeros((x.shape[0], 3), dtype=float)
            return _normal_fn

        mesh = rec["mesh"]
        faces = np.asarray(mesh.faces, dtype=np.int32)
        vertex_normals = np.asarray(rec["vertex_normals"], dtype=float)
        face_normals = np.asarray(rec["face_normals"], dtype=float)
        tri = faces[int(face_id)]
        na = vertex_normals[tri[0]]
        nb = vertex_normals[tri[1]]
        nc = vertex_normals[tri[2]]
        face_n = np.asarray(face_normals[int(face_id)], dtype=float)
        data = self.mj_ho.data
        brot = np.asarray(data.xmat[int(hand_body_id)], dtype=float).reshape(3, 3)

        def _normal_fn(xi_batch):
            xi_batch = np.asarray(xi_batch, dtype=float).reshape(-1, 2)
            v = xi_batch[:, 0:1]
            w = xi_batch[:, 1:2]
            u = 1.0 - v - w
            n_local = u * na.reshape(1, 3) + v * nb.reshape(1, 3) + w * nc.reshape(1, 3)
            n_local_norm = np.linalg.norm(n_local, axis=1, keepdims=True)
            bad_local = n_local_norm[:, 0] < 1e-12
            if np.any(bad_local):
                n_local[bad_local] = face_n.reshape(1, 3)
                n_local_norm = np.linalg.norm(n_local, axis=1, keepdims=True)
            n_local = n_local / (n_local_norm + 1e-12)
            n_world = n_local @ brot.T
            n_world_norm = np.linalg.norm(n_world, axis=1, keepdims=True)
            return n_world / (n_world_norm + 1e-12)

        return _normal_fn

    def build_motion_grasp_matrices(
        self,
        c_obj: np.ndarray,
        J_obs_contact: np.ndarray,
        m: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        c_obj = np.asarray(c_obj, dtype=float)
        J_obs_contact = np.asarray(J_obs_contact, dtype=float)
        n = int(c_obj.shape[0]) if c_obj.ndim == 2 else 0
        if n == 0:
            return np.zeros((6, 0), dtype=float), np.zeros((0, m), dtype=float)
        if J_obs_contact.shape != (3 * n, m):
            raise ValueError(f"J_obs_contact must be {(3 * n, m)}, got {J_obs_contact.shape}")

        obj_body_id = self.get_object_body_id()
        if obj_body_id is None:
            return np.zeros((6, 3 * n), dtype=float), J_obs_contact.copy()

        obj_pos = np.asarray(self.mj_ho.data.xpos[int(obj_body_id)], dtype=float).reshape(3)
        G = np.zeros((3 * n, 6), dtype=float)
        for i in range(n):
            r_i = c_obj[i] - obj_pos
            G[3 * i : 3 * i + 3, :] = np.concatenate(
                [np.eye(3, dtype=float), -self._skew(r_i)],
                axis=1,
            )
        G_pinv = np.linalg.pinv(G)
        return G_pinv, J_obs_contact.copy()
