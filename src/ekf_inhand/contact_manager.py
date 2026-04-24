from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .observation_model import xi_f_update_eq11
from .state import InhandState


@dataclass(frozen=True)
class ContactKey:
    hand_body_id: int
    obj_body_id: int
    hand_geom_id: int
    obj_geom_id: int


@dataclass(frozen=True)
class ContactMatchStats:
    matched_count: int
    new_count: int
    lost_count: int
    prev_count: int
    curr_count: int


@dataclass(frozen=True)
class ContactMatchEvents:
    # index mapping between previous and current contact lists
    persist_prev_idx: list[int]
    persist_curr_idx: list[int]
    new_curr_idx: list[int]
    lost_prev_idx: list[int]
    prev_count: int
    curr_count: int


ContactCache = dict[ContactKey, list[dict[str, np.ndarray | int]]]


def _estimate_contact_force_eq23(
    J_block: np.ndarray,
    n_obj: np.ndarray,
    tau_meas: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """
    Paper Eq.(23) scalar least-squares inversion:
      f_j = (J_j^T n_j)^+ tau

    For scalar f_j, this is equivalent to solving
      min_f ||a_j * f - tau||_2, where a_j = J_j^T n_j (shape m,).

    - Regular case (||a_j|| > eps): closed-form LS
        f_j = (a_j^T tau) / (a_j^T a_j)
      which equals Moore-Penrose pseudoinverse.
    - Degenerate case (||a_j|| <= eps): return 0.0
      to keep the estimator numerically safe and deterministic.
    """
    a_j = np.asarray(J_block, dtype=float).T @ np.asarray(n_obj, dtype=float).reshape(3)
    tau = np.asarray(tau_meas, dtype=float).reshape(-1)
    if a_j.shape != tau.shape:
        raise ValueError(
            f"Eq.(23) dimension mismatch: a_j {a_j.shape} vs tau {tau.shape}"
        )
    denom = float(np.dot(a_j, a_j))
    if denom <= float(eps):
        return 0.0
    return float(np.dot(a_j, tau) / denom)


def build_hq_inputs_with_contact_matching(
    mj_ho,
    m: int,
    tau_meas: np.ndarray | None,
    finger_dof_idx: np.ndarray | None,
    prev_contact_cache: ContactCache | None,
    obj_xi_mapper=None,
    hand_surface_mapper=None,
    hand_normal_fn_builder=None,
    rho_plus: float | None = None,
    rho_minus: float | None = None,
    contact_score_eps: float = 1e-9,
    nearest_dist_threshold: float = 5e-3,
):
    """
    Build Eq.(8) inputs with contact correspondence matching:
      q_hat = q_prev + J^+ (c_obj_t - c_f_{t-1})

    Returns:
      c_obj: (n, 3)
      c_f_prev: (n, 3)
      contact_normals_obj: (n, 3)
      J_obs: (3n, m)
      J_pinv: (m, 3n)
      xi_new_init: (2*k,), ordered by match_events.new_curr_idx
      f_new_init: (k,), ordered by match_events.new_curr_idx
      next_contact_cache: ContactCache
      match_stats: ContactMatchStats
      match_events: ContactMatchEvents
    """
    import mujoco  # type: ignore[import-not-found]

    data = mj_ho.data
    model = mj_ho.model
    if finger_dof_idx is None:
        finger_dof_idx = np.arange(m, dtype=np.int32)
    else:
        finger_dof_idx = np.asarray(finger_dof_idx, dtype=np.int32)
    tau_meas = (
        np.zeros((m,), dtype=float)
        if tau_meas is None
        else np.asarray(tau_meas, dtype=float).reshape(-1)
    )
    if tau_meas.shape[0] != m:
        raise ValueError(f"tau_meas dim must be {m}, got {tau_meas.shape[0]}")

    grouped_prev: dict[ContactKey, list[tuple[int, dict[str, np.ndarray | int]]]] = {}
    prev_flat_count = 0
    for k, entry_list in (prev_contact_cache or {}).items():
        slots = []
        for entry in entry_list:
            e = {
                "pos": np.asarray(entry["pos"], dtype=float),
            }
            if "xi_f" in entry:
                e["xi_f"] = np.asarray(entry["xi_f"], dtype=float)
            if "n_f" in entry:
                e["n_f"] = np.asarray(entry["n_f"], dtype=float)
            if "face_id" in entry:
                e["face_id"] = int(entry["face_id"])
            if "obj_face_id" in entry:
                e["obj_face_id"] = int(entry["obj_face_id"])
            slots.append((prev_flat_count, e))
            prev_flat_count += 1
        grouped_prev[k] = slots

    c_obj_lst = []
    c_f_prev_lst = []
    n_obj_lst = []
    J_blocks = []
    xi_obj_curr_lst = []
    f_curr_lst = []
    next_contact_cache: ContactCache = {}
    matched_count = 0
    new_count = 0
    persist_prev_idx: list[int] = []
    persist_curr_idx: list[int] = []
    new_curr_idx: list[int] = []

    if (rho_plus is None) != (rho_minus is None):
        raise ValueError("rho_plus and rho_minus must be both set or both None")
    use_hysteresis = rho_plus is not None and rho_minus is not None
    if use_hysteresis and float(rho_plus) <= float(rho_minus):
        raise ValueError("hysteresis requires rho_plus > rho_minus")

    for contact in data.contact:
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        body1_id = int(model.geom(geom1_id).bodyid)
        body2_id = int(model.geom(geom2_id).bodyid)
        body1_name = model.body(body1_id).name
        body2_name = model.body(body2_id).name
        body1_is_hand = "hand:rh_" in body1_name
        body2_is_hand = "hand:rh_" in body2_name
        body1_is_obj = "object" in body1_name
        body2_is_obj = "object" in body2_name
        if not ((body1_is_hand and body2_is_obj) or (body2_is_hand and body1_is_obj)):
            continue

        c_pos = np.asarray(contact.pos, dtype=float)
        n = np.asarray(contact.frame[0:3], dtype=float)
        if body1_is_hand and body2_is_obj:
            hand_body_id = body1_id
            obj_body_id = body2_id
            hand_geom_id = geom1_id
            obj_geom_id = geom2_id
            n_obj = -n
        else:
            hand_body_id = body2_id
            obj_body_id = body1_id
            hand_geom_id = geom2_id
            obj_geom_id = geom1_id
            n_obj = n

        key = ContactKey(
            hand_body_id=hand_body_id,
            obj_body_id=obj_body_id,
            hand_geom_id=hand_geom_id,
            obj_geom_id=obj_geom_id,
        )

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jac(model, data, jacp, jacr, c_pos, hand_body_id)
        J_block = jacp[:, finger_dof_idx]
        # Eq.(23): f_j = (J_j^T n_j)^+ tau
        # Here we use equivalent scalar LS closed-form with explicit degeneracy guard.
        f_j = _estimate_contact_force_eq23(
            J_block=J_block,
            n_obj=n_obj,
            tau_meas=tau_meas,
        )
        d_eff = float(contact.dist) if float(contact.dist) > float(contact_score_eps) else float(contact_score_eps)
        w_j = abs(float(f_j)) / np.sqrt(d_eff)

        curr_idx = len(c_obj_lst)
        # finger-side current mapping (Eq.(10)(11) related quantities)
        xi_f_curr = np.zeros((2,), dtype=float)
        n_f_curr = -n_obj.copy()
        face_id_curr = -1
        prev_candidates = grouped_prev.get(key, [])
        was_in_contact = len(prev_candidates) > 0
        if use_hysteresis:
            if (not was_in_contact) and (w_j <= float(rho_plus)):
                continue
            if was_in_contact and (w_j < float(rho_minus)):
                continue
        obj_face_id_curr = -1
        if len(prev_candidates) > 0:
            dists = np.array(
                [np.linalg.norm(c_pos - np.asarray(p[1]["pos"], dtype=float)) for p in prev_candidates],
                dtype=float,
            )
            best_i = int(np.argmin(dists))
            if float(dists[best_i]) <= nearest_dist_threshold:
                prev_idx, prev_entry = prev_candidates.pop(best_i)
                c_prev = np.asarray(prev_entry["pos"], dtype=float)
                matched_count += 1
                persist_prev_idx.append(int(prev_idx))
                persist_curr_idx.append(int(curr_idx))
                # use previous face hints for faster re-projection
                obj_face_hint = int(prev_entry.get("obj_face_id", -1))
                hand_face_hint = int(prev_entry.get("face_id", -1))
                if obj_xi_mapper is not None:
                    xi_obj, obj_face_id_curr = obj_xi_mapper(
                        obj_body_id, c_pos, obj_face_hint
                    )
                    xi_obj = np.asarray(xi_obj, dtype=float).reshape(-1)
                    obj_face_id_curr = int(obj_face_id_curr)
                    if xi_obj.shape[0] != 2:
                        raise ValueError(
                            f"obj_xi_mapper must return xi shape (2,), got {xi_obj.shape}"
                        )
                else:
                    obj_pos = np.asarray(data.xpos[obj_body_id], dtype=float)
                    obj_rot = np.asarray(data.xmat[obj_body_id], dtype=float).reshape(3, 3)
                    c_obj_local = obj_rot.T @ (c_pos - obj_pos)
                    xi_obj = c_obj_local[:2].copy()
                if hand_surface_mapper is not None:
                    xi_f_curr, n_f_curr, face_id_curr = hand_surface_mapper(
                        hand_body_id, c_pos, n_obj, hand_face_hint
                    )
                    xi_f_curr = np.asarray(xi_f_curr, dtype=float).reshape(-1)
                    n_f_curr = np.asarray(n_f_curr, dtype=float).reshape(-1)
                    face_id_curr = int(face_id_curr)
                    if xi_f_curr.shape[0] != 2:
                        raise ValueError(
                            f"hand_surface_mapper xi_f must be (2,), got {xi_f_curr.shape}"
                        )
                    if n_f_curr.shape[0] != 3:
                        raise ValueError(
                            f"hand_surface_mapper n_f must be (3,), got {n_f_curr.shape}"
                        )
                # Eq.(11): update xi_f,t from xi_f,t-1 and normals
                if "xi_f" in prev_entry and "n_f" in prev_entry:
                    xi_prev = np.asarray(prev_entry["xi_f"], dtype=float).reshape(1, 2)
                    n_f_prev = np.asarray(prev_entry["n_f"], dtype=float).reshape(1, 3)
                    n_obj_t = n_obj.reshape(1, 3)
                    if hand_normal_fn_builder is not None:
                        face_id_prev = int(prev_entry.get("face_id", -1))
                        normal_fn = hand_normal_fn_builder(hand_body_id, face_id_prev)
                        xi_f_curr = xi_f_update_eq11(
                            xi_f_prev=xi_prev,
                            n_obj_t=n_obj_t,
                            n_f_prev=n_f_prev,
                            normal_fn=normal_fn,
                        ).reshape(-1)
                    else:
                        xi_f_curr = xi_prev.reshape(-1)
            else:
                c_prev = c_pos.copy()
                new_count += 1
                new_curr_idx.append(int(curr_idx))
                if obj_xi_mapper is not None:
                    xi_obj, obj_face_id_curr = obj_xi_mapper(obj_body_id, c_pos, -1)
                    xi_obj = np.asarray(xi_obj, dtype=float).reshape(-1)
                    obj_face_id_curr = int(obj_face_id_curr)
                    if xi_obj.shape[0] != 2:
                        raise ValueError(
                            f"obj_xi_mapper must return xi shape (2,), got {xi_obj.shape}"
                        )
                else:
                    obj_pos = np.asarray(data.xpos[obj_body_id], dtype=float)
                    obj_rot = np.asarray(data.xmat[obj_body_id], dtype=float).reshape(3, 3)
                    c_obj_local = obj_rot.T @ (c_pos - obj_pos)
                    xi_obj = c_obj_local[:2].copy()
                if hand_surface_mapper is not None:
                    xi_f_curr, n_f_curr, face_id_curr = hand_surface_mapper(
                        hand_body_id, c_pos, n_obj, -1
                    )
                    xi_f_curr = np.asarray(xi_f_curr, dtype=float).reshape(-1)
                    n_f_curr = np.asarray(n_f_curr, dtype=float).reshape(-1)
                    face_id_curr = int(face_id_curr)
        else:
            c_prev = c_pos.copy()
            new_count += 1
            new_curr_idx.append(int(curr_idx))
            if obj_xi_mapper is not None:
                xi_obj, obj_face_id_curr = obj_xi_mapper(obj_body_id, c_pos, -1)
                xi_obj = np.asarray(xi_obj, dtype=float).reshape(-1)
                obj_face_id_curr = int(obj_face_id_curr)
                if xi_obj.shape[0] != 2:
                    raise ValueError(
                        f"obj_xi_mapper must return xi shape (2,), got {xi_obj.shape}"
                    )
            else:
                obj_pos = np.asarray(data.xpos[obj_body_id], dtype=float)
                obj_rot = np.asarray(data.xmat[obj_body_id], dtype=float).reshape(3, 3)
                c_obj_local = obj_rot.T @ (c_pos - obj_pos)
                xi_obj = c_obj_local[:2].copy()
            if hand_surface_mapper is not None:
                xi_f_curr, n_f_curr, face_id_curr = hand_surface_mapper(
                    hand_body_id, c_pos, n_obj, -1
                )
                xi_f_curr = np.asarray(xi_f_curr, dtype=float).reshape(-1)
                n_f_curr = np.asarray(n_f_curr, dtype=float).reshape(-1)
                face_id_curr = int(face_id_curr)

        c_obj_lst.append(c_pos)
        c_f_prev_lst.append(c_prev)
        n_obj_lst.append(n_obj)
        J_blocks.append(J_block)
        xi_obj_curr_lst.append(xi_obj)
        f_curr_lst.append(f_j)
        next_contact_cache.setdefault(key, []).append(
            {
                "pos": c_pos.copy(),
                "xi_f": xi_f_curr.copy(),
                "n_f": n_f_curr.copy(),
                "face_id": int(face_id_curr),
                "obj_face_id": int(obj_face_id_curr),
            }
        )

    n_contacts = len(c_obj_lst)
    prev_count = int(prev_flat_count)
    curr_count = int(n_contacts)
    lost_count = max(0, prev_count - matched_count)
    matched_prev_set = set(persist_prev_idx)
    lost_prev_idx = [i for i in range(prev_count) if i not in matched_prev_set]
    stats = ContactMatchStats(
        matched_count=matched_count,
        new_count=new_count,
        lost_count=lost_count,
        prev_count=prev_count,
        curr_count=curr_count,
    )
    events = ContactMatchEvents(
        persist_prev_idx=persist_prev_idx,
        persist_curr_idx=persist_curr_idx,
        new_curr_idx=new_curr_idx,
        lost_prev_idx=lost_prev_idx,
        prev_count=prev_count,
        curr_count=curr_count,
    )
    if n_contacts == 0:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=float),
            np.zeros((0, m), dtype=float),
            np.zeros((m, 0), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
            {},
            stats,
            events,
        )

    c_obj = np.stack(c_obj_lst, axis=0)
    c_f_prev = np.stack(c_f_prev_lst, axis=0)
    contact_normals_obj = np.stack(n_obj_lst, axis=0)
    J_obs = np.concatenate(J_blocks, axis=0)  # (3n, m)
    J_pinv = np.linalg.pinv(J_obs)  # (m, 3n)
    xi_obj_curr = np.stack(xi_obj_curr_lst, axis=0)  # (n, 2)
    f_curr = np.asarray(f_curr_lst, dtype=float).reshape(-1)  # (n,)
    if len(events.new_curr_idx) > 0:
        idx = np.asarray(events.new_curr_idx, dtype=np.int32)
        xi_new_init = xi_obj_curr[idx].reshape(-1)
        f_new_init = f_curr[idx]
    else:
        xi_new_init = np.zeros((0,), dtype=float)
        f_new_init = np.zeros((0,), dtype=float)
    return (
        c_obj,
        c_f_prev,
        contact_normals_obj,
        J_obs,
        J_pinv,
        xi_new_init,
        f_new_init,
        next_contact_cache,
        stats,
        events,
    )


def shrink_state_cov_for_lost_contacts(
    state_prev: InhandState,
    P_prev: np.ndarray,
    Q_prev: np.ndarray,
    lost_prev_idx: list[int],
) -> tuple[InhandState, np.ndarray, np.ndarray]:
    """Remove lost contacts from y/P/Q using previous contact indexing."""
    n_prev = state_prev.n_contacts
    lost_set = set(int(i) for i in lost_prev_idx)
    keep = [i for i in range(n_prev) if i not in lost_set]
    if len(keep) == n_prev:
        return state_prev, P_prev, Q_prev

    xi_parts = [state_prev.xi[2 * i : 2 * i + 2] for i in keep]
    f_parts = [state_prev.f[i : i + 1] for i in keep]
    xi_new = np.concatenate(xi_parts, axis=0) if xi_parts else np.zeros((0,), dtype=float)
    f_new = np.concatenate(f_parts, axis=0) if f_parts else np.zeros((0,), dtype=float)
    state_new = InhandState(x=state_prev.x.copy(), xi=xi_new, f=f_new)

    keep_idx = [0, 1, 2, 3, 4, 5]
    for i in keep:
        keep_idx.append(6 + 2 * i)
        keep_idx.append(6 + 2 * i + 1)
    keep_idx += [6 + 2 * n_prev + i for i in keep]
    keep_idx = np.asarray(keep_idx, dtype=np.int32)
    P_new = np.asarray(P_prev)[np.ix_(keep_idx, keep_idx)]
    Q_new = np.asarray(Q_prev)[np.ix_(keep_idx, keep_idx)]
    return state_new, P_new, Q_new


def expand_state_cov_for_new_contacts(
    state_prev: InhandState,
    P_prev: np.ndarray,
    Q_prev: np.ndarray,
    new_count: int,
    xi_init: np.ndarray | None = None,
    f_init: np.ndarray | None = None,
    xi_init_var: float = 1e-4,
    f_init_var: float = 1e-3,
    q_new_var: float = 1e-4,
) -> tuple[InhandState, np.ndarray, np.ndarray]:
    """Append new contacts into y/P/Q with diagonal initialization."""
    k = int(new_count)
    if k <= 0:
        return state_prev, P_prev, Q_prev

    xi_add = (
        np.zeros((2 * k,), dtype=float)
        if xi_init is None
        else np.asarray(xi_init, dtype=float).reshape(-1)
    )
    f_add = (
        np.zeros((k,), dtype=float)
        if f_init is None
        else np.asarray(f_init, dtype=float).reshape(-1)
    )
    if xi_add.shape[0] != 2 * k:
        raise ValueError(f"xi_init must have shape (2*new_count,), got {xi_add.shape}")
    if f_add.shape[0] != k:
        raise ValueError(f"f_init must have shape (new_count,), got {f_add.shape}")

    state_new = InhandState(
        x=state_prev.x.copy(),
        xi=np.concatenate([state_prev.xi, xi_add], axis=0),
        f=np.concatenate([state_prev.f, f_add], axis=0),
    )

    old_dim = state_prev.dim
    new_dim = state_new.dim
    P_new = np.zeros((new_dim, new_dim), dtype=float)
    Q_new = np.zeros((new_dim, new_dim), dtype=float)
    P_new[:old_dim, :old_dim] = np.asarray(P_prev, dtype=float)
    Q_new[:old_dim, :old_dim] = np.asarray(Q_prev, dtype=float)

    xi_start = 6 + 2 * state_prev.n_contacts
    f_start = 6 + 2 * state_new.n_contacts + state_prev.n_contacts
    for i in range(2 * k):
        P_new[xi_start + i, xi_start + i] = float(xi_init_var)
        Q_new[xi_start + i, xi_start + i] = float(q_new_var)
    for i in range(k):
        P_new[f_start + i, f_start + i] = float(f_init_var)
        Q_new[f_start + i, f_start + i] = float(q_new_var)
    return state_new, P_new, Q_new


def audit_state_cov_contact_alignment(
    state_prev: InhandState,
    P_prev: np.ndarray,
    Q_prev: np.ndarray,
    state_after_shrink: InhandState,
    P_after_shrink: np.ndarray,
    Q_after_shrink: np.ndarray,
    state_after_expand: InhandState,
    P_after_expand: np.ndarray,
    Q_after_expand: np.ndarray,
    match_events: ContactMatchEvents,
    xi_new_init: np.ndarray,
    f_new_init: np.ndarray,
) -> None:
    """
    Audit Eq.(24)(25) index consistency across:
      - state y packing ([x, xi, f])
      - covariance blocks P/Q
      - contact add/remove events
    Raises RuntimeError on any mismatch.
    """
    n_prev = state_prev.n_contacts
    if match_events.prev_count != n_prev:
        raise RuntimeError(
            f"audit failed: events.prev_count={match_events.prev_count} != state_prev.n_contacts={n_prev}"
        )
    if P_prev.shape != (state_prev.dim, state_prev.dim):
        raise RuntimeError(f"audit failed: P_prev shape {P_prev.shape} != {(state_prev.dim, state_prev.dim)}")
    if Q_prev.shape != (state_prev.dim, state_prev.dim):
        raise RuntimeError(f"audit failed: Q_prev shape {Q_prev.shape} != {(state_prev.dim, state_prev.dim)}")

    lost_set = set(int(i) for i in match_events.lost_prev_idx)
    keep = [i for i in range(n_prev) if i not in lost_set]
    n_keep = len(keep)
    if state_after_shrink.n_contacts != n_keep:
        raise RuntimeError(
            f"audit failed after shrink: n_contacts={state_after_shrink.n_contacts} expected={n_keep}"
        )
    if not np.allclose(state_after_shrink.x, state_prev.x):
        raise RuntimeError("audit failed after shrink: x block changed unexpectedly")

    xi_expect = (
        np.concatenate([state_prev.xi[2 * i : 2 * i + 2] for i in keep], axis=0)
        if n_keep > 0
        else np.zeros((0,), dtype=float)
    )
    f_expect = (
        np.concatenate([state_prev.f[i : i + 1] for i in keep], axis=0)
        if n_keep > 0
        else np.zeros((0,), dtype=float)
    )
    if not np.allclose(state_after_shrink.xi, xi_expect):
        raise RuntimeError("audit failed after shrink: xi order/content mismatch")
    if not np.allclose(state_after_shrink.f, f_expect):
        raise RuntimeError("audit failed after shrink: f order/content mismatch")
    if P_after_shrink.shape != (state_after_shrink.dim, state_after_shrink.dim):
        raise RuntimeError("audit failed after shrink: P shape mismatch")
    if Q_after_shrink.shape != (state_after_shrink.dim, state_after_shrink.dim):
        raise RuntimeError("audit failed after shrink: Q shape mismatch")

    k_new = len(match_events.new_curr_idx)
    if state_after_expand.n_contacts != n_keep + k_new:
        raise RuntimeError(
            f"audit failed after expand: n_contacts={state_after_expand.n_contacts} expected={n_keep + k_new}"
        )
    if not np.allclose(state_after_expand.x, state_after_shrink.x):
        raise RuntimeError("audit failed after expand: x block changed unexpectedly")

    xi_new_init = np.asarray(xi_new_init, dtype=float).reshape(-1)
    f_new_init = np.asarray(f_new_init, dtype=float).reshape(-1)
    if xi_new_init.shape[0] != 2 * k_new:
        raise RuntimeError(
            f"audit failed after expand: xi_new_init shape {xi_new_init.shape} != {(2 * k_new,)}"
        )
    if f_new_init.shape[0] != k_new:
        raise RuntimeError(
            f"audit failed after expand: f_new_init shape {f_new_init.shape} != {(k_new,)}"
        )

    xi_expand_expect = np.concatenate([state_after_shrink.xi, xi_new_init], axis=0)
    f_expand_expect = np.concatenate([state_after_shrink.f, f_new_init], axis=0)
    if not np.allclose(state_after_expand.xi, xi_expand_expect):
        raise RuntimeError("audit failed after expand: xi append/order mismatch")
    if not np.allclose(state_after_expand.f, f_expand_expect):
        raise RuntimeError("audit failed after expand: f append/order mismatch")
    if P_after_expand.shape != (state_after_expand.dim, state_after_expand.dim):
        raise RuntimeError("audit failed after expand: P shape mismatch")
    if Q_after_expand.shape != (state_after_expand.dim, state_after_expand.dim):
        raise RuntimeError("audit failed after expand: Q shape mismatch")
