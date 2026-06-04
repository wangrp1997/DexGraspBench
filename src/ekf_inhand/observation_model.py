from typing import Callable

import numpy as np

from .state import InhandState


def _project_uv_to_triangle_domain(xi: np.ndarray) -> np.ndarray:
    """
    Project barycentric uv parameters to the valid triangle domain:
      v >= 0, w >= 0, v + w <= 1
    """
    x = np.asarray(xi, dtype=float).reshape(-1, 2).copy()
    x[:, 0] = np.maximum(x[:, 0], 0.0)
    x[:, 1] = np.maximum(x[:, 1], 0.0)
    s = x[:, 0] + x[:, 1]
    over = s > 1.0
    if np.any(over):
        # Radial projection onto the simplex edge v + w = 1.
        x[over, 0] /= s[over]
        x[over, 1] /= s[over]
    return x


def h_tau_eq12_13(
    state: InhandState,
    J: np.ndarray,
    contact_normals_obj: np.ndarray,
) -> np.ndarray:
    """
    Paper Eq.(12)(13):
      h_tau(y_t) = J^T * lambda_t
      lambda_i = n_i * f_i

    Args:
      state: y = [x, xi, f]
      J: contact Jacobian in joint-space mapping, shape (3n, m)
      contact_normals_obj: object-surface normals n_i, shape (n, 3)
    Returns:
      tau_hat, shape (m,)
    """
    n = state.n_contacts
    J = np.asarray(J, dtype=float)
    normals = np.asarray(contact_normals_obj, dtype=float)

    if n == 0:
        if J.ndim != 2:
            raise ValueError(f"J must be 2D, got {J.shape}")
        return np.zeros((J.shape[1],), dtype=float)

    if normals.shape != (n, 3):
        raise ValueError(
            f"contact_normals_obj must have shape {(n, 3)}, got {normals.shape}"
        )
    if J.shape[0] != 3 * n:
        raise ValueError(
            f"J first dim must be 3n={3*n} for n={n} contacts, got {J.shape[0]}"
        )

    # lambda stacked as [n1*f1, n2*f2, ..., nn*fn] in R^(3n)
    lam = (normals * state.f.reshape(n, 1)).reshape(-1)
    tau_hat = J.T @ lam
    return tau_hat.reshape(-1)


def h_q_eq8_11(
    state: InhandState,
    q_prev: np.ndarray,
    hq_impl: Callable[[InhandState, np.ndarray], np.ndarray] | None = None,
    J_pinv: np.ndarray | None = None,
    c_obj: np.ndarray | None = None,
    c_f_prev: np.ndarray | None = None,
) -> np.ndarray:
    """
    Paper Eq.(8)-(11) wrapper.

    Priority:
    1) If `hq_impl` is provided, use external implementation.
    2) Else, run Eq.(8) directly if required inputs are provided:
         h_q(y_t) = h_q(y_{t-1}) + J^+ (c_o,t - c_f,t-1)
       where:
         - J_pinv is J^+
         - c_obj is stacked c_o,t
         - c_f_prev is stacked c_f,t-1

    Eq.(9)-(11) are geometry update equations for c_o/c_f/xi_f and are
    expected to be handled by upstream geometry callbacks before calling here.
    """
    q_prev = np.asarray(q_prev, dtype=float).reshape(-1)
    if hq_impl is not None:
        q_hat = np.asarray(hq_impl(state, q_prev), dtype=float).reshape(-1)
        return q_hat

    if J_pinv is None or c_obj is None or c_f_prev is None:
        raise NotImplementedError(
            "Eq.(8) requires either hq_impl or (J_pinv, c_obj, c_f_prev)."
        )

    J_pinv = np.asarray(J_pinv, dtype=float)
    c_obj = np.asarray(c_obj, dtype=float)
    c_f_prev = np.asarray(c_f_prev, dtype=float)

    if c_obj.shape != c_f_prev.shape:
        raise ValueError(
            f"c_obj and c_f_prev must have same shape, got {c_obj.shape} vs {c_f_prev.shape}"
        )
    if c_obj.ndim != 2 or c_obj.shape[1] != 3:
        raise ValueError(
            f"contact point arrays must have shape (n,3), got {c_obj.shape}"
        )

    delta_c = (c_obj - c_f_prev).reshape(-1)  # (3n,)
    if J_pinv.shape[1] != delta_c.shape[0]:
        raise ValueError(
            f"J_pinv second dim must be 3n={delta_c.shape[0]}, got {J_pinv.shape}"
        )
    if J_pinv.shape[0] != q_prev.shape[0]:
        raise ValueError(
            f"J_pinv first dim must match q_prev dim {q_prev.shape[0]}, got {J_pinv.shape}"
        )

    return q_prev + J_pinv @ delta_c


def xi_f_update_eq11(
    xi_f_prev: np.ndarray,
    n_obj_t: np.ndarray,
    n_f_prev: np.ndarray,
    dn_f_dxi_at_prev: np.ndarray | None = None,
    normal_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Paper Eq.(11):
      xi_f,t = xi_f,t-1 + (∂n_f/∂xi_f)|_{xi_f,t-1} * ( (-n_o,t) - n_f,t-1 )

    Shapes:
      xi_f_prev: (n,2)
      n_obj_t: (n,3)
      n_f_prev: (n,3)
      dn_f_dxi_at_prev: (n,2,3), optional
      normal_fn: optional normal field callback for numerical derivative.
                 Signature: normal_fn(xi) -> normals with shape (n,3)
      eps: finite difference step for numerical derivative
    Returns:
      xi_f_t: (n,2)
    """
    xi_f_prev = np.asarray(xi_f_prev, dtype=float)
    n_obj_t = np.asarray(n_obj_t, dtype=float)
    n_f_prev = np.asarray(n_f_prev, dtype=float)
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    if xi_f_prev.ndim != 2 or xi_f_prev.shape[1] != 2:
        raise ValueError(f"xi_f_prev must have shape (n,2), got {xi_f_prev.shape}")
    n = xi_f_prev.shape[0]
    if n_obj_t.shape != (n, 3):
        raise ValueError(f"n_obj_t must have shape {(n, 3)}, got {n_obj_t.shape}")
    if n_f_prev.shape != (n, 3):
        raise ValueError(f"n_f_prev must have shape {(n, 3)}, got {n_f_prev.shape}")
    if dn_f_dxi_at_prev is None:
        if normal_fn is None:
            raise ValueError(
                "Provide either dn_f_dxi_at_prev or normal_fn for numerical approximation."
            )
        dn_f_dxi_at_prev = approx_dn_f_dxi_numeric(
            xi_f_prev=xi_f_prev,
            normal_fn=normal_fn,
            eps=eps,
        )
    else:
        dn_f_dxi_at_prev = np.asarray(dn_f_dxi_at_prev, dtype=float)
        if dn_f_dxi_at_prev.shape != (n, 2, 3):
            raise ValueError(
                f"dn_f_dxi_at_prev must have shape {(n, 2, 3)}, got {dn_f_dxi_at_prev.shape}"
            )

    normal_err = (-n_obj_t) - n_f_prev  # (n,3)
    delta_xi = np.einsum("nij,nj->ni", dn_f_dxi_at_prev, normal_err)  # (n,2)
    xi_next = xi_f_prev + delta_xi
    # Keep xi_f on the valid triangle parameter domain.
    return _project_uv_to_triangle_domain(xi_next)


def approx_dn_f_dxi_numeric(
    xi_f_prev: np.ndarray,
    normal_fn: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Numerical approximation of ∂n_f/∂xi_f with central finite differences.

    Args:
      xi_f_prev: (n,2)
      normal_fn: maps xi (n,2) -> normals (n,3)
      eps: finite difference step
    Returns:
      dn_f_dxi: (n,2,3)
    """
    xi_f_prev = np.asarray(xi_f_prev, dtype=float)
    if xi_f_prev.ndim != 2 or xi_f_prev.shape[1] != 2:
        raise ValueError(f"xi_f_prev must have shape (n,2), got {xi_f_prev.shape}")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    n = xi_f_prev.shape[0]
    # Fixed stability range to avoid too small/too large finite-diff steps.
    eps_eff = float(np.clip(float(eps), 1e-7, 1e-4))
    xi_center = _project_uv_to_triangle_domain(xi_f_prev)
    dn = np.zeros((n, 2, 3), dtype=float)
    for k in range(2):
        xi_plus = xi_center.copy()
        xi_minus = xi_center.copy()
        xi_plus[:, k] += eps_eff
        xi_minus[:, k] -= eps_eff
        xi_plus = _project_uv_to_triangle_domain(xi_plus)
        xi_minus = _project_uv_to_triangle_domain(xi_minus)
        n_plus = np.asarray(normal_fn(xi_plus), dtype=float)
        n_minus = np.asarray(normal_fn(xi_minus), dtype=float)
        if n_plus.shape != (n, 3):
            raise ValueError(f"normal_fn(xi_plus) must return {(n,3)}, got {n_plus.shape}")
        if n_minus.shape != (n, 3):
            raise ValueError(f"normal_fn(xi_minus) must return {(n,3)}, got {n_minus.shape}")
        step = (xi_plus[:, k] - xi_minus[:, k]).reshape(-1, 1)
        # Boundary-safe derivative: fallback to nominal denominator when step is tiny.
        tiny = np.abs(step[:, 0]) < 1e-12
        if np.any(tiny):
            step[tiny, 0] = 2.0 * eps_eff
        dn[:, k, :] = (n_plus - n_minus) / step
    return dn


def h_full_eq7(
    state: InhandState,
    q_prev: np.ndarray,
    J: np.ndarray,
    contact_normals_obj: np.ndarray,
    hq_impl: Callable[[InhandState, np.ndarray], np.ndarray] | None = None,
    J_pinv: np.ndarray | None = None,
    c_obj: np.ndarray | None = None,
    c_f_prev: np.ndarray | None = None,
) -> np.ndarray:
    """
    Paper Eq.(7): h(y_t) = [h_q(y_t); h_tau(y_t)].
    """
    q_hat = h_q_eq8_11(
        state=state,
        q_prev=q_prev,
        hq_impl=hq_impl,
        J_pinv=J_pinv,
        c_obj=c_obj,
        c_f_prev=c_f_prev,
    )
    tau_hat = h_tau_eq12_13(state=state, J=J, contact_normals_obj=contact_normals_obj)
    return np.concatenate([q_hat, tau_hat], axis=0)


def H_tau_eq19_analytic(
    n_contacts: int,
    dim_y: int,
    J_obs: np.ndarray,
    contact_normals_obj: np.ndarray,
    m_tau: int,
) -> np.ndarray:
    """
    Analytic ∂h_tau/∂y for Eq.(12)(13) when J and object normals are held fixed.

    Only force columns ∂h_tau/∂f_i are nonzero:
      h_tau = J^T [n_1 f_1, ..., n_n f_n]^T  =>  ∂h_tau/∂f_i = J^T[:, 3i:3i+3] n_i
    """
    n = int(n_contacts)
    J = np.asarray(J_obs, dtype=float)
    normals = np.asarray(contact_normals_obj, dtype=float)
    H_tau = np.zeros((m_tau, dim_y), dtype=float)
    if n == 0:
        return H_tau
    if normals.shape != (n, 3):
        raise ValueError(
            f"contact_normals_obj must have shape {(n, 3)}, got {normals.shape}"
        )
    if J.shape[0] != 3 * n:
        raise ValueError(
            f"J first dim must be 3n={3*n} for n={n} contacts, got {J.shape[0]}"
        )
    if J.shape[1] != m_tau:
        raise ValueError(
            f"J second dim must be m_tau={m_tau}, got {J.shape[1]}"
        )
    for i in range(n):
        col_f = 6 + 2 * n + i
        H_tau[:, col_f] = J.T[:, 3 * i : 3 * i + 3] @ normals[i]
    return H_tau


def H_full_eq19_hybrid(
    y_vec: np.ndarray,
    h_func: Callable[[np.ndarray], np.ndarray],
    J_obs: np.ndarray,
    contact_normals_obj: np.ndarray,
    m_q: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Hybrid Eq.(19) Jacobian: numeric for h_q rows, analytic for h_tau rows (w.r.t. f).
    """
    y = np.asarray(y_vec, dtype=float).reshape(-1)
    state = InhandState.unpack(y)
    H = H_full_eq19_numeric(y_vec=y, h_func=h_func, eps=eps)
    h0 = np.asarray(h_func(y), dtype=float).reshape(-1)
    m_tau = h0.shape[0] - int(m_q)
    if m_tau > 0 and state.n_contacts > 0:
        H_tau = H_tau_eq19_analytic(
            n_contacts=state.n_contacts,
            dim_y=y.shape[0],
            J_obs=J_obs,
            contact_normals_obj=contact_normals_obj,
            m_tau=m_tau,
        )
        H[int(m_q) :, :] = H_tau
    return H


def H_full_eq19_numeric(
    y_vec: np.ndarray,
    h_func: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Paper Eq.(19): H_t = ∂h/∂y, numerical Jacobian (finite difference).
    """
    y = np.asarray(y_vec, dtype=float).reshape(-1)
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    h0 = np.asarray(h_func(y), dtype=float).reshape(-1)
    dim_z = h0.shape[0]
    dim_x = y.shape[0]
    H = np.zeros((dim_z, dim_x), dtype=float)

    for i in range(dim_x):
        y_perturb = y.copy()
        y_perturb[i] += eps
        h_perturb = np.asarray(h_func(y_perturb), dtype=float).reshape(-1)
        if h_perturb.shape[0] != dim_z:
            raise ValueError(
                f"h_func output dim changed under perturbation: {dim_z} -> {h_perturb.shape[0]}"
            )
        H[:, i] = (h_perturb - h0) / eps
    return H

