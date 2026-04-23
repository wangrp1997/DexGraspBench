from typing import Callable

import numpy as np

from .state import InhandState


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
    return xi_f_prev + delta_xi


def approx_dn_f_dxi_numeric(
    xi_f_prev: np.ndarray,
    normal_fn: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Numerical approximation of ∂n_f/∂xi_f with forward finite differences.

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
    n0 = np.asarray(normal_fn(xi_f_prev), dtype=float)
    if n0.shape != (n, 3):
        raise ValueError(f"normal_fn(xi_f_prev) must return {(n,3)}, got {n0.shape}")

    dn = np.zeros((n, 2, 3), dtype=float)
    for k in range(2):
        xi_eps = xi_f_prev.copy()
        xi_eps[:, k] += eps
        nk = np.asarray(normal_fn(xi_eps), dtype=float)
        if nk.shape != (n, 3):
            raise ValueError(f"normal_fn(xi_eps) must return {(n,3)}, got {nk.shape}")
        dn[:, k, :] = (nk - n0) / eps
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

