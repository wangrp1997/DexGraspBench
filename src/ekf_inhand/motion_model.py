import numpy as np

from .state import InhandState


def predict_state_eq6(
    state_prev: InhandState,
    u_t: np.ndarray,
    dt: float,
    G_pinv: np.ndarray,
    J: np.ndarray,
) -> InhandState:
    """
    Paper Eq.(6): y_t = y_{t-1} + [G^+ J u_t * dt; 0_{3n}]

    State convention:
      y = [x(6), xi(2n), f(n)]  -> total dim = 6 + 3n

    Notes:
    - This function implements the exact additive structure of Eq.(6).
    - Contact substate [xi, f] remains unchanged in prediction.
    """
    u_t = np.asarray(u_t, dtype=float).reshape(-1)
    G_pinv = np.asarray(G_pinv, dtype=float)
    J = np.asarray(J, dtype=float)
    dt = float(dt)

    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if G_pinv.shape[0] != 6:
        raise ValueError(f"G_pinv first dim must be 6, got {G_pinv.shape}")
    if J.ndim != 2:
        raise ValueError(f"J must be 2D, got shape {J.shape}")
    if J.shape[1] != u_t.shape[0]:
        raise ValueError(
            f"J/u mismatch: J shape {J.shape}, u_t dim {u_t.shape[0]}"
        )
    if G_pinv.shape[1] != J.shape[0]:
        raise ValueError(
            f"G_pinv/J mismatch: G_pinv shape {G_pinv.shape}, J shape {J.shape}"
        )

    delta_x = (G_pinv @ J @ u_t) * dt
    if delta_x.shape[0] != 6:
        raise ValueError(f"delta_x dim must be 6, got {delta_x.shape[0]}")

    x_pred = state_prev.x + delta_x
    return InhandState(x=x_pred, xi=state_prev.xi.copy(), f=state_prev.f.copy())


def jacobian_F_eq16(
    state_prev: InhandState,
    u_t: np.ndarray,
    dt: float,
    G_pinv: np.ndarray,
    J: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Paper Eq.(16): F_t = ∂f/∂y.

    Numerical Jacobian (finite difference) around current state.
    This follows the chosen implementation strategy:
    - first run with numerical Jacobians to keep the pipeline correct,
    - then replace with analytic/autodiff Jacobians if needed.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")

    y0 = state_prev.pack()
    dim = y0.shape[0]
    F = np.zeros((dim, dim), dtype=float)

    def _predict_from_vec(y_vec: np.ndarray) -> np.ndarray:
        s = InhandState.unpack(y_vec)
        s_pred = predict_state_eq6(
            state_prev=s,
            u_t=u_t,
            dt=dt,
            G_pinv=G_pinv,
            J=J,
        )
        return s_pred.pack()

    f0 = _predict_from_vec(y0)
    for i in range(dim):
        y_perturb = y0.copy()
        y_perturb[i] += eps
        f_perturb = _predict_from_vec(y_perturb)
        F[:, i] = (f_perturb - f0) / eps
    return F


def predict_covariance_eq15(
    P_prev: np.ndarray,
    F_t: np.ndarray,
    Q_t: np.ndarray,
) -> np.ndarray:
    """
    Paper Eq.(15): P_t = F_{t-1} P_{t-1} F_{t-1}^T + Q_t
    """
    P_prev = np.asarray(P_prev, dtype=float)
    F_t = np.asarray(F_t, dtype=float)
    Q_t = np.asarray(Q_t, dtype=float)

    if P_prev.ndim != 2 or P_prev.shape[0] != P_prev.shape[1]:
        raise ValueError(f"P_prev must be square, got shape {P_prev.shape}")
    dim = P_prev.shape[0]
    if F_t.shape != (dim, dim):
        raise ValueError(f"F_t shape must be {(dim, dim)}, got {F_t.shape}")
    if Q_t.shape != (dim, dim):
        raise ValueError(f"Q_t shape must be {(dim, dim)}, got {Q_t.shape}")

    return F_t @ P_prev @ F_t.T + Q_t

