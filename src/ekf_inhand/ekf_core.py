from dataclasses import dataclass
from typing import Callable

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter


Array = np.ndarray


@dataclass
class EkfStepResult:
    y_next: Array
    P_next: Array
    innovation: Array
    K: Array


def ekf_step_filterpy(
    y: Array,
    P: Array,
    u: Array,
    z: Array,
    F: Array,
    Q: Array,
    R: Array,
    h_fn: Callable[[Array], Array],
    H_fn: Callable[[Array], Array],
) -> EkfStepResult:
    """
    One EKF predict+update step using FilterPy.

    Notes:
    - This is a thin engine wrapper only.
    - Motion/observation semantics come from paper-specific model modules.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float)
    F = np.asarray(F, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    dim_x = y.shape[0]
    dim_z = z.shape[0]
    dim_u = u.shape[0]

    if P.shape != (dim_x, dim_x):
        raise ValueError(f"P shape must be ({dim_x}, {dim_x}), got {P.shape}")
    if F.shape != (dim_x, dim_x):
        raise ValueError(f"F shape must be ({dim_x}, {dim_x}), got {F.shape}")
    if Q.shape != (dim_x, dim_x):
        raise ValueError(f"Q shape must be ({dim_x}, {dim_x}), got {Q.shape}")
    if R.shape != (dim_z, dim_z):
        raise ValueError(f"R shape must be ({dim_z}, {dim_z}), got {R.shape}")

    ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dim_u=dim_u)
    ekf.x = y.reshape(dim_x, 1)
    ekf.P = P.copy()
    ekf.F = F.copy()
    ekf.Q = Q.copy()
    ekf.R = R.copy()
    ekf.B = np.zeros((dim_x, dim_u), dtype=float)

    # IMPORTANT:
    # FilterPy predict_x computes:
    #   x <- F x + B u
    # where x is (dim_x,1). If u is passed as shape (dim_u,), numpy may
    # broadcast (dim_x,1) + (dim_x,) to (dim_x,dim_x), corrupting state shape.
    # Enforce column-vector u to keep x shape stable at (dim_x,1).
    u_col = u.reshape(dim_u, 1)

    # Predict with externally provided F/Q and zero-B control pathway.
    ekf.predict(u=u_col)

    def _hj(x_col: Array) -> Array:
        x = x_col.reshape(-1)
        H = np.asarray(H_fn(x), dtype=float)
        if H.shape != (dim_z, dim_x):
            raise ValueError(
                f"H_fn output shape must be ({dim_z}, {dim_x}), got {H.shape}"
            )
        return H

    def _hx(x_col: Array) -> Array:
        x = x_col.reshape(-1)
        z_hat = np.asarray(h_fn(x), dtype=float).reshape(-1)
        if z_hat.shape[0] != dim_z:
            raise ValueError(
                f"h_fn output dim must be {dim_z}, got {z_hat.shape[0]}"
            )
        return z_hat.reshape(dim_z, 1)

    ekf.update(z.reshape(dim_z, 1), HJacobian=_hj, Hx=_hx)

    return EkfStepResult(
        y_next=ekf.x.reshape(-1).copy(),
        P_next=ekf.P.copy(),
        innovation=ekf.y.reshape(-1).copy(),
        K=ekf.K.copy(),
    )

