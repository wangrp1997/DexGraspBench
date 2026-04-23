from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ekf_core import EkfStepResult
from .ekf_core import ekf_step_filterpy
from .motion_model import jacobian_F_eq16
from .motion_model import predict_covariance_eq15
from .motion_model import predict_state_eq6
from .observation_model import H_full_eq19_numeric
from .observation_model import h_full_eq7
from .state import InhandState


@dataclass
class RunnerStepResult:
    state_pred: InhandState
    P_pred: np.ndarray
    ekf_update: EkfStepResult


def run_one_step_smoke(
    state_prev: InhandState,
    P_prev: np.ndarray,
    u_t: np.ndarray,
    z_t: np.ndarray,
    dt: float,
    G_pinv: np.ndarray,
    J_motion: np.ndarray,
    Q_t: np.ndarray,
    R_t: np.ndarray,
    q_prev: np.ndarray,
    J_obs: np.ndarray,
    contact_normals_obj: np.ndarray,
    hq_impl: Callable[[InhandState, np.ndarray], np.ndarray] | None,
    J_pinv: np.ndarray | None = None,
    c_obj: np.ndarray | None = None,
    c_f_prev: np.ndarray | None = None,
    jac_eps: float = 1e-6,
) -> RunnerStepResult:
    """
    Minimal runner that chains:
    1) motion prediction (Eq.6)
    2) covariance prediction (Eq.15)
    3) EKF update through FilterPy core

    This is a smoke-level integration function for interface validation.
    """
    # 1) state prediction
    state_pred = predict_state_eq6(
        state_prev=state_prev,
        u_t=u_t,
        dt=dt,
        G_pinv=G_pinv,
        J=J_motion,
    )

    # 2) covariance prediction
    F_t = jacobian_F_eq16(
        state_prev=state_prev,
        u_t=u_t,
        dt=dt,
        G_pinv=G_pinv,
        J=J_motion,
        eps=jac_eps,
    )
    P_pred = predict_covariance_eq15(P_prev=P_prev, F_t=F_t, Q_t=Q_t)

    # 3) measurement model wrappers on vector state
    def _h_vec(y_vec: np.ndarray) -> np.ndarray:
        s = InhandState.unpack(y_vec)
        return h_full_eq7(
            state=s,
            q_prev=q_prev,
            J=J_obs,
            contact_normals_obj=contact_normals_obj,
            hq_impl=hq_impl,
            J_pinv=J_pinv,
            c_obj=c_obj,
            c_f_prev=c_f_prev,
        )

    def _H_vec(y_vec: np.ndarray) -> np.ndarray:
        return H_full_eq19_numeric(y_vec=y_vec, h_func=_h_vec, eps=jac_eps)

    step_res = ekf_step_filterpy(
        y=state_pred.pack(),
        P=P_pred,
        u=u_t,
        z=z_t,
        F=F_t,
        Q=Q_t,
        R=R_t,
        h_fn=_h_vec,
        H_fn=_H_vec,
    )

    return RunnerStepResult(
        state_pred=state_pred,
        P_pred=P_pred,
        ekf_update=step_res,
    )

