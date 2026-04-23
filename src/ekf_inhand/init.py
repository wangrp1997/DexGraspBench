from dataclasses import dataclass

import numpy as np

from .state import InhandState


@dataclass
class EkfInitConfig:
    """Initialization hyper-parameters for y0 and P0."""

    init_contacts: int = 0
    x0_std: float = 0.01
    xi0_std: float = 0.05
    f0_std: float = 0.5
    p0_x: float = 1e-2
    p0_xi: float = 1e-1
    p0_f: float = 1e-1
    seed: int = 0


def build_initial_state(
    x0_hint: np.ndarray,
    cfg: EkfInitConfig,
) -> InhandState:
    """
    Build initial EKF state y0 = [x0, xi0, f0].

    Notes:
    - `x0_hint` should come from an external rough initializer (not EKF GT feedback).
    - This function only creates a consistent starting point for the filter.
    """
    x0_hint = np.asarray(x0_hint, dtype=float).reshape(-1)
    if x0_hint.shape[0] != 6:
        raise ValueError(f"x0_hint dimension must be 6, got {x0_hint.shape[0]}")
    if cfg.init_contacts < 0:
        raise ValueError(f"init_contacts must be >= 0, got {cfg.init_contacts}")

    rng = np.random.default_rng(cfg.seed)
    x0 = x0_hint + rng.normal(loc=0.0, scale=cfg.x0_std, size=6)
    xi0 = (
        rng.normal(loc=0.0, scale=cfg.xi0_std, size=2 * cfg.init_contacts)
        if cfg.init_contacts > 0
        else np.zeros((0,), dtype=float)
    )
    f0 = (
        rng.normal(loc=0.0, scale=cfg.f0_std, size=cfg.init_contacts)
        if cfg.init_contacts > 0
        else np.zeros((0,), dtype=float)
    )

    return InhandState(x=x0, xi=xi0, f=f0)


def build_initial_covariance(state0: InhandState, cfg: EkfInitConfig) -> np.ndarray:
    """
    Build diagonal initial covariance P0 matching y0 dimension.
    """
    n = state0.n_contacts
    diag_x = np.full((6,), cfg.p0_x, dtype=float)
    diag_xi = np.full((2 * n,), cfg.p0_xi, dtype=float)
    diag_f = np.full((n,), cfg.p0_f, dtype=float)
    diag = np.concatenate([diag_x, diag_xi, diag_f], axis=0)
    return np.diag(diag)

