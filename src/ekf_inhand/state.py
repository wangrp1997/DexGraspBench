from dataclasses import dataclass

import numpy as np


@dataclass
class InhandState:
    """Paper state y = [x(6), xi(2n), f(n)]."""

    x: np.ndarray
    xi: np.ndarray
    f: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).reshape(-1)
        self.xi = np.asarray(self.xi, dtype=float).reshape(-1)
        self.f = np.asarray(self.f, dtype=float).reshape(-1)

        if self.x.shape[0] != 6:
            raise ValueError(f"x dimension must be 6, got {self.x.shape[0]}")
        if self.xi.shape[0] % 2 != 0:
            raise ValueError(f"xi dimension must be 2n, got {self.xi.shape[0]}")
        if self.xi.shape[0] // 2 != self.f.shape[0]:
            raise ValueError(
                f"xi/f mismatch: xi has {self.xi.shape[0] // 2} contacts but f has {self.f.shape[0]}"
            )

    @property
    def n_contacts(self) -> int:
        return int(self.f.shape[0])

    @property
    def dim(self) -> int:
        return int(6 + 3 * self.n_contacts)

    def pack(self) -> np.ndarray:
        return np.concatenate([self.x, self.xi, self.f], axis=0)

    @staticmethod
    def unpack(y_vec: np.ndarray) -> "InhandState":
        y = np.asarray(y_vec, dtype=float).reshape(-1)
        if y.shape[0] < 6:
            raise ValueError(f"y dimension must be >= 6, got {y.shape[0]}")
        rest = y.shape[0] - 6
        if rest % 3 != 0:
            raise ValueError(f"y dimension must satisfy 6+3n, got {y.shape[0]}")

        n = rest // 3
        x = y[:6]
        xi = y[6 : 6 + 2 * n]
        f = y[6 + 2 * n :]
        return InhandState(x=x, xi=xi, f=f)

