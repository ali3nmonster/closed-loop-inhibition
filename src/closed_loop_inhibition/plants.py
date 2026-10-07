"""Continuous plant dynamics with exact propagation under held inputs."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm


def _finite_scalar(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return value


@lru_cache(maxsize=512)
def _discretize(tau: float, zeta: float, dt: float) -> tuple[NDArray, NDArray]:
    """Cache private matrices; callers receive copies through ``discretize``."""
    inverse_tau_squared = (1.0 / tau) ** 2
    augmented = np.array(
        [[0.0, 1.0, 0.0], [-inverse_tau_squared, -2.0 * (zeta / tau), inverse_tau_squared],
         [0.0, 0.0, 0.0]]
    )
    transition = expm(augmented * dt)
    if not np.all(np.isfinite(transition)):
        raise ValueError("Parameters and dt produced a nonfinite transition")
    return transition[:2, :2], transition[:2, 2]


@dataclass(frozen=True)
class Oscillator:
    """Normalized mass–spring–damper with position in state component zero.

    ``q_dot = v`` and ``v_dot = (-q - 2*zeta*tau*v + u + d) / tau**2``.
    Time is measured in seconds; control and disturbance have the same units
    as position. Changing ``tau`` preserves the static input-to-position gain.
    """

    tau: float = 0.5
    zeta: float = 0.15

    def __post_init__(self) -> None:
        tau = _finite_scalar(self.tau, "tau")
        zeta = _finite_scalar(self.zeta, "zeta")
        if tau <= 0:
            raise ValueError("tau must be positive")
        if zeta < 0:
            raise ValueError("zeta must be nonnegative")
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            coefficients = np.array([np.float64(tau) ** -2, 2 * (np.float64(zeta) / tau)])
        if not np.all(np.isfinite(coefficients)) or coefficients[0] == 0:
            raise ValueError("tau and zeta must yield finite, representable dynamics")
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "zeta", zeta)

    @property
    def A(self) -> NDArray[np.float64]:
        return np.array([[0.0, 1.0], [-(1.0 / self.tau) ** 2, -2.0 * (self.zeta / self.tau)]])

    @property
    def B(self) -> NDArray[np.float64]:
        return np.array([0.0, (1.0 / self.tau) ** 2])

    def discretize(self, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return exact ``Ad, Bd`` for constant input over nonnegative ``dt``.

        An augmented exponential avoids an inverse of ``A``. Returned matrices
        can safely be modified; the cached transition remains private.
        """
        dt = _finite_scalar(dt, "dt")
        if dt < 0:
            raise ValueError("dt must be nonnegative")
        if dt == 0:
            return np.eye(2), np.zeros(2)
        ad, bd = _discretize(self.tau, self.zeta, dt)
        return ad.copy(), bd.copy()

    def propagate(
        self, state: ArrayLike, dt: float, control: float = 0.0, disturbance: float = 0.0
    ) -> NDArray[np.float64]:
        """Advance without mutating ``state``; both inputs are held over ``dt``."""
        state = np.asarray(state, dtype=float)
        if state.shape != (2,) or not np.all(np.isfinite(state)):
            raise ValueError("state must be a finite array with shape (2,)")
        control = _finite_scalar(control, "control")
        disturbance = _finite_scalar(disturbance, "disturbance")
        combined_input = control + disturbance
        if not np.isfinite(combined_input):
            raise ValueError("control plus disturbance must be finite")
        ad, bd = self.discretize(dt)
        return ad @ state + bd * combined_input
