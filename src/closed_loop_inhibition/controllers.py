"""Classical control baselines and local linear stability calculations."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .plants import Oscillator, _finite_scalar
from .records import PolicyInput


@dataclass(frozen=True)
class ClassicalPDController:
    """PD plus static reference feedforward, using the latest available sample.

    This policy does not access live state, compensate delay, or clip actions.
    The runner applies actuator limits equally to all policies.
    """

    kp: float = 2.0
    kd: float = 1.0
    tau: float = 0.5

    def __post_init__(self) -> None:
        for name in ("kp", "kd", "tau"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if self.tau <= 0:
            raise ValueError("tau must be positive")

    def __call__(self, snapshot: PolicyInput) -> float:
        obs = snapshot.latest
        return float(
            obs.reference + self.kp * (obs.reference - obs.position)
            + self.kd * self.tau * (obs.reference_velocity - obs.velocity)
        )


@dataclass(frozen=True)
class PredictorPDController:
    """PD on a causal model prediction at the scheduled action-application time.

    Reconstruct from the latest captured state and the already-applied command
    history. Hold the last known command from inference start to application;
    assume zero unknown disturbance and locally constant reference velocity.
    No future state, disturbance, reference, or pending command is consulted.

    This prediction is exact only when those assumptions hold. In particular,
    overlapping jobs or delayed actuators can apply other pending commands
    while this job runs; those commands are unavailable through PolicyInput.
    """

    plant: Oscillator = Oscillator()
    kp: float = 2.0
    kd: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.plant, Oscillator):
            raise ValueError("plant must be an Oscillator")
        for name in ("kp", "kd"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))

    def __call__(self, snapshot: PolicyInput) -> float:
        obs = snapshot.latest
        start = _finite_scalar(snapshot.time, "snapshot.time")
        target = _finite_scalar(snapshot.planned_apply_time, "planned_apply_time")
        at = _finite_scalar(obs.capture_time, "capture_time")
        if target < start or at > start or obs.available_time > start:
            raise ValueError("Prediction requires a causally available observation and future application")
        state = np.array([obs.position, obs.velocity])
        held = obs.applied_action
        previous_action_time = -np.inf
        for action in snapshot.actions:
            action_time = _finite_scalar(action.time, "action.time")
            if action_time < previous_action_time or action_time > start:
                raise ValueError("Applied actions must be ordered and available at inference start")
            previous_action_time = action_time
            if action_time < obs.capture_time:
                continue
            state = self.plant.propagate(state, action_time - at, held)
            at, held = action_time, action.value
        state = self.plant.propagate(state, target - at, held)
        reference = obs.reference + obs.reference_velocity * (target - obs.capture_time)
        return float(
            reference + self.kp * (reference - state[0])
            + self.kd * self.plant.tau * (obs.reference_velocity - state[1])
        )


def continuous_closed_loop_poles(
    plant: Oscillator, kp: float = 2.0, kd: float = 1.0
) -> NDArray:
    """Poles of continuous, unsaturated, zero-delay PD around fixed reference."""
    kp, kd = _finite_scalar(kp, "kp"), _finite_scalar(kd, "kd")
    return np.linalg.eigvals(plant.A - np.outer(plant.B, [kp, kd * plant.tau]))


def sampled_closed_loop_matrix(
    plant: Oscillator, kp: float, kd: float, period: float, delay_steps: int = 0
) -> NDArray[np.float64]:
    """Exact fixed-cadence, held-action map with integer-step observation delay.

    For ``u_k = -K x_(k-m)``, the augmented state is
    ``[x_k, x_(k-1), ..., x_(k-m)]``. The first block evolves as
    ``x_(k+1) = Ad x_k - Bd K x_(k-m)`` and lower blocks shift history.
    This assumes simultaneous sampling/application, no saturation, and a fixed
    reference. It does not describe arbitrary event-driven serial schedules.
    """
    period = _finite_scalar(period, "period")
    if period <= 0:
        raise ValueError("period must be positive")
    if isinstance(delay_steps, bool) or not isinstance(delay_steps, Integral) or delay_steps < 0:
        raise ValueError("delay_steps must be a nonnegative integer")
    kp, kd = _finite_scalar(kp, "kp"), _finite_scalar(kd, "kd")
    ad, bd = plant.discretize(period)
    width = 2 * (delay_steps + 1)
    transition = np.zeros((width, width))
    transition[:2, :2] = ad
    transition[:2, -2:] -= np.outer(bd, [kp, kd * plant.tau])
    if delay_steps:
        transition[2:, :-2] = np.eye(width - 2)
    return transition
