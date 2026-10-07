"""Causal, bounded information shared by a teacher and learned controllers.

The encoder receives only immutable ``PolicyInput`` records. Neither training
targets nor features have access to future signals, live plant state, pending
commands, or the simulator's full action log. Feature scaling uses declared
physical units; it is not fitted to training or evaluation trajectories.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np

from .plants import Oscillator, _finite_scalar
from .records import AppliedAction, PolicyInput
from .signals import PiecewiseConstant
from .timing import TimingConfig, simulate


FEATURE_NAMES = (
    "position",
    "tau_times_velocity",
    "reference",
    "tau_times_reference_velocity",
    "captured_action_over_limit",
    "observation_age_over_tau",
    "sensor_delay_over_tau",
    "planned_action_delay_over_tau",
    "current_action_over_limit",
    "current_action_age_over_tau",
)
NUM_FEATURES = len(FEATURE_NAMES)


def _check_window(max_tokens: int, history_seconds: float) -> float:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, Integral) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    history_seconds = _finite_scalar(history_seconds, "history_seconds")
    if history_seconds <= 0:
        raise ValueError("history_seconds must be positive")
    return history_seconds


def _positive(value: float, name: str) -> float:
    value = _finite_scalar(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def shared_snapshot(
    snapshot: PolicyInput, max_tokens: int = 11, history_seconds: float = 0.5
) -> PolicyInput:
    """Validate causality, then return the common immutable information budget.

    Observations are oldest first, bounded relative to the latest capture and
    then limited to ``max_tokens``. Only the latest already-applied command is
    exposed. This deliberately removes the predictor teacher's access to the
    full applied-action history. An empty action history means the simulator's
    initial zero command at time zero. Invalid future input raises even if it
    would subsequently be discarded by the history window.
    """
    history_seconds = _check_window(max_tokens, history_seconds)
    time = _finite_scalar(snapshot.time, "snapshot.time")
    apply_time = _finite_scalar(snapshot.planned_apply_time, "planned_apply_time")
    if time < 0 or apply_time < time:
        raise ValueError("Snapshot time must be nonnegative and application cannot be in the past")
    if not snapshot.observations:
        raise ValueError("A snapshot needs at least one available observation")

    previous_capture = -np.inf
    for observation in snapshot.observations:
        capture = _finite_scalar(observation.capture_time, "capture_time")
        available = _finite_scalar(observation.available_time, "available_time")
        if capture < 0 or capture < previous_capture or not capture <= available <= time:
            raise ValueError("Observations must be ordered, causally captured, and available")
        previous_capture = capture
        for name in ("position", "velocity", "reference", "reference_velocity", "applied_action"):
            _finite_scalar(getattr(observation, name), f"observation.{name}")

    previous_action_time = -np.inf
    for action in snapshot.actions:
        action_time = _finite_scalar(action.time, "action.time")
        _finite_scalar(action.value, "action.value")
        if action_time < 0 or action_time < previous_action_time or action_time > time:
            raise ValueError("Applied actions must be ordered and already applied")
        previous_action_time = action_time

    cutoff = snapshot.latest.capture_time - history_seconds
    # The simulator has a nanosecond clock; tolerate only roundoff at a window
    # boundary, not a full clock tick or a genuinely unavailable observation.
    retained = tuple(obs for obs in snapshot.observations if obs.capture_time >= cutoff - 1e-12)
    actions = (snapshot.actions[-1],) if snapshot.actions else (AppliedAction(0.0, 0.0, None),)
    return PolicyInput(time, apply_time, retained[-max_tokens:], actions)


def encode_snapshot(
    snapshot: PolicyInput,
    tau: float = 0.5,
    action_limit: float = 5.0,
    max_tokens: int = 11,
    history_seconds: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 ``[tokens, 10]`` features and a valid-prefix boolean mask.

    Rows preserve temporal order. Padding is on the right and is exactly zero.
    ``tau`` and ``action_limit`` must match the declared controller deployment
    configuration. Position and reference have unit scale. Action features are
    normalized but never clipped, preserving the information in the snapshot.
    """
    tau = _positive(tau, "tau")
    action_limit = _positive(action_limit, "action_limit")
    bounded = shared_snapshot(snapshot, max_tokens, history_seconds)
    tokens = np.zeros((max_tokens, NUM_FEATURES), dtype=np.float32)
    valid = np.zeros(max_tokens, dtype=np.bool_)
    current = bounded.actions[0]
    for row, obs in enumerate(bounded.observations):
        tokens[row] = (
            obs.position,
            tau * obs.velocity,
            obs.reference,
            tau * obs.reference_velocity,
            obs.applied_action / action_limit,
            (bounded.time - obs.capture_time) / tau,
            (obs.available_time - obs.capture_time) / tau,
            (bounded.planned_apply_time - bounded.time) / tau,
            current.value / action_limit,
            (bounded.time - current.time) / tau,
        )
        valid[row] = True
    if not np.all(np.isfinite(tokens)):
        raise ValueError("Snapshot features exceed finite float32 representation")
    return tokens, valid


@dataclass(frozen=True)
class SharedTeacher:
    """Restrict the teacher to the student's snapshot and common actuator limit."""

    teacher: Callable[[PolicyInput], float]
    max_tokens: int = 11
    history_seconds: float = 0.5
    action_limit: float = 5.0

    def __post_init__(self) -> None:
        if not callable(self.teacher):
            raise ValueError("teacher must be callable")
        _check_window(self.max_tokens, self.history_seconds)
        object.__setattr__(self, "action_limit", _positive(self.action_limit, "action_limit"))

    def __call__(self, snapshot: PolicyInput) -> float:
        bounded = shared_snapshot(snapshot, self.max_tokens, self.history_seconds)
        action = _finite_scalar(self.teacher(bounded), "teacher action")
        return float(np.clip(action, -self.action_limit, self.action_limit))


def collect_demonstrations(
    plant: Oscillator,
    teacher: Callable[[PolicyInput], float],
    episodes: list[dict],
    max_tokens: int = 11,
    action_limit: float = 5.0,
    history_seconds: float = 0.5,
) -> dict[str, np.ndarray]:
    """Record bounded teacher rollouts for episode-level supervised splits.

    Episode dictionaries contain a unique ``id`` and optionally ``duration``
    (6 seconds), ``initial_state`` ((0, 0)), ``schedule`` (serial),
    ``compute_duration`` (0.05 seconds), ``reference_changes``,
    ``disturbance_changes``, ``reference_initial``, and ``disturbance_initial``.
    All sensing, reporting, and fixed-cadence decision intervals are 0.05 s;
    sensor and actuator latency are zero. Only declared inference delay varies.

    Returns ``tokens`` [N, T, 10], ``valid`` [N, T], ``targets`` [N] (clipped
    commands divided by action_limit), and ``episode_ids`` [N]. Episodes are
    not split or shuffled here. Censored trajectories raise rather than
    silently adding a selected prefix to the demonstration distribution.
    """
    if not episodes:
        raise ValueError("At least one demonstration episode is required")
    ids = [episode["id"] for episode in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("Demonstration episode ids must be distinct")
    bounded_teacher = SharedTeacher(teacher, max_tokens, history_seconds, action_limit)
    action_limit = bounded_teacher.action_limit
    token_rows, valid_rows, targets, episode_ids = [], [], [], []

    for episode in episodes:
        def record(snapshot: PolicyInput) -> float:
            tokens, valid = encode_snapshot(
                snapshot, plant.tau, action_limit, max_tokens, history_seconds
            )
            command = bounded_teacher(snapshot)
            token_rows.append(tokens)
            valid_rows.append(valid)
            targets.append(command / action_limit)
            episode_ids.append(episode["id"])
            return command

        config = TimingConfig(
            duration=episode.get("duration", 6.0),
            observation_interval=0.05,
            decision_interval=0.05,
            sample_interval=0.05,
            compute_duration=episode.get("compute_duration", 0.05),
            schedule=episode.get("schedule", "serial"),
            history_seconds=history_seconds,
            action_limit=action_limit,
            max_abs_state=1e6,
        )
        result = simulate(
            plant,
            record,
            config,
            initial_state=episode.get("initial_state", (0.0, 0.0)),
            reference=PiecewiseConstant(
                initial=episode.get("reference_initial", 0.0),
                changes=episode.get("reference_changes", ()),
            ),
            disturbance=PiecewiseConstant(
                initial=episode.get("disturbance_initial", 0.0),
                changes=episode.get("disturbance_changes", ()),
            ),
        )
        if result.failure_reason is not None:
            raise ValueError(f"Demonstration episode {episode['id']!r} was censored: {result.failure_reason}")

    return {
        "tokens": np.stack(token_rows),
        "valid": np.stack(valid_rows),
        "targets": np.asarray(targets, dtype=np.float32),
        "episode_ids": np.asarray(episode_ids),
    }
