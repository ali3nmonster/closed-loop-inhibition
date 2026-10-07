"""Finite-horizon measurements with explicit censoring of incomplete episodes."""

import math

import numpy as np

from .plants import _finite_scalar
from .timing import SimulationResult


_SCORE_KEYS = (
    "tracking_ise", "tracking_rmse", "peak_abs_error", "peak_abs_position",
    "max_abs_velocity", "action_effort", "action_variation", "max_abs_action",
)


def summarize(
    result: SimulationResult, last_change_time: float = 0.0, settling_tolerance: float = 0.02
) -> dict:
    """Summarize a complete trajectory, returning only JSON-safe scalar values.

    Tracking scores and peaks use reporting samples. Action effort integrates
    the actual zero-order-held command exactly through the final sample;
    variation includes each applied jump, including the initial jump from zero.
    Settling is the first sample at or after ``last_change_time`` whose entire
    remaining suffix stays within the position-error band. It is evidence only
    for that remaining horizon, not an asymptotic stability statement.

    Failed or incomplete episodes have null scores, so truncated integrals
    cannot rank favorably against full episodes. ``observed_duration`` reports
    the actual reporting coverage, which can end before the failure event.
    """
    last_change_time = _finite_scalar(last_change_time, "last_change_time")
    settling_tolerance = _finite_scalar(settling_tolerance, "settling_tolerance")
    if last_change_time < 0 or settling_tolerance < 0:
        raise ValueError("last_change_time and settling_tolerance must be nonnegative")

    summary = {key: None for key in _SCORE_KEYS}
    summary.update({
        "settling_time": None,
        "settling_observed_for": None,
        "horizon_edge_settling": False,
        "observed_duration": 0.0,
        "last_change_time": last_change_time,
        "remaining_observation_horizon": 0.0,
        "censored": result.failure_reason is not None,
        "failure_reason": result.failure_reason,
    })
    if not result.samples:
        summary["censored"] = True
        summary["failure_reason"] = result.failure_reason or "no_reporting_samples"
        return summary

    samples = np.array([
        [sample["time"], sample["position"], sample["velocity"], sample["reference"]]
        for sample in result.samples
    ], dtype=float)
    if not np.all(np.isfinite(samples)):
        raise ValueError("Reporting samples must contain finite times and states")
    times, positions, velocities, references = samples.T
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Reporting sample times must be nonnegative and strictly increasing")
    duration = float(times[-1] - times[0])
    end = float(times[-1])
    summary["observed_duration"] = duration
    summary["remaining_observation_horizon"] = max(0.0, end - last_change_time)
    if result.failure_reason is not None:
        return summary
    if times[0] != 0 or duration <= 0 or abs(end - result.config.duration) > 1e-9:
        summary["censored"] = True
        summary["failure_reason"] = "incomplete_reporting_horizon"
        return summary

    effort = 0.0
    variation = 0.0
    max_action = 0.0
    held = 0.0
    previous_time = 0.0
    last_record_time = -np.inf
    for action in result.applied_actions:
        at = _finite_scalar(action.time, "applied action time")
        value = _finite_scalar(action.value, "applied action value")
        if at < 0 or at < last_record_time:
            raise ValueError("Applied action times must be nonnegative and ordered")
        last_record_time = at
        if at > end:
            continue
        effort += held * held * (at - previous_time)
        variation += abs(value - held)
        max_action = max(max_action, abs(value))
        held, previous_time = value, at
    effort += held * held * (end - previous_time)

    errors = positions - references
    # NumPy 1.26 is supported by pyproject; trapezoid was named trapz there.
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    with np.errstate(over="ignore", invalid="ignore"):
        ise = float(trapezoid(errors * errors, times))
    values = {
        "tracking_ise": ise,
        "tracking_rmse": math.sqrt(ise / duration),
        "peak_abs_error": float(np.max(np.abs(errors))),
        "peak_abs_position": float(np.max(np.abs(positions))),
        "max_abs_velocity": float(np.max(np.abs(velocities))),
        "action_effort": float(effort),
        "action_variation": float(variation),
        "max_abs_action": float(max_action),
    }
    if not all(math.isfinite(value) for value in values.values()):
        summary["censored"] = True
        summary["failure_reason"] = "nonfinite_metric"
        return summary
    summary.update(values)

    inside = np.abs(errors) <= settling_tolerance
    inside_to_end = np.logical_and.accumulate(inside[::-1])[::-1]
    candidates = np.flatnonzero(inside_to_end & (times >= last_change_time))
    if candidates.size:
        index = int(candidates[0])
        entry = float(times[index])
        summary["settling_time"] = entry - last_change_time
        summary["settling_observed_for"] = end - entry
        summary["horizon_edge_settling"] = index == len(times) - 1
    return summary
