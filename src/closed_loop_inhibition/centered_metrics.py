"""Signed operating-point diagnostics with the existing censoring contract."""

import math

import numpy as np

from .metrics import summarize
from .timing import SimulationResult


def mean_applied_action(run: SimulationResult) -> float | None:
    """Return the exact time mean of the actual held action over the horizon.

    The initial command is zero until the first applied action. Application
    times, including clipping already performed by the simulator, determine
    interval weights; neither dispatch counts nor reporting samples do.
    Actions at or beyond the endpoint have no weight. Failed or incomplete
    runs return ``None``, using the same validation and censoring as summarize.
    """
    if summarize(run)["censored"]:
        return None
    end = float(run.samples[-1]["time"])
    previous_time, held = 0.0, 0.0
    intervals = []
    for action in run.applied_actions:
        if action.time > end:
            break
        intervals.append(held * (action.time - previous_time))
        previous_time, held = action.time, action.value
    intervals.append(held * (end - previous_time))
    value = math.fsum(intervals) / end
    return float(value) if math.isfinite(value) else None


def mean_position(run: SimulationResult) -> float | None:
    """Return the signed sampled position integral divided by the horizon.

    Trapezoidal integration respects nonuniform reporting times and is exact
    for a position trace that is linear between samples. It is an estimate of
    the physical trajectory mean, unlike the exact held-action mean above.
    Failed or incomplete runs return ``None``.
    """
    if summarize(run)["censored"]:
        return None
    times = np.asarray([sample["time"] for sample in run.samples], dtype=float)
    positions = np.asarray([sample["position"] for sample in run.samples], dtype=float)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    value = float(trapezoid(positions, times) / times[-1])
    return value if math.isfinite(value) else None
