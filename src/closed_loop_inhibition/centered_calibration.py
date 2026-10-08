"""Clipping-aware command centering on independent calibration histories.

Offsets and centers are physical command values. Calibration uses equal weight
for every scenario, irrespective of its number of decision-time samples. An
offset fitted here is a fixed policy parameter: it does not preserve the mean
on new histories or guarantee that the controller's equilibrium is unchanged.
"""

import math
from numbers import Real

import numpy as np


def _vector(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real values")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return array


def _scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _group_masks(groups, length):
    labels = np.asarray(groups)
    if labels.ndim != 1 or len(labels) != length:
        raise ValueError("groups must be one-dimensional and match the values")
    if not all(isinstance(label, (str, np.str_)) for label in labels):
        raise ValueError("groups must contain string labels")
    return [labels == label for label in np.unique(labels)]


def _mean(values):
    # Scaling avoids overflow when the finite values have a very large mean.
    scale = float(np.max(np.abs(values)))
    return float(np.mean(values / scale) * scale) if scale else 0.0


def weighted_mean(values, groups):
    """Mean of within-scenario means, with every scenario weighted equally."""
    values = _vector(values, "values")
    masks = _group_masks(groups, len(values))
    return _mean(np.array([_mean(values[mask]) for mask in masks]))


def centered_commands(raw, *, action_limit, center=0.0, gain=1.0, offset=0.0):
    """Return ``clip(center + gain * (raw - center) + offset, -limit, limit)``.

    Transform raw, pre-actuator commands; clipping before this transformation
    would implement a different policy. The fitted center and offset must stay
    fixed across evaluation histories, including pulse and sham arms.
    """
    raw = _vector(raw, "raw")
    limit = _scalar(action_limit, "action_limit", positive=True)
    center = _scalar(center, "center")
    gain = _scalar(gain, "gain", nonnegative=True)
    offset = _scalar(offset, "offset")
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = center + gain * (raw - center) + offset
    if not np.isfinite(transformed).all():
        raise ValueError("command transformation exceeds finite float64 representation")
    return np.clip(transformed, -limit, limit)


def fit_command_offset(native_raw, changed_raw, groups, action_limit, atol=1e-10):
    """Fit a constant offset preserving the native clipped calibration mean.

    Solve ``mean_groups(clip(changed_raw + b)) == mean_groups(clip(native_raw))``
    using deterministic bisection with guaranteed saturation bounds. Inputs
    should be raw commands on the same independent, no-disturbance histories.
    For a gain control, provide the raw centered gain transformation as
    ``changed_raw``; the extra fitted offset corrects clipping-induced shifts.

    Returned errors are signed changed-minus-native means in physical units;
    ``raw_mean_error`` means *before offset correction*, after actuator clipping.
    No input arrays are changed. A nonrepresentable bracket or an unresolved
    mean tolerance raises rather than silently reporting a successful fit.
    """
    native = _vector(native_raw, "native_raw")
    changed = _vector(changed_raw, "changed_raw")
    if native.shape != changed.shape:
        raise ValueError("native_raw and changed_raw must have the same length")
    limit = _scalar(action_limit, "action_limit", positive=True)
    tolerance = _scalar(atol, "atol", positive=True)
    masks = _group_masks(groups, len(native))

    def mean(values):
        return _mean(np.array([_mean(values[mask]) for mask in masks]))

    target = mean(np.clip(native, -limit, limit))
    uncorrected = mean(np.clip(changed, -limit, limit))
    raw_error = uncorrected - target

    def result(offset, corrected, iterations):
        return {
            "offset": float(offset), "target_mean": target,
            "corrected_mean": float(corrected),
            "mean_error": float(corrected - target),
            "raw_mean_error": float(raw_error), "iterations": iterations,
        }

    if abs(raw_error) <= tolerance:
        return result(0.0, uncorrected, 0)
    with np.errstate(over="ignore", invalid="ignore"):
        low = float(-limit - np.max(changed))
        high = float(limit - np.min(changed))
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("offset bracket exceeds finite float64 representation")

    # These endpoints saturate every command. Handle exact endpoint targets
    # directly, avoiding an unnecessary numerical search on a flat objective.
    if target == -limit:
        return result(low, -limit, 0)
    if target == limit:
        return result(high, limit, 0)

    for iteration in range(1, 161):
        midpoint = low / 2.0 + high / 2.0
        corrected = mean(centered_commands(changed, action_limit=limit, offset=midpoint))
        if abs(corrected - target) <= tolerance:
            return result(midpoint, corrected, iteration)
        if midpoint == low or midpoint == high:
            break
        if corrected < target:
            low = midpoint
        else:
            high = midpoint
    raise ValueError("offset calibration cannot resolve the requested mean tolerance")
