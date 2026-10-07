"""Numerical summaries and calibration for paired suppression interventions.

Each group is an independent paired disturbance experiment; time samples within
one group are not independent replicates. Group RMS summaries give every group
equal weight, regardless of its number of reporting samples. Commands supplied
here must already be in physical units with the common actuator limit applied.
"""

import math
from numbers import Integral, Real

import numpy as np


def _vector(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real values")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional numeric array") from error
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return array


def _nonnegative_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _group_masks(groups, length):
    labels = np.asarray(groups)
    if labels.ndim != 1 or len(labels) != length:
        raise ValueError("groups must be one-dimensional and match the values")
    if not all(isinstance(label, (str, np.str_)) for label in labels):
        raise ValueError("groups must contain string labels")
    return [(str(label), labels == label) for label in np.unique(labels)]


def _rms(values):
    """Avoid intermediate squaring overflow for finite float64 inputs."""
    scale = float(np.max(np.abs(values)))
    return scale * float(np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0


def _paired_arrays(base_pulse, base_sham, changed_pulse, changed_sham):
    names = ("base_pulse", "base_sham", "changed_pulse", "changed_sham")
    arrays = tuple(_vector(values, name) for name, values in zip(
        names, (base_pulse, base_sham, changed_pulse, changed_sham)
    ))
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("all paired command arrays must have the same length")
    return arrays


def _difference(first, second):
    with np.errstate(over="ignore", invalid="ignore"):
        difference = first - second
    if not np.isfinite(difference).all():
        raise ValueError("command differences exceed finite float64 representation")
    return difference


def weighted_rms(values, groups):
    """sqrt(mean_groups(mean_within_group(values**2))), with equal group weights."""
    values = _vector(values, "values")
    masks = _group_masks(groups, len(values))
    return _rms(np.array([_rms(values[mask]) for _, mask in masks]))


def immediate_output_rms(base_pulse, base_sham, changed_pulse, changed_sham, groups):
    """Equal-group RMS of intervention-induced command change over both arms.

    Pulse and sham squared changes receive equal weight within each group.
    This measures immediate intervention magnitude, separately from its effect
    on the pulse-minus-sham disturbance response.
    """
    base_pulse, base_sham, changed_pulse, changed_sham = _paired_arrays(
        base_pulse, base_sham, changed_pulse, changed_sham
    )
    masks = _group_masks(groups, len(base_pulse))
    pulse_change = _difference(changed_pulse, base_pulse)
    sham_change = _difference(changed_sham, base_sham)
    return _rms(np.array([
        _rms(np.concatenate((pulse_change[mask], sham_change[mask])))
        for _, mask in masks
    ]))


def response_statistics(
    base_pulse, base_sham, changed_pulse, changed_sham, groups, amplitudes,
    *, min_fractional_increase=0.05, min_positive_fraction=0.75, baseline_epsilon=1e-8,
):
    """Summarize causal changes in pulse-minus-sham command response by group.

    Response RMS is divided by the absolute disturbance amplitude, which must
    be nonzero and constant within a group. Fractional increase is
    ``changed_response_rms / baseline_response_rms - 1``. Positive fraction is
    the fraction of valid groups with a strictly positive increase. Median and
    positive fraction summarize valid groups only; any baseline at or below
    ``baseline_epsilon`` makes the overall candidate ineligible. With no valid
    groups, the median is None and the positive fraction is zero.

    Eligibility also requires median increase strictly above the requested
    threshold and positive fraction at least the requested minimum. This is a
    discovery criterion for suppression of command response, not a stability
    or biological E/I-balance claim.
    """
    base_pulse, base_sham, changed_pulse, changed_sham = _paired_arrays(
        base_pulse, base_sham, changed_pulse, changed_sham
    )
    masks = _group_masks(groups, len(base_pulse))
    amplitudes = _vector(amplitudes, "amplitudes")
    if len(amplitudes) != len(base_pulse) or np.any(amplitudes == 0):
        raise ValueError("amplitudes must match commands and be nonzero")
    threshold = _nonnegative_scalar(min_fractional_increase, "min_fractional_increase")
    minimum_fraction = _nonnegative_scalar(min_positive_fraction, "min_positive_fraction")
    if minimum_fraction > 1:
        raise ValueError("min_positive_fraction must lie in [0, 1]")
    epsilon = _nonnegative_scalar(baseline_epsilon, "baseline_epsilon")
    baseline_response = _difference(base_pulse, base_sham)
    changed_response = _difference(changed_pulse, changed_sham)
    rows, increases = [], []
    for group, mask in masks:
        amplitude = float(amplitudes[mask][0])
        if not np.all(amplitudes[mask] == amplitude):
            raise ValueError("disturbance amplitude must be constant within each group")
        baseline_rms = _rms(baseline_response[mask]) / abs(amplitude)
        changed_rms = _rms(changed_response[mask]) / abs(amplitude)
        if not math.isfinite(baseline_rms) or not math.isfinite(changed_rms):
            raise ValueError("amplitude-normalized response exceeds finite representation")
        baseline_valid = baseline_rms > epsilon
        increase = changed_rms / baseline_rms - 1.0 if baseline_valid else None
        if increase is not None:
            if not math.isfinite(increase):
                raise ValueError("fractional response increase exceeds finite representation")
            increases.append(increase)
        rows.append({
            "group": group, "amplitude": amplitude,
            "baseline_response_rms": baseline_rms, "changed_response_rms": changed_rms,
            "fractional_increase": increase, "baseline_valid": baseline_valid,
        })
    median = float(np.median(increases)) if increases else None
    positive_fraction = float(np.mean(np.asarray(increases) > 0)) if increases else 0.0
    baselines_valid = all(row["baseline_valid"] for row in rows)
    return {
        "per_group": rows,
        "median_fractional_increase": median,
        "positive_fraction": positive_fraction,
        "eligible": bool(baselines_valid and median > threshold and positive_fraction >= minimum_fraction),
    }


def norm_match_head(head_rows, candidate_layer, candidate_head):
    """Choose the closest log-RMS head in the candidate's layer, then head index.

    Returns a copy of the chosen row with ``norm_ratio`` (control/candidate).
    Zero-norm alternatives are not eligible; the candidate must have positive
    norm. Matching is observational calibration, not proof of causal similarity.
    """
    identifiers = (candidate_layer, candidate_head)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in identifiers):
        raise ValueError("candidate layer and head must be nonnegative integers")
    rows = []
    seen = set()
    for row in head_rows:
        layer, head = row["layer"], row["head"]
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in (layer, head)):
            raise ValueError("head rows require nonnegative integer layer and head")
        if (layer, head) in seen:
            raise ValueError("head rows must have distinct (layer, head) identifiers")
        seen.add((layer, head))
        norm = _nonnegative_scalar(row["residual_rms"], "residual_rms")
        rows.append({**row, "layer": int(layer), "head": int(head), "residual_rms": norm})
    candidates = [row for row in rows if (row["layer"], row["head"]) == identifiers]
    if not candidates or candidates[0]["residual_rms"] == 0:
        raise ValueError("candidate head must be present with positive residual RMS")
    candidate_norm = candidates[0]["residual_rms"]
    alternatives = [row for row in rows if row["layer"] == candidate_layer
                    and row["head"] != candidate_head and row["residual_rms"] > 0]
    if not alternatives:
        raise ValueError("no positive-norm alternative head in the candidate layer")
    selected = min(alternatives, key=lambda row: (
        abs(math.log(row["residual_rms"]) - math.log(candidate_norm)), row["head"]
    ))
    ratio = selected["residual_rms"] / candidate_norm
    if not math.isfinite(ratio):
        raise ValueError("matched residual norm ratio exceeds finite representation")
    return {**selected, "norm_ratio": ratio}


def pick_rms_match(scales, achieved_rms, target_rms, tolerance=0.1):
    """Select RMS closest to target; ties prefer scale closest to one, then lower.

    ``matched`` is true when the relative RMS error is at most ``tolerance``.
    The selected setting is returned even when no available setting matches.
    """
    scales = _vector(scales, "scales")
    achieved_rms = _vector(achieved_rms, "achieved_rms")
    if len(scales) != len(achieved_rms) or np.any(scales < 0) or np.any(achieved_rms < 0):
        raise ValueError("scales and achieved_rms must match in length and be nonnegative")
    target = _nonnegative_scalar(target_rms, "target_rms")
    if target == 0:
        raise ValueError("target_rms must be positive")
    tolerance = _nonnegative_scalar(tolerance, "tolerance")
    selected = min(range(len(scales)), key=lambda index: (
        abs(achieved_rms[index] - target), abs(scales[index] - 1), scales[index]
    ))
    relative_error = float(abs(achieved_rms[selected] - target) / target)
    if not math.isfinite(relative_error):
        raise ValueError("relative RMS error exceeds finite representation")
    return {
        "scale": float(scales[selected]), "achieved_rms": float(achieved_rms[selected]),
        "target_rms": target, "relative_error": relative_error,
        "matched": bool(relative_error <= tolerance or math.isclose(relative_error, tolerance, rel_tol=1e-12)),
    }
