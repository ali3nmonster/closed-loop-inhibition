"""Mean-command correction must respect clipping and scenario-level weights."""

import json

import numpy as np
import pytest

from closed_loop_inhibition.centered_calibration import (
    centered_commands, fit_command_offset, weighted_mean,
)


def test_fit_removes_constant_shift_in_physical_command_units():
    native = np.array([-0.8, 0.1, 0.4])
    changed = native + 0.2
    saved_native, saved_changed = native.copy(), changed.copy()
    fitted = fit_command_offset(native, changed, ["a", "b", "b"], 5.0)
    assert fitted["offset"] == pytest.approx(-0.2, abs=1e-10)
    assert fitted["raw_mean_error"] == pytest.approx(0.2)
    assert abs(fitted["mean_error"]) <= 1e-10
    np.testing.assert_array_equal(native, saved_native)
    np.testing.assert_array_equal(changed, saved_changed)
    json.dumps(fitted, allow_nan=False)


def test_fit_is_clipping_aware_and_uses_raw_changed_commands():
    fitted = fit_command_offset([-0.4, 0.4], [1.6, -0.2], ["a", "a"], 1.0)
    assert fitted["offset"] == pytest.approx(-0.7, abs=1e-10)
    assert fitted["raw_mean_error"] == pytest.approx(0.4)
    corrected = centered_commands([1.6, -0.2], action_limit=1, offset=fitted["offset"])
    np.testing.assert_allclose(corrected, [0.9, -0.9], atol=1e-10)
    # Subtracting the already-clipped mean mismatch does not meet the target.
    assert np.mean(centered_commands([1.6, -0.2], action_limit=1, offset=-0.4)) == pytest.approx(0.2)


@pytest.mark.parametrize("sign", [-1, 1])
def test_fully_saturated_target_has_finite_exact_solution(sign):
    fitted = fit_command_offset([sign * 10, sign * 20], [-100, 100], ["a", "b"], 1)
    assert fitted["offset"] == sign * 101
    assert fitted["corrected_mean"] == sign
    assert fitted["mean_error"] == 0


def test_identity_and_already_centered_changes_keep_zero_offset():
    values = [-8, 0.2, 9]
    fitted = fit_command_offset(values, values, ["a", "b", "c"], 5)
    assert fitted["offset"] == 0
    assert fitted["iterations"] == 0
    assert fitted["mean_error"] == 0
    assert fit_command_offset([-0.5, 0.5], [-0.9, 0.9], ["a", "a"], 1)["offset"] == 0


def test_scenarios_are_equally_weighted_and_duplicated_samples_do_not_reweight():
    assert weighted_mean([1, 3, 3, 3], ["a", "b", "b", "b"]) == pytest.approx(2)
    original = fit_command_offset([0, 0], [0.1, 0.3], ["a", "b"], 1)
    repeated = fit_command_offset([0, 0, 0, 0], [0.1, 0.3, 0.3, 0.3], ["a", "b", "b", "b"], 1)
    assert repeated["offset"] == pytest.approx(original["offset"], abs=1e-10)
    assert repeated["raw_mean_error"] == pytest.approx(0.2)
    assert weighted_mean([1e308, 1e308], ["a", "a"]) == pytest.approx(1e308)


def test_centered_gain_is_applied_before_the_common_actuator_clip():
    raw = np.array([0.25, 0.5, 1.0, 2.0])
    result = centered_commands(raw, action_limit=1, center=0.5, gain=2, offset=0.1)
    np.testing.assert_allclose(result, [0.1, 0.6, 1, 1])
    np.testing.assert_array_equal(raw, [0.25, 0.5, 1, 2])
    assert centered_commands([0.5], action_limit=1, center=0.5, gain=3)[0] == 0.5


def test_centering_gain_around_native_mean_still_requires_clipping_correction():
    native = np.array([-0.8, 0.4, 0.9])
    groups = ["a", "a", "a"]
    center = weighted_mean(native, groups)
    changed_raw = center + 2 * (native - center)
    before = centered_commands(native, action_limit=1, center=center, gain=2)
    assert weighted_mean(before, groups) != pytest.approx(center)
    fitted = fit_command_offset(native, changed_raw, groups, 1)
    after = centered_commands(native, action_limit=1, center=center, gain=2, offset=fitted["offset"])
    assert weighted_mean(after, groups) == pytest.approx(center, abs=1e-10)


def test_fitted_offset_is_a_constant_and_does_not_center_each_new_history():
    fitted = fit_command_offset([0, 0], [0.1, 0.3], ["a", "b"], 1)
    new_commands = centered_commands([0.5, 0.7], action_limit=1, offset=fitted["offset"])
    assert np.mean(new_commands) == pytest.approx(0.4, abs=1e-10)


@pytest.mark.parametrize("values,groups", [
    ([], []), ([[1]], ["a"]), ([float("nan")], ["a"]),
    ([float("inf")], ["a"]), ([1 + 2j], ["a"]), ([1], [0]), ([1, 2], ["a"]),
])
def test_invalid_weighted_mean_inputs_are_rejected(values, groups):
    with pytest.raises(ValueError):
        weighted_mean(values, groups)


@pytest.mark.parametrize("kwargs", [
    {"action_limit": 0}, {"action_limit": -1}, {"action_limit": True},
    {"center": float("nan")}, {"gain": -1}, {"gain": float("inf")},
    {"offset": float("inf")},
])
def test_invalid_command_transform_is_rejected(kwargs):
    with pytest.raises(ValueError):
        centered_commands([0.1], **{"action_limit": 1, **kwargs})


@pytest.mark.parametrize("kwargs", [{"atol": 0}, {"atol": -1}, {"atol": float("nan")}])
def test_invalid_tolerance_is_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_command_offset([0], [0.1], ["a"], 1, **kwargs)


def test_mismatched_lengths_and_nonfinite_arithmetic_are_rejected():
    with pytest.raises(ValueError):
        fit_command_offset([0], [0.1, 0.2], ["a"], 1)
    with pytest.raises(ValueError):
        centered_commands([1e308], action_limit=1, gain=2)
    with pytest.raises(ValueError):
        fit_command_offset([0], [1e308], ["a"], 1e308)
