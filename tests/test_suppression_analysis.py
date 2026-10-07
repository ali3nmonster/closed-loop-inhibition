"""Check paired-response selection and equal-group intervention calibration."""

from copy import deepcopy
import json

import numpy as np
import pytest

from closed_loop_inhibition.suppression_analysis import (
    immediate_output_rms,
    norm_match_head,
    pick_rms_match,
    response_statistics,
    weighted_rms,
)


def test_weighted_rms_gives_equal_weight_to_groups_not_reporting_rows():
    assert weighted_rms([1, 3, 3, 3], ["a", "b", "b", "b"]) == pytest.approx(np.sqrt(5))
    assert weighted_rms([1, 3], ["a", "b"]) == pytest.approx(np.sqrt(5))
    assert weighted_rms([1e200, -1e200], ["a", "b"]) == pytest.approx(1e200)


def test_immediate_rms_averages_both_arms_without_cancelling_common_changes():
    # Group a has changes (pulse, sham)=(1, 3); b=(2, 2).
    result = immediate_output_rms([2, 4, 4], [1, 2, 2], [3, 6, 6], [4, 4, 4], ["a", "b", "b"])
    assert result == pytest.approx(np.sqrt((5 + 4) / 2))
    assert immediate_output_rms([1], [0], [3], [2], ["a"]) == pytest.approx(2)


def test_response_uses_paired_differences_amplitude_normalization_and_group_votes():
    result = response_statistics(
        [4, -2, 4, 4, 3], [2, 0, 0, 0, 1],
        [6, -4, 5, 5, 2], [2, 0, 0, 0, 1],
        ["a", "b", "c", "c", "d"], [2, -2, 4, 4, 2],
    )
    assert [row["baseline_response_rms"] for row in result["per_group"]] == pytest.approx([1, 1, 1, 1])
    assert [row["changed_response_rms"] for row in result["per_group"]] == pytest.approx([2, 2, 1.25, 0.5])
    assert result["median_fractional_increase"] == pytest.approx(0.625)
    assert result["positive_fraction"] == 0.75
    assert result["eligible"]


def test_common_shift_in_both_arms_does_not_create_disturbance_response_suppression():
    result = response_statistics([1, 2], [0, 1], [4, 5], [3, 4], ["a", "b"], [1, 1])
    assert result["median_fractional_increase"] == 0
    assert result["positive_fraction"] == 0
    assert not result["eligible"]


def test_response_thresholds_and_zero_baseline_guard():
    args = ([1, 1], [0, 0], [1.25, 1.25], [0, 0], ["a", "b"], [1, 1])
    assert response_statistics(*args)["eligible"]
    assert not response_statistics(*args, min_fractional_increase=0.25)["eligible"]
    no_response = response_statistics([1, 0], [0, 0], [2, 1], [0, 0], ["a", "b"], [1, 1])
    assert not no_response["eligible"]
    assert no_response["per_group"][1]["fractional_increase"] is None
    all_zero = response_statistics([0], [0], [1], [0], ["a"], [1])
    assert not all_zero["eligible"]
    assert all_zero["median_fractional_increase"] is None
    json.dumps(all_zero, allow_nan=False)
    near_zero = response_statistics([1e-8], [0], [1], [0], ["a"], [1])
    assert not near_zero["eligible"]


def test_duplicating_rows_within_one_group_changes_none_of_the_group_statistics():
    arrays = [np.array(values, dtype=float) for values in (
        [1, 3, 2], [0, 1, 0], [2, 4, 3], [0, 1, 0], [1, 1, -2],
    )]
    groups = np.array(["a", "a", "b"])
    indices = [0, 1, 0, 1, 0, 1, 2]
    expanded = [values[indices] for values in arrays]
    original = response_statistics(*arrays[:4], groups, arrays[4])
    repeated = response_statistics(*expanded[:4], groups[indices], expanded[4])
    assert repeated == original
    assert weighted_rms(arrays[0], groups) == pytest.approx(weighted_rms(expanded[0], groups[indices]))
    assert immediate_output_rms(*arrays[:4], groups) == pytest.approx(
        immediate_output_rms(*expanded[:4], groups[indices])
    )


def test_norm_matching_uses_same_layer_log_distance_and_preserves_input_rows():
    rows = [
        {"layer": 0, "head": 0, "residual_rms": 2.0},
        {"layer": 0, "head": 1, "residual_rms": 1.0},
        {"layer": 0, "head": 2, "residual_rms": 3.0, "label": "control"},
        {"layer": 1, "head": 0, "residual_rms": 2.0},
    ]
    original = deepcopy(rows)
    result = norm_match_head(rows, 0, 0)
    assert result == {**rows[2], "norm_ratio": 1.5}
    assert rows == original


def test_norm_match_ties_use_head_index_and_skip_zero_norm_controls():
    rows = [
        {"layer": 0, "head": 3, "residual_rms": 1.0},
        {"layer": 0, "head": 2, "residual_rms": 0.5},
        {"layer": 0, "head": 1, "residual_rms": 2.0},
        {"layer": 0, "head": 0, "residual_rms": 0.0},
    ]
    assert norm_match_head(rows, 0, 3)["head"] == 1


def test_rms_calibration_selects_closest_and_reports_unmatched_targets():
    result = pick_rms_match([1, 0.8, 0.5], [0, 0.2, 0.6], 0.22)
    assert result["scale"] == 0.8
    assert result["matched"]
    assert result["relative_error"] == pytest.approx(0.02 / 0.22)
    unmatched = pick_rms_match([1, 0.5], [0, 1], 0.4)
    assert unmatched["scale"] == 1
    assert not unmatched["matched"]
    assert pick_rms_match([0.5], [1.1], 1, tolerance=0.1)["matched"]


def test_rms_match_ties_prefer_scale_closest_to_one_then_lower_scale():
    assert pick_rms_match([0.25, 0.75], [0.75, 1.25], 1)["scale"] == 0.75
    assert pick_rms_match([1.25, 0.75], [0.75, 1.25], 1)["scale"] == 0.75


@pytest.mark.parametrize("values,groups", [
    ([], []), ([float("nan")], ["a"]), ([float("inf")], ["a"]),
    ([[1]], ["a"]), ([1, 2], ["a"]), ([1 + 2j], ["a"]), ([1], [0]),
])
def test_invalid_weighted_rms_input_is_rejected(values, groups):
    with pytest.raises(ValueError):
        weighted_rms(values, groups)


@pytest.mark.parametrize("amplitudes", [[0, 0], [1], [1, 2], [1, float("nan")]])
def test_invalid_disturbance_amplitudes_are_rejected(amplitudes):
    with pytest.raises(ValueError):
        response_statistics([1, 2], [0, 0], [2, 3], [0, 0], ["a", "a"], amplitudes)


@pytest.mark.parametrize("kwargs", [
    {"min_positive_fraction": 1.1}, {"min_positive_fraction": -0.1},
    {"min_fractional_increase": float("nan")}, {"baseline_epsilon": -1},
])
def test_invalid_response_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        response_statistics([1], [0], [2], [0], ["a"], [1], **kwargs)


@pytest.mark.parametrize("scales,achieved,target", [
    ([0.5], [1], 0), ([0.5], [float("nan")], 1),
    ([0.5, 1], [1], 1), ([-0.5], [1], 1), ([0.5], [-1], 1),
])
def test_invalid_calibration_inputs_are_rejected(scales, achieved, target):
    with pytest.raises(ValueError):
        pick_rms_match(scales, achieved, target)


def test_norm_match_rejects_missing_degenerate_or_duplicate_candidates():
    candidate = {"layer": 0, "head": 0, "residual_rms": 1.0}
    for rows in ([], [candidate], [candidate, candidate], [{**candidate, "residual_rms": 0}]):
        with pytest.raises(ValueError):
            norm_match_head(rows, 0, 0)
