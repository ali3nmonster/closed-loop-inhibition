"""Protect conditional-history interpretation and collective-control matching."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import closed_loop_inhibition.collective_dynamics as dynamics
from closed_loop_inhibition.neural import make_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    base = json.loads((ROOT / "configs/timescale_maps.json").read_text())
    protocol = json.loads((ROOT / "configs/collective_suppression.json").read_text())
    protocol["frequency"].update(frequencies=[2.], phases=[0., np.pi / 2], duration=1., burn_in=.2)
    protocol["dynamics"].update(duration=.3, pulse_onset=.1, pulse_width=.05,
                                 calibration_seeds=[1020001], confirmation_seeds=[1040001])
    base["model"]["width"] = 8
    return base, protocol


@pytest.fixture
def model(settings):
    torch.set_num_threads(2)
    torch.manual_seed(991)
    return make_model("transformer", **settings[0]["model"]).eval()


def test_harmonic_fit_separates_phase_dc_and_nonlinear_residual():
    times = np.arange(4., 12., .05)
    values = 3 * np.sin(2 * np.pi * 2 * times) - 4 * np.cos(2 * np.pi * 2 * times) + 2
    result = dynamics.fit_harmonic(times, values, 2.)
    assert result["amplitude"] == pytest.approx(5.)
    assert result["phase_rad"] == pytest.approx(np.arctan2(-4., 3.))
    assert result["dc"] == pytest.approx(2.)
    assert result["fit_residual_rms"] < 1e-12
    nonlinear = dynamics.fit_harmonic(times, values + np.sin(2 * np.pi * 4 * times), 2.)
    assert nonlinear["amplitude"] == pytest.approx(5.)
    assert nonlinear["fit_residual_rms"] == pytest.approx(1 / np.sqrt(2))


@pytest.mark.parametrize("size", range(11))
def test_alternative_group_is_same_size_distinct_and_has_minimal_overlap(size):
    identifiers = [str(index) for index in range(10)]
    selected = identifiers[:size]
    alternative = dynamics.alternative_group(identifiers, selected, 23)
    assert alternative == dynamics.alternative_group(identifiers, selected, 23)
    if size in (0, 10):
        assert alternative is None
    else:
        assert len(alternative) == size
        assert set(alternative) != set(selected)
        assert len(set(alternative) & set(selected)) == max(0, 2 * size - 10)


def test_direction_reports_opposite_controls_despite_equal_effect_size():
    native = (np.ones(4), np.zeros(4))
    weak = (np.ones(4) * 1.1, np.zeros(4))
    opposite = (np.ones(4) * .9, np.zeros(4))
    groups = np.asarray(["a", "a", "b", "b"])
    result = dynamics._direction(native, weak, opposite, groups)
    assert result["weak_response_delta"] == pytest.approx(.1)
    assert result["control_response_delta"] == pytest.approx(-.1)
    assert result["command_change_cosine"] == pytest.approx(-1.)
    assert result["response_direction_concordant"] is False


def test_gain_magnitude_tie_cannot_choose_opposite_response_direction(settings):
    protocol = settings[1]
    rows = [{"parameter": .9, "achieved_rms": .1, "direction_eligible": False},
            {"parameter": 1.1, "achieved_rms": .1, "direction_eligible": True}]
    selected, reason = dynamics.match_directional_gain(rows, .1, .1, protocol["dynamics"])
    assert selected["parameter"] == 1.1 and selected["matched"]
    assert reason is None
    missing, reason = dynamics.match_directional_gain(rows, .1, 0., protocol["dynamics"])
    assert missing is None and reason == "joint_response_change_below_direction_floor"
    missing, reason = dynamics.match_directional_gain(rows[:1], .1, .1, protocol["dynamics"])
    assert missing is None and reason == "no_gain_value_with_matching_response_direction"


def test_empty_selection_is_identity_and_unclassifiable_remains_undefined(settings, model, monkeypatch):
    base, protocol = settings
    monkeypatch.setattr(dynamics, "_bank", lambda *args, **kwargs: pytest.fail("No calibration is needed"))
    empty = dynamics.prepare_dynamics(base, protocol, .2, .05, 11, model, [])
    assert empty["status"] == "empty_group"
    assert empty["variants"]["joint_weak"]["branch_scales"] == {}
    assert empty["variants"]["joint_weak"]["achieved_rms"] == 0.
    assert "gain_matched" not in empty["variants"]
    assert "alternative_matched" not in empty["variants"]
    missing = dynamics.prepare_dynamics(base, protocol, .2, .05, 11, model, None)
    assert missing["status"] == "unclassifiable"
    assert set(missing["variants"]) == {"native"}


def test_frequency_histories_cached_and_exogenous_noise_free(settings):
    base, protocol = settings
    first = dynamics.frequency_bank(base, protocol, .2, 2., 0.)
    assert first is dynamics.frequency_bank(base, protocol, .2, 2., 0.)
    changed = deepcopy(base)
    changed["noise_std"] = 100.
    second = dynamics.frequency_bank(changed, protocol, .2, 2., 0.)
    assert first["sha256"] == second["sha256"]
    assert np.all(first["decision_times"] >= protocol["frequency"]["burn_in"])
    assert np.all(first["decision_times"] < protocol["frequency"]["duration"])
    np.testing.assert_array_equal(first["sham"][0][:, :, :2], 0.)
    with pytest.raises(ValueError, match="Nyquist"):
        dynamics.frequency_bank(base, protocol, .2, 10., 0.)


def test_frequency_empty_group_is_zero_effect_and_missing_group_is_null(settings, model, monkeypatch):
    base, protocol = settings
    protocol["frequency"]["amplitude_epsilon"] = 1e-12
    monkeypatch.setattr(dynamics, "NeuralPolicy", lambda *args, **kwargs: pytest.fail("No random-model own loop in frequency assay"))
    result = dynamics.frequency_assay(base, protocol, .2, .05, 11,
                                       {"initial": model, "selected": model},
                                       {"initial": [], "selected": None})
    initial = [row for row in result["rows"] if row["checkpoint"] == "initial"]
    assert len(initial) == 2
    for row in initial:
        assert row["available"] and row["baseline_valid"] and row["phase_valid"]
        assert row["gain_ratio_percent"] == pytest.approx(0., abs=1e-12)
        assert row["phase_difference_rad"] == pytest.approx(0., abs=1e-12)
    for row in result["rows"]:
        if row["checkpoint"] == "selected":
            assert row["available"] is False
            assert row["weak_amplitude"] is None and row["gain_ratio_percent"] is None
    json.dumps(result, allow_nan=False)


def test_frequency_floor_masks_ratios_and_phase_but_preserves_absolute_amplitudes(settings, model):
    base, protocol = settings
    protocol["frequency"]["amplitude_epsilon"] = 100.
    result = dynamics.frequency_assay(base, protocol, .2, .05, 11, {"selected": model}, {"selected": []})
    for row in result["rows"]:
        assert not row["baseline_valid"] and not row["phase_valid"]
        assert row["gain_ratio_percent"] is None and row["phase_difference_rad"] is None
        assert row["native_phase_rad"] is None and row["weak_phase_rad"] is None
        assert row["native_amplitude"] >= 0. and row["weak_amplitude"] >= 0.


def test_empty_group_own_loops_preserve_identity_and_count_unique_noise_seeds(settings, model):
    base, protocol = settings
    # A constructed zero-command controller, not an untrained random model
    # used to claim anything about learned closed-loop behavior.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    prepared = dynamics.prepare_dynamics(base, protocol, .2, .05, 11, model, [])
    result = dynamics.confirm_dynamics(base, protocol, .2, .05, 11, model, prepared)
    assert set(result["summary"]) == {"native", "joint_weak", "joint_strong"}
    assert len(result["rollouts"]) == 6  # Three variants, one seed, two pulse signs.
    for summary in result["summary"].values():
        assert summary["independent_noise_seeds"] == 1
        assert summary["sham"]["censored"] == 0.
        assert summary["delta_from_native"]["sham"]["position_rms"] == 0.
        assert summary["delta_from_native"]["paired_recovery"]["normalized_position_energy"] == 0.
    json.dumps(result, allow_nan=False)


def test_confirmation_refuses_mutated_protocol_or_cross_stage_seed_reuse(settings, model):
    base, protocol = settings
    prepared = dynamics.prepare_dynamics(base, protocol, .2, .05, 11, model, [])
    changed = deepcopy(protocol)
    changed["dynamics"]["weak_scale"] = .8
    with pytest.raises(ValueError, match="does not match"):
        dynamics.confirm_dynamics(base, changed, .2, .05, 11, model, prepared)
    reused = deepcopy(protocol)
    reused["dynamics"]["calibration_seeds"] = protocol["discovery_seeds"]
    with pytest.raises(ValueError, match="all be disjoint"):
        dynamics.prepare_dynamics(base, reused, .2, .05, 11, model, [])
