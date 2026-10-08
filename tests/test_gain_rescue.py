"""Independent response fitting, causal tape reuse and frozen-fit checks."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import closed_loop_inhibition.gain_rescue as rescue
from closed_loop_inhibition.collective_dynamics import _digest
from closed_loop_inhibition.neural import make_model
from closed_loop_inhibition.timescale_diagnostics import _clipped


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    base = json.loads((ROOT / "configs/timescale_maps.json").read_text())
    base["model"]["width"] = 8
    protocol = {"delay_cue": .05, "calibration_seeds": [5010001], "confirmation_seeds": [5030001],
                "probe": {"duration": .3, "pulse_onset": .1, "pulse_width": .05, "pulse_amplitudes": [-.02, .02]},
                "baseline_epsilon": .0001, "min_fractional_increase": .01, "min_positive_fraction": .75,
                "match_tolerance": .05, "response_direction_epsilon": 1e-12,
                "gain_bounds": [.25, 4.], "response_energy_floor": 1e-16}
    return base, protocol


@pytest.fixture
def model(settings):
    torch.set_num_threads(2)
    torch.manual_seed(381)
    return make_model("transformer", **settings[0]["model"]).eval()


def inherited(base):
    return {"tau": .2, "noise_tau": .05, "seed": 11, "group": ["L0H0"], "status": "prepared",
            "base_config_sha256": _digest(base), "unavailable_controls": {},
            "variants": {"native": {"branch_scales": {}, "offset": 0., "gain": 1., "center": 0.},
                         "joint_weak": {"branch_scales": {"L0H0": .9}, "offset": .001, "gain": 1., "center": 0., "achieved_rms": .002},
                         "joint_strong": {"branch_scales": {"L0H0": 1.1}, "offset": -.001, "gain": 1., "center": 0., "achieved_rms": .002},
                         "unused_inherited_control": {"some": "historical metadata"}}}


def test_exact_affine_response_rescue_and_native_sham_mean():
    groups = np.asarray(["a", "a", "b", "b"])
    raw = (np.asarray([.1, .2, .04, -.12]), np.asarray([.01, .02, .03, .04]))
    native = tuple(.8 * value - .01 for value in raw)
    fit, error = rescue.fit_response_gain(raw, native, native, groups, .5, [.25, 4.], 1e-16)
    assert error is None and fit["gain"] == pytest.approx(.8)
    assert not fit["gain_at_bound"]
    commands = _clipped(raw, {"action_limit": .5}, fit)
    np.testing.assert_allclose(commands, native, atol=1e-10)
    diagnostics = rescue.waveform_diagnostics(commands, native, native, raw, groups, 1e-16)
    assert diagnostics["response_error_rms"] < 1e-10
    assert diagnostics["response_cosine"] == pytest.approx(1.)
    assert diagnostics["sham_mean_change_from_native"] == pytest.approx(0., abs=1e-10)


def test_equal_probe_weighting_not_equal_row_weighting():
    groups = np.asarray(["a", "b", "b", "b"])
    source = (np.ones(4) * .01, np.zeros(4))
    target = (np.asarray([.01, .03, .03, .03]), np.zeros(4))
    fit, error = rescue.fit_response_gain(source, target, target, groups, .5, [.25, 4.], 1e-16)
    assert error is None
    assert fit["gain"] == pytest.approx(2.)  # Equal rows would give 2.5.
    assert fit["mean_fit"]["mean_error"] == 0.


def test_mean_target_separate_from_response_target_and_clipping_reportable():
    groups = np.asarray(["a", "a"])
    raw = (np.asarray([1., .1]), np.asarray([.8, -.2]))
    target = (np.asarray([.5, .3]), np.asarray([.1, -.1]))
    native = (np.asarray([.2, .4]), np.asarray([.2, 0.]))
    fit, error = rescue.fit_response_gain(raw, target, native, groups, .5, [.25, 4.], 1e-16)
    output = _clipped(raw, {"action_limit": .5}, fit)
    assert error is None
    assert np.mean(output[1]) == pytest.approx(.1, abs=1e-10)
    diagnostic = rescue.waveform_diagnostics(output, target, native, raw, groups, 1e-16)
    assert diagnostic["response_error_rms"] > .01
    assert diagnostic["sham_mean_change_from_target"] == pytest.approx(.1, abs=1e-10)


@pytest.mark.parametrize("target_sign, expected_gain", [(20., 4.), (-1., .25)])
def test_gain_bounds_declared_and_reported(target_sign, expected_gain):
    groups = np.asarray(["a", "a"])
    source = (np.asarray([.01, -.01]), np.zeros(2))
    target = (target_sign * source[0], source[1])
    fit, _ = rescue.fit_response_gain(source, target, target, groups, .5, [.25, 4.], 1e-16)
    assert fit["gain"] == expected_gain
    assert fit["gain_at_bound"]
    assert fit["unconstrained_gain"] == pytest.approx(target_sign)


def test_degenerate_responses_remain_unavailable():
    groups = np.asarray(["a", "a"])
    zero = (np.zeros(2), np.zeros(2))
    response = (np.ones(2) * .01, np.zeros(2))
    fit, error = rescue.fit_response_gain(zero, response, response, groups, .5, [.25, 4.], 1e-16)
    assert fit is None and error == "source_response_below_energy_floor"
    fit, error = rescue.fit_response_gain(response, zero, zero, groups, .5, [.25, 4.], 1e-16)
    assert fit is None and error == "target_response_below_energy_floor"
    diagnostic = rescue.waveform_diagnostics(zero, zero, zero, zero, groups, 1e-16)
    assert diagnostic["relative_response_error"] is None
    assert diagnostic["residual_over_intervention"] is None
    assert diagnostic["response_cosine"] is None
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("delay", [0., .025, .05, .075, .1])
def test_prepare_confirm_keep_inherited_settings_and_weights(settings, model, delay):
    base, protocol = settings
    old = inherited(base)
    original = deepcopy(old)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    prepared = rescue.prepare_rescue(base, protocol, .2, .05, 11, model, old, delay)
    assert old == original
    assert prepared["physical_rollouts"] == 3
    assert prepared["status"] == "prepared"
    assert set(prepared["variants"]) == set(rescue.INHERITED_VARIANTS + rescue.CONTROL_VARIANTS)
    for name in rescue.INHERITED_VARIANTS:
        assert prepared["variants"][name] == original["variants"][name]
    for name in rescue.CONTROL_VARIANTS:
        assert prepared["variants"][name]["calibration_delay"] == delay
        assert abs(prepared["calibration"]["fixed_history"]["variants"][name]["waveform"]["sham_mean_change_from_native"]) <= 1e-10
    frozen = deepcopy(prepared)
    confirmed = rescue.confirm_rescue(base, protocol, .2, .05, 11, model, prepared, delay)
    assert prepared == frozen
    assert confirmed["physical_rollouts"] == 15
    assert len(confirmed["rollouts"]) == 10
    assert confirmed["frozen_settings"] == frozen["variants"]
    assert confirmed["prepared_sha256"] == _digest(frozen)
    assert set(row["noise_seed"] for row in prepared["calibration"]["rollouts"]) == {5010001}
    assert set(row["noise_seed"] for row in confirmed["rollouts"]) == {5030001}
    assert all(row["action_lag_min"] == pytest.approx(delay) for row in confirmed["timing"])
    assert all(value["calibration_delay"] == delay for value in confirmed["summary"].values())
    assert confirmed["fixed_history"]["native_replay_passed"]
    assert confirmed["fixed_history"]["bank_sha256"] != prepared["calibration"]["fixed_history"]["bank_sha256"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0., atol=0.)
    json.dumps(prepared, allow_nan=False)
    json.dumps(confirmed, allow_nan=False)


def test_zero_intervention_rescue_is_identity(settings, model):
    base, protocol = settings
    old = inherited(base)
    old["group"] = []
    for name in rescue.INHERITED_VARIANTS:
        old["variants"][name] = deepcopy(old["variants"]["native"])
    prepared = rescue.prepare_rescue(base, protocol, .2, .05, 11, model, old, .075)
    for name in rescue.CONTROL_VARIANTS:
        assert prepared["variants"][name]["gain"] == 1.
        assert prepared["variants"][name]["offset"] == 0.
    confirmed = rescue.confirm_rescue(base, protocol, .2, .05, 11, model, prepared, .075)
    for summary in confirmed["summary"].values():
        assert summary["delta_from_native"]["sham"]["position_rms"] == 0.
        assert summary["delta_from_native"]["paired_recovery"]["normalized_position_energy"] == 0.


def test_censored_calibration_keeps_original_variants_and_null_controls(settings, model):
    base, protocol = settings
    base["evaluation"]["max_abs_state"] = 1e-8
    prepared = rescue.prepare_rescue(base, protocol, .2, .05, 11, model, inherited(base), .075)
    assert prepared["status"] == "calibration_censored"
    assert prepared["physical_rollouts"] == 3
    assert prepared["calibration"]["fixed_history"]["available"] is False
    assert set(prepared["variants"]) == set(rescue.INHERITED_VARIANTS)
    assert set(prepared["unavailable_controls"]) == set(rescue.CONTROL_VARIANTS)
    confirmed = rescue.confirm_rescue(base, protocol, .2, .05, 11, model, prepared, .075)
    assert confirmed["physical_rollouts"] == 9
    for name in rescue.INHERITED_VARIANTS:
        assert confirmed["summary"][name]["sham"]["censored"] == 1.
        assert confirmed["summary"][name]["sham"]["position_rms"] is None
        assert confirmed["summary"][name]["paired_recovery"]["normalized_position_energy"] is None
    for name in rescue.CONTROL_VARIANTS:
        assert confirmed["summary"][name] is None
        assert confirmed["fixed_history"]["variants"][name] is None
    json.dumps(confirmed, allow_nan=False)


def test_complete_task_failure_retains_metrics_and_calibration(settings, model):
    base, protocol = settings
    base["evaluation"]["failure_rms"] = 1e-12
    prepared = rescue.prepare_rescue(base, protocol, .2, .05, 11, model, inherited(base), .075)
    assert prepared["status"] == "prepared"
    row = prepared["calibration"]["summary"]["native"]["sham"]
    assert row["task_failure"] == 1. and row["censored"] == 0.
    assert row["position_rms"] is not None


def test_preparation_binds_cell_delay_protocol(settings, model):
    base, protocol = settings
    prepared = rescue.prepare_rescue(base, protocol, .2, .05, 11, model, inherited(base), .075)
    with pytest.raises(ValueError, match="Prepared rescue"):
        rescue.confirm_rescue(base, protocol, .2, .05, 11, model, prepared, .05)
    with pytest.raises(ValueError, match="Prepared rescue"):
        rescue.confirm_rescue(base, {**protocol, "confirmation_seeds": [1]}, .2, .05, 11, model, prepared, .075)
    with pytest.raises(ValueError, match="Inherited preparation"):
        rescue.prepare_rescue(base, protocol, .2, .05, 22, model, inherited(base), .075)


@pytest.mark.parametrize("change", [{"calibration_seeds": [5030001]}, {"calibration_seeds": []},
                                     {"confirmation_seeds": [1, 1]}, {"confirmation_seeds": [True]},
                                     {"gain_bounds": [0., 4.]}, {"gain_bounds": [4., .25]},
                                     {"response_energy_floor": 0.}])
def test_invalid_protocol_rejected(settings, model, change):
    base, protocol = settings
    with pytest.raises(ValueError):
        rescue.prepare_rescue(base, {**protocol, **change}, .2, .05, 11, model, inherited(base), .075)
