"""Scientific invariants for the two-timescale suppression assays."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from closed_loop_inhibition.neural import make_model
from closed_loop_inhibition.timescale_diagnostics import (
    _bank, _calibrate, _head_assay, _match, choose_candidate, confirm_model, paired_recovery,
    prepare_model, response_summary,
)
from closed_loop_inhibition.timescale_maps import make_plant, teacher_policy


@pytest.fixture
def config():
    return {
        "period": .05, "delay": .05, "zeta": .15, "noise_dt": .0025,
        "noise_std": .02, "state_scale": .1, "output_scale": .1,
        "action_limit": .5, "max_tokens": 3, "history_seconds": .1,
        "evaluation": {"duration": .3, "burn_in": .1,
                       "max_abs_state": 100, "failure_rms": .1},
        "diagnostics": {
            "duration": .3, "pulse_onset": .1, "pulse_width": .05,
            "pulse_amplitudes": [-.02, .02], "common_amplitudes": [-.02, -.01, .01, .02],
            "discovery_seeds": [910001], "calibration_seeds": [920001],
            "confirmation_seeds": [930001, 930002], "weak_scale": .9, "strong_scale": 1.1,
            "min_fractional_increase": .01, "min_positive_fraction": .75,
            "baseline_epsilon": 1e-4, "match_tolerance": .05,
            "gain_grid": {"min": 1, "max": 1.5, "step": .1},
            "alternative_grid": {"min": .5, "max": 1.5, "step": .5},
        },
    }


def test_response_uses_physical_floor_and_absolute_effect(config):
    summary = response_summary(
        ([1e-5, 1e-5, .01, .01], [0, 0, 0, 0]),
        ([2e-5, 2e-5, .012, .012], [0, 0, 0, 0]),
        ["small", "small", "large", "large"], [.01, .01, .02, .02],
        config["diagnostics"],
    )
    assert not summary["eligible"]
    assert summary["median_fractional_increase"] == pytest.approx(.2)
    small = next(row for row in summary["per_group"] if row["group"] == "small")
    assert small["fractional_increase"] is None
    assert small["absolute_response_change"] == pytest.approx(1e-5)
    assert not small["baseline_valid"]


def test_constant_offset_is_not_a_paired_response_change(config):
    summary = response_summary(([.01, -.01], [0, 0]),
                               ([.21, .19], [.2, .2]), ["a", "a"], [.02, .02],
                               config["diagnostics"])
    assert summary["median_fractional_increase"] == pytest.approx(0, abs=1e-12)
    assert not summary["eligible"]


def test_unequal_group_lengths_have_equal_scenario_weight(config):
    native = ([.01, .01, .01, .01], [0, 0, 0, 0])
    changed = ([.02, .02, .02, .005], [0, 0, 0, 0])
    summary = response_summary(native, changed, ["a", "a", "a", "b"],
                               [.02] * 4, config["diagnostics"])
    assert summary["median_fractional_increase"] == pytest.approx(.25)
    assert summary["positive_fraction"] == .5
    assert not summary["eligible"]


def test_candidate_selection_uses_only_eligible_heads_and_stable_ties():
    rows = [{"layer": 0, "head": 0, "eligible": False, "median_fractional_increase": 100},
            {"layer": 1, "head": 2, "eligible": True, "median_fractional_increase": .2},
            {"layer": 1, "head": 1, "eligible": True, "median_fractional_increase": .2}]
    assert choose_candidate(rows)["head"] == 1
    assert choose_candidate(rows[:1]) is None


def test_unresponsive_checkpoint_is_unclassifiable(config):
    model = make_model("transformer", input_dim=10, max_tokens=3, width=8)
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()
    bank = {"pulse": (np.ones((2, 3, 10), dtype=np.float32), np.ones((2, 3), dtype=bool)),
            "sham": (np.zeros((2, 3, 10), dtype=np.float32), np.ones((2, 3), dtype=bool)),
            "groups": np.array(["a", "a"]), "amplitudes": np.array([.02, .02])}
    result = _head_assay(model, bank, config)
    assert not result["classifiable"]
    assert result["eligible_fraction"] is None
    assert result["eligible_count"] is None
    assert result["selected"] is None


def test_common_teacher_bank_identical_across_noise_timescales(config):
    config["noise_std"] = 0
    teacher = teacher_policy(config, make_plant(config, .1))
    first, failure = _bank(config, .1, .01, teacher, [123], [-.02, .02])
    second, second_failure = _bank(config, .1, 1., teacher, [123], [-.02, .02])
    assert failure is None and second_failure is None
    for arm in ("pulse", "sham"):
        np.testing.assert_array_equal(first[arm][0], second[arm][0])
        np.testing.assert_array_equal(first[arm][1], second[arm][1])
    assert not np.array_equal(first["pulse"][0], first["sham"][0])


def test_unmatched_control_is_retained_and_flagged(config):
    rows = [{"parameter": 1., "achieved_rms": 0., "offset": 0.},
            {"parameter": 1.5, "achieved_rms": .001, "offset": .02}]
    result = _match(rows, .01, config["diagnostics"])
    assert result["parameter"] == 1.5
    assert not result["matched"]
    assert result["relative_error"] == pytest.approx(.9)
    assert result["offset"] == .02


def test_censored_recovery_never_reports_partial_window_success():
    run = SimpleNamespace(failure_reason="state_guard")
    result = paired_recovery(run, run, 1, .02)
    assert not result["complete"]
    assert result["normalized_position_energy"] is None


def test_paired_recovery_subtracts_own_background_and_integrates_held_actions():
    sham = SimpleNamespace(failure_reason=None, config=SimpleNamespace(duration=2.),
                           applied_actions=[SimpleNamespace(time=0., value=3.)], samples=[
        {"time": 0., "position": 5., "action": 3.},
        {"time": 1., "position": 5., "action": 3.},
        {"time": 2., "position": 5., "action": 3.},
    ])
    pulse = deepcopy(sham)
    for row in pulse.samples:
        row["position"] += 2
        row["action"] += 4
    pulse.applied_actions[0].value += 4
    result = paired_recovery(pulse, sham, 1, .5)
    assert result["incremental_position_rms"] == 2
    assert result["incremental_action_rms"] == 4
    assert result["normalized_position_energy"] == 16


def test_incomplete_recovery_cannot_look_successful_without_guard():
    run = SimpleNamespace(failure_reason=None, config=SimpleNamespace(duration=4.),
                          samples=[{"time": 0.}, {"time": 2.}])
    result = paired_recovery(run, run, 1., .02)
    assert not result["complete"]
    assert result["normalized_position_energy"] is None


def test_recovery_action_integral_includes_between_report_application():
    sham = SimpleNamespace(failure_reason=None, config=SimpleNamespace(duration=2.),
                           applied_actions=[], samples=[
                               {"time": 0., "position": 0.},
                               {"time": 2., "position": 0.}])
    pulse = deepcopy(sham)
    pulse.applied_actions = [SimpleNamespace(time=1.5, value=2.)]
    result = paired_recovery(pulse, sham, 1., .02)
    assert result["incremental_action_rms"] == pytest.approx(np.sqrt(2))


def test_preparation_never_accesses_confirmation_streams(config, tmp_path, monkeypatch):
    import closed_loop_inhibition.timescale_maps as maps
    model = make_model("transformer", input_dim=10, max_tokens=3, width=8)
    with torch.no_grad():
        model.readout.weight.zero_()
        model.readout.bias.zero_()
    for name, epoch in (("initial", 0), ("epoch_025", 25), ("selected", 25)):
        torch.save({"state_dict": model.state_dict(), "epoch": epoch}, tmp_path / f"{name}.pt")
    real_run = maps.run_episode
    used_seeds = []

    def tracked(*args, **kwargs):
        used_seeds.append(args[4])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(maps, "run_episode", tracked)
    prepared = prepare_model(config, .1, .05, 11, model, tmp_path)
    assert not set(used_seeds) & set(config["diagnostics"]["confirmation_seeds"])
    assert prepared["discovery"]["selected"] is None
    assert prepared["calibration"] is None
    used_seeds.clear()
    confirmed = confirm_model(config, .1, .05, 11, model, prepared)
    assert set(used_seeds) == set(config["diagnostics"]["confirmation_seeds"])
    assert set(confirmed["summary"]) == {"native"}
    assert confirmed["fixed_history"] is None
    assert confirmed["summary"]["native"]["independent_noise_seeds"] == 2


def test_overlapping_discovery_and_confirmation_streams_rejected(config):
    config["diagnostics"]["confirmation_seeds"] = config["diagnostics"]["discovery_seeds"]
    with pytest.raises(ValueError, match="disjoint"):
        _bank(config, .1, .05, lambda _: 0, [1], [.02])


def test_calibrated_variants_run_and_keep_frozen_mean_offsets(config):
    torch.manual_seed(148)
    model = make_model("transformer", input_dim=10, max_tokens=3, width=8)
    teacher = teacher_policy(config, make_plant(config, .1))
    bank, failure = _bank(config, .1, .05, teacher, [920001], [-.02, .02])
    assert failure is None
    candidate = {"layer": 0, "head": 0}
    calibration = {"available": True, "failure": None,
                   **_calibrate(model, bank, config, candidate)}
    for name, variant in calibration["variants"].items():
        if name != "native":
            assert abs(variant["mean_error"]) <= 1e-10
    prepared = {"tau": .1, "noise_tau": .05, "seed": 148,
                "discovery": {"selected": candidate}, "calibration": calibration}
    before = json.dumps(prepared, sort_keys=True, allow_nan=False)
    confirmed = confirm_model(config, .1, .05, 148, model, prepared)
    assert before == json.dumps(prepared, sort_keys=True, allow_nan=False)
    assert set(confirmed["summary"]) == {
        "native", "centered_weak", "centered_strong", "gain_matched", "alternative_matched"}
    assert confirmed["fixed_history"]["available"]
    assert len(confirmed["rollouts"]) == 20
    json.dumps(confirmed, allow_nan=False)
