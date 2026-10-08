"""Checks that latency interventions preserve controllers and physical clocks."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm
import torch

import closed_loop_inhibition.delay_sweep as sweep
from closed_loop_inhibition.collective_dynamics import _digest, confirm_dynamics
from closed_loop_inhibition.neural import make_model
from closed_loop_inhibition.records import AppliedAction, Observation, PolicyInput
from closed_loop_inhibition.timescale_maps import NeuralPolicy, make_plant, run_episode
from closed_loop_inhibition.timing import TimingConfig, simulate


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings():
    base = json.loads((ROOT / "configs/timescale_maps.json").read_text())
    base["model"]["width"] = 8
    protocol = {"delay_cue": .05, "confirmation_seeds": [3010001],
                "probe": {"duration": .3, "pulse_onset": .1, "pulse_width": .05,
                          "pulse_amplitudes": [-.02, .02]},
                "baseline_epsilon": .0001, "min_fractional_increase": .01,
                "min_positive_fraction": .75, "match_tolerance": .05,
                "response_direction_epsilon": 1e-12}
    return base, protocol


@pytest.fixture
def model(settings):
    torch.set_num_threads(2)
    torch.manual_seed(381)
    return make_model("transformer", **settings[0]["model"]).eval()


def prepared(base):
    return {"tau": .2, "noise_tau": .05, "seed": 11, "group": ["L0H0"], "status": "prepared",
            "base_config_sha256": _digest(base), "unavailable_controls": {},
            "variants": {"native": {"branch_scales": {}, "offset": 0., "gain": 1., "center": 0.},
                         "joint_weak": {"branch_scales": {"L0H0": .9}, "offset": .001, "gain": 1., "center": 0., "achieved_rms": .002},
                         "joint_strong": {"branch_scales": {"L0H0": 1.1}, "offset": -.001, "gain": 1., "center": 0., "achieved_rms": .002},
                         "gain_matched": {"branch_scales": {}, "offset": -.001, "gain": 1.1, "center": .001,
                                          "matched": True, "magnitude_matched": True, "direction_matched": True,
                                          "matched_and_direction": True, "achieved_rms": .002, "relative_error": .01}}}


def test_fixed_policy_hides_only_explicit_delay_context(settings):
    base, protocol = settings
    class Spy(torch.nn.Module):
        def forward(self, tokens, valid):
            self.tokens = tokens.clone()
            return tokens[:, 0, 7]
    model = Spy()
    policy = sweep.FixedDelayPolicy(model, make_plant(base, .2), base, delay_cue=.05)
    snapshot = PolicyInput(.1, .125, (Observation(1, .1, .1, .01, .02, 0., 0., .03),),
                           (AppliedAction(.075, .04, 0),))
    first = policy(snapshot)
    first_tokens = model.tokens.clone()
    second = policy(replace(snapshot, planned_apply_time=.3))
    assert first == second
    torch.testing.assert_close(model.tokens, first_tokens, rtol=0., atol=0.)
    assert model.tokens[0, 0, 7] == .25
    assert model.tokens[0, 0, 9] == .125  # Actual action age is still true.
    policy(replace(snapshot, actions=(AppliedAction(.1, .04, 0),)))
    assert model.tokens[0, 0, 9] == 0.
    assert snapshot.planned_apply_time == .125


def test_recorded_token_clamp_preserves_padding_and_all_other_features():
    tokens = np.arange(60, dtype=np.float32).reshape(2, 3, 10)
    valid = np.asarray([[True, False, False], [True, True, False]])
    original = tokens.copy()
    changed = sweep.clamp_delay_tokens(tokens, valid, .2, .05)
    np.testing.assert_array_equal(tokens, original)
    mask = np.zeros_like(tokens, dtype=bool)
    mask[:, :, 7] = valid
    np.testing.assert_array_equal(changed[~mask], original[~mask])
    np.testing.assert_array_equal(changed[mask], .25)


@pytest.mark.parametrize("delay", [.025, .075, .2])
def test_fractional_and_multiperiod_feedback_match_independent_discretization(delay):
    tau, zeta, h, dt, duration = .2, .15, .05, .025, .4
    gain = np.asarray([1.2, .08])
    run = simulate(make_plant({"zeta": zeta}, tau),
                   lambda snapshot: -float(gain @ [snapshot.latest.position, snapshot.latest.velocity]),
                   TimingConfig(duration=duration, observation_interval=h, decision_interval=h,
                                schedule="fixed_cadence", compute_duration=delay, sample_interval=dt),
                   initial_state=(.01, -.02))
    # Independent exact discretization at half-period resolution. Commands are
    # generated from current states, scheduled, and held after their true delay.
    matrix = np.asarray([[0., 1., 0.], [-1 / tau**2, -2 * zeta / tau, 1 / tau**2], [0., 0., 0.]])
    discrete = expm(matrix * dt)
    state, action, pending = np.asarray([.01, -.02]), 0., {}
    expected_states, expected_actions = [], []
    for step in range(round(duration / dt) + 1):
        if step % round(h / dt) == 0:
            pending[step + round(delay / dt)] = -float(gain @ state)
        action = pending.get(step, action)
        expected_states.append(state.copy())
        expected_actions.append(action)
        state = discrete[:2, :2] @ state + discrete[:2, 2] * action
    np.testing.assert_allclose([[row["position"], row["velocity"]] for row in run.samples], expected_states, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose([row["action"] for row in run.samples], expected_actions, rtol=1e-11, atol=1e-12)
    audit = sweep.timing_audit(run, h, delay)
    assert audit["passed"] and audit["dispatch_count"] == 9
    assert audit["action_lag_min"] == pytest.approx(delay)
    assert audit["action_lag_max"] == pytest.approx(delay)


def test_clock_audit_accepts_censored_prefix_and_rejects_wrong_lag():
    run = simulate(make_plant({"zeta": .15}, .2), lambda snapshot: 1.,
                   TimingConfig(duration=.4, observation_interval=.05, decision_interval=.05,
                                schedule="fixed_cadence", compute_duration=.025, max_abs_state=.001))
    assert run.failure_reason
    assert sweep.timing_audit(run, .05, .025)["censored"]
    with pytest.raises(AssertionError, match="timing"):
        sweep.timing_audit(run, .05, .05)


def test_confirm_uses_frozen_settings_and_no_extra_probe_rollouts(settings, model, monkeypatch):
    base, protocol = settings
    inherited = prepared(base)
    original = deepcopy(inherited)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    actual_run_episode = sweep.run_episode
    calls = []
    def spy(*args, **kwargs):
        calls.append((args[4], kwargs.get("pulse"), kwargs["record"]))
        return actual_run_episode(*args, **kwargs)
    monkeypatch.setattr(sweep, "run_episode", spy)
    result = sweep.confirm_delay(base, protocol, .2, .05, 11, model, inherited, .075)
    assert inherited == original
    assert result["frozen_settings"] == original["variants"]
    assert result["physical_rollouts"] == len(calls) == 12
    assert sum(call[2] for call in calls) == 3  # Reuse native sham and both pulse arms.
    assert len(result["rollouts"]) == 8 and len(result["timing"]) == 12
    assert result["delay"] == .075 and result["delay_cue"] == .05
    fixed = result["fixed_history"]
    assert fixed["available"] and fixed["probe_groups"] == 2
    assert fixed["native_replay_max_abs_error"] < 1e-7
    assert fixed["variants"]["gain_matched"]["matched_at_calibration"] is True
    assert result["summary"]["gain_matched"]["matched_and_direction"] is True
    assert all(row["action_lag_min"] == pytest.approx(.075) for row in result["timing"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0., atol=0.)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("delay", [0., .075, .2])
def test_zero_intervention_preserves_own_loop_and_fixed_history_at_every_delay(settings, model, delay):
    base, protocol = settings
    inherited = prepared(base)
    inherited["group"] = []
    for label in inherited["variants"]:
        inherited["variants"][label] = deepcopy(inherited["variants"]["native"])
    result = sweep.confirm_delay(base, protocol, .2, .05, 11, model, inherited, delay)
    for summary in result["summary"].values():
        assert summary["delta_from_native"]["sham"]["position_rms"] == 0.
        assert summary["delta_from_native"]["paired_recovery"]["normalized_position_energy"] == 0.
    for row in result["fixed_history"]["variants"].values():
        assert row["immediate_output_rms"] == 0.
        assert row["sham_command_mean_change"] == 0.


def test_censored_native_bank_remains_unavailable_without_prefix_averaging(settings, model):
    base, protocol = settings
    base["evaluation"]["max_abs_state"] = 1e-8
    result = sweep.confirm_delay(base, protocol, .2, .05, 11, model, prepared(base), .075)
    assert result["fixed_history"]["available"] is False
    assert result["fixed_history"]["failure"]["reason"] == "native_rollout_censored"
    for summary in result["summary"].values():
        assert summary["sham"]["censored"] == 1.
        assert summary["sham"]["position_rms"] is None
        assert summary["paired_recovery"]["normalized_position_energy"] is None
    json.dumps(result, allow_nan=False)


def test_anchor_reproduces_existing_dynamics_with_same_streams(settings, model):
    base, protocol = settings
    old = json.loads((ROOT / "configs/collective_suppression.json").read_text())
    old["dynamics"].update(protocol["probe"], confirmation_seeds=protocol["confirmation_seeds"])
    inherited = prepared(base)
    inherited["protocol_sha256"] = _digest(old)
    before = confirm_dynamics(base, old, .2, .05, 11, model, inherited)
    after = sweep.confirm_delay(base, protocol, .2, .05, 11, model, inherited, .05)
    assert after["rollouts"] == before["rollouts"]
    for label in before["summary"]:
        for arm in ("sham", "pulse", "paired_recovery", "delta_from_native"):
            assert after["summary"][label][arm] == before["summary"][label][arm]


def test_zero_policy_reference_is_delay_independent_and_counts_unique_shams(settings):
    base, protocol = settings
    reference = sweep.passive_cell(base, protocol, .2, .05)
    assert reference["physical_rollouts"] == 3
    assert len(reference["rollouts"]) == 2 and len(reference["timing"]) == 3
    assert reference["summary"]["zero_policy"]["sham"]["action_rms"] == 0.
    assert reference["summary"]["zero_policy"]["independent_noise_seeds"] == 1
    for delay in (0., .075, .2):
        changed = {**base, "delay": delay}
        run = run_episode(changed, .2, .05, lambda snapshot: 0., 3010001, duration=.3)
        original = run_episode(base, .2, .05, lambda snapshot: 0., 3010001, duration=.3)
        for field in ("position", "velocity"):
            np.testing.assert_allclose([row[field] for row in run.samples], [row[field] for row in original.samples], atol=1e-12, rtol=1e-12)


def test_delay_cue_policy_equals_original_at_training_delay(settings, model):
    base, protocol = settings
    plant = make_plant(base, .2)
    native = NeuralPolicy(model, plant, base)
    clamped = sweep.FixedDelayPolicy(model, plant, base, delay_cue=protocol["delay_cue"])
    original = run_episode(base, .2, .05, native, 3010001, duration=.3)
    changed = run_episode(base, .2, .05, clamped, 3010001, duration=.3)
    assert original.jobs == changed.jobs
    assert original.samples == changed.samples


def test_invalid_preparation_cue_or_streams_are_rejected(settings, model):
    base, protocol = settings
    inherited = prepared(base)
    with pytest.raises(ValueError, match="model cell"):
        sweep.confirm_delay(base, protocol, .2, .05, 22, model, inherited, .05)
    with pytest.raises(ValueError, match="training delay"):
        sweep.confirm_delay(base, {**protocol, "delay_cue": .1}, .2, .05, 11, model, inherited, .05)
    with pytest.raises(ValueError, match="distinct"):
        sweep.confirm_delay(base, {**protocol, "confirmation_seeds": [1, 1]}, .2, .05, 11, model, inherited, .05)
    with pytest.raises(ValueError, match="nonnegative"):
        sweep.confirm_delay(base, protocol, .2, .05, 11, model, inherited, -.05)
