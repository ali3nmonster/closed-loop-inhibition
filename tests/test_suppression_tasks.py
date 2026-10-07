"""Check paired intervention controls, recovery windows, and causal probes."""

from dataclasses import replace
import json

import numpy as np
import pytest

from closed_loop_inhibition.controllers import ClassicalPDController
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.records import AppliedAction
from closed_loop_inhibition.suppression_tasks import (
    collect_paired_probes, make_scenarios, paired_rollouts, score_pair,
)
from closed_loop_inhibition.timing import SimulationResult, TimingConfig


def scenario(amplitude=0.4):
    return {"id": "example", "initial_state": [0.0, 0.0], "reference": 0.0,
            "pulse_start": 1.0, "pulse_duration": 0.25, "amplitude": amplitude, "duration": 5.0}


def recorded(positions, times=None, duration=5.0):
    times = np.linspace(0, duration, len(positions)) if times is None else times
    samples = [{"time": float(t), "position": float(q), "velocity": 0.0, "reference": 0.0,
                "action": 0.0, "disturbance": 0.0} for t, q in zip(times, positions)]
    return SimulationResult(samples, [], [], [AppliedAction(0.0, 0.0, None)], TimingConfig(duration=duration))


def test_scenario_generation_is_reproducible_and_independent_across_seeded_splits():
    first = make_scenarios(12, 101, "discovery")
    assert first == make_scenarios(12, 101, "discovery")
    second = make_scenarios(12, 103, "confirmation")
    assert not {s["id"] for s in first} & {s["id"] for s in second}
    signature = lambda s: json.dumps({k: v for k, v in s.items() if k != "id"}, sort_keys=True)
    assert not {signature(s) for s in first} & {signature(s) for s in second}
    for item in first + second:
        assert 0.3 <= abs(item["amplitude"]) <= 0.6
        assert item["pulse_start"] * 20 == pytest.approx(round(item["pulse_start"] * 20))
        assert 1.0 <= item["pulse_start"] <= 1.4


def test_pair_uses_the_same_active_intervention_for_pulse_and_sham():
    plant = Oscillator()
    timing = TimingConfig(duration=5.0, observation_interval=0.05, compute_duration=0.05, sample_interval=0.01)
    # The intervention introduces autonomous motion even without a pulse.
    native = ClassicalPDController()
    intervened = lambda s: native(s) + 0.3
    pulse, sham = paired_rollouts(plant, intervened, scenario(), timing)
    _, native_sham = paired_rollouts(plant, native, scenario(), timing)
    at = np.array([s["time"] for s in sham.samples])
    before = at < 1.0
    pulse_q = np.array([s["position"] for s in pulse.samples])
    sham_q = np.array([s["position"] for s in sham.samples])
    np.testing.assert_array_equal(pulse_q[before], sham_q[before])
    assert np.max(np.abs(sham_q)) > 0.05
    correct = score_pair(pulse, sham, scenario())
    wrong = score_pair(pulse, native_sham, scenario())
    assert abs(correct["J_response"] - wrong["J_response"]) > 0.01
    assert correct["sham_last_second_rmse"] > 0.01


def test_normalized_pair_scores_remove_common_drift_but_report_absolute_behavior():
    sham = recorded([1.0] * 6)
    pulse = recorded([1.4] * 6)
    pulse.applied_actions.append(AppliedAction(0.2, 2.0, 0))
    score = score_pair(pulse, sham, scenario())
    assert score["J_response"] == pytest.approx(3.0)
    assert score["peak_response"] == pytest.approx(1.0)
    assert score["pulse_action_effort"] == pytest.approx(4 * 4.8)
    assert score["pulse_action_variation"] == pytest.approx(2.0)
    assert score["sham_action_variation"] == 0.0
    assert score["sham_tracking_rmse"] == pytest.approx(1.0)
    assert score["pulse_tracking_rmse"] == pytest.approx(1.4)
    assert score["sham_peak_abs_error"] == pytest.approx(1.0)
    assert score["pulse_last_second_rmse"] == pytest.approx(1.4)
    negative = score_pair(recorded([0.2] * 6), sham, scenario(amplitude=-0.8))
    assert negative["J_response"] == pytest.approx(score["J_response"])
    assert negative["peak_response"] == pytest.approx(score["peak_response"])
    json.dumps(score, allow_nan=False)


def test_recovery_bounds_interpolate_position_before_squaring():
    pulse = recorded([0.0, 0.0, 4.0, 4.0, 4.0, 4.0])
    sham = recorded([0.0] * 6)
    score = score_pair(pulse, sham, scenario(amplitude=1.0))
    # Recovery starts at 1.25: interpolated position is 1; it reaches 4 at 2.
    # Trapezoidal integration is .75*(1^2+4^2)/2 + 2.25*4^2.
    assert score["J_response"] == pytest.approx(42.375)
    assert score["peak_response"] == pytest.approx(4.0)
    assert score["recovery_start"] == 1.25
    assert score["recovery_end"] == 4.25


@pytest.mark.parametrize("case", ["failure", "truncated", "short_recovery", "duration_mismatch"])
def test_incomplete_pairs_have_null_scores(case):
    pulse, sham = recorded([0, 1]), recorded([0, 0])
    setting = scenario()
    if case == "failure":
        pulse.failure_reason = "state_guard"
    elif case == "truncated":
        pulse.samples[-1]["time"] = 4.0
    elif case == "short_recovery":
        setting["pulse_start"] = 2.0
    else:
        pulse.config = replace(pulse.config, duration=6.0)
    score = score_pair(pulse, sham, setting)
    assert score["censored"] is True
    assert score["failure_reason"]
    assert all(score[key] is None for key in ("J_response", "peak_response", "sham_tracking_rmse", "pulse_action_effort",
                                            "pulse_action_variation", "sham_action_variation"))
    json.dumps(score, allow_nan=False)


def test_probe_pairs_use_available_snapshots_not_unobserved_pulse_state():
    timing = TimingConfig(duration=5.0, observation_interval=0.05, sensor_delay=0.2,
                          compute_duration=0.05, schedule="fixed_cadence", decision_interval=0.05,
                          history_seconds=0.5, sample_interval=0.05, action_limit=5.0)
    seen = []

    def policy(snapshot):
        assert snapshot.latest.capture_time <= snapshot.time - 0.2 + 1e-9
        assert snapshot.latest.available_time <= snapshot.time
        seen.append(snapshot.time)
        return 0.0

    encoding = {"max_tokens": 11, "history_seconds": 0.5, "action_limit": 5.0}
    probes = collect_paired_probes(Oscillator(), policy, scenario(), timing, encoding, window_seconds=0.15)
    np.testing.assert_allclose(probes["times"], [1.0, 1.05, 1.1, 1.15])
    np.testing.assert_array_equal(probes["pulse_tokens"], probes["sham_tokens"])
    assert probes["pulse_tokens"].shape == (4, 11, 10)
    assert probes["valid"].dtype == np.bool_
    assert len(seen) == 194  # Both arms, dispatches from .2 through 5 seconds.


def test_probe_collection_rejects_censoring_or_incomplete_window():
    timing = TimingConfig(duration=5.0, max_abs_state=0.01)
    with pytest.raises(ValueError, match="censored"):
        collect_paired_probes(Oscillator(), lambda _: 1.0, scenario(), timing, {})
    with pytest.raises(ValueError, match="window"):
        collect_paired_probes(Oscillator(), lambda _: 0.0, scenario(), timing, {}, window_seconds=5.0)
    with pytest.raises(ValueError, match="duration"):
        paired_rollouts(Oscillator(), lambda _: 0.0, scenario(), replace(timing, duration=4.0))
