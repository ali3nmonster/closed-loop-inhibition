"""Check held-action accounting, sampled settling, censoring and convergence."""

import json

import numpy as np
import pytest
from scipy.integrate import quad

from closed_loop_inhibition.metrics import summarize
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.records import AppliedAction
from closed_loop_inhibition.signals import PiecewiseConstant
from closed_loop_inhibition.timing import SimulationResult, TimingConfig, simulate


def test_unit_tracking_error_has_unit_ise_and_rmse():
    result = simulate(Oscillator(), lambda _: 0.0, TimingConfig(duration=1.0),
                      reference=PiecewiseConstant(initial=1.0))
    metrics = summarize(result)
    assert metrics["tracking_ise"] == pytest.approx(1.0)
    assert metrics["tracking_rmse"] == pytest.approx(1.0)
    assert metrics["peak_abs_error"] == pytest.approx(1.0)
    assert metrics["action_effort"] == 0
    assert metrics["settling_time"] is None
    assert metrics["censored"] is False
    json.dumps(metrics, allow_nan=False)


def test_action_effort_uses_application_time_not_reporting_interpolation():
    result = simulate(Oscillator(), lambda _: 2.0,
                      TimingConfig(duration=1.0, compute_duration=0.1, sample_interval=0.3))
    metrics = summarize(result)
    assert metrics["action_effort"] == pytest.approx(4 * 0.9)
    assert metrics["action_variation"] == pytest.approx(2.0)
    assert metrics["max_abs_action"] == 2.0


def _recorded_result(errors, times=None, failure_reason=None):
    times = np.linspace(0, 1, len(errors)) if times is None else times
    samples = [{"time": float(t), "position": float(error), "velocity": 0.0,
                "reference": 0.0, "action": 0.0, "disturbance": 0.0}
               for t, error in zip(times, errors)]
    return SimulationResult(samples, [], [], [AppliedAction(0.0, 0.0, None)],
                            TimingConfig(duration=1.0), failure_reason)


def test_action_variation_counts_all_applied_jumps_and_effort_uses_held_intervals():
    result = _recorded_result([0, 0])
    result.applied_actions += [AppliedAction(0.0, 2.0, 0), AppliedAction(0.2, -1.0, 1),
                               AppliedAction(0.2, 3.0, 2), AppliedAction(1.0, 1.0, 3)]
    metrics = summarize(result)
    assert metrics["action_effort"] == pytest.approx(4 * 0.2 + 9 * 0.8)
    assert metrics["action_variation"] == pytest.approx(2 + 3 + 4 + 2)
    assert metrics["max_abs_action"] == 3.0


def test_settling_requires_entire_remaining_suffix_after_last_change():
    result = _recorded_result([0.1, 0.01, 0.03, 0.01, 0.0])
    metrics = summarize(result, last_change_time=0.2)
    assert metrics["settling_time"] == pytest.approx(0.75 - 0.2)
    assert metrics["settling_observed_for"] == pytest.approx(0.25)
    assert metrics["remaining_observation_horizon"] == pytest.approx(0.8)
    assert metrics["horizon_edge_settling"] is False


def test_final_sample_only_settling_is_flagged_and_future_changes_are_not_scored():
    result = _recorded_result([0.1, 0.1, 0])
    metrics = summarize(result, last_change_time=0.2)
    assert metrics["settling_time"] == pytest.approx(0.8)
    assert metrics["settling_observed_for"] == 0.0
    assert metrics["horizon_edge_settling"] is True
    future = summarize(result, last_change_time=1.1)
    assert future["settling_time"] is None
    assert future["remaining_observation_horizon"] == 0.0
    assert future["horizon_edge_settling"] is False


def test_divergence_guard_censors_scores_instead_of_rewarding_truncation():
    result = simulate(Oscillator(), lambda _: 10.0,
                      TimingConfig(duration=1.0, compute_duration=0.0, max_abs_state=0.1))
    assert result.failure_reason is not None
    metrics = summarize(result)
    assert metrics["censored"] is True
    assert metrics["failure_reason"] == result.failure_reason
    assert 0 <= metrics["observed_duration"] < 1.0
    for name in ("tracking_ise", "tracking_rmse", "action_effort", "action_variation", "settling_time"):
        assert metrics[name] is None
    json.dumps(metrics, allow_nan=False)


def test_failure_before_first_sample_and_unmarked_incomplete_horizon_are_censored():
    empty = _recorded_result([], failure_reason="initial_state_guard")
    assert summarize(empty)["observed_duration"] == 0.0
    assert summarize(empty)["tracking_ise"] is None
    incomplete = _recorded_result([1, 1], times=[0, 0.5])
    metrics = summarize(incomplete)
    assert metrics["censored"] is True
    assert metrics["failure_reason"] == "incomplete_reporting_horizon"
    assert metrics["tracking_ise"] is None


def test_tracking_quadrature_converges_without_changing_controller_timing():
    plant = Oscillator(tau=0.5, zeta=0.15)
    initial = (0.8, -0.2)
    exact_ise = quad(lambda t: plant.propagate(initial, t)[0] ** 2, 0, 1, epsabs=1e-12)[0]
    errors = []
    jobs = []
    for interval in [0.1, 0.05, 0.025]:
        result = simulate(plant, lambda _: 0.0,
                          TimingConfig(duration=1.0, compute_duration=0.07, sample_interval=interval),
                          initial_state=initial)
        errors.append(abs(summarize(result)["tracking_ise"] - exact_ise))
        jobs.append(result.jobs)
    assert errors[1] < errors[0] / 3
    assert errors[2] < errors[1] / 3
    assert jobs[0] == jobs[1] == jobs[2]


@pytest.mark.parametrize("kwargs", [{"last_change_time": -1}, {"last_change_time": np.inf},
                                   {"settling_tolerance": -1}, {"settling_tolerance": np.nan}])
def test_invalid_settling_settings_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        summarize(_recorded_result([0, 0]), **kwargs)
