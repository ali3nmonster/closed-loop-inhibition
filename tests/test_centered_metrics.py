"""Time-weighted operating-point measurements and censoring."""

import numpy as np
import pytest

from closed_loop_inhibition.centered_metrics import mean_applied_action, mean_position
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.records import AppliedAction
from closed_loop_inhibition.timing import SimulationResult, TimingConfig, simulate


def _changing_actions():
    values = (1.0, -2.0, 4.0, 7.0)
    return simulate(
        Oscillator(), lambda snapshot: values[round(snapshot.time / 0.3)],
        TimingConfig(duration=1.0, observation_interval=0.3,
                     decision_interval=0.3, schedule="fixed_cadence",
                     compute_duration=0.2, sample_interval=0.37),
    )


def test_action_mean_weights_actual_held_intervals_and_initial_zero():
    run = _changing_actions()
    # Zero until .2, then 1 until .5, -2 until .8, and 4 until 1.
    assert [action.time for action in run.applied_actions] == [0.0, 0.2, 0.5, 0.8]
    assert mean_applied_action(run) == pytest.approx(0.3 - 0.6 + 0.8)
    assert mean_applied_action(run) != pytest.approx(
        np.mean([action.value for action in run.applied_actions])
    )


def test_action_mean_uses_clipped_applied_values_not_policy_requests():
    run = simulate(
        Oscillator(), lambda _: -100.0,
        TimingConfig(duration=1.0, compute_duration=0.2,
                     action_limit=2.0, sample_interval=0.3),
    )
    assert mean_applied_action(run) == pytest.approx(-2.0 * 0.8)


def test_commands_at_or_after_horizon_do_not_contribute():
    run = _changing_actions()
    before = mean_applied_action(run)
    run.applied_actions += [AppliedAction(1.0, -30.0, 10),
                            AppliedAction(1.2, 1000.0, 11)]
    assert mean_applied_action(run) == before


def test_same_time_commands_have_zero_duration_until_last_value():
    run = simulate(Oscillator(), lambda _: 0.0, TimingConfig(duration=1.0))
    run.applied_actions = [AppliedAction(0.0, 0.0, None),
                           AppliedAction(0.2, 50.0, 1),
                           AppliedAction(0.2, -2.0, 2),
                           AppliedAction(0.5, 1.0, 3)]
    assert mean_applied_action(run) == pytest.approx(-2.0 * 0.3 + 1.0 * 0.5)


def test_initial_zero_is_used_even_without_an_explicit_action_record():
    run = simulate(Oscillator(), lambda _: 0.0, TimingConfig(duration=1.0))
    run.applied_actions = []
    assert mean_applied_action(run) == 0.0


def test_constant_equilibrium_position_has_its_signed_mean():
    run = simulate(
        Oscillator(), lambda _: -2.0,
        TimingConfig(duration=1.0, compute_duration=0.0, sample_interval=0.3),
        initial_state=(-2.0, 0.0),
    )
    assert mean_position(run) == pytest.approx(-2.0)


def test_linear_position_on_nonuniform_grid_integrates_exactly():
    times = [0.0, 0.1, 0.7, 1.0]
    samples = [{"time": time, "position": 3 * time - 1, "velocity": 3.0,
                "reference": 0.0} for time in times]
    run = SimulationResult(samples, [], [], [], TimingConfig(duration=1.0))
    assert mean_position(run) == pytest.approx(0.5)
    assert mean_position(run) != pytest.approx(np.mean([3 * time - 1 for time in times]))


@pytest.mark.parametrize("metric", [mean_applied_action, mean_position])
def test_failed_run_is_censored(metric):
    run = simulate(Oscillator(), lambda _: 10.0,
                   TimingConfig(duration=1.0, compute_duration=0.0, max_abs_state=0.1))
    assert run.failure_reason is not None
    assert metric(run) is None


@pytest.mark.parametrize("metric", [mean_applied_action, mean_position])
@pytest.mark.parametrize("coverage", ["empty", "late_start", "early_end"])
def test_unmarked_incomplete_reporting_is_censored(metric, coverage):
    run = _changing_actions()
    if coverage == "empty":
        run.samples = []
    elif coverage == "late_start":
        run.samples = run.samples[1:]
    else:
        run.samples = run.samples[:-1]
    assert metric(run) is None


@pytest.mark.parametrize("metric", [mean_applied_action, mean_position])
def test_invalid_action_records_use_existing_validation(metric):
    run = _changing_actions()
    run.applied_actions.append(AppliedAction(0.1, 1.0, 20))
    with pytest.raises(ValueError, match="ordered"):
        metric(run)
