"""Structured forcing has its own clock and measurement noise stays causal."""

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.records import AppliedAction, Observation, PolicyInput
from closed_loop_inhibition.structured_signals import (
    MeasurementNoisePolicy,
    SampledSignal,
    ou_tape,
    rectangular_pulse,
    sinusoid_tape,
)
from closed_loop_inhibition.timing import TimingConfig, simulate


def test_ou_exact_stationary_transition_and_seeded_replay():
    kwargs = dict(duration=0.02, dt=0.0025, correlation_time=0.05, std=0.02, seed=19)
    tape = ou_tape(**kwargs)
    draws = np.random.default_rng(19).standard_normal(len(tape.values))
    expected = [0.02 * draws[0]]
    decay = math.exp(-0.0025 / 0.05)
    innovation_std = 0.02 * math.sqrt(1 - decay**2)
    for draw in draws[1:]:
        expected.append(decay * expected[-1] + innovation_std * draw)
    np.testing.assert_allclose(tape.values, expected, rtol=1e-14, atol=1e-17)
    assert tape == ou_tape(**kwargs)
    assert tape != ou_tape(**{**kwargs, "seed": 20})
    # The actual sample mean is retained rather than forcibly removed.
    assert np.mean(tape.values) != 0.0
    assert np.sqrt(np.mean(np.square(tape.values))) != 0.02


def test_long_ou_tape_has_requested_stationary_variance_and_correlation():
    tape = ou_tape(duration=200.0, dt=0.0025, correlation_time=0.05, std=0.02, seed=827)
    values = np.asarray(tape.values)
    assert abs(np.mean(values)) < 0.002
    assert np.std(values) == pytest.approx(0.02, rel=0.08)
    lag = 20  # One requested correlation time.
    correlation = np.corrcoef(values[:-lag], values[lag:])[0, 1]
    assert correlation == pytest.approx(math.exp(-1), abs=0.04)


def test_signal_clock_is_integer_aligned_and_holds_final_value():
    tape = ou_tape(duration=0.009, dt=0.0025, correlation_time=0.01, std=0.02, seed=8)
    assert tape.times == (0.0, 0.0025, 0.005, 0.0075)
    piecewise = tape.to_piecewise_constant()
    for time in [0.0, 0.002499999, 0.0025, 0.005, 0.0075, 0.009, 0.02]:
        assert tape.at(time) == piecewise.at(time)
    assert tape.at(0.002499999) == tape.values[0]
    assert tape.at(0.0025) == tape.values[1]
    assert tape.at(0.02) == tape.values[-1]


def test_signal_is_immutable_and_copies_mutable_arguments():
    times, values = [0, 0.05], [1.0, -1.0]
    tape = SampledSignal(times, values)
    times[1], values[1] = 0.1, 99.0
    assert tape.times == (0.0, 0.05)
    assert tape.values == (1.0, -1.0)
    with pytest.raises(FrozenInstanceError):
        tape.values = (4.0, 5.0)
    with pytest.raises(TypeError):
        tape.values[0] = 8.0


@pytest.mark.parametrize("times,values", [
    ([], []), ([0], []), ([0.1], [0]), ([0, 0], [0, 1]),
    ([0, -0.1], [0, 1]), ([0, 1.4e-9], [0, 1]),
    ([0, float("nan")], [0, 1]), ([0], [float("inf")]),
])
def test_invalid_sampled_signals_are_rejected(times, values):
    with pytest.raises(ValueError):
        SampledSignal(times, values)


@pytest.mark.parametrize("change", [
    {"duration": 0}, {"dt": 0}, {"dt": 0.1e-9}, {"dt": 2.2e-9},
    {"correlation_time": 0}, {"correlation_time": float("nan")},
    {"std": -1}, {"std": float("inf")}, {"seed": -1}, {"seed": 1.5},
])
def test_invalid_ou_parameters_are_rejected(change):
    kwargs = dict(duration=1, dt=0.0025, correlation_time=0.05, std=0.02, seed=1)
    with pytest.raises(ValueError):
        ou_tape(**{**kwargs, **change})


def test_sinusoid_and_rectangular_pulse_have_specified_temporal_structure():
    tape = sinusoid_tape(duration=1, dt=0.125, frequency=1, amplitude=2, phase=math.pi / 2)
    np.testing.assert_allclose(tape.values, 2 * np.cos(2 * np.pi * np.asarray(tape.times)), atol=1e-15)
    assert tape.at(0.2) == tape.values[1]
    pulse = rectangular_pulse(onset=0.03, duration=0.04, amplitude=0.7)
    assert pulse.at(0.029999999) == 0
    assert pulse.at(0.03) == 0.7
    assert pulse.at(0.069999999) == 0.7
    assert pulse.at(0.07) == 0


class RecordingPolicy:
    def __init__(self):
        self.snapshots = []

    def __call__(self, snapshot):
        self.snapshots.append(snapshot)
        return 0.0


def test_measurement_noise_changes_only_position_at_capture_time():
    observation = Observation(0, 0.01, 0.04, 1.0, 2.0, 3.0, 4.0, 0.2)
    actions = (AppliedAction(0.0, 0.2, None),)
    snapshot = PolicyInput(0.05, 0.08, (observation,), actions)
    tape = SampledSignal((0.0, 0.02, 0.05), (0.1, 100.0, 200.0))
    inner = RecordingPolicy()
    wrapper = MeasurementNoisePolicy(inner, tape)
    assert wrapper(snapshot) == 0.0
    assert wrapper(replace(snapshot, time=0.09, planned_apply_time=0.1)) == 0.0
    for seen in inner.snapshots:
        assert seen.latest == replace(observation, position=1.1)
        assert seen.actions is actions
    assert inner.snapshots[0].time == 0.05
    assert inner.snapshots[0].planned_apply_time == 0.08
    assert snapshot.latest is observation
    assert snapshot.latest.position == 1.0
    with pytest.raises(FrozenInstanceError):
        inner.snapshots[0].latest.position = 8.0


@pytest.mark.parametrize("schedule", ["serial", "fixed_cadence"])
def test_measurement_noise_preserves_true_physics_and_repeated_observations(schedule):
    tape = ou_tape(duration=0.3, dt=0.0025, correlation_time=0.05, std=0.02, seed=10)
    config = TimingConfig(duration=0.3, observation_interval=0.025, decision_interval=0.01,
                          compute_duration=0.03, sensor_delay=0.015, schedule=schedule,
                          sample_interval=0.005)
    plant = Oscillator(tau=0.1, zeta=0.15)
    physical_force = rectangular_pulse(onset=0.0325, duration=0.0525, amplitude=0.4)
    clean, noisy = RecordingPolicy(), RecordingPolicy()
    original = simulate(plant, clean, config, initial_state=(0.1, 0.2), disturbance=physical_force)
    changed = simulate(plant, MeasurementNoisePolicy(noisy, tape), config,
                       initial_state=(0.1, 0.2), disturbance=physical_force)
    assert changed.samples == original.samples
    assert changed.applied_actions == original.applied_actions
    assert len(noisy.snapshots) == len(clean.snapshots)
    seen = {}
    for clean_snapshot, noisy_snapshot in zip(clean.snapshots, noisy.snapshots):
        for a, b in zip(clean_snapshot.observations, noisy_snapshot.observations):
            assert b == replace(a, position=a.position + tape.at(a.capture_time))
            if a.index in seen:
                assert b == seen[a.index]
            seen[a.index] = b


def test_physical_noise_tape_does_not_depend_on_controller_schedule():
    tape = ou_tape(duration=0.15, dt=0.0025, correlation_time=0.01, std=0.02, seed=17)
    results = []
    for schedule in ("serial", "fixed_cadence"):
        result = simulate(Oscillator(tau=0.05), RecordingPolicy(),
                          TimingConfig(duration=0.15, observation_interval=0.05,
                                       decision_interval=0.05, compute_duration=0.075,
                                       schedule=schedule, sample_interval=0.0025),
                          disturbance=tape.to_piecewise_constant())
        results.append(result)
        assert [sample["disturbance"] for sample in result.samples] == list(tape.values)
    np.testing.assert_allclose(
        [[row["position"], row["velocity"]] for row in results[0].samples],
        [[row["position"], row["velocity"]] for row in results[1].samples],
        atol=1e-14,
    )
