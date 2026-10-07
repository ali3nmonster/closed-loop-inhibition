"""Behavioral checks for causal, continuously evolving closed-loop simulations.

The reference trajectories below come from elementary oscillator solutions or
sampled-data control equations, independently of the simulator event machinery.
"""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.linalg import expm

from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.records import PolicyInput
from closed_loop_inhibition.signals import PiecewiseConstant
from closed_loop_inhibition.timing import TimingConfig, simulate


class RecordingPolicy:
    def __init__(self, action=0.0):
        self.inputs: list[PolicyInput] = []
        self.action = action

    def __call__(self, inputs):
        self.inputs.append(inputs)
        return self.action(inputs) if callable(self.action) else self.action


def actual_actions(result):
    """Ignore an optional record of the initial zero-order-hold value."""
    return [action for action in result.applied_actions if action.job_id is not None]


def states(result):
    return np.array([[sample["position"], sample["velocity"]] for sample in result.samples])


def test_measurement_compute_and_actuator_delays_have_distinct_clocks():
    policy = RecordingPolicy(0.7)
    result = simulate(
        Oscillator(tau=1.0, zeta=0.0),
        policy,
        TimingConfig(
            duration=0.16,
            observation_interval=0.01,
            sensor_delay=0.03,
            compute_duration=0.05,
            actuator_delay=0.02,
            sample_interval=0.01,
        ),
        initial_state=(1.0, 0.0),
    )

    first = result.jobs[0]
    assert first["capture_time"] == pytest.approx(0.0)
    assert first["available_time"] == pytest.approx(0.03)
    assert first["start_time"] == pytest.approx(0.03)
    assert first["complete_time"] == pytest.approx(0.08)
    assert first["apply_time"] == pytest.approx(0.10)
    assert actual_actions(result)[0].time == pytest.approx(0.10)
    assert policy.inputs[0].planned_apply_time == pytest.approx(0.10)
    assert policy.inputs[0].latest.position == pytest.approx(1.0)
    assert all(sample["action"] == 0.0 for sample in result.samples if sample["time"] < 0.10 - 1e-12)


def test_plant_keeps_moving_through_inference():
    result = simulate(
        Oscillator(tau=1.0, zeta=0.0),
        RecordingPolicy(100.0),
        TimingConfig(duration=0.2, compute_duration=0.25, sample_interval=0.01),
        initial_state=(1.0, 0.0),
    )
    times = np.array([sample["time"] for sample in result.samples])
    expected = np.column_stack((np.cos(times), -np.sin(times)))
    np.testing.assert_allclose(states(result), expected, atol=1e-11, rtol=1e-11)
    assert not actual_actions(result)
    assert result.samples[-1]["position"] != result.samples[0]["position"]


def test_disturbance_clock_does_not_wait_for_an_observation_or_inference():
    change_time = 0.035
    result = simulate(
        Oscillator(tau=1.0, zeta=0.0),
        RecordingPolicy(0.0),
        TimingConfig(
            duration=0.19,
            observation_interval=0.1,
            compute_duration=0.1,
            sample_interval=0.005,
        ),
        disturbance=PiecewiseConstant(changes=((change_time, 1.0),)),
    )
    elapsed = np.maximum(np.array([sample["time"] for sample in result.samples]) - change_time, 0.0)
    expected = np.column_stack((1.0 - np.cos(elapsed), np.sin(elapsed)))
    np.testing.assert_allclose(states(result), expected, atol=1e-11, rtol=1e-11)
    at_change = next(sample for sample in result.samples if abs(sample["time"] - change_time) < 1e-12)
    assert at_change["disturbance"] == 1.0


@pytest.mark.parametrize("compute_duration", [0.005, 0.075])
def test_fixed_cadence_does_not_slow_when_computation_takes_longer(compute_duration):
    policy = RecordingPolicy(0.0)
    result = simulate(
        Oscillator(),
        policy,
        TimingConfig(
            duration=0.125,
            observation_interval=0.01,
            schedule="fixed_cadence",
            decision_interval=0.02,
            compute_duration=compute_duration,
        ),
    )
    np.testing.assert_allclose([job["start_time"] for job in result.jobs], np.arange(7) * 0.02)
    for job in result.jobs:
        assert job["complete_time"] - job["start_time"] == pytest.approx(compute_duration)
    assert len(policy.inputs) == 7


def test_serial_dispatch_consumes_latest_observation_without_queuing_old_frames():
    policy = RecordingPolicy(0.0)
    result = simulate(
        Oscillator(),
        policy,
        TimingConfig(duration=0.16, observation_interval=0.02, compute_duration=0.05),
    )
    np.testing.assert_allclose([job["start_time"] for job in result.jobs], [0.0, 0.05, 0.10, 0.15])
    np.testing.assert_allclose([inputs.latest.capture_time for inputs in policy.inputs], [0.0, 0.04, 0.10, 0.14])
    assert [inputs.latest.index for inputs in policy.inputs] == [0, 2, 5, 7]


@pytest.mark.parametrize("schedule", ["serial", "fixed_cadence"])
def test_one_snapshot_is_not_reused_for_multiple_dispatches(schedule):
    policy = RecordingPolicy(0.0)
    result = simulate(
        Oscillator(),
        policy,
        TimingConfig(
            duration=0.13,
            observation_interval=0.04,
            compute_duration=0.005,
            schedule=schedule,
            decision_interval=0.01,
        ),
    )
    assert [inputs.latest.index for inputs in policy.inputs] == [0, 1, 2, 3]
    np.testing.assert_allclose([job["start_time"] for job in result.jobs], [0.0, 0.04, 0.08, 0.12])


def test_policy_observes_only_available_history_and_previously_applied_actions():
    policy = RecordingPolicy(lambda inputs: -inputs.latest.position)
    result = simulate(
        Oscillator(),
        policy,
        TimingConfig(
            duration=0.23,
            observation_interval=0.01,
            sensor_delay=0.025,
            compute_duration=0.045,
            actuator_delay=0.015,
            schedule="fixed_cadence",
            decision_interval=0.02,
            history_seconds=0.06,
        ),
        initial_state=(1.0, 0.0),
        reference=PiecewiseConstant(changes=((0.075, 2.0),)),
    )
    for inputs in policy.inputs:
        assert isinstance(inputs.observations, tuple)
        assert isinstance(inputs.actions, tuple)
        assert inputs.observations
        assert all(obs.available_time <= inputs.time + 1e-12 for obs in inputs.observations)
        assert all(obs.capture_time <= inputs.time - 0.025 + 1e-12 for obs in inputs.observations)
        assert all(obs.capture_time >= inputs.latest.capture_time - 0.06 - 1e-12 for obs in inputs.observations)
        assert all(action.time <= inputs.time + 1e-12 for action in inputs.actions)
        assert all(obs.reference == (0.0 if obs.capture_time < 0.075 else 2.0) for obs in inputs.observations)
        assert inputs.planned_apply_time - inputs.time == pytest.approx(0.06)
    assert [job["observation_indices"] for job in result.jobs] == [
        [obs.index for obs in inputs.observations] for inputs in policy.inputs
    ]
    assert result.failure_reason is None


def test_delayed_measurement_contains_capture_state_not_delivery_state():
    policy = RecordingPolicy(0.0)
    simulate(
        Oscillator(tau=1.0, zeta=0.0),
        policy,
        TimingConfig(duration=0.12, sensor_delay=0.05, compute_duration=0.02),
        initial_state=(1.0, 0.0),
    )
    for inputs in policy.inputs:
        for obs in inputs.observations:
            assert obs.position == pytest.approx(np.cos(obs.capture_time), abs=1e-11)
            assert obs.velocity == pytest.approx(-np.sin(obs.capture_time), abs=1e-11)
            assert obs.available_time - obs.capture_time == pytest.approx(0.05)
    assert policy.inputs[0].latest.position == 1.0
    assert policy.inputs[0].time == pytest.approx(0.05)


def test_controller_inputs_are_immutable_snapshots():
    policy = RecordingPolicy(0.5)
    simulate(
        Oscillator(),
        policy,
        TimingConfig(duration=0.1, compute_duration=0.02),
        initial_state=(1.0, 0.0),
    )
    first = policy.inputs[0]
    with pytest.raises(FrozenInstanceError):
        first.time = 99.0
    with pytest.raises(FrozenInstanceError):
        first.latest.position = 99.0
    if first.actions:
        with pytest.raises(FrozenInstanceError):
            first.actions[0].value = 99.0
    assert first.latest.capture_time == 0.0
    assert first.latest.position == 1.0
    assert len(first.observations) == 1


def test_simultaneous_signal_changes_and_scheduled_actions_precede_capture():
    policy = RecordingPolicy(1.0)
    result = simulate(
        Oscillator(),
        policy,
        TimingConfig(
            duration=0.065,
            observation_interval=0.02,
            compute_duration=0.02,
            actuator_delay=0.02,
            sample_interval=0.02,
        ),
        reference=PiecewiseConstant(changes=((0.04, 7.0),)),
        disturbance=PiecewiseConstant(changes=((0.04, 0.3),)),
    )
    observed = {
        obs.index: obs
        for inputs in policy.inputs
        for obs in inputs.observations
    }
    at_change = next(obs for obs in observed.values() if abs(obs.capture_time - 0.04) < 1e-12)
    assert at_change.reference == 7.0
    assert at_change.applied_action == 1.0
    sample = next(sample for sample in result.samples if abs(sample["time"] - 0.04) < 1e-12)
    assert sample["reference"] == 7.0
    assert sample["disturbance"] == 0.3
    assert sample["action"] == 1.0


def test_instantaneous_inference_updates_hold_before_sampling_without_rewriting_observation():
    policy = RecordingPolicy(1.0)
    result = simulate(
        Oscillator(), policy, TimingConfig(duration=0.03, compute_duration=0.0)
    )
    assert policy.inputs[0].latest.applied_action == 0.0
    assert result.samples[0]["time"] == 0.0
    assert result.samples[0]["action"] == 1.0
    assert actual_actions(result)[0].time == 0.0


@pytest.mark.parametrize("delay_steps", [0, 1, 4])
def test_trajectory_matches_independent_sampled_data_delay_equations(delay_steps):
    """Check an actual delayed feedback loop against its exact discrete dynamics."""
    tau, zeta, h = 0.7, 0.2, 0.02
    gain = np.array([1.4, 0.35])
    initial_state = np.array([0.7, -0.2])
    step_count = 30
    config = TimingConfig(
        duration=step_count * h,
        observation_interval=h,
        schedule="fixed_cadence",
        decision_interval=h,
        compute_duration=delay_steps * h,
        sample_interval=h,
    )
    policy = RecordingPolicy(lambda inputs: -float(gain @ [inputs.latest.position, inputs.latest.velocity]))
    result = simulate(Oscillator(tau=tau, zeta=zeta), policy, config, initial_state=initial_state)

    # Exact zero-order-hold discretization, calculated without simulator helpers.
    augmented = np.array([[0.0, 1.0, 0.0], [-1.0 / tau**2, -2.0 * zeta / tau, 1.0 / tau**2], [0.0, 0.0, 0.0]])
    discrete = expm(augmented * h)
    phi, gamma = discrete[:2, :2], discrete[:2, 2]
    expected = [initial_state]
    expected_actions = []
    for step in range(step_count + 1):
        action = 0.0 if step < delay_steps else -float(gain @ expected[step - delay_steps])
        expected_actions.append(action)
        if step < step_count:
            expected.append(phi @ expected[-1] + gamma * action)

    assert len(result.samples) == step_count + 1
    np.testing.assert_allclose(states(result), np.array(expected), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose([sample["action"] for sample in result.samples], expected_actions, atol=1e-10, rtol=1e-10)
    assert result.failure_reason is None


def test_jobs_whose_delivery_falls_after_horizon_are_recorded_but_never_applied():
    result = simulate(
        Oscillator(),
        RecordingPolicy(1.0),
        TimingConfig(
            duration=0.055,
            observation_interval=0.02,
            schedule="fixed_cadence",
            decision_interval=0.02,
            compute_duration=0.1,
        ),
    )
    assert len(result.jobs) == 3
    assert all(job["apply_time"] > result.config.duration for job in result.jobs)
    assert not actual_actions(result)
    assert all(sample["action"] == 0.0 for sample in result.samples)


@pytest.mark.parametrize("command,limit", [(100.0, 0.7), (-100.0, 0.7)])
def test_action_saturation_is_applied_and_logged(command, limit):
    result = simulate(
        Oscillator(),
        RecordingPolicy(command),
        TimingConfig(duration=0.1, compute_duration=0.02, action_limit=limit),
    )
    clipped = float(np.clip(command, -limit, limit))
    assert result.jobs
    assert all(job["raw_action"] == command for job in result.jobs)
    assert all(job["action"] == clipped for job in result.jobs)
    assert all(action.value == clipped for action in actual_actions(result))
    assert all(abs(sample["action"]) <= limit for sample in result.samples)


def test_simulation_is_deterministic_including_job_and_event_logs():
    config = TimingConfig(
        duration=0.25,
        sensor_delay=0.023,
        compute_duration=0.037,
        actuator_delay=0.011,
        observation_interval=0.017,
        sample_interval=0.013,
    )
    def run():
        return simulate(
            Oscillator(),
            RecordingPolicy(lambda inputs: -inputs.latest.position - 0.3 * inputs.latest.velocity),
            config,
            initial_state=(0.3, -0.2),
            disturbance=PiecewiseConstant(changes=((0.071, 0.2), (0.189, -0.1))),
        )
    first, second = run(), run()
    assert first.samples == second.samples
    assert first.jobs == second.jobs
    assert first.events == second.events
    assert first.applied_actions == second.applied_actions
    assert first.failure_reason == second.failure_reason


def test_reporting_grid_does_not_set_the_physical_or_controller_clock():
    config = TimingConfig(
        duration=0.5,
        observation_interval=0.02,
        compute_duration=0.035,
        sensor_delay=0.011,
        actuator_delay=0.007,
        sample_interval=0.1,
    )

    def run(configuration):
        return simulate(
            Oscillator(tau=0.5, zeta=0.1),
            RecordingPolicy(lambda inputs: -1.3 * inputs.latest.position - 0.2 * inputs.latest.velocity),
            configuration,
            initial_state=(0.7, 0.1),
            disturbance=PiecewiseConstant(changes=((0.073, 0.2), (0.179, -0.1))),
        )

    coarse = run(config)
    fine = run(replace(config, sample_interval=0.007))
    assert [job["start_time"] for job in coarse.jobs] == [job["start_time"] for job in fine.jobs]
    assert [job["observation_indices"] for job in coarse.jobs] == [job["observation_indices"] for job in fine.jobs]
    np.testing.assert_allclose(
        [job["action"] for job in coarse.jobs],
        [job["action"] for job in fine.jobs],
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(states(coarse)[-1], states(fine)[-1], atol=1e-11, rtol=1e-11)
