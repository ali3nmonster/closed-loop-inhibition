"""Independent physical and analytical checks before learned control."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from closed_loop_inhibition.controllers import (
    ClassicalPDController,
    PredictorPDController,
    continuous_closed_loop_poles,
    sampled_closed_loop_matrix,
)
from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.records import AppliedAction, Observation, PolicyInput


def test_free_underdamped_motion_matches_analytical_solution():
    plant = Oscillator(tau=0.7, zeta=0.2)
    q0, v0, elapsed = 1.2, -0.4, 1.9
    decay = plant.zeta / plant.tau
    frequency = np.sqrt(1 - plant.zeta**2) / plant.tau
    a, b = q0, (v0 + decay * q0) / frequency
    wave = a * np.cos(frequency * elapsed) + b * np.sin(frequency * elapsed)
    derivative = frequency * (-a * np.sin(frequency * elapsed) + b * np.cos(frequency * elapsed))
    expected = np.exp(-decay * elapsed) * np.array([wave, derivative - decay * wave])
    assert_allclose(plant.propagate([q0, v0], elapsed), expected, atol=1e-13)


@pytest.mark.parametrize("zeta", [0.0, 0.15, 1.0, 2.0])
def test_exact_propagation_matches_independent_ode_solver(zeta):
    plant = Oscillator(tau=0.4, zeta=zeta)
    state = np.array([0.6, -0.2])
    control, disturbance, elapsed = 0.8, -0.3, 0.73

    def equation(_time, value):
        q, v = value
        return [v, (-q - 2 * zeta * plant.tau * v + control + disturbance) / plant.tau**2]

    numerical = solve_ivp(equation, [0, elapsed], state, rtol=1e-11, atol=1e-13)
    assert numerical.success
    assert_allclose(plant.propagate(state, elapsed, control, disturbance), numerical.y[:, -1], atol=2e-11)
    assert_allclose(state, [0.6, -0.2], atol=0)


def test_equilibrium_and_semigroup_preserve_held_input():
    plant = Oscillator()
    assert_allclose(plant.propagate([1.1, 0], 10.0, 1.5, -0.4), [1.1, 0], atol=1e-13)
    original = np.array([0.7, -0.9])
    direct = plant.propagate(original, 0.9, control=0.3, disturbance=0.1)
    first = plant.propagate(original, 0.2, control=0.3, disturbance=0.1)
    segmented = plant.propagate(first, 0.7, control=0.3, disturbance=0.1)
    assert_allclose(segmented, direct, atol=2e-14)
    unchanged = plant.propagate(original, 0, 0.3)
    assert_allclose(unchanged, original, atol=0)
    assert not np.shares_memory(unchanged, original)


def test_cached_matrices_cannot_be_corrupted_by_caller():
    plant = Oscillator()
    first_a, first_b = plant.discretize(0.1)
    expected_a, expected_b = first_a.copy(), first_b.copy()
    first_a[:] = 0
    first_b[:] = 0
    recovered_a, recovered_b = plant.discretize(0.1)
    assert_allclose(recovered_a, expected_a, atol=0)
    assert_allclose(recovered_b, expected_b, atol=0)


@pytest.mark.parametrize("kwargs", [{"tau": 0}, {"tau": -1}, {"tau": np.inf}, {"zeta": -0.1}, {"zeta": np.nan}])
def test_invalid_physical_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        Oscillator(**kwargs)


@pytest.mark.parametrize(
    "state,dt,control,disturbance",
    [([0, 0], -0.1, 0, 0), ([0, 0], np.inf, 0, 0), ([0], 0.1, 0, 0),
     ([0, np.nan], 0.1, 0, 0), ([0, 0], 0.1, np.inf, 0), ([0, 0], 0.1, 0, np.nan)],
)
def test_invalid_propagation_inputs_fail_explicitly(state, dt, control, disturbance):
    with pytest.raises(ValueError):
        Oscillator().propagate(state, dt, control, disturbance)


def test_continuous_poles_match_characteristic_polynomial():
    plant = Oscillator(tau=0.7, zeta=0.15)
    kp, kd = 2.0, 1.0
    expected = np.roots([plant.tau**2, (2 * plant.zeta + kd) * plant.tau, 1 + kp])
    assert_allclose(np.sort_complex(continuous_closed_loop_poles(plant, kp, kd)), np.sort_complex(expected))
    # Shrinking the sample interval approaches the continuous generator.
    period = 1e-6
    sampled = sampled_closed_loop_matrix(plant, kp, kd, period)
    generator = (sampled - np.eye(2)) / period
    assert_allclose(np.sort_complex(np.linalg.eigvals(generator)), np.sort_complex(expected), atol=5e-6)


@pytest.mark.parametrize("delay_steps", [0, 1, 4])
def test_augmented_map_matches_delayed_held_action_rollout(delay_steps):
    plant = Oscillator(tau=0.5, zeta=0.15)
    kp, kd, period = 2.0, 1.0, 0.06
    transition = sampled_closed_loop_matrix(plant, kp, kd, period, delay_steps)
    history = [np.array([0.8 - i * 0.1, -0.2 + i * 0.03]) for i in range(delay_steps + 1)]
    augmented = np.concatenate(history)
    for _ in range(40):
        delayed = history[-1]
        control = -kp * delayed[0] - kd * plant.tau * delayed[1]
        following = plant.propagate(history[0], period, control)
        history = [following] + history[:-1]
        augmented = transition @ augmented
        assert_allclose(augmented, np.concatenate(history), atol=1e-12)


def _snapshot(state, *, capture=0.0, start=0.0, target=0.0, reference=0.7, reference_velocity=0.0, applied=0.0, actions=()):
    observation = Observation(0, capture, capture, *state, reference, reference_velocity, applied)
    return PolicyInput(start, target, (observation,), tuple(actions))


def test_pd_uses_available_sample_and_reference_feedforward():
    controller = ClassicalPDController(kp=2.0, kd=1.0, tau=0.5)
    snapshot = _snapshot([0.7, 0.0], start=2, target=3)
    assert controller(snapshot) == pytest.approx(0.7)
    moving = _snapshot([0.2, -0.4], reference=0.7, reference_velocity=0.3)
    assert controller(moving) == pytest.approx(0.7 + 2 * 0.5 + 0.5 * 0.7)


def test_predictor_reconstructs_known_history_and_holds_last_command_to_application():
    plant = Oscillator()
    initial = np.array([0.2, -0.1])
    actions = (AppliedAction(-0.1, 8, None), AppliedAction(0.1, 0.5, 1), AppliedAction(0.2, -0.3, 2))
    snapshot = _snapshot(initial, start=0.3, target=0.5, applied=0.1, actions=actions, reference_velocity=0.2)
    expected_state = plant.propagate(initial, 0.1, 0.1)
    expected_state = plant.propagate(expected_state, 0.1, 0.5)
    expected_state = plant.propagate(expected_state, 0.3, -0.3)
    reference = 0.7 + 0.5 * 0.2
    expected_action = reference + 2 * (reference - expected_state[0]) + 0.5 * (0.2 - expected_state[1])
    assert PredictorPDController(plant)(snapshot) == pytest.approx(expected_action)
    assert_allclose(initial, [0.2, -0.1], atol=0)


def test_predictor_rejects_future_action_leakage():
    snapshot = _snapshot([0, 0], start=0.1, target=0.3, actions=(AppliedAction(0.2, 1, 1),))
    with pytest.raises(ValueError, match="available at inference start"):
        PredictorPDController()(snapshot)
