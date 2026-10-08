"""Cross-check frequency, stochastic, and delay references independently."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from closed_loop_inhibition.dynamics_validation import (
    continuous_force_response,
    delayed_pd_reference,
    free_ringdown,
    ou_open_loop_covariance,
    plant_timescales,
    sampled_frequency_response,
    sampled_mode_summary,
)
from closed_loop_inhibition.plants import Oscillator


@pytest.mark.parametrize("tau", [0.5, 0.2, 0.1, 0.05])
def test_ringdown_matches_independent_numerical_ode_across_plant_speeds(tau):
    plant = Oscillator(tau=tau)
    times = np.linspace(0, 8 * tau, 201)
    state = [0.3, -0.4]

    def derivative(_time, x):
        q, v = x
        return [v, (-q - 2 * plant.zeta * tau * v) / tau**2]

    ode = solve_ivp(derivative, [0, times[-1]], state, t_eval=times, rtol=2e-12, atol=1e-13)
    assert ode.success
    assert_allclose(free_ringdown(plant, times, state), ode.y.T, atol=2e-11, rtol=2e-10)


def test_timescales_distinguish_tau_period_and_envelope():
    plant = Oscillator(tau=0.5, zeta=0.15)
    scales = plant_timescales(plant)
    assert scales["natural_angular_frequency"] == 2
    assert scales["natural_period"] == pytest.approx(np.pi)
    assert scales["damped_period"] == pytest.approx(3.177565,
                                                     rel=1e-5)
    assert scales["envelope_decay_time"] == pytest.approx(10 / 3)
    initial = [0.7, 0.1]
    after_period = free_ringdown(plant, [scales["damped_period"]], initial)[0]
    assert_allclose(after_period, np.asarray(initial) * np.exp(-scales["damped_period"] / scales["envelope_decay_time"]))
    assert plant_timescales(Oscillator(zeta=0))["envelope_decay_time"] is None
    assert plant_timescales(Oscillator(zeta=1))["damped_period"] is None


def test_continuous_force_response_static_resonant_and_high_frequency_limits():
    plant = Oscillator(tau=0.2, zeta=0.15)
    dc, resonant, fast = continuous_force_response(plant, [0, 1 / plant.tau, 1e6 / plant.tau])
    assert dc == 1
    assert resonant == pytest.approx(-1j / (2 * plant.zeta))
    assert fast.real * 1e12 == pytest.approx(-1, rel=1e-9)
    assert fast.imag < 0


@pytest.mark.parametrize("delay_steps", [0, 1, 4])
def test_sampled_dc_gains_are_independent_of_delay_and_force_grid(delay_steps):
    plant = Oscillator()
    kwargs = dict(plant=plant, kp=2, kd=1, period=0.05, delay_steps=delay_steps,
                  angular_frequencies=[0], force_dt=0.0025)
    assert sampled_frequency_response(**kwargs)[0] == pytest.approx(1 / 3)
    assert sampled_frequency_response(**kwargs, channel="position_noise")[0] == pytest.approx(-2 / 3)


@pytest.mark.parametrize("delay_steps", [0, 2])
@pytest.mark.parametrize("channel,force_dt", [("force", 0.05), ("force", 0.01), ("position_noise", 0.01)])
def test_sampled_phasor_matches_settled_time_domain_response(delay_steps, channel, force_dt):
    plant = Oscillator()
    h, dt, omega, periods = 0.05, force_dt, 2 * np.pi, 1000
    stride = int(round(h / dt))
    fine_times = np.arange(periods * stride) * dt
    capture_times = np.arange(periods + 1) * h
    force = np.cos(omega * fine_times) if channel == "force" else np.zeros(len(fine_times))
    noise = np.cos(omega * capture_times) if channel == "position_noise" else None
    run = delayed_pd_reference(plant, [0, 0], force, fine_dt=dt, period=h,
                               delay_steps=delay_steps, position_noise=noise)
    fit_times = capture_times[600:-1]
    fit_values = run["states"][::stride, 0][600:-1]
    design = np.column_stack((np.cos(omega * fit_times), -np.sin(omega * fit_times)))
    coefficients = np.linalg.lstsq(design, fit_values, rcond=None)[0]
    measured = coefficients[0] + 1j * coefficients[1]
    theory = sampled_frequency_response(plant, 2, 1, h, delay_steps, [omega],
                                         channel=channel, force_dt=dt)[0]
    assert_allclose(measured, theory, atol=1e-11, rtol=1e-10)


def test_force_grid_converges_to_continuous_open_loop_response():
    plant = Oscillator(tau=0.2)
    omega = np.array([0.2, 2, 5, 10])
    continuous = continuous_force_response(plant, omega)
    sampled = sampled_frequency_response(plant, 0, 0, 0.05, 0, omega, force_dt=0.00005)
    assert_allclose(sampled, continuous, rtol=3e-4, atol=1e-8)


def test_cold_start_delay_and_sensor_noise_have_causal_application():
    plant = Oscillator()
    run = delayed_pd_reference(plant, [0, 0], np.zeros(12), fine_dt=0.025,
                               period=0.05, delay_steps=2,
                               position_noise=[1, 0, 0, 0, 0, 0, 0])
    assert_allclose(run["states"][:5], 0, atol=0)
    assert_allclose(run["actions"][:4], 0, atol=0)
    assert_allclose(run["actions"][4:6], -2, atol=0)
    assert run["states"][5, 0] < 0
    # Noise perturbs the sensed value, not the physical state instantaneously.
    assert run["states"][0, 0] == 0


def test_piecewise_force_and_delayed_reference_match_independent_ode_integration():
    plant = Oscillator(tau=0.1)
    dt, stride, delay = 0.01, 5, 1
    force = 0.1 * np.sin(np.arange(50) * 0.7)
    run = delayed_pd_reference(plant, [0.2, -0.1], force, fine_dt=dt,
                               period=stride * dt, delay_steps=delay)
    independent_states = [np.array([0.2, -0.1])]
    commands = []
    held = 0.0
    for step, external in enumerate(force):
        if step % stride == 0:
            q, v = independent_states[-1]
            commands.append(-2 * q - plant.tau * v)
            capture = step // stride
            held = commands[capture - delay] if capture >= delay else 0.0

        def derivative(_time, state):
            q, v = state
            return [v, (-q - 2 * plant.zeta * plant.tau * v + held + external) / plant.tau**2]

        solution = solve_ivp(derivative, [0, dt], independent_states[-1], rtol=1e-12, atol=1e-13)
        assert solution.success
        independent_states.append(solution.y[:, -1])
    assert_allclose(run["states"], independent_states, rtol=1e-10, atol=2e-11)


def test_mode_summary_agrees_with_passive_decay_and_detects_delay_instability():
    plant = Oscillator(tau=0.5)
    passive = sampled_mode_summary(plant, 0, 0, 0.05, 4)
    assert passive["spectral_radius"] == pytest.approx(np.exp(-plant.zeta * 0.05 / plant.tau))
    assert passive["growth_rate"] == pytest.approx(-plant.zeta / plant.tau)
    assert passive["envelope_time"] == pytest.approx(plant.tau / plant.zeta)
    assert passive["classification"] == "stable"
    fast = Oscillator(tau=0.05)
    no_delay = sampled_mode_summary(fast, 2, 1, 0.05)
    delayed = sampled_mode_summary(fast, 2, 1, 0.05, 1)
    assert no_delay["stable"]
    assert not delayed["stable"]
    assert delayed["growth_rate"] > 0
    # Keep the unstable finite trajectory: it is a result, not an excluded run.
    run = delayed_pd_reference(fast, [1e-4, 0], np.zeros(200), fine_dt=0.05,
                               period=0.05, delay_steps=1)
    assert np.all(np.isfinite(run["states"]))
    assert abs(run["states"][-1, 0]) > 1e20
    # Growth inferred from disjoint windows follows the dominant pole modulus.
    q = run["states"][:, 0]
    slope = np.polyfit(np.arange(80, 201) * 0.05, np.log(np.abs(q[80:])), 1)[0]
    assert slope == pytest.approx(delayed["growth_rate"], rel=0.03)


def test_held_ou_covariance_matches_continuous_limit_and_force_variance():
    plant = Oscillator(tau=0.2, zeta=0.15)
    tc, std = 0.08, 0.4
    covariance = ou_open_loop_covariance(plant, 1e-5, tc, std)
    qd = std**2 / (1 + 2 * plant.zeta * plant.tau / tc + (plant.tau / tc) ** 2)
    qq = qd * (1 + plant.tau / (2 * plant.zeta * tc))
    vv = qd / (2 * plant.zeta * plant.tau * tc)
    assert covariance[2, 2] == pytest.approx(std**2, rel=1e-10)
    assert covariance[0, 0] == pytest.approx(qq, rel=2e-4)
    assert covariance[1, 1] == pytest.approx(vv, rel=2e-4)
    assert covariance[0, 2] == pytest.approx(qd, rel=2e-4)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert_allclose(ou_open_loop_covariance(plant, 0.0025, tc, 0), 0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"fine_dt": 0}, {"fine_dt": 0.03}, {"delay_steps": -1}, {"delay_steps": True},
    {"position_noise": [0]}, {"position_noise": [np.nan] * 3},
])
def test_invalid_reference_configuration_fails(kwargs):
    settings = {"fine_dt": 0.025, "period": 0.05, **kwargs}
    with pytest.raises(ValueError):
        delayed_pd_reference(Oscillator(), [0, 0], np.zeros(4), **settings)


def test_invalid_analytic_queries_fail_explicitly():
    with pytest.raises(ValueError, match="underdamped"):
        free_ringdown(Oscillator(zeta=1), [0, 1], [1, 0])
    with pytest.raises(ValueError, match="Nyquist"):
        sampled_frequency_response(Oscillator(), 2, 1, 0.05, 0, [100])
    with pytest.raises(ValueError, match="integer multiple"):
        sampled_frequency_response(Oscillator(), 2, 1, 0.05, 0, [1], force_dt=0.03)
    with pytest.raises(ValueError, match="positive plant damping"):
        ou_open_loop_covariance(Oscillator(zeta=0), 0.01, 0.1, 1)
    with pytest.raises(ValueError, match="singular"):
        continuous_force_response(Oscillator(tau=0.5, zeta=0), [2])
