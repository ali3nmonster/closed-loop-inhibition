"""Independent checks of the physical augmented map and local linear analysis."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from closed_loop_inhibition import loop_mechanism as mechanism
from closed_loop_inhibition.neural import make_model


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base():
    torch.set_num_threads(1)
    return json.loads((ROOT / "configs/timescale_maps.json").read_text())


@pytest.fixture
def transformer(base):
    torch.manual_seed(773)
    return make_model("transformer", max_tokens=base["max_tokens"], width=8).eval()


class LinearHistory(nn.Module):
    """Independent controller with history, captured/current action and age."""

    def __init__(self, *, history=False, kp=1.2, kd=.8):
        super().__init__()
        weights = [-kp, -kd, 0., 0., .08 if history else 0., 0., 0.,
                   .01 if history else 0., .05 if history else 0., .02 if history else 0.]
        self.weights = nn.Parameter(torch.tensor(weights))
        self.history = history

    def forward(self, tokens, valid):
        last = valid.sum(dim=1) - 1
        current = tokens[torch.arange(tokens.shape[0]), last]
        result = current @ self.weights
        if self.history:
            result = result - .04 * tokens[:, 0, 0]
        return result


@pytest.mark.parametrize("delay,pending,age", [(0., 0, .05), (.025, 0, .025), (.05, 0, 0.),
                                              (.075, 1, .025), (.1, 1, 0.), (.175, 3, .025)])
def test_phase_counts_and_action_ages(base, delay, pending, age):
    cycle = mechanism.EventCycleMap(LinearHistory(history=True), base, .2, delay)
    assert cycle.dimension == 34 + pending
    assert cycle.pending_count == pending
    assert cycle.current_action_age == pytest.approx(age)
    tokens, valid = cycle.tokens(cycle.uniform_state(.003))
    assert bool(valid.all())
    np.testing.assert_allclose(tokens[:, 7].detach().numpy(), .25)
    np.testing.assert_allclose(tokens[:, 9].detach().numpy(), age / .2)


@pytest.mark.parametrize("delay", [0., .025, .05, .075, .1, .175])
def test_original_event_simulator_matches_map_with_transformer_gates(base, transformer, delay):
    original = deepcopy(transformer.state_dict())
    settings = {"branch_scales": {"L0H2": .9, "L1MLP": 1.1},
                "gain": .8, "offset": .002, "center": -.001}
    cycle = mechanism.EventCycleMap(transformer, base, .2, delay, settings)
    parity = mechanism.simulator_parity(cycle)
    assert parity["passed"]
    assert parity["mature_steps_compared"] == 30
    assert not parity["censored"]
    assert parity["state_error_max_normalized"] < 2e-6
    for name, value in transformer.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0., rtol=0.)


@pytest.mark.parametrize("delay", [0., .025, .05, .075, .1])
def test_linear_controller_history_and_pipeline_match_event_simulator(base, delay):
    cycle = mechanism.EventCycleMap(LinearHistory(history=True), base, .2, delay)
    assert mechanism.simulator_parity(cycle)["passed"]


def test_integer_latency_keeps_capture_before_completion(base):
    cycle = mechanism.EventCycleMap(LinearHistory(), base, .2, .05)
    state = cycle.uniform_state(.01)
    state[33] = .07  # held physical action .007
    command = float(cycle.command(state)) / cycle.output_scale
    following = cycle.step(state).detach().numpy()
    assert following[32] == pytest.approx(.07)  # capture sees the old hold
    assert following[33] == pytest.approx(command)  # dispatch sees new application
    zero = mechanism.EventCycleMap(LinearHistory(), base, .2, 0.)
    zero_command = float(zero.command(state)) / zero.output_scale
    following = zero.step(state).detach().numpy()
    assert following[32] == pytest.approx(zero_command)
    assert following[33] == pytest.approx(zero_command)


def test_pending_commands_are_hidden_from_policy_but_affect_plant(base):
    cycle = mechanism.EventCycleMap(LinearHistory(history=True), base, .2, .075)
    state = cycle.uniform_state(.003)
    changed = state.copy()
    changed[-1] += .1
    torch.testing.assert_close(cycle.tokens(state)[0], cycle.tokens(changed)[0])
    assert float(cycle.command(state)) == float(cycle.command(changed))
    assert not np.allclose(cycle.step(state).detach().numpy(), cycle.step(changed).detach().numpy())


@pytest.mark.parametrize("delay", [0., .025, .05, .075, .1])
def test_full_map_jacobian_predicts_independent_finite_differences(base, transformer, delay):
    cycle = mechanism.EventCycleMap(transformer, base, .2, delay)
    rng = np.random.default_rng(743)
    state = rng.normal(0., .02, size=cycle.dimension)
    a, b, c, k = cycle.linearize(state)
    direction = rng.normal(size=cycle.dimension)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    with torch.no_grad():
        numerical = (cycle.step(state + epsilon * direction).numpy()
                     - cycle.step(state - epsilon * direction).numpy()) / (2 * epsilon)
        input_difference = (cycle.step(state, epsilon).numpy() - cycle.step(state, -epsilon).numpy()) / (2 * epsilon)
        command_difference = (float(cycle.command(state + epsilon * direction))
                              - float(cycle.command(state - epsilon * direction))) / (2 * epsilon)
    np.testing.assert_allclose(a @ direction, numerical, atol=3e-8, rtol=3e-6)
    np.testing.assert_allclose(b, input_difference, atol=3e-10, rtol=3e-8)
    assert k @ direction == pytest.approx(command_difference, abs=3e-9, rel=3e-6)
    assert c[cycle.position_index] == base["state_scale"]


def test_zero_delay_pd_poles_match_exact_sampled_feedback(base):
    policy = LinearHistory(kp=1.2, kd=.8)
    cycle = mechanism.EventCycleMap(policy, base, .2, 0.)
    eq, info = cycle.equilibrium()
    assert info["state_residual_max"] < 1e-12
    a, b, c, k = cycle.linearize(eq)
    ad, bd = cycle.plant.discretize(cycle.period)
    actual_weights = policy.weights.detach().numpy()
    gain = -actual_weights[:2] * cycle.output_scale / cycle.state_scale * [1., cycle.tau]
    expected = np.linalg.eigvals(ad - np.outer(bd, gain))
    observed = np.linalg.eigvals(a)
    observed = observed[np.abs(observed) > 1e-8]
    np.testing.assert_allclose(np.sort_complex(observed), np.sort_complex(expected), atol=2e-14)


@pytest.mark.parametrize("delay", [0., .025, .05, .075, .1])
def test_passive_transfer_matches_independent_exact_plant_discretization(base, delay):
    cycle = mechanism.EventCycleMap(LinearHistory(kp=0., kd=0.), base, .2, delay)
    eq, info = cycle.equilibrium()
    a, b, c, k = cycle.linearize(eq)
    frequencies = [0., .5, 1., 4., 8.]
    analysis = mechanism.linear_analysis(a, b, c, cycle.period, frequencies, 20)
    ad, bd = cycle.plant.discretize(cycle.period)
    for row in analysis["frequency_response"]:
        z = np.exp(2j * np.pi * row["frequency_hz"] * cycle.period)
        expected = np.array([1., 0.]) @ np.linalg.solve(z * np.eye(2) - ad, bd)
        assert complex(row["real"], row["imag"]) == pytest.approx(expected, abs=3e-14)
    assert analysis["dominant_decay_rate_per_s"] == pytest.approx(cycle.plant.zeta / cycle.tau)
    expected_frequency = np.sqrt(1. - cycle.plant.zeta**2) / (2 * np.pi * cycle.tau)
    assert analysis["dominant_frequency_hz"] == pytest.approx(expected_frequency)


def test_offset_equilibrium_is_solved_for_actual_controller(base):
    cycle = mechanism.EventCycleMap(LinearHistory(kp=1., kd=.8), base, .2, .075,
                                   {"offset": .003, "gain": 1.5, "center": .001})
    state, info = cycle.equilibrium()
    expected = (.003 + .001 * (1 - 1.5)) / (1 + 1.5 * cycle.output_scale / cycle.state_scale)
    assert info["position"] == pytest.approx(expected, abs=2e-13)
    np.testing.assert_allclose(cycle.step(state).detach().numpy(), state, atol=2e-12)
    assert info["differentiable"] and not info["clipped"]


def test_linear_impulse_predictions_and_sign_symmetry(base):
    cycle = mechanism.EventCycleMap(LinearHistory(history=True), base, .2, .075)
    eq, _ = cycle.equilibrium()
    a, b, c, _ = cycle.linearize(eq)
    data = mechanism.impulse_analysis(cycle, eq, a, b, c, amplitudes=[1e-4, .02], width=.1, duration=1.)
    assert data["verification_relative_error_max"] < 1e-8
    assert len(data["records"]) == 6
    for row in data["records"]:
        assert row["relative_linear_prediction_error"] < 1e-8
        assert not row["censored"]
    assert data["records"][2]["sampled_recovery_integral_normalized"] == pytest.approx(
        data["records"][5]["sampled_recovery_integral_normalized"], rel=1e-9)


def test_singular_frequency_is_explicitly_unavailable_and_strict_json():
    report = mechanism.linear_analysis(np.eye(2), np.ones(2), np.ones(2), .05, [0.], 3)
    row = report["frequency_response"][0]
    assert not row["available"]
    assert row["condition_number"] is None
    json.dumps(report, allow_nan=False)


def test_analyze_model_records_five_physical_runs_and_signed_pulses(base, transformer):
    protocol = {"delay_cue": .05, "mechanism": {"duration": .6, "pulse_width": .1,
                 "pulse_amplitudes": [-.02, -.0001, .0001, .02], "frequencies": [.5, 4.]}}
    prepared = {"variants": {
        "native": {}, "joint_weak": {"branch_scales": {"L0H1": .9}},
        "joint_strong": {"branch_scales": {"L0H1": 1.1}},
        "gain_matched": {"gain": 1.1}, "weak_rescue": {"branch_scales": {"L0H1": .9}, "gain": .95}}}
    report = mechanism.analyze_model(base, protocol, .2, .01, 11, transformer, prepared, .075)
    assert report["status"] == "complete"
    assert report["physical_rollouts"] == 5
    for row in report["variants"].values():
        assert row["simulator_parity"]["passed"]
        assert set(x["amplitude"] for x in row["impulses"]["records"]) == {-.02, -.0001, -1e-6, 1e-6, .0001, .02}
        assert row["dimension"] == 35
    json.dumps(report, allow_nan=False)


def test_parity_failure_cannot_be_sealed_as_a_result(base, transformer, monkeypatch):
    monkeypatch.setattr(mechanism, "simulator_parity", lambda *args, **kwargs: {"passed": False, "physical_rollouts": 1})
    with pytest.raises(AssertionError, match="does not match"):
        mechanism.analyze_model(base, {}, .2, .01, 11, transformer, {"variants": {"native": {}}}, .05)


@pytest.mark.parametrize("delay", [-.01, float("nan"), .0250000001])
def test_invalid_delay_is_rejected(base, delay):
    with pytest.raises(ValueError):
        mechanism.EventCycleMap(LinearHistory(), base, .2, delay)


def test_nyquist_and_unpaired_pulses_are_rejected(base):
    with pytest.raises(ValueError, match="Nyquist"):
        mechanism.linear_analysis(np.eye(2) * .5, np.ones(2), np.ones(2), .05, [10.], 3)
    with pytest.raises(ValueError, match="paired"):
        mechanism.analyze_model(base, {"mechanism": {"pulse_amplitudes": [.02]}}, .2, .01, 11,
                                LinearHistory(), {"variants": {"native": {}}}, .05)
