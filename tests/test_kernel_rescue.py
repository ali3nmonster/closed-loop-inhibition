"""Independent algebra, causality, event parity, and frozen kernel controls."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from closed_loop_inhibition import kernel_rescue as rescue
from closed_loop_inhibition.collective_dynamics import _digest
from closed_loop_inhibition.loop_mechanism import EventCycleMap
from closed_loop_inhibition.neural import make_model
from closed_loop_inhibition.records import AppliedAction, Observation, PolicyInput

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    base = json.loads((ROOT / 'configs/timescale_maps.json').read_text())
    base['model']['width'] = 8
    protocol = {'calibration_delay': .05, 'delays': [.05, .1], 'delay_cue': .05,
                'calibration_seeds': [6030001], 'confirmation_seeds': [6050001],
                'probe': {'duration': .8, 'pulse_onset': .6, 'pulse_width': .05, 'pulse_amplitudes': [-.02, .02]},
                'confirmation_durations': [.8, 1.], 'gain_bounds': [.25, 4.], 'response_energy_floor': 1e-16,
                'baseline_epsilon': .0001, 'min_fractional_increase': .01, 'min_positive_fraction': .75,
                'mechanism': {'duration': .8, 'pulse_width': .1, 'pulse_amplitudes': [-.02, .02],
                              'frequencies': [.5, 2.], 'parity_duration': 1.}}
    torch.manual_seed(642)
    model = make_model('transformer', **base['model']).eval()
    inherited = {'tau': .2, 'noise_tau': .05, 'seed': 11, 'group': ['L0H0'],
                 'base_config_sha256': _digest(base),
                 'variants': {'native': {'branch_scales': {}, 'offset': 0., 'gain': 1., 'center': 0.},
                              'joint_weak': {'branch_scales': {'L0H0': .9}, 'offset': .001, 'gain': 1., 'center': 0.}}}
    return base, protocol, model, inherited


def prepare(setup):
    base, protocol, model, inherited = setup
    return rescue.prepare_kernel(base, protocol, .2, .05, 11, model, inherited)


def snapshot(q, v=0., action=None, count=11):
    action = q if action is None else action
    obs = tuple(Observation(index=i, capture_time=i * .05, available_time=i * .05,
                            position=q, velocity=v, reference=0., reference_velocity=0., applied_action=action)
                for i in range(count))
    now = (count - 1) * .05
    return PolicyInput(now, now + .05, obs, (AppliedAction(now, action, None),))


def test_equal_probe_scalar_fit_without_mean_refit():
    groups = np.asarray(['a', 'b', 'b', 'b'])
    source = (np.ones(4) * .01, np.zeros(4))
    target = (np.asarray([.01, .03, .03, .03]), np.zeros(4))
    fit, error = rescue.fit_scalar(source, target, groups, [.25, 4.], 1e-16)
    assert error is None
    assert fit['gain'] == pytest.approx(2.)
    assert not fit['mean_offset_refit']
    assert fit['response_error_energy'] == pytest.approx(.0001)


@pytest.mark.parametrize('target,expected', [(10., 4.), (-1., .25)])
def test_scalar_bounds(target, expected):
    source = (np.array([.1, -.1]), np.zeros(2))
    fit, _ = rescue.fit_scalar(source, (target * source[0], source[1]), np.array(['a', 'a']), [.25, 4.], 1e-16)
    assert fit['gain'] == expected and fit['gain_at_bound']


def test_scalar_undefined_response_not_silent_default():
    zeros = (np.zeros(2), np.zeros(2))
    fit, error = rescue.fit_scalar(zeros, zeros, np.array(['a', 'a']), [.25, 4.], 1e-16)
    assert fit is None and 'below_energy_floor' in error


def test_observed_warmup_aligns_newest_suffix_and_ignores_padding():
    tokens = np.zeros((1, 11, 10), dtype=np.float32)
    valid = np.zeros((1, 11), dtype=bool)
    valid[:, :2] = True
    tokens[0, 0, [0, 1, 4]] = [1, 2, 3]
    tokens[0, 1, [0, 1, 4, 8]] = [4, 5, 6, 7]
    tokens[0, 2:] = np.nan
    anchor = np.arange(34, dtype=float) + 100
    coordinates = rescue.observed_coordinates(tokens, valid, anchor)[0]
    np.testing.assert_array_equal(coordinates[:27], anchor[:27])
    np.testing.assert_array_equal(coordinates[27:], [1, 2, 3, 4, 5, 6, 7])
    assert set(sum(rescue.feature_groups(11).values(), [])) == set(range(34))


@pytest.mark.parametrize('bad', ['no_valid', 'nonprefix', 'nan'])
def test_observation_bank_validation(bad):
    tokens = np.zeros((1, 11, 10), dtype=np.float32)
    valid = np.ones((1, 11), dtype=bool)
    if bad == 'no_valid': valid[:] = False
    if bad == 'nonprefix': valid[:, 0] = False
    if bad == 'nan': tokens[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        rescue.observed_coordinates(tokens, valid, np.zeros(34))


def test_frozen_preparation_preserves_native_weak_and_model(setup):
    base, protocol, model, inherited = setup
    old = deepcopy(inherited)
    weights = deepcopy(model.state_dict())
    prepared = prepare(setup)
    assert prepared['status'] == 'prepared'
    assert prepared['physical_rollouts'] == 3
    assert prepared['eligible']
    assert set(prepared['variants']) == set(rescue.VARIANTS)
    assert prepared['variants']['native'] == old['variants']['native']
    assert prepared['variants']['joint_weak'] == old['variants']['joint_weak']
    assert inherited == old
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], atol=0., rtol=0.)
    anchor = prepared['anchor']
    delta = np.asarray(anchor['native_gradient']) - anchor['scalar_fit']['gain'] * np.asarray(anchor['weak_gradient'])
    np.testing.assert_allclose(prepared['variants']['weak_kernel']['correction']['kernel'], delta, atol=0., rtol=0.)
    assert len(anchor['coordinates']) == 34
    assert set(anchor['decomposition']) == set(rescue.feature_groups(11))
    json.dumps(prepared, allow_nan=False)


@pytest.mark.parametrize('delay', [.05, .1])
def test_exact_common_anchor_local_feedback_and_complete_map_identity(setup, delay):
    base, protocol, model, _ = setup
    prepared = prepare(setup)
    native = rescue.KernelCycleMap(model, base, .2, delay, prepared['variants']['native'])
    anchor = native.uniform_state(prepared['anchor']['position'])
    native_a, native_b, native_c, native_k = native.linearize(anchor)
    for name in rescue.CORRECTIONS:
        changed = rescue.KernelCycleMap(model, base, .2, delay, prepared['variants'][name])
        assert float(changed.command(anchor)) == pytest.approx(prepared['anchor']['position'], abs=1e-14)
        np.testing.assert_allclose(changed.step(anchor).detach().numpy(), anchor, atol=1e-13)
        if name == 'weak_kernel':
            a, b, c, k = changed.linearize(anchor)
            np.testing.assert_allclose(k, native_k, atol=1e-14)
            np.testing.assert_allclose(a, native_a, atol=1e-13)
            np.testing.assert_array_equal(b, native_b)
            np.testing.assert_array_equal(c, native_c)
            # Redundant shift-register coordinates have a large nilpotent
            # block: near-zero eigenvalues are ill-conditioned even when A
            # agrees to machine precision. The dominant radius is meaningful.
            assert max(abs(np.linalg.eigvals(a))) == pytest.approx(max(abs(np.linalg.eigvals(native_a))), abs=1e-12)
    if delay == .1:
        changed_state = anchor.copy()
        changed_state[-1] += .3
        kernel = rescue.KernelCycleMap(model, base, .2, delay, prepared['variants']['weak_kernel'])
        assert float(kernel.command(changed_state)) == float(kernel.command(anchor))
        assert not np.allclose(kernel.step(changed_state).detach().numpy(), kernel.step(anchor).detach().numpy())


def test_runtime_canonical_equilibrium_and_clipping_after_correction(setup):
    base, protocol, model, _ = setup
    prepared = prepare(setup)
    q = prepared['anchor']['position']
    for name in rescue.CORRECTIONS:
        policy = rescue.KernelPolicy(model, base, .2, prepared['variants'][name])
        assert policy(snapshot(q)) == pytest.approx(q, abs=5e-9)
    setting = deepcopy(prepared['variants']['weak_kernel'])
    setting['correction']['kernel'] = [0.] * 34
    setting['correction']['kernel'][-4] = 100.
    policy = rescue.KernelPolicy(model, base, .2, setting)
    assert policy(snapshot(q + 1.)) == base['action_limit']
    assert policy.records[0]['clipped_count'] == 1


@pytest.mark.parametrize('delay', [.05, .1])
def test_corrected_physical_event_simulator_parity(setup, delay):
    base, _, model, _ = setup
    prepared = prepare(setup)
    for name in rescue.VARIANTS:
        cycle = rescue.KernelCycleMap(model, base, .2, delay, prepared['variants'][name])
        assert rescue.simulator_parity(cycle, duration=1.)['passed']


def test_frozen_confirmation_two_delays_two_horizons_and_own_diagnostics(setup):
    base, protocol, model, _ = setup
    prepared = prepare(setup)
    frozen = deepcopy(prepared)
    for delay, duration in [(delay, duration) for delay in [.05, .1] for duration in [.8, 1.]]:
        record = rescue.confirm_kernel(base, protocol, .2, .05, 11, model, prepared, delay, duration=duration)
        assert record['physical_rollouts'] == 15
        assert record['duration'] == duration
        assert record['fixed_history']['native_replay_passed']
        assert record['prepared_sha256'] == _digest(frozen)
        assert record['frozen_settings'] == frozen['variants']
        assert all(x['action_lag_min'] == pytest.approx(delay) for x in record['timing'])
        assert set(r['noise_seed'] for r in record['rollouts']) == {6050001}
        diagnostic = record['correction_on_own_histories']['weak_kernel']['rollouts']
        assert len(diagnostic) == 3
        assert all(row['warmup_decision_count'] == 10 for row in diagnostic)
        assert all(row['components']['raw_command_change']['rms'] > 0 for row in diagnostic)
        assert set(diagnostic[0]['components']) >= {'kernel_older_position', 'kernel_latest_velocity', 'kernel_current_action'}
        json.dumps(record, allow_nan=False)
    assert prepared == frozen


def test_empty_group_all_variants_exact_identity(setup):
    base, protocol, model, inherited = setup
    inherited['group'] = []
    inherited['variants']['joint_weak'] = deepcopy(inherited['variants']['native'])
    prepared = prepare(setup)
    assert not prepared['eligible']
    for delay in [.05, .1]:
        record = rescue.confirm_kernel(base, protocol, .2, .05, 11, model, prepared, delay)
        for name in rescue.VARIANTS:
            assert record['summary'][name]['delta_from_native']['sham']['position_rms'] == 0.
            assert record['summary'][name]['delta_from_native']['paired_recovery']['normalized_position_energy'] == 0.
            assert record['fixed_history']['variants'][name]['waveform']['command_error_rms'] == 0.


def test_complete_failure_retained_and_censoring_controls_not_silently_fit(setup):
    base, protocol, model, inherited = setup
    base['evaluation']['failure_rms'] = 1e-12
    inherited['base_config_sha256'] = _digest(base)
    prepared = prepare(setup)
    assert prepared['status'] == 'prepared'
    native = prepared['calibration']['summary']['native']['sham']
    assert native['task_failure'] == 1. and native['censored'] == 0.
    assert native['position_rms'] is not None
    base['evaluation']['max_abs_state'] = 1e-8
    inherited['base_config_sha256'] = _digest(base)
    censored = prepare(setup)
    assert censored['status'] == 'calibration_censored'
    assert set(censored['variants']) == {'native', 'joint_weak', 'weak_equilibrium'}
    assert set(censored['unavailable_controls']) == {'weak_scalar', 'weak_kernel'}
    report = rescue.confirm_kernel(base, protocol, .2, .05, 11, model, censored, .1)
    assert report['summary']['native']['sham']['position_rms'] is None
    assert report['summary']['weak_kernel'] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('changed', ['cell', 'protocol', 'delay', 'duration'])
def test_confirmation_binding_rejects_changes(setup, changed):
    base, protocol, model, _ = setup
    prepared = prepare(setup)
    kw = dict(tau=.2, noise_tau=.05, seed=11, model=model, prepared=prepared, delay=.1)
    if changed == 'cell': kw['seed'] = 22
    if changed == 'protocol': protocol['confirmation_seeds'] = [6050002]
    if changed == 'delay': kw['delay'] = .075
    if changed == 'duration': kw['duration'] = 3.
    with pytest.raises(ValueError):
        rescue.confirm_kernel(base, protocol, **kw)


def test_analysis_identity_is_construction_check_with_five_independent_parities(setup):
    base, protocol, model, _ = setup
    prepared = prepare(setup)
    report = rescue.analyze_kernel(base, protocol, .2, .05, 11, model, prepared, .1)
    assert report['physical_rollouts'] == 5
    assert report['status'] == 'complete'
    assert report['anchor_identity']['weak_kernel']['full_map_error_max'] < 1e-13
    for name in rescue.CORRECTIONS:
        assert report['variants'][name]['equilibrium']['method'].startswith('known_native_anchor')
    assert 'algebraic construction' in report['anchor_identity']['weak_kernel']['interpretation']
    json.dumps(report, allow_nan=False)


def test_complete_kernel_matches_independent_finite_difference_at_anchor(setup):
    base, _, model, _ = setup
    prepared = prepare(setup)
    native = rescue.KernelCycleMap(model, base, .2, .1, prepared['variants']['native'])
    corrected = rescue.KernelCycleMap(model, base, .2, .1, prepared['variants']['weak_kernel'])
    anchor = native.uniform_state(prepared['anchor']['position'])
    direction = np.random.default_rng(9482).normal(size=len(anchor))
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    with torch.no_grad():
        native_diff = (native.step(anchor + epsilon * direction).numpy()
                       - native.step(anchor - epsilon * direction).numpy()) / (2 * epsilon)
        corrected_diff = (corrected.step(anchor + epsilon * direction).numpy()
                          - corrected.step(anchor - epsilon * direction).numpy()) / (2 * epsilon)
        finite_difference = float(corrected.command(anchor + .1 * direction) - native.command(anchor + .1 * direction))
    np.testing.assert_allclose(native_diff, corrected_diff, atol=1e-8, rtol=1e-6)
    assert abs(finite_difference) > 1e-9  # Local matching has not substituted native globally.


class LargeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(10.))

    def forward(self, tokens, valid):
        return self.bias + tokens[:, 0, 0]


def test_raw_correction_precedes_actuator_clip(setup):
    base, _, _, _ = setup
    # Raw model output at zero is 1.0, above the 0.5 actuator bound.
    # Anchoring then adding a 0.01 observation must give 0.01. Clipping the
    # inherited output first would instead yield -0.49 or another wrong value.
    setting = {'branch_scales': {}, 'offset': 0., 'gain': 1., 'center': 0.,
               'correction': {'anchor_coordinates': [0.] * 34, 'anchor_coordinates_float32': [0.] * 34,
                              'anchor_position': 0., 'weak_anchor_raw_float32': 1.,
                              'weak_anchor_raw_float64': 1., 'gain': 1., 'kernel': [0.] * 34}}
    policy = rescue.KernelPolicy(LargeLinear(), base, .2, setting)
    assert policy(snapshot(.01, action=0.)) == pytest.approx(.01, abs=1e-7)
    cycle = rescue.KernelCycleMap(LargeLinear(), base, .2, .05, setting)
    assert float(cycle.command(cycle.uniform_state(.01))) == pytest.approx(.01, abs=1e-14)


def test_calibration_columns_independently_reconstruct_scalar_fit_and_kernel(setup):
    base, _, _, _ = setup
    prepared = prepare(setup)
    waveforms = prepared['calibration']['fixed_history']['waveforms']
    numerator = denominator = 0.
    correction = prepared['variants']['weak_kernel']['correction']
    for row in waveforms:
        source = np.asarray(row['raw_commands']['joint_weak']['pulse']) - row['raw_commands']['joint_weak']['sham']
        target = np.asarray(row['commands']['native']['pulse']) - row['commands']['native']['sham']
        numerator += np.mean(source * target)
        denominator += np.mean(source * source)
        for arm in ('pulse', 'sham'):
            weak = np.asarray(row['raw_commands']['joint_weak'][arm])
            history = np.asarray(row['observed_coordinates'][arm])
            expected = (correction['anchor_position'] + correction['gain'] * (weak - correction['weak_anchor_raw_float32'])
                        + (history - correction['anchor_coordinates_float32']) @ correction['kernel'])
            np.testing.assert_allclose(row['raw_commands']['weak_kernel'][arm], expected, atol=2e-16)
            np.testing.assert_allclose(row['commands']['weak_kernel'][arm], np.clip(expected, -base['action_limit'], base['action_limit']), atol=2e-16)
    assert correction['gain'] == pytest.approx(np.clip(numerator / denominator, .25, 4.), abs=1e-14)


def test_invalid_anchor_retains_native_weak_and_marks_corrections_unavailable(setup, monkeypatch):
    base, protocol, model, _ = setup
    def unavailable(*args, **kwargs):
        raise ValueError('native anchor is at a clipping kink')
    monkeypatch.setattr(rescue, '_anchor', unavailable)
    prepared = prepare(setup)
    assert prepared['status'] == 'anchor_unavailable'
    assert prepared['physical_rollouts'] == 3
    assert set(prepared['variants']) == {'native', 'joint_weak'}
    assert set(prepared['unavailable_controls']) == set(rescue.CORRECTIONS)
    assert not prepared['anchor']['available']
    confirmed = rescue.confirm_kernel(base, protocol, .2, .05, 11, model, prepared, .1)
    assert confirmed['physical_rollouts'] == 6
    assert confirmed['summary']['native'] is not None
    assert all(confirmed['summary'][name] is None for name in rescue.CORRECTIONS)
    mechanism = rescue.analyze_kernel(base, protocol, .2, .05, 11, model, prepared, .1)
    assert mechanism['physical_rollouts'] == 2
    assert mechanism['anchor_identity'] == {}
    assert set(mechanism['unavailable_controls']) == set(rescue.CORRECTIONS)
    json.dumps(prepared, allow_nan=False)
    json.dumps(confirmed, allow_nan=False)
    json.dumps(mechanism, allow_nan=False)
