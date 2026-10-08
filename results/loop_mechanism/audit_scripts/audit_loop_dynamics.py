"""Independent saved-result mechanics audit; imports no experiment modules."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(base, record, variant):
    """Build A,B,C algebraically from physical plant and the command gradient.

    This deliberately does not differentiate or call the implemented map.
    The copied controller gradient K is physical command per normalized state.
    """
    tau, h, delay = record['tau'], record['period'], record['delay']
    count = variant['history_count']
    pending = variant['pending_command_count']
    n = 3 * count + 1 + pending
    assert n == variant['dimension']
    tick_h, tick_l = round(h * 1e9), round(delay * 1e9)
    whole, remainder = divmod(tick_l, tick_h)
    phase = remainder / 1e9
    assert pending == whole - (remainder == 0 and tick_l > 0)
    expected_age = 0. if tick_l and not remainder else h - phase
    assert abs(variant['current_action_age'] - expected_age) < 1e-12
    assert abs(variant['remainder_delay'] - phase) < 1e-12
    scale = np.diag([base['state_scale'], base['state_scale'] / tau])

    def discretize(dt):
        matrix = np.zeros((3, 3))
        matrix[0, 1] = 1.
        matrix[1, :] = [-1. / tau**2, -2. * base['zeta'] / tau, 1. / tau**2]
        matrix = expm(dt * matrix)
        return np.linalg.solve(scale, matrix[:2, :2] @ scale), np.linalg.solve(scale, matrix[:2, 2])

    k = np.asarray(variant['command_jacobian'])
    assert k.shape == (n,)
    if pending:
        assert np.max(np.abs(k[-pending:])) < 1e-14, 'Policy saw pending commands'
    output_scale = base['output_scale']
    history_select = np.eye(n)[3 * (count - 1):3 * (count - 1) + 2]
    held = np.eye(n)[3 * count]
    queued = [np.eye(n)[3 * count + 1 + index] for index in range(pending)]
    incoming = queued + [k / output_scale]
    ah, bh = discretize(h)
    if tick_l == 0:
        physical = ah @ history_select + np.outer(bh, k)
        capture = next_held = k / output_scale
        next_pending = []
    elif remainder == 0:
        physical = ah @ history_select + np.outer(bh * output_scale, held)
        capture, next_held, next_pending = held, incoming[0], incoming[1:]
    else:
        first_a, first_b = discretize(phase)
        final_a, final_b = discretize(h - phase)
        physical = (final_a @ first_a @ history_select
                    + np.outer(final_a @ first_b * output_scale, held)
                    + np.outer(final_b * output_scale, incoming[0]))
        capture = next_held = incoming[0]
        next_pending = incoming[1:]
        np.testing.assert_allclose(final_a @ first_b + final_b, bh, rtol=1e-13, atol=1e-14)
    a = np.zeros((n, n))
    a[:3 * (count - 1), 3:3 * count] = np.eye(3 * (count - 1))
    a[3 * (count - 1):3 * (count - 1) + 2] = physical
    a[3 * count - 1] = capture
    a[3 * count] = next_held
    for index, values in enumerate(next_pending):
        a[3 * count + 1 + index] = values
    b = np.zeros(n)
    b[3 * (count - 1):3 * (count - 1) + 2] = bh
    c = np.zeros(n)
    c[3 * (count - 1)] = base['state_scale']
    return a, b, c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    root = Path('/home/ball/transformer-closed-loop-inhibition')
    config = read(directory / 'config.json')
    base = read(directory / 'base_config.json')
    manifest = read(directory / 'run_manifest.json')
    assert 'mechanism' in manifest['stages'], 'Mechanism stage is not sealed'
    models = config.get('model_subset') or [dict(tau=tau, noise_tau=noise, seed=seed)
        for tau in config['plant_taus'] for noise in base['noise_taus'] for seed in base['seeds']]
    expected = {(m['tau'], m['noise_tau'], m['seed'], delay) for m in models for delay in config['delays']}
    seen = set()
    maxima = Counter()
    counts = Counter()
    statuses = Counter()
    exclusions = []
    input_hashes = {}

    def compare(name, actual, expected_value, *, atol=2e-11, rtol=2e-10):
        if expected_value is None:
            assert actual is None, (name, actual, expected_value)
            counts[name] += 1
            return
        assert actual is not None, (name, actual, expected_value)
        lhs, rhs = np.asarray(actual), np.asarray(expected_value)
        assert lhs.shape == rhs.shape, (name, lhs.shape, rhs.shape)
        assert np.isfinite(lhs).all() and np.isfinite(rhs).all(), name
        np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol, err_msg=name)
        error = float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.
        maxima[name] = max(maxima[name], error)
        counts[name] += int(lhs.size)

    for path in sorted((directory / 'mechanism').glob('*.json')):
        record = read(path)
        identity = (record['tau'], record['noise_tau'], record['seed'], record['delay'])
        assert identity in expected and identity not in seen
        seen.add(identity)
        input_hashes[str(path.relative_to(root))] = digest(path)
        assert manifest['completed_sha256'][str(path.relative_to(root))] == digest(path)
        prepared_path = root / record['dependencies']['calibration']['path']
        prepared = read(prepared_path)
        assert digest(prepared_path) == record['dependencies']['calibration']['sha256']
        input_hashes[str(prepared_path.relative_to(root))] = digest(prepared_path)
        assert set(record['variants']) == set(prepared['variants'])
        assert {'native', 'joint_weak', 'joint_strong'} <= set(record['variants'])
        assert record['physical_rollouts'] == len(record['variants'])
        assert record['period'] == base['period']
        assert record['delay_cue'] == config['delay_cue']
        counts['records'] += 1
        counts['physical_simulator_validation_rollouts'] += record['physical_rollouts']
        for name, variant in record['variants'].items():
            counts['variants'] += 1
            statuses[variant['status']] += 1
            assert variant['settings'] == prepared['variants'][name]
            parity = variant['simulator_parity']
            assert parity['passed'] and not parity['censored'] and parity['failure_reason'] is None
            assert parity['physical_rollouts'] == 1
            assert parity['state_error_max_normalized'] <= config['mechanism']['parity_state_tolerance']
            assert parity['command_error_max_physical'] <= config['mechanism']['parity_command_tolerance']
            assert parity['timing']['passed'] and not parity['timing']['censored']
            counts['simulator_steps_compared'] += parity['mature_steps_compared']
            maxima['simulator_state_error_normalized'] = max(maxima['simulator_state_error_normalized'], parity['state_error_max_normalized'])
            maxima['simulator_command_error_physical'] = max(maxima['simulator_command_error_physical'], parity['command_error_max_physical'])
            if 'state_jacobian' not in variant:
                exclusions.append({'identity': list(identity), 'variant': name, 'status': variant['status'],
                                   'reason': variant['numerical_issues']})
                continue
            assert not variant['numerical_issues'], (identity, name, variant['numerical_issues'])
            a, b, c = reconstruct(base, record, variant)
            saved_a = np.asarray(variant['state_jacobian'])
            compare('augmented_jacobian', saved_a, a)
            compare('force_input_jacobian', variant['force_input_jacobian'], b)
            compare('physical_position_readout', variant['position_readout'], c)
            eq = variant['equilibrium']
            assert eq['converged'] and eq['differentiable']
            assert eq['state_residual_max'] <= config['mechanism']['equilibrium_tolerance']
            maxima['equilibrium_residual_max'] = max(maxima['equilibrium_residual_max'], eq['state_residual_max'])
            compare('equilibrium_action', eq['action'], eq['position'])
            compare('equilibrium_raw_command', np.clip(eq['raw_command'], -base['action_limit'], base['action_limit']), eq['action'])
            expected_eq = ([eq['position'] / base['state_scale'], 0., eq['action'] / base['output_scale']] * variant['history_count']
                           + [eq['action'] / base['output_scale']] * (1 + variant['pending_command_count']))
            compare('equilibrium_state', variant['equilibrium_state'], expected_eq)
            linear = variant['linear']
            # Recompute spectra from the serialized full matrix after the
            # independent physical construction check above. Tiny perturbations
            # of nilpotent redundant-history blocks can move their zero poles.
            poles = np.linalg.eigvals(saved_a)
            reported_poles = np.array([complex(p['real'], p['imag']) for p in linear['poles']])
            rows, cols = linear_sum_assignment(np.abs(poles[:, None] - reported_poles[None, :]))
            compare('poles', np.abs(poles[rows] - reported_poles[cols]), np.zeros(len(poles)), atol=1e-9, rtol=0.)
            rho = float(np.max(np.abs(poles)))
            dominant = poles[np.argmax(np.abs(poles))]
            compare('spectral_radius', linear['spectral_radius'], rho)
            assert linear['locally_asymptotically_stable'] == (rho < 1.)
            decay = -math.log(rho) / record['period'] if rho else None
            compare('dominant_decay', linear['dominant_decay_rate_per_s'], decay)
            frequency = abs(np.angle(dominant)) / (2 * np.pi * record['period'])
            compare('dominant_frequency', linear['dominant_frequency_hz'], frequency)
            continuous_pole = np.log(complex(dominant)) / record['period'] if rho else None
            damping = (-float(continuous_pole.real) / abs(continuous_pole)
                       if continuous_pole is not None and abs(continuous_pole) > 0 else None)
            compare('dominant_damping', linear['dominant_damping_ratio'], damping)
            for transfer in linear['frequency_response']:
                frequency = transfer['frequency_hz']
                assert frequency in config['mechanism']['frequencies']
                matrix = np.exp(2j * np.pi * frequency * record['period']) * np.eye(len(a)) - a
                condition = float(np.linalg.cond(matrix))
                if not transfer['available']:
                    assert not np.isfinite(condition) or condition > 1e14
                    continue
                value = c @ np.linalg.solve(matrix, b)
                compare('transfer_real', transfer['real'], value.real)
                compare('transfer_imag', transfer['imag'], value.imag)
                compare('transfer_magnitude', transfer['magnitude'], abs(value))
                compare('transfer_phase', transfer['phase_degrees'], np.angle(value, deg=True))
                compare('resolvent_condition', transfer['condition_number'], condition, atol=1e-8, rtol=1e-8)
                assert transfer['steady_state_interpretation_valid'] == (rho < 1.)
            power = np.eye(len(a))
            norms = [1.]
            impulse = np.zeros(len(a))
            unit_q = [0.]
            steps = round(config['mechanism']['duration'] / record['period'])
            for n in range(steps):
                power = a @ power
                norms.append(float(np.linalg.norm(power, 2)))
                impulse = a @ impulse + (b if n == 0 else 0.)
                unit_q.append(float(c @ impulse))
            compare('transient_norm_curve', linear['transient_norm'], norms)
            compare('transient_norm_max', linear['transient_norm_max'], max(norms))
            compare('transient_norm_peak_time', linear['transient_norm_peak_time'], int(np.argmax(norms)) * record['period'])
            compare('unit_force_impulse', linear['unit_one_period_force_position'], unit_q)
            impulses = variant['impulses']
            times = np.arange(steps + 1) * record['period']
            compare('impulse_times', impulses['times'], times)
            expected_amplitudes = set(config['mechanism']['pulse_amplitudes']) | {
                -config['mechanism'].get('verification_amplitude', 1e-6), config['mechanism'].get('verification_amplitude', 1e-6)}
            assert set(p['amplitude'] for p in impulses['records']) == expected_amplitudes
            assert len(impulses['records']) == len(expected_amplitudes)
            width_steps = round(config['mechanism']['pulse_width'] / record['period'])
            verification_errors = []
            for pulse in impulses['records']:
                counts['map_pulse_trajectories'] += 1
                amplitude = pulse['amplitude']
                if pulse['censored']:
                    counts['censored_map_pulse_trajectories'] += 1
                    assert pulse['sampled_recovery_integral_normalized'] is None
                    assert pulse['relative_linear_prediction_error'] is None
                    continue
                q = np.asarray(pulse['position_response'])
                assert q.shape == times.shape
                perturbation = np.zeros(len(a))
                predicted = [0.]
                for n in range(steps):
                    perturbation = a @ perturbation + b * (amplitude if n < width_steps else 0.)
                    predicted.append(float(c @ perturbation))
                predicted = np.asarray(predicted)
                compare('linear_pulse_prediction', pulse['linear_position_response'], predicted)
                error = float(np.linalg.norm(q - predicted) / np.linalg.norm(predicted))
                compare('full_horizon_linear_prediction_error', pulse['relative_linear_prediction_error'], error, atol=2e-9, rtol=2e-8)
                short_steps = min(len(q), round(.5 / record['period']) + 1)
                short_error = float(np.linalg.norm(q[:short_steps] - predicted[:short_steps]) / np.linalg.norm(predicted[:short_steps]))
                compare('half_second_linear_prediction_error', pulse['first_half_second_relative_linear_prediction_error'], short_error, atol=2e-9, rtol=2e-8)
                if pulse['verification_probe']:
                    verification_errors.append(short_error)
                energy = float(np.trapezoid((q / amplitude)**2, dx=record['period']))
                compare('map_pulse_recovery_integral', pulse['sampled_recovery_integral_normalized'], energy)
                compare('map_pulse_peak', pulse['peak_position_over_amplitude'], np.max(np.abs(q)) / abs(amplitude))
                tail = q[width_steps:]
                filtered = tail[np.abs(tail) > max(np.max(np.abs(q)) * 1e-5, 1e-14)]
                crosses = int(np.sum(filtered[1:] * filtered[:-1] < 0))
                assert pulse['post_pulse_sign_crossings'] == crosses
                outside = np.flatnonzero(np.abs(tail) > np.max(np.abs(q)) * config['mechanism']['settling_fraction'])
                settling = ((outside[-1] + 1) * record['period']
                            if len(outside) and outside[-1] < len(q) - width_steps - 1
                            else 0. if not len(outside) else None)
                compare('settling_time', pulse['settling_time_after_pulse_2pct_peak'], settling)
                compare('observed_map_duration', pulse['observed_duration'], config['mechanism']['duration'])
            assert len(verification_errors) == 2
            compare('tiny_verification_error', impulses['verification_relative_error_max'], max(verification_errors), atol=2e-9, rtol=2e-8)
            assert max(verification_errors) <= config['mechanism'].get('linearization_relative_tolerance', .01)
            maxima['tiny_verification_relative_error'] = max(maxima['tiny_verification_relative_error'], max(verification_errors))
    assert seen == expected, f'Missing model/delay records: {expected - seen}'
    assert counts['variants'] == counts['physical_simulator_validation_rollouts']
    assert counts['variants'] == sum(statuses.values())
    report = {'passed': not exclusions, 'directory': str(directory), 'audit_script_sha256': digest(__file__),
              'independent_of_experiment_modules': True, 'counts': dict(counts),
              'variant_status_counts': dict(statuses), 'explicit_unavailable_analyses': exclusions,
              'maximum_absolute_errors': dict(maxima), 'input_sha256': input_hashes,
              'method': 'independent event-phase algebra with SciPy matrix exponential; saved command Jacobian supplies policy derivative; NumPy spectral, transfer and trajectory-metric recomputation'}
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(json.dumps({k: v for k, v in report.items() if k != 'input_sha256'}, allow_nan=False, sort_keys=True))


if __name__ == '__main__':
    main()
