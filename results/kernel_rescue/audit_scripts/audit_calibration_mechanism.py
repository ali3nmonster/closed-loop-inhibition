#!/usr/bin/env python3
"""Independent arithmetic audit; imports no experiment implementation helpers."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm

REPO = Path('/home/ball/transformer-closed-loop-inhibition')
NAMES = ('native', 'joint_weak', 'weak_equilibrium', 'weak_scalar', 'weak_kernel')


def read(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Audit:
    def __init__(self):
        self.errors = []
        self.counts = Counter()
        self.maximum = {}
        self.comparisons = 0

    def check(self, condition, context):
        self.comparisons += 1
        if not condition:
            self.errors.append(context)

    def close(self, actual, expected, category, context, tolerance=1e-11):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            self.check(False, f'{context}: invalid shape or nonfinite {category}')
            return
        error = float(np.max(np.abs(a - b), initial=0.))
        previous = self.maximum.get(category, {}).get('absolute_error', -1.)
        if error > previous:
            self.maximum[category] = {'absolute_error': error, 'context': context, 'tolerance': tolerance}
        self.check(error <= tolerance, f'{context}: {category} error={error:.16g} > {tolerance}')


def label(key):
    population, tau, noise, seed = key
    return f'{population}_tau_{tau:g}_noise_{noise:g}_seed_{seed}'


def key(record):
    return record['population'], record['tau'], record['noise_tau'], record['seed']


def expected_models(base, protocol):
    if 'model_subset' in protocol:
        return {(r['population'], r['tau'], r['noise_tau'], r['seed']) for r in protocol['model_subset']}
    return {(population, tau, noise, seed)
            for population, seeds in [('existing', base['seeds']), ('fresh', protocol['fresh_seeds'])]
            for tau, noise, seed in itertools.product(protocol['plant_taus'], base['noise_taus'], seeds)}


def expected_physical_jacobians(base, tau, delay, derivative):
    """Exact ZOH in normalized q,tau*v coordinates, plus hand-built queue.

    Section: capture precedes completion-generated application at an integer
    delay boundary. Captured action is the old hold; the post-completion hold
    shifts the pending FIFO. The network's command uses observations and hold.
    """
    count = min(base['max_tokens'], math.floor(base['history_seconds'] / base['period'] + 1e-9) + 1)
    periods = round(delay / base['period'])
    if periods not in (1, 2) or abs(periods * base['period'] - delay) > 1e-10:
        raise ValueError('Audit restricted to the prospective 50/100 ms integer-phase comparison')
    dimension = 3 * count + periods
    held = 3 * count
    latest = 3 * (count - 1)
    derivative = np.asarray(derivative)
    if len(derivative) != dimension:
        raise ValueError('Incorrect full command derivative dimension')
    continuous = np.array([[0., 1. / tau, 0.],
                           [-1. / tau, -2. * base['zeta'] / tau, 1. / (tau * base['state_scale'])],
                           [0., 0., 0.]])
    held_transition = expm(continuous * base['period'])
    a = np.zeros((dimension, dimension))
    # Oldest observation drops, every remaining (q,v,captured u) shifts back.
    a[:latest, 3:3 * count] = np.eye(latest)
    a[latest:latest + 2, latest:latest + 2] = held_transition[:2, :2]
    a[latest:latest + 2, held] = held_transition[:2, 2] * base['output_scale']
    a[latest + 2, held] = 1.
    if periods == 1:
        a[held, :] = derivative / base['output_scale']
    else:
        a[held, held + 1] = 1.
        a[held + 1, :] = derivative / base['output_scale']
    b = np.zeros(dimension)
    b[latest:latest + 2] = held_transition[:2, 2]
    c = np.zeros(dimension)
    c[latest] = base['state_scale']
    return a, b, c


def audit_preparation(audit, record, base, protocol, context):
    audit.counts['preparations'] += 1
    if record['status'] == 'upstream_unavailable':
        audit.counts['upstream_unavailable'] += 1
        return
    variants, missing = record['variants'], record['unavailable_controls']
    audit.check(set(variants).isdisjoint(missing) and set(variants) | set(missing) == set(NAMES), context + ': available/missing controls')
    audit.check(record['calibration_delay'] == protocol['calibration_delay'], context + ': calibration delay')
    audit.check(record['base_config_sha256'] == digest(base), context + ': base digest')
    audit.check(record['protocol_sha256'] == digest(protocol), context + ': protocol digest')
    bank = record['calibration']['fixed_history']
    if not bank['available']:
        audit.counts['unavailable_calibration_banks'] += 1
        return
    waveforms = bank['waveforms']
    audit.check(len(waveforms) == len(protocol['calibration_seeds']) * len(protocol['probe']['pulse_amplitudes']), context + ': calibration probe count')
    if not record['anchor']['available']:
        audit.counts['unavailable_anchors'] += 1
        return
    anchor = record['anchor']
    h = np.asarray(anchor['coordinates'])
    h32 = np.asarray(anchor['coordinates_float32'])
    count = anchor['history_count']
    expected_anchor = np.asarray([anchor['position'] / base['state_scale'], 0., anchor['position'] / base['output_scale']] * count + [anchor['position'] / base['output_scale']])
    audit.close(h, expected_anchor, 'equilibrium_coordinates', context)
    native_k, weak_k = np.asarray(anchor['native_gradient']), np.asarray(anchor['weak_gradient'])
    cross, source_energy, target_energy = [], [], []
    for waveform in waveforms:
        commands, raw = waveform['commands'], waveform['raw_commands']
        weak_response = np.asarray(raw['joint_weak']['pulse']) - raw['joint_weak']['sham']
        native_response = np.asarray(commands['native']['pulse']) - commands['native']['sham']
        cross.append(float(np.mean(weak_response * native_response)))
        source_energy.append(float(np.mean(weak_response**2)))
        target_energy.append(float(np.mean(native_response**2)))
        for name in variants:
            for arm in ('pulse', 'sham'):
                audit.close(commands[name][arm], np.clip(raw[name][arm], -base['action_limit'], base['action_limit']), 'clipped_command', f'{context}/{waveform["group"]}/{name}/{arm}')
                audit.counts['calibration_command_samples'] += len(commands[name][arm])
            audit.close(waveform['responses'][name], np.asarray(commands[name]['pulse']) - commands[name]['sham'], 'paired_command_response', context + '/' + name)
        for name in ('weak_equilibrium', 'weak_scalar', 'weak_kernel'):
            if name not in variants:
                continue
            correction = variants[name]['correction']
            audit.close(correction['anchor_coordinates'], h, 'frozen_anchor_coordinates', context + '/' + name)
            audit.close(correction['anchor_coordinates_float32'], h32, 'frozen_float32_anchor_coordinates', context + '/' + name)
            audit.close(correction['weak_anchor_raw_float32'], anchor['weak_anchor_raw_float32'], 'frozen_float32_anchor_command', context + '/' + name)
            audit.close(correction['weak_anchor_raw_float64'], anchor['weak_anchor_raw_float64'], 'frozen_float64_anchor_command', context + '/' + name)
            audit.close(correction['anchor_position'], anchor['position'], 'frozen_anchor_position', context + '/' + name)
            for arm in ('pulse', 'sham'):
                original = np.asarray(raw['joint_weak'][arm])
                if correction.get('identity', False):
                    expected = original
                else:
                    coordinates = np.asarray(waveform['observed_coordinates'][arm])
                    expected = (anchor['position'] + correction['gain'] * (original - correction['weak_anchor_raw_float32'])
                                + (coordinates - h32) @ np.asarray(correction['kernel']))
                audit.close(raw[name][arm], expected, 'anchored_raw_command', f'{context}/{waveform["group"]}/{name}/{arm}', 2e-12)
                audit.counts['reconstructed_corrected_command_samples'] += len(expected)
    if 'weak_scalar' not in variants:
        audit.counts['unavailable_scalar_fits'] += 1
        return
    fit = anchor['scalar_fit']
    if fit.get('identity_intervention'):
        expected_gain = 1.
        audit.counts['identity_preparations'] += 1
    else:
        unconstrained = float(np.mean(cross) / np.mean(source_energy))
        expected_gain = float(np.clip(unconstrained, *protocol['gain_bounds']))
        audit.close(fit['unconstrained_gain'], unconstrained, 'unconstrained_gain', context)
        audit.close(fit['source_response_energy'], np.mean(source_energy), 'source_response_energy', context)
        audit.close(fit['target_response_energy'], np.mean(target_energy), 'target_response_energy', context)
        audit.check(fit['gain_at_bound'] == (expected_gain != unconstrained), context + ': bound flag')
        audit.counts['reconstructed_scalar_fits'] += 1
    audit.close(fit['gain'], expected_gain, 'bounded_gain', context)
    audit.close(variants['weak_scalar']['correction']['gain'], expected_gain, 'frozen_scalar_gain', context)
    audit.close(variants['weak_kernel']['correction']['gain'], expected_gain, 'frozen_kernel_gain', context)
    identity = variants['weak_kernel']['correction'].get('identity', False)
    expected_delta = np.zeros_like(native_k) if identity else native_k - expected_gain * weak_k
    audit.close(variants['weak_kernel']['correction']['kernel'], expected_delta, 'kernel_construction', context)
    audit.close(anchor['kernel'], expected_delta, 'saved_kernel_construction', context)
    for name in ('weak_equilibrium', 'weak_scalar'):
        audit.close(variants[name]['correction']['kernel'], np.zeros_like(native_k), 'zero_kernel', context + '/' + name)
    indices = []
    for name, decomposition in anchor['decomposition'].items():
        subset = decomposition['indices']
        indices += subset
        for field, expected in [('native', native_k[subset]), ('weak', weak_k[subset]),
                                ('scalar', expected_gain * weak_k[subset]), ('correction', expected_delta[subset])]:
            audit.close(decomposition[field], expected, 'feature_decomposition', context + '/' + name + '/' + field)
        audit.close(decomposition['correction_l2'], np.linalg.norm(expected_delta[subset]), 'feature_decomposition_norm', context + '/' + name)
    audit.check(sorted(indices) == list(range(len(h))), context + ': disjoint exhaustive feature decomposition')


def audit_mechanism(audit, record, prepared, base, protocol, context):
    audit.counts['mechanism_records'] += 1
    if record['status'] == 'upstream_unavailable':
        return
    audit.check(record['prepared_sha256'] == digest(prepared), context + ': preparation binding')
    audit.check(record['base_config_sha256'] == digest(base), context + ': base binding')
    audit.check(record['protocol_sha256'] == digest(protocol), context + ': protocol binding')
    anchor = prepared['anchor']
    q = anchor.get('position')
    variants = record['variants']
    native_a = np.asarray(variants['native']['state_jacobian'])
    native_k = np.asarray(variants['native']['command_jacobian'])
    for name, variant in variants.items():
        audit.counts['mechanism_variants'] += 1
        audit.check(variant['settings'] == prepared['variants'][name], context + '/' + name + ': frozen coefficients')
        a, b, c, k = [np.asarray(variant[field]) for field in ('state_jacobian', 'force_input_jacobian', 'position_readout', 'command_jacobian')]
        expected_a, expected_b, expected_c = expected_physical_jacobians(base, record['tau'], record['delay'], k)
        audit.close(a, expected_a, 'independent_physical_map_A', context + '/' + name)
        audit.close(b, expected_b, 'independent_physical_map_B', context + '/' + name)
        audit.close(c, expected_c, 'independent_physical_map_C', context + '/' + name)
        radius = float(max(abs(np.linalg.eigvals(a))))
        audit.close(variant['linear']['spectral_radius'], radius, 'recomputed_spectral_radius', context + '/' + name)
        audit.check(variant['linear']['locally_asymptotically_stable'] == (radius < 1.), context + '/' + name + ': stability flag')
        if radius > 0:
            audit.close(variant['linear']['dominant_decay_rate_per_s'], -math.log(radius) / base['period'], 'recomputed_decay_rate', context + '/' + name)
        parity = variant['simulator_parity']
        audit.check(parity['passed'] and parity['state_error_max_normalized'] <= protocol['mechanism']['parity_state_tolerance']
                    and parity['command_error_max_physical'] <= protocol['mechanism']['parity_command_tolerance'], context + '/' + name + ': parity bounds')
        if name in ('weak_equilibrium', 'weak_scalar', 'weak_kernel') and anchor.get('available', True):
            audit.close(variant['equilibrium']['position'], q, 'restored_equilibrium', context + '/' + name)
            correction = prepared['variants'][name]['correction']
            identity = correction.get('identity', False)
            observed_k = np.asarray(anchor['native_gradient']) if identity else correction['gain'] * np.asarray(anchor['weak_gradient']) + np.asarray(correction['kernel'])
            expected_k = np.pad(observed_k, (0, len(k) - len(observed_k)))
            audit.close(k, expected_k, 'independent_constructed_gradient', context + '/' + name)
            if name == 'weak_kernel':
                audit.close(k, native_k, 'kernel_native_K_identity', context, protocol['mechanism']['kernel_jacobian_tolerance'])
                audit.close(a, native_a, 'kernel_native_A_identity', context, protocol['mechanism']['kernel_jacobian_tolerance'])
                native_radius = float(max(abs(np.linalg.eigvals(native_a))))
                audit.close(radius, native_radius, 'kernel_native_dominant_radius', context, 1e-9)
                saved = record['anchor_identity'][name]
                audit.close(saved['jacobian_error_max'], np.max(abs(k - native_k)), 'saved_gradient_identity_error', context)
                audit.close(saved['full_map_error_max'], np.max(abs(a - native_a)), 'saved_map_identity_error', context)
                audit.close(saved['dominant_radius_error'], radius - native_radius, 'saved_radius_identity_error', context)
                audit.counts['constructed_local_matches'] += 1
        if record['delay'] == .1:
            audit.close(k[-1], 0., 'pending_command_hidden_from_policy', context + '/' + name)
        for response in variant['linear']['frequency_response']:
            if not response['available']:
                continue
            z = np.exp(2j * np.pi * response['frequency_hz'] * base['period'])
            transfer = c @ np.linalg.solve(z * np.eye(len(a)) - a, b)
            audit.close([response['real'], response['imag']], [transfer.real, transfer.imag], 'physical_frequency_response', context + '/' + name)
            audit.counts['frequency_responses'] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/tmp/kernel_calibration_mechanism_audit.json'))
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    protocol, base = read(directory / 'config.json'), read(directory / 'base_config.json')
    manifest = read(directory / 'run_manifest.json')
    audit = Audit()
    expected = expected_models(base, protocol)
    def stage_paths(stage):
        paths = sorted((directory / stage).glob('*.json'))
        # A running stage can publish new records after our manifest snapshot.
        # Partial audits inspect only artifacts sealed in that snapshot.
        return [path for path in paths if not args.allow_partial or
                str(path.relative_to(REPO)) in manifest['completed_sha256']]
    preparations = {}
    for path in stage_paths('prepare'):
        record = read(path)
        preparations[key(record)] = record
        audit_preparation(audit, record, base, protocol, path.name)
    audit.check(set(preparations) == expected, 'Full prospective preparation population retained')
    mechanism_keys = set()
    for path in stage_paths('mechanism'):
        record = read(path)
        identity = key(record)
        mechanism_keys.add(identity + (record['delay'],))
        audit_mechanism(audit, record, preparations[identity], base, protocol, path.name)
    required_mechanisms = {identity + (delay,) for identity in expected for delay in protocol['delays']}
    if not args.allow_partial:
        audit.check(mechanism_keys == required_mechanisms, 'Every prospective mechanism condition retained')
    confirmation_keys = set()
    for path in stage_paths('confirm'):
        record = read(path)
        identity = key(record)
        confirmation_keys.add(identity + (record['delay'], record['duration']))
        audit.counts['confirmation_records'] += 1
        if record['status'] == 'upstream_unavailable':
            continue
        prepared = preparations[identity]
        audit.check(record['prepared_sha256'] == digest(prepared), path.name + ': preparation binding')
        audit.check(record['frozen_settings'] == prepared['variants'], path.name + ': exact frozen settings at delay/horizon')
        audit.check(record['unavailable_controls'] == prepared['unavailable_controls'], path.name + ': unavailable controls unchanged')
    required_confirmations = {identity + (delay, duration) for identity in expected for delay in protocol['delays'] for duration in protocol['confirmation_durations']}
    if not args.allow_partial:
        audit.check(confirmation_keys == required_confirmations, 'Every prospective delay/horizon condition retained')
    checked_paths = [directory / 'config.json', directory / 'base_config.json']
    for stage in ('prepare', 'mechanism', 'confirm'):
        checked_paths += stage_paths(stage)
    for path in checked_paths:
        relative = str(path.relative_to(REPO))
        audit.check(manifest['completed_sha256'].get(relative) == hashlib.sha256(path.read_bytes()).hexdigest(), relative + ': sealed artifact digest')
    result = {'passed': not audit.errors, 'complete_population': mechanism_keys == required_mechanisms and confirmation_keys == required_confirmations and set(preparations) == expected,
              'directory': str(directory), 'scope': 'independent saved-calibration arithmetic, full event-map reconstruction, local identity and frozen-condition audit; no experiment helper imports',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'checks': audit.comparisons,
              'counts': dict(audit.counts), 'maximum_errors': audit.maximum, 'errors': audit.errors,
              'expected_models': len(expected), 'expected_mechanism_records': len(required_mechanisms),
              'expected_confirmation_records': len(required_confirmations)}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'maximum_errors'}, indent=2, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
