"""Read-only independent replication/provenance audit; output is /tmp only."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path('/home/ball/transformer-closed-loop-inhibition')
sys.path.insert(0, str(ROOT / 'src'))
from closed_loop_inhibition.timescale_maps import load_checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(data):
    h = hashlib.sha256()
    for name, value in sorted(data.items()):
        h.update(name.encode())
        value = np.asarray(value)
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def close(actual, expected, label, errors, tolerance=1e-10):
    if actual is None or expected is None:
        assert actual is expected, (label, actual, expected)
    else:
        error = abs(float(actual) - float(expected))
        assert error <= tolerance * max(1., abs(float(expected))), (label, actual, expected, error)
        errors.append(error)


def binding(row):
    path = ROOT / row['path']
    assert sha(path) == row['sha256'], path
    return path


def average(rows):
    result = {}
    for key in sorted({key for row in rows for key in row if key != 'failure_reason'}):
        values = [row.get(key) for row in rows]
        result[key] = (math.fsum(values) / len(values)
                       if all(isinstance(value, (int, float, bool)) for value in values) else None)
    return result


def audit(output, allow_incomplete=False):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    manifest = read(output / 'run_manifest.json')
    base, protocol = manifest['base_config'], manifest['protocol_config']
    assert read(output / 'config.json') == protocol
    assert read(output / 'base_config.json') == base
    all_complete = set(manifest['stages']) == {'discovery_bank', 'fresh_train', 'fresh_discovery', 'prepare', 'passive', 'confirm', 'mechanism'}
    assert allow_incomplete or all_complete
    def paths(stage):
        prefix = str((output / stage).relative_to(ROOT)) + '/'
        return [ROOT / relative for relative in manifest['completed_sha256']
                if relative.startswith(prefix) and relative.endswith('.json')]
    checked_hashes = {}
    for category in ('scientific_sources_sha256', 'completed_sha256'):
        for relative, expected in manifest[category].items():
            assert sha(ROOT / relative) == expected, relative
        checked_hashes[category] = len(manifest[category])
    if not protocol['development']:
        assert not manifest['source_dirty']
        for relative, expected in manifest['scientific_sources_sha256'].items():
            blob = subprocess.check_output(['git', 'show', f"{manifest['source_commit']}:{relative}"], cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest() == expected, relative
    ancestors = []
    for key in ('loop_manifest', 'delay_manifest', 'parent_manifest', 'ancestor_manifest'):
        path = ROOT / manifest['inherited'][key]
        assert sha(path) == manifest['inherited'][key + '_sha256']
        record = read(path)
        for category in ('scientific_sources_sha256', 'completed_sha256'):
            for relative, expected in record[category].items():
                assert sha(ROOT / relative) == expected, relative
        ancestors.append(dict(path=str(path.relative_to(ROOT)), sources=len(record['scientific_sources_sha256']),
                              artifacts=len(record['completed_sha256'])))

    models = [(pop, tau, noise, seed) for pop, seeds in [('existing', base['seeds']), ('fresh', protocol['fresh_seeds'])]
              for tau in protocol['plant_taus'] for noise in base['noise_taus'] for seed in seeds]
    if protocol.get('model_subset'):
        models = [(r['population'], r['tau'], r['noise_tau'], r['seed']) for r in protocol['model_subset']]
    fresh_models = {m for m in models if m[0] == 'fresh'}
    expected_jobs = {
        'fresh_train': fresh_models, 'fresh_discovery': fresh_models, 'prepare': set(models),
        'discovery_bank': {(m[1],) for m in fresh_models},
        'confirm': {(*m, delay, duration) for m in models for delay in protocol['delays'] for duration in protocol['confirmation_durations']},
        'mechanism': {(*m, delay) for m in models for delay in protocol['delays']},
        'passive': {(tau, noise, duration) for tau, noise in {(m[1], m[2]) for m in models}
                    for duration in protocol['confirmation_durations']}}
    for stage, expected in expected_jobs.items():
        actual = []
        for path in paths(stage):
            row = read(path)
            if stage == 'discovery_bank':
                job = row['tau'],
            elif stage == 'passive':
                job = row['tau'], row['noise_tau'], row['duration']
            else:
                job = row['population'], row['tau'], row['noise_tau'], row['seed']
                if stage in ('confirm', 'mechanism'):
                    job += row['delay'],
                if stage == 'confirm':
                    job += row['duration'],
            actual.append(job)
        assert len(set(actual)) == len(actual), stage
        assert set(actual) <= expected, stage
        if stage in manifest['stages']:
            assert set(actual) == expected, stage
    original_training = deepcopy(base['training'])
    training_rows = paths('fresh_train')
    assert len(training_rows) == len(fresh_models) or (allow_incomplete and 'fresh_train' not in manifest['stages'])
    independent_streams, data_content, selected_hashes, initial_weights = [], {'train': [], 'validation': []}, [], defaultdict(list)
    trained, failed, validation_errors = {}, {}, []
    checkpoint_count = 0
    for path in training_rows:
        row = read(path)
        model = row['population'], row['tau'], row['noise_tau'], row['seed']
        assert model in fresh_models
        conf = row['training_config']
        restored = deepcopy(conf)
        for name in ('train', 'validation'):
            restored['training'][name + '_seed'] = original_training[name + '_seed']
        if protocol['development']:
            for key in protocol.get('development_training', {}):
                restored['training'][key] = original_training[key]
        assert restored == base, model
        index = protocol['fresh_seeds'].index(row['seed']) * len(base['noise_taus']) + base['noise_taus'].index(row['noise_tau'])
        for split, number in [('train', 'episodes'), ('validation', 'validation_episodes')]:
            first = protocol['training_streams'][split + '_base'] + index * protocol['training_streams']['stride']
            assert conf['training'][split + '_seed'] == first
            independent_streams.append(set(range(first, first + conf['training'][number])))
        assert row['training_config_sha256'] == digest(conf)
        if row['status'] == 'training_failed':
            assert row['physical_rollouts'] is None
            failed[model] = row
            continue
        assert row['status'] == 'trained'
        trained[model] = row
        directory = Path(row['dependencies']['artifact_directory'])
        data = {}
        for split, number in [('train', 'episodes'), ('validation', 'validation_episodes')]:
            dataset = directory / 'data' / f"tau_{row['tau']:g}_noise_{row['noise_tau']:g}" / f'{split}.npz'
            with np.load(dataset, allow_pickle=False) as arrays:
                assert arrays['config_sha256'].item() == digest(conf)
                data[split] = {key: arrays[key].copy() for key in ('tokens', 'valid', 'targets', 'episode_ids')}
            values = data[split]
            episodes, counts = np.unique(values['episode_ids'], return_counts=True)
            assert episodes.tolist() == list(range(conf['training'][number]))
            assert np.all(counts == round(conf['training']['duration'] / conf['period']) + 1)
            assert values['valid'].dtype == np.bool_
            assert values['tokens'].shape[1:] == (conf['max_tokens'], conf['model']['input_dim'])
            assert np.isfinite(values['tokens']).all() and np.isfinite(values['targets']).all()
            data_content[split].append(array_hash(values))
        history = row['history']
        assert [r['epoch'] for r in history] == list(range(1, conf['training']['epochs'] + 1))
        assert all(r['seed'] == row['seed'] and r['noise_tau'] == row['noise_tau'] and r['tau'] == row['tau'] for r in history)
        best = min(history, key=lambda r: r['validation_mse'])
        summary = row['summary']
        assert summary['epochs_run'] == conf['training']['epochs']
        assert summary['best_epoch'] == best['epoch']
        assert summary['validation_mse'] == best['validation_mse']
        assert summary['train_examples'] == len(data['train']['targets'])
        assert summary['validation_examples'] == len(data['validation']['targets'])
        assert summary['parameters'] == 17889
        for key in ('checkpoint', 'initial_checkpoint', 'intermediate_checkpoint', 'final_checkpoint'):
            checkpoint = Path(summary[key])
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            checkpoint_count += 1
            assert payload['config'] == conf and payload['config_sha256'] == digest(conf)
            assert payload['model_config'] == base['model']
            assert (payload['tau'], payload['noise_tau'], payload['seed']) == model[1:]
            if key == 'initial_checkpoint':
                initial_weights[row['seed']].append(array_hash({k: v.numpy() for k, v in payload['state_dict'].items()}))
                assert payload['epoch'] == 0
            elif key == 'checkpoint':
                assert payload['epoch'] == best['epoch'] and payload['validation_mse'] == best['validation_mse']
                assert sha(checkpoint) == summary['checkpoint_sha256']
                selected_hashes.append(sha(checkpoint))
            elif key == 'intermediate_checkpoint':
                assert payload['epoch'] == conf['training']['intermediate_epoch']
            else:
                assert payload['epoch'] == conf['training']['epochs']
        model_net = load_checkpoint(summary['checkpoint'], conf)
        validation = data['validation']
        with torch.inference_mode():
            predictions = model_net(torch.from_numpy(validation['tokens']), torch.from_numpy(validation['valid']))
            mse = float(torch.sum((predictions - torch.from_numpy(validation['targets'])) ** 2)) / len(predictions)
        close(mse, best['validation_mse'], 'selected-validation-MSE', validation_errors, tolerance=1e-7)
    assert all(not a & b for index, a in enumerate(independent_streams) for b in independent_streams[index + 1:])
    new_probe_streams = [set(protocol[name]) for name in ('discovery_seeds', 'offset_calibration_seeds',
                                                         'calibration_seeds', 'confirmation_seeds')]
    assert all(not tapes & probes for tapes in independent_streams for probes in new_probe_streams)
    assert all(not a & b for index, a in enumerate(new_probe_streams) for b in new_probe_streams[index + 1:])
    assert all(len(set(values)) == len(values) for values in data_content.values())
    assert len(set(selected_hashes)) == len(selected_hashes)
    assert all(len(set(values)) == 1 for values in initial_weights.values())
    assert len({values[0] for values in initial_weights.values()}) == len(initial_weights)

    branch_errors, discoveries, discovery_states = [], {}, Counter()
    for path in paths('discovery_bank'):
        row = read(path)
        with np.load(binding(row['bank']), allow_pickle=False) as arrays:
            assert set(np.unique(arrays['noise_seeds'])) == set(protocol['discovery_seeds'])
            assert set(np.unique(arrays['amplitudes'])) == set(protocol['discovery_probe']['amplitudes'])
            assert len(np.unique(arrays['groups'])) == len(protocol['discovery_seeds']) * len(protocol['discovery_probe']['amplitudes'])
            assert arrays['pulse_tokens'].shape == arrays['sham_tokens'].shape
            assert arrays['pulse_tokens'].shape[0] == row['metadata']['decision_rows']
    for path in paths('fresh_discovery'):
        row = read(path)
        model = row['population'], row['tau'], row['noise_tau'], row['seed']
        if model in failed:
            assert row['status'] == 'upstream_unavailable'
            continue
        assert model in trained and len(row['branches']) == len(row['branch_ids']) == 10
        assert row['bank_metadata']['noise_seeds'] == sorted(protocol['discovery_seeds'])
        selected = []
        all_classifiable = True
        for branch in row['branches']:
            result = branch['weak']
            groups = result['per_group']
            valid = all(r['native_response_rms'] > protocol['baseline_epsilon'] for r in groups)
            assert result['classifiable'] == valid
            all_classifiable &= valid
            fractions = []
            for probe in groups:
                before, after, amplitude = probe['native_response_rms'], probe['changed_response_rms'], abs(probe['amplitude'])
                close(probe['absolute_change'], after - before, 'discovery-absolute', branch_errors)
                close(probe['normalized_absolute_change'], (after - before) / amplitude, 'discovery-normalized', branch_errors)
                expected = after / before - 1 if before > protocol['baseline_epsilon'] else None
                close(probe['fractional_change'], expected, 'discovery-fraction', branch_errors)
                if expected is not None:
                    fractions.append(expected)
            med = float(np.median(fractions)) if valid else None
            positive = sum(v > 0 for v in fractions) / len(fractions) if valid else None
            close(result['median_fractional_change'], med, 'discovery-median', branch_errors)
            close(result['positive_fraction'], positive, 'discovery-positive', branch_errors)
            eligible = valid and med > .01 and positive >= .75
            assert result['eligible'] == eligible
            if eligible:
                selected.append(branch['branch'])
        expected = selected if all_classifiable else None
        assert row['selected'] == expected
        assert row['group'] == (expected or [])
        assert row['status'] == ('unclassifiable' if expected is None else 'empty_group' if not expected else 'selected')
        settings = row['settings']
        assert settings['selected'] == expected and settings['group'] == row['group']
        assert settings['variants']['native'] == dict(branch_scales={}, center=0., gain=1., offset=0.)
        if settings['status'] == 'prepared':
            weak = settings['variants']['joint_weak']
            assert weak['branch_scales'] == {name: .9 for name in row['group']}
            if not row['group']:
                assert weak['gain'] == 1. and weak['offset'] == weak['center'] == 0.
                assert weak['identity_group']
        discoveries[model] = row
        discovery_states[row['status']] += 1

    prepared = {}
    for path in paths('prepare'):
        row = read(path)
        model = row['population'], row['tau'], row['noise_tau'], row['seed']
        assert model in models and model not in prepared
        prepared[model] = row
        metadata = row['dependencies']
        if model in failed:
            assert row['status'] == 'upstream_unavailable'
            continue
        if model[0] == 'existing':
            source = read(ROOT / metadata['prepared_path'])
            assert sha(ROOT / metadata['prepared_path']) == metadata['prepared_sha256']
            assert sha(ROOT / metadata['discovery_path']) == metadata['discovery_sha256']
        else:
            source = read(binding(metadata['fresh_discovery']))['settings']
            assert read(binding(metadata['training'])) == trained[model]
        binding(metadata['checkpoint'])
        assert row['inherited_preparation_sha256'] == digest(source)
        assert row['variants']['native'] == source['variants']['native']
        assert row['variants']['joint_weak'] == source['variants']['joint_weak']
        assert row['group'] == source['group']
        assert row['calibration_delay'] == row['delay'] == row['delay_cue'] == .05
        assert row['base_config_sha256'] == digest(base)
        assert row['protocol_sha256'] == digest(protocol)
        for setting in row['variants'].values():
            if 'correction' in setting:
                assert setting['calibration_delay'] == .05
    assert set(prepared) == set(models) or (allow_incomplete and 'prepare' not in manifest['stages'])

    numeric_errors, physical, failure_count, censored_count = [], Counter(), Counter(), Counter()
    summary_count, timing_count, energy_count = 0, 0, 0
    for stage in ('fresh_discovery', 'prepare', 'confirm', 'passive'):
        for path in paths(stage):
            row = read(path)
            if row.get('status') == 'upstream_unavailable':
                assert row['physical_rollouts'] == 0
                continue
            data = row.get('calibration', row)
            if stage == 'confirm':
                model = row['population'], row['tau'], row['noise_tau'], row['seed']
                source = prepared[model]
                assert read(binding(row['dependencies']['calibration'])) == source
                assert row['prepared_sha256'] == digest(source)
                assert row['frozen_settings'] == source['variants']
                assert row['unavailable_controls'] == source['unavailable_controls']
                assert row['calibration_delay'] == row['delay_cue'] == .05
                assert row['delay'] in [.05, .1] and row['duration'] in [4., 12.]
            duration = row.get('duration', 4.)
            delay = row.get('delay', .05)
            seeds = protocol['confirmation_seeds'] if stage in ('confirm', 'passive') else protocol[
                'offset_calibration_seeds' if stage == 'fresh_discovery' else 'calibration_seeds']
            by_variant, trials = defaultdict(list), {}
            for observation in data['rollouts']:
                assert observation['noise_seed'] in seeds
                assert observation['amplitude'] in [-.02, .02]
                by_variant[observation['variant']].append(observation)
                for arm, amplitude in [('sham', 0.), ('pulse', observation['amplitude'])]:
                    key = observation['variant'], observation['noise_seed'], amplitude
                    value = observation[arm]
                    if key in trials:
                        assert trials[key] == value
                    trials[key] = value
                recovery = observation['paired_recovery']
                if recovery['complete']:
                    expected = recovery['incremental_position_rms'] ** 2 * (duration - 1.) / observation['amplitude'] ** 2
                    close(recovery['normalized_position_energy'], expected, 'energy-RMS', numeric_errors)
                    energy_count += 1
                else:
                    assert all(recovery[key] is None for key in ('incremental_position_rms', 'incremental_action_rms', 'normalized_position_energy'))
            assert len(trials) == row['physical_rollouts']
            physical[stage] += len(trials)
            for key, value in trials.items():
                failure_count[stage] += value['task_failure']
                censored_count[stage] += value['censored']
                if value['censored']:
                    assert value['position_rms'] is None and value['task_failure']
                else:
                    assert isinstance(value['position_rms'], (int, float))
                    assert value['task_failure'] == (value['position_rms'] > base['evaluation']['failure_rms'])
                    close(value['duration_observed'], duration, 'physical-horizon', numeric_errors)
            assert len(data['timing']) == len(trials)
            assert {(r['variant'], r['noise_seed'], r['amplitude']) for r in data['timing']} == set(trials)
            for clock in data['timing']:
                assert clock['passed']
                for key in ('action_lag_min', 'action_lag_max'):
                    if clock[key] is not None:
                        close(clock[key], delay, 'actual-action-lag', numeric_errors, 2e-9)
                for key in ('dispatch_interval_min', 'dispatch_interval_max'):
                    if clock[key] is not None:
                        close(clock[key], .05, 'actual-decision-period', numeric_errors, 2e-9)
                if not clock['censored']:
                    assert clock['dispatch_count'] == round(duration / .05) + 1
                    assert clock['applied_count'] == round((duration - delay) / .05) + 1
                timing_count += 1
            recalculated = {}
            for variant, observations in by_variant.items():
                assert len(observations) == len(seeds) * 2
                shams = {r['noise_seed']: r['sham'] for r in observations}
                assert len(shams) == len(seeds)
                aggregates = dict(sham=average(list(shams.values())), pulse=average([r['pulse'] for r in observations]),
                                  paired_recovery=average([r['paired_recovery'] for r in observations]))
                recalculated[variant] = aggregates
                saved = data['summary'][variant]
                for arm in aggregates:
                    assert set(saved[arm]) == set(aggregates[arm])
                    for key, value in aggregates[arm].items():
                        close(saved[arm][key], value, 'strict-mean', numeric_errors)
                        summary_count += 1
            reference = recalculated.get('native', recalculated.get('zero_policy'))
            for variant, aggregates in recalculated.items():
                for arm in aggregates:
                    for key, value in aggregates[arm].items():
                        target = reference[arm][key]
                        delta = value - target if value is not None and target is not None else None
                        close(data['summary'][variant]['delta_from_native'][arm][key], delta, 'native-delta', numeric_errors)
                        summary_count += 1
    for stage in ('fresh_train', 'discovery_bank', 'mechanism'):
        for path in paths(stage):
            row = read(path)
            if row['physical_rollouts'] is not None:
                physical[stage] += row['physical_rollouts']
            if stage == 'mechanism' and row.get('status') != 'upstream_unavailable':
                model = row['population'], row['tau'], row['noise_tau'], row['seed']
                source = prepared[model]
                assert read(binding(row['dependencies']['calibration'])) == source
                assert set(row['variants']) == set(source['variants'])
                assert len(row['variants']) == row['physical_rollouts']
                for name, variant in row['variants'].items():
                    assert variant['settings'] == source['variants'][name]
                    assert variant['simulator_parity']['passed']
    if all_complete and (output / 'execution_audit.json').exists():
        execution = read(output / 'execution_audit.json')
        for stage, count in physical.items():
            assert execution['stages'][stage]['known_physical_rollouts'] == count
        assert execution['known_physical_rollouts'] == sum(physical.values())
        if not failed:
            assert execution['total_physical_rollouts'] == sum(physical.values())
    return dict(passed=True, audit_script_sha256=sha(__file__),
                all_stages_complete=all_complete, snapshot_stages=manifest['stages'],
                output=str(output), source_commit=manifest['source_commit'], checked_hashes=checked_hashes,
                checked_ancestors=ancestors, fresh_models=len(fresh_models), trained=len(trained), failed_training=len(failed),
                checkpoint_payloads_checked=checkpoint_count, unique_training_array_hashes=len(set(data_content['train'])),
                unique_validation_array_hashes=len(set(data_content['validation'])), unique_selected_checkpoint_hashes=len(set(selected_hashes)),
                independent_initialization_seeds=len(initial_weights), initialization_weights_identical_across_noise_per_seed=True,
                independent_demonstration_stream_families=len(independent_streams),
                recomputed_validation_mse_error_max=max(validation_errors, default=0.),
                discovery_statuses=dict(discovery_states), discovery_numeric_checks=len(branch_errors),
                discovery_recomputation_error_max=max(branch_errors, default=0.), preparations_bound=len(prepared),
                summary_and_delta_recomputations=summary_count, physical_timing_audits=timing_count,
                recovery_energy_identities=energy_count, metric_recomputation_error_max=max(numeric_errors, default=0.),
                physical_by_stage=dict(physical), known_physical_rollouts=sum(physical.values()),
                physical_task_failures=dict(failure_count), physical_censoring=dict(censored_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/kernel_rescue')
    parser.add_argument('--audit-output', type=Path, default=Path('/tmp/kernel_replication_audit.json'))
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    result = audit(args.output, args.allow_incomplete)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
