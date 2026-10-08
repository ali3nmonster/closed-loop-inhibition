"""Supplemental reporting-grid check; frozen settings, no parameter selection."""

from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('centered_convergence_runner', ROOT / 'experiments/run_centered_suppression.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def compare_arm(coarse, fine):
    coarse_actions = [(action.time, action.value) for action in coarse.applied_actions]
    fine_actions = [(action.time, action.value) for action in fine.applied_actions]
    same_times = [row[0] for row in coarse_actions] == [row[0] for row in fine_actions]
    maximum_action = max((abs(a[1] - b[1]) for a, b in zip(coarse_actions, fine_actions)), default=0.0) if same_times else None
    fine_samples = {round(row['time'] * 1e9): row for row in fine.samples}
    same_samples = all(round(row['time'] * 1e9) in fine_samples for row in coarse.samples)
    errors = {}
    for field in ('position', 'velocity', 'action'):
        errors[f'maximum_shared_grid_{field}_difference'] = max(
            (abs(row[field] - fine_samples[round(row['time'] * 1e9)][field]) for row in coarse.samples), default=0.0
        ) if same_samples else None
    return {'coarse_failure_reason': coarse.failure_reason, 'fine_failure_reason': fine.failure_reason,
            'coarse_applied_actions': len(coarse_actions), 'fine_applied_actions': len(fine_actions),
            'action_times_identical': same_times, 'maximum_applied_action_difference': maximum_action,
            'coarse_times_present_in_fine': same_samples, **errors}


def main():
    output = ROOT / 'results/centered_suppression_pilot'
    artifacts = ROOT / 'runs/centered_suppression_pilot'
    if (output / 'convergence.json').exists():
        raise ValueError('Supplemental convergence already exists; preserve its result')
    runner.verify_stage(output, artifacts, 'calibration')
    config = json.loads((output / 'config.json').read_text())
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    selection, _ = runner.verify_inherited_selection(config)
    pretrained, models, checkpoints = runner.prior.load_frozen_models(config)
    plant = runner.Oscillator(**pretrained['plant'])
    scenario = json.loads((output / 'scenarios.json').read_text())['calibration'][0]
    calibration = json.loads((output / 'calibration.json').read_text())
    rows = []
    for item in selection:
        seed = item['seed']
        for delay in (0.0, 0.2):
            cell = next(row for row in calibration if row['seed'] == seed and row['schedule'] == 'fixed_cadence' and row['compute_duration'] == delay)
            coarse_timing = runner.prior.timing(config, 'fixed_cadence', delay)
            fine_timing = replace(coarse_timing, sample_interval=coarse_timing.sample_interval / 2)
            for variant in ('native', 'centered_weak'):
                policy = runner.make_policy(models[seed], item, cell['variants'][variant], plant, pretrained['encoding'])
                coarse = runner.paired_rollouts(plant, policy, scenario, coarse_timing)
                fine = runner.paired_rollouts(plant, policy, scenario, fine_timing)
                scores_coarse = runner.score_pair(*coarse, scenario, recovery_seconds=config['recovery_seconds'])
                scores_fine = runner.score_pair(*fine, scenario, recovery_seconds=config['recovery_seconds'])
                complete = not scores_coarse['censored'] and not scores_fine['censored']
                difference = scores_fine['J_response'] - scores_coarse['J_response'] if complete else None
                relative = abs(difference) / abs(scores_coarse['J_response']) if complete and abs(scores_coarse['J_response']) > 1e-12 else None
                rows.append({'seed': seed, 'variant': variant, 'schedule': 'fixed_cadence', 'compute_duration': delay,
                             'scenario_id': scenario['id'], 'coarse_interval': coarse_timing.sample_interval,
                             'fine_interval': fine_timing.sample_interval, 'complete': complete,
                             'coarse_J_response': scores_coarse['J_response'], 'fine_J_response': scores_fine['J_response'],
                             'signed_J_difference_fine_minus_coarse': difference, 'absolute_relative_J_difference': relative,
                             'pulse': compare_arm(coarse[0], fine[0]), 'sham': compare_arm(coarse[1], fine[1])})
                print(json.dumps({'stage': 'supplemental_convergence', 'seed': seed, 'variant': variant,
                                  'delay': delay, 'relative_J_difference': relative}), flush=True)
    runner.verify_stage(output, artifacts, 'calibration')
    arms = [row[label] for row in rows for label in ('pulse', 'sham')]
    summary = {'generated_at': datetime.now(timezone.utc).isoformat(),
               'purpose': 'Supplemental reporting-grid convergence on one calibration scenario; no retuning or confirmation selection',
               'calibration_manifest_sha256': runner.prior.sha(output / 'calibration_manifest.json'),
               'script_sha256': runner.prior.sha(Path(__file__)),
               'checkpoint_sha256': checkpoints, 'source_sha256': runner.prior.source_hashes(),
               'pairs_compared': len(rows), 'supplemental_rollouts': len(rows) * 4,
               'all_pairs_complete': all(row['complete'] for row in rows),
               'maximum_absolute_J_difference': max(abs(row['signed_J_difference_fine_minus_coarse']) for row in rows if row['complete']),
               'maximum_relative_J_difference': max(row['absolute_relative_J_difference'] for row in rows if row['absolute_relative_J_difference'] is not None),
               'all_action_times_identical': all(row['action_times_identical'] for row in arms),
               'maximum_applied_action_difference': max(row['maximum_applied_action_difference'] for row in arms if row['maximum_applied_action_difference'] is not None),
               'maximum_shared_grid_position_difference': max(row['maximum_shared_grid_position_difference'] for row in arms if row['maximum_shared_grid_position_difference'] is not None),
               'results': rows}
    runner.prior.write_json(output / 'convergence.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('source_sha256', 'checkpoint_sha256', 'results')}), flush=True)


if __name__ == '__main__':
    main()
