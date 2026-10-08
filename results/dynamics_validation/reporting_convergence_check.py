"""Four prospectively selected reporting-grid checks after the main validation.

Uses the frozen main-run source/configuration and the exact saved input tapes.
Only the reporting interval changes, from 10 ms to 5 ms. No source is modified.
"""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/dynamics_validation'
ARTIFACTS = ROOT / 'runs/dynamics_validation'
spec = importlib.util.spec_from_file_location('dynamics_reporting_runner', ROOT / 'experiments/run_dynamics_validation.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

from closed_loop_inhibition.plants import Oscillator
from closed_loop_inhibition.structured_signals import MeasurementNoisePolicy, SampledSignal
from closed_loop_inhibition.timing import simulate


def main():
    destination = OUTPUT / 'reporting_convergence.json'
    if destination.exists():
        raise ValueError('Refusing to overwrite existing convergence outcomes')
    main_manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    if not main_manifest.get('completed_at_utc'):
        raise ValueError('The main validation must be complete before supplemental runs')
    runner.verify_manifest(main_manifest)
    for name, expected in main_manifest['output_sha256'].items():
        if runner.sha(OUTPUT / name) != expected:
            raise ValueError(f'Original main output hash mismatch: {name}')
    for name, expected in main_manifest['artifact_sha256'].items():
        if runner.sha(ARTIFACTS / name) != expected:
            raise ValueError(f'Original main artifact hash mismatch: {name}')
    config = json.loads((OUTPUT / 'config.json').read_text())
    baseline_rows = json.loads((OUTPUT / 'noise_response.json').read_text())
    with np.load(ARTIFACTS / 'noise_tc_0.01_seed_101.npz') as stored:
        tape = SampledSignal(tuple(stored['times']), tuple(stored['values']))
    settings = config['noise']
    rows = []
    for tau, delay in [(0.05, 0), (0.5, 4)]:
        for channel in ['force', 'position_noise']:
            base = next(row for row in baseline_rows if row['controller'] == 'pd'
                        and row['tau'] == tau and row['delay_steps'] == delay
                        and row['channel'] == channel and row['correlation_time'] == 0.01
                        and row['seed'] == 101)
            if not base['completed']:
                raise ValueError('The predeclared baseline was incomplete; do not substitute another condition')
            stem = f'noise_response_pd_tau_{tau}_m_{delay}_{channel}_tc_0.01_seed_101'
            with np.load(ARTIFACTS / f'{stem}.npz') as coarse:
                coarse_times, coarse_states, coarse_actions = (coarse[key].copy()
                                                               for key in ['times', 'states', 'actions'])
            plant = Oscillator(tau=tau, zeta=config['plant']['zeta'])
            policy = runner.pd_policy(config, tau)
            if channel == 'position_noise':
                policy = MeasurementNoisePolicy(policy, tape)
            run = simulate(plant, policy, runner.timing(config, duration=settings['duration'],
                           delay_steps=delay, sample_interval=0.005),
                           disturbance=tape.to_piecewise_constant() if channel == 'force' else None)
            fine_times, fine_states = runner.sample_arrays(run)
            fine_actions = np.asarray([sample['action'] for sample in run.samples])
            metrics = runner.finite_window_metrics(run, settings['burn_in'], 0.01)
            row = {'tau': tau, 'delay_steps': delay, 'controller': 'pd', 'channel': channel,
                   'correlation_time': 0.01, 'seed': 101, 'coarse_reporting_dt': 0.01,
                   'fine_reporting_dt': 0.005, 'completed': metrics['completed'],
                   'failure_reason': run.failure_reason}
            if metrics['completed']:
                indices = np.rint(coarse_times / 0.005).astype(int)
                if np.max(np.abs(fine_times[indices] - coarse_times)) > 1e-10:
                    raise ValueError('The reporting grids do not share the claimed times')
                state_error = runner.normalized_state_error(fine_states[indices], coarse_states, tau)
                action_max_error = float(np.max(np.abs(fine_actions[indices] - coarse_actions)))
                position_rms_error = abs(metrics['rms_position'] - base['rms_position']) / max(base['rms_position'], 1e-300)
                action_rms_error = abs(metrics['rms_action'] - base['rms_action']) / max(base['rms_action'], 1e-300)
                row.update(shared_state_normalized_error=state_error,
                           shared_action_maximum_absolute_error=action_max_error,
                           coarse_position_rms=base['rms_position'], fine_position_rms=metrics['rms_position'],
                           position_rms_relative_difference=position_rms_error,
                           coarse_action_rms=base['rms_action'], fine_action_rms=metrics['rms_action'],
                           action_rms_relative_difference=action_rms_error,
                           passed=(state_error <= 1e-8 and action_max_error <= 1e-10
                                   and position_rms_error <= 0.005 and action_rms_error <= 1e-10))
            else:
                row['passed'] = False
            rows.append(row)
            np.savez_compressed(ARTIFACTS / f'{stem}_reporting_0.005.npz',
                                times=fine_times, states=fine_states, actions=fine_actions)
            print(f"Reporting convergence {tau=}, {delay=}, {channel=}: passed={row['passed']}", flush=True)
            runner.write_json(destination, {'complete': False, 'rows': rows})
    runner.verify_manifest(main_manifest)
    for name, expected in main_manifest['output_sha256'].items():
        if runner.sha(OUTPUT / name) != expected:
            raise ValueError(f'Original main output changed: {name}')
    for name, expected in main_manifest['artifact_sha256'].items():
        if runner.sha(ARTIFACTS / name) != expected:
            raise ValueError(f'Original main artifact changed: {name}')
    result = {'complete': True, 'supplemental_rollouts': 4,
              'all_passed': all(row['passed'] for row in rows),
              'main_manifest_sha256': runner.sha(OUTPUT / 'manifest.json'),
              'script_sha256': runner.sha(__file__),
              'original_main_outputs_verified': len(main_manifest['output_sha256']),
              'original_main_artifacts_verified': len(main_manifest['artifact_sha256']),
              'criteria': {'shared_state_normalized_error': 1e-8,
                           'shared_action_maximum_absolute_error': 1e-10,
                           'position_rms_relative_difference': 0.005,
                           'action_rms_relative_difference': 1e-10},
              'rows': rows}
    runner.write_json(destination, result)


if __name__ == '__main__':
    main()
