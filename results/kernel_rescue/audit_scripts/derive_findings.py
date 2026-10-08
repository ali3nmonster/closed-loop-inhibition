"""Descriptive endpoints and local-mode summaries from complete sealed records."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('results', type=Path)
ROOT = parser.parse_args().results
read = lambda p: json.loads(p.read_text())
confirm = [read(p) for p in sorted((ROOT/'confirm').glob('*.json'))]
mechanism = [read(p) for p in sorted((ROOT/'mechanism').glob('*.json'))]
prepare = [read(p) for p in sorted((ROOT/'prepare').glob('*.json'))]
passive = {(r['noise_tau'], r['duration']): r for r in map(read, (ROOT/'passive').glob('*.json'))}
assert len(confirm) == 208 and len(mechanism) == 104 and len(prepare) == 52
variants = ('native', 'joint_weak', 'weak_equilibrium', 'weak_scalar', 'weak_kernel')
def summarize(values):
    a = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(a))
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), min=float(a.min()), max=float(a.max()))
energy = lambda r, v: r['summary'][v]['paired_recovery']['normalized_position_energy']
results = {'scope': 'Descriptive all-model endpoints; seed-block inference is in summary.json; local matching is by construction.', 'endpoints': [], 'local_modes': [], 'unstable_equilibria': [], 'within_model_timing': []}
for population in ('existing', 'fresh'):
    for duration in (4., 12.):
        for delay in (.05, .1):
            rows = [r for r in confirm if (r['population'],r['duration'],r['delay']) == (population,duration,delay)]
            row = dict(population=population, duration=duration, delay=delay, n=len(rows), variants={})
            for v in variants:
                effects = [100*(energy(r,v)/energy(r,'native')-1) for r in rows]
                row['variants'][v] = dict(recovery_percent=summarize(effects), recovery_absolute=summarize([energy(r,v)-energy(r,'native') for r in rows]), recovery=summarize([energy(r,v) for r in rows]), improves_native=sum(e<0 for e in effects))
                for field in ('position_rms','action_rms'):
                    row['variants'][v][field+'_percent'] = summarize([100*(r['summary'][v]['sham'][field]/r['summary']['native']['sham'][field]-1) for r in rows])
                for field in ('relative_response_error','residual_over_intervention'):
                    row['variants'][v][field] = summarize([r['fixed_history']['variants'][v]['waveform'][field] for r in rows])
            for field in ('position_rms','recovery'):
                vals = [(r['summary']['native']['sham'][field]/passive[(r['noise_tau'],duration)]['summary']['zero_policy']['sham'][field] if field != 'recovery' else energy(r,'native')/energy(passive[(r['noise_tau'],duration)],'zero_policy')) for r in rows]
                row['native_to_passive_'+field] = {**summarize(vals), 'better_count': sum(x<1 for x in vals)}
            results['endpoints'].append(row)
        low = {(r['noise_tau'], r['seed']): r for r in confirm if (r['population'],r['duration'],r['delay']) == (population,duration,.05)}
        high = {(r['noise_tau'], r['seed']): r for r in confirm if (r['population'],r['duration'],r['delay']) == (population,duration,.1)}
        row = dict(population=population,duration=duration,variants={})
        for v in variants[1:]:
            lo = np.array([energy(r,v)-energy(r,'native') for r in low.values()])
            hi = np.array([energy(high[k],v)-energy(high[k],'native') for k in low])
            row['variants'][v] = dict(absolute_interaction=summarize(hi-lo), positive_interactions=int(np.sum(hi-lo>0)), helpful_to_harmful=int(np.sum((lo<0)&(hi>0))))
        row['kernel_better_than_scalar_at_100'] = sum(energy(r,'weak_kernel')<energy(r,'weak_scalar') for r in high.values())
        results['within_model_timing'].append(row)
    for delay in (.05,.1):
        rows=[r for r in mechanism if (r['population'],r['delay'])==(population,delay)]
        row=dict(population=population,delay=delay,variants={})
        for v in variants:
            row['variants'][v]={field:summarize([r['variants'][v]['linear'][field] for r in rows]) for field in ('dominant_decay_rate_per_s','dominant_frequency_hz','spectral_radius')}
            row['variants'][v]['stable_count']=sum(r['variants'][v]['linear']['locally_asymptotically_stable'] for r in rows)
            for r in rows:
                if not r['variants'][v]['linear']['locally_asymptotically_stable']:
                    results['unstable_equilibria'].append(dict(population=population,delay=delay,noise_tau=r['noise_tau'],seed=r['seed'],variant=v,rho=r['variants'][v]['linear']['spectral_radius']))
        row['weak_decay_faster_than_native']=sum(r['variants']['joint_weak']['linear']['dominant_decay_rate_per_s']>r['variants']['native']['linear']['dominant_decay_rate_per_s'] for r in rows)
        results['local_modes'].append(row)
results['fit_gain']=summarize([r['anchor']['scalar_fit']['gain'] for r in prepare])
results['gain_bound_hits']=sum(r['anchor']['scalar_fit']['gain_at_bound'] for r in prepare)
results['max_kernel_matrix_error']=max(r['anchor_identity']['weak_kernel']['full_map_error_max'] for r in mechanism)
results['max_kernel_command_derivative_error']=max(r['anchor_identity']['weak_kernel']['jacobian_error_max'] for r in mechanism)
all_modes=[v for r in mechanism for v in r['variants'].values()]
results['max_parity_state_error']=max(v['simulator_parity']['state_error_max_normalized'] for v in all_modes)
results['max_parity_command_error']=max(v['simulator_parity']['command_error_max_physical'] for v in all_modes)
results['parity_steps']=sum(v['simulator_parity']['mature_steps_compared'] for v in all_modes)
results['max_tiny_pulse_relative_error']=max(v['impulses']['verification_relative_error_max'] for v in all_modes)
(ROOT/'mechanistic_findings.json').write_text(json.dumps(results,indent=2,sort_keys=True,allow_nan=False)+'\n')
print('Wrote',ROOT/'mechanistic_findings.json')
