import json
from pathlib import Path
import numpy as np

p = Path(__file__).resolve().parents[1]
records = [json.loads(x.read_text()) for x in (p / 'mechanism').glob('*.json')]
assert len(records) == 60 and all(r['status'] == 'complete' for r in records)
confirmation = {(r['noise_tau'],r['seed'],r['delay']):r for r in [json.loads(x.read_text()) for x in (p/'confirm').glob('*.json')]}
lookup = {(r['noise_tau'],r['seed'],r['delay']):r for r in records}
models = sorted(set((r['noise_tau'],r['seed']) for r in records))
variants = ['native','joint_weak','joint_strong','gain_matched','weak_rescue']
metrics = ['spectral_radius','dominant_decay_rate_per_s','dominant_frequency_hz','dominant_damping_ratio','transient_norm_max']

def energy(r,v,amp=.02,linear=False):
    rows=[x for x in r['variants'][v]['impulses']['records'] if abs(x['amplitude']) == amp]
    if linear:
        return float(np.mean([np.trapezoid(np.asarray(x['linear_position_response'])**2,dx=r['period'])/x['amplitude']**2 for x in rows]))
    return float(np.mean([x['sampled_recovery_integral_normalized'] for x in rows]))

def noisy(r,v):
    return confirmation[r['noise_tau'],r['seed'],r['delay']]['summary'][v]['paired_recovery']['normalized_position_energy']

def relative(r,v,fn):
    return 100*(fn(r,v)/fn(r,'native')-1)

out={'n_models':len(models),'by_delay':{},'delay_comparisons':{},'global':{}}
for delay in [0.,.025,.05,.075,.1]:
    subset = sorted([r for r in records if r['delay']==delay],key=lambda r:(r['noise_tau'],r['seed']))
    block={}
    for v in variants:
        data={m:float(np.mean([r['variants'][v]['linear'][m] for r in subset])) for m in metrics}
        data['rho_min']=float(min(r['variants'][v]['linear']['spectral_radius'] for r in subset))
        data['rho_max']=float(max(r['variants'][v]['linear']['spectral_radius'] for r in subset))
        data['equilibrium_q_mean']=float(np.mean([r['variants'][v]['equilibrium']['position'] for r in subset]))
        data['equilibrium_abs_q_mean']=float(np.mean([abs(r['variants'][v]['equilibrium']['position']) for r in subset]))
        data['own_eq_energy_mean']=float(np.mean([energy(r,v) for r in subset]))
        data['own_eq_energy_pct']=float(np.mean([relative(r,v,energy) for r in subset]))
        own_effects = [relative(r,v,energy) for r in subset]
        data['own_eq_energy_pct_median'] = float(np.median(own_effects))
        data['own_eq_energy_pct_min'] = float(min(own_effects))
        data['own_eq_energy_pct_max'] = float(max(own_effects))
        data['own_eq_small_energy_pct']=float(np.mean([relative(r,v,lambda rr,vv:energy(rr,vv,.0001)) for r in subset]))
        data['own_eq_linear_energy_pct']=float(np.mean([relative(r,v,lambda rr,vv:energy(rr,vv,.02,True)) for r in subset]))
        data['noisy_energy_pct']=float(np.mean([relative(r,v,noisy) for r in subset]))
        data['own_eq_harm_count']=sum(energy(r,v)>energy(r,'native') for r in subset)
        data['noisy_harm_count']=sum(noisy(r,v)>noisy(r,'native') for r in subset)
        data['rho_delta']=float(np.mean([r['variants'][v]['linear']['spectral_radius']-r['variants']['native']['linear']['spectral_radius'] for r in subset]))
        data['rho_increase_count']=sum(r['variants'][v]['linear']['spectral_radius']>r['variants']['native']['linear']['spectral_radius'] for r in subset)
        data['decay_delta']=float(np.mean([r['variants'][v]['linear']['dominant_decay_rate_per_s']-r['variants']['native']['linear']['dominant_decay_rate_per_s'] for r in subset]))
        data['ringdown_noisy_sign_agreement']=sum(np.sign(energy(r,v)-energy(r,'native'))==np.sign(noisy(r,v)-noisy(r,'native')) for r in subset)
        data['finite_amp_linear_energy_relative_deviation_pct']=float(np.mean([100*(energy(r,v)/energy(r,v,.02,True)-1) for r in subset]))
        data['finite_amp_linear_error_mean']=float(np.mean([q['relative_linear_prediction_error'] for r in subset for q in r['variants'][v]['impulses']['records'] if abs(q['amplitude'])==.02]))
        settling=[q['settling_time_after_pulse_2pct_peak'] for r in subset for q in r['variants'][v]['impulses']['records'] if abs(q['amplitude'])==.02]
        data['finite_amp_settling_mean']=float(np.mean(settling)) if all(s is not None for s in settling) else None
        data['finite_amp_unsettled_pulses']=sum(s is None for s in settling)
        data['formal_resolvent_peak_magnitude_mean']=float(np.mean([max(q['magnitude'] for q in r['variants'][v]['linear']['frequency_response']) for r in subset]))
        block[v]=data
    out['by_delay'][str(delay)]=block
for low,high in [(0.,.1),(.025,.075),(.05,.1)]:
    block={}
    for v in variants:
        endpoints=[(lookup[noise,seed,low],lookup[noise,seed,high]) for noise,seed in models]
        block[v]={
            'own_eq_helpful_to_harmful':sum(relative(a,v,energy)<0 and relative(b,v,energy)>0 for a,b in endpoints),
            'own_eq_harmful_to_helpful':sum(relative(a,v,energy)>0 and relative(b,v,energy)<0 for a,b in endpoints),
            'noisy_helpful_to_harmful':sum(relative(a,v,noisy)<0 and relative(b,v,noisy)>0 for a,b in endpoints),
            'noisy_harmful_to_helpful':sum(relative(a,v,noisy)>0 and relative(b,v,noisy)<0 for a,b in endpoints),
            'equilibrium_q_max_change':max(abs(b['variants'][v]['equilibrium']['position']-a['variants'][v]['equilibrium']['position']) for a,b in endpoints),
            'within_variant_decay_decreases_with_delay':sum(b['variants'][v]['linear']['dominant_decay_rate_per_s']<a['variants'][v]['linear']['dominant_decay_rate_per_s'] for a,b in endpoints),
            'rho_effect_negative_to_positive_count':sum((a['variants'][v]['linear']['spectral_radius']-a['variants']['native']['linear']['spectral_radius'])<0 and (b['variants'][v]['linear']['spectral_radius']-b['variants']['native']['linear']['spectral_radius'])>0 for a,b in endpoints),
            'rho_effect_positive_to_negative_count':sum((a['variants'][v]['linear']['spectral_radius']-a['variants']['native']['linear']['spectral_radius'])>0 and (b['variants'][v]['linear']['spectral_radius']-b['variants']['native']['linear']['spectral_radius'])<0 for a,b in endpoints),
            'ringdown_energy_pct_interaction':float(np.mean([relative(b,v,energy)-relative(a,v,energy) for a,b in endpoints])),
            'noisy_energy_pct_interaction':float(np.mean([relative(b,v,noisy)-relative(a,v,noisy) for a,b in endpoints])),
        }
    out['delay_comparisons'][f'{low} to {high}']=block
out['global']={
    'all_locally_stable':all(v['linear']['locally_asymptotically_stable'] for r in records for v in r['variants'].values()),
    'rho_max':max(v['linear']['spectral_radius'] for r in records for v in r['variants'].values()),
    'rho_min':min(v['linear']['spectral_radius'] for r in records for v in r['variants'].values()),
    'all_equilibria_unclipped':all(not v['equilibrium']['clipped'] for r in records for v in r['variants'].values()),
    'clipped_pulse_decisions':sum(q['clipped_decision_count'] for r in records for v in r['variants'].values() for q in v['impulses']['records']),
    'censored_map_pulses':sum(q['censored'] for r in records for v in r['variants'].values() for q in v['impulses']['records']),
}
(p/'mechanistic_findings.json').write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False,default=lambda x:x.item())+'\n')
print(json.dumps(out,sort_keys=True,allow_nan=False,default=lambda x:x.item()))
