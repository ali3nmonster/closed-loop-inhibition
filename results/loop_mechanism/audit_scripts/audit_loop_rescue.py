"""Independent arithmetic audit; imports no experiment analysis modules."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,default=Path('/tmp/loop_rescue_audit.json'));args=parser.parse_args()
root=Path('/home/ball/transformer-closed-loop-inhibition'); base=json.loads((args.directory/'base_config.json').read_text());protocol=json.loads((args.directory/'config.json').read_text())
counters=Counter();max_error=0.;failed=[]
def check(actual,expected,label,atol=1e-10):
 global max_error
 counters['numeric_checks']+=1
 if actual is None or expected is None:
  assert actual is expected,(label,actual,expected);return
 error=float(np.max(np.abs(np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float))));max_error=max(max_error,error)
 assert np.allclose(actual,expected,atol=atol,rtol=1e-10),(label,actual,expected,error)
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()
def scalar(v):return isinstance(v,(int,float,bool))
def analyze_rollouts(result,seeds,variants):
 rows=result['rollouts']; assert set(row['variant'] for row in rows)==set(variants)
 assert result['physical_rollouts']==len(variants)*len(seeds)*(1+len(protocol['probe']['pulse_amplitudes']))
 assert len(result['timing'])==result['physical_rollouts'];counters['physical_rollouts']+=result['physical_rollouts']
 recomputed={}
 for name in variants:
  selected=[r for r in rows if r['variant']==name];assert len(selected)==len(seeds)*len(protocol['probe']['pulse_amplitudes'])
  shams={}
  for r in selected:
   assert r['noise_seed'] in seeds and r['amplitude'] in protocol['probe']['pulse_amplitudes']
   if r['noise_seed'] in shams:assert shams[r['noise_seed']]==r['sham']
   shams[r['noise_seed']]=r['sham'];recovery=r['paired_recovery']
   if recovery['complete']:
    check(recovery['normalized_position_energy'],recovery['incremental_position_rms']**2*(protocol['probe']['duration']-protocol['probe']['pulse_onset'])/r['amplitude']**2,'position-integral identity')
   else:assert recovery['normalized_position_energy'] is None
  recomputed[name]={}
  for arm,values in [('sham',list(shams.values())),('pulse',[r['pulse'] for r in selected]),('paired_recovery',[r['paired_recovery'] for r in selected])]:
   recomputed[name][arm]={}
   for key in set().union(*(v.keys() for v in values))-{'failure_reason'}:
    expected=float(np.mean([v[key] for v in values])) if all(scalar(v.get(key)) for v in values) else None
    check(result['summary'][name][arm][key],expected,f'{name}/{arm}/{key}');recomputed[name][arm][key]=expected
  assert result['summary'][name]['independent_noise_seeds']==len(shams)
  for arm in ['sham','pulse','paired_recovery']:
   counters['task_failures']+=sum(int(v['task_failure']) for v in (list(shams.values()) if arm=='sham' else [r['pulse'] for r in selected])) if arm!='paired_recovery' else 0
 for name in variants:
  for arm,values in recomputed[name].items():
   for key,val in values.items():
    native=recomputed['native'][arm][key]
    check(result['summary'][name]['delta_from_native'][arm][key],val-native if val is not None and native is not None else None,'delta')

def history(hist,settings,prepared=False):
 if not hist['available']:return
 waves=hist['waveforms'];variants=list(settings);floor=protocol['response_energy_floor']
 # Each probe receives equal weight, regardless of number of decision rows.
 def mean(v):return float(np.mean([np.mean(a) for a in v]))
 outputs={name:{arm:[np.asarray(w['commands'][name][arm]) for w in waves] for arm in ['pulse','sham']} for name in variants}
 raw={name:{arm:[np.asarray(w['raw_commands'][name][arm]) for w in waves] for arm in ['pulse','sham']} for name in variants}
 response={name:[p-s for p,s in zip(outputs[name]['pulse'],outputs[name]['sham'])] for name in variants}
 for name in variants:
  s=settings[name]
  for i,w in enumerate(waves):
   check(w['responses'][name],response[name][i],'stored pulse-sham waveform')
  for arm in ['pulse','sham']:
   transformed=[s['center']+s['gain']*(r-s['center'])+s['offset'] for r in raw[name][arm]]
   for y,z in zip(outputs[name][arm],transformed):check(y,np.clip(z,-base['action_limit'],base['action_limit']),'affine+clip commands')
   check(hist['variants'][name]['clipping_fraction'][arm],mean([np.abs(z)>=base['action_limit'] for z in transformed]),'clipping')
  target='joint_weak' if name=='gain_matched' else 'native'
  def metric(indices):
   m=lambda v:float(np.mean([np.mean(v[i]) for i in indices]))
   err=m([(a-b)**2 for a,b in zip(response[name],response[target])]);energy=m([a*a for a in response[name]]);targete=m([a*a for a in response[target]]);intere=m([(a-b)**2 for a,b in zip(response['joint_weak'],response['native'])])
   return {'response_error_rms':np.sqrt(err),'output_response_rms':np.sqrt(energy),'target_response_rms':np.sqrt(targete),'relative_response_error':np.sqrt(err/targete) if targete>floor else None,'intervention_response_rms':np.sqrt(intere),'residual_over_intervention':np.sqrt(err/intere) if intere>floor else None,'response_cosine':float(np.clip(m([a*b for a,b in zip(response[name],response[target])])/np.sqrt(energy*targete),-1,1)) if energy>floor and targete>floor else None,'sham_mean_change_from_native':m([a-b for a,b in zip(outputs[name]['sham'],outputs['native']['sham'])]),'sham_mean_change_from_target':m([a-b for a,b in zip(outputs[name]['sham'],outputs[target]['sham'])]),'command_error_rms':np.sqrt((m([(a-b)**2 for a,b in zip(outputs[name]['pulse'],outputs[target]['pulse'])])+m([(a-b)**2 for a,b in zip(outputs[name]['sham'],outputs[target]['sham'])]))/2)}
  for key,value in metric(range(len(waves))).items():check(hist['variants'][name]['waveform'][key],value,'aggregate waveform '+key)
  for i,w in enumerate(waves):
   stored=next(r for r in hist['variants'][name]['waveform']['per_group'] if r['group']==w['group'])
   for key,value in metric([i]).items():check(stored[key],value,'pergroup waveform '+key)
  group_changes=[];absolute=[]
  for i,w in enumerate(waves):
   row=next(r for r in hist['variants'][name]['response']['per_group'] if r['group']==w['group'])
   nr=float(np.sqrt(np.mean(response['native'][i]**2)));cr=float(np.sqrt(np.mean(response[name][i]**2)));valid=nr>protocol['baseline_epsilon']
   check(row['baseline_response_rms'],nr,'baseline rms');check(row['changed_response_rms'],cr,'changed rms');check(row['absolute_response_change'],cr-nr,'response delta');check(row['fractional_increase'],cr/nr-1 if valid else None,'response fraction');assert row['baseline_valid']==valid
   if valid:group_changes.append(cr/nr-1)
   absolute.append(cr-nr)
  record=hist['variants'][name]['response'];check(record['median_fractional_increase'],float(np.median(group_changes)) if group_changes else None,'median response');check(record['median_absolute_response_change'],float(np.median(absolute)),'median absolute');check(record['positive_fraction'],float(np.mean(np.array(group_changes)>0)) if group_changes else 0.,'positive fraction')
  counters['variant_histories']+=1
  if prepared and name in ['gain_matched','weak_rescue']:
   source=settings[name]['source_variant']; dx=[p-s for p,s in zip(raw[source]['pulse'],raw[source]['sham'])];dy=response[target]
   g=mean([a*b for a,b in zip(dx,dy)])/mean([a*a for a in dx]);check(s['unconstrained_gain'],g,'LS gain');check(s['gain'],np.clip(g,*protocol['gain_bounds']),'bounded gain');check(s['source_response_energy'],mean([a*a for a in dx]),'source energy');check(s['target_response_energy'],mean([a*a for a in dy]),'target energy');check(s['center'],mean(raw[source]['sham']),'center');check(s['mean_fit']['target_mean'],mean(outputs['native']['sham']),'mean target');check(s['mean_fit']['corrected_mean'],mean(outputs[name]['sham']),'mean corrected');assert abs(mean(outputs[name]['sham'])-mean(outputs['native']['sham']))<=1.0001e-10
   counters['gain_fits']+=1

prepared_paths=sorted((args.directory/'prepare').glob('*.json'))
if not prepared_paths:prepared_paths=sorted((args.directory/'calibration').glob('*.json'))
for path in prepared_paths:
 p=json.loads(path.read_text());assert p['protocol_sha256']==digest(protocol);assert p['base_config_sha256']==digest(base)
 analyze_rollouts(p['calibration'],protocol['calibration_seeds'],['native']);history(p['calibration']['fixed_history'],p['variants'],True);counters['prepared_records']+=1
 inherited=json.loads((root/'results/collective_suppression/dynamics_prepared'/f"tau_{p['tau']:g}_noise_{p['noise_tau']:g}_seed_{p['seed']}.json").read_text());assert p['inherited_preparation_sha256']==digest(inherited)
 for name in ['native','joint_weak','joint_strong']:assert p['variants'][name]==inherited['variants'][name]
 confirms=[c for c in (args.directory/'confirm').glob('*.json') if c.name==path.name]
 if not confirms:
  confirms=[c for c in (args.directory/'confirm').glob('*.json') if (lambda o:(o['tau'],o['noise_tau'],o['seed'],o['delay'])==(p['tau'],p['noise_tau'],p['seed'],p['delay']))(json.loads(c.read_text()))]
 if confirms:
  assert len(confirms)==1
  c=json.loads(confirms[0].read_text());assert c['prepared_sha256']==digest(p);assert c['frozen_settings']==p['variants'];analyze_rollouts(c,protocol['confirmation_seeds'],list(p['variants']));history(c['fixed_history'],p['variants']);counters['confirmation_records']+=1
out={'passed':True,'counts':dict(counters),'numeric_absolute_error_max':max_error,'protocol_sha256':digest(protocol),'independent_implementation':True};args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
