import hashlib,json,subprocess
from pathlib import Path
root=Path('/home/ball/transformer-closed-loop-inhibition'); out=root/'results/loop_mechanism'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(out/'run_manifest.json')
audit={'passed':True,'source_commit':manifest['source_commit'],'source_clean_at_launch':not manifest['source_dirty'],'inventories':{}}
for label,relative in [('current','results/loop_mechanism/run_manifest.json'),('delay','results/delay_sweep/run_manifest.json'),('collective','results/collective_suppression/run_manifest.json'),('training','results/timescale_maps/run_manifest.json')]:
 r=read(root/relative)
 counts={}
 for category in ['scientific_sources_sha256','completed_sha256']:
  for p,h in r[category].items():assert sha(root/p)==h,(label,p)
  counts[category]=len(r[category])
 audit['inventories'][label]=counts
for p,h in manifest['scientific_sources_sha256'].items():
 raw=subprocess.check_output(['git','show',manifest['source_commit']+':'+p],cwd=root)
 assert hashlib.sha256(raw).hexdigest()==h,p
assert manifest['source_dirty'] is False
assert set(manifest['stages'])=={'prepare','passive','confirm','mechanism'}
audit['source_bytes_match_commit']=True
audit['preparation_records']=len(list((out/'prepare').glob('*.json')))
audit['confirmation_records']=len(list((out/'confirm').glob('*.json')))
audit['mechanism_records']=len(list((out/'mechanism').glob('*.json')))
audit['passive_records']=len(list((out/'passive').glob('*.json')))
assert [audit[k] for k in ['preparation_records','confirmation_records','mechanism_records','passive_records']]==[60,60,60,4]
Path('/tmp/loop_mechanism_provenance_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print(json.dumps(audit,indent=2))
