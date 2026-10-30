"""Build descriptive controlled comparisons from saved runs; never trains."""
import csv, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
from datetime import datetime, timezone
OUT=ROOT/'results'/('a2_controlled_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
OUT.mkdir(exist_ok=False)
base=ROOT/'results'/'comparison_20261027T001236_958528Z'
runs=[base/'A0_seed42',base/'A1_seed42']
for seed in (42,1042):
    matches=list((ROOT/'results').glob(f'comparison_*/A2_seed{seed}/summary.json'))
    assert len(matches)==1, matches
    runs.append(matches[0].parent)
for name in ('A0','A1'):
    matches=list((ROOT/'results').glob(f'comparison_*/{name}_seed1042/summary.json'))
    runs.extend(p.parent for p in matches if 'evaluated' in json.loads(p.read_text()))
rows=[]; manifest={}
fig,axes=plt.subplots(2,2,figsize=(13,8))
for p in runs:
    c=json.loads((p/'config.json').read_text()); s=json.loads((p/'summary.json').read_text())
    t=list(csv.DictReader((p/'training.csv').open())); e=list(csv.DictReader((p/'evaluation.csv').open()))
    assert len(t)==300 and len(e)==200 and [int(x['reset_seed']) for x in e]==list(range(10000,10200))
    assert c['early_stopping'] is None and c['max_steps']==1000 and not c['evaluation_exploration']
    assert sum(int(x['length']) for x in t)==s['total_steps']
    rows.append({k:s[k] for k in ['configuration','training_seed','mean_return','median_return','std_return','successes','total_steps','training_seconds','evaluation_seconds']})
    for f in p.iterdir():
        if f.is_file(): manifest[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    for col,x in enumerate(('episode','cumulative_steps')):
        for row,names in enumerate((('A0','A2'),('A2','A1'))):
            if c['name'] in names:
                axes[row,col].plot([int(v[x]) for v in t],[float(v['return']) for v in t],label=p.name,alpha=.65,linewidth=.7)
for row in range(2):
    for col in range(2):
        axes[row,col].set(xlabel=('Episode','Actual environment steps')[col],ylabel='Raw training return',title=('A0 vs A2: learning rate','A2 vs A1: architecture')[row]); axes[row,col].legend()
fig.tight_layout(); fig.savefig(OUT/'raw_learning_curves.png',dpi=150)
with (OUT/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'evidence.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
