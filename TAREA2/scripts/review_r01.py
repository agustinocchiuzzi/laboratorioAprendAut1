"""Auditoría R01 sin entrenar ni modificar resultados; salida separada obligatoria."""
import argparse
import ast
import base64
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import scipy
from scipy import stats
from test_ablacion import ABLACIONES, comparar_una, ajustar_familias

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    notebook = json.loads((ROOT/'entrega_borrador.ipynb').read_text())
    cell = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('RUNS ='))
    tree = ast.parse(cell)
    packed = tree.body[0].value.args[0].args[0].args[0]
    runs = json.loads(zlib.decompress(base64.b64decode(ast.literal_eval(packed))))
    manifest = json.loads((ROOT/'analysis/team_study/manifest.json').read_text())
    assert {r['folder'] for r in runs} == set(manifest['selected_runs']) and len(runs) == 32
    means, records, by_key = {}, [], {}
    for run in runs:
        folder = ROOT/run['folder']
        config = json.loads((folder/'config.json').read_text())
        assert config == run['config']
        v = next((name for switch, name in [('use_target','no_target'),('use_replay','no_replay'),('use_exploration','no_exploration')] if not config[switch]), 'full')
        seed = config['training_seed']
        assert (v, seed) not in by_key
        hashes = {}
        for name in ['evaluation','training']:
            p = folder/f'{name}.csv'
            rows = list(csv.DictReader(p.open()))
            assert rows == run[name], (folder, name)
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            assert hashes[p.name] == manifest['source_hashes'][run['folder']][p.name]
        assert [int(x['reset_seed']) for x in run['evaluation']] == list(range(10000,10200))
        assert [int(x['reset_seed']) for x in run['training']] == list(range(seed,seed+300))
        values = np.array([float(x['return']) for x in run['evaluation']])
        assert np.isfinite(values).all()
        mean = float(values.mean())
        assert abs(mean-run['summary']['mean_return']) < 1e-10
        means.setdefault(v,{})[seed] = mean
        by_key[v,seed] = run
        records.append(dict(variant=v, seed=seed, mean=mean, folder=run['folder'], hashes=hashes,
                            platform={k:config.get(k) for k in ['hardware','machine','python','packages','torch_threads','torch_interop_threads','source_revision','source_notebook_sha256']}))
    seeds = list(range(42,7043,1000))
    assert all(sorted(means[v]) == seeds for v in ['full',*ABLACIONES])
    tests = [dict(variante=v, **comparar_una(means['full'],means[v])) for v in ABLACIONES]
    ajustar_familias(tests)
    prefixes = {}
    for v in ABLACIONES:
        prefixes[v] = {}
        for seed in seeds:
            count = 0
            for x,y in zip(by_key['full',seed]['training'], by_key[v,seed]['training']):
                if x != y: break
                count += 1
            prefixes[v][seed] = count
    # Enumeración independiente: mismas dos pruebas, no tests adicionales.
    for t in tests:
        ranks = stats.rankdata(np.abs(t['diferencias']))
        totals = [sum(r for r,b in zip(ranks,bits) if b) for bits in itertools.product([0,1],repeat=8)]
        p = min(1., 2*sum(x <= t['W'] for x in totals)/256)
        assert abs(p-t['p_wilcoxon']) < 1e-14
        us = [sum(c)-36 for c in itertools.combinations(range(1,17),8)]
        p = min(1., 2*sum(u <= min(t['U'],64-t['U']) for u in us)/len(us))
        assert abs(p-t['p_mann_whitney']) < 1e-14
        t['min_abs_difference'] = min(abs(x) for x in t['diferencias'])
        t['min_gap_abs_differences'] = float(np.diff(sorted(abs(x) for x in t['diferencias'])).min())
        pooled = sorted([*means['full'].values(),*means[t['variante']].values()])
        t['min_gap_pooled_means'] = float(np.diff(pooled).min())
    old = json.loads((ROOT/'analysis/team_study/tests.json').read_text())['filas']
    delta = {t['variante']:{k:t[k]-o[k] for k in ['diferencia_media','p_mann_whitney','p_holm','p_wilcoxon','p_wilcoxon_holm']} for t,o in zip(tests,old)}
    result = dict(status='technical_review_report_pending_H03', python=sys.version, scipy=scipy.__version__, numpy=np.__version__,
                  sources=records, means=means, tests=tests, identical_training_row_prefixes=prefixes,
                  difference_from_previous=delta,
                  descriptive={v:dict(mean=float(np.mean(list(m.values()))),sd=float(np.std(list(m.values()),ddof=1))) for v,m in means.items()},
                  checks='32 original/embedded records equal; 6400 returns; seeds and hashes checked; exact p enumerated independently')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'R01_calculos.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    with (args.output/'R01_medias.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['training_seed','full',*ABLACIONES])
        for seed in seeds:w.writerow([seed,*[means[v][seed] for v in ['full',*ABLACIONES]]])
    print(json.dumps({k:result[k] for k in ['tests','identical_training_row_prefixes','descriptive','difference_from_previous']},indent=2))


if __name__ == '__main__':
    main()
