"""Select and validate 32 policies without selecting seeds by performance."""
import csv
import hashlib
import json
import os
from pathlib import Path

from study_tools import aggregate, read_json
from test_ablacion import comparar_una, holm, ABLACIONES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/team_study'


def main():
    own = read_json(ROOT / 'preparation/thiago_analysis_20261101/validated_summary.json')['sources']
    fran = read_json(ROOT / 'preparation/remote_fran_0686cbf/import_manifest.json')['runs']
    selected, extra = [], []
    for rel in own:
        c = read_json(ROOT / rel / 'config.json')
        (selected if not c['use_target'] else extra).append(rel)
    selected += fran
    exploration = []
    for path in sorted((ROOT / 'results').glob('comparison_20261102*/A0_no_exploration_seed*/config.json')):
        if read_json(path)['episodes'] == 300:
            exploration.append(str(path.parent.relative_to(ROOT)))
    selected += exploration
    protocol = read_json(ROOT / 'protocols/team_final.json')
    summary = aggregate(protocol, [ROOT / p for p in selected])
    assert len(selected) == 32 and summary['complete']
    summary['sources'] = selected
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'validated_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    rows = []
    for rel in selected + extra:
        c, s = read_json(ROOT / rel / 'config.json'), read_json(ROOT / rel / 'summary.json')
        v = 'no_target' if not c['use_target'] else 'no_replay' if not c['use_replay'] else 'no_exploration' if not c['use_exploration'] else 'full'
        owner = 'Thiago' if rel in own else 'Fran' if rel in fran else 'Agustín'
        rows.append(dict(variant=v, seed=c['training_seed'], owner=owner,
            selected=rel in selected, mean_return=s['mean_return'], success_rate=s['successes']/200,
            successes=s['successes'], evaluated=200, training_steps=s['total_steps'],
            torch=c['packages']['torch'], hardware=c['hardware'], folder=rel))
    with (OUT / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    means = {v: {r['training_seed']: r['mean_return'] for r in item['per_training_seed']}
        for v, item in summary['variants'].items()}
    tests = [dict(variante=v, **comparar_una(means['full'], means[v])) for v in ABLACIONES]
    for row, p, w in zip(tests, holm([r['p_mann_whitney'] for r in tests], 3), holm([r['p_wilcoxon'] for r in tests], 3)):
        row.update(p_holm=p, p_wilcoxon_holm=w, significativo=p < .05)
    assert all(r['n_completo'] == r['n_ablacion'] == 8 for r in tests)
    result = dict(completo=True, alfa=.05, filas=tests,
        method_status='Agustin proposal executed for review; not a claim of preregistration or automatic satisfaction of assumptions.',
        limitations='Conditions were trained on different machines/package versions. MW assumes independence; shared seed design also motivates the paired sensitivity analysis. Wilcoxon assumes symmetry of paired differences. Neither test removes platform confounding.')
    (OUT / 'tests.json').write_text(json.dumps(result, indent=2) + '\n')
    manifest = dict(status='32_validated_policies_review_pending', selected_runs=selected,
        additional_full_runs=extra, reference_selection='All eight full runs from Fran, as used by Agustin policy analysis. Selection based on provenance/completeness, never score.',
        source_hashes={rel: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/rel).iterdir() if p.is_file()} for rel in selected},
        statistics_status='Proposed methods executed; team methodological review remains pending.',
        reserved_evaluation_performed=False)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    labels = {'full':'Completo', 'no_target':'Sin red objetivo', 'no_replay':'Sin replay', 'no_exploration':'Sin exploración'}
    for variant in ['full', *ABLACIONES]:
        curves = []
        for row in rows:
            if row['selected'] and row['variant'] == variant:
                with (ROOT / row['folder'] / 'training.csv').open() as handle:
                    values = np.array([float(r['return']) for r in csv.DictReader(handle)])
                curves.append([values[max(0,i-19):i+1].mean() for i in range(len(values))])
        array = np.array(curves)
        assert array.shape == (8, 300)
        mean, sd = array.mean(axis=0), array.std(axis=0, ddof=1)
        line, = ax.plot(np.arange(1,301), mean, label=labels[variant], linewidth=1.3)
        ax.fill_between(np.arange(1,301), mean-sd, mean+sd, color=line.get_color(), alpha=.10)
    ax.set(xlabel='Episodio de entrenamiento', ylabel='Retorno suavizado')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(OUT / 'learning_curves_summary.png', dpi=180)
    fig.savefig(ROOT / 'informe/figuras/curvas_aprendizaje.png', dpi=180)
    plt.close(fig)
    print('Validated 32 selected policies; retained 3 additional full policies.')
    print(json.dumps(tests, indent=2))


if __name__ == '__main__':
    main()
