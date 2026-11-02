"""Build reproducible descriptive artifacts from Thiago's saved training manifest."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['status'] != 'completed_and_validated':
        raise ValueError('Expected a completed, validated training manifest')
    folders = [Path(manifest['existing_run'])] + [Path(j['folder']) for j in manifest['jobs']]
    # Resolve old absolute paths using the stable results-relative suffix on another machine.
    folders = [p if p.exists() else ROOT / 'results' / p.parent.name / p.name for p in folders]
    from study_tools import aggregate, read_json
    protocol = read_json(ROOT / 'protocols/team_draft.json')
    summary = aggregate(protocol, folders)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    series, metrics, data, selected = [], [], {}, set(folders)
    for folder in folders:
        config = read_json(folder / 'config.json')
        result = read_json(folder / 'summary.json')
        if config['source_notebook_sha256'] != manifest['source_notebook_sha256']:
            raise ValueError(f'{folder}: source notebook hash differs')
        variant = 'full' if config['use_target'] else 'no_target'
        seed = config['training_seed']
        with (folder / 'training.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        raw = np.array([float(r['return']) for r in rows])
        smooth = np.array([raw[max(0, i - 19):i + 1].mean() for i in range(len(raw))])
        data[variant, seed] = smooth
        for r, value in zip(rows, smooth):
            series.append(dict(variant=variant, seed=seed, episode=int(r['episode']),
                cumulative_steps=int(r['cumulative_steps']), raw_return=float(r['return']),
                trailing_mean_20=float(value)))
        metrics.append(dict(variant=variant, seed=seed, mean_return=result['mean_return'],
            success_rate=result['successes'] / result['evaluated'],
            successes=result['successes'], evaluated=result['evaluated'],
            training_steps=result['total_steps'], training_seconds=result['training_seconds'],
            folder=str(folder.relative_to(ROOT))))
    full_seeds = sorted(s for v, s in data if v == 'full')
    target_seeds = sorted(s for v, s in data if v == 'no_target')
    common = sorted(set(full_seeds) & set(target_seeds))
    assert full_seeds == [42, 1042, 2042] and len(target_seeds) == 8 and len(metrics) == 11
    paired = []
    for seed in common:
        f = next(r for r in metrics if r['variant'] == 'full' and r['seed'] == seed)
        t = next(r for r in metrics if r['variant'] == 'no_target' and r['seed'] == seed)
        paired.append(dict(seed=seed, full_mean_return=f['mean_return'],
            no_target_mean_return=t['mean_return'], difference_full_minus_no_target=f['mean_return']-t['mean_return']))

    audit = []
    for path in sorted((ROOT / 'results').glob('*/config.json')):
        # Included for compatibility with flat result directories if present.
        audit.append(dict(folder=str(path.parent.relative_to(ROOT)), status='not_selected', detail='Outside selected manifest'))
    for path in sorted((ROOT / 'results').glob('*/*/config.json')):
        folder = path.parent
        if folder in selected:
            status, detail = 'included', 'Selected manifest; validated'
        else:
            try:
                aggregate(protocol, [folder])
                status, detail = 'compatible_not_selected', 'Historical run; not added automatically (may duplicate seed or differ in source revision)'
            except (ValueError, KeyError, OSError) as error:
                status, detail = 'excluded', str(error).replace(str(ROOT) + '/', '')
        audit.append(dict(folder=str(folder.relative_to(ROOT)), status=status, detail=detail))

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    summary['sources'] = [str(p.relative_to(ROOT)) for p in folders]
    (out / 'validated_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_csv(out / 'metrics.csv', sorted(metrics, key=lambda r: (r['variant'], r['seed'])))
    write_csv(out / 'learning_curves.csv', series)
    write_csv(out / 'paired_evaluation.csv', paired)
    write_csv(out / 'local_run_audit.csv', audit)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    x = np.arange(1, 301)
    for ax, variant, seeds, title in [
        (axes[0], 'full', full_seeds, 'Agente completo: 3 semillas disponibles'),
        (axes[1], 'no_target', target_seeds, 'Sin red objetivo: 8 semillas completas')]:
        for seed in seeds:
            ax.plot(x, data[variant, seed], label=str(seed), linewidth=1.3)
        ax.set(title=title, xlabel='Episodio de entrenamiento', ylabel='Retorno suavizado')
        ax.legend(title='Semilla', ncol=4, fontsize=8)
        ax.grid(alpha=.2)
    for variant, label, color in [('full', 'Completo', '#2563eb'), ('no_target', 'Sin red objetivo', '#c2410c')]:
        values = np.stack([data[variant, s] for s in common])
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        axes[2].plot(x, mean, label=f'{label}: n={len(common)}', color=color)
        axes[2].fill_between(x, mean-sd, mean+sd, color=color, alpha=.15)
    axes[2].set(title='Comparación preliminar: solo las mismas 3 semillas',
        xlabel='Episodio de entrenamiento', ylabel='Retorno suavizado')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=.2)
    fig.suptitle('Curvas de entrenamiento · A0 · estudio parcial\nMedia móvil causal de hasta 20 episodios; banda inferior: ±1 DE entre semillas, no IC', fontsize=12)
    for suffix in ['png', 'pdf']:
        fig.savefig(out / f'learning_curves.{suffix}', dpi=160)
    plt.close(fig)

    missing = '\n'.join(f'- {v}: {row["missing_training_seeds"]}' for v, row in summary['variants'].items() if row['missing_training_seeds'])
    report = f'''# Análisis descriptivo parcial de Thiago

11 políticas validadas: 8 sin red objetivo y 3 completas. No hay corridas finales compatibles sin replay o sin exploración disponibles en este conjunto local; los pilotos y resultados históricos incompatibles se registran en local_run_audit.csv.

## Archivos

- [Curvas PNG](learning_curves.png) y [PDF](learning_curves.pdf).
- [Datos de curvas](learning_curves.csv), con retornos originales y suavizados.
- [Métricas por política](metrics.csv).
- [Evaluación emparejada](paired_evaluation.csv): completo menos sin red objetivo, para semillas 42, 1042 y 2042.
- [Auditoría de resultados locales](local_run_audit.csv).
- [Resumen validado](validated_summary.json).
- [Paquete para compartir](thiago_runs.zip), con los archivos originales de las 11 políticas y estos resultados.

## Cómo leer las curvas

Cada línea de los dos primeros paneles representa una política entrenada con una semilla. El retorno se suaviza con una media de los últimos 20 episodios (menos al inicio). El tercer panel usa únicamente las 3 semillas presentes en ambas condiciones: muestra la media de las curvas suavizadas y ±1 desviación estándar muestral entre semillas. La banda no es un intervalo de confianza. No se rellenan semillas faltantes ni se seleccionan las mejores corridas.

Los CSV conservan episodios y pasos acumulados. El presupuesto es igual en episodios; los pasos y tiempos efectivamente consumidos pueden diferir. No inferir eficiencia computacional a partir de tiempos de ejecuciones con distinta carga de máquina.

## Límites

El estudio de 32 políticas sigue incompleto. La comparación emparejada tiene solo 3 semillas; no se realizaron pruebas estadísticas ni se declara significación. Un resultado bajo es un resultado válido, no un motivo para descartar una semilla.

Evaluación greedy de los pesos finales: 200 episodios, semillas de desarrollo 10000–10199, usadas previamente para calibración. No se usaron semillas reservadas. La unidad de análisis es la política entrenada, no cada episodio de evaluación.

A0 fue elegida por el usuario. La compatibilidad se validó contra team_draft.json; las demás decisiones de ese documento siguen figurando como borrador. El manifiesto identifica la revisión y el hash del notebook de las corridas.

## Corridas que faltan reunir

{missing}

Pedir las carpetas completas a Fran y Agustín y validar su compatibilidad antes de incorporarlas. No reemplazar esos faltantes con pilotos ni con entrenamientos de presupuestos distintos.
'''
    (out / 'README.md').write_text(report)
    html = f'''<!doctype html><html lang="es"><meta charset="utf-8"><style>
body{{margin:0;font-family:var(--font-sans);color:var(--foreground)}}
p{{line-height:1.5}}img{{width:100%;height:auto;display:block}}
</style><p><strong>Estudio parcial: 11 de 32 políticas.</strong> A0; 300 episodios por entrenamiento. La comparación inferior usa las mismas tres semillas en ambas condiciones.</p>
<img src="{out.resolve() / 'learning_curves.png'}" alt="Curvas individuales de entrenamiento y media con desviación estándar para las tres semillas comunes">
<p>Evaluación final del estudio y pruebas estadísticas pendientes. Las bandas representan dispersión entre semillas, no intervalos de confianza.</p></html>'''
    (out / 'curves.html').write_text(html)
    artifacts = sorted(out.iterdir())
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in folders for p in sorted(folder.iterdir()) if p.is_file()}
    (out / 'source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    with zipfile.ZipFile(out / 'thiago_runs.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for folder in folders:
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    archive.write(path, str(path.relative_to(ROOT)))
        for path in artifacts + [out / 'source_hashes.json']:
            archive.write(path, 'analysis/' + path.name)
        archive.write(args.manifest, 'analysis/training_manifest.json')
        archive.write(ROOT / 'protocols/team_draft.json', 'analysis/team_draft.json')
        archive.write(ROOT / 'protocols/reference_selection.json', 'analysis/reference_selection.json')
        archive.write(Path(__file__), 'analysis/analyze_thiago_runs.py')
    with zipfile.ZipFile(out / 'thiago_runs.zip') as archive:
        assert archive.testzip() is None
    print(json.dumps({'output': str(out.resolve()), 'runs': len(metrics), 'paired_seeds': common,
        'audit_counts': {s: sum(r['status'] == s for r in audit) for s in sorted({r['status'] for r in audit})},
        'package_bytes': (out / 'thiago_runs.zip').stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
