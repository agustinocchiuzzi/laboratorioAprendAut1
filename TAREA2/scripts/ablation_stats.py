"""Compara cada ablación contra el DQN completo usando las corridas guardadas.

Unidad: la recompensa media de evaluación de cada política entrenada (una por
semilla de entrenamiento). Como todas las condiciones usan las mismas semillas, se
comparan semilla a semilla. El test estadístico todavía no está: lo propone Agustín
y se agrega en `comparar` cuando el equipo lo confirme.

Uso, desde TAREA2/:
    python scripts/ablation_stats.py --runs results/comparison_*/A0*_seed* --output ablacion/
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from study_tools import DEFAULT_PROTOCOL, aggregate, read_json

ABLACIONES = ['no_target', 'no_replay', 'no_exploration']
NOMBRES = {'full': 'DQN completo', 'no_target': 'Sin red objetivo',
           'no_replay': 'Sin replay', 'no_exploration': 'Sin exploración'}


def comparar(variantes):
    """Diferencias semilla a semilla de cada ablación contra el completo (sin test todavía)."""
    completo = {r['training_seed']: r['mean_return'] for r in variantes['full']['per_training_seed']}
    filas = []
    for nombre in ABLACIONES:
        ablacion = {r['training_seed']: r['mean_return'] for r in variantes[nombre]['per_training_seed']}
        semillas = sorted(set(completo) & set(ablacion))
        if not semillas:
            continue
        a = np.array([ablacion[s] for s in semillas])
        c = np.array([completo[s] for s in semillas])
        filas.append(dict(variante=nombre, semillas=semillas, n=len(semillas),
                          media_ablacion=float(a.mean()), media_completo=float(c.mean()),
                          diferencia_media=float((a - c).mean()),
                          diferencia_mediana=float(np.median(a - c)),
                          completo_gana=int((c > a).sum())))
    return filas


def graficar(variantes, ruta):
    """Un punto por semilla de entrenamiento; la raya negra es la media."""
    fig, ax = plt.subplots(figsize=(7, 4))
    orden = ['full'] + ABLACIONES
    for i, nombre in enumerate(orden):
        medias = [r['mean_return'] for r in variantes[nombre]['per_training_seed']]
        if not medias:
            continue
        x = np.full(len(medias), i) + np.linspace(-0.1, 0.1, len(medias))
        ax.scatter(x, medias, s=30, zorder=3)
        ax.hlines(np.mean(medias), i - 0.25, i + 0.25, color='black')
    ax.axhline(200, color='green', linestyle='--', linewidth=1, label='Umbral de éxito (200)')
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([NOMBRES[n] for n in orden])
    ax.set_ylabel('Recompensa media de evaluación (200 ep)')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    resumen = aggregate(read_json(args.protocol), args.runs)
    comparaciones = comparar(resumen['variants'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'ablacion.json').write_text(json.dumps(
        dict(resumen=resumen, comparaciones=comparaciones),
        indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    graficar(resumen['variants'], args.output / 'ablacion.png')

    print('Recompensa media de evaluación por semilla de entrenamiento:')
    for nombre in ['full'] + ABLACIONES:
        filas = resumen['variants'][nombre]['per_training_seed']
        valores = '  '.join(f"{r['training_seed']}:{r['mean_return']:7.1f}" for r in filas)
        print(f"  {NOMBRES[nombre]:<16} {valores or '(sin corridas)'}")
    print('\nComparación contra el DQN completo (semillas en común):')
    for f in comparaciones:
        print(f"  {NOMBRES[f['variante']]:<16} n={f['n']}  dif. media={f['diferencia_media']:8.1f}  "
              f"completo gana {f['completo_gana']}/{f['n']}")
    if not resumen['complete']:
        print('\nAtención: faltan semillas; los resultados son parciales.')


if __name__ == '__main__':
    main()
