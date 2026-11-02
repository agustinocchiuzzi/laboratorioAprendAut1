"""Test estadístico propuesto por Agustín para el estudio de ablación.

Unidad de análisis: la recompensa media de evaluación (200 episodios greedy,
semillas 10000-10199) de cada política entrenada. Cada condición aporta 8 números,
uno por semilla de entrenamiento.

Test principal: U de Mann-Whitney bilateral, completo contra cada ablación.
Corrección: Holm-Bonferroni sobre las 3 comparaciones planeadas (alfa = 0.05).
Análisis de sensibilidad: Wilcoxon de rangos con signo emparejando por semilla.
Tamaños de efecto: diferencia de medias, de medianas, estimador de Hodges-Lehmann
y probabilidad de superioridad P(completo > ablación) = U / (n1 * n2).

Uso, desde TAREA2/:
    python scripts/test_ablacion.py --runs results/comparison_20261031*/A0*_seed* \
        results/comparison_20261102*/A0*_seed* --output ablacion/
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from study_tools import DEFAULT_PROTOCOL, aggregate, read_json

ABLACIONES = ['no_target', 'no_replay', 'no_exploration']
ALFA = 0.05


def holm(pvalores, m):
    """Holm-Bonferroni: p ajustados para m comparaciones planeadas.

    Si falta alguna comparación (m > len(pvalores)) los p devueltos no son finales.
    """
    orden = sorted(range(len(pvalores)), key=lambda i: pvalores[i])
    ajustados = [None] * len(pvalores)
    maximo = 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, min(1.0, (m - rango) * pvalores[i]))   # monotonía
        ajustados[i] = maximo
    return ajustados


def comparar_una(completo, ablacion):
    """Compara dos diccionarios {semilla: media} y devuelve estadísticos y efectos."""
    c = np.array(list(completo.values()))
    a = np.array(list(ablacion.values()))
    u = stats.mannwhitneyu(c, a, alternative='two-sided', method='exact')
    fila = dict(
        n_completo=len(c), n_ablacion=len(a),
        media_completo=float(c.mean()), media_ablacion=float(a.mean()),
        diferencia_media=float(c.mean() - a.mean()),
        diferencia_mediana=float(np.median(c) - np.median(a)),
        hodges_lehmann=float(np.median(np.subtract.outer(c, a))),
        U=float(u.statistic), p_mann_whitney=float(u.pvalue),
        prob_superioridad=float(u.statistic / (len(c) * len(a))),
    )
    comunes = sorted(set(completo) & set(ablacion))
    if len(comunes) >= 5:
        dif = np.array([completo[s] - ablacion[s] for s in comunes])
        w = stats.wilcoxon(dif, alternative='two-sided', method='exact')
        fila.update(n_pares=len(comunes), W=float(w.statistic), p_wilcoxon=float(w.pvalue),
                    completo_gana=int((dif > 0).sum()))
    return fila


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    resumen = aggregate(read_json(args.protocol), args.runs)
    medias = {nombre: {r['training_seed']: r['mean_return'] for r in v['per_training_seed']}
              for nombre, v in resumen['variants'].items()}

    filas = []
    for nombre in ABLACIONES:
        if medias.get(nombre):
            filas.append(dict(variante=nombre, **comparar_una(medias['full'], medias[nombre])))
    completas = resumen['complete'] and len(filas) == len(ABLACIONES)
    for fila, p_holm in zip(filas, holm([f['p_mann_whitney'] for f in filas], m=len(ABLACIONES))):
        fila['p_holm'] = p_holm
        fila['significativo'] = bool(p_holm < ALFA) if completas else None

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'tests.json').write_text(json.dumps(
        dict(alfa=ALFA, comparaciones_planeadas=ABLACIONES, completo=completas, filas=filas),
        indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f"{'variante':<16}{'dif. media':>11}{'HL':>8}{'P(sup)':>8}{'U':>6}"
          f"{'p MW':>9}{'p Holm':>9}{'p Wilc.':>9}  gana")
    for f in filas:
        print(f"{f['variante']:<16}{f['diferencia_media']:11.1f}{f['hodges_lehmann']:8.1f}"
              f"{f['prob_superioridad']:8.2f}{f['U']:6.0f}{f['p_mann_whitney']:9.4f}"
              f"{f['p_holm']:9.4f}{f.get('p_wilcoxon', float('nan')):9.4f}"
              f"  {f.get('completo_gana', '-')}/{f.get('n_pares', '-')}")
    if not completas:
        print('\nAtención: faltan comparaciones; los p de Holm se calcularon con m = 3 pero '
              'pueden cambiar cuando estén todas (no son finales).')


if __name__ == '__main__':
    main()
