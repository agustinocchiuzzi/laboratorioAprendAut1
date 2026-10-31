# Corridas de Fran (31/10)

Protocolo: `protocols/team_draft.json` (A0, 300 episodios, evaluación greedy en
semillas 10000–10199). Lanzadas con `scripts/run_notebook.py --mode comparison
--config A0`, en Windows 11, Python 3.12.10, torch 2.15.0+cpu, un hilo por corrida.
Las 11 corridas en paralelo tardaron 148 s en total.

| Variante | Semilla | Carpeta | Pasos | Entrenamiento | Recompensa media | Éxitos /200 |
|---|---|---|---|---|---|---|
| sin replay | 42 | `results/comparison_20261031T172932_700486Z` | 91990 | 52 s | −28.2 | 1 |
| sin replay | 1042 | `results/comparison_20261031T172934_265556Z` | 63829 | 37 s | 122.8 | 86 |
| sin replay | 2042 | `results/comparison_20261031T172936_332164Z` | 106776 | 60 s | 6.2 | 0 |
| sin replay | 3042 | `results/comparison_20261031T172938_474464Z` | 66230 | 38 s | −129.5 | 0 |
| sin replay | 4042 | `results/comparison_20261031T172940_323746Z` | 97625 | 55 s | −129.2 | 0 |
| sin replay | 5042 | `results/comparison_20261031T172942_212531Z` | 75651 | 43 s | 107.6 | 67 |
| sin replay | 6042 | `results/comparison_20261031T172944_154688Z` | 94588 | 53 s | 38.3 | 31 |
| sin replay | 7042 | `results/comparison_20261031T172946_364487Z` | 88765 | 49 s | 5.1 | 4 |
| completo | 42 | `results/comparison_20261031T172948_767233Z` | 158659 | 104 s | 174.9 | 68 |
| completo | 1042 | `results/comparison_20261031T172950_978673Z` | 153312 | 100 s | 183.2 | 85 |
| completo | 2042 | `results/comparison_20261031T172952_979246Z` | 159763 | 105 s | 80.6 | 63 |

## Definición de "sin replay"

Se mantiene la del código integrado: cada actualización usa solo la transición
actual (batch efectivo 1), con `learning_starts=1000` y `train_frequency=4`.
La red hace las mismas actualizaciones que la referencia, pero solo aprende de
1 de cada 4 transiciones. Hay que declararlo en el informe.

## La misma semilla no se reproduce entre máquinas

`completo, semilla 42` ya la había corrido Thiago (macOS ARM, torch 2.14.1:
`results/comparison_20261027T001236_958528Z/A0_seed42`). La configuración es
idéntica, pero el retorno del primer episodio ya difiere en el tercer decimal
(−351.1578 vs −351.1590) y las corridas se separan después. Cada corrida sigue
siendo una política entrenada válida; no se deben esperar resultados idénticos
entre las máquinas de cada uno. Para el estudio se usan las corridas nuevas.

## Comparación estadística

`scripts/ablation_stats.py` valida las corridas contra el protocolo y compara cada
ablación con el completo: Wilcoxon de rangos con signo pareado por semilla (dos
colas) y corrección de Holm por las tres comparaciones. Es una propuesta; el test
lo confirma el equipo. Con las 8 semillas el p mínimo posible es 0.0078.

```sh
python scripts/ablation_stats.py --runs results/comparison_2026103*/A0*_seed* --output ablacion/
```

Hace falta que estén las 32 corridas en la misma carpeta `results/`.
