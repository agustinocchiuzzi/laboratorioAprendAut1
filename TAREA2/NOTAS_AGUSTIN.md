# Corridas y análisis de Agustín (2/11)

Detalle completo, con la explicación de cada script: `AgustinAgente/EXPLICACION_AgenteDQN.md`, Parte 2 (secciones 13 a 21).

## Protocolo

Confirmo `protocols/team_draft.json` (A0, 300 episodios, 8 semillas, evaluación greedy en
10000–10199). Agregué el campo `statistics`:

* Test principal: **U de Mann-Whitney bilateral exacto**, completo contra cada ablación (8 contra 8).
* **Holm-Bonferroni** por las 3 comparaciones, α = 0,05.
* Wilcoxon emparejado por semilla solo como análisis de sensibilidad.
* Script: `scripts/test_ablacion.py` (usa `aggregate` de `study_tools.py`, así valida contra el protocolo).

Falta que Thiago confirme.

## Sin exploración: 8 semillas

`python scripts/run_notebook.py --mode comparison --config A0 --variant no_exploration --seed <s>`,
Windows 11, Python 3.12, torch 2.14.1+cpu, un hilo por corrida, 8 en paralelo (~5 min en total).

| Semilla | Carpeta | Pasos | Entrenamiento | Recompensa media | Éxitos /200 |
|---|---|---|---|---|---|
| 42 | `results/comparison_20261102T210818_015183Z` | 32968 | 18 s | −168.7 | 0 |
| 1042 | `results/comparison_20261102T210827_266117Z` | 228980 | 227 s | 62.6 | 15 |
| 2042 | `results/comparison_20261102T210838_316189Z` | 241521 | 245 s | 150.7 | 74 |
| 3042 | `results/comparison_20261102T210848_166764Z` | 215083 | 222 s | −70.2 | 0 |
| 4042 | `results/comparison_20261102T210858_823203Z` | 116643 | 122 s | −133.3 | 0 |
| 5042 | `results/comparison_20261102T210909_733641Z` | 175372 | 188 s | 65.9 | 33 |
| 6042 | `results/comparison_20261102T210920_232824Z` | 184590 | 198 s | 119.2 | 42 |
| 7042 | `results/comparison_20261102T210931_103280Z` | 167025 | 180 s | 180.3 | 123 |

No corrí semillas del completo: Fran ya tiene las 8.

**Avisos para Thiago** (me pasaron los dos):
* En Windows hay que correr con `PYTHONUTF8=1`; si no, falla al escribir "≥" en `runner.log`.
* Lanzar las corridas con unos segundos de diferencia: si dos arrancan en el mismo instante
  chocan en el nombre de la carpeta (`FileExistsError`).

## Test parcial (falta sin red objetivo)

| Variante | Dif. de medias | p Mann-Whitney | Completo gana |
|---|---|---|---|
| Sin replay | −138.7 | 0.007 | 8/8 |
| Sin exploración | −112.0 | 0.065 | 7/8 |

Los p de Holm no son finales hasta que estén las tres comparaciones.

## Análisis de política

`python scripts/analisis_politica.py --runs <carpetas> --output analisis_politica` → figuras y
`metricas_politica.json` en `analisis_politica/`. Solo carga pesos; no entrena. Usa las semillas
10000–10019 para los episodios reales; no toca las reservadas 30000+.

## Informe

`informe/informe.tex` + `informe/figuras/`: plantilla IEEE con secciones, tablas y figuras, **sin
texto redactado** (la letra prohíbe usar IA para redactar). Compilar en Overleaf.
