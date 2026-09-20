# Entrega e integración de Naive Bayes

Esta es la guía de entrega de NB. `informe.tex` contiene el texto integrado y
`notebook.ipynb` ejecuta la selección, el ajuste final y el análisis desde el
ZIP; no necesita resultados, modelos o CSV precalculados.

## Configuración cerrada

Los dos clasificadores usan la variante `plus_points_and_goals`, las mismas diez
entradas y decisión de tres clases E/L/V:

```text
home_win_rate_last_5             away_win_rate_last_5
home_win_rate_season             away_win_rate_season
home_win_rate_as_home_all        home_win_rate_h2h_as_home
home_points_per_match_5          away_points_per_match_5
home_goal_diff_per_match_5       away_goal_diff_per_match_5
```

- NB propio: `MEstimateCategoricalNB(m=0.1, min_categories=4)`.
- Referencia: `CategoricalNB(alpha=0.025, min_categories=4, force_alpha=True)`.
- Las dos tasas de últimos cinco partidos usan cortes fijos 0.3/0.6; los
  terciles y medianas de las otras ocho entradas se aprenden sólo de train.
- La selección usa train expansivo y validación anual en 2021, 2022 y 2023.
  Primero compara cinco suavizados (30 ajustes entre ambos NB); después compara
  siete variantes con el suavizado fijo (42 ajustes). El test no interviene.

Con cuatro códigos declarados, incluido el cero reservado, `alpha=m/4=0.025`.
Ambos modelos tienen las mismas entradas, conteos y priors de clase empíricos;
la ejecución verificada encontró cero diferencias entre sus 472 predicciones.

## Reproducción portátil desde el ZIP original

Desde la raíz del repositorio, con Python 3.12 disponible como `python3.12`:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/verify_final_nb.py
```

El verificador exige Python 3.12 y scikit-learn 1.9, crea una copia temporal
con sólo `notebook.ipynb`, `requirements.txt`, `src/*.py` y
`data/raw/futbol_uruguayo.zip`, y ejecuta las 21 celdas de código. No puede
leer `results/` dentro de esa copia. Sus salidas regenerables se escriben en
`results/nb_delivery/` y no son entradas del notebook.

La evidencia vigente corresponde a Python 3.12.14 y scikit-learn 1.9.1:

- 37 ajustes por NB (15 de suavizado, 21 de atributos y uno final), un ajuste
  del baseline y cero ajustes de ID3 o Random Forest.
- Selección cerrada y 472 predicciones idénticas a la evidencia previa.
- NB propio y CategoricalNB: accuracy 0.478814, macro-F1 0.445369, 226/472.
- Baseline de diez años: accuracy 0.461864, macro-F1 0.356446, 218/472.
- Antes de retirar la suite de tests se aprobaron 41 pruebas: implementación NB, equivalencia,
  preprocesamiento, causalidad, límites temporales, selección y evaluación.

`results/nb_delivery/verification.json` registra versiones, hashes, ajustes y
la comparación. Los detalles de implementación, selección y evaluación se
mantienen en `naive_bayes.md`, `nb_selection.md` y `nb_final_evaluation.md`;
esta guía es la fuente para reproducir y conectar NB.

## Qué deben conectar los demás integrantes

- Mantener sin cambios las filas finales de NB y baseline del informe:
  accuracy/macro-F1 0.4788/0.4454 para ambos NB y 0.4619/0.3564 para el
  baseline. Las filas de ID3 y Random Forest siguen como `Pend.`.
- ID3 debe integrar su variante ya seleccionada antes de evaluar test. Random
  Forest y los árboles de referencia deben completar su ajuste final, métricas,
  matrices y análisis sin usar las métricas NB para tomar decisiones.
- Las ramas finales de árboles siguen desactivadas en el notebook (`RUN_FINAL_TEST=False`).
  Ejecutar el notebook no significa que ID3 o Random Forest estén evaluados.
- Tras incorporar esos resultados, revisar resumen, conclusiones conjuntas y
  el límite de cinco páginas del informe. No repetir la selección NB ni
  modificar sus parámetros según el test.

## Declaración de IA y datos personales

El informe declara únicamente el hecho comprobable de que Codex asistió la
parte NB en análisis, integración y verificaciones. No hay datos comprobados
para completar autoría, correos, otras herramientas usadas personalmente ni la
revisión humana. Esos datos deben ser aportados y confirmados por cada
integrante; no se infieren desde el repositorio.

## Integración y verificaciones

La suite `tests/` se retiró por decisión del equipo al preparar la integración;
los conteos de pruebas anteriores documentan verificaciones ya realizadas, no
una suite incluida en esta entrega. Se conserva `scripts/verify_final_nb.py`
para comprobar el flujo completo desde el ZIP y contrastar selección y predicciones.

Los experimentos compartidos de validación y atributos son evidencia histórica:
para repetirlos con sus versiones originales seguir
[feature_findings.md](feature_findings.md). Sus manifiestos se conservan intactos.
Integrar NB no implica que la evaluación de ID3/Random Forest ni la entrega
conjunta estén terminadas.
