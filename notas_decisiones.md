# Decisiones de modelado

Este documento registra decisiones para que los experimentos sean comparables.
Las marcadas **por confirmar** deben contrastarse con las notas del curso.
La política vigente y la auditoría reproducible se documentan en
[docs/data_policy.md](docs/data_policy.md); prevalecen sobre salidas históricas.

## Variable objetivo

- Solo se admiten registros F sin conflictos de fecha/par de equipos; las
  filas E/P y todas las variantes de un encuentro ambiguo se excluyen también
  de los historiales, sin reconstruir goles.
- `L`: `gh > ga`.
- `V`: `gh < ga`.
- `E`: `gh == ga`.
- Los goles solo construyen la etiqueta y atributos historicos de partidos
  anteriores; nunca se usan los goles del partido que se predice.

## Particion temporal

- Entrenamiento final: desde 1932 hasta 2023 inclusive.
- Evaluacion final: 2024 y 2025.
- Selección de hiperparámetros: folds comunes de años completos 2021, 2022 y
  2023, con entrenamiento expansivo anterior a cada año. `src/evaluation.py`
  crea un pipeline nuevo por fold para todos los modelos.
- Dentro de cada fold una fecha completa pertenece a un único lado de la
  partición; un año validado puede integrar el entrenamiento de folds posteriores.
- La evaluacion 2024-2025 no participa en la seleccion de atributos ni parametros.

## Politica de atributos historicos

La corrida representa predicciones sucesivas: un partido puede usar resultados de
fechas anteriores, incluso si esas fechas pertenecen a 2024-2025. El modelo no se
reentrena durante test. Partidos del mismo dia no se actualizan entre si.

Los casos sin historial previo (denominador = 0: debut del equipo, sin partidos
en la ventana, primer partido del ano, primer head-to-head) se imputan con un
valor neutro constante, distinto de una tasa real `0.0`:

- `NEUTRAL_WIN_RATE = 0.5` para toda proporcion de victorias;
- `NEUTRAL_POINTS_PER_MATCH = 4/3` (puntos esperados por partido con prior
  uniforme sobre `L`/`E`/`V`) para puntos por partido;
- `NEUTRAL_GOAL_DIFF_PER_MATCH = 0.0` para diferencia de gol por partido.

Asi, "no ganó ningún partido con historial" queda en `0.0` real mientras que
"no hay historial" queda en el neutro, pero el neutro puede coincidir con una tasa real de 0.5; no identifica
por sí solo la ausencia de historial. El caso `denominador = 0` se resuelve explicitamente en
`src/features.py`; nunca cae en un `0/0` silencioso. Se incorporan las
siguientes tasas, siempre calculadas antes de la fecha del partido: victorias en
los últimos cinco partidos de cada equipo, victorias de cada equipo dentro del
año calendario, victorias históricas del local actuando de local y victorias del
local frente a ese visitante con la misma localía. El último atributo es
head-to-head orientado: no mezcla partidos con la localía invertida.

## Atributos del modelo

No se usan nombres de equipos ni el mes como atributos. El modelo se entrena solo
con seis tasas de victoria: `win_rate_last_5` (local y visitante), `win_rate_season`
(local y visitante), `win_rate_as_home_all` (local) y `win_rate_h2h_as_home`
(local). Las columnas restantes calculadas por `build_causal_match_features`
quedan para auditoria y para los datasets procesados.

## Discretizacion

Cada tasa se convierte en tres categorias (`baja`, `media`, `alta`):

- `win_rate_last_5` (valores casi discretos: 0, 0.2, 0.4, 0.6, 0.8, 1.0): cortes
  fijos `[0.3, 0.6]`, es decir 0-1 triunfos de 5 = baja, 2 = media, 3+ = alta cuando hay cinco antecedentes;
  con menos se usa la proporción sobre los disponibles.
  Al ser fijos no dependen de los datos y no requieren ajuste con train (sin
  riesgo de leakage).
- `win_rate_season`, `win_rate_as_home_all` y `win_rate_h2h_as_home`: cuantiles
  (equal-frequency) con tres bines, calculados **solo con train** dentro de cada
  fold y reutilizados en validacion/test. Los valores repetidos pueden producir grupos desiguales o menos de tres
  intervalos. No se fuerza un tercio exacto por categoría.

Comparacion opcional para el informe: congelar todos los historiales al 31/12/2023
y medir cuanto cambia el resultado. No mezclar ambas politicas en una misma tabla.

## Baseline de diez anos

Para cada equipo se calcula:

```text
partidos ganados / partidos jugados
```

usando las tasas causales de `build_causal_match_features` en la ventana
`[fecha - 10 años, fecha)`. Incluye fechas anteriores de validación/test, igual
que los demás modelos; sin antecedentes usa 0.5. `fit` no almacena un historial
congelado y `predict` no necesita etiquetas ni modifica estado. Se predice `L` o `V` segun cual equipo tenga mayor proporcion. Si hay
empate exacto, se elige `L` de forma deterministica. **Por confirmar:** preguntar
si el equipo docente espera otro desempate.

## Naive Bayes propio

Los atributos se discretizan con `MixedTypeDiscretizer` dentro de cada fold:
`win_rate_last_5` con cortes fijos, el resto con cuantiles fit-en-train (ver
[Discretizacion](#discretizacion)). No quedan atributos categoricos.

```text
P(X_j=v | Y=c) = (n_jvc + m * p_jv) / (n_c + m)
p_jv = 1 / cantidad_de_valores_del_atributo_j
```

Se suman log-probabilidades para evitar underflow. **Por confirmar:** validar que
el curso define `m` con prior uniforme y no con frecuencias marginales.

## Arbol propio

Se implementa ID3 categorico:

- criterio: entropia y ganancia de informacion;
- split multiway;
- cada atributo se usa a lo sumo una vez por rama;
- se detiene si la mejor ganancia no supera `min_info_gain`;
- valores desconocidos usan la distribucion mayoritaria del nodo;
- profundidad maxima inicial: 8.

**Por confirmar:** validar si se esperaba ID3 multiway o un arbol binario con
umbrales numericos.

## Metricas

- principal para seleccion: macro-F1;
- finales: accuracy, macro-F1, precision/recall/F1 por clase;
- matriz de confusion con orden `E`, `L`, `V`;
- analisis cualitativo a partir de las predicciones sobre test del notebook.

## Comparadores y selección de hiperparámetros

- `src/model_selection.py` declara las grillas y los desempates antes de la
  corrida; `scripts/run_validation.py` ejecuta solo validación hasta 2023.
- Los cuatro métodos requeridos usan los mismos tres folds y macro-F1 medio.
  Se conservan los dos árboles de referencia y el baseline como controles.
- CategoricalNB recibe los mismos códigos que el NB propio, con cuatro categorías
  incluyendo cero. Su grilla `alpha=m/4` permite contrastar el suavizado equivalente.
- Random Forest mantiene 300 árboles y semilla 42, y selecciona profundidad y
  hoja mínima. Se conserva la configuración original sin límite/hoja 1.
- Las curvas registran error `1-accuracy` y macro-F1 de train y validación;
  solo el macro-F1 de validación selecciona parámetros.
- Resultados, grillas y limitaciones: [docs/validation_findings.md](docs/validation_findings.md).
- No se ejecutó el test ni el reajuste final. El notebook los deja desactivados
  con `RUN_FINAL_TEST = False`. Las salidas históricas siguen invalidadas.

## Reproducibilidad

- Python 3.12;
- scikit-learn 1.9;
- semilla 42;
- datos limpios, figuras y modelos generados desde comandos documentados;
- ninguna salida generada se considera fuente.

## Comparación acotada de atributos

Se ejecutó el plan de [atributos](docs/feature_experiment_plan.md) con los
hiperparámetros previamente elegidos, sin reajustarlos. El
[resumen de validación](docs/feature_findings.md) registra siete variantes y el
diagnóstico sin predecir E, evaluado siempre contra las tres clases reales.
Random Forest conserva las seis entradas actuales; ID3 y ambos NB seleccionan
las entradas actuales más puntos y diferencia de gol recientes de ambos equipos.
No se realizó evaluación final ni reajuste con todo el histórico.
