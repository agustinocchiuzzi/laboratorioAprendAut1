# Selección consolidada de Naive Bayes

Se repitió exclusivamente la experimentación NB existente con Python **3.12.14**
y scikit-learn **1.9.1**: 2 modelos × 5 suavizados × 3 folds = **30 ajustes**,
seguidos por 2 modelos × 7 variantes × 3 folds = **42 ajustes**. No se amplió la
búsqueda ni se reajustaron otros clasificadores. Se confirman `m=0.1`,
`alpha=0.025` y `plus_points_and_goals` para ambos métodos.

## Protocolo verificado

| Entrenamiento | Validación | Filas train / validación |
|---|---|---:|
| Hasta 2020 | 2021 completo | 13.744 / 364 |
| Hasta 2021 | 2022 completo | 14.108 / 297 |
| Hasta 2022 | 2023 completo | 14.405 / 300 |

La partición es idéntica en ambas etapas y ambos NB. El manifiesto registra las
fechas y hashes de posiciones de cada fold. Cada ajuste crea un pipeline nuevo;
cuantiles y medianas se aprenden solo de su entrenamiento. También registra los
cortes, medianas y cardinalidades por variante/fold. Los historiales se actualizan
solo después de cada fecha completa, incluidos los días anteriores del año de
validación, sin reajustar el clasificador ni el discretizador.

La lectura acotada hace una primera pasada solo por la columna fecha y excluye
las filas posteriores a 2023 antes de interpretar marcadores, limpiar registros
o construir etiquetas. La huella de datos corresponde exclusivamente al prefijo
admitido hasta 2023. No se consultaron resultados ni se evaluó 2024–2025.

## Selección en dos etapas

1. **Suavizado con las seis tasas originales.** Se mantienen `m` en
   `[0.1, 1, 10, 100, 1000]` y `alpha` en `[0.025, 0.25, 2.5, 25, 250]`.
   El criterio es macro-F1 medio anual con E/L/V e igual peso para los tres
   años, antes de redondear. Ante empate exacto se prefiere menor suavizado.
   Los primeros tres valores empatan en 0.363482: se eligen 0.1 y 0.025.
   El error medio de entrenamiento es 0.516039 y el de validación 0.565633.
2. **Atributos con ese suavizado fijo.** Se repiten solo las siete variantes del
   plan original. Selecciona macro-F1 medio; empates favorecen menos columnas y
   luego el orden declarado. El diagnóstico nunca-E no compite. No se vuelve a
   optimizar `m/alpha` sobre diez atributos ni se hace una búsqueda conjunta.

| Configuración, igual resultado para ambos NB | Accuracy | Macro-F1 | DE macro-F1 |
|---|---:|---:|---:|
| Seis tasas | 0.434367 | 0.363482 | 0.032202 |
| Diez atributos: + puntos y diferencia de gol | 0.427650 | 0.385798 | 0.033246 |

La variante de diez atributos mejora macro-F1 y reduce accuracy; se elige por el
criterio predeclarado. Su error medio train/validación es 0.526476/0.572350.
El macro-F1 por año es 0.373586 (2021), 0.431225 (2022) y 0.352584 (2023).
Son resultados de selección sobre folds reutilizados, no estimaciones
independientes de generalización. Las desviaciones son poblacionales (`ddof=0`).

## Configuración seleccionada antes del reajuste final

- **NB propio:** `MEstimateCategoricalNB(m=0.1, min_categories=4)`.
- **sklearn:** `CategoricalNB(alpha=0.025, min_categories=4, force_alpha=True,
  fit_prior=True, class_prior=None)`.
- **Ambos:** `make_feature_pipeline(..., 'discrete', columns)`, tres clases E/L/V,
  variante `plus_points_and_goals`, con estas columnas en este orden:

```text
home_win_rate_last_5
away_win_rate_last_5
home_win_rate_season
away_win_rate_season
home_win_rate_as_home_all
home_win_rate_h2h_as_home
home_points_per_match_5
away_points_per_match_5
home_goal_diff_per_match_5
away_goal_diff_per_match_5
```

Las cuatro últimas son promedios de hasta cinco partidos anteriores disponibles
por equipo. Los puntos son 3/1/0 y la diferencia es goles a favor menos goles en
contra. Sin antecedentes se conservan 4/3 puntos y 0 de diferencia. Se mantienen
los cortes 0.3/0.6 para las dos tasas recientes y terciles de entrenamiento para
las otras ocho columnas. Medianas e imputación conservan la política existente.

Se comprobó **K=4 en cada atributo, variante y fold**, incluido cero. El prior de
categorías es uniforme y el prior de clase es empírico en ambos métodos. Por eso
`(n+m/4)/(n_c+m)` equivale a `(n+alpha)/(n_c+4*alpha)` cuando `alpha=m/4`.
Con cardinalidades diferentes se necesitaría `alpha_j=m/K_j`; la equivalencia
con un único alpha no es una propiedad general de cualquier discretización.

## Vigencia y reproducción

Los manifiestos históricos tenían hashes anteriores a la corrección numérica
de NB y a `min_categories`. La revalidación no cambió ninguna de las **13.454
predicciones de validación** de los dos NB y las siete variantes, ni en tres
clases ni en el diagnóstico. Las diferencias de las métricas leídas desde CSV
son menores que `7e-17`; no cambió ninguna selección.

Los directorios generales `results/validation/` y `results/feature_experiments/`
se conservan intactos, con su procedencia histórica. La evidencia vigente de NB
está en `results/naive_bayes/`. El notebook de entrega calcula las filas NB
desde cero y distingue los otros resultados históricos, que son opcionales.
No se cambian hashes antiguos para hacerlos pasar por una corrida nueva.

La selección queda documentada como antecedente de la evaluación final; para
reproducir la entrega completa desde el ZIP, usar las instrucciones portables de
[nb_delivery.md](nb_delivery.md). No es necesario regenerar esta evidencia para
ejecutar el notebook.

- [Manifiesto y auditoría por fold](../results/naive_bayes/manifest.json).
- [Configuraciones finales exactas](../results/naive_bayes/final_configurations.json).
- [Curva NB propio](../results/naive_bayes/figures/nb_propio.png) y
  [curva CategoricalNB](../results/naive_bayes/figures/categorical_nb.png).
- [Grilla de suavizado](../results/naive_bayes/validation_summary.csv) y
  [comparación de atributos](../results/naive_bayes/feature_summary.csv).
- [Predicciones hasta 2023](../results/naive_bayes/validation_predictions.csv).

`final_nb_pipelines(load_nb_manifest(ROOT))` permite inspeccionar la selección
histórica sin ajustar modelos. El notebook de entrega construye la selección
en memoria, sin cargar ese manifiesto: sus pipelines quedan en
`NB_FINAL_PIPELINES` y sus diez columnas en `NB_FINAL_COLUMNS`. La sección 14
reproduce el reajuste final y la evaluación reservada.

Actualización: esa etapa está completada en [nb_final_evaluation.md](nb_final_evaluation.md).
La evidencia de selección aquí descrita permanece intacta; la evaluación y el
reajuste final tienen un manifiesto independiente en `results/naive_bayes/final/`.
La [entrega integrada](nb_delivery.md) se verifica con
`.venv/bin/python scripts/verify_final_nb.py` después de crear el entorno Python
3.12 indicado allí, y escribe evidencia separada en `results/nb_delivery/`.
