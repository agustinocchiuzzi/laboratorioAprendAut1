# Comparación y selección temporal (sin evaluación final)

Se completaron **31 configuraciones × 3 folds = 93 ajustes**, con 14.705 partidos
admitidos hasta 2023. Las seis tasas, la limpieza y la política histórica son las
ya declaradas: no se probaron atributos nuevos. El período 2024–2025 se excluyó
antes de construir atributos para esta búsqueda. No se reajustó ningún modelo
final con todo el histórico ni se ejecutó la evaluación reservada.

## Protocolo y grillas fijadas antes de la corrida

| Entrenamiento | Validación | Filas train / validación |
|---|---|---:|
| Hasta 2020 | 2021 completo | 13.744 / 364 |
| Hasta 2021 | 2022 completo | 14.108 / 297 |
| Hasta 2022 | 2023 completo | 14.405 / 300 |

Cada combinación ajusta un pipeline nuevo por fold. La discretización aprende
cuantiles/medianas exclusivamente del entrenamiento correspondiente. Las fechas
completas permanecen juntas y los historiales solo incluyen resultados de fechas
anteriores. Modelo y preprocesamiento permanecen fijos dentro de la validación.

| Modelo | Hiperparámetros explorados | Configuraciones |
|---|---|---:|
| ID3 | `min_info_gain`: 0; 0,001; 0,005; 0,01; 0,02; 0,05 | 6 |
| NB propio | `m`: 0,1; 1; 10; 100; 1000 | 5 |
| CategoricalNB | `alpha`: 0,025; 0,25; 2,5; 25; 250 | 5 |
| Random Forest | profundidad: 4, 8, sin límite; hoja mínima: 50, 20, 5, 1 | 12 |
| Controles existentes | DT con códigos, DT con tasas, baseline de diez años | 3 |

El bosque mantiene 300 árboles, `balanced_subsample`, `max_features='sqrt'`,
bootstrap y semilla 42; no se seleccionó el número de árboles. Los árboles de
referencia conservan entropía y semilla 42. ID3 mantiene profundidad máxima 8.
CategoricalNB utiliza `min_categories=4`, incluyendo el código reservado cero,
y `force_alpha=True`.

La métrica de selección es **macro-F1 medio de validación**, con clases fijas
E/L/V, `zero_division=0` e igual peso para cada año. No se promedian ejemplos
agrupados de los tres años ni se pondera por el tamaño de los folds. Los empates
exactos, antes de redondear, favorecen menor umbral/m/alpha; para el bosque,
menor profundidad y luego mayor hoja. El error es **1 − accuracy**. Las métricas
de entrenamiento son de resustitución, usadas como diagnóstico, no para elegir.

## Configuraciones seleccionadas y controles

| Modelo | Configuración | Macro-F1 validación, media ± DE | Error train | Error validación |
|---|---|---:|---:|---:|
| ID3 | ganancia mínima 0,005 | 0,381798 ± 0,016158 | 0,473186 | 0,576739 |
| NB propio | m = 0,1 | 0,363482 ± 0,032202 | 0,516039 | 0,565633 |
| CategoricalNB | alpha = 0,025 | 0,363482 ± 0,032202 | 0,516039 | 0,565633 |
| Random Forest | profundidad sin límite, hoja mínima 20 | **0,422767 ± 0,009563** | 0,403547 | 0,565298 |
| DT con códigos, fijo | entropía | 0,379811 ± 0,014377 | 0,472063 | 0,580804 |
| DT con tasas, fijo | entropía | 0,341531 ± 0,018977 | 0,001159 | 0,648557 |
| Baseline, fijo | tasa de victorias en diez años | 0,350521 ± 0,008642 | 0,524971 | 0,547660 |

DE es la desviación estándar poblacional entre los tres años (`ddof=0`), no un
intervalo de confianza. La tabla describe validación usada para seleccionar;
no constituye una estimación independiente del desempeño final.

| Modelo seleccionado | F1 2021 | F1 2022 | F1 2023 |
|---|---:|---:|---:|
| ID3 | 0,404642 | 0,370874 | 0,369878 |
| NB propio | 0,362013 | 0,403635 | 0,324798 |
| CategoricalNB | 0,362013 | 0,403635 | 0,324798 |
| Random Forest | 0,426883 | 0,431866 | 0,409553 |

## Lectura de los hallazgos

**ID3.** El umbral 0,005 mejora levemente el macro-F1 respecto a 0
(0,381798 frente a 0,380579). El umbral 0,01 logra menor error de validación
(0,571926), pero menor macro-F1 (0,363435), por lo que no se selecciona. Los
umbrales 0,02 y 0,05 reducen más el macro-F1. La pequeña diferencia entre
umbrales bajos no prueba superioridad estadística.

**Naive Bayes.** `m=0,1`, 1 y 10 empatan exactamente en macro-F1 de validación;
se elige 0,1 por la regla declarada. En CategoricalNB empatan `alpha=0,025`,
0,25 y 2,5, y se elige 0,025. En los tres folds cada atributo tiene cuatro
códigos en la tabla, contando cero: el suavizado `(n + m/4)/(N + m)` del modelo
propio corresponde a `alpha=m/4` de CategoricalNB. Las métricas de ambos coinciden
en cada punto emparejado de las grillas; no se interpretan como dos evidencias
independientes de superioridad. Los suavizados más fuertes reducen macro-F1.

**Random Forest.** El mejor valor medio de la búsqueda es 0,422767, con hoja
mínima 20 y profundidad sin límite. El bosque original sin límite y hoja 1 se
conservó: obtiene 0,375276 en validación y 0,998784 en entrenamiento. Su error
train es 0,001159 frente a 0,588063 en validación. Esta brecha es compatible
con sobreajuste; aumentar la hoja a 20 reduce esa brecha y mejora macro-F1 en
estos folds. La elección está acotada a la grilla y semilla utilizadas.

**Comparaciones útiles.** El árbol de scikit-learn con los mismos códigos queda
cerca de ID3 (0,379811 frente a 0,381798). Con tasas continuas y configuración
sin regularizar muestra una brecha amplia entre entrenamiento y validación.
El baseline logra menor error de validación que los modelos seleccionados, pero
menor macro-F1 que el bosque: esto ilustra por qué accuracy no reemplaza la
métrica de selección acordada. No se atribuyen causas por clase sin el análisis
correspondiente, ni se anticipan conclusiones sobre 2024–2025.

## Curvas y evidencia reproducible

Las cuatro figuras incluyen error de entrenamiento/validación y macro-F1 a lo
largo de las grillas. Para el bosque, cada fila corresponde a una profundidad
fija y el eje horizontal recorre la hoja mínima. Las estrellas indican el mejor
macro-F1 de cada panel; la selección global compara todas las profundidades.

- [Curvas ID3](../results/validation/figures/id3.png)
- [Curvas NB propio](../results/validation/figures/nb_propio.png)
- [Curvas CategoricalNB](../results/validation/figures/categorical_nb.png)
- [Curvas Random Forest](../results/validation/figures/random_forest.png)
- [Métricas por configuración y año](../results/validation/fold_metrics.csv)
- [Medias y desviaciones de toda la grilla](../results/validation/grid_summary.csv)
- [Configuraciones seleccionadas y controles](../results/validation/selected.csv)
- [Manifiesto: versiones, SHA-256, grillas y cortes por fold](../results/validation/manifest.json)

Esta corrida histórica se verificó con Python **3.12.14** y scikit-learn
**1.9.1**, usando el código conservado en
`0fc5945b03803b329226f1a0fe6b97cbad030aeb`. Para repetirla, preparar la copia
histórica y su entorno según [feature_findings.md](feature_findings.md), y
antes de ejecutar los experimentos de atributos ejecutar dentro de esa copia:

```sh
.venv/bin/python scripts/run_validation.py
```

Los manifiestos originales no se deben modificar para atribuir esta corrida al
código actual. En aquel notebook, `RUN_FINAL_TEST=False` desactivaba toda la
evaluación final. En el notebook actual desactiva únicamente las celdas legadas
de árboles: NB y baseline sí se ajustan y evalúan sobre 2024–2025. Su reproducción
vigente desde el ZIP está en [nb_delivery.md](nb_delivery.md).

Verificación final: 16 pruebas aprobadas; se recalcularon medias y selección desde
las 93 filas guardadas y se comprobaron los hashes. Se recorrieron todas las
celdas del notebook con la evaluación final desactivada, sin entrenar nuevos
clasificadores ni crear predicciones de test. Las cuatro figuras se revisaron
visualmente.
