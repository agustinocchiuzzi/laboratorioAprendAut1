# Experimentos acotados de atributos: hallazgos de validación

Se ejecutó el [plan declarado antes de la corrida](feature_experiment_plan.md):
**7 variantes × 4 modelos × 3 años = 84 ajustes**. Cada ajuste se reutilizó
para decisiones de tres clases y para el diagnóstico «nunca predice E».
Se conservaron los hiperparámetros elegidos en la etapa anterior, los folds
2021/2022/2023 y la limpieza auditada. No se probaron combinaciones adicionales
tras observar los resultados. La variante actual reprodujo las métricas de
validación de la corrida anterior en los cuatro modelos.

Las medias dan igual peso a cada año. El total de validación es 961 encuentros,
incluidos **273 empates reales**, en cada combinación modelo/variante/diagnóstico.
El período 2024–2025 se excluyó antes de crear atributos para estos experimentos.
No se realizó reajuste final ni evaluación sobre ese período.

## Selecciones por macro-F1 de validación

El diagnóstico sin E se informa aparte y no compite en la selección principal.
Se conserva la regla de mayor macro-F1 medio; ante empate exacto, menos atributos
y luego el orden del plan. Los hiperparámetros se mantienen fijos, por lo que
estas elecciones no equivalen a optimizar nuevamente cada modelo/variante.

| Modelo | Variante elegida | Accuracy | Macro-F1 | DE entre años |
|---|---|---:|---:|---:|
| Random Forest | Actuales | 0.434702 | 0.422767 | 0.009563 |
| ID3 | + puntos y diferencia de gol | 0.415377 | 0.400186 | 0.029046 |
| CategoricalNB | + puntos y diferencia de gol | 0.427650 | 0.385798 | 0.033246 |
| NB propio | + puntos y diferencia de gol | 0.427650 | 0.385798 | 0.033246 |

La configuración global elegida sigue siendo **Random Forest con las seis tasas
actuales**, 300 árboles, profundidad sin límite, hoja mínima 20,
`balanced_subsample` y semilla 42. Para ID3 se eligen las diez entradas de la
variante +puntos/+goles y se mantiene `min_info_gain=0.005`, profundidad máxima 8.
Ambos NB eligen esas mismas diez entradas, con `m=0.1` y `alpha=0.025`,
respectivamente. La mejora de los NB no altera la equivalencia de sus métricas
con el suavizado emparejado utilizado.

## Comparación completa de variantes

Cada celda muestra **accuracy / macro-F1** medio. NB propio y CategoricalNB
obtuvieron las mismas métricas por variante y año; se agrupan solo en esta tabla
para facilitar la lectura. Los archivos conservan ambos modelos por separado.

| Variante | ID3 | NB propio / CategoricalNB | Random Forest |
|---|---:|---:|---:|
| Actuales | 0.4233 / 0.3818 | 0.4344 / 0.3635 | 0.4347 / 0.4228 |
| + puntos | 0.4048 / 0.3827 | 0.4302 / 0.3713 | 0.4246 / 0.4107 |
| + diferencia de gol | 0.3999 / 0.3798 | 0.4262 / 0.3693 | 0.4290 / 0.4130 |
| + puntos y diferencia de gol | 0.4154 / 0.4002 | 0.4276 / 0.3858 | 0.4313 / 0.4169 |
| + tasas de empate | 0.4066 / 0.3917 | 0.4224 / 0.3841 | 0.4075 / 0.3931 |
| Sin historial local | 0.4010 / 0.3252 | 0.4247 / 0.3369 | 0.3970 / 0.3868 |
| Sin H2H local | 0.4143 / 0.3247 | 0.4273 / 0.3440 | 0.4323 / 0.4214 |

- **Puntos y goles juntos:** ID3 pasa de macro-F1 0,3818 a 0,4002, pero accuracy
  baja de 0,4233 a 0,4154. Ambos NB pasan de macro-F1 0,3635 a 0,3858 y accuracy
  de 0,4344 a 0,4277. Se elige esta variante por la métrica acordada, sin afirmar
  que mejore simultáneamente ambos criterios.
- **Tasas históricas de empate:** mejoran el macro-F1 de ID3 y NB respecto de
  las entradas actuales, pero no superan la combinación de puntos/goles. Para
  el bosque reducen accuracy y macro-F1. El recall medio de E en ID3 sube de
  0,1710 a 0,3365, mientras accuracy baja: detectar más empates no garantiza
  por sí solo mejor clasificación global.
- **Retirar cada historial de localía:** quitar el historial general del local
  reduce macro-F1 en los cuatro modelos. Quitar solo H2H también lo reduce;
  en Random Forest la diferencia es pequeña (0,4228 a 0,4214). No se declara
  irrelevante ese atributo ni superioridad estadística a partir de esa diferencia.
- **Bosque:** ninguna adición o ablación supera la configuración actual en esta
  comparación con hiperparámetros fijos. Este resultado no prueba que ninguna
  otra representación o configuración pueda mejorarla.

## Diagnóstico: nunca predecir empates

Se entrenó siempre con E/L/V y se conservaron las mismas verdades de validación.
El diagnóstico elige L/V según probabilidades (conteos del nodo alcanzado en
ID3), sin mirar la etiqueta real ni ajustar un umbral. Empates exactos de puntaje
favorecen L. Así, E tiene F1 y recall cero, y ese cero **sigue contando en el
promedio de tres clases**. No se presenta una métrica binaria filtrada.

Para las variantes seleccionadas por modelo:

| Modelo / variante | Accuracy tres clases | Accuracy nunca E | Macro-F1 tres clases | Macro-F1 nunca E |
|---|---:|---:|---:|---:|
| Random Forest / Actuales | 0.434702 | 0.434915 | 0.422767 | 0.337022 |
| ID3 / + puntos y diferencia de gol | 0.415377 | 0.416834 | 0.400186 | 0.316589 |
| CategoricalNB / + puntos y diferencia de gol | 0.427650 | 0.441447 | 0.385798 | 0.341427 |
| NB propio / + puntos y diferencia de gol | 0.427650 | 0.441447 | 0.385798 | 0.341427 |

En las **28 parejas** modelo/variante el diagnóstico reduce macro-F1. En varias
sube accuracy: por ejemplo, NB +puntos/goles pasa de 0,4277 a 0,4414, pero
macro-F1 cae de 0,3858 a 0,3414. En otras también baja accuracy, como ID3 actual.
Por tanto, no se infiere que evitar empates mejore siempre la proporción de
aciertos. En el bosque actual el cambio de accuracy es mínimo (0,434702 a
0,434915) y macro-F1 cae a 0,337022. Ese intercambio no justifica abandonar E
bajo el criterio de selección declarado.

## Evidencia, reproducibilidad y límites

- [Gráfico accuracy–macro-F1 de todas las comparaciones](../results/feature_experiments/accuracy_macro_f1.png)
- [Métricas y matrices por fold](../results/feature_experiments/fold_metrics.csv)
- [Resumen completo, incluidas ambas decisiones](../results/feature_experiments/summary.csv)
- [Configuraciones elegidas por modelo](../results/feature_experiments/selected.csv)
- [Predicciones de validación y verdades originales](../results/feature_experiments/validation_predictions.csv)
- [Manifiesto: columnas exactas, configuraciones, folds, hashes y versiones](../results/feature_experiments/manifest.json)

Los puntos/goles proceden de los últimos cinco partidos admitidos. Las tasas de
empate acumulan todo el historial anterior del equipo, sin distinguir localía,
y usan 1/3 sin antecedentes. Se computan todos los partidos de un día antes de
actualizar los historiales. Los cuantiles de los atributos añadidos se aprenden
solo en cada entrenamiento. Las ablaciones eliminan por separado exactamente
`home_win_rate_as_home_all` o `home_win_rate_h2h_as_home`.

Corrida con Python 3.12.14 y scikit-learn 1.9.1:

```sh
python3.12 scripts/run_feature_experiments.py
python3.12 -m unittest discover -s tests -v
```

Las desviaciones entre tres años son descriptivas, no intervalos de confianza.
Los folds ya participaron en elegir hiperparámetros y ahora atributos, de modo
que estas mejoras son hallazgos de selección, no estimaciones independientes de
generalización. No se amplió la grilla después de verlos. Los diagnósticos
finales legados del notebook permanecen desactivados; antes de habilitarlos habrá
que adaptarlos a las variantes elegidas. La evaluación final sigue pendiente.

Verificación completada: **23 pruebas aprobadas**. Se recalcularon las 168 filas
de métricas a partir de 26.908 predicciones de validación y se comprobaron las
56 filas de resumen, selecciones y hashes. Cada comparación conserva 961 casos
y 273 empates verdaderos. El notebook se recorrió con la evaluación final
desactivada, sin entrenar nuevos clasificadores ni producir predicciones de test.
