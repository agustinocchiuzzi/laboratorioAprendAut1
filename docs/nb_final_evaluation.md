# Evaluación final de Naive Bayes

Se completó la evaluación con **Python 3.12.14 y scikit-learn 1.9.1**, usando
las configuraciones cerradas en [la selección anterior](nb_selection.md).
Cada NB ajustó un pipeline nuevo con **14.705 partidos admitidos**, desde
1932-03-05 hasta 2023-12-16. El test común contiene **472 partidos**:
297 de 2024 y 175 de 2025, entre 2024-02-17 y 2025-06-30. No cubre todo 2025.
No se ejecutaron ni modificaron los modelos ID3 o Random Forest.

## Configuración y política de información

- NB propio: `MEstimateCategoricalNB(m=0.1, min_categories=4)`.
- Referencia: `CategoricalNB(alpha=0.025, min_categories=4, force_alpha=True,
  fit_prior=True, class_prior=None)`.
- Variante `plus_points_and_goals`: las seis tasas originales, puntos por partido
  y diferencia de gol por partido en hasta cinco antecedentes de cada equipo.
  Las diez columnas y su orden se copian de la configuración cerrada.
- Dos discretizadores nuevos, uno por NB: cortes fijos 0.3/0.6 para las dos tasas
  recientes; terciles y medianas aprendidos únicamente de todo train para las
  otras ocho entradas. Cuatro categorías por atributo, incluido el cero reservado.
- Priors de clase empíricos, decisión por argmax E/L/V; empates exactos a favor
  de la primera clase. No se aplicó la variante diagnóstica que prohíbe E.

La [política de datos](data_policy.md) se conserva: únicamente registros F,
repeticiones exactas eliminadas y conflictos en cuarentena. Solo se incorporan
resultados de **fechas estrictamente anteriores**, incluidos días anteriores del
test. Se construyen todos los partidos de una fecha antes de actualizar cualquier
historial de ese día. No se reajusta ni el preprocesamiento ni el clasificador
durante test. La huella de cada pipeline es idéntica antes y después de predecir.
También se verificó que construir los atributos con toda la serie deja intacto
el prefijo de entrenamiento calculado sin leer test.

La configuración y su manifiesto se verifican **antes** de leer resultados de
test; sus archivos permanecen intactos. Ningún resultado de esta evaluación se
utilizó para cambiar atributos, suavizado o reglas de decisión.

## Resultados sobre el bloque completo

| Modelo | Accuracy | Macro-F1 |
|---|---:|---:|
| NB propio | 0.478814 | 0.445369 |
| CategoricalNB | 0.478814 | 0.445369 |
| Baseline de diez años | 0.461864 | 0.356446 |

Ambos NB aciertan 226 partidos; el baseline, 218. Las diferencias son
**+0.016949 en accuracy** y **+0.088923 en macro-F1**. Son comparaciones
descriptivas de este test, sin afirmar significación estadística.
Macro-F1 promedia las tres clases E/L/V con igual peso; se usa `zero_division=0`.
Las métricas agrupan los 472 partidos, sin promediar primero por año.

Las métricas por clase son idénticas para ambos NB:

| Modelo | Clase | Precision | Recall | F1 | Soporte |
|---|:---:|---:|---:|---:|---:|
| NB propio / CategoricalNB | E | 0.353659 | 0.221374 | 0.272300 | 131 |
| NB propio / CategoricalNB | L | 0.547511 | 0.636842 | 0.588808 | 190 |
| NB propio / CategoricalNB | V | 0.449704 | 0.503311 | 0.475000 | 151 |
| Baseline de diez años | E | 0.000000 | 0.000000 | 0.000000 | 131 |
| Baseline de diez años | L | 0.502075 | 0.636842 | 0.561485 | 190 |
| Baseline de diez años | V | 0.419913 | 0.642384 | 0.507853 | 151 |

Matrices de confusión, **filas reales y columnas predichas E, L, V**:

```text
NB propio y CategoricalNB        Baseline de diez años
       E    L    V                      E    L    V
E     29   54   48               E      0   66   65
L     24  121   45               L      0  121   69
V     29   46   76               V      0   54   97
```

Los NB predicen 82 empates y recuperan 29 de los 131 reales. E sigue siendo
la clase con menor recall. El baseline nunca predice E. Los NB tienen mayor
F1 en E y L, y menor F1 en V: reconocer algunos empates viene acompañado de
menos aciertos visitantes en esta muestra. No se usa esa observación para
reoptimizar la decisión.

## Equivalencia y comparación justa con el baseline

**Cero diferencias entre las 472 predicciones de los NB.** Se comprobaron las
entradas discretizadas, los conteos por clase/categoría, las cardinalidades y
los priors. Con K=4 en los diez atributos y `alpha=m/4=0.025`,
`(n + m/4)/(n_c + m)` es la misma probabilidad que
`(n + alpha)/(n_c + 4*alpha)`. Los puntajes logarítmicos difieren como máximo
`5.329070518200751e-15`, por redondeo, sin cambiar ningún argmax.

El baseline utiliza las tasas del **mismo constructor causal y las mismas filas**.
Su ventana es `[fecha − 10 años, fecha)`, su neutro es 0.5 y los empates de tasas
favorecen L. Su `fit` registra el contrato de entrada, sin aprender parámetros.
La tabla antigua de `results/notebook.merged.executed.ipynb` tenía una evaluación
anterior a la política actual (accuracy redondeada 0.4640); no acredita iguales
registros e información. Se conservó y se recalculó el baseline actual, sin
presentar aquella cifra como una corrida comparable.

## Artefactos y reproducción

Los resultados están en [`results/naive_bayes/final/`](../results/naive_bayes/final/):

- `summary.csv`, `class_report.csv`: métricas completas sin redondear.
- `confusion_*.csv`, `confusion_matrices.png`: las tres matrices en E/L/V.
- `predictions.csv`: una fila por partido con fecha, local, visitante, resultado
  real, las tres predicciones y sus aciertos; 472 filas.
- `probabilities.csv`: puntajes conjuntos y probabilidades logarítmicas de los
  dos NB por partido y clase; 944 filas.
- `disagreements.csv`: esquema para identidad, puntajes y márgenes de cualquier
  discrepancia; sin filas porque no hubo diferencias.
- `test_features.csv`: las diez entradas NB y las dos tasas del baseline por partido.
- `*.joblib`: los dos pipelines ajustados y el baseline. La recarga reproduce
  todas las predicciones y también respeta el orden invertido de consultas.
- `configurations.json`, `preprocessing.json`, `equivalence.json`: parámetros,
  columnas, cortes, medianas, cardinalidades, priors y diagnóstico numérico.
- `requirements.lock.txt`, `manifest.json`: versiones, límites y soportes de las
  particiones, hashes de datos, selección, código y artefactos.
- `notebook.executed.ipynb`, `verification.json`: copia del notebook con las
  celdas de lectura NB ejecutadas y registro de verificación sin ajustes.

El notebook principal mantiene sus outputs vacíos y las celdas de árboles
desactivadas. **Actualización de entrega:** ahora reproduce selección, ajuste
final y análisis desde el ZIP, sin requerir estos artefactos históricos.
`scripts/verify_final_nb.py` ejecuta todas sus celdas en una carpeta temporal
sin resultados previos y guarda la copia ejecutada en `results/nb_delivery/`.
Ver [instrucciones y pendientes de entrega](nb_delivery.md).

```sh
# Dentro del entorno Python 3.12 creado según nb_delivery.md:
.venv/bin/python -m pip install -r results/naive_bayes/final/requirements.lock.txt
.venv/bin/python scripts/evaluate_final_nb.py

# Reproducción desde cero del notebook (incluye selección y ajuste de NB):
.venv/bin/python scripts/verify_final_nb.py

```

Antes de retirar la suite de tests se aprobaron **23 pruebas del subconjunto NB** en Python 3.12.14 y scikit-learn
1.9.1. Cubren el
ajuste único y exclusivo con train, preprocesadores independientes, rechazo de
seis atributos o decisiones distintas, invariancia frente a resultados actuales,
del mismo día y futuros, exclusiones y límites de la ventana del baseline.
La verificación histórica bloqueaba cualquier ajuste; la verificación actual
permite únicamente los 74 ajustes de NB (30 + 42 + 2 finales) y el baseline,
y bloquea los ajustes de árboles y bosque. Los artefactos históricos se preservan.
La ejecución histórica de la suite completa fue de 41 casos; ver
[nb_delivery.md](nb_delivery.md).
