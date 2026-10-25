# Plan acotado de mejoras (declarado antes de ejecutar)

Se comparan siete conjuntos sobre los mismos partidos admitidos y los folds
anuales completos 2021, 2022 y 2023. Se excluyen fechas posteriores a 2023 antes
de crear atributos. No se modifica la limpieza ni la política de historiales.

| Identificador | Cambio respecto de las seis tasas actuales | Total |
|---|---|---:|
| current | Ninguno | 6 |
| plus_points | Puntos medios en los últimos cinco partidos, local y visitante | 8 |
| plus_goal_difference | Diferencia de gol media en los últimos cinco, ambos equipos | 8 |
| plus_points_and_goals | Ambas adiciones anteriores | 10 |
| plus_draw_rates | Tasa de empates histórica de cada equipo, sin distinguir localía | 8 |
| without_home_history | Quitar solo `home_win_rate_as_home_all` | 5 |
| without_home_h2h | Quitar solo `home_win_rate_h2h_as_home` | 5 |

Puntos y diferencia de gol ya eran calculados para auditoría por el constructor:
3/1/0 puntos por victoria/empate/derrota; goles a favor menos goles en contra.
Se usan hasta cinco antecedentes disponibles; sin antecedentes, 4/3 puntos y
cero goles de diferencia, conforme a la política existente. Las tasas de empate
son nuevas: empates / partidos admitidos anteriores del equipo, acumulados desde
el inicio del archivo, con neutro fijo 1/3 cuando no hay antecedentes. El neutro
puede coincidir con una tasa real; no implica probabilidades conjuntas calibradas
con el neutro 0,5 de las tasas de victoria. Ningún resultado del día actual
actualiza entradas de otros partidos del mismo día.

Se usan los cuatro modelos con los hiperparámetros previamente seleccionados:
ID3 `min_info_gain=0.005` (profundidad máxima 8); NB propio `m=0.1`;
CategoricalNB `alpha=0.025`, cuatro códigos; Random Forest 300 árboles,
profundidad sin límite, hoja mínima 20, `balanced_subsample`, semilla 42.
No se retocan hiperparámetros para favorecer una variante. Esto compara atributos
condicionado a esas configuraciones, no el óptimo de cada conjunto de atributos.

Los modelos categóricos aprenden terciles de cada atributo añadido dentro de
cada entrenamiento; conservan los cortes 0,3/0,6 para las dos tasas recientes.
Random Forest consume valores continuos. Cada modelo/variante/fold crea un
pipeline nuevo. Son 7 × 4 × 3 = **84 ajustes**, sin búsqueda posterior adaptativa.

Se reutiliza cada ajuste para un diagnóstico sin predicciones E: se elige L/V
por mayor probabilidad estimada (o conteos de la hoja/nodo alcanzado en ID3), con
empates exactos a favor de L. **No se eliminan ni recodifican empates reales** del
entrenamiento o evaluación; macro-F1 siempre promedia E, L y V. El F1/recall de E
será cero en ese diagnóstico. No es un problema binario ni una variante elegible
para la selección principal. Sirve para mostrar el intercambio accuracy/macro-F1.

Selección principal: mayor macro-F1 medio anual de validación entre las siete
variantes de tres clases, por modelo, con igual peso por año. Ante empate exacto,
menos atributos y luego orden de la tabla; para un empate global restante, nombre
del modelo en orden lexicográfico. Accuracy se informa pero no elige. Se guardan
métricas por año, resúmenes, matrices de confusión y predicciones de validación
para auditar que todos los casos y las tres clases permanezcan en el denominador.
La reutilización de estos años para seleccionar modelos y atributos introduce
optimismo de selección: los valores no equivalen a una evaluación independiente.
No se reajustan modelos finales ni se consulta el rendimiento de 2024–2025.
