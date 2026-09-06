# Decisiones de modelado

Este documento registra decisiones para que los experimentos sean comparables.
Las marcadas **por confirmar** deben contrastarse con las notas del curso.

## Variable objetivo

- `L`: `gh > ga`.
- `V`: `gh < ga`.
- `E`: `gh == ga`.
- Los goles solo construyen la etiqueta y atributos historicos de partidos
  anteriores; nunca se usan los goles del partido que se predice.

## Particion temporal

- Entrenamiento final: desde 1932 hasta 2023 inclusive.
- Evaluacion final: 2024 y 2025.
- Seleccion de hiperparametros: ventana expansiva sobre bloques contiguos de fechas
  dentro de entrenamiento.
- Una fecha completa pertenece a un unico fold.
- La evaluacion 2024-2025 no participa en la seleccion de atributos ni parametros.

## Politica de atributos historicos

La corrida representa predicciones sucesivas: un partido puede usar resultados de
fechas anteriores, incluso si esas fechas pertenecen a 2024-2025. El modelo no se
reentrena durante test. Partidos del mismo dia no se actualizan entre si.

Los porcentajes sin partidos previos se codifican como `0.0`. Se incorporan las
siguientes tasas, siempre calculadas antes de la fecha del partido: victorias en
los últimos cinco partidos de cada equipo, victorias de cada equipo dentro del
año calendario, victorias históricas del local actuando de local y victorias del
local frente a ese visitante con la misma localía. El último atributo es
head-to-head orientado: no mezcla partidos con la localía invertida.

Comparacion opcional para el informe: congelar todos los historiales al 31/12/2023
y medir cuanto cambia el resultado. No mezclar ambas politicas en una misma tabla.

## Baseline de diez anos

Para cada equipo se calcula:

```text
partidos ganados / partidos jugados
```

usando partidos de entrenamiento en los diez anos anteriores a la fecha a
predecir. Se predice `L` o `V` segun cual equipo tenga mayor proporcion. Si hay
empate exacto, se elige `L` de forma deterministica. **Por confirmar:** preguntar
si el equipo docente espera otro desempate.

## Naive Bayes propio

Los atributos numericos se discretizan por cuantiles dentro de cada fold. Los
categoricos se codifican reservando cero para valores desconocidos.

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
- analisis cualitativo a partir de `results/test_predictions.csv`.

## Comparadores de scikit-learn

- `CategoricalNB` recibe exactamente la misma discretizacion que el Naive Bayes
  propio y ajusta `alpha`.
- `RandomForestClassifier` usa `class_weight="balanced_subsample"`, semilla 42 y
  ajusta profundidad maxima y cantidad minima de instancias por hoja.
- La corrida rapida reduce grillas y cantidad de arboles; solo la corrida `full`
  debe alimentar el informe final.

## Reproducibilidad

- Python 3.12;
- scikit-learn 1.9;
- semilla 42;
- datos limpios, figuras y modelos generados desde comandos documentados;
- ninguna salida generada se considera fuente.
