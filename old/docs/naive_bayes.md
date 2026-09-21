# Naive Bayes: implementación y verificación

La consigna `Tarea_1.pdf` solicita un NB propio con hiperparámetro `m`, tamaño
equivalente de muestra, y un comparador de scikit-learn. No especifica la
distribución del prior. Conservamos `MEstimateCategoricalNB` y su elección de
un prior uniforme para cada atributo categórico.

## Qué aprende `fit`

Para cada clase `c`, cuenta `n_c` ejemplos entre los `N` de entrenamiento y
estima **P(c) = n_c/N**, sin suavizar. En este problema las etiquetas son
`E`, `L`, `V`, pero la clase acepta otras etiquetas de clasificación.

Para cada atributo `j`, código `v` y clase `c`, cuenta `n_jvc` y calcula:

```text
P(X_j=v | c) = (n_jvc + m/K_j) / (n_c + m)
```

`m > 0` es el peso TOTAL del prior por atributo y clase: agrega `m`
observaciones ficticias, repartidas en `K_j` categorías. Cada categoría recibe
`m/K_j`, no `m`. Por ejemplo, `m=3`, `K_j=3` agrega un pseudoconteo a cada código.
Cuanto mayor sea `m` respecto a `n_c`, más pesa la distribución uniforme frente
a las frecuencias observadas. Las condicionales suman uno en cada clase.

Hay dos conceptos distintos: el **prior de clase** `P(c)` se aprende de las
etiquetas; el **prior de categorías** `p_jv=1/K_j` es nuestra elección para
suavizar las condicionales. Este último no usa las frecuencias marginales
`P(X_j=v)` del entrenamiento.

Por defecto `K_j=max(X_train[:,j])+1`: incluye todos los códigos desde cero,
incluso los intermedios sin observaciones. No es la cantidad de valores únicos
ni tiene por qué ser cuatro. `n_categories_` expone esos tamaños.
`min_categories` acepta un mínimo común o uno por atributo, declarado sin
mirar validación/test. Permite incluir un bin válido que no apareció en train.
La grilla actual declara cuatro códigos en ambos NB porque su discretizador
produce como máximo los bines `1,2,3`, más el cero reservado.

## Cómo se obtiene una predicción

Naive Bayes supone independencia de los atributos **condicionada a la clase**.
Para una fila `x` y cada clase, calcula:

```text
s(c) = log P(c) + suma_j log P(X_j=x_j | c)
predicción = clase con mayor s(c)
```

Sumar logaritmos equivale a multiplicar probabilidades y evita que ese producto
se redondee a cero. No se suman los códigos de los atributos. No hace falta
calcular `P(x)` para elegir la clase: es el mismo divisor para todas.
En un empate exacto se elige la primera clase en `classes_`, ordenada por
`np.unique` (para estas etiquetas: `E`, `L`, `V`).

`predict_proba` normaliza con softmax. `predict_log_proba` resta
`logsumexp(s)` directamente: tomar `log(predict_proba)` podía producir `-inf`
por redondeo aunque el logaritmo correcto fuera finito. Las tablas condicionales
también se calculan en log para conservar el suavizado con `m` muy pequeño.

### Ejemplo que se puede calcular a mano

Entrenamos con cinco filas, dos atributos y `m=3`:

| x1 | x2 | clase |
|---:|---:|:---|
| 1 | 1 | L |
| 1 | 2 | L |
| 2 | 1 | L |
| 2 | 2 | V |
| 2 | 1 | V |

Ambos atributos tienen `K=3` (códigos `0,1,2`). Los priors de clase son
`P(L)=3/5`, `P(V)=2/5`. Para predecir `[1,2]`:

```text
P(x1=1 | L) = (2+1)/(3+3) = 3/6
P(x2=2 | L) = (1+1)/(3+3) = 2/6
P(x1=1 | V) = (0+1)/(2+3) = 1/5
P(x2=2 | V) = (1+1)/(2+3) = 2/5

P(L) * P(x1=1 | L) * P(x2=2 | L) = 1/10
P(V) * P(x1=1 | V) * P(x2=2 | V) = 4/125

P(L | x) = 25/33; P(V | x) = 8/33; predicción = L
```

La prueba verifica tanto estas probabilidades como sus logaritmos.

## Desconocidos y preprocesamiento

- El código cero representa categorías desconocidas o faltantes categóricos.
  Se reserva aunque no aparezca en train y recibe suavizado. Si ya apareció,
  sus observaciones también se cuentan.
- Un código mayor o igual a `K_j` se remite al cero durante predicción. Esto
  agrupa los desconocidos en una sola categoría; no amplía el dominio ni cambia
  las tablas aprendidas. Los códigos dentro del dominio sin observaciones
  conservan sus propias columnas suavizadas. Valores negativos, fraccionarios
  o no finitos se rechazan: deben pasar antes por el preprocesamiento.
- `MixedTypeDiscretizer` aprende categorías, cuantiles y medianas solo en `fit`.
  `transform` reutiliza ese estado. Los números faltantes/no finitos se imputan
  con la mediana de train; si no hay números finitos, usa cero y luego discretiza.
  No se codifican como faltantes categóricos.
- Los cortes fijos `0.3,0.6` no se estiman de datos. Los cuantiles repetidos se
  deduplican y pueden generar menos de tres bines. Los pipelines crean un
  discretizador nuevo por fold; las pruebas verifican entrenamiento separado,
  extremos de validación, categorías nuevas, constantes y columnas sin datos.

## Cuándo equivale a `CategoricalNB`

[CategoricalNB 1.9](https://scikit-learn.org/1.9/modules/naive_bayes.html#categorical-naive-bayes)
usa `(n_jvc + alpha)/(n_c + alpha*K_j)`. Para obtener el mismo modelo se necesita:

1. Los mismos datos, etiquetas, códigos y dominios por atributo, incluido cero.
2. Prior uniforme de categorías y **alpha_j=m/K_j**.
3. El mismo prior de clase: en el comparador, `fit_prior=True`, `class_prior=None`.
4. Preservar el `alpha` solicitado (`force_alpha=True`, especialmente si es pequeño).
5. Al comparar consultas desconocidas, codificarlas igual. El comparador no
   remite por sí mismo los códigos fuera de rango a cero.

La API de `CategoricalNB` 1.9 usa un `alpha` escalar. Un solo valor coincide en
todos los atributos si todos tienen el mismo `K`. Si `K_1=2` y `K_2=4`, se
necesitarían respectivamente `m/2` y `m/4`; usar `m/4` en ambos cambia el modelo.
Las pruebas comparan atributos aislados en ese caso y detectan la discrepancia
del `alpha` común. También comprueban equivalencia completa para `K=2,3,5`.

En la grilla del repositorio, `min_categories=4` en ambos modelos mantiene
`alpha=m/4` aun con bines ausentes en un fold. Esa configuración no se debe
extrapolar a un discretizador con otro dominio.

## Verificación de la implementación

Antes de retirar la suite de tests se verificaron con datos sintéticos
m-estimate, priors, dominio declarado, desconocidos, cálculo logarítmico,
equivalencia con `CategoricalNB` y preprocesamiento ajustado sólo en train.
Esas pruebas no se incluyen en la entrega actual. La verificación completa vigente,
incluida la reproducción desde el ZIP y la evaluación final, está centralizada
en [nb_delivery.md](nb_delivery.md). No depende de una ruta temporal ni de la
`.venv` local.
