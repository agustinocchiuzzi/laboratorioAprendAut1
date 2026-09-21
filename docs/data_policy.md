# Política de datos e información histórica

Esta página documenta la preparación de datos. La selección temporal posterior
se registra por separado en [validation_findings.md](validation_findings.md). El ZIP original se conserva sin cambios. El alcance y las
exclusiones se fijaron por consistencia de los registros, no por rendimiento.

## Evidencia y alcance del objetivo

El [diccionario de Schoch](https://github.com/schochastics/football-data#codebook)
(consultado el 17/09/2026) define `F` como finalización a los 90 minutos, `E` como
prórroga y `P` como definición por penales. Describe `gh`/`ga` como goles que
incluyen prórroga y penales. El CSV suministrado, sin embargo, tiene **ocho filas
P con goles iguales** y cinco filas E con goles distintos. No contiene columnas
separadas de goles a los 90 minutos, en prórroga o en la tanda. El repositorio
público documenta datos hasta 2023; el archivo de la consigna llega a 2025. Su
diccionario orienta la interpretación, pero no prueba la exactitud de cada fila
de esta copia. No se equipara `full_time=E` con la clase objetivo empate.

Se adopta un objetivo conservador: resultado L/V/E de los registros identificados
como finalizados en tiempo reglamentario (`F`). Se excluyen **todas** las filas E/P
tanto de las muestras como de los historiales. No se reconstruye el marcador a
los 90 minutos, no se descuentan goles y no se asigna un empate supuesto. Esta
exclusión reduce la cobertura: el modelo no se evalúa sobre partidos definidos
fuera del tiempo reglamentario. `full_time` sirve para delimitar retrospectivamente
la muestra; nunca se entrega como atributo para predecir un partido.

La letra no especifica explícitamente si quiere resultado reglamentario o final:
esta es una decisión metodológica declarada, no una exigencia atribuida al curso.
Reintegrar estos encuentros requeriría evidencia verificable del marcador que
corresponda al objetivo y una nueva versión de la auditoría.

## Duplicados y conflictos

Se normalizan espacios y tipos antes de comparar; se conserva el primer registro
solo cuando **todas** sus columnas son iguales. El duplicado exacto es Montevideo
Wanderers–Defensor Sporting, 17/06/2025, 0–0: dos filas, una conservada.

La clave de control es fecha y par **no ordenado** de equipos. Después de quitar
repeticiones exactas, cualquier clave con más de un registro se pone completa en
cuarentena. También se detecta localía invertida: seleccionar la primera fila
introduciría una elección arbitraria. Incluso cuando dos marcadores coinciden en
L/V/E, no se cuentan como dos encuentros ni se elige un marcador para los atributos
históricos de goles. No se afirma que sean el mismo partido real: podría haber
fechas incorrectas o partidos distintos; el archivo no permite resolverlo.

| Fecha | Par de equipos | Registros ambiguos |
|---|---|---:|
| 1987-09-26 | Rampla Juniors Futbol Club / Miramar Misiones | 3 |
| 1994-10-01 | CA Penarol / Defensor Sporting | 3 |
| 1994-10-15 | CA Basanez / Institucion Atletica Sud America | 2 |
| 1994-10-15 | Montevideo Wanderers / River Plate | 2 |
| 1996-03-16 | Danubio / Nacional | 2 |
| 1997-08-30 | CA Penarol / Defensor Sporting | 2 |
| 1998-08-29 | Bella Vista / CA Penarol | 2 |

Cuatro grupos ya presentan múltiples marcadores con la misma localía (ocho filas).
Al incluir las localías invertidas, son siete grupos y **16 filas excluidas**.
Los marcadores registrados, líneas del CSV y motivos están en
[data_audit/flagged_records.csv](data_audit/flagged_records.csv); ninguna cifra fue
sustituida. `source_line` cuenta el encabezado como línea 1.

También se marcan las apariciones múltiples de un equipo en una fecha frente a
rivales distintos. Antes de las exclusiones hay 191 filas con esa advertencia,
127 combinaciones equipo/fecha en 48 fechas; 175 filas marcadas se conservan.
Una coincidencia equipo/fecha no basta para identificar cuál registro es erróneo.
Se mantienen las fechas originales y la advertencia: la causalidad se garantiza
respecto de **fechas registradas**, no de un calendario histórico verificado. No
se establecen alias nuevos para los clubes. Para varios partidos previos de un
mismo equipo en una fecha, el orden estable por local/visitante desempata la
ventana de últimos cinco; no supone conocer su horario real.

## Recuento reproducible

| Etapa | Filas |
|---|---:|
| Original | 15.207 |
| Repeticiones exactas descartadas | 1 |
| Registros ambiguos descartados | 16 |
| Prórroga descartados | 5 |
| Penales descartados | 8 |
| Admitidas | 15.177 |
| Entrenamiento, hasta 2023 | 14.705 |
| Evaluación, 2024–2025 | 472 |

En este archivo las exclusiones de ambigüedad y E/P no se superponen. Los datos
llegan hasta el 30/06/2025, no cubren ese año completo. Para reproducir la auditoría:

```sh
python3.12 scripts/audit_data.py
```

El script genera el CSV de registros señalados y
[data_audit/summary.json](data_audit/summary.json), con SHA-256 del ZIP. No entrena
modelos. La carga exige etiquetas F/E/P conocidas; ante faltantes o nuevas
etiquetas falla explícitamente para evitar una política implícita.

## Información disponible para todos los modelos

`build_causal_match_features` recibe únicamente partidos admitidos. Procesa todos
los encuentros de una fecha antes de actualizar cualquier historial de ese día.
Los atributos pueden usar resultados de fechas anteriores del mismo bloque de
validación o evaluación: es un protocolo de predicción sucesiva, no de pronóstico
de toda la temporada desde su inicio. Ni modelo ni discretizador se reajustan
dentro del bloque. Calcular previamente estas estadísticas causales equivale a
revelar los resultados al terminar cada fecha; las pruebas comprueban que cambiar
resultados actuales o futuros no altera entradas anteriores ni del mismo día.

El baseline recibe `home_win_rate_10y` y `away_win_rate_10y` del **mismo constructor**
que produce las seis tasas de los demás modelos. Ya no crea un historial congelado
en `fit`. La ventana es `[fecha - 10 años, fecha)`, incluye el extremo inferior y
excluye el día actual. Sin antecedentes usa 0,5, igual que las demás tasas. Ante
igualdad predice L; nunca predice E. `predict` solo compara tasas, no accede a
etiquetas ni actualiza estado, y repetirlo u ordenar sus consultas no cambia el
resultado. El valor neutro puede coincidir con una tasa real: no identifica por sí
solo ausencia de información.

## Validación y preprocesamiento comunes

`src/evaluation.py` define una única partición para todas las configuraciones:

| Entrenamiento expansivo | Validación |
|---|---|
| Fechas anteriores a 2021 | Todo 2021 |
| Fechas anteriores a 2022 | Todo 2022 |
| Fechas anteriores a 2023 | Todo 2023 |

La partición usa fechas, no número de filas, e incluye siempre fechas completas.
Los índices son posiciones del DataFrame recibido; el evaluador verifica que
correspondan al frame actual. Rechaza filas de test, folds parciales o posiciones
obsoletas. ID3, NB, árboles de referencia, Random Forest y baseline invocan el
mismo evaluador con `CV_FOLDS`. El comparador CategoricalNB usa esa misma interfaz en la selección temporal
posterior, documentada en `validation_findings.md`.

Cada fold crea un pipeline y un estimador nuevos. ID3 y NB categórico comparten
cortes fijos 0,3/0,6 para las tasas recientes y terciles ajustados exclusivamente en
entrenamiento para el resto; también las medianas se aprenden allí. Las tasas
continuas y el baseline pasan por selección explícita de columnas, sin usar
etiquetas, fechas ni goles actuales como entradas. Para el ajuste final se crea
otro discretizador con todo train. El notebook selecciona los parámetros propios
por macro-F1 medio con igual peso por año y menor parámetro ante empate. El ID3
se reajusta con el umbral elegido, no con el umbral fijo anterior.

Las métricas y texto numérico de la antigua ejecución del notebook se invalidaron.
Los archivos históricos de `results/` no se borraron ni se consideran resultados
de esta política. Las grillas no se ejecutaron durante la revisión de datos; su ejecución
posterior, exclusivamente en validación, se documenta en `validation_findings.md`.

## Verificación histórica sin experimentos

La suite de tests se retiró al preparar la integración. Las comprobaciones aquí
registradas son históricas; las pruebas compartidas siguen disponibles en la
copia del commit descrita en [feature_findings.md](feature_findings.md).
La reproducción actual desde el ZIP está en [nb_delivery.md](nb_delivery.md).

Las pruebas realizadas comprobaron exclusiones, conservación del original, límites de la
ventana, independencia de resultados futuros, actualización común del baseline,
fechas completas y ajuste de preprocesamiento por fold. Las pruebas de interfaz
entrenan únicamente sobre datos sintéticos diminutos; no buscan hiperparámetros
ni producen métricas experimentales sobre el fútbol uruguayo.

Verificación realizada el 17/09/2026 en un entorno temporal con **Python 3.12.14**
y **scikit-learn 1.9.1**: 12 pruebas aprobadas, dependencias compatibles y sintaxis
válida de todas las celdas. También se ejecutaron únicamente las celdas de carga,
atributos, partición y discretización del notebook sobre el ZIP real. Los folds
resultantes tienen 13.744/364, 14.108/297 y 14.405/300 filas de
entrenamiento/validación; las matrices finales tienen formas `(14705, 6)` y
`(472, 6)`. No se entrenaron clasificadores sobre esos datos ni se ejecutaron
las grillas. El entorno `.venv` existente no se sustituyó.
