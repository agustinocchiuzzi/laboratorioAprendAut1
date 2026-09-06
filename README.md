# Tarea 1 - Prediccion del futbol uruguayo

Proyecto de la asignatura **Aprendizaje Automatico** (FING, UdelaR, 2026). El
objetivo es predecir si un partido termina con victoria local (`L`), victoria
visitante (`V`) o empate (`E`).

La fecha oficial de entrega es el **21 de septiembre de 2026 a las 23:59**. El
enunciado esta en [`Tarea_1.pdf`](Tarea_1.pdf).

## Estado del proyecto

El repositorio ya incluye una base reproducible para:

- validar y limpiar el dataset original;
- construir atributos historicos sin usar el resultado del partido actual;
- separar entrenamiento (hasta 2023) y evaluacion final (2024-2025);
- hacer validacion cruzada temporal por bloques de fechas;
- entrenar el baseline solicitado;
- entrenar un Naive Bayes propio con hiperparametro `m`;
- entrenar un arbol ID3 propio con `min_info_gain`;
- comparar con `CategoricalNB` y `RandomForestClassifier` de scikit-learn;
- guardar metricas, matrices de confusion, curvas y predicciones para el informe.

Los resultados numericos todavia deben ser ejecutados, revisados criticamente y
discutidos por el grupo. No se deben copiar al informe sin entenderlos.

## Estructura

```text
data/raw/                  Dataset original del curso (fuente autoritativa)
data/processed/            Archivos generados; no se versionan
data/legacy/               Transformaciones manuales antiguas; no se usan
docs/modeling_decisions.md Supuestos metodologicos a validar con el curso
notebook/                  Material de apoyo original del curso
notebooks/                 Notebook autocontenido de experimentacion
report/                    Esqueleto del articulo IEEE
results/                   Metricas, graficas y modelos generados
src/aa_futbol/             Paquete principal
tests/                     Pruebas unitarias y de ausencia de leakage
```

## Instalacion

Se requiere **Python 3.12**. Desde la raiz del repositorio:

```bash
make setup
```

El comando crea `.venv` e instala las dependencias del proyecto, pruebas y
notebook. Si `python3.12` no existe, hay que instalarlo antes de ejecutar el
comando.

## Flujo reproducible

```bash
# 1. Regenerar el CSV limpio desde el ZIP original
make clean-data

# 2. Ejecutar pruebas
make test

# 3. Comprobar estilo
make lint

# 4. Corrida rapida para desarrollar y detectar errores
make experiment

# 5. Corrida final con grillas y cinco folds temporales
make experiment-full

# 6. Abrir el notebook
make notebook
```

La corrida completa escribe en `results/`:

- `metrics_summary.csv`: accuracy, macro-F1 y metricas por clase;
- `experiment_details.json`: parametros, reportes y matrices;
- `test_predictions.csv`: prediccion de cada modelo para analisis cualitativo;
- `validation_*.csv/png`: error temporal al variar hiperparametros;
- `confusion_*.png`: matrices de confusion finales;
- `models/*.joblib`: modelos ajustados con datos hasta 2023.

## Protocolo experimental

1. Los partidos se ordenan por fecha.
2. Los atributos de un partido solo usan resultados de fechas anteriores. Todos
   los partidos del mismo dia se procesan antes de actualizar historiales.
3. Los partidos hasta 2023 inclusive forman el conjunto de entrenamiento.
4. La seleccion de hiperparametros usa ventanas expansivas y fechas completas.
5. Los partidos de 2024 y 2025 se reservan para una unica evaluacion final.
6. Codificacion y discretizacion se ajustan dentro de cada `Pipeline`, usando
   solamente el fold de entrenamiento.
7. La semilla global es `42`.

Los atributos `gh`, `ga`, `winner` y `full_time` nunca ingresan a los modelos.

## Supuestos que debe confirmar el grupo

El enunciado no define completamente la forma del arbol ni la distribucion previa
del m-estimate. Esta implementacion usa:

- ID3 categorico, con ramas multiples y entropia;
- `P(v|c) = (n(v,c) + m p(v)) / (n(c) + m)`, con `p(v)` uniforme;
- empate en el baseline resuelto a favor del local;
- actualizacion causal de atributos entre fechas del conjunto de evaluacion, sin
  reentrenar los modelos.

Antes de cerrar la entrega, comparar estas decisiones con las diapositivas y
practicos del curso. Ver [`docs/modeling_decisions.md`](docs/modeling_decisions.md).

## Trabajo en equipo

No versionar `.venv`, `results/`, CSV procesados ni notebooks con salidas enormes.
Cada integrante debe poder reproducir una corrida desde el ZIP original. Las
reglas de colaboracion estan en [`CONTRIBUTING.md`](CONTRIBUTING.md).
GitHub Actions ejecuta pruebas y estilo automaticamente en cada push y pull
request.

El uso de herramientas de IA esta permitido en categoria 3, pero debe declararse.
El borrador de declaracion y la lista de verificaciones estan en
[`AI_USAGE.md`](AI_USAGE.md).
