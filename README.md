# Reproducción de la entrega final

La entrega vigente es `entrega/notebook.ipynb`, junto con `entrega/id3.py`,
`entrega/naive_bayes.py` e `Informe_final.pdf`. El notebook raíz reproduce
el mismo flujo usando los módulos de `src/`.

## Entorno y ejecución

Se verificó con Python 3.12.14 y scikit-learn 1.9.1. La semilla es 42 para
Random Forest, los árboles de referencia y las muestras ilustrativas.
Las versiones completas y hashes del código/dataset de la evaluación guardada
están en `results/finalized/environment.json`.

Desde la raíz del repositorio:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name aprendaut1 --display-name 'AprendAut1 Python 3.12'
```

Abrir el notebook en Jupyter o VS Code y seleccionar ese kernel. Comprobar
la ruta `RAW` de la primera celda de código. Para reproducir el informe,
cambiar `RUN_FINAL_TEST = True` y ejecutar todas las celdas en orden desde
un kernel nuevo. El valor predeterminado `False` permite revisar únicamente
validación y ajuste; omite evaluación e instancias nuevas.
El notebook de entrega contiene todo el procesamiento necesario y no lee
tablas de resultados ni archivos intermedios.

## Resultados vigentes

`results/validation/` conserva las búsquedas y selecciones. `results/finalized/`
contiene métricas, matrices, ejemplos, escenarios y comprobaciones del ajuste
final: 14.705 partidos hasta 2023 y 472 de 2024–2025. Las últimas celdas del
notebook reproducen esas tablas, incluidos los ejemplos y escenarios del informe.
El test fue inspeccionado previamente; no es una evaluación independiente nueva.

`scripts/verify_final_experiment.py` reproduce únicamente el ajuste y evaluación
con la configuración congelada, sin repetir la búsqueda de hiperparámetros.
Desde la raíz: `.venv/bin/python -B scripts/verify_final_experiment.py`.

`old/`, `notas_decisiones.md`, `results/notebook_execution.txt`,
`results/notebook.executed.ipynb`, `results/notebook.merged.executed.ipynb` y
`informe_overleaf.zip` son antecedentes de versiones anteriores; no son la
entrega final ni la evidencia vigente de sus resultados. El fuente actual
del PDF es `informe.tex` con las imágenes de `figuras/`.
