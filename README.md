# Tarea 1: predicción de resultados del fútbol uruguayo

Autores: Thiago Rivas, Juan Duarte y Agustin Occhiuzzi.

La entrega autocontenida está en `entrega/`: informe IEEE, notebook ejecutado,
las dos implementaciones propias, dataset crudo, dependencias e instrucciones.
El notebook raíz reproduce el mismo flujo usando los módulos de `src/`.

## Reproducción

Se verificó con Python 3.12.14 y scikit-learn 1.9.1. Las dependencias están
fijadas a las versiones utilizadas; la semilla es 42. Desde la raíz:

```sh
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r requirements.txt
.venv312/bin/python -m ipykernel install --user --name aprendaut1 --display-name 'AprendAut1 Python 3.12'
```

Abrir `entrega/notebook.ipynb` en Jupyter o VS Code, seleccionar ese kernel,
reiniciarlo y ejecutar todas las celdas. El directorio de trabajo debe ser
`entrega/`, donde están los modelos y `futbol_uruguayo.zip`.
`RUN_FINAL_TEST=True` está activado: se reproducen selección, ajuste final,
métricas, matrices, ejemplos y predicciones de partidos hipotéticos.
`False` conserva un modo opcional que omite evaluación y demostraciones.
El notebook no lee resultados precalculados ni archivos de procesamiento parcial.

## Resultados y verificaciones

Se usan 14.705 partidos hasta 2023 y 472 de 2024-2025. El test fue
inspeccionado previamente: estas cifras no representan un nuevo holdout intacto.
`results/validation/` conserva las búsquedas; `results/finalized/` contiene
métricas, matrices, ejemplos, escenarios, comprobaciones y versiones/hashes.

```sh
.venv312/bin/python -B -m unittest discover -s tests -v
.venv312/bin/python -B scripts/verify_final_experiment.py
```

El verificador reproduce el ajuste y evaluación con la configuración congelada,
sin repetir la búsqueda. La ejecución completa del notebook sí repite las búsquedas.

## Informe y paquete de entrega

El fuente vigente es `informe.tex`, con las imágenes de `figuras/`.
`scripts/plot_validation.py` regenera esas imágenes desde los registros de
validación. Compilar el fuente con una distribución LaTeX que incluya IEEEtran
(dos pasadas), revisar el PDF y guardar el resultado como `Informe_final.pdf`.

Después de guardar el notebook ejecutado y el PDF actualizado:

```sh
.venv312/bin/python scripts/build_submission.py
```

El comando sincroniza los archivos de `entrega/` y genera
`output/entrega/Tarea1_Rivas_Duarte_Occhiuzzi.zip`. Solo incluye los siete
archivos enumerados en `entrega/README.md`; no incluye modelos serializados,
datasets procesados, cachés ni resultados intermedios.

`old/`, `notas_decisiones.md`, los notebooks históricos de `results/` e
`informe_overleaf.zip` son antecedentes, no la entrega vigente. El ZIP de
Overleaf contiene una versión anterior y no debe usarse para compilar el informe.
