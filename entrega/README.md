# Aprendizaje Automático 2026 - Tarea 1

Thiago Rivas, Juan Duarte y Agustin Occhiuzzi.

## Contenido

- `Informe_final.pdf`: artículo en formato IEEE con métodos y resultados.
- `notebook.ipynb`: flujo completo ejecutado, con resultados y ejemplos.
- `id3.py`: implementación propia de ID3.
- `naive_bayes.py`: implementación propia de Naive Bayes con m-estimate.
- `futbol_uruguayo.zip`: dataset crudo original, leído directamente por el notebook.
- `requirements.txt`: dependencias verificadas.
- `README.md`: estas instrucciones.

La limpieza, atributos, discretización, particiones y clasificador base están
definidos dentro del notebook. No se necesitan otros archivos del repositorio.

## Ejecutar

Usar Python 3.12. Se verificó con Python 3.12.14 y scikit-learn 1.9.1.
Desde esta carpeta:

```sh
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r requirements.txt
.venv312/bin/python -m ipykernel install --user --name aprendaut1 --display-name 'AprendAut1 Python 3.12'
```

En Windows, los ejecutables del entorno están en `.venv312\Scripts\`.
Abrir `notebook.ipynb` con Jupyter o VS Code, seleccionar el kernel
**AprendAut1 Python 3.12**, reiniciarlo y ejecutar todas las celdas en orden.
Mantener esta carpeta como directorio de trabajo. `RAW` apunta al ZIP adjunto
y `RUN_FINAL_TEST=True` ya está activado. Las salidas también están guardadas
para poder leer el notebook sin ejecutarlo.

El notebook vuelve a calcular la selección y entrena todos los modelos desde
el ZIP. Incluye precisión, recall y F1 por clase, accuracy, macro-F1, matrices,
ejemplos reales y clasificación de tres partidos hipotéticos. La semilla es 42.

## Resultados de referencia

Entrenamiento: 14.705 partidos hasta 2023. Evaluación: 472 de 2024-2025.

| Modelo | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| ID3 | 0,4492 | 0,4238 |
| NB propio | 0,4788 | 0,4454 |
| CategoricalNB | 0,4788 | 0,4454 |
| Random Forest | 0,4258 | 0,4219 |
| Base de diez años | 0,4619 | 0,3564 |

Los históricos solo usan fechas anteriores a cada encuentro; modelos y cortes
permanecen fijos durante la evaluación. El bloque de test ya fue inspeccionado
en versiones anteriores y 2025 llega hasta junio; las limitaciones se explican
en el informe.
