# Revisión de consistencia: solo validación

Entrada: entrega/notebook.ipynb. Se ejecutaron todas las celdas con
RUN_FINAL_TEST=False: selección y ajuste final con train, sin predicciones
ni métricas de test. Para esta verificación se cambió RAW solo en memoria a
data/raw/futbol_uruguayo.zip; no se copió el dataset a entrega/.

Folds expansivos: 2021, 2022, 2023. Selección por macro-F1 medio anual sin
redondear. ID3: menor umbral en empate; NB: menor m; atributos: seis antes
que diez; RF: menor profundidad y luego mayor hoja. Semilla 42.
Los históricos siguen actualizándose solo tras cada fecha completa.

| Modelo | Entradas | Parámetros seleccionados | Macro-F1 validación | Error validación |
| --- | ---: | --- | ---: | ---: |
| ID3 | 10 | min_info_gain=0.005, max_depth=8 | 0.400186 | 0.584623 |
| NB propio | 10 | m=0.1, min_categories=4 | 0.385798 | 0.572350 |
| sklearn CategoricalNB | 10 | alpha=0.025, min_categories=4 | 0.385798 | 0.572350 |
| Random Forest | 6 | max_depth=None, min_samples_leaf=20, 300 árboles | 0.422767 | 0.565298 |

Los diez atributos son las seis tasas existentes más puntos por partido y
diferencia de gol por partido de los últimos cinco antecedentes de ambos equipos.
El bosque mantiene balanced_subsample. Los CSV conservan las métricas sin
redondear. Las curvas están incorporadas al notebook y en figuras/.
Los CSV ID3/NB corresponden a la selección inicial sobre seis tasas;
feature_comparison.csv contiene la comparación posterior con parámetros fijos.

Se comprobaron cortes fijos [0.3, 0.6] exclusivamente para las dos tasas
last_5, cuantiles del train para las demás entradas, propagación de atributos
y parámetros al ajuste final, y equivalencia de probabilidades de ambos NB
sobre train. Las pruebas incluyen el caso en que ganan seis atributos y
el empate exacto; todas las celdas finales se omiten con el test desactivado.

Verificación: .venv/bin/python -B -m unittest discover -s tests -v.
Entorno disponible: Python 3.14.7, scikit-learn 1.9.1; la consigna solicita
Python 3.12. No se cambiaron ni instalaron dependencias.

Esta carpeta registra la etapa de selección, anterior a la evaluación final;
`test_evaluated=false` en selected.json describe esa etapa, no el estado actual.
La evaluación posterior verificada está en `../finalized/`, con Python 3.12.14
y scikit-learn 1.9.1. `informe.tex` e `Informe_final.pdf` ya contienen esos
resultados. El ZIP de Overleaf es histórico. Las instrucciones vigentes están
en el [README de la raíz](../../README.md).
