# Análisis final de Naive Bayes: texto para el informe

Se analizaron las predicciones finales sobre 472 partidos admitidos entre
el 17/02/2024 y el 30/06/2025 (297 de 2024 y 175 de 2025, año incompleto),
sin volver a seleccionar ni ajustar modelos o hiperparámetros con test.
NB propio y CategoricalNB coinciden en las 472 predicciones.

| Modelo | Aciertos | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| NB propio / CategoricalNB | 226/472 | 0.4788 | 0.4454 |
| Baseline de diez años | 218/472 | 0.4619 | 0.3564 |

| Clase | Soporte | Precision NB | Recall NB | F1 NB | F1 baseline |
|---|---:|---:|---:|---:|---:|
| E (empate) | 131 | 0.3537 | 0.2214 | 0.2723 | 0.0000 |
| L (victoria local) | 190 | 0.5475 | 0.6368 | 0.5888 | 0.5615 |
| V (victoria visitante) | 151 | 0.4497 | 0.5033 | 0.4750 | 0.5079 |

L obtiene las mejores precision, recall y F1, seguida de V y E.
NB reconoce 29 de 131 empates reales y predice 82: omite 102 y genera
53 falsos empates. El baseline nunca predice E.

La matriz NB, con filas reales y columnas predichas, es:

| Real / Predicción | E | L | V |
|:---:|---:|---:|---:|
| E | 29 | 54 | 48 |
| L | 24 | 121 | 45 |
| V | 29 | 46 | 76 |

Predominan los empates confundidos con victorias: 102/246 errores (41.46%).
E→L es la confusión individual más frecuente (54), seguida de E→V (48).
Las confusiones L↔V suman 91 (36.99%) y los falsos empates 53 (21.54%).

NB mejora 1.69 puntos porcentuales de accuracy y 0.0889 de macro-F1.
Corrige 73 errores del baseline, pero pierde 65 de sus aciertos: saldo +8.
Por clase, recupera 29 empates adicionales, mantiene 121 aciertos locales
y baja de 97 a 76 visitantes. La mejora no es uniforme: aumenta F1 de E
y L, pero disminuye F1 de V. No se comprobó significación estadística.

Para seleccionar ejemplos se ordenó por fecha, local y visitante y se tomó
el primer partido de cada par (clase real, predicción NB). El notebook incluye
las nueve celdas; aquí se muestran E→E, E→L, L→E y V→V. El criterio es
reproducible e ilustrativo, no representativo de frecuencias. Los marcadores
se unieron desde los datos originales limpios, sin usarlos como atributos.

| Fecha | Local – Visitante | Marcador | Real | NB | Base |
|---|---|:---:|:---:|:---:|:---:|
| 2024-03-11 | Montevideo Wanderers – Deportivo Maldonado | 0-0 | E | E | L |
| 2024-02-25 | Cerro Largo FC – CA Fenix | 0-0 | E | L | L |
| 2024-04-07 | CA Cerro – Rampla Juniors Futbol Club | 3-0 | L | E | L |
| 2024-02-17 | CA Fenix – Danubio | 1-2 | V | V | V |

El primer caso ilustra un empate recuperado por NB; el tercero, un falso
empate donde el baseline acierta. No se atribuyen causas individuales.

Como diagnóstico posterior al test, se particionaron todos los casos según
el signo de la diferencia de puntos por partido en hasta cinco antecedentes,
sin buscar umbrales:

| Promedio reciente | Casos | Reales E/L/V | Aciertos NB (accuracy) | Aciertos base (accuracy) |
|---|---:|:---:|---:|---:|
| Local > visitante | 216 | 62/106/48 | 116 (53.70%) | 99 (45.83%) |
| Local = visitante | 39 | 9/15/15 | 15 (38.46%) | 21 (53.85%) |
| Local < visitante | 217 | 60/69/88 | 95 (43.78%) | 98 (45.16%) |

**Hecho observado:** NB presenta mayor accuracy con ventaja reciente local.
**Posible explicación:** esa ventaja podría aportar información útil, pero
también cambia la composición de clases y L es la clase mejor reconocida.
No se aislaron estos efectos. El grupo de igualdad solo tiene 39 casos;
los macro-F1 NB por grupo son 0.3788, 0.3737 y 0.3741, por lo que la
diferencia de accuracy no implica mejora uniforme entre clases.

Los tres bines por atributo pierden resolución y separan valores próximos
a sus cortes. Los históricos resumen ventanas distintas y las diez entradas
omiten sus denominadores. Una tasa de 0.5 puede ser neutra o una proporción
observada: no demuestra ausencia de historial, que requeriría comprobar
el conteo previo de la ventana correspondiente. NB supone independencia
condicionada a la clase; tasas, puntos y goles resumen partidos superpuestos
y podrían compartir información. No se midió esa dependencia ni su efecto.
Estas limitaciones no son causas demostradas de los errores.

La evaluación es secuencial: incorpora resultados de fechas estrictamente
anteriores, incluso del test, manteniendo fijos modelos y discretizadores.
Los hallazgos describen esta muestra y no garantizan desempeño futuro.

---

Este texto de trabajo está integrado de forma más compacta en `informe.tex`.
Respaldo: secciones 14.2–14.7 de `notebook.ipynb`, con las métricas completas
de ambos NB y baseline, matrices, nueve ejemplos y cálculos reproducibles.
Consigna: apartado 3 de `Tarea_1.pdf`.
