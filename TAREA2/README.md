# Tarea 2: LunarLander con DQN

## Qué se entrega

| Entregable | Archivo |
|---|---|
| Notebook (copia de la plantilla con los tres TODO resueltos) | [`LunarLander_DQN.ipynb`](LunarLander_DQN.ipynb) |
| Informe IEEE (fuente) | [`informe/informe.tex`](informe/informe.tex) + [`informe/figuras/`](informe/figuras/) |

El PDF del informe se genera compilando `informe/informe.tex` (por ejemplo en Overleaf, con
pdfLaTeX). Hay que comprobar que tenga como máximo 5 páginas.

## Resultados principales

- **Agente principal** (1200 episodios): 81 % de éxito (162/200) en escenarios de test (30000–30199) que no se
  usaron para ninguna decisión; en la evaluación de la plantilla (10000–10199), el mismo principal
  obtuvo 77,5 % (155/200), frente al 0 % (0/200) del agente aleatorio.
- **Calibración**: A0 contra 9 alternativas, 3 semillas cada una, en escenarios de validación.
  Solo se adoptó explorar durante 100 episodios en lugar de 200.
- **Ablación**: 4 condiciones × 8 semillas, todas en la misma computadora. El completo supera en
  promedio a las tres variantes, pero el test exacto bilateral de signos emparejado no rechaza ninguna hipótesis
  tras la corrección de Holm (tres comparaciones, alfa = 0,05). No rechazar no demuestra equivalencia.

## Cómo ejecutar el notebook

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # en Linux/macOS: .venv/bin/python
```

Abrir `LunarLander_DQN.ipynb` con ese entorno y ejecutar todas las celdas. El notebook trae
adentro los resultados de las 63 corridas (celda *Datos guardados*, sección 7.1), así que con
`REENTRENAR = False` reutiliza las 63 corridas principales. La ejecución completa sí incluye
entrenamientos cortos de reproducibilidad y nuevas evaluaciones. Al ejecutarse escribe
`aterrizaje_dqn.gif` (ignorado por git) y regenera las figuras de `informe/figuras/`.

Para reentrenar todo desde cero, poner `REENTRENAR = True` en la celda de bibliotecas adicionales.
Las 63 corridas tardan más de 1 hora con 8 núcleos y guardan sus resultados en `resultados/`. En
otra computadora los números no van a ser exactamente los mismos.

## Otros archivos

- `Tarea - Aprendizaje por Refuerzos (2026).pdf`: la letra.
- `AgustinAgente/`: apuntes de estudio y una primera versión del agente. No forman parte de la
  entrega.

Las versiones anteriores (notebooks intermedios, corridas en varias computadoras, scripts y
revisiones) se quitaron para simplificar. Siguen disponibles en el historial de git, en el commit
`3cfa592` de la rama `integration/assignment-2`.

## Análisis estadístico de la ablación

La unidad experimental es la política entrenada. Se promedian sus 200 retornos de evaluación
guardados y se alinean las condiciones por semilla. Para cada variante se calcula
`D_s = media_completo_s − media_variante_s`. El test principal elegido por los autores es
`scipy.stats.binomtest(k, n, p=0.5, alternative="two-sided")`: `k` cuenta diferencias positivas
y `n` diferencias no nulas. Los empates exactos se informan y se excluyen de `n` (en esta
versión no hay); si todos fueran empates, el test no es aplicable y se usa `p=1` para Holm.

Se contrasta si una diferencia no nula tiene igual probabilidad de ser positiva o negativa,
no la igualdad de medias. Se requiere independencia entre pares y se pierde información
sobre magnitudes; la diferencia de medias se conserva como descripción. La letra y la
plantilla piden un test adecuado, sin prescribir nuestra elección. El alcance es exploratorio:
las semillas 42, 1042 y 2042 reutilizan entrenamientos de calibración; cambiar el test no elimina
el sesgo de selección.

Para recalcular solo este análisis, mantener `REENTRENAR=False` y usar una copia temporal:
extraer `CORRIDAS_GUARDADAS` de la celda de datos, seleccionar las claves `estudio_`, recuperar
`SEMILLAS_ESTUDIO`, `EPISODIOS_TEST` y `NOMBRES`, y ejecutar únicamente la última celda
estadística con NumPy y SciPy. Ejecutar todo el notebook también entrena las comprobaciones
cortas de reproducibilidad, reevalúa políticas y regenera figuras, aun con ese indicador en False.
La tabla y la explicación del informe usan este mismo análisis emparejado.
