# Tarea 2: LunarLander con DQN

## Qué se entrega

| Entregable | Archivo |
|---|---|
| Notebook (copia de la plantilla con los tres TODO resueltos) | [`LunarLander_DQN.ipynb`](LunarLander_DQN.ipynb) |
| Informe IEEE (fuente) | [`informe/informe.tex`](informe/informe.tex) + [`informe/figuras/`](informe/figuras/) |

El PDF del informe se genera compilando `informe/informe.tex` (por ejemplo en Overleaf, con
pdfLaTeX). Hay que comprobar que tenga como máximo 5 páginas.

## Resultados principales

- **Agente principal** (1200 episodios): 81 % de éxito (162/200) en escenarios de test que no se
  usaron para ninguna decisión; el agente aleatorio, 0 %.
- **Calibración**: A0 contra 9 alternativas, 3 semillas cada una, en escenarios de validación.
  Solo se adoptó explorar durante 100 episodios en lugar de 200.
- **Ablación**: 4 condiciones × 8 semillas, todas en la misma computadora. El completo supera en
  promedio a las tres variantes, pero ninguna diferencia es significativa con Mann–Whitney y
  corrección de Holm.

## Cómo ejecutar el notebook

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # en Linux/macOS: .venv/bin/python
```

Abrir `LunarLander_DQN.ipynb` con ese entorno y ejecutar todas las celdas. El notebook trae
adentro los resultados de las 63 corridas (celda *Datos guardados*, sección 7.1), así que con
`REENTRENAR = False` **no reentrena** y tarda un par de minutos. Al ejecutarse escribe
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
