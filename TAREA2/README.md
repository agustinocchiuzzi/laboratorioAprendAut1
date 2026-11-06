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
  promedio a las tres variantes, pero ninguna diferencia es significativa con Wilcoxon pareado y
  corrección de Holm.

## Cómo ejecutar el notebook

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # en Linux/macOS: .venv/bin/python
```

Abrir `LunarLander_DQN.ipynb` con ese entorno y ejecutar todas las celdas. El notebook trae
adentro los resultados de las 63 corridas (celda *Datos guardados*, sección 7), así que **no
reentrena** y tarda unos 2 minutos. Al ejecutarse escribe `resultados/` y `aterrizaje_dqn.gif`
(ignorados por git) y regenera las figuras de `informe/figuras/`.

Para reentrenar todo desde cero: borrar `resultados/` y la celda *Datos guardados*. Las 63
corridas tardan cerca de 1 hora con 8 núcleos.

## Otros archivos

- `Tarea - Aprendizaje por Refuerzos (2026).pdf`: la letra.
- `AgustinAgente/`: apuntes de estudio y una primera versión del agente. No forman parte de la
  entrega.

Las versiones anteriores (notebooks intermedios, corridas en varias computadoras, scripts y
revisiones) se quitaron para simplificar. Siguen disponibles en el historial de git, en el commit
`3cfa592` de la rama `integration/assignment-2`.
