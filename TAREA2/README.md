# LunarLander DQN — integración de Tarea 2

El trabajo de Thiago (A), Fran (B) y Agustín (C) está reunido en la rama
`integration/assignment-2`. La integración modifica únicamente `TAREA2/`.

- [Integración, ramas revisadas, evidencia y plan de cierre](INTEGRATION.md)
- [Cuaderno común editable](lander.ipynb)
- [Piloto de integración ejecutado](integration-pilot_results.ipynb)
- [Original de Fran: replay](LunarLander_DQN_B.ipynb)
- [Original de Agustín: exploración](AgustinAgente/LunarLander_DQN_AgustinAgente.ipynb)
- [Propuesta para el estudio final](protocols/team_draft.json)
- [Herramientas y decisiones pendientes del equipo](TEAM_AGREEMENT.md)

Desde `TAREA2/`, usando el entorno ya instalado en esta máquina:

```sh
../../.venv/bin/python scripts/run_notebook.py --mode integration-pilot
../../.venv/bin/python -m unittest discover -s tests -v
```

En otra máquina, crear un entorno e instalar `requirements.txt`; usar su Python
en lugar de `../../.venv/bin/python`. `requirements-lock.txt` conserva las versiones
del entorno de A. El notebook original de C instala además SciPy para sus tests
estadísticos; la integración común no depende de SciPy.

El agente común permite desactivar red objetivo, replay o exploración por separado.
También admite capacidades distintas de replay, exploración lineal por episodios,
lineal por pasos o exponencial por episodios, y red objetivo dura o suave. Los defaults
conservan A0. El piloto usa 20 episodios de entrenamiento y 10 de evaluación por
condición; comprueba funcionamiento, no rendimiento final.

Para una corrida seleccionada de 300 episodios y 200 evaluaciones, indicar
`--mode comparison` explícitamente:

```sh
../../.venv/bin/python scripts/run_notebook.py --mode comparison --config A0 --seed 42 --variant full
../../.venv/bin/python scripts/run_notebook.py --mode comparison --config A0 --seed 42 --variant no_replay
```

Las otras variantes son `no_target` y `no_exploration`. `--mode pilot --config A0`
respeta el presupuesto corto. Las opciones `--epsilon-schedule`, `--replay-capacity`
y `--target-update` requieren `--config` y quedan registradas en `config.json`.
Cada ejecución crea una carpeta nueva. No ejecutar varios experimentos en el mismo
proceso/kernel: el constructor fija generadores globales.

Los resultados históricos tienen presupuestos y configuraciones diferentes.
El estudio final con una referencia común, las comparaciones estadísticas y el
informe IEEE siguen pendientes. Los cortes de política de A están disponibles en
`preparation/A0_seed42_policy_slices/` y pueden regenerarse con
`scripts/study_tools.py slices`.

Documentación histórica de A: [introducción](START_HERE_A.md),
[resultados iniciales](RUN_STATUS_A.md), [comandos originales](README_A.md).
