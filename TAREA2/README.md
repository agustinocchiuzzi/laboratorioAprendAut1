# Tarea 2 — LunarLander DQN

Entrada para la entrega y su reproducción. La limpieza afecta únicamente a `TAREA2/` del repositorio.

- [Notebook integrado ejecutado](entrega_borrador.ipynb): candidato autocontenido con resultados, pesos, curvas, GIF y cortes.
- [Informe IEEE](informe/informe.tex): todavía requiere redacción del equipo, compilación y revisión de los marcadores `REDACTAR`.
- [Estado y pendientes](CIERRE_02_11.md).
- [Resultados conjuntos](analysis/team_study/README.md) y [protocolo](protocols/team_final.json): 32 políticas seleccionadas, ocho por condición.
- [Diferencias entre plataformas](PLATAFORMAS.md): procedencia y límites para interpretar las ablaciones.
- [Apuntes de arquitectura](ARQUITECTURA_METODOLOGIA.md).
- [Referencias históricas y registro de limpieza](REFERENCIAS_LIMPIEZA.md): documentos anteriores íntegros, inventario y recuperación desde Git.

Se conservan `results/` con datos y pesos originales, `preparation/` con procedencia y fuentes del ensamblado, `analisis_politica/`, figuras del informe, protocolos, scripts y tests. No borrar esas fuentes: permiten reproducir y auditar el estudio. `lander.ipynb` es el cuaderno común para entrenamiento; `course_template.ipynb` conserva la plantilla docente. `AgustinAgente/EXPLICACION_AgenteDQN.md` se conserva porque el informe lo referencia.

Desde `TAREA2/`, crear un entorno con Python 3.12 e instalar dependencias:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

En Windows usar `.venv\Scripts\python.exe`. En esta máquina también está disponible el entorno `../../.venv`. `requirements-analysis-lock.txt` registra el entorno de análisis actual; `requirements-lock.txt`, el entorno histórico de A.

Reproducir el ensamblado y verificar el notebook sin entrenar:

```sh
python scripts/assemble_team_study.py
python scripts/build_delivery_draft.py --replace-draft
python scripts/verify_delivery_draft.py
```

Estos comandos actualizan los artefactos generados. Para nuevas corridas, consultar `python scripts/run_notebook.py --help`; cada corrida guarda configuración, semillas y versiones. El paquete a entregar debe incluir el notebook y el informe definitivo según la consigna. El estado actual aún requiere revisión del equipo.
