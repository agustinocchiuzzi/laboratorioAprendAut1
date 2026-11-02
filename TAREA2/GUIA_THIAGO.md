# Guía de trabajo de Thiago — cierre de Tarea 2

Objetivo: completar tus 11 entrenamientos, reunir las 32 corridas del equipo e integrar un notebook reproducible y la sección de arquitectura/metodología. El 3/11 queda para revisión y el 4/11 para entrega y correcciones.

Esta guía traduce el cronograma acordado a tareas verificables. No acredita que ya estén hechas. El archivo `protocols/team_draft.json` todavía está marcado como borrador: confirmar las decisiones técnicas con Fran y Agustín antes de tomar sus valores como definitivos.

## Por dónde empezar hoy, 1/11

- [ ] Confirmar cuál es la versión común del código y dónde quedó el protocolo definitivo.
- [ ] Pedir a cada compañero las carpetas de resultados y el estado de cada combinación variante/semilla.
- [ ] Comprobar tus 8 corridas sin red objetivo y tus 3 del agente completo.
- [ ] Identificar faltantes, fallos o configuraciones incompatibles antes de juntar métricas.
- [ ] Centralizar las corridas válidas, generar las curvas y pasar a Fran los datos por semilla.
- [ ] Resolver con el equipo las corridas pendientes. Si falta cómputo, priorizar completarlas; no iniciar nuevas búsquedas de hiperparámetros después del 1/11.

## 30/10 — Cerrar el protocolo

Tu responsabilidad es coordinar y dejar por escrito una única referencia y un presupuesto común.

- [ ] Registrar la revisión del código que usarán todos y las versiones del entorno.
- [ ] Confirmar arquitectura, optimizador, tasa de aprendizaje, gamma, replay, frecuencia de actualización, red objetivo y exploración.
- [ ] Confirmar cantidad de episodios, límite de pasos y checkpoint que se evaluará.
- [ ] Con Fran, cerrar qué significa «sin replay». La implementación integrada usa solo la transición actual, con batch efectivo de 1; dejar documentado ese efecto.
- [ ] Con Agustín, cerrar semillas de entrenamiento, semillas de evaluación, métricas, método estadístico y corrección para las tres comparaciones.
- [ ] Guardar el protocolo definitivo y compartirlo antes de lanzar las corridas restantes.

El borrador propone A0 (capas 64/64, learning rate 0.001), 300 episodios, hasta 1000 pasos por episodio y 200 evaluaciones greedy. Son propuestas existentes, no decisiones confirmadas por esta guía. Cada ablación debe comenzar desde pesos nuevos y cambiar únicamente el componente acordado.

**Para dar esta etapa por cerrada:** los tres pueden identificar el mismo código, configuración, presupuesto, lista de semillas y evaluación. El runner no carga automáticamente el JSON del protocolo: comprobar que sus opciones y la configuración efectiva coincidan con lo acordado.

## 31/10 — Ejecutar tus 11 corridas

Te corresponden 8 semillas de `no_target` y 3 de `full`. Fran realiza 8 de `no_replay` y 3 de `full`; Agustín, 8 de `no_exploration` y 2 de `full`.

Si se confirma la lista del borrador, esta asignación concreta evita duplicados. El reparto de semillas de `full` es una propuesta de organización:

| Semilla | Sin red objetivo | Sin replay | Sin exploración | Agente completo |
| --- | --- | --- | --- | --- |
| 42 | Thiago | Fran | Agustín | Thiago |
| 1042 | Thiago | Fran | Agustín | Thiago |
| 2042 | Thiago | Fran | Agustín | Thiago |
| 3042 | Thiago | Fran | Agustín | Fran |
| 4042 | Thiago | Fran | Agustín | Fran |
| 5042 | Thiago | Fran | Agustín | Fran |
| 6042 | Thiago | Fran | Agustín | Agustín |
| 7042 | Thiago | Fran | Agustín | Agustín |

- [ ] Llevar un registro con variante, semilla, responsable, estado, carpeta de salida e incidencia.
- [ ] Ejecutar una primera corrida de cada variante y revisar entre todos las salidas antes de lanzar las restantes.
- [ ] Verificar presupuesto, switches de componentes y evaluación en `config.json`.
- [ ] Guardar por corrida `config.json`, `training.csv`, `evaluation.csv`, `summary.json` y `online_weights.pt`.
- [ ] Conservar corridas débiles y fallidas, con la causa documentada. Un retorno bajo no justifica descartarlas ni repetir hasta obtener un buen resultado.

Ejemplo para esta máquina, desde `laboratorioAprendAut1/TAREA2/`, **solo si A0 y el presupuesto del runner coinciden con el protocolo definitivo**:

```sh
../../.venv/bin/python scripts/run_notebook.py --mode comparison --config A0 --seed 42 --variant no_target
../../.venv/bin/python scripts/run_notebook.py --mode comparison --config A0 --seed 42 --variant full
```

Cambiar la semilla para cada corrida asignada. Usar `comparison` explícitamente: el modo por defecto es un piloto corto. No ejecutar experimentos simultáneos dentro del mismo proceso o kernel, porque el agente fija generadores globales. En otra máquina, usar el Python del entorno instalado allí.

**Para cerrar:** tus 11 corridas están completas y localizables, con su configuración y archivos; el equipo tiene un registro de las 32 combinaciones previstas.

## 1/11 — Centralizar resultados y curvas

- [ ] Reunir las carpetas originales sin sobrescribir resultados históricos.
- [ ] Validar que las 32 corridas coincidan con el protocolo: 8 por condición, sin combinaciones repetidas ni faltantes.
- [ ] Verificar episodios completados, pasos, semillas y evaluación; separar los pilotos de las corridas finales.
- [ ] Registrar incidentes y repetir solo cuando exista un fallo técnico justificado, usando la misma configuración y semilla.
- [ ] Generar curvas de retorno durante entrenamiento para las cuatro condiciones, mostrando variabilidad entre semillas e indicando cualquier suavizado.
- [ ] Usar un eje de episodios común y documentar también los pasos consumidos: igual cantidad de episodios no implica igual cantidad de interacciones.
- [ ] Preparar una tabla por variante y semilla con retorno medio de evaluación, tasa de éxito, pasos y tiempo.
- [ ] Pasar a Fran esa tabla y el protocolo; pasar a Agustín los checkpoints y sus configuraciones.

La herramienta `scripts/study_tools.py aggregate` valida archivos y compatibilidad y produce estadísticas descriptivas. Usar `--protocol` con la ruta del protocolo definitivo, `--runs` con las carpetas de las corridas y `--output` con una ruta nueva. No realiza los tests estadísticos de Fran.

La unidad de comparación es la política entrenada con una semilla: hay 8 políticas por condición. Los 200 episodios de evaluación de una política no representan 200 entrenamientos independientes. Distinguir la dispersión entre semillas de la dispersión entre episodios de una sola política.

**Para cerrar:** conjunto de resultados validado, curvas y tabla compartidas, y lista explícita de cualquier faltante. Si no se alcanzan 8 semillas por condición, informar el número real y acordar cómo proceder sin ocultar la limitación.

## 2/11 — Integrar notebook y escribir metodología

- [ ] Integrar en el notebook final la configuración acordada, carga de resultados, curvas, tablas, análisis estadístico de Fran y análisis de política de Agustín.
- [ ] Distinguir las celdas que entrenan de las que cargan resultados guardados; documentar tiempos y pasos para reproducir cada parte.
- [ ] Escribir arquitectura: observación y acciones del entorno, red Q, capas, activaciones y selección de acciones.
- [ ] Escribir metodología: actualización DQN, replay, red objetivo, exploración, presupuesto, semillas, evaluación y definición exacta de cada ablación.
- [ ] Explicar cómo se seleccionó la configuración y qué resultados son de calibración frente a los del estudio final.
- [ ] Documentar hardware, dependencias, código utilizado y archivos necesarios para reproducir resultados.
- [ ] Coordinar con Agustín el espacio de tu sección dentro del informe IEEE.

**Para cerrar:** notebook integrado y texto de arquitectura/metodología listos para revisión; todas las figuras y tablas tienen una fuente identificable.

## 3/11 — Ejecutar y revisar

- [ ] Reiniciar el kernel y ejecutar el notebook en orden siguiendo el procedimiento de reproducción documentado.
- [ ] Comprobar rutas, imports, dependencias y disponibilidad de resultados y checkpoints.
- [ ] Verificar que no dependa de variables de sesiones anteriores ni de rutas personales.
- [ ] Contrastar cifras del notebook con tablas y texto del informe.
- [ ] Incorporar la revisión de métricas y conclusiones de Fran y la revisión de formato, referencias y páginas de Agustín.
- [ ] Hacer revisión cruzada: leer las secciones ajenas y resolver observaciones entre los tres.

**Para cerrar:** notebook verificado, cifras consistentes e informe revisado. No dejar la primera ejecución completa para el día de entrega.

## 4/11 — Empaquetar y entregar

- [ ] Comprobar con la consigna los archivos obligatorios, nombres, formato y límite de páginas.
- [ ] Preparar el paquete con notebook, informe y los archivos necesarios según la consigna.
- [ ] Abrir el paquete final y comprobar que los archivos estén completos y legibles.
- [ ] Hacer revisión conjunta, entregar y guardar constancia de recepción.

Reservar este día para correcciones y entrega. Si una corrección cambia resultados, revisar también las tablas y conclusiones afectadas.

## Referencias del proyecto

- [README de la integración y comandos](README.md).
- [Decisiones técnicas pendientes y formato de resultados](TEAM_AGREEMENT.md).
- [Protocolo todavía en borrador](protocols/team_draft.json).
- [Notebook común](lander.ipynb).
- [Documentación de integración](INTEGRATION.md).
