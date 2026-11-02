# Análisis descriptivo parcial de Thiago

11 políticas validadas: 8 sin red objetivo y 3 completas. No hay corridas finales compatibles sin replay o sin exploración disponibles en este conjunto local; los pilotos y resultados históricos incompatibles se registran en local_run_audit.csv.

## Archivos

- [Curvas PNG](learning_curves.png) y [PDF](learning_curves.pdf).
- [Datos de curvas](learning_curves.csv), con retornos originales y suavizados.
- [Métricas por política](metrics.csv).
- [Evaluación emparejada](paired_evaluation.csv): completo menos sin red objetivo, para semillas 42, 1042 y 2042.
- [Auditoría de resultados locales](local_run_audit.csv).
- [Resumen validado](validated_summary.json).
- [Paquete para compartir](thiago_runs.zip), con los archivos originales de las 11 políticas y estos resultados.

## Cómo leer las curvas

Cada línea de los dos primeros paneles representa una política entrenada con una semilla. El retorno se suaviza con una media de los últimos 20 episodios (menos al inicio). El tercer panel usa únicamente las 3 semillas presentes en ambas condiciones: muestra la media de las curvas suavizadas y ±1 desviación estándar muestral entre semillas. La banda no es un intervalo de confianza. No se rellenan semillas faltantes ni se seleccionan las mejores corridas.

Los CSV conservan episodios y pasos acumulados. El presupuesto es igual en episodios; los pasos y tiempos efectivamente consumidos pueden diferir. No inferir eficiencia computacional a partir de tiempos de ejecuciones con distinta carga de máquina.

## Límites

El estudio de 32 políticas sigue incompleto. La comparación emparejada tiene solo 3 semillas; no se realizaron pruebas estadísticas ni se declara significación. Un resultado bajo es un resultado válido, no un motivo para descartar una semilla.

Evaluación greedy de los pesos finales: 200 episodios, semillas de desarrollo 10000–10199, usadas previamente para calibración. No se usaron semillas reservadas. La unidad de análisis es la política entrenada, no cada episodio de evaluación.

A0 fue elegida por el usuario. La compatibilidad se validó contra team_draft.json; las demás decisiones de ese documento siguen figurando como borrador. El manifiesto identifica la revisión y el hash del notebook de las corridas.

## Corridas que faltan reunir

- full: [3042, 4042, 5042, 6042, 7042]
- no_replay: [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042]
- no_exploration: [42, 1042, 2042, 3042, 4042, 5042, 6042, 7042]

Pedir las carpetas completas a Fran y Agustín y validar su compatibilidad antes de incorporarlas. No reemplazar esos faltantes con pilotos ni con entrenamientos de presupuestos distintos.
