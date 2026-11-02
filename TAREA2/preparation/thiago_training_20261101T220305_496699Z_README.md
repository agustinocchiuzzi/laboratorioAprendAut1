# Corridas de Thiago completadas

Las 11 corridas asignadas a Thiago están completas: 8 sin red objetivo y 3 del agente completo (semillas 42, 1042 y 2042).

Configuración A0; 300 episodios, máximo 1000 pasos por episodio, sin parada anticipada. Evaluación greedy del checkpoint final: 200 episodios, semillas 10000–10199. Son semillas de desarrollo utilizadas previamente; no constituyen una evaluación final independiente.

Se comprobaron configuración, semillas, presupuesto, archivos requeridos, pasos totales y concordancia de los resúmenes con los retornos de evaluación. El hash del notebook coincide en las once corridas.

- [Registro de corridas y métricas](thiago_training_20261101T220305_496699Z_runs.csv)
- [Resumen descriptivo validado](thiago_training_20261101T220305_496699Z_validated_summary.json)
- [Manifiesto de ejecución](thiago_training_20261101T220305_496699Z.json)

## Carpetas de resultados

- full, semilla 42: [results/comparison_20261101T220308_894362Z/A0_seed42](../results/comparison_20261101T220308_894362Z/A0_seed42)
- full, semilla 1042: [results/comparison_20261101T220918_008885Z/A0_seed1042](../results/comparison_20261101T220918_008885Z/A0_seed1042)
- full, semilla 2042: [results/comparison_20261101T220958_025937Z/A0_seed2042](../results/comparison_20261101T220958_025937Z/A0_seed2042)
- no_target, semilla 42: [results/comparison_20261101T220036_407013Z/A0_no_target_seed42](../results/comparison_20261101T220036_407013Z/A0_no_target_seed42)
- no_target, semilla 1042: [results/comparison_20261101T220354_300376Z/A0_no_target_seed1042](../results/comparison_20261101T220354_300376Z/A0_no_target_seed1042)
- no_target, semilla 2042: [results/comparison_20261101T220420_336130Z/A0_no_target_seed2042](../results/comparison_20261101T220420_336130Z/A0_no_target_seed2042)
- no_target, semilla 3042: [results/comparison_20261101T220507_516940Z/A0_no_target_seed3042](../results/comparison_20261101T220507_516940Z/A0_no_target_seed3042)
- no_target, semilla 4042: [results/comparison_20261101T220609_838850Z/A0_no_target_seed4042](../results/comparison_20261101T220609_838850Z/A0_no_target_seed4042)
- no_target, semilla 5042: [results/comparison_20261101T220648_523949Z/A0_no_target_seed5042](../results/comparison_20261101T220648_523949Z/A0_no_target_seed5042)
- no_target, semilla 6042: [results/comparison_20261101T220727_859502Z/A0_no_target_seed6042](../results/comparison_20261101T220727_859502Z/A0_no_target_seed6042)
- no_target, semilla 7042: [results/comparison_20261101T220812_457694Z/A0_no_target_seed7042](../results/comparison_20261101T220812_457694Z/A0_no_target_seed7042)

## Próximo paso

Compartir estas carpetas con Fran y Agustín y reunir las corridas restantes: 8 sin replay de Fran, 8 sin exploración de Agustín y las 5 semillas del agente completo repartidas entre ambos. Para comparar, verificar que coincidan con el mismo código y protocolo. El estudio de 32 corridas aún no está completo en este registro.

La herramienta de agregación usa team_draft.json como referencia de compatibilidad; ese archivo sigue marcado como borrador. La elección de A0 por el usuario está registrada en protocols/reference_selection.json. No se realizaron tests estadísticos ni evaluación en las semillas reservadas.
