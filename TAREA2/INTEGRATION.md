# Integración de Tarea 2 — 30 de octubre de 2026

Rama: `integration/assignment-2`, creada desde `thiago` (`5246ac3`). Se hizo
`git fetch --all --prune` antes de revisar todas las ramas locales y remotas.
Los historiales B y C se integraron mediante merges, conservando sus archivos
originales. `main` continúa en `ea425a0`; Tarea 1 no recibe cambios.

## Ramas revisadas

| Rama remota | Commit | Contenido y decisión |
|---|---|---|
| `thiago` | `5246ac3` | Base A: agente, comparación A0/A1, resultados y herramientas. |
| `AlgoritmoFran` | `b2cdb60` | Tarea 2: agente B, replay 100k/10k/sin replay, salidas y GIF. Integrada. |
| `AgustinAgente` | `0c593eb` | Tarea 2: agente C, exploración, 15 corridas, pesos, explicación y GIF. Integrada. |
| `main` | `ea425a0` | Ancestro de las tres contribuciones; incluye plantilla de Tarea 2. |
| `feature/initial-ml-pipeline` | `1dc7647` | Tarea 1, ya ancestro de la base. |
| `fix/naive-bayes-consistency` | `0ca5912` | Tarea 1, ya ancestro de la base. |
| `naive-bayes-integration` | `0b986f3` | Tarea 1, ya ancestro de la base. |
| `rama-arboles-informe` | `17a0eb8` | Cambios exclusivos de Tarea 1 (ID3/NB); excluidos. |
| `remove-ai-punctuation` | `998fcc3` | Tarea 1, ya ancestro de la base. |
| `revision-final-entrega` | `fbc7958` | Cambios exclusivos de la entrega de Tarea 1; excluidos. |

Las ramas locales `main` y `thiago` coinciden con sus remotas.
`origin/HEAD` es un alias de `origin/main`. Los hashes completos y los cambios
desde el ancestro común están en [branch_audit.json](integration/branch_audit.json).

## Agente y herramientas comunes

`lander.ipynb` mantiene la interfaz y las funciones de entrenamiento/evaluación
del curso. Todo lo necesario para el agente sigue dentro de `AgenteDQN`;
los pesos A existentes conservan los mismos nombres y formas.

| Aporte | Interfaz integrada | Semántica |
|---|---|---|
| A: arquitectura/optimización | `hidden_dims`, `lr`, A0–A3 | Conserva A2/A3 y la segunda semilla que estaban sin commit. |
| A: red objetivo | `use_target=False` | Bootstrap online sin gradiente por el objetivo. |
| B: replay | `replay_capacity`, `use_replay=False` | Sin replay: transición actual, batch efectivo 1, memoria vacía. |
| C: exploración | `epsilon_schedule` | `linear_episodes` (A), `linear_steps` y `exponential_episodes` (C). |
| C: quitar exploración | `use_exploration=False` | Epsilon cero desde el inicio; decisiones greedy. |
| C: target suave | `target_update="soft"`, `tau` | Copia ponderada después de cada actualización del optimizador. |

A0 se conserva como referencia operativa por compatibilidad. **No es una selección
científica del mejor agente ni un acuerdo del equipo.** Los nuevos esquemas usan
los demás parámetros de la configuración seleccionada; no reproducen automáticamente
el protocolo C. Su original conserva el buffer circular y el generador propio;
el agente común mantiene el replay y flujo aleatorio de A para preservar sus defaults.

En el prototipo B, quitar replay también cambiaba de actualizar cada cuatro pasos
a actualizar en todos los pasos. La versión común conserva `learning_starts` y
`train_frequency` en ambos modos. El batch efectivo cambia de 64 a 1; debe
declararse como parte de la definición de esa ablación.

Los exports ahora guardan switches, esquema de epsilon, batch efectivo, tipo de
actualización objetivo, revisión Git y SHA-256 del notebook. El CSV registra el
epsilon del esquema seleccionado. El agregador rechaza esquemas de exploración
o target incompatibles y sigue leyendo los resultados antiguos de A.

El runner admite cuatro variantes y overrides sin copiar código entre notebooks.
`--config` respeta `--mode`: indicar `comparison` para 300 episodios; `pilot`
ejecuta 20. `integration-pilot` prueba siete condiciones: full, tres remociones,
dos esquemas de C y target suave.

Se conservaron las herramientas, resultados y grabaciones locales de A junto con
los originales B/C. Los manifiestos antiguos de `preparation/` son instantáneas
anteriores; el [nuevo manifiesto](preparation/integrated_job_manifest.json) reconoce
los tres switches y mantiene el protocolo como borrador.

## Evidencia disponible y límites

| Evidencia | Entrenamiento | Evaluación | Resultado guardado |
|---|---|---|---|
| A0, seed 42 | 300 episodios | 200, seeds 10000–10199 | Media 167.22, 113/200 éxitos. |
| A0, seed 1042 | 300 episodios | 200, seeds 10000–10199 | Media 209.76, 146/200 éxitos. |
| B principal, seed 42 | 1200 episodios | 200, seeds 10000–10199 | Media 249.86, 193/200 éxitos (salida del notebook). |
| B replay 100k / 10k / sin replay | 500 episodios, seed 42 | 50, seeds 10000–10049 | Medias 247.7 / 223.6 / −87.7 (salida redondeada). |
| C principal, seed 42 | 1200 episodios | 200, seeds 10000–10199 | Media 217.99, 130/200 éxitos; pesos disponibles. |
| C lineal / exponencial / sin exploración | 600 episodios, seeds 0–4 | 100 por política, seeds 10000–10099 | Medias entre seeds 127.1 / 177.1 / 6.1; resultados individuales JSON. |

Fuentes A: `results/comparison_20261027T001236_958528Z/` y
`results/comparison_20261029T233234_746501Z/`. También se conservan A1/A2/A3
con seeds 42 y 1042, comparaciones controladas y cortes de política.
Fuentes B: secciones 8 y 8b de `LunarLander_DQN_B.ipynb`.
Fuentes C: secciones 8 y 11 de su notebook y `AgustinAgente/resultados/`.

No se combinan como un ranking controlado: cambian arquitectura, optimización,
warmup, target, exploración, presupuestos y cantidad de semillas. B no incluye
CSV de retornos ni checkpoint; sus números conservan la precisión publicada.
Los tests estadísticos de C pertenecen a su experimento, no al estudio de las tres
ablaciones integradas. Las semillas 10000–10199 ya se usaron para comparar
configuraciones y se consideran evaluación de desarrollo.

## Validación

- 14 tests: switches, warmup, frecuencia, transición actual, epsilon, evaluación
  sin efectos, target suave, máscaras terminales, planificación y agregación.
- Ejecución completa de `integration-pilot_results.ipynb` en un kernel nuevo,
  incluidas las comprobaciones DQN originales. Siete agentes desde cero, 20
  episodios cada uno y 10 evaluaciones greedy; todos hicieron actualizaciones
  reales. Se conserva también el baseline aleatorio.
- El piloto comprueba funcionamiento; no reemplaza el estudio multisemilla.
- Verificación por hash de los originales B/C y los resultados previos de A;
  la plantilla del curso y Tarea 1 se conservaron.
- A0 y target-off reprodujeron exactamente todas las filas de entrenamiento y
  evaluación, y todos los tensores de pesos, de los pilotos originales de A.
  El CLI seleccionado respetó el modo corto, y el agregador validó las dos
  políticas A0 históricas. [Registro de validación](integration/validation.json).

## Plan de cierre

1. Revisar A/B/C y fijar referencia, presupuesto, semillas y definición de replay-off
   en `protocols/team_draft.json`. Elegir test estadístico y corrección por las tres
   comparaciones antes del estudio.
2. Entrenar referencia y tres ablaciones desde cero con el mismo presupuesto y
   semillas. La propuesta actual contiene ocho semillas y 32 políticas. Guardar
   cada corrida con el runner común, sin reutilizar pesos entre condiciones.
3. Validar y agregar con `study_tools.py aggregate`. Comparar medias por política
   entrenada; los episodios de evaluación no sustituyen réplicas independientes.
   Implementar el test acordado y verificar sus supuestos.
4. Generar cortes y rollouts del modelo final con las herramientas existentes e
   incorporarlos al notebook autocontenido. Los cortes A actuales son preparatorios.
5. Completar las secciones 10 y 11 con los experimentos definitivos y redactar el
   informe IEEE. Usar las semillas reservadas 30000–30199 después de congelar las
   decisiones. Ejecutar y revisar la entrega completa.

La integración está hecha; este plan enumera el trabajo científico y de entrega
que los branches aún no habían completado.
