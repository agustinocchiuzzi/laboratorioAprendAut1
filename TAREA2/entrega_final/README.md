# Carpeta de entrega — Tarea 2

Preparada el 4/11/2026 a partir de la versión vigente de `TAREA2/`. Es una **candidata para revisión y consolidación final**, no una certificación de cumplimiento académico.

## Archivos para subir

En [`para_entregar/`](para_entregar/) están los dos entregables solicitados:

- [`LunarLander_DQN.ipynb`](para_entregar/LunarLander_DQN.ipynb): copia de `../entrega_borrador.ipynb`, con código, datos incorporados y salidas guardadas.
- [`informe.pdf`](para_entregar/informe.pdf): copia del informe compilado, de cinco páginas.

Las copias coinciden con las fuentes de trabajo. H02 restauró material docente en el notebook y sincronizó ambas copias; el agente, los datos científicos y el informe se conservan.

El notebook actual tiene validación estática H02 en [`revision/verificacion.json`](revision/verificacion.json). La ejecución integral anterior corresponde exclusivamente al hash histórico registrado en [`revision/verificacion_pre_H02.json`](revision/verificacion_pre_H02.json). La ejecución del nuevo hash en kernel nuevo queda pendiente para C01. Ver la correspondencia y adaptaciones en [`revision/seguimiento/H02.md`](revision/seguimiento/H02.md).

## Revisión académica

El prompt completo está en [`PROMPT_REVISION_ACADEMICA.md`](PROMPT_REVISION_ACADEMICA.md). Podés iniciar la revisión con esta instrucción:

```text
Leé laboratorioAprendAut1/TAREA2/entrega_final/PROMPT_REVISION_ACADEMICA.md
y ejecutá la revisión que describe sobre esa carpeta y la evidencia del repositorio.
Devolvé la evaluación completa en español, sin modificar la entrega.
```

Si el agente ya trabaja dentro de `TAREA2/`, usá la ruta `entrega_final/PROMPT_REVISION_ACADEMICA.md`.

Antes de cerrar la entrega, leer [`revision/CONSOLIDACION.md`](revision/CONSOLIDACION.md): contiene los pendientes concretos detectados al preparar el paquete, incluido el contraste entre las dos versiones de la consigna sobre IA. No sustituye la revisión del prompt.

## Material de apoyo

- `apoyo/informe_fuente/`: LaTeX y las dos figuras PDF que utiliza. Permite seguir editando y recompilar con la clase `IEEEtran` y los paquetes declarados en el documento.
- `apoyo/requirements.txt`: dependencias generales para ejecutar el notebook.
- `apoyo/requirements-analysis-lock.txt`: versiones del entorno de análisis, con dependencias específicas de macOS.
- `revision/`: ambas consignas, plantilla docente, manifiesto de copias y verificación técnica. Es material interno de revisión.

El notebook puede analizar los datos y evaluar la política incorporada sin acceder a `results/` ni descargar pesos. Su ejecución normal mantiene `REENTRENAR = False` y las demostraciones docentes desactivadas; genera curvas, evaluaciones y cortes, por lo que no es instantánea. El GIF es opcional (`RUN_GIF=False`). Para correrlo, usar Python 3.12 con las dependencias instaladas, seleccionar ese entorno como kernel y ejecutar las celdas en orden. Un entorno nuevo requiere instalar `apoyo/requirements.txt`; la prueba de preparación utilizó el entorno ya existente del proyecto.

Las evidencias extensas, los demás pesos y las fuentes históricas permanecen en `TAREA2/`. No se duplicó todo el repositorio dentro del paquete. H01 incorporó las ocho calibraciones A0–A3, el código y las salidas originales de replay histórico, las 15 corridas de exploración y los 640 registros auxiliares de la tabla IV. Ver [seguimiento/H01.md](revision/seguimiento/H01.md). El bloque nuevo incluye decodificación, hashes, tablas recalculadas y reproducción costosa desactivada. La fuente actual se mantiene manualmente en `entrega_borrador.ipynb`; no regenerarla con el generador histórico, que perdería H02/R01/R03/H01.

## Consolidación final

1. Ejecutar la revisión con el prompt y resolver los puntos obligatorios que confirme.
2. Confirmar qué consigna publicó el equipo docente y contrastar con ella la declaración real de uso de IA.
3. Incorporar las correcciones también en las fuentes de trabajo de `TAREA2/`, recompilar el informe cuando corresponda y actualizar las copias de `para_entregar/`. Evitar mantener dos versiones divergentes.
4. Si se modifica el código o los datos incorporados, volver a comprobar la ejecución en kernel nuevo. Si cambia el PDF, volver a revisar sus páginas. Actualizar `revision/MANIFIESTO.json` y el registro de verificación: los actuales corresponden a los hashes de esta preparación.
5. Abrir exactamente los archivos de `para_entregar/` y subir los entregables que pida la plataforma del curso. No subir por defecto toda `entrega_final/`, ya que incluye consignas históricas, el prompt y notas internas.

El script original `scripts/verify_delivery_draft.py` escribe sobre `entrega_borrador.ipynb`; esta preparación verificó una copia en memoria y conservó las salidas originales. No confundir esas operaciones al hacer nuevas comprobaciones.

H01: las celdas nuevas se ejecutaron en un temporal vacío sin abrir archivos ni usar red; además pasó el recorrido ordinario completo mediante Python en otro temporal, bloqueando lecturas del repositorio, red y procesos. Las salidas de ese recorrido no reemplazan las guardadas. C01 todavía debe validar el kernel de notebook y la entrega completa.
