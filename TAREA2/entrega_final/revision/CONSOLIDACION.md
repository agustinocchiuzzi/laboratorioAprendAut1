# Estado de preparación y asuntos para revisar

Fecha: 4/11/2026. Este documento registra comprobaciones de empaquetado y observaciones preliminares. **No es la evaluación académica solicitada al futuro revisor.**

## Preparación realizada

- Se tomó como fuente `TAREA2/`, identificada como versión de trabajo vigente por la documentación del proyecto.
- Se copiaron el notebook integrado, el informe PDF, su fuente LaTeX y las dos figuras referenciadas. Se agregaron dependencias, plantilla docente y ambas versiones de la consigna como apoyo.
- Las copias son idénticas a sus fuentes; tamaños y SHA-256 están en `MANIFIESTO.json`.
- El PDF tiene cinco páginas. Se renderizaron y examinaron las cinco: no se observaron recortes ni superposiciones evidentes. Esta comprobación de maquetación no valida por sí sola las cifras ni las conclusiones.
- El notebook contiene 31 celdas, 18 de código, con contadores de ejecución consecutivos y sin salidas de error guardadas. Las 18 celdas de código se ejecutaron correctamente en un kernel nuevo desde un directorio inicialmente vacío, en 37,72 segundos. Se mantuvo el reentrenamiento desactivado y se conservaron intactos los bytes del notebook entregable. El resultado, las salidas de texto y las versiones utilizadas están en `verificacion.json`. La prueba utilizó el entorno ya instalado, no una instalación nueva de dependencias.
- No se reentrenaron políticas, no se reemplazaron resultados científicos y no se redactó ni modificó el informe en esta preparación.

## Cuestiones que deben resolverse o evaluarse

### 1. Confirmar la consigna docente vigente sobre IA

`consigna_historica_60af2e4.pdf`, recuperada del commit `60af2e4`, contiene en la página 2: «No está permitido utilizar herramientas de IA para la redacción del informe». También permite la corrección gramatical y de estilo.

La copia local actual no contiene esa prohibición. El historial registra una actualización local de la letra en `44f19b7`, pero eso no acredita por sí solo un cambio aprobado por los docentes. Las copias de la raíz, de `TAREA2/` y de `output/pdf/` coinciden entre sí; no son confirmaciones independientes de procedencia oficial.

El informe, página 5, declara asistencia de IA para redactar y editar, incluida la escritura de la prosa. Por tanto, el punto debe resolverse contrastando con la consigna oficial publicada. Si sigue vigente la prohibición histórica, esa declaración implica un problema de cumplimiento que no se resuelve ocultándola ni cambiando superficialmente el estilo. Si hubo una modificación docente válida, conservar la evidencia y evaluar con esa versión. Aquí no se afirma cuál es la regla vigente.

### 2. Actualizar o aclarar textos de cierre del notebook

Las celdas 1 y 31 todavía lo presentan como borrador del 2/11 e indican que falta redactar y compilar el informe. Ya existe un PDF de cinco páginas. La celda 16 también conserva una confirmación pendiente del protocolo/test. Hay que revisar cuáles de esos pendientes siguen abiertos antes de actualizar los textos; no marcarlos todos como resueltos automáticamente.

La portada del informe identifica a Thiago Rivas, Juan Duarte y Agustin Occhiuzzi, mientras que el notebook usa Thiago, Fran y Agustín. Corresponde aclarar si «Fran» es la misma persona o una atribución distinta y usar una identificación coherente. No se infiere un error de autoría solo por esa diferencia.

### 3. Comprobar el ajuste a la plantilla docente

La consigna pide una copia del cuaderno provisto. El candidato integrado reorganiza secciones: por ejemplo, `AgenteDQN` aparece en la sección 4, mientras que la consigna remite a la sección 6 de la plantilla. El revisor debe comparar código, interfaces y contenido con `course_template.ipynb` y determinar si corresponde restaurar la estructura o aclarar la correspondencia. El cambio de numeración por sí solo no demuestra una falla del algoritmo.

### 4. Revisar la cobertura de los experimentos en el paquete

El notebook incorpora los registros de las 32 políticas del estudio final y un checkpoint representativo. El informe también describe comparaciones de calibración e incluye una tabla de finales de episodio proveniente de `analysis/team_study/policy_all/`. El revisor debe comprobar cuánto de ese código y de esos resultados está efectivamente en el notebook que se subirá. Si falta evidencia exigida, incorporarla o delimitar las afirmaciones; la existencia de archivos externos no basta para acreditar autocontención.

### 5. Distinguir límites metodológicos de nuevas obligaciones

La evaluación reservada 30000–30199 se describe como no realizada. La consigna local no la exige explícitamente. El informe también declara diferencias de plataforma y un número acotado de semillas. El revisor debe juzgar si las conclusiones respetan esos límites; no exigir automáticamente nuevos entrenamientos o una evaluación reservada solo porque serían deseables.

## Cierre pendiente

- [ ] Ejecutar la revisión académica completa con el prompt.
- [ ] Confirmar la versión oficial de la consigna y la coherencia de la declaración de IA.
- [ ] Resolver los hallazgos obligatorios y documentar qué mejoras opcionales se dejan fuera.
- [ ] Actualizar textos y copias cuando corresponda; mantener manifiesto y verificación alineados con los archivos finales.
- [ ] Abrir el notebook y el PDF definitivos de `para_entregar/` antes de subirlos.

No se produjo un ZIP ni se efectuó ninguna entrega en una plataforma externa.
