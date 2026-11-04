# Prompts para corregir y cerrar la entrega de LunarLander

Fecha de preparación: 4 de noviembre de 2026.

Estos prompts convierten los hallazgos de la [guía de mejoras](GUIA_MEJORAS_Y_ESTADO_INFORME.md) en tareas concretas. Prepararlos no ejecuta ni da por resuelta ninguna corrección. Este archivo es material interno y permanece fuera de `para_entregar/`.

## Cómo usarlos

Copiá completo el bloque del hallazgo que quieras resolver en una conversación del agente que tenga acceso a este mismo repositorio. Cada bloque indica dónde leer el contexto y las reglas compartidas; no necesita que se pegue el historial de esta conversación.

**Ejecutalos de a uno:** varios modifican el mismo notebook o informe. No conviene lanzarlos en paralelo sobre los mismos archivos. Los números de celda y página de la guía corresponden a la versión revisada; después de reorganizar, hay que localizar los contenidos por nombres y texto.

Orden sugerido:

1. **H03:** comprobar la regla docente de IA. Si no puede confirmarse, dejar ese punto pendiente y continuar con el trabajo técnico independiente.
2. **H02 → H01:** ordenar la base del cuaderno e incorporar la evidencia experimental.
3. **R01 → R03:** resolver interpretación estadística y precisión de protocolos.
4. **R02 → R04 → R05 → R06:** cerrar nombres, instrucciones, figuras y redacción permitida.
5. **C01:** comprobación final, compilación, sincronización y estado de entrega.

Los prompts **O01–O05 son opcionales** y están fuera de esa secuencia. O01 y O02 preparan posibles trabajos adicionales sin consumir la evaluación reservada ni iniciar entrenamientos. Si se decide aplicar otro cambio opcional, hacerlo antes de C01.

| Prompt | Resultado esperado | Modelo recomendado | Razonamiento |
|---|---|---|---|
| [H03](#h03) | Regla de IA documentada o incertidumbre explícita | GPT-6.1 Sol | `high` |
| [H02](#h02) | Cuaderno con el contenido base conservado | GPT-6.1 Sol | `high` |
| [H01](#h01) | Experimentos del informe respaldados dentro del notebook | GPT-6.1 Sol | `high` |
| [R01](#r01) | Análisis estadístico coherente con el diseño | GPT-6 Astra | `high` |
| [R03](#r03) | Calibración y antecedentes descritos con su protocolo real | GPT-6 Astra | `high` |
| [R02](#r02) | Estado y nombres consistentes, sin inventar identidades | GPT-6.1 Sol | `medium` |
| [R04](#r04) | Dependencias explicadas dentro del entregable | GPT-6.1 Sol | `medium` |
| [R05](#r05) | Cortes con leyenda y escalas correctas | GPT-6.1 Terra | `high` |
| [R06](#r06) | Pasajes precisados dentro del alcance de IA permitido | GPT-6.1 Terra | `medium` |
| [O01](#o01) | Herramienta y protocolo para evaluación reservada, sin ejecutarla | GPT-6.1 Sol | `high` |
| [O02](#o02) | Diseño mínimo de experimentos adicionales, sin ejecutarlos | GPT-6 Astra | `high` |
| [O03](#o03) | Checkpoints originales incorporados y verificables | GPT-6.1 Sol | `high` |
| [O04](#o04) | Decisión razonada sobre análisis adicional de incertidumbre | GPT-6 Astra | `high` |
| [O05](#o05) | Limpieza acotada del código residual | GPT-6.1 Sol | `medium` |
| [C01](#c01) | Validación de los archivos finales y estado verificable | GPT-6 Astra | `high` |

## Selección del modelo y del razonamiento

Recomendación preparada el **4 de noviembre de 2026**, con los modelos y niveles comprobados en el catálogo de T3. La asignación responde al tipo de tarea; no es el resultado de una evaluación comparativa de estos prompts.

- **GPT-6.1 Sol** (`gpt-6.1-sol`): código, datos incorporados, dependencias y trazabilidad. Usar `high` cuando hay que coordinar varias fuentes o preservar interfaces; `medium` para correcciones pequeñas.
- **GPT-6 Astra** (`gpt-6-astra`): interpretación estadística, límites de los experimentos y revisión final. Usar `high` en R01, R03, O02, O04 y C01, donde importa sostener conclusiones frente a evidencia incompleta o heterogénea.
- **GPT-6.1 Terra** (`gpt-6.1-terra`): revisión visual y cambios acotados de presentación. Usar `high` para R05, que combina figuras con escalas y código; `medium` para la redacción puntual de R06.

Esta distribución es un criterio práctico basado en la [guía oficial de selección de modelos](https://developers.openai.com/api/docs/guides/model-selection). Los niveles sugeridos siguen la orientación de usar razonamiento moderado o alto para diagnóstico y código; reservar niveles superiores para problemas cuya evaluación muestre una mejora real, según la [guía de configuración y verificación](https://developers.openai.com/api/docs/guides/deployment-checklist). No hace falta activar `max` o `ultra` en todos los prompts.

**Seleccioná el modelo y el razonamiento en T3 antes de pegar cada bloque.** Mencionar un modelo dentro del texto no cambia automáticamente el que está ejecutando la conversación. No se ha iniciado ninguna tarea ni cambiado ningún modelo al preparar esta tabla.

Si preferís mantener un único modelo, **GPT-6 Astra con `high`** es una alternativa razonable para toda la secuencia; no implica que vaya a ser la opción más rápida o económica. Mantené el orden de ejecución y leé los registros de seguimiento entre tareas.

La elección de modelo no vuelve obligatorios O01–O05, no autoriza entrenamientos y no resuelve la falta de una consigna oficial en H03 ni de confirmación de nombres en R02. Esos puntos se resuelven con evidencia o información del equipo.

## Contrato común

Todos los prompts remiten a estas reglas. Aplicarlas al ejecutar una tarea, no al leer este documento como material de referencia.

### Localización y alcance

- La raíz relevante es `laboratorioAprendAut1/TAREA2/`, relativa al espacio de trabajo; si el directorio actual ya es `TAREA2/`, usarlo directamente. No confundir con cuadernos de la raíz externa ni con `TAREA 1/`.
- Leer primero las instrucciones aplicables del repositorio, la guía y el hallazgo solicitado. Comprobar el estado real: un pendiente de la guía puede haberse resuelto desde su creación.
- Los hallazgos son conclusiones de una revisión, no una fuente infalible. Contrastarlos con artefactos y datos; si la evidencia actual contradice uno, explicarlo y evitar una modificación innecesaria.
- Resolver solo la tarea seleccionada. No iniciar subagentes, entrenamientos, descargas de datos o pesos, instalaciones, envíos de mensajes, publicaciones, commits ni push como efecto lateral.

### Fuentes, copias y conservación

- Inspeccionar el estado de Git, los hashes y las diferencias relevantes antes de editar. Preservar cambios existentes del equipo.
- Fuentes de trabajo registradas: `entrega_borrador.ipynb` e `informe/informe.tex`. Copias candidatas: `entrega_final/para_entregar/` y `entrega_final/apoyo/informe_fuente/`.
- Si las copias siguen coincidiendo, usar las fuentes de trabajo y sincronizar las copias afectadas. Si divergen, revisar qué cambió y conservar las modificaciones válidas; no sobrescribir una versión con otra por la fecha del nombre o por comodidad.
- Inspeccionar los generadores antes de usarlos, en particular `scripts/build_delivery_draft.py`: una regeneración puede borrar correcciones manuales. Mantener coherente la vía de construcción o documentar la fuente elegida para evitar que se pierdan los cambios.
- Conservar los resultados científicos originales: `results/`, pesos, CSV, configuraciones, manifiestos de selección y evaluaciones originales. Tampoco modificar `lander.ipynb` para reescribir el código histórico usado en las corridas. Las notas nuevas no deben falsear protocolos ni decisiones anteriores.
- Se puede incorporar una copia sin pérdida de datos al notebook y crear salidas de revisión separadas. No reemplazar resultados originales por reevaluaciones ni excluir corridas por su desempeño.

### Regla sobre IA y texto del informe

- H03 es una condición documental concreta, surgida de la discrepancia entre las dos consignas de la p. 2; no una sospecha basada en el estilo.
- Mientras la regla vigente no esté confirmada, realizar código, empaquetado, recálculos y diagnósticos técnicos. Para cambios sustantivos de prosa del informe, dejar indicaciones precisas para los autores en `entrega_final/revision/seguimiento/`, sin escribir una nueva redacción entregable ni insertarla en el LaTeX.
- Si hay respaldo verificable de que el alcance de asistencia solicitado está permitido, aplicar únicamente las ediciones autorizadas. Si solo se permite corrección gramatical y de estilo, mantener ese alcance y dejar las decisiones y nueva argumentación a los autores.
- No ocultar ni falsear la declaración de asistencia. No deducir permiso de la omisión de una frase en un PDF local. No hacer que un acuerdo o una vivencia del equipo parezca documentado si no lo está.

### Verificación y registro

- Inspeccionar el código antes de ejecutarlo. Trabajar con copias temporales para pruebas que escriben archivos. No activar entrenamientos, instalaciones automáticas, descargas ni interfaces interactivas durante una ejecución de comprobación.
- Hacer las comprobaciones breves necesarias para el cambio. Reservar la ejecución integral final para C01, salvo que antes haga falta resolver un riesgo concreto. No ampliar pruebas sin motivo.
- Distinguir revisión estática, salidas guardadas y ejecución nueva. Una verificación anterior solo corresponde a los hashes que registra.
- Registrar en `entrega_final/revision/seguimiento/<ID>.md`: archivos cambiados, evidencia, comprobaciones realmente efectuadas, límites y estado de la tarea. Crear esa carpeta si hace falta. No marcar todo el proyecto como terminado al cerrar un único hallazgo.
- No reutilizar un `passed` anterior para hashes nuevos. Conservar la verificación previa como evidencia histórica e indicar qué falta volver a comprobar. C01 actualizará la verificación final.
- Si se modifica LaTeX sin compilar todavía, indicar expresamente que el PDF está pendiente de actualizar. No presentar una edición de la fuente como cambio ya visible en el PDF.
- Al cerrar, entregar un resumen breve con enlaces a archivos, alcance del cambio, verificación y pendientes. No prometer aprobación ni nota docente.

## Prompts principales

### H03

**Confirmar la regla docente de IA. Solo comprobación documental.**

Modelo recomendado: **GPT-6.1 Sol**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/ (o en el directorio actual si ya es TAREA2). Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md y el hallazgo H03 de GUIA_MEJORAS_Y_ESTADO_INFORME.md, en esa misma carpeta.

Resolvé hasta donde permita la evidencia la discrepancia sobre IA. No cambies el informe, el notebook ni ninguna consigna.

1. Leé las dos consignas de entrega_final/revision/, en especial Implementación, p. 2, y la declaración de IA del informe, p. 5.
2. Separá la evidencia local de una eventual actualización docente oficial. El commit de un alumno, una copia repetida o la omisión local de la prohibición no prueban autorización.
3. Buscá primero si el proyecto o los recursos conectados contienen una fuente oficial identificable. Si hay una URL oficial pertinente, comprobala. No atribuyas a este curso reglas encontradas para otro año o grupo.
4. Si no hay una fuente suficiente, dejá el estado «no verificable» y especificá exactamente qué enlace, archivo o aclaración docente falta. Podés solicitar ese dato, pero no contactar ni enviar mensajes al docente por tu cuenta.
5. Si hay confirmación oficial, registrá procedencia, fecha, texto pertinente y alcance: desarrollo de código, corrección de estilo, redacción u otros usos. Contrastalo con el uso realmente declarado, sin reducir ni maquillar la declaración.

Escribí entrega_final/revision/seguimiento/H03.md con: evidencia, conclusión condicional o confirmada, y qué acciones pueden continuar. Si falta la regla, el empaquetado y los análisis técnicos pueden avanzar; H03 queda abierto. No inventes una regularización ni una aprobación docente.

Entregá el enlace al registro y un estado inequívoco: resuelto conforme, requiere regularización, o no verificable. No generes nueva prosa para el informe.
```

### H02

**Conservar la plantilla y resolver la estructura del notebook.**

Modelo recomendado: **GPT-6.1 Sol**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Si ya estás dentro, usá esa raíz. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md y el hallazgo H02 de la guía ubicada allí.

Corregí la correspondencia del notebook de entrega con entrega_final/revision/course_template.ipynb. Hacé cambios concretos y acotados; no cambies el algoritmo ni sus resultados.

1. Compará plantilla y candidato por secciones, clases, firmas y funciones, no solo por números de celda. Identificá qué está conservado, reorganizado u omitido.
2. Reintegrá el material suministrado que falta, incluido jugar_lunar_lander, la presentación del entorno y las demostraciones omitidas. Conservá las firmas y la lógica de entrenamiento, evaluación y visualización suministradas.
3. Ubicá claramente los tres bloques solicitados: AgenteDQN, cortes de política y ablación. Conservá las extensiones útiles, los datos incorporados, el checkpoint representativo y las salidas relevantes.
4. Conservá el código de las demostraciones, pero evitá que «Ejecutar todo» lance juego interactivo, instalaciones, descargas o entrenamientos. Usá activaciones explícitas deshabilitadas por defecto y explicá las adaptaciones de ejecución. No elimines el código necesario para reproducir el entrenamiento.
5. No conviertas una diferencia de numeración en motivo para reescribir funciones correctas. Comprobá que el núcleo de AgenteDQN y los datos científicos no cambiaron.

Validá el esquema del notebook y compará las interfaces o árboles sintácticos de las funciones suministradas. Verificá estáticamente los interruptores de ejecución. Hacé solo las comprobaciones adicionales breves necesarias.

Registrá en entrega_final/revision/seguimiento/H02.md una correspondencia plantilla → sección/celda final, las adaptaciones hechas y lo que verificaste. Sincronizá las copias afectadas según el contrato común. No presentes una ejecución integral anterior como validación del nuevo hash.
```

### H01

**Incorporar la evidencia experimental al notebook sin reentrenar.**

Modelo recomendado: **GPT-6.1 Sol**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Si ya estás dentro, usá esa raíz. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md y H01 de la guía ubicada allí. Aplicá esta tarea sobre la estructura actual, preservando H02 si ya fue resuelto.

Hacé que el notebook entregable incluya código y resultados de los experimentos que presenta el informe. El estudio final ya tiene datos incorporados: no lo dupliques ni lo reemplaces.

Incorporá, con procedencia y separación explícita:
- Las ocho corridas históricas A0–A3 de calibración, semillas 42 y 1042: configuraciones, presupuesto, código aplicable y datos originales de entrenamiento/evaluación. Localizá los CSV del 27 y 29 de octubre; no confundas el A0 histórico con el completo seleccionado después.
- La comparación histórica de capacidad de replay y el prototipo sin replay. Recuperá de Git el notebook pertinente siguiendo REFERENCIAS_LIMPIEZA.md. Conservar código y salidas guardadas es válido; no inventes retornos individuales que no se conservaron ni atribuyas esos resultados al agente final.
- La comparación histórica de exploración presentada en el informe, usando los datos de AgustinAgente/resultados/exploracion/ y su código histórico real.
- Los 640 registros auxiliares de analysis/team_study/policy_all/metricas_politica.json y su agregación para reproducir la tabla IV. Conservá esta tabla: hay evidencia disponible para incorporarla.

Mostrá tablas y resultados legibles; si usás datos comprimidos, incluí el decodificador, la procedencia y comprobaciones de integridad. Los códigos históricos deben quedar aislados para que no sobrescriban AgenteDQN, RUNS, la política final ni otras variables del estudio. Conservá todos los entrenamientos desactivados por defecto.

Recalculá desde datos individuales las cifras que lo permiten. Etiquetá el replay histórico como salida original guardada cuando no sea posible ese recálculo. No ejecutes los experimentos históricos para llenar el vacío ni actualices resultados para hacerlos coincidir con la tabla.

Probá las celdas nuevas en un directorio temporal sin resultados externos y verificá que el recorrido ordinario no lea rutas del repositorio ni use red. Es aceptable que la reproducción costosa sea opcional, pero el código y los resultados presentados deben estar dentro del notebook.

Sincronizá las copias afectadas y registrá en entrega_final/revision/seguimiento/H01.md un mapa «afirmación/tabla del informe → código y evidencia incorporados», recálculos efectuados y límites restantes. No modifiques la prosa del informe para eludir la autocontención.
```

### R01

**Ajustar la interpretación estadística al diseño experimental.**

Modelo recomendado: **GPT-6 Astra**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md, R01 de la guía y el estado de H03 si existe.

Resolvé la coherencia entre diseño, hipótesis y conclusión estadística. No reentrenes ni selecciones el test por el valor p que produce.

1. Reconstruí las ocho medias por condición desde los retornos individuales incorporados u originales, emparejando por semilla de entrenamiento. Los episodios de evaluación no son réplicas independientes de entrenamiento.
2. Comprobá qué aleatoriedad se comparte entre variantes y qué diferencias de plataforma persisten. No confundas trayectorias distintas con prueba de independencia ni la coincidencia nominal de semillas con un emparejamiento causal perfecto.
3. Recalculá MW bilateral exacto, Wilcoxon emparejado y Holm para las tres comparaciones de cada familia. Comprobá ceros y empates pertinentes. Documentá hipótesis y supuestos de cada método, incluyendo simetría de diferencias para Wilcoxon.
4. Mantené la trazabilidad de que MW fue el análisis originalmente propuesto. Proponé y documentá la interpretación que respeta los bloques por semilla, dejando MW como complementario condicionado. No reescribas retrospectivamente el protocolo como si la decisión nueva hubiera sido previa.
5. No agregues una batería de pruebas. Si hay una razón concreta para usar signos en lugar de asumir simetría, explicá antes qué hipótesis cambia y aplicá la misma familia completa de contrastes. No uses solo un p favorable sin ajustar.
6. Conservá efectos descriptivos, incertidumbre y límites de plataforma. Ausencia de significación no prueba equivalencia, y MW no es exclusivamente un test de medias.

Aplicá los ajustes de código, etiquetas técnicas y tablas que correspondan. Para la argumentación del informe, respetá H03: con autorización verificable hacé una edición mínima y coherente de resumen, VI y conclusiones; si falta o no permite nueva prosa, dejá indicaciones precisas para los autores en entrega_final/revision/seguimiento/R01.md, sin insertar nueva redacción.

Registrá datos usados, números recalculados, decisión metodológica y comprobaciones. Si algo difiere de la revisión previa, investigalo: no fuerces los resultados esperados. No marques R01 totalmente cerrado si la parte pertinente del informe sigue sin resolver.
```

### R03

**Precisar las diferencias de protocolo de la calibración.**

Modelo recomendado: **GPT-6 Astra**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md, R03 de la guía y el estado de H03 si existe.

Auditá y corregí la descripción de los protocolos de calibración, sin repetir experimentos ni cambiar resultados.

1. Localizá las ocho corridas exactas que sustentan la tabla II y compará sus configuraciones. Verificá la diferencia registrada de macOS para semilla 42 entre A0/A1 y A2/A3, y lo que ocurre para 1042.
2. Distinguí «solo cambia este hiperparámetro configurado» de «todos los factores de ejecución están controlados». No atribuyas una diferencia de desempeño al sistema operativo sin evidencia causal.
3. Inspeccioná el código histórico de replay: verificá cuándo empieza a aprender y con qué frecuencia, tanto con replay como sin él. Contrastalo con la ablación final, que usa inicio en 1000 pasos y frecuencia de cuatro.
4. Conservá separados el ensayo de capacidad 100k/10k, el prototipo histórico sin replay y la ablación integrada. No rebautices el ensayo de capacidad como ensayo de tamaños de minilote.
5. Revisá que las celdas de calibración incorporadas mediante H01 tengan la misma precisión documental.

Corregí metadatos descriptivos o notas técnicas del notebook si hace falta. Para el texto de la sección III del informe, aplicá solo la edición sustantiva permitida por la regla confirmada en H03. Si no está confirmada o no lo permite, dejá una lista de correcciones fundamentadas para los autores en entrega_final/revision/seguimiento/R03.md, sin escribir prosa entregable.

La tarea queda resuelta cuando las descripciones corresponden a los protocolos reales y los límites están expresados donde se hacen las comparaciones. No propongas nuevas corridas como primera solución: el informe no necesita demostrar optimalidad general.
```

### R02

**Actualizar estados y aclarar nombres sin asumir identidades.**

Modelo recomendado: **GPT-6.1 Sol**. Razonamiento: **`medium`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md y R02 de la guía.

Actualizá los textos de estado y la identificación del equipo en el notebook y las fuentes relacionadas. Revisá los artefactos actuales antes de repetir pendientes del 2/11.

1. Comprobá que el informe existe, su versión y páginas; distinguí PDF ya compilado de fuente que haya cambiado después.
2. Quitá o actualizá los pendientes obsoletos de redacción/compilación cuando la evidencia realmente los cierre. No declares «listo para entregar» si H01, H02, H03 o la validación final siguen abiertos.
3. Conservá como no realizada la evaluación reservada, salvo evidencia explícita nueva. No conviertas un acuerdo metodológico pendiente en acuerdo confirmado.
4. Buscá evidencia inequívoca sobre la relación entre «Fran» y «Juan Duarte». No infieras que son la misma persona, ni cambies la atribución de las corridas por coincidencias parciales. Si no hay respaldo, pedí esa aclaración concreta al equipo y continuá con los textos independientes.
5. Una vez confirmados los nombres, unificá la identificación formal y explicá los alias necesarios para relacionarla con la procedencia de los experimentos.

Preservá los registros históricos; no cambies el autor original de una corrida ni la declaración de IA para simplificar la presentación. Evitá nuevas modificaciones de prosa ajenas a esta tarea.

Sincronizá las copias afectadas y registrá en entrega_final/revision/seguimiento/R02.md qué pendientes se cerraron, qué nombres se confirmaron y qué sigue sin evidencia. No insertes marcadores de autor pendientes en el PDF destinado a entregar.
```

### R04

**Hacer autocontenidas las instrucciones de dependencias.**

Modelo recomendado: **GPT-6.1 Sol**. Razonamiento: **`medium`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md y R04 de la guía.

Corregí las instrucciones del notebook para que un docente que reciba solamente el .ipynb y el PDF sepa preparar el entorno.

1. Inspeccioná los imports y el recorrido ordinario real del notebook, además de apoyo/requirements.txt, apoyo/requirements-analysis-lock.txt y verificacion.json. Revisá también las dependencias de las celdas nuevas.
2. Incorporá dentro del notebook una lista suficiente de paquetes y una instrucción de instalación explícita para el kernel adecuado. No remitas exclusivamente a un archivo que no se subirá.
3. Separá dependencias del análisis ordinario, visualización y reproducción opcional cuando esa distinción sea útil. No ejecutes instalaciones ni agregues dependencias nuevas por comodidad.
4. Indicá las versiones y plataforma que realmente se verificaron. No presentes el lock macOS, ni paquetes como appnope, como requisitos universales de Windows/Linux.
5. Conservá instalación y entrenamientos fuera de la ejecución ordinaria. Si faltan paquetes en el entorno actual, informá qué falta en lugar de descargarlos automáticamente.

Verificá que los imports ordinarios estén cubiertos y que las instrucciones no dependan de rutas privadas, archivos ausentes ni del repositorio. Si no se hizo una instalación limpia, no afirmes que se comprobó portabilidad universal.

Aplicá la edición, sincronizá la copia entregable y registrá en entrega_final/revision/seguimiento/R04.md qué entorno quedó documentado y qué se verificó. No cambies la versión científica del experimento ni reevalues políticas por modificar documentación de instalación.
```

### R05

**Completar leyenda y escalas de los cortes de política.**

Modelo recomendado: **GPT-6.1 Terra**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md, R05 de la guía y el estado de H03 si existe.

Corregí la presentación de los cortes manteniendo los mismos pesos, estados de referencia, rangos, resolución y resultados.

1. Localizá el corte representativo y la función leyenda_acciones. Añadí la leyenda que falta, con correspondencia exacta entre colores y las cuatro acciones. No dupliques leyendas que ya estén correctas.
2. Consultá la implementación de LunarLander de la versión usada para identificar cómo se construyen las observaciones. Verificá las escalas de posición, velocidad y velocidad angular, y el ángulo en radianes. No inventes unidades físicas directas.
3. Aclarar la escala no requiere transformar los valores del corte: preferí etiquetar correctamente las observaciones existentes y señalar que no se añadió normalización propia.
4. Conservá semilla/checkpoint, valores de las seis coordenadas fijas y la distinción entre puntos visitados proyectados y estados sintéticos. Los valores Q no son probabilidades de éxito.
5. Regenerá solo las figuras afectadas, desde datos o pesos disponibles, en una salida temporal antes de reemplazar las figuras de presentación. No sobreescribas archivos históricos de análisis ni recalcules experimentos que no hagan falta.

Inspeccioná visualmente las figuras nuevas: leyenda completa, etiquetas legibles y ausencia de recortes. Si modificás el PDF, compilalo y examiná las páginas afectadas y el total de páginas. Respetá H03 para cualquier nueva prosa del informe; si no está permitida o confirmada, dejá la indicación a los autores en entrega_final/revision/seguimiento/R05.md.

Sincronizá las copias de presentación pertinentes y registrá qué cambió y cómo comprobaste que los valores representados permanecen iguales.
```

### R06

**Precisar el resumen, los pilotos y las conclusiones.**

Modelo recomendado: **GPT-6.1 Terra**. Razonamiento: **`medium`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md, R06 de la guía y el resultado de H03 si existe. Leé los pasajes en el PDF actual y su contexto, no solo el resumen de la revisión.

Atendé únicamente estos tres puntos:

1. Resumen: separar el agregado original de ocho políticas completas del resultado de la comparación local con el agente aleatorio. Verificá números, denominadores y plataforma. No sugerir que 40,44 % y 0 % provienen de un único experimento homogéneo; la comparación local guardada es 31,5 % frente a 0 %.
2. Calibración: precisar qué se comprobó realmente mediante los pilotos. Buscá registros contemporáneos. No atribuyas a los autores las pruebas ejecutadas por el revisor ni inventes errores, vivencias o motivaciones. Si falta evidencia, pedí una aclaración concreta o señalá la afirmación que no puede sostenerse; continuá con los otros dos puntos.
3. Conclusiones: distinguir para qué servirían más semillas, plataformas comparables y una evaluación reservada. No presentar esas extensiones como exigencias de la consigna ni como soluciones intercambiables para todos los límites.

Si H03 acredita que estas ediciones sustantivas con asistencia están permitidas, aplicá cambios mínimos sobre los pasajes; no reescribas todo el informe. Si falta confirmación o la regla solo permite estilo/gramática, prepará en entrega_final/revision/seguimiento/R06.md un diagnóstico y una lista de contenidos que los autores deben aclarar por sí mismos, sin generar prosa nueva para insertar.

Conservá las limitaciones bien explicadas, la declaración veraz de IA y la voz del equipo. No uses «parecer humano» como objetivo ni elimines precisión técnica para lograrlo.

Si editás la fuente, sincronizá las copias correspondientes e indicá si el PDF fue o no recompilado. Al compilar, mantené el límite de cinco páginas incluyendo referencias. Registrá con exactitud qué parte quedó aplicada y qué parte requiere intervención de los autores.
```

## Prompts opcionales

Estos trabajos no son necesarios para cerrar los requisitos actuales. No ejecutarlos automáticamente como parte de H01–R06.

### O01

**Preparar una evaluación reservada sin ejecutarla ni consumir sus escenarios.**

Modelo recomendado: **GPT-6.1 Sol**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md y O01 de la guía.

Prepará el protocolo y, si falta, una herramienta para reevaluar los 32 checkpoints finales en las semillas reservadas 30000–30199. Este pedido no autoriza a ejecutar esa evaluación ni a entrenar.

Identificá los checkpoints mediante el manifiesto de selección, comprobá hashes y fijá las decisiones antes de mirar resultados: ocho políticas por condición, evaluación greedy sin aprendizaje, 200 escenarios comunes, misma plataforma de evaluación y registro de versiones. Conservá retorno, longitud y motivo de finalización por episodio, identificando la política, el checkpoint y la semilla.

Usá una salida nueva claramente separada de los CSV originales. No reemplaces ni mezcles desarrollo, visualización y reservado. La herramienta debe requerir activación explícita y fallar de forma clara si falta un peso o su hash no coincide.

Podés hacer comprobaciones estáticas y una prueba mínima con escenarios de desarrollo, identificada como tal. No uses escenarios reservados en una prueba previa. Dejá el comando exacto y el alcance previsto para una autorización posterior del equipo, sin afirmar que la evaluación fue realizada.

Registrá en entrega_final/revision/seguimiento/O01.md qué decisiones quedan fijadas, qué falta ejecutar y cómo impedir que se ajuste el agente tras observar el reservado. No cambies las cifras ni la declaración de evaluación reservada del informe actual.
```

### O02

**Diseñar la extensión mínima para una pregunta experimental concreta. Sin entrenar.**

Modelo recomendado: **GPT-6 Astra**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md y O02 de la guía.

Prepará una propuesta mínima de experimentos adicionales que responda a un límite concreto del estudio, sin ejecutar entrenamientos ni modificar la entrega actual.

Separá tres objetivos distintos: medir variabilidad con más semillas, separar plataforma de componente, o comparar costo a igual número de interacciones. Determiná qué afirmación adicional permitiría cada uno y cuál no resolvería. Si no hay un objetivo priorizado por el equipo, presentá alternativas pequeñas y sus costos; no elijas arbitrariamente un estudio grande.

Para cada alternativa necesaria especificá condiciones, factores que cambian y se mantienen, semillas, presupuesto, elección del checkpoint, plataforma/versiones, evaluación, unidad experimental y análisis previsto. No confundas igualdad de episodios con igualdad de pasos o actualizaciones. Si proponés más semillas, no prometas una potencia que no calculaste con supuestos explícitos.

Usá tiempos históricos solo como estimaciones orientativas y declará su incertidumbre. No cambies protocolos pasados ni presentes el plan como prerregistrado antes de su creación.

Guardá el diseño en entrega_final/revision/seguimiento/O02.md. Terminá explicando por qué se puede entregar el trabajo actual sin ejecutarlo y qué decisión del equipo haría falta para emprenderlo. No lances procesos, descargas ni trabajos en segundo plano.
```

### O03

**Incorporar los pesos originales restantes sin reevaluar.**

Modelo recomendado: **GPT-6.1 Sol**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md y O03 de la guía.

Incorporá al notebook los checkpoints originales de las 32 políticas seleccionadas, usando el manifiesto del estudio. El objetivo es permitir reevaluación posterior sin entrenar; no ejecutar reevaluaciones en esta tarea.

Comprobá cada hash antes de copiar y asociá de forma inequívoca variante, semilla, configuración, ruta de procedencia y checksum. Reutilizá el checkpoint de semilla 42 que ya está incorporado en vez de duplicarlo. Si falta un archivo o no coincide, no lo sustituyas por otro ni regeneres pesos: registrá la discrepancia.

Usá una representación incorporada sin pérdida y un cargador explícito; preservá el recorrido ordinario breve. Verificá carga segura de state_dict, nombres y formas de tensores, arquitectura y correspondencia con los archivos originales. No cargues pickles arbitrarios ni active código histórico para reconstruir resultados.

Comprobá que el aumento de tamaño sea razonable y, si existe un límite docente de subida documentado, que se respete. Si el límite impide incluirlos, explicá el conflicto sin eliminar evidencia obligatoria ni cambiar a descargas externas silenciosas.

Sincronizá las copias afectadas y registrá en entrega_final/revision/seguimiento/O03.md número de checkpoints válidos, tamaño final y comprobaciones. No marques como reevaluadas políticas que solo cargaste.
```

### O04

**Decidir si hace falta un análisis adicional de incertidumbre.**

Modelo recomendado: **GPT-6 Astra**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md, R01 y O04 de la guía. Comprobá primero si R01 ya está resuelto.

Evaluá si una afirmación concreta del informe necesita una mejora adicional de incertidumbre. No agregues intervalos, bootstrap, potencia o tests por cumplir una lista.

Si el informe ya limita correctamente sus conclusiones y el análisis actual es suficiente, documentá por qué conviene conservarlo. Si hay un beneficio claro, elegí una sola mejora vinculada a un estimando explícito y explicá sus supuestos antes de calcularla.

Cualquier remuestreo debe respetar la política entrenada como unidad y el emparejamiento por semilla que corresponda. No trates los 200 episodios por política como 200 entrenamientos independientes. Conservá el tratamiento de comparaciones múltiples cuando corresponda y no selecciones análisis por significación.

Usá únicamente datos guardados y cálculos breves. No reentrenes, no presentes ausencia de significación como equivalencia y no afirmes que más técnicas corrigen la confusión de plataforma o la selección de escenarios.

Registrá en entrega_final/revision/seguimiento/O04.md la decisión y, si calculaste algo, datos, código reproducible, resultado y límites. Para incorporarlo al informe respetá H03; no amplíes el PDF más allá de cinco páginas ni elimines contenido obligatorio para agregar una mejora marginal.
```

### O05

**Eliminar el código residual que no pertenece al recorrido del notebook.**

Modelo recomendado: **GPT-6.1 Sol**. Razonamiento: **`medium`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md y O05 de la guía.

Revisá el main() heredado de la sección estadística del notebook, originalmente en la celda 29, que referencia nombres como DEFAULT_PROTOCOL. Confirmá primero si sigue existiendo, si es invocado y si otra celda depende de él.

Si es código residual no utilizado, retiralo del notebook junto con imports que hayan quedado realmente sin uso. Conservá holm, comparar_una y cualquier función necesaria para el análisis. No borres la interfaz válida del script independiente del repositorio ni aproveches para refactorizar el agente.

Si la función ahora tiene un uso real, no la elimines a ciegas: resolvé ese uso con el menor cambio compatible con el notebook autocontenido. No agregues dependencias externas para mantener una función innecesaria.

Verificá sintaxis y ejecución breve de las celdas estadísticas con los datos incorporados. Los cálculos y resultados deben quedar iguales. No hace falta crear una nueva batería de tests para esta limpieza.

Sincronizá el candidato y registrá en entrega_final/revision/seguimiento/O05.md qué se retiró o reparó y cómo se comprobó que no cambió el análisis.
```

## Cierre final

### C01

**Validar la versión efectivamente corregida y emitir un estado final.**

Modelo recomendado: **GPT-6 Astra**. Razonamiento: **`high`**.

```text
Trabajá en laboratorioAprendAut1/TAREA2/. Leé el «Contrato común» de entrega_final/revision/PROMPTS_CORRECCIONES_ENTREGA.md, la guía y los registros disponibles de entrega_final/revision/seguimiento/. No supongas que un prompt fue aplicado porque existe en este archivo.

Realizá el cierre técnico de la entrega sin reentrenar, descargar datos/pesos ni publicarla. Revisá el estado real y verificá los archivos exactos que terminarán en entrega_final/para_entregar/.

1. Inventariá fuentes y copias, comprobá diferencias y preservá los cambios del equipo. Confirmá qué hallazgos se resolvieron, cuáles tienen una alternativa justificada y cuáles siguen abiertos. Los campos de seguimiento son evidencia auxiliar, no prueba suficiente por sí solos.
2. Contrastá el notebook con la plantilla; verificá los tres bloques, funciones e interfaces, dependencias y el código/resultados de los experimentos presentados. Comprobá que no haya rutas externas activas en el recorrido ordinario.
3. Inspeccioná todos los efectos de «Ejecutar todo» antes de lanzarlo. Confirmá entrenamiento, instalación, juego interactivo y evaluación reservada desactivados. Usá el entorno instalado; si falta una dependencia, registrá el bloqueo concreto sin instalarla automáticamente.
4. Ejecutá una copia en kernel nuevo desde un directorio temporal inicialmente sin archivos de resultados. Registrá tiempo, versiones, celdas, errores y archivos creados. No eludas una restricción del entorno si impide iniciar el kernel: indicá qué no se pudo verificar.
5. Conservá inalterados los datos experimentales incorporados y originales. Distinguí las salidas nuevas de reevaluación local de las originales. Si cambia una salida por plataforma, investigá y etiquetá la procedencia; no reescribas el estudio para que coincida con la reevaluación.
6. Recalculá las tablas y los contrastes relevantes desde los datos disponibles, aprovechando comprobaciones previas de hashes idénticos cuando sea adecuado. Verificá denominadores, semillas, umbral de éxito, ddof, familias de tests y diferencia entre agregado y política individual.
7. Compilá el informe desde la fuente final con las herramientas disponibles y examiná visualmente todas sus páginas, figuras y tablas. Comprobá máximo cinco páginas totales, referencias incluidas. No elimines contenido obligatorio ni achiques el texto hasta hacerlo ilegible para cumplir el límite. Si hacen falta cambios sustantivos de prosa, respetá H03 y dejá la tarea específica a los autores cuando corresponda.
8. Comprobá correspondencia entre resumen, cuerpo, tablas, figuras, conclusiones, nombres y declaración de IA. No marques la regla docente como resuelta sin respaldo oficial.
9. Sincronizá notebook, PDF y fuentes afectadas. Archivá la verificación anterior antes de reemplazarla. Generá hashes y una verificación nueva vinculados a los bytes finales; no vuelvas a editar esos archivos después sin invalidar o repetir las comprobaciones afectadas. Actualizá MANIFIESTO.json sin inventar procedencia.

Escribí entrega_final/revision/seguimiento/ESTADO_CIERRE.md con:
- Estado por hallazgo y evidencia verificable.
- Veredicto técnico del informe y veredicto del paquete, separados del estado de IA.
- Qué se ejecutó ahora, qué se verificó estáticamente y qué se tomó de registros anteriores.
- Archivos finales, hashes, páginas, versiones y límites pendientes.
- Checklist exacto de lo que se subirá: notebook y PDF; otros archivos solo si la consigna/plataforma los pide.

Repará los problemas menores dentro del alcance autorizado que detectes; no te limites a enumerarlos. Si queda un impedimento real, identificá el cambio mínimo pendiente y no declares «listo para entregar». La evaluación reservada, más semillas y las otras mejoras opcionales no son bloqueos automáticos.

Entregá los enlaces a los dos candidatos y al estado final. No hagas commits, push, envío de mensajes ni subida a la plataforma docente. No prometas una nota o aprobación.
```

## Estado de este archivo

Contiene nueve prompts principales, cinco opcionales y uno de cierre. Su preparación no modifica los entregables ni resuelve los hallazgos. El estado real se registra al ejecutar cada tarea y se consolida mediante C01.
