# Guía de mejoras antes de entregar y estado del informe

Fecha: 4 de noviembre de 2026.

Documento de trabajo del equipo, elaborado a partir de la revisión académica realizada en esta conversación. No forma parte del informe ni de los archivos que se propone subir al docente. Las prioridades son el juicio del revisor; no constituyen una rúbrica oficial ni una promesa de aprobación.

**Estado general: el paquete requiere correcciones importantes de autocontención. El algoritmo y el trabajo experimental principal están respaldados. No hace falta comenzar por nuevos entrenamientos.**

Esta guía registra acciones pendientes; su creación no significa que se hayan aplicado. No se modificaron el notebook ni el PDF para escribirla.

## 1. Versión a la que se aplica

Los hashes se volvieron a comprobar al crear esta guía y coinciden con los de la revisión anterior.

| Artefacto | SHA-256 |
|---|---|
| [Notebook candidato](../para_entregar/LunarLander_DQN.ipynb) | `d73dc13c806b4cdd0da0b9a0a6ad59ebc82d72a3abeeead7bc4d943046a5cfda` |
| [Informe candidato](../para_entregar/informe.pdf) | `8c4f69838c781340d02e50c24cf7b1cc2683c065f7d347bfa2bb01e6fb8476b6` |

Las celdas del notebook se numeran desde 1, contando también las de Markdown. Las páginas citadas son las del PDF evaluado. Si se reorganizan las celdas o se recompila el informe, hay que actualizar esas referencias cuando corresponda.

Material de referencia:

- [Consigna local](consigna_local.pdf), dos páginas.
- [Consigna histórica](consigna_historica_60af2e4.pdf), dos páginas.
- [Plantilla docente](course_template.ipynb).
- [Fuente del informe](../apoyo/informe_fuente/informe.tex).
- [Verificación previa de ejecución](verificacion.json) y [manifiesto del paquete](MANIFIESTO.json).
- [Protocolo final del repositorio](../../protocols/team_final.json) y [manifiesto de las 32 políticas](../../analysis/team_study/manifest.json).

## 2. Estado final del informe actual

### Veredicto separado por alcance

| Objeto | Estado | Qué significa |
|---|---|---|
| Informe PDF, en lo técnico y académico | Entregable con correcciones menores de análisis y presentación | Cubre los cinco contenidos, tiene cinco páginas y sus cifras recalculables coinciden con los registros. Conviene precisar la interpretación estadística, algunos protocolos y el resumen. |
| Trabajo respaldado por el repositorio | Sustancialmente completo y técnicamente sólido | Existen entrenamientos, calibración, resultados, pesos y análisis adicionales. No se encontró un fallo central del DQN. |
| Paquete que recibe el docente en `para_entregar/` | Requiere correcciones importantes | El notebook no incorpora toda la evidencia de los experimentos descritos en el informe y debe ajustarse al cuaderno base. |
| Cumplimiento de la regla docente sobre IA | No verificable con la documentación local disponible | La versión histórica prohíbe la redacción con IA; la local omite esa prohibición; el informe declara asistencia en la escritura de la prosa. Falta confirmar la regla oficial vigente. |

**No entregaría los dos archivos exactamente como están.** Los impedimentos de cierre son la evidencia que no llega dentro del notebook, el ajuste formal a la plantilla y la regla de IA sin confirmar. No lo son el desempeño obtenido, las ocho semillas disponibles ni la falta de evaluación reservada.

### Qué ya está comprobado

- El PDF tiene **cinco páginas totales, incluidas las referencias**. Se leyeron y examinaron visualmente las cinco durante la revisión. No se observaron recortes, superposiciones importantes ni marcadores pendientes.
- El informe incluye solución DQN, calibración, aprendizaje y comparación con el azar, análisis de política y ablaciones.
- El notebook contiene 31 celdas, 18 de código, con salidas guardadas y sin errores almacenados.
- Hay datos incorporados de **32 políticas**, con 9.600 filas de entrenamiento y 6.400 de evaluación. Se contrastaron con los originales y con los 160 hashes del manifiesto de esas corridas.
- La arquitectura es `8 → 64 → 64 → 4`, con 4.996 parámetros. Se verificó la correspondencia del checkpoint incorporado con el completo de semilla 42.
- Pasaron siete pruebas breves de comportamiento del agente sobre transiciones sintéticas. No fueron corridas de entrenamiento del estudio.
- Las dos figuras fuente del PDF se regeneraron en una carpeta temporal y coincidieron en sus píxeles renderizados.
- Se recalcularon las tablas de calibración A0–A3, evaluación final, finales de episodio y contrastes estadísticos desde los datos individuales disponibles.
- Los resultados históricos de replay coinciden con las salidas guardadas del notebook recuperado de Git. No se recalcularon desde retornos individuales persistidos, porque no se localizaron esos registros individuales.

### Resultados centrales que deben conservarse

| Condición | Retorno medio entre ocho políticas | DE muestral entre políticas | Éxitos sobre 1.600 | Tasa de éxito |
|---|---:|---:|---:|---:|
| Completo | 137,8 | 62,0 | 647 | 40,44 % |
| Sin red objetivo | 47,2 | 126,6 | 352 | 22,00 % |
| Sin replay | −0,8 | 94,5 | 189 | 11,81 % |
| Sin exploración | 25,8 | 132,8 | 287 | 17,94 % |

El completo supera estrictamente el 10 % en siete de sus ocho políticas. La excepción es la semilla 3042, con 7/200 éxitos. La consigna no exige retorno medio de 200 ni que todas las semillas superen el umbral.

La comparación local de la política completa de semilla 42 con el azar es otro resultado: 168,86 de retorno y 63/200 éxitos frente a −183,17 y 0/200. Está respaldada por las salidas guardadas y la verificación previa; no se volvió a ejecutar al redactar esta guía. No debe reemplazar la evaluación original de esa política.

### Alcance de las verificaciones de ejecución

`verificacion.json` registra una ejecución anterior de exactamente estos hashes, con kernel nuevo y directorio inicialmente vacío: 18 celdas de código, 37,72 segundos, sin reentrenamiento. Se aprovechó esa evidencia durante la revisión.

Eso no acredita una instalación limpia en todas las plataformas. Durante la revisión no se lanzaron entrenamientos completos, descargas ni una nueva ejecución integral del notebook. Al crear esta guía se comprobaron nuevamente los hashes, sin repetir el análisis experimental.

## 3. Lista maestra de acciones

Las prioridades no son ponderaciones de una nota. Los responsables quedan a asignar por el equipo; no se presuponen acuerdos.

| ID | Categoría | Acción | Estado | Tipo de trabajo |
|---|---|---|---|---|
| [H01](#h01) | Obligatorio | Incorporar la evidencia de los experimentos presentados | Pendiente | Empaquetado, edición y ejecución breve |
| [H02](#h02) | Obligatorio | Restituir material suministrado y correspondencia con la plantilla | Pendiente | Edición y ejecución breve |
| [H03](#h03) | Obligatorio condicionado | Confirmar la regla oficial de IA y actuar según corresponda | Pendiente | Comprobación documental |
| [R01](#r01) | Importante y recomendable | Ajustar la jerarquía e interpretación de los contrastes | Pendiente | Edición; recálculo breve si se añade una alternativa |
| [R02](#r02) | Importante y recomendable | Actualizar estado del notebook y aclarar nombres | Pendiente | Edición |
| [R03](#r03) | Importante y recomendable | Precisar diferencias de protocolo de la calibración | Pendiente | Edición |
| [R04](#r04) | Importante y recomendable | Incluir instrucciones de dependencias utilizables | Pendiente | Edición; instalación limpia si se decide verificarla |
| [R05](#r05) | Importante y recomendable | Completar leyenda y escalas de los cortes | Pendiente | Edición y regeneración breve |
| [R06](#r06) | Importante y recomendable | Precisar tres pasajes del informe | Pendiente | Edición |

## 4. Correcciones obligatorias

### H01

**Completar la evidencia experimental dentro del notebook.**

**Ubicación:** informe, sección III y tabla II; sección V y tabla IV. Notebook, celdas 13 y 23–26.

**Requisito:** consigna, p. 2, Entregables 1: cuaderno autocontenido con código y resultados de los experimentos realizados.

**Problema concreto:** los datos finales sí están incorporados. Lo que falta dentro del notebook es la evidencia de calibración que presenta el informe y los registros auxiliares que sustentan la tabla IV. Que existan en el repositorio no los hace accesibles al docente que recibe solamente `para_entregar/`.

#### Acciones

- [ ] Incorporar configuraciones, semillas, presupuesto, código y resultados de las ocho políticas de calibración A0–A3 con semillas 42 y 1042.
- [ ] Identificar explícitamente cuáles son las corridas de calibración: no confundir el A0 histórico con el completo seleccionado para el estudio integrado.
- [ ] Incorporar el código y los resultados históricos de capacidad de replay que se citan en el informe, con su procedencia.
- [ ] Incorporar el código, configuración y datos de la comparación histórica de exploración presentada.
- [ ] Si se mantiene la tabla IV, incorporar los registros de la reevaluación auxiliar de las 32 políticas y el código que agrega los 640 episodios.
- [ ] Si se decide retirar la tabla IV por ser accesoria, retirar también sus afirmaciones dependientes y revisar referencias cruzadas. Esto no elimina la obligación de respaldar la calibración presentada.
- [ ] Mantener desactivados los entrenamientos históricos y finales durante la ejecución ordinaria.
- [ ] Evitar que clases, variables o funciones históricas sobrescriban silenciosamente el agente y los datos finales.

#### Dónde está la evidencia

- A0–A3: corridas de `TAREA2/results/` del 27 y 29 de octubre; `results/a2_controlled_20261029T233548_194093Z/` sirve como índice auxiliar, pero las cifras deben salir de los CSV originales.
- Exploración: [resultados históricos](../../AgustinAgente/resultados/exploracion/).
- Finales de episodio: [metricas_politica.json de policy_all](../../analysis/team_study/policy_all/metricas_politica.json).
- Código de política: [analisis_politica.py](../../scripts/analisis_politica.py).
- Código histórico retirado del árbol actual: recuperación desde Git documentada en [REFERENCIAS_LIMPIEZA.md](../../REFERENCIAS_LIMPIEZA.md). El commit allí indicado es `2643141f75c630319f0b0b24fbd09eed4beb8fee`; las rutas son `TAREA2/LunarLander_DQN_B.ipynb` y `TAREA2/AgustinAgente/LunarLander_DQN_AgustinAgente.ipynb`.

**Límite que debe preservarse:** para replay histórico se localizaron código, gráfica y tabla impresa, pero no retornos individuales persistidos. Se pueden conservar esas salidas originales indicando su alcance. No inventar un archivo de datos ni atribuir esos números a la implementación final, cuyo calendario es diferente.

**Terminado cuando:** una copia del notebook, en un directorio sin las carpetas del repositorio, permite consultar los experimentos presentados y recalcular los resultados para los que se incorporaron datos individuales. Cada salida histórica no recalculable queda identificada como tal.

**No exige:** un CSV separado, descargar pesos ni repetir entrenamientos.

### H02

**Restituir los bloques suministrados del cuaderno base.**

**Ubicación:** encabezado y secciones 1–9 de la plantilla, frente al notebook candidato.

**Requisito:** consigna pp. 1–2: entrega en una copia del cuaderno provisto. La plantilla pide el cuaderno completo y señala que, fuera de los TODO, el material ya está escrito y «no hay que tocarlo».

**Lo que ya está bien:** las interfaces y las funciones principales suministradas conservan el mismo árbol sintáctico: agentes base, entrenamiento, evaluación, media móvil, gráfica de progreso y funciones del GIF. La diferencia de numeración no es por sí sola un error.

**Desviación:** faltan otros bloques suministrados, como `jugar_lunar_lander`, demostraciones y parte de la presentación original del entorno. La corrección es formal y de organización; no se propone rehacer DQN.

#### Acciones

- [ ] Comparar el candidato con `revision/course_template.ipynb` y reintegrar el material suministrado omitido.
- [ ] Conservar firmas y lógica de las funciones docentes.
- [ ] Ubicar implementación, política y ablación en correspondencia clara con los tres TODO.
- [ ] Añadir los datos y análisis extra sin eliminar contenido base.
- [ ] Mantener una vía explícita para cargar resultados guardados y ejecutar la revisión sin reentrenar.

**Terminado cuando:** no falta material suministrado, los tres bloques están resueltos y la ejecución ordinaria sigue siendo breve y segura para los resultados originales.

### H03

**Confirmar la consigna oficial vigente sobre IA.**

**Ubicación:** ambas consignas, p. 2, Implementación; informe, p. 5, declaración de IA.

**Condición:** la histórica prohíbe IA para redactar; la local omite la prohibición. El informe declara asistencia en la escritura de prosa. El commit local que modifica el PDF no demuestra que la cátedra haya cambiado la regla.

#### Acciones

- [ ] Obtener la versión publicada oficialmente o una aclaración docente verificable.
- [ ] Registrar procedencia, fecha y alcance de esa regla.
- [ ] Si existe una actualización docente válida, evaluar el uso declarado contra esa versión.
- [ ] Si la prohibición histórica sigue vigente, seguir la regularización indicada por la cátedra.
- [ ] Conservar una declaración veraz del uso real de IA.

**Terminado cuando:** existe respaldo oficial suficiente y se ha resuelto la correspondencia entre lo permitido y lo realizado.

**No sirve como solución:** quitar la declaración, cambiar «redacción» por «corrección» si no describe lo ocurrido, introducir informalidad artificial o afirmar autorización a partir de una omisión local.

## 5. Mejoras importantes y recomendables

### R01

**Hacer coherente la interpretación estadística con las semillas compartidas.**

**Ubicación:** informe VI-A, pp. 4–5 y tabla V; notebook 27–30.

**Diagnóstico:** Mann–Whitney presupone muestras independientes. Se comparten semillas de inicialización y entrenamiento; completo/sin replay incluso tienen prefijos de entrenamiento idénticos en varias semillas. La independencia no queda justificada por describir las corridas como políticas distintas.

**Por qué no invalida toda la evaluación:** ya se presenta Wilcoxon emparejado con Holm y la conclusión sobre replay se mantiene. La unidad experimental está bien definida: una política entrenada, no cada episodio de evaluación.

- [ ] Mantener constancia de cuál fue el test originalmente propuesto, sin inventar un prerregistro distinto.
- [ ] Apoyar la conclusión en el análisis que respeta los bloques por semilla; presentar MW como complementario y condicionado a sus supuestos.
- [ ] Explicitar hipótesis, independencia entre pares y supuesto de simetría de Wilcoxon.
- [ ] Conservar Holm sobre los tres contrastes de cada familia.
- [ ] Mantener la distinción entre ausencia de significación y equivalencia.

Valores recalculados que sirven para verificar la edición:

| Ablación | Diferencia media, completo menos ablación | MW con Holm | Wilcoxon con Holm |
|---|---:|---:|---:|
| Sin objetivo | 90,63 | 0,13038 | 0,10938 |
| Sin replay | 138,67 | 0,02098 | 0,02344 |
| Sin exploración | 112,01 | 0,12991 | 0,07812 |

Si el equipo prefiere evitar el supuesto de simetría, puede considerar el test de signos, explicando que contrasta otra hipótesis. En la revisión se obtuvo, con Holm, 0,140625; 0,0234375; 0,140625, respectivamente. **Es una alternativa motivada por este diseño, no una obligación de agregar otro test.**

**Terminado cuando:** el texto, la hipótesis, el diseño y la interpretación de los valores p son coherentes. No afirmar que MW es exclusivamente un test de medias ni que los 1.600 episodios son réplicas independientes de entrenamiento.

**Trabajo:** edición; recálculo breve si se incorpora otra opción. No requiere entrenar.

### R02

**Actualizar estado y nombres.**

**Ubicación:** notebook, celdas 1, 15 y 31; informe, portada y IV-A.

- [ ] Sustituir el estado de «borrador del 2/11» por el estado real de la versión final.
- [ ] Retirar pendientes ya resueltos: redacción, compilación y existencia del PDF de cinco páginas.
- [ ] Mantener únicamente pendientes reales; no declarar realizada la evaluación reservada.
- [ ] Aclarar la correspondencia entre «Fran» y «Juan Duarte» y usar nombres consistentes.
- [ ] Confirmar con el equipo los acuerdos metodológicos que el notebook todavía presenta como pendientes.

**Terminado cuando:** portada, atribución de corridas, autores y estado del trabajo se entienden sin inferencias. La diferencia de nombres no demuestra por sí sola una atribución falsa.

**Trabajo:** solo edición. Evita que una entrega terminada parezca incompleta.

### R03

**Precisar diferencias de protocolo en la calibración.**

**Ubicación:** informe III, p. 2.

- [ ] Aclarar que, para semilla 42, A0/A1 registran macOS 26.7.1 y A2/A3 macOS 27.2. Para 1042 no aparece esa diferencia en los metadatos comparados.
- [ ] Conservar la distinción entre cambios de hiperparámetros configurados y control de todos los factores de ejecución.
- [ ] Aclarar que el prototipo histórico sin replay actualizaba en cada transición desde el inicio, mientras que con replay esperaba completar el lote y actualizaba cada cuatro pasos.
- [ ] Mantener ese antecedente separado de la ablación final, que conserva inicio en 1.000 pasos y frecuencia de cuatro.

**Terminado cuando:** la narración coincide con configuraciones y código históricos. Se conserva la comparación de capacidad 100k/10k sin presentar el prototipo sin replay como una intervención idéntica a la final.

**Trabajo:** solo edición. No hace falta repetir la calibración para sostener la elección acotada que ya presenta el informe.

### R04

**Incluir instrucciones de dependencias que funcionen desde la entrega.**

**Ubicación:** notebook, celda 2; archivos de dependencias en `apoyo/`.

- [ ] Incluir dentro del notebook la lista de dependencias y una instrucción suficiente de instalación.
- [ ] No remitir exclusivamente a un `requirements.txt` que no se entrega junto al notebook.
- [ ] Identificar versiones comprobadas, Python y plataforma de análisis.
- [ ] Aclarar que `requirements-analysis-lock.txt` registra un entorno concreto; contiene dependencias de plataforma, como `appnope`.
- [ ] Separar instalación opcional de ejecución ordinaria; no hacer descargas innecesarias al consultar resultados.

**Terminado cuando:** alguien con solo el notebook y el PDF sabe preparar el entorno. No afirmar instalación universal verificada si solo se ejecutó sobre el entorno ya instalado.

**Trabajo:** edición. Una instalación limpia adicional es útil, pero no se ha demostrado un fallo que obligue a rehacer el entorno.

### R05

**Completar leyenda y escalas de los cortes.**

**Ubicación:** notebook, celdas 24–25; informe V y figura 2.

- [ ] Llamar a `leyenda_acciones` en el corte representativo del notebook; ya está definida, pero no se utiliza allí.
- [ ] Mantener la leyenda correcta de las otras figuras y del PDF.
- [ ] Aclarar que los ejes corresponden a las observaciones escaladas del entorno; no etiquetar automáticamente posiciones y velocidades como unidades físicas directas.
- [ ] Identificar el ángulo en radianes y distinguir la velocidad angular observada de una lectura directa sin escala.
- [ ] Precisar que «no normalizamos» significa que no se agregó un preprocesamiento propio a las observaciones entregadas por Gymnasium.

**Terminado cuando:** el gráfico puede interpretarse por sí solo, conserva su semilla/checkpoint y sus seis valores fijos, y las unidades son compatibles con el entorno.

**Trabajo:** edición y regeneración breve. No requiere entrenar ni ampliar los cortes.

### R06

**Precisar tres pasajes del informe y preservar una voz auténtica.**

| Ubicación | Fragmento o asunto | Acción |
|---|---|---|
| P. 1, resumen | «40,44 % […] frente a un 0 % del agente aleatorio» | Separar el agregado original de ocho políticas de la comparación local de una política: 31,5 % frente a 0 %. No sugerir que todo proviene de un único experimento homogéneo. |
| P. 2, III | «Los pilotos cortos permitieron comprobar la implementación» | Los autores deben precisar qué comprobaron realmente y con qué evidencia. No atribuir a esos pilotos las pruebas ejecutadas después durante la revisión. |
| P. 5, VIII | Ampliar semillas, repetir plataformas y realizar evaluación reservada | Vincular cada mejora con su propósito: variabilidad de entrenamiento, confusión experimental o generalización. No presentarlas como una lista obligatoria para aprobar. |

**Terminado cuando:** cada afirmación tiene alcance y evidencia identificables. La edición debe preservar el sentido y las decisiones reales del equipo, sin inventar motivaciones ni experiencias.

**Trabajo:** solo edición. No se recomienda reescribir todo el informe para que «parezca menos asistido».

## 6. Qué puede quedar como está con una limitación clara

Categoría de todos los puntos siguientes: **opcional; puede quedar sin corregir**.

| ID | Mejora no exigida para el cierre | Límite que se acepta | Cuándo pasaría a ser necesaria |
|---|---|---|---|
| O01 | Ejecutar evaluación reservada 30000–30199 | Los resultados actuales son de desarrollo, utilizados durante la selección. El informe ya lo declara. | Para afirmar evaluación independiente del proceso de calibración. |
| O02 | Más semillas, cruce de plataformas e igualdad de presupuesto por pasos | Variabilidad limitada; no se aísla completamente plataforma/componente ni se compara igual costo computacional. | Para conclusiones causales más fuertes, convergencia o eficiencia a igual presupuesto de interacciones. |
| O03 | Incorporar los otros 31 checkpoints | No se puede reevaluar inmediatamente cada política original sin recuperar pesos o reentrenar. | Si se pretende ofrecer reevaluación directa de todas las redes sin entrenamiento. |
| O04 | Añadir bootstrap, más intervalos, potencia o una batería de tests | La incertidumbre podría describirse mejor, pero ya hay dispersión, tamaños de efecto y ajustes múltiples. | Si una afirmación adicional necesita específicamente ese análisis. Resolver R01 tiene prioridad. |
| O05 | Limpiar el `main()` residual de la celda 29 | Referencia nombres no definidos, pero no se invoca en el recorrido ordinario. Es confuso, no un fallo actual de ese recorrido. | Si se pretende presentar esa función como una interfaz ejecutable. |

No hace falta incorporar Double DQN, replay priorizado, más capas ni una búsqueda exhaustiva. Tampoco corresponde exigir éxito en todas las semillas o retorno medio de al menos 200: no es el criterio expresado por la tarea.

## 7. Qué conviene conservar

- **Agente:** arquitectura, objetivo de Bellman, SmoothL1, Adam, clipping, sincronización y calendarios coinciden con la explicación del informe. No reemplazar un algoritmo comprobado por preferencia personal.
- **Terminación y truncamiento:** el código detiene ambos tipos de episodio, pero solo `terminated` anula el valor futuro. La explicación es correcta.
- **Evaluación:** decisiones greedy sin aprendizaje; aleatorio evaluado sobre los mismos escenarios de la comparación local.
- **Datos originales:** mantener las 32 políticas seleccionadas, incluidas las de mal desempeño. No sustituirlas por reevaluaciones locales.
- **Unidad experimental:** una política entrenada. Conservar la distinción frente a los 200 episodios usados para estimar su desempeño.
- **Transparencia:** mantener diferencias de plataforma, protocolo histórico, presupuesto por episodios y ausencia de evaluación reservada.
- **Política:** conservar el cuidado al distinguir estados sintéticos, proyecciones visitadas, valores Q y probabilidades de éxito.
- **Informe:** cinco páginas totales, cinco contenidos, figuras legibles y referencias pertinentes. No ampliar por rellenar ni recortar limitaciones importantes para introducir mejoras opcionales.

La revisión encontró una voz académica cuidada y uniforme, con evidencia concreta de comprensión. El estilo no permite autenticar autoría. La asistencia en prosa está declarada y debe tratarse por H03, no mediante cambios cosméticos.

## 8. Plan mínimo de cierre y aceptación

El trabajo de empaquetado puede avanzar mientras se obtiene la confirmación documental de IA. No es necesario detener las mejoras independientes de esa respuesta.

| Orden | Acción | Criterio para darla por terminada |
|---|---|---|
| 1 | Resolver H03 | Regla oficial identificada y cumplimiento resuelto de manera veraz. |
| 2 | Completar H01 | Experimentos presentados accesibles dentro del notebook; procedencia y límites históricos explícitos. |
| 3 | Resolver H02 y R02 | Cuaderno base conservado; autores, estructura y estado final coherentes. |
| 4 | Resolver R01, R03 y R06 | Hipótesis, protocolos, resumen y conclusiones concordantes con la evidencia. |
| 5 | Resolver R04 y R05 | Instrucciones de entorno suficientes y cortes interpretables. |
| 6 | Ejecutar una copia del notebook final | Kernel nuevo, directorio sin resultados externos, reentrenamiento desactivado, sin errores y sin modificar los datos originales. |
| 7 | Recompilar y revisar el PDF final | Máximo cinco páginas totales; sin recortes; cifras y referencias cruzadas coherentes con el notebook. |
| 8 | Actualizar copias, hashes y verificación | Los archivos de `para_entregar/` son exactamente los que se acaban de revisar. |

### Cuidado con las copias

El paquete se preparó a partir de `TAREA2/entrega_borrador.ipynb` y `TAREA2/informe/informe.pdf`. La fuente LaTeX también está duplicada en `apoyo/informe_fuente/`.

- [ ] Elegir y registrar cuál será la fuente de trabajo durante el cierre.
- [ ] Aplicar las correcciones sobre esa fuente y actualizar después las copias correspondientes.
- [ ] No dejar un PDF nuevo en `TAREA2/informe/` y uno anterior en `para_entregar/`, o viceversa.
- [ ] Actualizar `MANIFIESTO.json` y `verificacion.json` tras los cambios. No conservar una marca de ejecución aprobada asociada a hashes anteriores como si validara los archivos nuevos.
- [ ] Abrir por última vez los archivos que efectivamente se subirán.

### Checklist de subida

- [ ] `para_entregar/LunarLander_DQN.ipynb`, con código, resultados incorporados, salidas e instrucciones suficientes.
- [ ] `para_entregar/informe.pdf`, concordante con el notebook y de hasta cinco páginas, referencias incluidas.
- [ ] Archivos adicionales solo si la plataforma docente los solicita; no usarlos para suplir la autocontención exigida del notebook.
- [ ] Fuentes, datos originales, manifiesto y verificación actualizados conservados internamente.
- [ ] Esta guía permanece como documento interno de revisión, fuera de `para_entregar/`.

## 9. Preguntas de defensa oral

No son una prueba de autoría ni agregan requisitos de entrega. Sirven para que cada integrante compruebe que entiende lo que se está presentando.

1. ¿Qué ocurre en el paso 1000 cuando coinciden la primera actualización y la sincronización de la red objetivo?
2. ¿Cómo cambia el objetivo de Bellman entre `terminated=True` y un truncamiento por tiempo?
3. ¿Qué transiciones utiliza exactamente la variante sin replay y qué cambia además de la memoria?
4. ¿Por qué el completo puede cumplir con media 137,8 y qué diferencia hay entre cumplimiento agregado y por semilla?
5. ¿Qué separan A0–A2 y A2–A1, y qué factores no quedaron completamente controlados?
6. ¿Qué unidad experimental usan los tests, qué hipótesis contrastan y qué implica compartir semillas?
7. ¿Por qué una proyección de estados visitados no valida automáticamente un corte sintético, y por qué un final «posado» no garantiza retorno ≥200?

## 10. Registro de cierre

Completar después de realizar las acciones; no cambiar estos campos a terminado por la sola existencia de esta guía.

| Comprobación final | Estado al crear la guía | Evidencia que falta registrar |
|---|---|---|
| Regla docente de IA resuelta | Pendiente | Fuente oficial y decisión aplicable |
| Evidencia de calibración y análisis auxiliar incorporada | Pendiente | Celdas y salidas finales |
| Correspondencia con plantilla revisada | Pendiente | Comparación final |
| Recomendaciones aceptadas o descartadas con motivo | Pendiente | Decisiones del equipo |
| Ejecución de la versión corregida en kernel nuevo | Pendiente | Fecha, entorno, hashes y resultado |
| PDF corregido revisado visualmente | Pendiente | Fecha, hash y número de páginas |
| Dos entregables definitivos identificados | Pendiente | Hashes finales |

**Estado de cierre actual: todavía no listo para subir.** El trabajo central está realizado; falta cerrar el paquete, precisar algunos argumentos y resolver la regla de IA. No se necesita rehacer el estudio para atender las correcciones señaladas.
