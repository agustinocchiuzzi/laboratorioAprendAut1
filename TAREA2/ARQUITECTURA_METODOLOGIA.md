# Arquitectura y metodología — apuntes de estudio asistidos por IA

**No usar este archivo como texto del informe.** Se conserva como material explicativo del código. La consigna original, página 2, indica: «No está permitido utilizar herramientas de IA para la redacción del informe». El equipo debe escribir sus propias secciones; este archivo no forma parte del informe IEEE. Fuente: `Tarea - Aprendizaje por Refuerzos (2026).pdf`.

Texto basado en la implementación y las corridas validadas de Thiago y Fran. Describe el protocolo ejecutado; el equipo debe confirmar que sus demás corridas coincidan antes de redactarlo como un estudio completo de 32 políticas. Adaptar su extensión al informe IEEE de hasta 5 páginas.

## Arquitectura del agente

Se implementó un agente Deep Q-Network (DQN) para LunarLander-v3, con acciones discretas y viento desactivado. El estado contiene ocho componentes: posición y velocidad en ambos ejes, ángulo, velocidad angular y contacto de cada pata con el suelo. La red recibe esos ocho valores y produce cuatro estimaciones de valor de acción. Se utilizan dos capas ocultas de 64 unidades con activación ReLU y una capa de salida lineal. La red online tiene 4996 parámetros entrenables. No se normalizan observaciones ni se recortan recompensas.

La referencia A0 utiliza un buffer de replay de 50 000 transiciones y minibatches de 64 muestras elegidas uniformemente sin reemplazo. El aprendizaje comienza al alcanzar 1000 interacciones y se actualiza cada cuatro interacciones. Una segunda red, inicialmente idéntica a la online y sin gradientes, calcula los valores objetivo. Sus pesos se reemplazan por los de la red online cada 1000 interacciones.

El objetivo de aprendizaje es

\[
y = r + \gamma (1-d) \max_{a'} Q_{\theta^-}(s',a'),\qquad \gamma=0.99,
\]

donde \(d\) indica terminación real del episodio. El truncamiento por tiempo finaliza el rollout, pero no elimina el término de valor futuro. Se minimiza SmoothL1 con parámetro beta igual a 1 mediante Adam, con tasa de aprendizaje 0.001, betas (0.9, 0.999), epsilon numérico de 10⁻⁸ y sin weight decay. La norma del gradiente se limita a 10.

Durante el entrenamiento se usa una política epsilon-greedy. Epsilon comienza en 1 y disminuye linealmente hasta 0.05 al completar 200 episodios; se mantiene en ese valor después. En evaluación se elige siempre una acción de valor máximo, sin aprendizaje ni exploración.

## Diseño experimental

Se eligió A0 como referencia operativa a partir de los experimentos de desarrollo disponibles y de su compatibilidad con la integración común. En las dos semillas comparables de la exploración de tasas de aprendizaje, A0 obtuvo mayor retorno medio agregado que las alternativas 0.0005 y 0.005. Esta evidencia limitada justifica la selección para el estudio, pero no demuestra optimalidad ni superioridad general frente a configuraciones entrenadas con otros presupuestos.

El diseño previsto contiene cuatro condiciones: agente completo, sin red objetivo, sin replay y sin exploración. Cada condición se entrena desde pesos nuevos con ocho semillas: 42, 1042, 2042, 3042, 4042, 5042, 6042 y 7042. Las semillas inicializan Python, NumPy, PyTorch y el espacio de acciones; el reset del episodio i, contado desde cero, usa la semilla base más i. Compartir la lista de semillas no implica compartir trayectorias entre condiciones. Diferencias de dispositivo o versiones pueden afectar la reproducibilidad.

Cada política dispone de 300 episodios de hasta 1000 pasos, sin parada anticipada, y se evalúa el checkpoint final. El presupuesto es igual en episodios, pero los pasos efectivos y el costo computacional pueden diferir. Se registran ambos junto con el tiempo de entrenamiento y evaluación.

Las ablaciones se definen así:

- Sin red objetivo: el bootstrap utiliza la red online, manteniendo el objetivo fuera del cálculo de gradientes.
- Sin replay: se utiliza únicamente la transición actual, con batch efectivo de uno y sin almacenar experiencias. Se mantienen el warmup y la frecuencia de actualización. El cambio de batch forma parte de la definición de esta ablación.
- Sin exploración: epsilon es cero desde el comienzo del entrenamiento y las acciones se eligen de forma greedy.

Las demás configuraciones se mantienen iguales a la referencia. El protocolo original de cada integrante y sus pilotos se consideran calibración y no se combinan automáticamente con el estudio común.

## Evaluación y unidad de análisis

Las corridas disponibles se evaluaron durante 200 episodios con semillas de reset 10000–10199, iguales para cada política, sin exploración ni aprendizaje. Se registran los retornos individuales, su media y la tasa de éxito, definida operacionalmente como la proporción de episodios con retorno mayor o igual a 200. Esta métrica no constituye una clasificación visual de cada aterrizaje.

Esas semillas ya se utilizaron para seleccionar configuraciones y se describen como evaluación de desarrollo. Las semillas 30000–30199 están reservadas para una comprobación posterior al cierre de todas las decisiones; este documento no afirma que dicha evaluación se haya realizado.

La unidad experimental es una política entrenada con una semilla. Los 200 episodios de evaluación estiman el rendimiento de esa política y no equivalen a 200 entrenamientos independientes. Para las comparaciones se preserva la correspondencia por semilla. La elección del test, sus supuestos, el manejo de empates y la corrección por tres comparaciones debe registrarse con Fran y Agustín antes de interpretar los contrastes finales. No se realizaron esos tests en el análisis parcial de Thiago.

## Curvas y reproducibilidad

Las curvas conservan el retorno original por episodio y muestran una media móvil causal de hasta 20 episodios. En el análisis preliminar solo se agregan conjuntamente las semillas presentes en ambas condiciones; las bandas representan una desviación estándar muestral entre semillas de entrenamiento y no intervalos de confianza. Los pasos acumulados se conservan para permitir también análisis por cantidad de interacciones.

Cada corrida guarda configuración, versiones de paquetes, revisión Git, hash SHA-256 del notebook fuente, datos de entrenamiento y evaluación, métricas y pesos finales. Se valida que las métricas coincidan con los retornos originales, que el presupuesto y las semillas sean correctos y que no existan duplicados en el conjunto seleccionado. Las corridas de bajo rendimiento se conservan. Repeticiones por fallos técnicos deben justificarse y documentarse.

## Estado actualizado al cierre del 2/11

Las 32 políticas están integradas: completo y sin replay de Fran, sin red objetivo de Thiago y sin exploración de Agustín, ocho semillas por condición. Las tres completas de Thiago se conservan aparte. El notebook incorpora las cuatro condiciones, cortes, GIF y los métodos estadísticos propuestos. La revisión metodológica, la redacción propia del informe y la verificación de sus cinco páginas siguen pendientes. Ver `CIERRE_02_11.md` y `analysis/team_study/README.md` para los artefactos actuales.
