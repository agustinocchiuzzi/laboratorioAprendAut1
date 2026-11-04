# Guía para redactar mi parte del informe (Agustín, 4/11)

**Qué es este archivo:** material para leer y después escribir **con mis palabras**. Son ideas,
cifras verificadas y advertencias, en viñetas. **No es texto para copiar al informe.**

Fuentes: `informe/informe.tex` (versión de Thiago del 3/11, idéntica a
`entrega_final/apoyo/informe_fuente/informe.tex`), la revisión en `entrega_final/revision/`
(guía, H03, R01, R05, R06) y los datos de `analysis/team_study/`. Todas las cifras de abajo se
recalcularon desde esos datos.

---

## 0. Qué me queda (en orden)

Entrega: **hoy 4/11, 23:59** (actividad "Entrega L2" en EVA).

| # | Tarea | Tiempo aprox. | Sección de esta guía |
|---|---|---|---|
| 1 | Hablar con Thiago y Fran del tema IA (antes de escribir nada) | 15 min | 1 |
| 2 | Redactar la **sección V, Análisis de la política** | 60–75 min | 2 |
| 3 | Redactar la **sección VII, Limitaciones** | 20 min | 3 |
| 4 | Aportar a resumen, conclusiones y declaración de IA (compartido) | 20 min | 4 |
| 5 | Revisar referencias | 10 min | 6 |
| 6 | Compilar en Overleaf, revisar 5 páginas, sincronizar copias | 20 min | 7 |

Lo técnico de mi lado **ya está hecho**: 8 corridas sin exploración (integradas en el estudio de 32
políticas), propuesta y script estadístico, cortes y figuras, plantilla IEEE. La revisión R05
(leyenda y escalas de los cortes en el notebook) también quedó resuelta por Thiago. **Lo único que
falta de mi parte es texto.**

---

## 1. Primero: el tema IA (H03), decidir en equipo

Hechos (de `entrega_final/revision/seguimiento/H03.md`):

- La letra **original** (commit `60af2e4`) dice: "No está permitido utilizar herramientas de IA para
  la redacción del informe. Sí es aceptable su uso para la corrección gramatical y de estilo."
- La letra que está hoy en el repo (subida por Thiago el 3/11) **no tiene la primera frase**, pero
  conserva "Sí es aceptable…", que solo tiene sentido después de la prohibición. **No hay evidencia
  de que el cambio venga de los docentes.**
- El informe actual declara (p. 5) que la IA ayudó a "redactar y editar este informe", "incluyó la
  escritura de la prosa".
- **Mis secciones V y VII, tal como están hoy, no las escribí yo.**

Qué hacer:

- [ ] Entrar a EVA (curso 43, actividad "Entrega L2") y mirar la letra publicada ahí. Es la única
      fuente que vale.
- [ ] Si la prohibición sigue: las secciones las tenemos que escribir nosotros (lo que propone esta
      guía para las mías) y la declaración de IA tiene que decir la verdad sobre lo que quede.
      No sirve solo borrar o suavizar la declaración.
- [ ] Si los docentes realmente la quitaron: guardar la evidencia (captura con fecha) y aun así
      revisar que la declaración sea exacta.
- [ ] Ante la duda, preguntar a los docentes por el foro antes de entregar.

---

## 2. Sección V: Análisis de la política

**Lo que pide la letra:** "Analizar qué aprendió la política mediante cortes que fijen algunas
dimensiones del estado y muestren la acción elegida o el valor Q en función de las demás".

**Presupuesto:** hoy ocupa ≈ 650 palabras + figura 2 (ancho completo) + tabla IV. Mantener **≤ 650**
para no pasar las 5 páginas.

**Material visual:** figura 2 = `informe/figuras/cortes_politica.pdf` (completo, semilla 42).
Figuras extra (solo para entender, no entran por espacio): `analysis/team_study/policy_all/`
(`cortes_8_semillas_angulo.png`, `cortes_8_semillas_altura.png`, `acciones_por_altura.png`).

Estructura sugerida en 5 bloques:

### 2.1 Método (≈ 120 palabras)

Ideas a cubrir:
- Se usan los pesos finales guardados; no se entrena nada para el análisis.
- Política analizada: completo, **semilla 42**. Motivo: es la primera del protocolo; **no** se eligió
  por rendimiento (decirlo explícitamente).
- Estado de referencia: x = 0, y = 0,8, vx = 0, vy = −0,3, θ = 0, ω = 0, sin contacto (c1 = c2 = 0).
  En palabras: sobre la plataforma, a media altura, bajando despacio, derecha.
- Tres cortes, cada uno varía 2 dimensiones y deja las otras 6 fijas: (x, vx), (θ, ω), (y, vy).
  Grilla de 101 × 101 puntos.
- En cada punto: **acción greedy** (argmax Q, fila de arriba de la figura) y **max Q** (fila de abajo).
- **Escalas (pedido de la revisión R05, hoy no está en el informe):** los ejes son las
  **observaciones que entrega Gymnasium**, que ya vienen escaladas; no son metros ni m/s.
  θ sí está en **radianes**; ω es la velocidad angular escalada (×0,4). "No normalizamos" significa
  que **no agregamos** una normalización propia, no que las observaciones sean físicas crudas.

  | Observación | Cómo la construye Gymnasium 1.4.0 |
  |---|---|
  | x | (pos.x − 10) / 10 |
  | y | (pos.y − altura de la plataforma) / (20/3) |
  | vx | vel.x / 5 |
  | vy | vel.y · 2/15 |
  | θ | ángulo, en radianes |
  | ω | velocidad angular · 0,4 |

  En el informe alcanza con una frase que diga esto; la tabla es para mí.
- Qué hace cada acción (lo medimos en el entorno): 1 = motor izquierdo (empuja a la izquierda y
  aumenta θ), 2 = principal (frena la caída), 3 = motor derecho (empuja a la derecha y reduce θ).

### 2.2 Qué muestra cada corte (≈ 250 palabras)

**Altura – vy (panel derecho):**
- Debajo de una frontera inclinada, motor principal.
- La frontera depende de la altura: cerca del suelo activa el motor con caídas lentas
  (vy ≈ −0,2); arriba tolera caídas más rápidas (vy ≈ −0,85 a y = 1,5).
- Lectura: frena según la altura que le queda, como lo haría un piloto.
- Por encima de la frontera hay inacción y también motor derecho. **No es una regla binaria de
  frenado.**

**Ángulo – ω (panel central):**
- Dos regiones de motores laterales opuestos separadas por una frontera que depende de θ **y** de ω.
- Del lado donde θ + ω es positivo usa el que reduce θ; del otro, el que lo aumenta.
- Lectura: corrige a la vez la inclinación y el giro (idea de controlador proporcional-derivativo).
- La heurística de Gymnasium tiene una regla de la misma forma (con otra pendiente). Es una
  comparación cualitativa: **no decir que la red implementa un controlador lineal**.
- En las **8 semillas** del completo aparece la misma organización (`cortes_8_semillas_angulo.png`);
  cambian las fronteras y dónde aparecen el motor principal y la inacción. Esto muestra que es algo
  robusto y no un accidente de una semilla.

**x – vx (panel izquierdo):**
- Con vx positiva grande usa el motor izquierdo, que reduce la deriva hacia la derecha.
- La división **no es simétrica** y depende también de x.
- Con un solo estado de referencia **no se puede** afirmar una preferencia global por un lado.

**Fila de abajo (max Q):**
- Q es el **retorno descontado esperado**; **no es una probabilidad** de aterrizar ni el retorno del
  episodio.
- Hay zonas con Q alto y caída muy rápida que el agente no visita: la red extrapola. Esto muestra que
  una predicción de la red, sola, no valida una acción.

### 2.3 Cuidado con los estados sintéticos (≈ 60 palabras)

- Con 6 coordenadas fijas, muchos puntos del corte son estados raros o inalcanzables.
- Por eso en el análisis se superpusieron los estados visitados en 20 episodios reales.
- Pero esos puntos son **proyecciones**: sus otras 6 coordenadas no son las del corte. Ayudan a ver
  qué zonas importan, **no validan** las decisiones del corte.

### 2.4 Cómo terminan los episodios: tabla IV (≈ 150 palabras)

Datos: reevaluación auxiliar de las 32 redes en macOS (`analysis/team_study/policy_all/`),
**20 episodios por red** (semillas 10000–10019), 160 por condición.

| Condición | Posado | Choque/salida | Truncado (1000 pasos) | R ≥ 200 |
|---|---:|---:|---:|---:|
| Completo | 105 | 7 | 48 | 68 |
| Sin red objetivo | 66 | 54 | 40 | 44 |
| Sin replay | 22 | 135 | 3 | 21 |
| Sin exploración | 35 | 48 | 77 | 27 |

Cómo se clasifica: si el episodio termina y el último paso vale exactamente +100 → posado; si
termina de otra forma → choque o salida de la pantalla; si se corta a los 1000 pasos → truncado.

Ideas a cubrir:
- Completo: de sus 55 episodios no posados, **48 son truncados**. Cuando falla, casi nunca choca:
  **queda flotando** sin terminar de posarse.
- Sin replay: **135 de 160 terminan en choque o salida**. Falla de otra manera.
- Sin exploración: muchos truncados (77) y también choques (48). Mezcla los dos fracasos.
- **Posado ≠ éxito**: el completo tiene 105 posados pero solo 68 con R ≥ 200. Un aterrizaje que
  gasta mucho combustible puede quedar por debajo del umbral.
- Es una reevaluación auxiliar, en otra plataforma: **no reemplaza** la evaluación de 200 episodios
  del estudio y no son tasas de éxito.

### 2.5 Cierre de la sección (≈ 50 palabras, opcional)

- Una oración que conecte: los cortes muestran frenado y corrección del ángulo, y los finales
  muestran que el límite del completo es posarse, no chocar.

**No afirmar en la sección V:**
- que la semilla 42 es representativa del promedio (rinde 174,9; la media del completo es 137,8);
- probabilidades a partir de Q;
- causalidad del componente a partir de la tabla IV (es descriptiva);
- que el corte describe todo el espacio de estados.

---

## 3. Sección VII: Limitaciones

**Presupuesto:** hoy ≈ 130 palabras. Se puede llevar a **≈ 180** si se acorta algo en V.
La revisión pide (R06) **ligar cada limitación con su consecuencia** y no presentarlas como
obligaciones.

Limitaciones, con su dato y su consecuencia:

| Limitación | Dato | Consecuencia para las conclusiones |
|---|---|---|
| Presupuesto por episodios, no por pasos | Las corridas usaron entre **32 968 y 241 521** pasos | No comparamos igual costo ni afirmamos convergencia |
| Pocas semillas | 8 por condición; DE entre semillas de 62 a 133 | Poca potencia: "no significativo" no es "sin efecto" (exploración −112 y objetivo −91 en promedio) |
| Plataformas distintas | Completo y sin replay en Windows/torch 2.15; sin exploración en Windows/torch 2.14.1; sin objetivo en macOS | El test no separa el efecto del componente del de la máquina |
| Escenarios de desarrollo | Evaluación en 10000–10199, también usados en la calibración | Las cifras pueden favorecer lo elegido; la evaluación reservada 30000–30199 no se hizo |
| 300 episodios | Los agentes seguían mejorando (mi agente de 1200 episodios pasó de 219 a 247 entre los episodios 600 y 1200) | Comparamos políticas a mitad de camino |
| Definición de "sin replay" | Minilote efectivo 1 y sin reutilizar transiciones | La conclusión vale para esa ablación concreta, no para cualquier forma de quitar replay |
| Cortes | 3 planos, 6 coordenadas fijas, estados sintéticos | Describen la política localmente, no en todo el espacio |
| Reevaluaciones auxiliares | Tabla IV y comparación con el azar | Sirven para analizar redes guardadas, no agregan réplicas de entrenamiento |

- Algunas ya aparecen en otras secciones (plataformas en VI-B, escenarios en IV-A): en VII basta con
  nombrarlas brevemente, sin repetir números.
- **Agregar (falta hoy):** que las semillas no reproducen el mismo resultado entre máquinas (Fran lo
  comprobó: la semilla 42 del completo difiere en el tercer decimal del primer episodio entre su
  máquina y la de Thiago).

---

## 4. Partes compartidas (resolver con Thiago y Fran)

### Resumen (R06)
- Hoy mezcla dos cosas en una frase: el **40,44 %** del completo (8 políticas, 1600 episodios,
  estudio original) y el **0 %** del aleatorio, que sale de **otra** comparación: una sola política
  (semilla 42) reevaluada localmente con **31,5 %** (63/200) frente a 0/200 del aleatorio.
- Separarlas: el 0 % se compara con el 31,5 %, no con el 40,44 %.
- Si se adopta R01 (abajo), el resumen tiene que decir con qué test se concluye.

### Conclusiones (R01 y R06)
- Replay: es la única evidencia firme, pero calificarla: vale **bajo los supuestos del test** y para
  **esta** ablación (que también cambia el minilote y la reutilización), sin aislarla de la plataforma.
- Objetivo y exploración: **inconclusos**, no "innecesarios".
- Si se proponen mejoras (más semillas, misma plataforma, evaluación reservada), decir **para qué**
  sirve cada una: variabilidad, separar el efecto de la máquina, generalización.

### Declaración de IA
- Tiene que describir lo que realmente pasó (ver sección 1). Si mis secciones las reescribo yo, la
  declaración tiene que reflejarlo.

---

## 5. Mi propuesta estadística y el cambio R01 (para entender y defender)

- **Lo que propuse (2/11):** Mann-Whitney bilateral exacto como test principal + Holm por las 3
  comparaciones; Wilcoxon emparejado por semilla como sensibilidad.
- **Lo que decidió la revisión R01 (4/11):** dar prioridad a **Wilcoxon emparejado + Holm**,
  porque el diseño comparte semillas entre condiciones (mismos pesos iniciales y mismos `reset`; los
  primeros ≈ 10 episodios de completo y sin replay coinciden exactamente). MW queda como
  complemento, porque supone muestras independientes y eso no se cumple del todo.
- **No se eligió por el p:** con cualquiera de los dos, la conclusión es la misma. Solo sin replay es
  significativo tras Holm.

| Ablación | Δ media | p MW | MW + Holm | p Wilcoxon | Wilcoxon + Holm | Completo gana |
|---|---:|---:|---:|---:|---:|---:|
| Sin red objetivo | 90,6 | 0,130 | 0,130 | 0,109 | 0,109 | 7/8 |
| Sin replay | 138,7 | 0,007 | **0,021** | 0,008 | **0,023** | 8/8 |
| Sin exploración | 112,0 | 0,065 | 0,130 | 0,039 | 0,078 | 7/8 |

- Ojo con exploración: Wilcoxon **sin** ajustar da 0,039 (< 0,05), pero **con Holm** da 0,078. No es
  significativo. No citar el 0,039 como resultado.
- La sección VI la escribe Fran; yo tengo que poder explicarla si me preguntan.

---

## 6. Referencias (mi tarea del 3/11)

Las 6 actuales están **bien** (verifiqué autor, revista, volumen y páginas):

| Clave | Referencia | Se cita en |
|---|---|---|
| towers2024 | Gymnasium, arXiv:2407.17032 | Intro |
| mnih2015 | DQN, Nature 518, 529–533 | Intro, II |
| sutton2018 | Sutton y Barto, 2.ª ed. | II |
| mann1947 | Mann-Whitney, Ann. Math. Statist. 18(1) | VI |
| holm1979 | Holm, Scand. J. Statist. 6(2) | VI |
| wilcoxon1945 | Wilcoxon, Biometrics Bulletin 1(6) | VI |

Sugerencia (solo si entra en las 5 páginas): citar **Adam**, que se usa y no tiene referencia:
D. P. Kingma y J. Ba, "Adam: A method for stochastic optimization," en *Proc. ICLR*, 2015,
arXiv:1412.6980.

---

## 7. Formato y cierre

- [ ] Editar **una sola fuente**: `TAREA2/informe/informe.tex`. Al final copiarla a
      `entrega_final/apoyo/informe_fuente/informe.tex`.
- [ ] Compilar en Overleaf (pdfLaTeX, clase IEEEtran; subir `informe.tex` y `figuras/`). En esta
      máquina no hay LaTeX.
- [ ] **≤ 5 páginas en total, referencias incluidas** (hoy son exactamente 5: cualquier agregado
      tiene que compensarse recortando).
- [ ] Revisar que la figura 2 no quede cortada y que tablas y figuras estén referenciadas en el texto.
- [ ] Copiar el PDF nuevo a `informe/informe.pdf` **y** a `entrega_final/para_entregar/informe.pdf`.
      No dejar dos PDF distintos.
- [ ] Antes de subir: abrir los dos archivos de `entrega_final/para_entregar/` (notebook y PDF).
      Subir solo esos dos.

---

## 8. Preguntas de defensa sobre mi parte

- **¿Por qué la semilla 42 en los cortes?** Es la primera del protocolo; elegir la mejor sería sesgar.
- **¿Qué es un corte?** Evaluar la red en una grilla que varía 2 dimensiones con las otras 6 fijas, y
  mirar la acción ganadora y el max Q.
- **¿Los puntos visitados validan el corte?** No: son proyecciones de estados con otras 6 coordenadas.
- **¿Q alto quiere decir que va a aterrizar?** No: es un retorno descontado estimado, no una
  probabilidad, y puede estar extrapolado.
- **¿Por qué posado no es éxito?** El éxito es R ≥ 200; un aterrizaje con mucho combustible puede no
  llegar (105 posados contra 68 con R ≥ 200).
- **¿Qué aprendió la política?** Frenar según la altura y corregir θ y ω a la vez; esto último se
  repite en las 8 semillas.
- **¿Por qué falla el completo?** Sobre todo por quedar flotando hasta el paso 1000 (48 de 55 fallos),
  no por chocar.
- **¿Sin exploración no sirve explorar?** Es 112 puntos peor en promedio y pierde en 7 de 8 semillas,
  pero con 8 semillas y mucha dispersión no alcanza la significancia. Es inconcluso, no "innecesario".
- **¿Cómo se clasifica un final?** Gymnasium pone exactamente +100 (quieta) o −100 (choque o salida)
  en el último paso; si se corta a los 1000 pasos es truncado.
