# Agente DQN para LunarLander: explicación completa (enfoque C)

Este documento acompaña al cuaderno `LunarLander_DQN_AgustinAgente.ipynb`. La idea es que,
leyéndolo de punta a punta, se pueda explicar **qué hace cada línea del agente, por qué está
ahí y qué teoría la respalda**, y defender los resultados del enfoque C (exploración,
semillas y evaluación).

Contenido:

1. [Qué se pidió y qué se entrega](#1-qué-se-pidió-y-qué-se-entrega)
2. [El problema: LunarLander-v3](#2-el-problema-lunarlander-v3)
3. [Teoría: de aprendizaje por refuerzos a DQN](#3-teoría-de-aprendizaje-por-refuerzos-a-dqn)
4. [Teoría del enfoque C: exploración, semillas y evaluación](#4-teoría-del-enfoque-c-exploración-semillas-y-evaluación)
5. [El código del agente, línea por línea](#5-el-código-del-agente-línea-por-línea)
6. [Ejemplo numérico completo de un paso de aprendizaje](#6-ejemplo-numérico-completo-de-un-paso-de-aprendizaje)
7. [El resto del cuaderno, celda por celda](#7-el-resto-del-cuaderno-celda-por-celda)
8. [Hiperparámetros y semillas exactos](#8-hiperparámetros-y-semillas-exactos)
9. [Resultados](#9-resultados)
10. [Qué funcionó, qué falló, qué queda incierto](#10-qué-funcionó-qué-falló-qué-queda-incierto)
11. [Cómo correrlo](#11-cómo-correrlo)
12. [Preguntas probables y cómo responderlas](#12-preguntas-probables-y-cómo-responderlas)

**Parte 2: trabajo en el estudio del equipo (30/10 al 2/11)**

13. [Dónde estamos: el estudio del equipo y en qué se diferencia de mi agente](#13-dónde-estamos-el-estudio-del-equipo-y-en-qué-se-diferencia-de-mi-agente)
14. [30/10: protocolo y método estadístico](#14-3010-protocolo-y-método-estadístico)
15. [31/10: las 8 corridas sin exploración](#15-3110-las-8-corridas-sin-exploración)
16. [El script `test_ablacion.py`, línea por línea](#16-el-script-test_ablacionpy-línea-por-línea)
17. [1/11: análisis de la política (cortes y visualizaciones)](#17-111-análisis-de-la-política-cortes-y-visualizaciones)
18. [El script `analisis_politica.py`, por bloques](#18-el-script-analisis_politicapy-por-bloques)
19. [2/11: plantilla IEEE y qué tengo que escribir yo](#19-211-plantilla-ieee-y-qué-tengo-que-escribir-yo)
20. [Estado al 2/11 y pendientes](#20-estado-al-211-y-pendientes)
21. [Más preguntas probables (parte 2)](#21-más-preguntas-probables-parte-2)

---

## 1. Qué se pidió y qué se entrega

La consigna del grupo reparte focos; a mí me toca el **C**:

| Persona | Foco | Evidencia a traer |
|---|---|---|
| C | Esquema de exploración, semillas y evaluación | Comparación pequeña de **dos esquemas de exploración** + interruptor de prototipo para **apagar la exploración** |

Y cada persona trae: cuaderno ejecutable con el agente completo, curva de aprendizaje y
resultado de evaluación, hiperparámetros y semillas exactos, tiempo de entrenamiento y una
explicación de qué funcionó, qué falló y qué queda incierto.

Archivos de esta carpeta:

| Archivo | Qué es |
|---|---|
| `LunarLander_DQN_AgustinAgente.ipynb` | Copia del cuaderno de la cátedra con el TODO 1 (agente) resuelto y la sección 11 con el experimento de exploración. Ya ejecutado, con todas las salidas. |
| `EXPLICACION_AgenteDQN.md` | Este documento. |
| `resultados/dqn_principal.pt` | Pesos de la red Q del agente principal (1200 episodios, semilla 42). |
| `resultados/dqn_principal_curvas.json` | Recompensa y longitud de cada episodio del entrenamiento principal, tiempo, ε final. |
| `resultados/exploracion/<config>_s<semilla>.json` | Una corrida del experimento de exploración: curva, validaciones, recompensas de test y tiempo. |
| `resultados/configuracion.json` | Hiperparámetros, semillas, tiempos y versiones usadas. |
| `aterrizaje_dqn.gif` | Los 10 episodios del GIF de la sección 9. |

---

## 2. El problema: LunarLander-v3

Una nave tiene que posarse sobre la plataforma ubicada en `(0, 0)`.

**Estado** (8 números reales): posición `x, y`, velocidades `vx, vy`, ángulo `θ`, velocidad
angular `ω`, y dos indicadores 0/1 de si cada pata toca el suelo.

**Acciones** (4, discretas): `0` nada, `1` motor lateral izquierdo, `2` motor principal,
`3` motor lateral derecho.

**Recompensa** por paso: además de lo que dice el cuaderno (+10 por pata apoyada, −0,3 por
encender el principal, −0,03 por un lateral), Gymnasium da una recompensa de "forma"
(*shaping*): sube cuando la nave se acerca a la plataforma, baja la velocidad y se endereza, y baja
en el caso contrario. Al final, +100 si se posa quieta, −100 si se estrella o se va de la pantalla.

**Éxito**: un episodio con recompensa total ≥ 200. La consigna pide **más de 10 % de éxito**
evaluando sin exploración.

**Terminado vs truncado** (importante para el código):

* `terminado = True` → el episodio terminó *de verdad* (aterrizó o se estrelló). No hay futuro:
  el valor del estado siguiente es 0.
* `truncado = True` → se cortó a los 1000 pasos por reloj. La nave seguía volando: el futuro
  existe, solo que no lo vimos. Por eso el agente recibe solo `terminado` y **sí** usa el valor del
  estado siguiente cuando se trunca.

---

## 3. Teoría: de aprendizaje por refuerzos a DQN

### 3.1 El marco: un proceso de decisión de Markov (MDP)

En cada paso `t` el agente ve un estado $s_t$, elige una acción $a_t$, el entorno le devuelve una
recompensa $r_t$ y un nuevo estado $s_{t+1}$. El objetivo es maximizar el **retorno descontado**

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$$

con $0 \le \gamma < 1$. Con $\gamma = 0{,}99$ una recompensa que llega dentro de 100 pasos pesa
$0{,}99^{100} \approx 0{,}37$: el agente "mira" a unos cientos de pasos, que es la escala de un
aterrizaje.

### 3.2 La función Q

$Q^\pi(s,a)$ es el retorno esperado si en $s$ hago $a$ y después sigo la política $\pi$. La
**Q óptima** $Q^*$ cumple la **ecuación de Bellman de optimalidad**:

$$Q^*(s,a) = \mathbb{E}\left[\, r + \gamma \max_{a'} Q^*(s',a') \,\right]$$

"Lo que vale hacer $a$ en $s$ es lo que gano ahora más lo mejor que puedo conseguir desde donde
caigo". Si conozco $Q^*$, la política óptima es simplemente $\pi^*(s) = \arg\max_a Q^*(s,a)$.

### 3.3 Q-learning

Q-learning aprende $Q^*$ sin modelo del entorno, a partir de transiciones $(s,a,r,s')$:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\big[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{objetivo } y} - Q(s,a)\big]$$

Es *off-policy*: aprende la política greedy aunque las acciones las haya elegido una política
exploratoria. Eso es lo que permite reutilizar transiciones viejas (memoria de repetición).

### 3.4 ¿Por qué una red? (aproximación de funciones)

En LunarLander el estado es continuo: no hay tabla posible. Se aproxima $Q(s,a) \approx Q(s,a;\theta)$
con una red neuronal de parámetros $\theta$ que recibe $s$ y devuelve **los 4 valores Q a la vez**.
Se entrena minimizando el error entre la predicción y el objetivo de Bellman:

$$L(\theta) = \mathbb{E}\Big[\ \ell\big(y - Q(s,a;\theta)\big)\Big], \qquad y = r + \gamma\,(1-\text{terminado}) \max_{a'} Q(s',a';\theta^-)$$

### 3.5 Las tres piezas de DQN (Mnih et al., 2015)

Hacer Q-learning con una red "a secas" es inestable (la llamada *tríada mortal*: aproximación de
funciones + bootstrapping + off-policy). DQN agrega:

1. **Memoria de repetición (*experience replay*)**. Las transiciones consecutivas están muy
   correlacionadas (el paso 101 se parece al 100). Entrenar en orden viola la suposición i.i.d.
   del descenso por gradiente estocástico y hace que la red "olvide" lo que no está viendo ahora.
   Guardar las transiciones en una memoria grande y entrenar con **minilotes sorteados al azar**
   rompe esa correlación y reutiliza cada experiencia muchas veces.
2. **Red objetivo**. Si el objetivo $y$ se calcula con la misma red que se está entrenando, cada
   paso de gradiente mueve también el objetivo: es perseguir la propia cola. Se usa una copia
   $\theta^-$ que cambia lentamente. El artículo la copia entera cada $C$ pasos; acá se usa la
   versión **suave** $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ con $\tau = 0{,}001$, que es
   una copia gradual (en promedio, $\theta^-$ va "unos mil pasos de gradiente atrás").
3. **Recorte del error**. El artículo recorta el error a $[-1, 1]$; equivale a usar la **pérdida de
   Huber**: cuadrática para errores chicos, lineal para errores grandes. Así un objetivo raro
   (por ejemplo, un −100 por choque) no produce un gradiente gigante.

A eso se le suma la **exploración ε-greedy**, que es el tema del enfoque C.

### 3.6 Diferencias con el artículo original y por qué

| Artículo (Atari) | Este agente | Motivo |
|---|---|---|
| Entrada: 4 imágenes 84×84, red convolucional | Entrada: 8 números, MLP 128×128 | El estado ya es un vector compacto |
| RMSProp | Adam | Más robusto a la escala de los gradientes, estándar hoy |
| Red objetivo copiada cada 10 000 pasos | Actualización suave τ = 0,001 cada paso de gradiente | Más estable en problemas chicos; es la misma idea |
| Memoria de 1 000 000 | 100 000 | Entrenamientos de ~10⁵–10⁶ pasos |
| ε lineal de 1 a 0,1 en 10⁶ cuadros | Se comparan dos esquemas (enfoque C) | Es justamente lo que se estudia |
| Recorte del error | Pérdida de Huber + recorte de la norma del gradiente a 10 | Equivalente + una protección extra |

---

## 4. Teoría del enfoque C: exploración, semillas y evaluación

### 4.1 El dilema exploración–explotación

Si el agente siempre elige la acción que hoy cree mejor (**explotar**), nunca prueba las otras y
nunca descubre que eran mejores: sus estimaciones Q de acciones no probadas no se corrigen. Si
siempre elige al azar (**explorar**), aprende sobre todo pero nunca aprovecha lo aprendido y además
solo ve estados "de principiante" (por ejemplo, nunca llega cerca del suelo con la nave
estabilizada).

**ε-greedy**: con probabilidad ε acción al azar, si no la de mayor Q. Se arranca con ε = 1 (todo
al azar: no sabemos nada) y se baja hasta un piso chico (0,01) a medida que la red mejora.

### 4.2 Los dos esquemas comparados

**Lineal por pasos** (`exploracion='lineal'`):

$$\varepsilon(t) = \varepsilon_0 + \min\left(1, \tfrac{t}{T}\right)(\varepsilon_f - \varepsilon_0), \quad \varepsilon_0 = 1,\ \varepsilon_f = 0{,}01,\ T = 50\,000 \text{ pasos}$$

| pasos $t$ | 0 | 10 000 | 25 000 | 50 000 | 80 000 |
|---|---|---|---|---|---|
| ε | 1,00 | 0,80 | 0,505 | 0,01 | 0,01 |

**Exponencial por episodio** (`exploracion='exponencial'`):

$$\varepsilon_{k+1} = \max(\varepsilon_f,\ 0{,}995\cdot\varepsilon_k) \quad\Rightarrow\quad \varepsilon_k = 0{,}995^k$$

| episodio $k$ | 0 | 100 | 200 | 460 | 600 | 919+ |
|---|---|---|---|---|---|---|
| ε | 1,00 | 0,61 | 0,37 | 0,10 | 0,05 | 0,01 |

La diferencia clave no es la forma de la curva sino **en qué unidad se mide el tiempo**. Al
principio los episodios son cortos (la nave cae en ~100 pasos), así que 50 000 pasos son solo
~200–250 episodios: el esquema lineal deja de explorar muy temprano, justo cuando los episodios
empiezan a alargarse (la nave aprende a flotar y aparecen episodios de 1000 pasos). El exponencial
mantiene ε ≈ 0,1–0,3 durante cientos de episodios, el período en que el agente descubre cómo
aterrizar.

**Sin exploración** (`exploracion='ninguna'`, el interruptor): ε = 0 siempre. Es la "ablación" de la
exploración. Ojo: el agente no es totalmente determinista en lo que ve, porque cada episodio
arranca en una posición y con un empujón al azar del entorno, y la red inicial tiene pesos al azar.
Esa es toda la variedad de experiencias que tiene.

### 4.3 Semillas y reproducibilidad

Un entrenamiento de DQN usa azar en **seis** lugares, y hay que fijar todos:

| Fuente de azar | Dónde se fija |
|---|---|
| `random` de Python | `random.seed(semilla)` en el constructor |
| NumPy global | `np.random.seed(semilla)` |
| Inicialización de pesos de PyTorch | `torch.manual_seed(semilla)` antes de crear las redes |
| Acción al azar de ε-greedy (`action_space.sample()`) | `accion_space.seed(semilla)` — **no lo alcanza** `np.random.seed` |
| Moneda de ε y sorteo del minilote | `self.rng = np.random.default_rng(semilla)`, un generador propio del agente |
| Estado inicial de cada episodio | `env.reset(seed=...)` en `entrenar_agente` / `evaluar_agente` |

Además se fija `torch.set_num_threads(1)`: con varios hilos algunas sumas en punto flotante pueden
hacerse en otro orden y dar resultados distintos en el último decimal, que en RL se amplifican.

La sección 11.1 lo **verifica**: dos corridas con semilla 7 dan exactamente las mismas recompensas
y los mismos pesos; con semilla 8 dan otras.

¿Por qué un generador propio (`self.rng`) y no `np.random`? Porque así el agente no depende de que
nadie más use `np.random` entre medio (por ejemplo una celda que sortea semillas del GIF):
el flujo de números aleatorios del agente es solo suyo.

### 4.4 Evaluación honesta: tres juegos de semillas

| Uso | Semillas de `env.reset` | Para qué |
|---|---|---|
| Entrenamiento (experimento) | `100 000 + 1000·semilla + episodio` | Cada semilla del agente ve escenarios distintos; no se pisa con lo demás |
| Validación | 50 000 … 50 009 | Curva greedy durante el entrenamiento; sirve para mirar y elegir |
| Test | 10 000 … 10 099 (y 10 000 … 10 199 en la sección 8) | Número final; no se usa para decidir nada |

Es la misma lógica que train/validación/test en aprendizaje supervisado: si elijo la configuración
mirando el test, el número de test deja de ser una estimación imparcial.

La evaluación se hace **sin exploración** (`explorar=False`): lo que se quiere medir es la política
aprendida, no el ruido de ε.

### 4.5 La unidad de análisis es la semilla, no el episodio

Si comparo dos configuraciones con 1 semilla cada una y 100 episodios de test, tengo 100 números
por lado, pero **no son independientes respecto de lo que quiero comparar**: todos vienen de la
misma red, que pudo haber tenido suerte o mala suerte en el entrenamiento. La variabilidad
importante en RL es **entre entrenamientos**. Por eso:

* cada configuración se entrena con 5 semillas (0–4),
* cada agente se resume con su recompensa media en las 100 semillas de test,
* el test estadístico compara esos 5 números contra los 5 de la otra configuración.

### 4.6 El test estadístico: U de Mann-Whitney

Compara dos grupos independientes sin suponer normalidad. Idea: junto los 10 valores, los ordeno
y miro si los de un grupo tienden a quedar arriba. $U$ cuenta, entre los $5\times5 = 25$ pares
(uno de cada grupo), cuántas veces gana el grupo A. Si A gana los 25, $U = 25$; si no hay
diferencia se espera $U \approx 12{,}5$.

Ejemplo: A = {210, 205, 230, 190, 220}, B = {120, 60, 150, 100, 170}. Cada valor de A supera a cada
valor de B ⇒ $U = 25$. Hay $\binom{10}{5} = 252$ formas de repartir 10 rangos en dos grupos de 5,
y solo 2 son tan extremas (A todo arriba o B todo arriba) ⇒ $p = 2/252 \approx 0{,}008$.
Ese es el p-valor más chico posible con 5 contra 5: **solo una separación completa o casi completa
sale significativa al 5 %**.

¿Por qué no un t-test? El t de Student supone normalidad, que con 5 valores no se puede
comprobar, y las recompensas de LunarLander son bimodales (aterriza ≈ +250, se estrella ≈ −100).
Se reporta el t de Welch como referencia, pero la conclusión se apoya en Mann-Whitney.

---

## 5. El código del agente, línea por línea

El código completo está en la celda de la sección 6 del cuaderno. Acá va por bloques.

### 5.1 Cabecera e interfaz

```python
class AgenteDQN(Agente):
```
Hereda de `Agente`, la clase abstracta de la cátedra. Eso obliga a implementar `elegir_accion` y
`aprender`; `fin_episodio` es opcional y acá se usa.

### 5.2 La red Q (clase anidada)

```python
    class RedQ(nn.Module):
        def __init__(self, dim_estado, n_acciones, capas_ocultas):
            super().__init__()
```
La letra pide que todo lo que necesita el agente esté **dentro de la clase**, por eso la red es una
clase anidada. `nn.Module` es la clase base de PyTorch para redes; `super().__init__()` la inicializa
(sin esto PyTorch no registra los parámetros).

```python
            capas = []
            entrada = dim_estado
            for salida in capas_ocultas:
                capas.append(nn.Linear(entrada, salida))
                capas.append(nn.ReLU())
                entrada = salida
```
Arma la lista de capas. Con `capas_ocultas=(128, 128)` queda:
`Linear(8→128), ReLU, Linear(128→128), ReLU`. `nn.Linear(a, b)` calcula $Wx + b$ con $W$ de
$b\times a$. `ReLU` ($\max(0, x)$) es la no linealidad; sin ella, varias capas lineales seguidas
serían equivalentes a una sola. `entrada = salida` encadena: la salida de una capa es la entrada
de la siguiente.

```python
            capas.append(nn.Linear(entrada, n_acciones))   # salida lineal
            self.red = nn.Sequential(*capas)
```
Última capa: 128 → 4, **sin** activación, porque los valores Q pueden ser negativos y de cualquier
tamaño. `nn.Sequential` aplica las capas en orden.

Cantidad de parámetros: $(8\cdot128+128) + (128\cdot128+128) + (128\cdot4+4) = 1152 + 16\,512 + 516 = 18\,180$.

```python
        def forward(self, x):
            return self.red(x)
```
`forward` define qué pasa cuando se llama `red_q(x)`: entra un lote de estados de forma `(N, 8)`,
sale `(N, 4)`, un Q por acción.

### 5.3 El constructor: firma e hiperparámetros

```python
    def __init__(self, estado_space, accion_space, semilla=None,
                 capas_ocultas=(128, 128), lr=5e-4, gamma=0.99,
                 tam_buffer=100_000, tam_lote=64, min_buffer=1_000,
                 entrenar_cada=4, tau=1e-3,
                 exploracion='lineal', eps_inicio=1.0, eps_fin=0.01,
                 pasos_decaimiento=50_000, factor_decaimiento=0.995):
```
Los tres primeros argumentos son los que usa la cátedra (`AgenteDQN(env.observation_space,
env.action_space, semilla=SEMILLA)`). El resto son hiperparámetros con valores por defecto, que en
el cuaderno se pasan siempre explícitos con `**HIPERPARAMETROS`. `exploracion` es el interruptor
del enfoque C.

### 5.4 Semillas

```python
        if semilla is not None:
            random.seed(semilla)
            np.random.seed(semilla)
            torch.manual_seed(semilla)
            accion_space.seed(semilla)
        self.rng = np.random.default_rng(semilla)
```
Ver sección 4.3. `torch.manual_seed` va **antes** de crear las redes porque los pesos iniciales se
sortean al construir cada `nn.Linear`. `default_rng(None)` (sin semilla) toma entropía del sistema,
así que el agente sin semilla funciona igual, solo que no es reproducible.

### 5.5 Datos del problema y del aprendizaje

```python
        self.accion_space = accion_space
        self.dim_estado = estado_space.shape[0]     # 8
        self.n_acciones = accion_space.n            # 4
```
Se leen del entorno en vez de escribir 8 y 4 a mano.

```python
        self.gamma = gamma
        self.tam_lote = tam_lote
        self.min_buffer = max(min_buffer, tam_lote)
        self.entrenar_cada = entrenar_cada
        self.tau = tau
```
`min_buffer`: no se entrena hasta tener 1000 transiciones. Entrenar con 64 transiciones casi
idénticas al principio sobreajusta la red a esas pocas experiencias. El `max` garantiza que haya al
menos un minilote completo.

### 5.6 Las dos redes y el optimizador

```python
        self.red_q = self.RedQ(self.dim_estado, self.n_acciones, capas_ocultas)
        self.red_objetivo = self.RedQ(self.dim_estado, self.n_acciones, capas_ocultas)
        self.red_objetivo.load_state_dict(self.red_q.state_dict())
        self.red_objetivo.eval()
        self.optimizador = optim.Adam(self.red_q.parameters(), lr=lr)
```
* `red_q` ($\theta$) es la que se entrena y la que decide acciones.
* `red_objetivo` ($\theta^-$) solo calcula objetivos. `load_state_dict` copia los pesos para que
  arranquen iguales. `.eval()` la marca como "modo evaluación" (no cambia nada en un MLP sin dropout
  ni batchnorm, pero documenta la intención).
* El optimizador recibe **solo** los parámetros de `red_q`: la red objetivo nunca se actualiza por
  gradiente, solo por la copia suave.

### 5.7 La memoria de repetición

```python
        self.tam_buffer = tam_buffer
        self.buf_estados = np.zeros((tam_buffer, self.dim_estado), dtype=np.float32)
        self.buf_acciones = np.zeros(tam_buffer, dtype=np.int64)
        self.buf_recompensas = np.zeros(tam_buffer, dtype=np.float32)
        self.buf_siguientes = np.zeros((tam_buffer, self.dim_estado), dtype=np.float32)
        self.buf_terminados = np.zeros(tam_buffer, dtype=np.float32)
        self.pos_buffer = 0
        self.n_buffer = 0
```
En lugar de un `deque` de tuplas se usan **cinco arreglos preasignados**, uno por campo de la
transición. Ventajas: sortear un minilote es una sola indexación (`buf_estados[idx]`), sin bucles
de Python, y convertirlo a tensor es inmediato. Memoria: 100 000 × (8+8+3) números ≈ 8 MB.

* `float32` porque PyTorch trabaja en float32.
* `int64` para las acciones porque `gather` exige índices enteros largos.
* `terminados` en float (0.0 / 1.0) para poder multiplicar directamente en la fórmula del objetivo.

**Es un buffer circular**: `pos_buffer` dice dónde escribir; al llegar al final vuelve a 0 y pisa
lo más viejo. Ejemplo con capacidad 5:

| transición nº | posición donde se escribe | `n_buffer` |
|---|---|---|
| 1, 2, 3, 4, 5 | 0, 1, 2, 3, 4 | 1 … 5 |
| 6 | 0 (pisa la nº 1) | 5 |
| 7 | 1 (pisa la nº 2) | 5 |

### 5.8 Configuración de la exploración

```python
        if exploracion not in ('lineal', 'exponencial', 'ninguna'):
            raise ValueError(f"exploracion desconocida: {exploracion!r}")
        self.exploracion = exploracion
        self.eps_inicio = eps_inicio
        self.eps_fin = eps_fin
        self.pasos_decaimiento = pasos_decaimiento
        self.factor_decaimiento = factor_decaimiento
        self.epsilon = 0.0 if exploracion == 'ninguna' else eps_inicio
```
Validar el nombre evita que un error de tipeo (`'exponecial'`) entrene silenciosamente sin
decaimiento. Con `'ninguna'` ε arranca y queda en 0. El atributo se llama `epsilon` porque
`entrenar_agente` lo imprime con `getattr(agente, 'epsilon', 0.0)`.

### 5.9 Contadores

```python
        self.pasos = 0
        self.episodios = 0
        self.actualizaciones = 0
```
`pasos` lo usa el esquema lineal y la regla "entrenar cada 4"; los otros dos son informativos.

### 5.10 `elegir_accion`: ε-greedy

```python
    def elegir_accion(self, estado, explorar=True):
        if explorar and self.rng.random() < self.epsilon:
            return int(self.accion_space.sample())
```
`self.rng.random()` es un número uniforme en [0, 1); es menor que ε con probabilidad ε. Si
`explorar=False` (evaluación, GIF) la condición es falsa siempre y nunca se explora, sin importar ε.

```python
        with torch.no_grad():
            q = self.red_q(torch.as_tensor(estado, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())
```
* `torch.no_grad()`: no construir el grafo de gradientes (no vamos a hacer `backward`), más rápido.
* `torch.as_tensor(...)` convierte el arreglo de 8 números en tensor; `unsqueeze(0)` lo vuelve un
  lote de tamaño 1, forma `(1, 8)`, porque la red espera lotes.
* `q` tiene forma `(1, 4)`; `argmax(dim=1)` da el índice de la acción de mayor Q; `.item()` lo
  saca como número de Python e `int(...)` asegura el tipo que espera Gymnasium.

### 5.11 `aprender`: guardar, actualizar ε, entrenar

```python
    def aprender(self, estado, accion, recompensa, siguiente_estado, terminado):
        i = self.pos_buffer
        self.buf_estados[i] = estado
        self.buf_acciones[i] = accion
        self.buf_recompensas[i] = recompensa
        self.buf_siguientes[i] = siguiente_estado
        self.buf_terminados[i] = float(terminado)
        self.pos_buffer = (i + 1) % self.tam_buffer
        self.n_buffer = min(self.n_buffer + 1, self.tam_buffer)
        self.pasos += 1
```
Escribe la transición en la posición `i`, avanza el puntero circular con `% tam_buffer` y cuenta
cuántas válidas hay (tope: la capacidad). `terminado` llega desde `entrenar_agente` **sin** mezclar
con `truncado` (sección 2).

```python
        if self.exploracion == 'lineal':
            fraccion = min(1.0, self.pasos / self.pasos_decaimiento)
            self.epsilon = self.eps_inicio + fraccion * (self.eps_fin - self.eps_inicio)
```
Esquema lineal: `fraccion` va de 0 a 1 en 50 000 pasos y después queda en 1. Ejemplo: en el paso
25 000, `fraccion = 0,5` y ε = 1 + 0,5·(0,01 − 1) = 0,505.

```python
        if self.pasos % self.entrenar_cada == 0 and self.n_buffer >= self.min_buffer:
            self._paso_de_gradiente()
```
Un paso de gradiente cada 4 pasos de entorno (como en el artículo). Entrenar en cada paso
cuadruplica el costo sin mejorar mucho, porque los datos nuevos entre paso y paso son casi iguales.
Con un minilote de 64 cada 4 pasos, cada transición se usa en promedio 16 veces mientras está en
la memoria.

### 5.12 `_paso_de_gradiente`: el corazón de DQN

```python
        idx = self.rng.integers(0, self.n_buffer, size=self.tam_lote)
```
Sortea 64 índices uniformes entre las transiciones válidas (con reemplazo, como el artículo).

```python
        estados = torch.from_numpy(self.buf_estados[idx])
        acciones = torch.from_numpy(self.buf_acciones[idx])
        recompensas = torch.from_numpy(self.buf_recompensas[idx])
        siguientes = torch.from_numpy(self.buf_siguientes[idx])
        terminados = torch.from_numpy(self.buf_terminados[idx])
```
Arma el minilote. Formas: `estados (64, 8)`, `acciones (64,)`, `recompensas (64,)`,
`siguientes (64, 8)`, `terminados (64,)`.

```python
        q_actual = self.red_q(estados).gather(1, acciones.unsqueeze(1)).squeeze(1)
```
`self.red_q(estados)` da `(64, 4)`: los 4 Q de cada estado. Solo interesa el Q de la acción que
realmente se tomó. `acciones.unsqueeze(1)` → `(64, 1)`; `gather(1, ...)` toma, en cada fila, la
columna indicada → `(64, 1)`; `squeeze(1)` → `(64,)`. Ejemplo de una fila: Q = [3,1, 5,0, 4,2, 2,7]
y acción 2 ⇒ se toma 4,2.

```python
        with torch.no_grad():
            q_siguiente = self.red_objetivo(siguientes).max(dim=1).values
            objetivo = recompensas + self.gamma * (1.0 - terminados) * q_siguiente
```
El objetivo de Bellman $y = r + \gamma (1 - \text{terminado}) \max_{a'} Q(s', a'; \theta^-)$.
* Se calcula con la **red objetivo** (estabilidad, sección 3.5).
* `no_grad`: el objetivo es la "etiqueta"; no se propaga gradiente a través de él.
* `(1 − terminados)` anula el término futuro si el episodio terminó de verdad.

```python
        perdida = F.smooth_l1_loss(q_actual, objetivo)
```
Pérdida de Huber (con β = 1): $\ell(\delta) = \tfrac12\delta^2$ si $|\delta|<1$, y $|\delta| - \tfrac12$ si no.
Su derivada está acotada a [−1, 1], que es exactamente el "recorte del error" del artículo. Se
promedia sobre los 64 ejemplos.

```python
        self.optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(self.red_q.parameters(), 10.0)
        self.optimizador.step()
        self.actualizaciones += 1
```
El ciclo estándar de PyTorch: borrar gradientes viejos (si no, se acumulan), calcular gradientes
por retropropagación, **recortar** la norma total del gradiente a 10 (protección extra contra pasos
gigantes) y aplicar el paso de Adam.

```python
        with torch.no_grad():
            for p_obj, p in zip(self.red_objetivo.parameters(), self.red_q.parameters()):
                p_obj.mul_(1.0 - self.tau).add_(self.tau * p)
```
Actualización suave: $\theta^- \leftarrow (1-\tau)\theta^- + \tau\theta$, parámetro por parámetro, in place
(los métodos con `_` modifican el tensor). Ejemplo con un solo peso: $\theta = 1$, $\theta^- = 0$,
τ = 0,001 → tras 1 paso $\theta^- = 0{,}001$; tras $k$ pasos $\theta^- = 1 - 0{,}999^k$; llega a la mitad
en $k \approx 693$ pasos de gradiente (≈ 2800 pasos de entorno). Es la "memoria" de la red objetivo.

### 5.13 `fin_episodio`: esquema exponencial

```python
    def fin_episodio(self):
        self.episodios += 1
        if self.exploracion == 'exponencial':
            self.epsilon = max(self.eps_fin, self.epsilon * self.factor_decaimiento)
```
`entrenar_agente` lo llama al terminar cada episodio. El esquema exponencial vive acá porque mide
el tiempo en **episodios**; el lineal vive en `aprender` porque lo mide en **pasos**. Con `'ninguna'`
no se toca ε (queda en 0).

### 5.14 La prueba rápida al final de la celda

```python
env = gym.make('LunarLander-v3')
_prueba = AgenteDQN(env.observation_space, env.action_space, semilla=SEMILLA)
_estado, _ = env.reset(seed=SEMILLA)
print("Acción greedy:", _prueba.elegir_accion(_estado, explorar=False), ...)
```
Solo verifica que el agente se construye y elige una acción válida; imprime los 18 180 parámetros.

---

## 6. Ejemplo numérico completo de un paso de aprendizaje

Supongamos una transición en mitad del entrenamiento:

* Estado $s$: la nave baja algo rápido sobre la plataforma.
* Acción $a = 2$ (motor principal).
* Recompensa $r = 1{,}7$ (el *shaping* premia frenar, menos 0,3 por usar el motor).
* `terminado = False`.

La **red objetivo** evalúa el estado siguiente: $Q(s', \cdot;\theta^-) = [10{,}0;\ 12{,}5;\ 14{,}0;\ 11{,}0]$.
El máximo es 14,0 (seguir usando el motor principal).

**Objetivo**: $y = 1{,}7 + 0{,}99 \cdot (1 - 0) \cdot 14{,}0 = 1{,}7 + 13{,}86 = 15{,}56$.

La **red Q** predecía $Q(s, 2;\theta) = 13{,}0$. Error $\delta = 13{,}0 - 15{,}56 = -2{,}56$.

**Huber**: como $|\delta| = 2{,}56 > 1$, $\ell = 2{,}56 - 0{,}5 = 2{,}06$ y la derivada respecto de la
predicción es $-1$ (con error cuadrático sería $-2{,}56$: Huber limita el empujón).

El gradiente mueve $\theta$ para **subir** $Q(s, 2)$ hacia 15,56. Solo se corrige la salida de la
acción 2; las otras tres salidas no reciben gradiente directo en este ejemplo (pero sí cambian un
poco porque comparten las capas ocultas).

**Caso terminal**: si en cambio la nave se estrella, $r = -100$, `terminado = True` ⇒
$y = -100 + 0{,}99\cdot 0 \cdot (\dots) = -100$. No importa lo que diga la red objetivo sobre $s'$.

**Caso truncado** (paso 1000, la nave flotando): `terminado = False` aunque el episodio se cortó ⇒
se usa $\gamma \max Q(s', \cdot)$ normalmente, porque la nave podría haber seguido.

Con 64 transiciones así, la pérdida es el promedio de los 64 $\ell$, y un único paso de Adam
corrige todo el minilote a la vez.

---

## 7. El resto del cuaderno, celda por celda

Las secciones 1 a 5, 8 y 9 son de la cátedra y no se tocaron, salvo:

* **Celda de instalación**: agrega `scipy` (tests estadísticos) y `joblib` (paralelismo).
* **Celda de imports**: agrega `contextlib`, `io`, `json`, `Path`, `Parallel/delayed`, `stats`;
  fija `torch.set_num_threads(1)`; crea la carpeta `resultados/`.

### 7.1 Sección 6.1: `HIPERPARAMETROS` y `EXPLORACION_PRINCIPAL`

Un diccionario único con todos los hiperparámetros, que se pasa con `**HIPERPARAMETROS` a cada
agente. Así es imposible que el entrenamiento principal y el experimento usen valores distintos sin
querer. `EXPLORACION_PRINCIPAL = 'exponencial'` se eligió con un prototipo de una semilla y lo
confirman las curvas de validación de la sección 11 (ver sección 9.3).

### 7.2 Sección 7: entrenamiento principal con caché

```python
REENTRENAR = False
RUTA_PESOS = CARPETA_RESULTADOS / 'dqn_principal.pt'
RUTA_CURVAS = CARPETA_RESULTADOS / 'dqn_principal_curvas.json'
```
Si los archivos existen y `REENTRENAR` es `False`, se cargan los pesos (`load_state_dict`) y las
curvas en vez de reentrenar 1200 episodios. Si no, se entrena con `entrenar_agente` (sin
modificar), se mide el tiempo con `time.time()` y se guarda todo. La letra sugiere justamente
"guardar los resultados en disco para poder re-analizarlos sin reentrenar".

### 7.3 Sección 10

El TODO 2 (cortes de la política) es trabajo grupal y queda pendiente; la celda imprime un aviso
en vez de cortar la ejecución, para que el cuaderno corra entero.

### 7.4 Sección 11.1: `huella_entrenamiento`

Crea un agente nuevo, lo entrena 20 episodios en silencio (`contextlib.redirect_stdout` tira lo que
imprime `entrenar_agente`) y devuelve las recompensas y todos los pesos aplanados en un vector
(`torch.cat` de cada parámetro `.flatten()`). Se llama con semillas 7, 7 y 8, y un `assert`
garantiza que las dos primeras son idénticas. Si alguien rompe la reproducibilidad (por ejemplo,
borra `accion_space.seed`), el cuaderno falla ahí.

### 7.5 Sección 11.2: `entrenar_con_validacion`

```python
for inicio in range(0, num_episodios, cada):
    n = min(cada, num_episodios - inicio)
    with contextlib.redirect_stdout(io.StringIO()):
        r, l = entrenar_agente(agente, env, num_episodios=n, verbose=False,
                               semilla=semilla + inicio)
```
Entrena por bloques de 50 episodios **reusando** `entrenar_agente`. Como esa función hace
`env.reset(seed=semilla + episodio)` con el episodio contado desde 0 en cada llamada, pasarle
`semilla + inicio` hace que el episodio global `inicio + j` use la semilla `semilla + inicio + j`:
idéntico a entrenar de un tirón. El agente conserva su estado (memoria, ε, contadores) entre
bloques porque es el mismo objeto.

```python
    val = evaluar_agente(agente, env, num_episodios=episodios_val, semilla=SEMILLA_VALIDACION)
    validacion.append({'episodio': ..., 'recompensa': ..., 'tasa_exito': ..., 'epsilon': agente.epsilon})
```
Después de cada bloque, 10 episodios greedy sobre las semillas de validación. Se guarda también ε
para graficar el esquema.

### 7.6 Sección 11.3: el experimento en paralelo

* `CONFIGURACIONES`: lo único que cambia entre variantes es `exploracion`; se combina con
  `{**HIPERPARAMETROS, **CONFIGURACIONES[nombre]}` (el segundo diccionario pisa al primero).
* `correr_configuracion(nombre, semilla)`: crea entorno y agente, entrena 600 episodios con
  validación, mide el tiempo, evalúa en test (100 episodios, semillas 10 000+) y devuelve un
  diccionario de tipos simples (listas, floats) para poder guardarlo en JSON.
* `pendientes`: solo las combinaciones cuyo JSON no existe. Si se corta a la mitad, al volver a
  correr solo se entrena lo que falta.
* `Parallel(n_jobs=...)(delayed(f)(n, s) for ...)`: `joblib` lanza procesos separados (uno por
  núcleo) y reparte las 15 corridas. Cada proceso usa un hilo de PyTorch. Como cada corrida fija
  sus propias semillas, el resultado no depende del orden ni de en qué proceso cayó.

### 7.7 Sección 11.4: gráficas

* `media_movil` (de la cátedra) suaviza cada curva de entrenamiento; se promedia entre las 5
  semillas y se sombrea ± 1 desviación estándar.
* La curva de validación greedy muestra la política sin ruido de ε.
* La tercera gráfica muestra ε en cada esquema.

### 7.8 Sección 11.5: tabla y test

Imprime, para cada configuración, la recompensa media de test de cada semilla, su media, su
desviación estándar (con `ddof=1`, muestral), la tasa de éxito promedio y los minutos por corrida.
Después `stats.mannwhitneyu(..., alternative='two-sided')` y `stats.ttest_ind(..., equal_var=False)`
(Welch) contra `lineal`. La gráfica de puntos muestra cada semilla para que se vea la dispersión y
no solo el promedio.

### 7.9 Sección 11.6: tiempos y `configuracion.json`

Imprime minutos por corrida, el tiempo del entrenamiento principal, y una estimación de cuánto
costaría la ablación (4 configuraciones × 5 semillas) secuencial y en paralelo. Guarda todo lo
necesario para reproducir en `resultados/configuracion.json`.

---

## 8. Hiperparámetros y semillas exactos

| Hiperparámetro | Valor | Por qué |
|---|---|---|
| Capas ocultas | 128, 128 (ReLU) | Suficiente para 8 entradas; estándar en LunarLander |
| Optimizador / `lr` | Adam, 5·10⁻⁴ | Valor habitual para DQN en este entorno |
| γ | 0,99 | Horizonte de cientos de pasos |
| Memoria | 100 000 transiciones | Cubre los últimos ~100–300 episodios |
| Minilote | 64 | Estándar |
| `min_buffer` | 1 000 | No entrenar con datos casi idénticos al inicio |
| `entrenar_cada` | 4 | Como el artículo |
| τ (red objetivo) | 0,001 por paso de gradiente | Copia suave |
| Pérdida | Huber (`smooth_l1`), recorte de norma 10 | Estabilidad |
| ε inicial / final | 1,0 / 0,01 | |
| Lineal | 50 000 pasos | Esquema 1 |
| Exponencial | ×0,995 por episodio | Esquema 2 (**elegido**) |
| Episodios | 1200 (principal), 600 (experimento) | Presupuesto |

| Semillas | Valor |
|---|---|
| Agente principal | 42; `reset` de entrenamiento 42 … 1241 |
| Agentes del experimento | 0, 1, 2, 3, 4; `reset` 100 000 + 1000·semilla + episodio |
| Validación | 50 000 … 50 009 |
| Test sección 8 | 10 000 … 10 199 (200 episodios) |
| Test sección 11 | 10 000 … 10 099 (100 episodios) |
| GIF | 10 semillas sorteadas en [20 000, 30 000) con `default_rng(42)` |

Versiones: gymnasium 1.2.3, torch 2.10.0+cpu, Python 3.12, Windows 11, 8 núcleos, sin GPU.

---

## 9. Resultados

Todo lo que sigue sale de la ejecución guardada en el cuaderno (Windows 11, CPU de 8 núcleos, sin GPU).

### 9.1 Entrenamiento principal (exponencial, semilla 42, 1200 episodios)

**Tiempo: 10,1 minutos** (558 856 pasos de entorno, un solo proceso).

| Episodio | Recompensa media (100 ep) | Longitud media | ε |
|---|---|---|---|
| 120 | −128 | 132 | 0,55 |
| 240 | −30 | 529 | 0,30 |
| 300 | −20 | 839 | 0,22 |
| 360 | 61 | 787 | 0,17 |
| 480 | 203 | 502 | 0,09 |
| 600 | 219 | 506 | 0,05 |
| 900 | 228 | 441 | 0,01 |
| 1200 | **247** | 380 | 0,01 |

Cómo leer la curva (es la gráfica de la sección 7 del cuaderno):

1. **Episodios 0–150**: la nave cae y se estrella (~100 pasos, recompensa ≈ −150). Es casi todo
   exploración al azar.
2. **Episodios 150–350**: la nave aprende a **no estrellarse**: flota. Los episodios se alargan a
   800–1000 pasos (muchos se truncan) y la recompensa sube hacia 0. Es un "óptimo local" típico:
   flotar evita el −100.
3. **Episodios 350–500**: aprende a **bajar y posarse**. La recompensa pasa los 200 y los episodios
   se acortan (aterrizar termina el episodio).
4. **Episodios 500–1200**: refina. La caída alrededor del episodio 660 (188) es la inestabilidad
   habitual de DQN; se recupera sola.

Últimos 100 episodios de entrenamiento (todavía con ε = 0,01): media 246,7, **92 % de éxito**.

### 9.2 Evaluación sin exploración (sección 8, 200 episodios, semillas 10 000–10 199)

| Agente | Tasa de éxito (R ≥ 200) | Media | Mediana | Desv. estándar |
|---|---|---|---|---|
| **DQN** | **65,0 %** (130/200) | **218,0** | 237,6 | 58,3 |
| Aleatorio | 0,0 % | −183,2 | −149,8 | 112,0 |

* Cumple holgadamente el criterio de la letra (> 10 %).
* Desglose del DQN: 81 % ≥ 150, 99 % ≥ 100, **100 % ≥ 0**, y **ningún choque** (R < −50: 0 %).
  Los episodios que no llegan a 200 no son fracasos catastróficos: son aterrizajes "caros" (mucho
  combustible) o episodios en los que la nave queda flotando hasta el paso 1000.
* GIF (sección 9): 8 de 10 episodios exitosos, media 231. Los dos que fallan (151 y 118) duran
  1000 pasos: la nave se queda flotando cerca del suelo sin terminar de posarse.

¿Por qué 92 % en entrenamiento y 65 % en evaluación? Son escenarios distintos (semillas nuevas) y
el umbral de 200 es un corte duro: buena parte de la masa está entre 150 y 250, así que pequeñas
diferencias de escenario cruzan el umbral. La mediana (238) está por encima de 200.

### 9.3 Experimento de exploración (sección 11, 3 configuraciones × 5 semillas × 600 episodios)

**Tiempo de pared: 24,9 minutos** con 8 procesos en paralelo (unos 10–12 minutos por corrida).

Recompensa media de test (100 episodios, semillas 10 000–10 099, sin exploración) de cada semilla:

| Config | s0 | s1 | s2 | s3 | s4 | Media | Desv. | Éxito medio | Min/corrida |
|---|---|---|---|---|---|---|---|---|---|
| lineal | 99,5 | 177,1 | 99,9 | 80,7 | 178,1 | 127,1 | 46,8 | 40,6 % | 11,6 |
| **exponencial** | 213,3 | 169,2 | 137,1 | 175,5 | 190,6 | **177,1** | 28,1 | **48,0 %** | 9,7 |
| sin_exploracion | −6,5 | 0,4 | −29,3 | −27,3 | 93,2 | 6,1 | 50,3 | 4,6 % | 10,8 |

Tests contra `lineal` (n = 5 por grupo, bilaterales):

| Variante | Diferencia de medias | Mann-Whitney U | p (MW) | p (Welch) |
|---|---|---|---|---|
| exponencial | +50,0 | 19 | **0,22** | 0,08 |
| sin_exploracion | −121,0 | 1 | **0,016** | 0,004 |

Validación greedy (semillas 50 000+) durante el entrenamiento:

| Config | Media de los últimos 3 controles | Primer control con media ≥ 200 | Media últimos 100 ep. de entrenamiento |
|---|---|---|---|
| lineal | 92 | nunca (en 600 ep) | 8 / 163 / 114 / 31 / 181 |
| exponencial | 173 | episodios 350–500 (las 5 semillas) | 212 / 203 / 211 / 216 / 204 |
| sin_exploracion | −6 | nunca | −22 / −10 / −36 / −44 / 52 |

**Cómo se interpreta:**

* **Apagar la exploración es claramente peor** que explorar (p = 0,016 con Mann-Whitney: 24 de los
  25 pares semilla-contra-semilla los gana `lineal`). 4 de 5 semillas sin exploración no aterrizan
  nunca en test (0 % de éxito); quedan en el "óptimo local" de flotar. Esto es evidencia directa de
  que la exploración es un componente necesario de DQN en este problema, y deja lista una variante
  para la ablación.
* **Exponencial es mejor que lineal en promedio** (+50 puntos en test, curvas de validación y de
  entrenamiento consistentemente arriba; las 5 semillas exponenciales superan 200 de media en
  entrenamiento, las lineales ninguna), **pero con 5 semillas la diferencia en test no es
  estadísticamente significativa** (p = 0,22). No se puede afirmar con este experimento que un
  esquema sea mejor que el otro; solo que la tendencia favorece al exponencial.
* La explicación más probable: con `pasos_decaimiento = 50 000`, el lineal llega a ε = 0,01 hacia el
  episodio ~200 (ver la tercera gráfica de la sección 11.4), **antes** de que el agente haya
  aprendido a aterrizar. El exponencial sigue en ε ≈ 0,1–0,3 entre los episodios 250 y 450, justo
  cuando aparecen los aterrizajes.
* Un detalle de costo: el exponencial es la corrida **más barata** (9,7 min) porque aprende a
  aterrizar antes y los episodios se acortan; sin exploración es cara porque muchos episodios
  flotan hasta el paso 1000.

**Cómo se eligió el esquema principal.** La decisión de usar `exponencial` en el entrenamiento
principal se tomó con un prototipo previo de una semilla (a los 600 episodios: exponencial ≈ 212 de
media, lineal ≈ 119, sin exploración ≈ −60 a los 300), y la confirman las curvas de **validación**
del experimento. Los números de **test** no se usaron para elegir.

### 9.4 Presupuesto para la ablación

Con 600 episodios por corrida, unos 11 minutos por corrida cuando corren 8 en paralelo:

* 4 configuraciones (DQN completo + 3 variantes) × 5 semillas = 20 corridas
* secuencial ≈ 231 minutos; **con 8 procesos ≈ 35 minutos** (3 tandas).

Si se quieren más semillas para tener más potencia estadística (ver sección 10), 10 semillas
duplicarían el costo (≈ 70 minutos).

---

## 10. Qué funcionó, qué falló, qué queda incierto

**Funcionó**

* El agente completo resuelve el problema: **65 % de éxito** sin exploración (mínimo pedido: 10 %),
  media 218 contra −183 del aleatorio y ningún choque en 200 episodios de evaluación.
* El entrenamiento es **exactamente reproducible**: misma semilla ⇒ mismas recompensas y mismos
  pesos (verificado con `assert` en la sección 11.1).
* La separación entrenamiento / validación / test de semillas y la evaluación por semilla de
  entrenamiento quedan armadas para reutilizarlas en la ablación.
* El interruptor `exploracion='ninguna'` muestra un efecto grande y significativo.
* Correr en paralelo con `joblib` y guardar cada corrida en JSON hace que el experimento se pueda
  reanalizar sin reentrenar y retomar si se corta.

**Falló o anduvo peor de lo esperado**

* El esquema lineal con 50 000 pasos explora muy poco tiempo: en 600 episodios ninguna semilla llegó
  a 200 de media en validación. El valor de `pasos_decaimiento` se eligió sin mirar cuántos pasos
  dura cada episodio; fue una mala elección, no un defecto del esquema lineal en sí.
* Con 5 semillas el test no alcanza para separar `exponencial` de `lineal` (p = 0,22).
* En evaluación, el 35 % de los episodios no llega a 200; los que fallan suelen ser naves que flotan
  hasta el paso 1000 sin posarse.
* Primer intento de paralelizar: falló con "A task has failed to un-serialize" mientras corrían a la
  vez otros 3 entrenamientos; en una máquina sin otra carga anduvo. Lo más probable es falta de
  memoria; si vuelve a pasar, bajar `n_jobs`.

**Queda incierto**

* Si la ventaja del exponencial es real o ruido: haría falta más semillas (10 o más) o comparar
  contra un lineal con `pasos_decaimiento` más largo (por ejemplo 150 000), que probablemente
  empataría. La hipótesis es que **la duración de la exploración** importa más que la forma de la
  curva, pero no está probada.
* Cuánto varía el agente principal de 1200 episodios con otras semillas: solo se entrenó con la
  semilla 42. Las corridas de 600 episodios del experimento dan entre 19 % y 74 % de éxito en test
  según la semilla, así que la varianza entre semillas es grande.
* Si 600 episodios alcanzan para la ablación: a los 600 episodios los agentes todavía mejoran
  (el principal pasa de 219 a 247 entre los episodios 600 y 1200). Las variantes se compararían
  "a mitad de camino"; para la ablación es aceptable si todas usan el mismo presupuesto, pero hay
  que decirlo en el informe.
* Los demás hiperparámetros (red, `lr`, memoria, τ) no se calibraron; son valores estándar y les
  corresponden a los focos A y B.

---

## 11. Cómo correrlo

1. Abrir `LunarLander_DQN_AgustinAgente.ipynb` **desde esta carpeta** (las rutas `resultados/` son
   relativas).
2. Ejecutar todo. La primera celda instala dependencias.
3. Como `resultados/` ya trae los pesos y los JSON, no se reentrena nada: carga y grafica en un par
   de minutos. Para reentrenar desde cero: borrar `resultados/` o poner `REENTRENAR = True`
   (entrenamiento principal) y borrar `resultados/exploracion/` (experimento).

En Windows, si `Box2D` falla con "nombre de archivo demasiado largo", el entorno virtual está en una
ruta muy larga: crearlo en una ruta corta (por ejemplo `C:\Users\<usuario>\rlvenv`).

---

## 12. Preguntas probables y cómo responderlas

**¿Por qué DQN y no Q-learning tabular?** El estado es continuo de 8 dimensiones; una tabla
necesitaría discretizar y explotaría en tamaño. La red generaliza entre estados parecidos.

**¿Por qué la red devuelve 4 salidas en vez de recibir la acción como entrada?** Así con una sola
pasada se obtienen los 4 Q, que es lo que necesitan tanto el `argmax` (elegir acción) como el `max`
(objetivo).

**¿Por qué `terminado` y no `terminado or truncado`?** Truncar es un corte artificial por tiempo:
el estado siguiente tiene futuro. Si se tratara como terminal, el agente aprendería que flotar 1000
pasos "termina" en 0, lo cual es falso.

**¿Para qué la red objetivo?** Para que el objetivo $y$ no se mueva con cada paso de gradiente.
Sin ella la red persigue un blanco que ella misma desplaza y puede divergir.

**¿Para qué la memoria de repetición?** Rompe la correlación temporal entre muestras y reutiliza
experiencia (cada transición se usa ~16 veces).

**¿Por qué Huber?** Limita el gradiente de errores grandes (un −100 por choque no desestabiliza).

**¿Por qué el exponencial anda mejor que el lineal?** Porque mide el tiempo en episodios. El lineal
llega a ε = 0,01 en 50 000 pasos, que con episodios cortos al principio son ~250 episodios: deja de
explorar antes de que el agente haya visto aterrizajes. El exponencial sigue explorando
(ε ≈ 0,1–0,3) durante los episodios 250–450, que es cuando aparecen los primeros aterrizajes.
Esto **no** significa que "exponencial" sea mejor en general: un lineal con más pasos (por ejemplo
150 000) podría andar igual. Lo que muestra el experimento es que **cuánto tiempo se explora**
importa más que la forma de la curva.

**¿Por qué sin exploración aprende algo igual?** Porque el entorno ya mete variedad (posición y
empujón inicial al azar) y la red inicial al azar elige acciones "arbitrarias". Pero se queda
atascado: si la red cree que una acción es mala nunca la vuelve a probar y no corrige el error.

**¿Por qué 5 semillas y Mann-Whitney?** La varianza entre entrenamientos es grande en RL; la
unidad independiente es la semilla. Mann-Whitney no supone normalidad. Con 5 contra 5 el p mínimo
es 0,008, así que un p < 0,05 implica que casi todas las semillas de un grupo superan a casi todas
las del otro.

**¿Por qué no evaluar con los mismos 100 episodios del entrenamiento?** Porque la curva de
entrenamiento incluye exploración (ε) y escenarios que el agente ya vio; la evaluación greedy sobre
semillas nuevas mide la política real.

**¿Cómo sé que es reproducible?** La sección 11.1 entrena dos veces con la misma semilla y compara
recompensas y pesos con igualdad exacta; un `assert` corta el cuaderno si no coinciden.

---

# Parte 2: trabajo en el estudio del equipo (30/10 al 2/11)

Todo lo de esta parte está en la rama **`AgustinAblacion`**, que sale de `FranAblacion`
(la última versión del equipo, con el código integrado, el protocolo y las corridas de Fran).

| Archivo nuevo o modificado | Qué es |
|---|---|
| `protocols/team_draft.json` | Mi confirmación del protocolo y la propuesta estadística (campo `statistics`) |
| `scripts/test_ablacion.py` | El test propuesto, listo para que Fran lo use o lo integre en `ablation_stats.py` |
| `results/comparison_20261102T2108*` y `T2109*` | Mis 8 corridas sin exploración (una carpeta por semilla) |
| `scripts/analisis_politica.py` | Genera los cortes de política y las métricas de episodios reales |
| `analisis_politica/` | Las figuras y `metricas_politica.json` |
| `informe/informe.tex` + `informe/figuras/` | Plantilla IEEE para Overleaf (estructura, tablas y figuras; **sin texto redactado**) |

---

## 13. Dónde estamos: el estudio del equipo y en qué se diferencia de mi agente

El plan del equipo:

| Día | Mi tarea | Estado |
|---|---|---|
| 30/10 | Proponer semillas, evaluación y método estadístico | Hecho (sección 14) |
| 31/10 | 8 semillas sin exploración (+ 2 del completo) | Hechas las 8 (sección 15). Las 2 del completo ya las corrió Fran: no se repiten |
| 1/11 | Cortes de política y visualizaciones | Hecho (secciones 17 y 18) |
| 2/11 | Análisis de política, limitaciones, formato IEEE | Formato IEEE hecho; **el texto lo tengo que escribir yo** (sección 19) |

**El estudio usa el agente integrado (`lander.ipynb`), no el mío.** Thiago integró las tres
contribuciones (A, B y C) en un solo `AgenteDQN` con tres interruptores (`use_target`,
`use_replay`, `use_exploration`). La configuración de referencia es **A0**:

| | Mi agente (Parte 1) | A0, el del estudio |
|---|---|---|
| Red | 128 × 128 | 64 × 64 |
| `lr` | 5·10⁻⁴ | 10⁻³ |
| Memoria | 100 000, arreglos de NumPy, muestreo **con** reemplazo | 50 000, `deque`, `random.sample` (**sin** reemplazo) |
| Red objetivo | Suave, τ = 0,001 | **Copia completa cada 1000 pasos** (como el artículo) |
| Exploración | Exponencial por episodio (0,995) | **Lineal por episodio**: 1 → 0,05 en 200 episodios |
| Presupuesto | 1200 episodios | **300 episodios** |
| Evaluación | 200 ep., semillas 10 000+ | Igual: 200 ep., semillas 10 000–10 199 |

Lo que sí viene de mi foco: el interruptor `use_exploration`, los esquemas de ε alternativos
(`linear_steps`, `exponential_episodes`) y la idea de medir ε en **episodios**, que es justo
lo que usa A0. Mi experimento de la Parte 1 (sección 9.3) es la evidencia de calibración de
esa decisión.

**Cómo funciona `use_exploration=False` en el código integrado** (celda del agente en `lander.ipynb`):

```python
self.epsilon = epsilon_start if use_exploration else 0.0
...
if explorar and self.use_exploration and random.random() < self.epsilon:
    return int(self.accion_space.sample())
...
def epsilon_en(self, episodios, pasos):
    if not self.use_exploration:
        return 0.0
```

* ε arranca en 0 y `epsilon_en` (que actualiza ε al final de cada episodio) siempre devuelve 0.
* En `elegir_accion`, el `and self.use_exploration` hace que ni siquiera se tire la moneda: el
  agente es greedy siempre, también durante el entrenamiento.
* Todo lo demás (red, memoria, red objetivo, semillas) queda idéntico al completo: **la
  ablación cambia una sola cosa**.

---

## 14. 30/10: protocolo y método estadístico

### 14.1 Qué confirmé del protocolo

El protocolo (`protocols/team_draft.json`) ya lo había aprobado Fran. Lo confirmé tal cual:

* **Referencia A0**, 300 episodios de hasta 1000 pasos, sin parada temprana. Se evalúa el
  modelo final, no el mejor punto de la curva, para no favorecer a ninguna variante.
* **Semillas de entrenamiento**: 42, 1042, 2042, …, 7042 (8 por condición, las mismas para todas).
* **Evaluación**: greedy, sin aprender, 200 episodios en las semillas 10 000–10 199
  (*desarrollo*). Las semillas **30 000–30 199 quedan reservadas** y nadie las mira hasta que
  todo esté congelado.
* **Unidad estadística**: la recompensa media de evaluación de cada política. 200 episodios de
  una política no son 200 réplicas independientes; las réplicas son las semillas de entrenamiento.

El estado quedó en `approved_by_fran_and_agustin_pending_thiago`.

### 14.2 La propuesta estadística

Quedó escrita en el campo `statistics` del protocolo:

| Pieza | Elección |
|---|---|
| Test principal | **U de Mann-Whitney bilateral, exacto**: completo contra cada ablación, 8 contra 8 |
| Comparaciones múltiples | **Holm-Bonferroni** sobre las 3 comparaciones planeadas, α = 0,05 |
| Sensibilidad | Wilcoxon de rangos con signo emparejado por semilla (solo se reporta) |
| Tamaños de efecto | Diferencia de medias y de medianas, Hodges-Lehmann, probabilidad de superioridad |

**Por qué Mann-Whitney** (lo mismo que en la sección 4.6, ahora con 8 semillas):

* No supone normalidad. Con 8 valores no se puede verificar, y las recompensas son bimodales
  (políticas que aterrizan ≈ 200, políticas que se estrellan ≈ −150).
* Con 8 contra 8 hay $\binom{16}{8} = 12\,870$ formas de repartir los rangos; el p mínimo es
  2/12 870 ≈ 0,0002. Hay potencia para detectar separaciones claras aun después de corregir.
* Ya lo había elegido el **29/10** (Parte 1 de este documento, commit `0c593eb`), antes de ver
  ninguna corrida del estudio. Eso importa: el test se fija **antes** de mirar los datos.

**Por qué corregir por comparaciones múltiples.** Si hago 3 tests a α = 0,05 y ninguna ablación
tiene efecto, la probabilidad de que al menos uno salga significativo por azar es
1 − 0,95³ ≈ 14 %, no 5 %. Holm controla eso:

1. Ordeno los 3 p de menor a mayor: p₍₁₎ ≤ p₍₂₎ ≤ p₍₃₎.
2. Comparo p₍₁₎ con 0,05/3; si pasa, p₍₂₎ con 0,05/2; si pasa, p₍₃₎ con 0,05/1.
3. En cuanto uno no pasa, ese y los siguientes no son significativos.

Equivale a un "p ajustado" = (3 − rango) × p, tomando el máximo acumulado para que sea monótono.
Es menos conservador que Bonferroni (que multiplica todos por 3) y sigue controlando el error.

Ejemplo con números inventados: p = 0,004; 0,020; 0,300.
Ajustados: 3·0,004 = 0,012 ✓; 2·0,020 = 0,040 ✓; 1·0,300 = 0,300 ✗ ⇒ dos significativas.
Con Bonferroni: 0,012 ✓; 0,060 ✗; 0,900 ✗ ⇒ solo una.

**Por qué Wilcoxon solo como sensibilidad.** Las semillas emparejan las condiciones (misma
semilla ⇒ mismos pesos iniciales y misma secuencia de `reset`), así que un test emparejado es
defendible. Pero el emparejamiento es débil: las trayectorias divergen en pocos pasos, y Fran
comprobó que la misma semilla ni siquiera reproduce el mismo resultado en dos máquinas
distintas (`NOTAS_FRAN.md`). Por eso el principal es el no emparejado.

**Tamaños de efecto** (un p chico no dice cuánto empeora; esto sí):

* *Hodges-Lehmann*: la mediana de las 64 diferencias (completo − ablación) entre todos los pares.
  Es el estimador que acompaña naturalmente a Mann-Whitney.
* *Probabilidad de superioridad* = U / 64: la probabilidad de que una política completa
  elegida al azar supere a una de la ablación elegida al azar. 0,5 = sin diferencia; 1 = siempre.

### 14.3 Resultado parcial (falta "sin red objetivo")

Salida de `scripts/test_ablacion.py` con las corridas actuales:

| Variante | Media | Dif. de medias | HL | P(sup.) | U | p MW | p Holm (*) | p Wilcoxon | Completo gana |
|---|---|---|---|---|---|---|---|---|---|
| Completo | 137,8 | — | | | | | | | |
| Sin replay | −0,8 | 138,7 | 140,7 | 0,89 | 57 | **0,0070** | 0,021 | 0,0078 | 8/8 |
| Sin exploración | 25,8 | 112,0 | 99,2 | 0,78 | 50 | **0,0650** | 0,130 | 0,0391 | 7/8 |

(*) Los p de Holm no son finales hasta que estén las tres comparaciones.

Lectura:

* **Sin replay** es peor que el completo de forma clara y significativa incluso con la corrección.
* **Sin exploración** es peor en promedio (−112) y pierde en 7 de 8 semillas, pero con 8 semillas
  y Mann-Whitney **no alcanza significancia** (p = 0,065). La razón es la dispersión: sin
  exploración hay semillas que andan bien (7042: 180; 2042: 151) y otras que fracasan del todo
  (42: −169). El Wilcoxon emparejado da p = 0,039, **pero no corresponde cambiar de test después
  de ver los resultados**: sería elegir el análisis que da lo que uno quiere (*p-hacking*). En el
  informe se reporta el test fijado de antemano, y el Wilcoxon como sensibilidad.

---

## 15. 31/10: las 8 corridas sin exploración

### 15.1 Cómo se corrieron

Desde `TAREA2/`, una corrida por semilla, en paralelo:

```bash
PYTHONUTF8=1 python scripts/run_notebook.py --mode comparison --config A0 --variant no_exploration --seed 42
```

* `run_notebook.py` ejecuta solo las celdas de definiciones de `lander.ipynb` y llama a
  `entrenar_y_guardar`. Así todos usamos **exactamente el mismo código** (cada `config.json`
  guarda la revisión de git; `lander.ipynb` no cambió desde la integración del 30/10).
* Cada corrida crea `results/comparison_<fecha-hora>/A0_no_exploration_seed<s>/` con los cinco
  archivos del protocolo: `config.json`, `training.csv`, `evaluation.csv`, `summary.json` y
  `online_weights.pt`.
* Se validaron con `ablation_stats.py` de Fran: el script rechaza semillas no planeadas,
  duplicadas o configuraciones distintas a la del protocolo, y las aceptó.

### 15.2 Dos problemas que aparecieron (y cómo se resolvieron)

1. **`UnicodeEncodeError` al imprimir "≥"**. En Windows, el archivo `runner.log` se abre con la
   codificación del sistema (cp1252), que no tiene el carácter ≥. La corrida fallaba al imprimir
   el resumen final. Solución: `PYTHONUTF8=1`, que obliga a Python a usar UTF-8. Las corridas
   fallidas se borraron (no habían terminado limpias) y se volvieron a correr.
2. **Dos corridas con la misma carpeta**. La carpeta se nombra con la fecha y hora; si dos procesos
   arrancan en el mismo instante, el segundo falla con `FileExistsError` (a Fran le pasó lo mismo con
   3042). Solución: lanzar cada corrida 10 segundos después de la anterior.

**Para avisar al equipo**: ambos problemas le pueden pasar a Thiago con sus corridas.

### 15.3 Resultados

| Semilla | Pasos de entorno | Entrenamiento | Recompensa media (200 ep.) | Éxitos / 200 |
|---|---|---|---|---|
| 42 | 32 968 | 18 s | −168,7 | 0 |
| 1042 | 228 980 | 227 s | 62,6 | 15 |
| 2042 | 241 521 | 245 s | 150,7 | 74 |
| 3042 | 215 083 | 222 s | −70,2 | 0 |
| 4042 | 116 643 | 122 s | −133,3 | 0 |
| 5042 | 175 372 | 188 s | 65,9 | 33 |
| 6042 | 184 590 | 198 s | 119,2 | 42 |
| 7042 | 167 025 | 180 s | 180,3 | 123 |
| **Media** | | | **25,8** | **36 (18 %)** |

Comparación: el completo (8 semillas de Fran) tiene media 137,8 y 81 éxitos de 200 en promedio (40 %).

Las 8 corrieron en paralelo en unos 5 minutos de reloj, en Windows 11 con 8 núcleos y
torch 2.14.1 (Fran usó 2.15.0). No cambia el código, pero las corridas no son reproducibles bit
a bit entre versiones ni entre máquinas, como ya documentó Fran.

**Interpretación:**

* La **semilla 42** es el caso extremo: solo 33 000 pasos en 300 episodios (≈ 110 pasos por
  episodio). Se estrella en cada episodio y nunca ve otra cosa; sin ε no hay ningún mecanismo que
  le haga probar algo distinto.
* Las semillas que **sí** aprenden (7042, 2042) lo logran porque el entorno ya tiene variedad
  propia: cada episodio arranca con un empujón al azar y la red inicial elige acciones casi
  arbitrarias. Esa "exploración implícita" a veces alcanza y a veces no.
* Esa es la diferencia con quitar la memoria de repetición: sin exploración el resultado es una
  **lotería** (mucha varianza entre semillas); sin replay es **consistentemente malo**.
* Sin exploración los entrenamientos son **más largos** (≈ 170 000 pasos contra ≈ 152 000 del
  completo) porque muchas políticas aprenden a flotar hasta el paso 1000.

---

## 16. El script `test_ablacion.py`, línea por línea

```python
from study_tools import DEFAULT_PROTOCOL, aggregate, read_json
```
Reutiliza las funciones de Thiago: `aggregate` lee las carpetas de corridas, las valida contra el
protocolo y devuelve, por variante, la recompensa media de cada semilla. Así el test usa
**exactamente** los mismos números que la tabla de Fran.

```python
ABLACIONES = ['no_target', 'no_replay', 'no_exploration']
ALFA = 0.05
```
Las tres comparaciones planeadas. Que estén fijas en el código (y no "las que haya") es lo que
hace que Holm use siempre m = 3.

```python
def holm(pvalores, m):
    orden = sorted(range(len(pvalores)), key=lambda i: pvalores[i])
    ajustados = [None] * len(pvalores)
    maximo = 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, min(1.0, (m - rango) * pvalores[i]))
        ajustados[i] = maximo
    return ajustados
```
* `orden`: los índices de los p de menor a mayor.
* El p de rango 0 (el más chico) se multiplica por m = 3, el siguiente por 2, el último por 1.
* `min(1.0, …)`: un p no puede pasar de 1.
* `max(maximo, …)`: el ajustado nunca puede ser menor que el de un p más chico (monotonía); si no,
  podría rechazarse una hipótesis "más débil" sin rechazar una "más fuerte".
* Se devuelven en el orden original para asociarlos con cada variante.

```python
def comparar_una(completo, ablacion):
    c = np.array(list(completo.values()))
    a = np.array(list(ablacion.values()))
    u = stats.mannwhitneyu(c, a, alternative='two-sided', method='exact')
```
`completo` y `ablacion` son diccionarios `{semilla: recompensa media}`. `method='exact'` calcula el
p enumerando las repartijas de rangos en vez de usar la aproximación normal, que con 8 por grupo
no es buena. El orden `(c, a)` hace que **U cuente cuántos pares gana el completo**.

```python
        diferencia_media=float(c.mean() - a.mean()),
        diferencia_mediana=float(np.median(c) - np.median(a)),
        hodges_lehmann=float(np.median(np.subtract.outer(c, a))),
        U=float(u.statistic), p_mann_whitney=float(u.pvalue),
        prob_superioridad=float(u.statistic / (len(c) * len(a))),
```
* `np.subtract.outer(c, a)` arma la matriz 8 × 8 de todas las diferencias cᵢ − aⱼ; su mediana es
  el estimador de Hodges-Lehmann.
* La probabilidad de superioridad es U sobre los 64 pares.

```python
    comunes = sorted(set(completo) & set(ablacion))
    if len(comunes) >= 5:
        dif = np.array([completo[s] - ablacion[s] for s in comunes])
        w = stats.wilcoxon(dif, alternative='two-sided', method='exact')
```
El análisis de sensibilidad: empareja por semilla. Con menos de 5 pares el Wilcoxon no puede dar
p < 0,05 nunca, así que no se calcula. `completo_gana` cuenta en cuántas semillas el completo
supera a la ablación.

```python
    completas = len(filas) == len(ABLACIONES)
    for fila, p_holm in zip(filas, holm([f['p_mann_whitney'] for f in filas], m=len(ABLACIONES))):
        fila['p_holm'] = p_holm
        fila['significativo'] = bool(p_holm < ALFA)
```
Holm sobre los p de Mann-Whitney (el test principal, no el Wilcoxon). Si falta una variante,
`completas` es falso y el script avisa que los p de Holm no son finales.

El resto escribe `tests.json` e imprime la tabla.

---

## 17. 1/11: análisis de la política (cortes y visualizaciones)

La letra pide "cortes que fijen algunas dimensiones del estado y muestren la acción elegida o el
valor Q en función de las demás". Se hizo eso y se le agregaron dos cosas que hacen el análisis
más honesto: los **estados que el agente realmente visita** y **episodios reales**.

### 17.1 Método

* **Estado fijo**: x = 0, y = 0,8, vx = 0, vy = −0,3, θ = ω = 0, sin patas apoyadas. Es "la nave
  sobre la plataforma, a media altura, bajando despacio y derecha". Es el mismo que usa
  `study_tools.py slices`, para que las figuras sean comparables.
* **Tres cortes**, cada uno varía dos dimensiones en una grilla de 101 × 101 y deja las otras seis
  fijas: (x, vx), (θ, ω) y (y, vy).
* En cada punto se evalúa la red: **acción greedy** (argmax Q) y **max Q** (lo que la red cree que
  vale ese estado).
* **Puntos negros**: estados visitados en 20 episodios greedy reales (semillas 10 000–10 019, de
  desarrollo). Advertencia: un punto negro tiene las *otras seis* dimensiones con otros valores,
  así que no está exactamente en el corte. Sirven para ver **qué zonas del plano importan** y
  cuáles son estados sintéticos que el agente nunca encuentra (donde la red puede decir cualquier cosa).
* **Qué hace cada acción** (medido en el entorno, no supuesto): la 1 empuja la nave a la izquierda
  y la gira en sentido antihorario (sube θ); la 3 la empuja a la derecha y la gira en sentido
  horario (baja θ); la 2 frena la caída.

### 17.2 Figura `cortes_completo_s42.png` (completo, semilla 42)

Se usa la semilla 42 porque es la primera del protocolo, **no** porque sea la mejor: elegir la
mejor sería hacer trampa.

* **Corte altura – velocidad vertical** (arriba a la derecha). Debajo de una frontera inclinada
  la red usa el **motor principal**. La frontera va de vy ≈ −0,2 cerca del suelo a vy ≈ −0,85 a
  altura 1,5: **cuanto más cerca del suelo, menos velocidad de caída tolera**. Es lo que haría un
  piloto: a mucha altura puede caer rápido y frenar después; cerca del suelo no. Por encima de esa
  frontera (bajando despacio o subiendo) no hace nada. Las trayectorias negras muestran el patrón
  de aterrizaje: caer acelerando, frenar con el motor y "rebotar" sobre la frontera hasta llegar
  al suelo.
* **Corte ángulo – velocidad angular** (arriba al centro). La frontera entre motor izquierdo
  (naranja) y derecho (azul) es una línea inclinada que pasa cerca del origen con pendiente ≈ −1:
  la red dispara el motor derecho (que baja θ) cuando **θ + ω > 0** y el izquierdo cuando es
  negativo. Es un **controlador proporcional-derivativo**: corrige según cuánto está inclinada
  (θ) y según hacia dónde está girando (ω). El controlador heurístico de Gymnasium tiene la misma
  forma con pendiente 0,5 (dispara cuando 0,5·θ + ω > 0,05). Los puntos negros se concentran en
  θ ∈ [−0,3; 0,4], ω ≈ 0: la nave casi nunca se inclina mucho.
* **Corte posición – velocidad horizontal** (arriba a la izquierda). Si va rápido a la derecha
  (vx > 0,45) usa el motor izquierdo, que la empuja a la izquierda. **No es simétrico**: por
  debajo de vx ≈ 0,45 usa el derecho, incluso con vx = 0. Una hipótesis: en el estado fijo la nave
  baja (vy = −0,3) y la red prefiere ese motor por defecto. Los estados visitados se concentran a
  la derecha de la plataforma (x ∈ [0; 0,45]), coherente con esa tendencia. Es una
  **limitación** a mencionar: la política aprendida no respeta la simetría del problema.
* **Fila de abajo (max Q)**: los valores van de ≈ −50 a ≈ 150. Q es el **retorno descontado
  esperado**, no una probabilidad. Que Q sea alto con vy = −2 a mucha altura (abajo a la derecha)
  parece raro, pero es una zona que el agente no visita: la red extrapola.

### 17.3 Figura `cortes_8_semillas_angulo.png`

El corte θ – ω para las **8 semillas** del completo. En **las 8** aparece la misma estructura:
motor izquierdo abajo, derecho arriba, frontera inclinada cerca del origen. Cambian la pendiente y
dónde aparecen las zonas de motor principal y de "nada", casi siempre en regiones que no se
visitan. **Conclusión: la regla de control del ángulo es una propiedad robusta de lo que aprende
DQN, no un accidente de una semilla.** La figura `cortes_8_semillas_altura.png` hace lo mismo
para altura – vy.

### 17.4 Figura `cortes_completo_vs_sin_exploracion.png`

Semillas 42 y 7042 (la primera y la última del protocolo), completo contra sin exploración:

* **Sin exploración, semilla 42** (la que se estrella siempre): en el corte de altura usa motor
  principal para cualquier vy < 0, sin depender de la altura, y los estados visitados muestran
  caídas mucho más rápidas (vy hasta −1,7) que el completo nunca alcanza. Los 20 episodios terminan
  en choque. En el corte del ángulo la frontera izquierda/derecha es mucho más plana y aparece una
  banda ancha de motor principal donde el completo corrige el giro.
* **Sin exploración, semilla 7042** (la mejor sin exploración): la estructura se parece a la del
  completo. Aprendió algo razonable **a pesar** de no explorar.

Esto ilustra la sección 15.3: sin exploración, a veces se aprende y a veces no.

### 17.5 Episodios reales: cómo termina cada uno (`metricas_politica.json`)

20 episodios greedy por política (semillas 10 000–10 019), sumados sobre las 8 semillas
(160 episodios por variante):

| Variante | Posa (+100) | Choca (−100) | Flota hasta el paso 1000 |
|---|---|---|---|
| Completo | **101** (63 %) | **7** (4 %) | 52 (33 %) |
| Sin exploración | 35 (22 %) | 49 (31 %) | 76 (48 %) |
| Sin replay | 23 (14 %) | **135** (84 %) | 2 (1 %) |

Cómo se clasifica: Gymnasium reemplaza la recompensa del último paso por exactamente +100 si la
nave queda quieta y −100 si choca o sale; si el episodio se trunca, flotó hasta el paso 1000.

**El hallazgo más útil para el informe**: los fracasos del completo **no son choques**, son
episodios en los que la nave queda flotando cerca del suelo sin terminar de posarse (y pierde
puntos por gastar combustible). Las semillas más débiles del completo (3042, 4042, 5042) son las que
más flotan. Cada ablación falla de forma distinta: sin replay se estrella, sin exploración mezcla
los dos fracasos.

**Uso de acciones según la altura** (`acciones_por_altura.png`): el completo usa el motor principal
en ≈ 60 % de los pasos entre altura 0,2 y 1,2, y casi nada arriba de 1,3 (al principio deja caer la
nave). Cerca del suelo (y < 0,1) aumentan "nada" y el motor izquierdo: son los pasos con las patas
apoyadas, corrigiendo el equilibrio.

### 17.6 Acuerdo con el controlador heurístico: lo que **no** funcionó

Se midió en qué fracción de los estados visitados la acción del DQN coincide con `heuristic` de
Gymnasium (un controlador escrito a mano que aterriza bien). Resultado: **≈ 30 %** para el
completo (23–35 % según la semilla), y valores parecidos o incluso mayores para políticas que se
estrellan (sin replay 7042: 54 %). **No sirve como medida de calidad**: hay muchas formas distintas
de aterrizar bien, y el DQN usa el motor principal mucho más seguido que la heurística, que muchas
veces no hace nada. Queda en las métricas, pero no se usa como argumento en el informe.

---

## 18. El script `analisis_politica.py`, por bloques

```python
ACCIONES = ['Nada', 'Motor izq. (gira +)', 'Motor principal', 'Motor der. (gira −)']
ESTADO_FIJO = np.array([0, .8, 0, -.3, 0, 0, 0, 0], dtype=np.float32)
CORTES = [(0, 2, (-1, 1), (-1.5, 1.5), ...), (4, 5, ...), (1, 3, ...)]
SEMILLA_EVAL, EPISODIOS = 10_000, 20
```
Constantes: nombres de las acciones (con su efecto medido), el estado fijo y los tres cortes
(índice de la dimensión en x, en y, sus rangos y etiquetas). Los rangos cubren la zona donde vuela
la nave. Los episodios reales usan semillas de desarrollo y **nunca** las reservadas 30 000+.

```python
def variante(config):
    if not config.get('use_target', True): return 'no_target'
    ...
```
Deduce la variante de los interruptores guardados en `config.json` de cada corrida, en vez de
confiar en el nombre de la carpeta.

```python
def cargar_red(carpeta):
    h1, h2 = config['hidden_dims']
    red = nn.Sequential(nn.Linear(8, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 4))
    red.load_state_dict(torch.load(carpeta / 'online_weights.pt', map_location='cpu', weights_only=True))
    red.eval()
```
Reconstruye la red con la misma forma que `AgenteDQN` (las claves `0.weight`, `2.weight` y
`4.weight` coinciden porque los ReLU ocupan las posiciones 1 y 3) y carga los pesos finales.
`weights_only=True` evita ejecutar código arbitrario al cargar el archivo. **No se entrena nada.**

```python
def corte(red, ix, iy, rx, ry):
    x, y = np.linspace(*rx, RESOLUCION), np.linspace(*ry, RESOLUCION)
    xx, yy = np.meshgrid(x, y)
    estados = np.tile(ESTADO_FIJO, (xx.size, 1))
    estados[:, ix], estados[:, iy] = xx.ravel(), yy.ravel()
    q = q_de(red, estados).reshape(RESOLUCION, RESOLUCION, 4)
```
`meshgrid` arma la grilla de 101 × 101 puntos; `np.tile` repite el estado fijo 10 201 veces y se
pisan las dos columnas que varían. Una sola pasada de la red evalúa los 10 201 estados, y el
resultado se vuelve a dar forma de grilla con 4 Q por punto.

```python
def jugar(red, episodios=EPISODIOS):
    ...
            a = int(q_de(red, s[None])[0].argmax())
    ...
            fin = 'posado' if terminado and r == 100 else 'choque' if terminado else 'flotando'
```
Juega episodios greedy y guarda cada estado y acción. `s[None]` agrega la dimensión de lote. La
clasificación del final usa que Gymnasium pone exactamente ±100 en el último paso (sección 17.5).

```python
def acuerdo_heuristica(estados, acciones):
    env = gym.make('LunarLander-v3').unwrapped
    esperadas = np.array([heuristic(env, s) for s in estados])
    return float((esperadas == acciones).mean()), esperadas
```
`heuristic(env, s)` es la función de Gymnasium; solo usa `env` para saber si es la versión continua.

```python
def dibujar_corte(ax_a, ax_q, red, spec, visitados=None, titulo=''):
    ax_a.pcolormesh(x, y, q.argmax(axis=2), shading='auto', cmap=COLORES, norm=NORMA)
    if visitados is not None:
        ax_a.scatter(visitados[:, ix], visitados[:, iy], s=1, c='black', alpha=.15)
```
`pcolormesh` pinta cada celda con el color de la acción ganadora (`BoundaryNorm` asigna un color a
cada entero 0–3). Los estados visitados van encima, semitransparentes, para que se vea dónde se
concentran.

```python
    for (var, semilla), (red, carpeta) in sorted(politicas.items()):
        estados, acciones, finales = jugar(red)
        ...
        metricas.append(dict(..., posado=..., choque=..., flotando=..., uso_acciones=..., ...))
```
`main` primero juega los episodios de todas las políticas (24: 8 por variante) y guarda las
métricas. Después arma las cuatro figuras con esos mismos episodios, así los puntos negros y la
tabla salen de los mismos datos.

```python
        tramo = np.clip(np.digitize(est[:, 1], bordes) - 1, 0, len(bordes) - 2)
        fracciones = np.array([[np.mean(acc[tramo == t] == a) ...
```
Para `acciones_por_altura.png`: `np.digitize` asigna cada estado a un tramo de altura (15 tramos
entre 0 y 1,5) y se calcula qué fracción de los pasos de cada tramo usó cada acción. Se dibuja como
barras horizontales apiladas.

Uso (desde `TAREA2/`, tarda unos 2 minutos):

```bash
python scripts/analisis_politica.py --runs results/comparison_20261031*/A0*_seed* results/comparison_20261102*/A0*_seed* --output analisis_politica
```

---

## 19. 2/11: plantilla IEEE y qué tengo que escribir yo

**La letra prohíbe usar IA para redactar el informe** ("No está permitido utilizar herramientas
de IA para la redacción del informe"). Por eso `informe/informe.tex` tiene:

* El preámbulo IEEE de la Tarea 1 (mismos autores), en español.
* Las secciones en el orden de los puntos a–e de la letra, con quién escribe cada una.
* **Tablas con datos** (hiperparámetros de A0, cómo terminan los episodios, ablación parcial) y
  las figuras de política ya incluidas.
* Las referencias (Mnih 2015, Sutton y Barto, Gymnasium, Mann-Whitney, Holm).
* Marcadores rojos `[REDACTAR: …]` con los puntos que debe cubrir cada sección. **Son una guía,
  no texto para copiar**: hay que borrarlos y escribir con nuestras palabras. Antes de entregar,
  buscar "REDACTAR" y verificar que no quede ninguno.
* Un presupuesto de palabras por sección para no pasar las 5 páginas.

No hay LaTeX instalado en esta máquina: compilar en Overleaf (subir `informe.tex` y la carpeta
`figuras/`). La plantilla todavía no se compiló nunca.

**Lo que me toca escribir a mí** (secciones V y VII), con el material de este documento:

| Sección | Material |
|---|---|
| V. Análisis de la política | 17.1 (método), 17.2–17.4 (lectura de figuras), 17.5 (cómo terminan los episodios); aclarar que los cortes son estados sintéticos |
| VII. Limitaciones | La lista de abajo |

Limitaciones a cubrir (todas verificadas en los datos):

1. **8 semillas**: el efecto de quitar la exploración (−112) no llega a ser significativo; con
   más semillas podría serlo. "No significativo" no quiere decir "sin efecto".
2. **300 episodios**: los agentes siguen mejorando (mi agente de 1200 episodios pasa de 219 a 247
   entre los episodios 600 y 1200). Comparamos políticas "a mitad de camino".
3. **Reproducibilidad entre máquinas**: misma semilla, distinta máquina o versión de torch ⇒
   distinto resultado (comprobado por Fran). Cada corrida sigue siendo válida como réplica.
4. **Semillas de evaluación de desarrollo** (10 000–10 199) también se usaron para calibrar; la
   evaluación independiente en 30 000+ no se hizo todavía.
5. **Cortes sintéticos**: estados con seis dimensiones fijas pueden no ser alcanzables; por eso se
   superponen los estados visitados.
6. **Definición de "sin replay"** (minilote efectivo 1): otra definición podría dar otro resultado.
7. **Asimetría de la política** en el corte x – vx, y fracasos por flotar en lugar de posarse.

---

## 20. Estado al 2/11 y pendientes

| | Estado |
|---|---|
| Protocolo | Confirmado por Fran y por mí; **falta Thiago** |
| Test estadístico | Propuesto y programado (`scripts/test_ablacion.py`); falta que el equipo lo confirme y que Fran lo use |
| Completo (8) | Hecho (Fran) |
| Sin replay (8) | Hecho (Fran) |
| Sin exploración (8) | **Hecho (yo)** |
| Sin red objetivo (8) | **Falta (Thiago)**: no está en GitHub. Hasta que esté, los p de Holm no son finales |
| Cortes y visualizaciones | **Hecho (yo)** |
| Plantilla IEEE | **Hecha (yo)**, sin texto y sin compilar |
| Redactar análisis de política y limitaciones | **Pendiente (yo)**, a mano |
| Evaluación en semillas reservadas 30 000+ | No se hizo; decidir en equipo si se hace al congelar todo |

Cuando Thiago suba sus corridas, volver a correr los dos scripts agregando sus carpetas:

```bash
python scripts/test_ablacion.py --runs results/comparison_20261031*/A0*_seed* results/comparison_20261102*/A0*_seed* <carpetas de Thiago> --output ablacion
python scripts/analisis_politica.py --runs <las mismas carpetas> --output analisis_politica
```

y completar las filas "pendiente" de las tablas del informe.

---

## 21. Más preguntas probables (parte 2)

**¿Por qué Mann-Whitney y no t de Student?** No supone normalidad; con 8 valores bimodales no se
puede verificar. Además lo fijamos antes de ver los datos.

**¿Por qué corregir por tres comparaciones?** Con tres tests a 5 % la chance de al menos un falso
positivo es ≈ 14 %. Holm la baja a 5 % y es menos conservador que Bonferroni.

**Wilcoxon da significativo para "sin exploración", ¿por qué no lo usan?** Porque el test se
fijó antes, y cambiarlo después de ver cuál da p < 0,05 invalida el p. Se reporta como sensibilidad
y se discute.

**Entonces, ¿la exploración no sirve?** No es eso. En promedio es 112 puntos peor y pierde en 7 de
8 semillas; con 8 semillas y esa variabilidad no alcanza para descartar el azar. Además, en la
Parte 1 (5 semillas, 600 episodios, otra configuración) sí fue significativo (p = 0,016).

**¿Por qué sin exploración a veces aprende?** El entorno da variedad propia (empujón inicial al
azar) y la red inicial actúa casi al azar. Es exploración "gratis" pero no controlada: depende de
la suerte de la semilla.

**¿Qué aprendió la política?** Un controlador proporcional-derivativo para el ángulo
(θ + ω > 0 ⇒ motor que baja θ) y un freno con el motor principal cuya velocidad límite depende de
la altura. La regla del ángulo aparece en las 8 semillas.

**¿Los cortes muestran estados reales?** No necesariamente: fijan seis dimensiones. Por eso
superponemos los estados visitados y solo interpretamos las zonas donde hay puntos.

**¿Por qué fallan los episodios del completo?** Dos de cada tres fracasos son flotar hasta el paso
1000 sin posarse, no choques (7 choques en 160 episodios).

---

## 22. Actualización del 4/11: el informe

Thiago integró el estudio completo (32 políticas, con sus 8 sin red objetivo), escribió y compiló el
informe (5 páginas) y armó `entrega_final/` con una revisión. Mis 8 corridas, el test y los cortes
están incorporados. Lo que me queda (redactar con mis palabras las secciones V y VII, aportar a
resumen y conclusiones, revisar referencias y el cierre) está en
[GUIA_INFORME_AGUSTIN.md](GUIA_INFORME_AGUSTIN.md), junto con el problema pendiente de la regla de IA.
