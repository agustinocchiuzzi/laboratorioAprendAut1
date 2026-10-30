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
