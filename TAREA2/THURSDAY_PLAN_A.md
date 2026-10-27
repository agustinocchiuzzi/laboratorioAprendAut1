> **Implementation update:** The Thursday agent, checks, A0/A1 runs and evaluations are complete. See [RUN_STATUS_A.md](RUN_STATUS_A.md) for measured results and remaining human tasks. The guide below is retained as the original plan.

**Person A: DQN implementation guide for Thursday, October 29, 2026**

Build one complete, understandable DQN for `LunarLander-v3`, compare two architecture/learning-rate configurations, and demonstrate a switch that removes the target network. Bring measured results even if the policy is weak. The settings below are proposed starting points, not experimentally validated recommendations.

Confirmed available time: 4-6 hours of hands-on work, plus unattended training. Prioritize the complete agent, two configurations with one training seed each, evaluation, and the target-off check. Repeat both configurations with a second seed only if time remains. This guide plans implementation; no training or evaluation has been run.

**Start from the course notebook**

The unmodified [course notebook](course_template.ipynb) is now in this folder. Make your working copy `lander.ipynb`. Its three TODO blocks are still unimplemented in the downloaded original.

The [assignment PDF](Tarea%20-%20Aprendizaje%20por%20Refuerzos%20%282026%29.pdf) requires the implementation and additional agent helpers to stay inside `AgenteDQN`. Preserve the supplied agent interface, training/evaluation functions, and plotting utilities. Hyperparameters and experiment call sites can be adjusted. The final submission is due November 4; it also needs policy analysis, at least three ablations with multiple training seeds and statistical tests, and a maximum five-page IEEE article. Those full experiments are beyond Thursday's checkpoint.

The assignment permits AI assistance for implementation, but the team must write the report itself; grammar and style correction are permitted. Use this as a development checklist and keep your own observations.

**What to have ready on Thursday**

- [ ] A complete `AgenteDQN`, running with replay, a target network, and epsilon-greedy exploration.
- [ ] Two trained configurations with raw learning curves and a clearly labelled moving average.
- [ ] Greedy evaluation results for both configurations and the random agent on the same evaluation seeds.
- [ ] Exact settings, seeds, package versions, hardware/device, episode counts, environment steps, and runtime.
- [ ] `use_target=False` implemented and checked with actual learning updates.
- [ ] Saved model weights and raw numerical results, so plots can be rebuilt without training.
- [ ] Your brief notes on what worked, failed, and is still uncertain.

For Thursday, a fresh kernel must reproduce the agent and experiments through section 8. Sections 10 and 11 remain explicitly deferred. If you want the entire notebook to execute with Run All, guard those two unfinished TODO blocks with an explicit `RUN_FINAL_EXPERIMENTS=False` flag and print that they are deferred. Remove that deferral for the final submission. Section 9's GIF is optional evidence for the meeting.

**Schedule and stopping points**

| When | Hands-on budget | Work | Finish when |
|---|---:|---|---|
| Monday, Oct 26 | 45-60 min | Open a working copy, run dependencies and the random agent, inspect the interface | One complete random episode and baseline evaluation work |
| Tuesday, Oct 27 | 2-2.5 h | Implement the whole agent and target switch; check one learning update | Shapes, gradients, terminal handling, target copying, and greedy evaluation are correct |
| Wednesday, Oct 28 | 1-1.5 h + training | Time a short pilot, run A0 and A1, save each result immediately | Two curves, evaluation results, and runtimes exist |
| Thursday, Oct 29, before meeting | 30-45 min | Check clean execution, inspect saved evidence, prepare your observations | You can run the notebook and explain the comparison |

If you have only 1-3 hours, shorten both training runs equally and use one training seed. Preserve correctness checks, evaluation, and saved evidence. If you have extra time, repeat both configurations with a second training seed before adding more hyperparameters.

**Implementation order**

1. **Get the kernel working.** Run the notebook's dependency cells and sections 2-5 before writing DQN code. The shell inspected here has Python 3.14 and no PyTorch/Gymnasium; your notebook kernel may differ. A Python 3.12/3.13 kernel or Colab is a practical starting choice. If Box2D setup stalls, resolve that first. Gymnasium documents an additional SWIG/source-build requirement on Python 3.14+. Keep the working package versions in your experiment record. [Box2D installation documentation](https://gymnasium.farama.org/environments/box2d/)

2. **Implement `__init__`.** Keep `estado_space, accion_space, semilla=None` and add configurable `hidden_dims`, `lr`, and `use_target`, plus the shared settings below. Seed Python, NumPy, PyTorch, and `accion_space.seed(semilla)` before network initialization. Create `self.red_q`, the optional target copy, an Adam optimizer over `red_q` parameters only, replay storage, counters, and `epsilon`. Use `nn.Sequential` and a `deque` inside the class so extra top-level network/buffer classes are unnecessary.

3. **Build the Q network.** Map eight state values to four unbounded Q-values with two hidden linear layers and ReLU. The output layer is linear. An MLP fits the vector observations; a convolutional network is unnecessary. Initialize the target with a separate copy of the online network's weights. It must have separate parameter storage. The environment dimensions are documented by [Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

4. **Implement `elegir_accion(self, estado, explorar=True)`.** During training, choose a random action with probability `epsilon`; otherwise return the integer index of the largest online Q-value. With `explorar=False`, always choose greedily under `torch.no_grad()`. Evaluation must not advance the exploration schedule, consume exploration RNG draws, append replay transitions, or modify training counters.

5. **Implement `aprender(self, estado, accion, recompensa, siguiente_estado, terminado)`.** Store copies of both observations with the action, reward, and `terminado`. Count environment interactions here. After warmup, sample a uniform minibatch and take one optimizer step every four environment interactions. Compute the value of the chosen action with `gather`; compare it with a detached Bellman target using Huber loss. Zero gradients, backpropagate, clip the gradient norm, and step Adam. Keep helper methods inside `AgenteDQN`. [PyTorch's DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) explains replay, action selection, and the learning update; its CartPole hyperparameters are not LunarLander results.

6. **Implement `fin_episodio(self)`.** Increment the completed-episode counter and update epsilon once per training episode using the fixed schedule below. The supplied trainer already calls this method. Keep the readable `self.epsilon` attribute because the progress logger uses it.

7. **Connect the existing notebook functions.** Section 7 calls `entrenar_agente`; section 8 calls `evaluar_agente`. Use their real signatures below. Add experiment records around these calls without rewriting the supplied loops.

**The learning update and your target-network switch**

For each transition, train the online prediction toward:

```text
y = reward + gamma * (1 - terminated) * max_a Q_bootstrap(next_state, a)
Q_bootstrap = target network when use_target=True; online network otherwise
loss = Huber(Q_online(state, chosen_action), y)
```

The detached next-state estimate is used in both modes. Removing the target network means bootstrapping from the current online network; setting `gamma=0` would remove future rewards and test a different change.

Illustrative code inside your learning helper, after constructing the batch tensors:

```python
# estados/siguientes: float32 [B, 8]
# acciones: int64 [B, 1]; recompensas: float32 [B]; terminados: bool [B]
q_elegidos = self.red_q(estados).gather(1, acciones).squeeze(1)
with torch.no_grad():
    red_bootstrap = self.red_objetivo if self.use_target else self.red_q
    q_siguientes = red_bootstrap(siguientes).max(dim=1).values
    objetivos = recompensas + self.gamma * (~terminados).float() * q_siguientes
loss = F.smooth_l1_loss(q_elegidos, objetivos, beta=1.0)
```

When enabled, hard-copy the online weights to the target after every 1,000 environment interactions. When disabled, do not create/use/synchronize a separate target. Keep replay, exploration, optimizer, discount, and update frequency unchanged.

The trainer already distinguishes termination and truncation: reset the episode on either, but remove bootstrapping only for `terminado=True`. At the time limit, use the actual next observation before reset. [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) explains why.

**Two deliberate configurations**

| Configuration | Network | Online parameters | Learning rate | Hypothesis to investigate |
|---|---|---:|---:|---|
| A0 | `8 -> 64 -> 64 -> 4`, ReLU in hidden layers | 4,996 | `1e-3` | A compact model with larger optimization steps may make useful progress quickly |
| A1 | `8 -> 128 -> 128 -> 4`, ReLU in hidden layers | 18,180 | `5e-4` | More capacity with smaller optimization steps may produce steadier learning |

These hypotheses are untested. Comparing A0 and A1 selects between two packages of settings. Because both architecture and learning rate change, it cannot identify their separate effects. Say that explicitly. If you prefer to isolate learning rate, decide before running to use `(128, 128)` for both and keep the two learning rates; architecture then has a design rationale but no comparative evidence. Do not run both experiment designs before Thursday.

Hold the following fixed in the chosen comparison:

```text
environment                  LunarLander-v3, continuous=False, wind disabled
observations/rewards         supplied values; no normalization or reward clipping
network initialization       PyTorch nn.Linear default, seeded before construction
optimizer                    Adam; betas=(0.9, 0.999), eps=1e-8, weight_decay=0
discount gamma               0.99
loss                         SmoothL1/Huber, beta=1.0, mean reduction
gradient clipping            global norm <= 10
replay capacity              50,000 transitions
minibatch size               64; uniform sampling without replacement per batch
learning starts              after 1,000 stored environment interactions
learning frequency           one optimizer step every 4 environment interactions
target network               enabled; initial copy; hard copy every 1,000 env steps
epsilon                      1.0 -> 0.05 linearly over 200 completed episodes
epsilon after episode e      max(0.05, 1.0 - 0.95 * min(e / 200, 1.0))
training budget              300 episodes per configuration, max 1,000 steps each
early stopping               disabled: umbral_resolucion=None
minimum training seed        42 for both configurations
optional second seed         1,042 for both configurations, if runtime allows
evaluation                   200 episodes; explorar=False; no learning
evaluation reset seeds       10,000 through 10,199, identical for every agent
random agent action seed     42, using a fresh AgenteAleatorio
device                       CPU for the initial comparison; record thread count
```

The supplied trainer budgets by episodes. Equal episodes do not imply equal environment interactions, because learned policies change episode lengths. Record total steps as well as runtime, and compare curves against cumulative steps too. Do not describe these runs as having equal compute.

**Checks before a long run**

- [ ] A batch of 64 observations produces Q-values of shape `(64, 4)`; selected Q-values and targets both have shape `(64,)`.
- [ ] One update changes online weights, produces a finite loss, and leaves targets detached.
- [ ] Target weights match initially, stay frozen between synchronization points, and match after a copy. Target parameters receive no gradients.
- [ ] On a synthetic terminal transition, the target equals the reward. On a nonterminal transition it includes the discounted next-state value.
- [ ] Repeated `explorar=False` calls leave epsilon, replay size, weights, and training counters unchanged.
- [ ] A fresh agent with `use_target=False` completes real optimizer updates without accessing a missing target network.
- [ ] Repeating a short run with the same seed and software/device setup gives consistent results. Seeds alone do not guarantee identical output across platforms or releases. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

**Run the comparison without accidentally continuing a previous agent**

First run a disposable 20-50 episode pilot to check correctness and timing. For the target-off prototype, use a separate short run and verify the update counter is positive; a run that never reaches warmup does not test learning. Label these pilot runs separately from A0/A1 evidence.

Then create a fresh environment and agent for every `(configuration, training seed)` pair. Reinitialize the network, optimizer, replay, epsilon, counters, and RNGs. Run sequentially in the notebook, so global RNG reseeding cannot interfere between agents. Do not train A1 from A0's checkpoint.

These are existing helper calls; the constructor options must be added in your implementation first:

```python
env = gym.make('LunarLander-v3')
agente = AgenteDQN(
    env.observation_space, env.action_space,
    semilla=42, hidden_dims=(64, 64), lr=1e-3, use_target=True,
    # Pass or fix all other settings exactly as recorded above.
)
inicio = time.perf_counter()
recompensas, longitudes = entrenar_agente(
    agente, env, num_episodios=300, max_pasos=1000,
    umbral_resolucion=None, semilla=42, verbose=True,
)
segundos_entrenamiento = time.perf_counter() - inicio
env.close()

env_eval = gym.make('LunarLander-v3')
inicio = time.perf_counter()
resultado = evaluar_agente(agente, env_eval, num_episodios=200, semilla=10_000)
segundos_evaluacion = time.perf_counter() - inicio
env_eval.close()
```

Evaluate a fresh `AgenteAleatorio` with the same 200 reset seeds and action-space seed 42. Save its raw returns once. For A1, change only `hidden_dims` and `lr` in a fresh run. For the optional repetition, change the training seed to 1,042 in both the constructor and `entrenar_agente`.

Use the final checkpoint at the fixed training budget for this comparison. Avoid selecting each configuration's best-looking episode or checkpoint. If you reduce the budget after the pilot, choose one common budget before running A0/A1 and record the change. With a much shorter run, the fixed 200-episode epsilon decay may remain incomplete; report that limitation.

**Evidence to save and how to read it**

For each run, save `config.json`, `training.csv`, `evaluation.csv`, `summary.json`, and the online weights in a unique directory such as `results/A0_seed42/`. Also save the notebook with its outputs. The config must include every setting above, seeds, Python/package versions, device, CPU thread count, and whether the target was enabled.

`training.csv`: one row per episode with episode number, environment reset seed, return, length, cumulative environment steps, and epsilon. Optional diagnostics: mean loss and Q-value magnitude. `evaluation.csv`: one row per evaluation seed with raw return and `return >= 200`. `summary.json`: configuration, training seed, completed episodes, total steps, actual optimizer updates, training seconds, evaluation seconds, mean/median/std return, successes, and number evaluated.

The existing evaluator returns the raw returns and all of these evaluation summaries; its standard deviation uses `ddof=0`. Report that consistently. Variation across evaluation episodes describes different scenarios for one trained policy, not training-seed variability.

Bring a plot of raw training return plus the supplied 100-episode moving average for each configuration. Early windows contain fewer than 100 observations. Also compare moving averages against cumulative environment steps. Label the smoothing and seeds. Training loss and exploratory training reward do not replace greedy evaluation.

Your comparison table should contain A0, A1, and the random baseline, with mean return, standard deviation, success count/total, training steps, and runtime. Do not fill results until measured. The assignment's success criterion is **strictly more than 10%** of evaluation episodes with return **at least 200**; with 200 episodes, that means at least 21 successes. Below-threshold performance is still useful Thursday evidence.

Use greedy mean return as the preliminary selection metric, while also checking success rate and runtime. If they disagree, bring the tradeoff to the group. One seed supports a provisional choice; two still provide limited evidence. These 200 evaluation seeds are being used for calibration, so describe the results as development evaluation. For an additional independent final check, reserve new seeds after settings are frozen, while retaining the course notebook's required evaluations.

Saving online weights plus architecture is enough to reload a policy for evaluation. Exact training continuation additionally needs target and optimizer states, replay, counters, exploration state, RNG states, and the episode/seed position; do not claim that a weights-only file resumes training exactly.

**Budget the remaining experiments from measurements**

After the pilot, estimate environment steps per second from measured steps and training time. A rough projection is `planned_steps / measured_steps_per_second`, plus evaluation time. Episode lengths and update costs can change as learning improves; recheck the projection after a longer run. Do not promise a wall-clock duration before measuring it.

For Thursday, the minimum is two main training runs, a shared random baseline evaluation, and a short target-off check. A second seed makes four main runs without adding configurations. Keep the target-off full ablation for the later study unless everything else is finished.

If performance is weak, first check the target mask, action indexing, detached targets, update count, epsilon evolution, and greedy evaluation. Finite/decreasing loss alone is insufficient evidence of improvement. Record training that plateaus, large Q-values, unstable rewards, or time-limit episodes; do not silently discard failed seeds. Fix correctness errors, then rerun affected comparisons. Avoid opening a large search over architectures, optimizers, and exploration settings.

**Prepare your own meeting notes**

Answer these from the saved evidence: Which configuration was better on which metric? How long did it train? Did the learning curve improve or collapse? What did the target-off check actually establish? What can two jointly changed settings and one seed not establish? What is the next single experiment you would run?

After Thursday, agree on one complete reference configuration before combining the three ablation switches. The final comparison should retrain each variant from scratch, remove one component at a time, use several training seeds and common evaluation seeds, and report an appropriate statistical comparison. Multiple evaluation episodes of one policy are not substitutes for independently trained seeds. Policy slices and the team's report remain separate final tasks.

**Source record**

The notebook was retrieved on October 26, 2026 from the [course GitLab link embedded in the PDF](https://gitlab.fing.edu.uy/aprendaut2/aa-26/-/blob/main/tarea%202/LunarLander_DQN_Tarea.ipynb). The local original is unchanged; SHA-256: `cf52a35513390b201d7ff3a6b3a006ca4e1160c022fb03090569861d095f13ed`. The schedules, configurations, and reduced training budget in this guide are planning choices; the course notebook defaults to 1,200 training episodes.
