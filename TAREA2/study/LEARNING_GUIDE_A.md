# Person A learning guide: understand and explain the saved DQN work

Study material, not submission prose. Based on the actual `TAREA2/` implementation and saved evidence inspected on October 29, 2026. No training was run to prepare this guide. Numerical teaching examples below are explicitly hypothetical. Paths are relative to `TAREA2/` unless linked otherwise.

## 1. Start with the task

**Reinforcement learning** means learning decisions from interaction and reward feedback. An **agent** chooses actions; an **environment** simulates their consequences. Here the environment is LunarLander-v3, with discrete engines, no wind, and no continuous-action control. The aim is a gentle, stable landing near the platform.

An **observation** is the information returned to the agent. Here it is a vector of eight numbers: horizontal position, height, horizontal velocity, vertical velocity, angle, angular velocity, left-leg contact and right-leg contact. The code calls this `estado` (state). A **state** ideally contains all information needed to predict future consequences; in this implementation the observation vector is used directly as the state representation. Inputs are not pixels. No extra observation normalization is applied.

An **action** is one discrete choice: 0 do nothing; 1 left orientation engine; 2 main engine; 3 right orientation engine, following the course action labels. Engine labels should not be confused with the direction of the resulting horizontal motion.

A **reward** is the scalar feedback after one action. It is not a label identifying the correct action. The course description lists contact bonuses, main/side engine costs and terminal ±100. That description is abbreviated: do not infer that summing those listed terms alone reconstructs every saved reward. The agent receives the simulator's returned reward directly, with no reward clipping or custom reward function.

A **step** is one action and one simulator response. An **episode** is one attempt starting at reset and ending at termination or cutoff. An episode's **return** is the sum of its rewards. For rewards 2, −1, 5, the measured return is 6. The evaluation success rule is return ≥200. Physical landing, simulator termination and the success threshold are different concepts.

A **policy** is a rule for choosing actions. Our greedy policy chooses the action with the largest learned value. **Greedy** means choosing the currently highest estimate; it does not mean the choice is guaranteed optimal.

A **discount factor**, gamma (γ), reduces the weight of distant rewards. The learning objective uses discounted future return, while saved episode returns are ordinary undiscounted sums. With γ=.99, a reward two steps later receives weight .99²=.9801 in a discounted sum.

## 2. What came from the course, and what was implemented

Read [course_template.ipynb](../course_template.ipynb) alongside [lander.ipynb](../lander.ipynb).

The course supplied environment explanations/setup examples, the `Agente` interface, the random reference agent, `entrenar_agente`, `evaluar_agente`, result-printing utilities, moving-average/progress plotting, and GIF helpers. The template DQN constructor raises `NotImplementedError`; its action/learning/episode methods are empty. Policy analysis and the ablation study are other incomplete template tasks.

The implemented contribution fills `AgenteDQN`, adds its two private learning helpers, correctness checks, experiment controls/configurations, fresh-run orchestration, saving, additional step-based curves and scripts for selecting runs, analysis and recording. It uses PyTorch's existing linear layers, automatic differentiation, Adam and loss functions; it does not implement those mathematical libraries from scratch.

The training loop definition and progress-plotting cell match the supplied template. Evaluation is course supplied too; experiment call sites around it were adapted. This distinction matters: do not claim to have invented the environment, the training loop or all DQN theory. Explain the implementation decisions you can trace in section 6 of the working notebook.

Older `README_A.md`, `START_HERE_A.md` and `RUN_STATUS_A.md` describe the original seed-42 checkpoint. They are useful background but incomplete inventories now. Actual saved evidence includes A2 and seed 1042 for all three configurations. Also, the current comparison controls include A2 even though an adjacent comment still says A0/A1.

## 3. The network, before the update

A **Q-value**, Q(s,a), estimates discounted future reward after choosing action a in state s and behaving well afterwards. DQN (**Deep Q-Network**) approximates these values with a neural network. Values are estimates, not probabilities, and can be negative or exceed 1. There is no softmax output.

A **parameter** is a learned weight or bias. An **architecture** is the arrangement and size of layers. This agent uses an **MLP**, a multilayer perceptron: fully connected layers with **ReLU**, the function max(0,x), after the two hidden layers. A hidden layer is an intermediate representation between input and output. The last layer is linear and returns four values, one per action.

A0/A2: `8 → 64 → 64 → 4`, with 4,996 parameters. A1: `8 → 128 → 128 → 4`, with 18,180 parameters. For A0, count `(8×64+64)+(64×64+64)+(64×4+4)=4996`. A wider network can represent more functions but does not guarantee better learning.

## 4. Every implemented AgenteDQN method

| Method | Actual responsibility | What to explain aloud |
|---|---|---|
| `__init__` | Validates settings; seeds generators; stores controls; initializes counters and replay; constructs online network, optional target copy and Adam | Establishes all state needed for interaction and learning |
| `elegir_accion` | Random branch during exploration; otherwise float32 `[1,8]` input and online `argmax` under `no_grad` | Chooses an action; does not train |
| `_calcular_objetivos` | Computes reward plus masked discounted maximum next-state value under `no_grad` | Builds fixed values to learn toward |
| `_actualizar_red` | Samples replay; builds tensors; gathers chosen Q-values; computes mean Huber loss; backpropagates, clips gradients, calls Adam; records update count/loss | Performs one learning update |
| `aprender` | Copies/stores a transition; increments step count; conditionally updates; then conditionally synchronizes target | Called every training step, although an optimizer update happens less often |
| `fin_episodio` | Increments completed episodes and recomputes epsilon | Exploration changes once per training episode |

Constructor details worth understanding: exactly two positive hidden sizes are required; gamma and epsilon bounds are checked; capacities/frequencies must be valid; learning rate and clipping threshold must be positive. These checks catch selected invalid inputs, not every possible malformed configuration. The device defaults to CPU. Counters begin at zero; `last_loss` begins as `None`. The input size is the product of the observation shape.

The **online network**, `red_q`, is the trainable network used for actions. The **target network**, `red_objetivo`, is an independent deep copy used to estimate future values. It starts equal, has gradients disabled and is placed in evaluation mode. Independent storage prevents online updates from silently changing it. In this MLP, evaluation mode has no dropout/batch-normalization behavior to change; disabling gradient recording is a separate operation.

Adam receives only online parameters, with betas (.9,.999), epsilon 1e−8 and zero weight decay. Python, NumPy, PyTorch and action-space generators are seeded when `semilla` is provided.

## 5. Replay and the precise learning update

A **transition** is `(s,a,r,s_next,terminated)`, the data from one interaction. **Replay** is storage of past transitions for later reuse. `memoria` is a bounded `deque` holding the most recent 50,000 transitions; inserting beyond capacity evicts the oldest. `aprender` copies both observation arrays as float32 so later mutation by a caller cannot corrupt stored experience.

A **minibatch** is a small group of examples used for one update. `random.sample` draws 64 distinct replay entries uniformly, without replacement within that batch. Entries can recur in later batches. This reduces adjacency/correlation compared with training only on consecutive latest steps and reuses experience. Samples are not perfectly independent, nor is this prioritized replay.

**Warmup** collects experience before updating. Defaults require at least 1,000 environment steps, enough replay entries, and a step divisible by 4. Thus the first update is at step 1000, then 1004, 1008, etc. With default settings the update count after N≥1000 steps is `floor(N/4)−249`; at 152,500 it is 37,876. Updating every fourth step is not the same as taking one action every four steps.

A **tensor** is an array used by PyTorch. Here batch shapes are:

| Quantity | Shape/type |
|---|---|
| states and next states | `[64,8]`, float32 |
| all online Q-values | `[64,4]`, float32 |
| actions after `unsqueeze(1)` | `[64,1]`, int64 |
| rewards and terminal flags | `[64]`, float32 / bool |
| selected predictions and targets | `[64]` |

`gather(1, acciones)` selects each row's column for the action actually taken. `squeeze(1)` removes the extra one-column dimension. The matching `[64]` shapes avoid accidentally broadcasting into a matrix of cross-example errors.

The **Bellman target** is a value constructed from immediate reward and estimated future value:

`y = r + γ × (1 − terminal) × max_a Q_bootstrap(s_next,a)`.

**Bootstrapping** means using an estimate to help build another estimate. This is not a known correct label. The next-state maximum comes from the target network when enabled, otherwise the online network. This is ordinary DQN, not Double DQN: it does not split next-action selection and evaluation between two networks.

A **detached target** is held fixed while computing the update's derivatives. `torch.no_grad()` surrounds target computation, so gradients do not flow through the next-state branch. This remains necessary in target-off mode. The **terminal mask** is `(~terminados).float()`: terminal True becomes 0, False becomes 1.

A **loss** measures prediction error. Huber loss (`smooth_l1_loss`, beta=1) is `0.5e²` for |e|<1 and `|e|−0.5` otherwise, where `e=prediction−target`. It is quadratic for small errors and linear for large errors. The implementation averages the 64 losses. It softens large-error influence; it does not eliminate instability or clip rewards.

A **gradient** tells how loss changes with a parameter. **Backpropagation** computes those derivatives through the network. `zero_grad(set_to_none=True)` clears old gradients; `loss.backward()` computes new ones. **Gradient clipping** rescales the combined parameter-gradient norm to at most 10. It does not bound Q-values, individual weights or episode returns.

An **optimizer** uses gradients to adjust parameters. **Adam** maintains running averages of gradients and squared gradients, bias-corrects them and scales updates using them. The **learning rate** sets the overall update scale; it is not a probability and is not a guaranteed fixed change in each Q-value. `step()` updates online weights; counters and detached scalar `last_loss` are then recorded.

**Target synchronization** is a hard copy: every 1,000 environment steps, `aprender` copies online weights into the target after any update at that step. This is not every 1,000 optimizer updates. The target is frozen between copies, but changes at copies. This helps keep the learning reference from following every individual update.

## 6. One transition and one update, numerically

This is a hypothetical hand calculation, not a recovered training sample. Suppose observation `s=[.1,.8,0,−.2,.05,0,0,0]` produces online values `[1,2,4,0]`. A greedy decision picks action 2, the main engine. During exploration a random branch could instead choose any action, including action 2.

Suppose the simulator returns reward 2, `s_next=[.1,.79,0,−.18,.04,0,0,0]`, `terminated=False`, `truncated=False`. `aprender` stores copies with the action and terminal flag. The observed reward is assumed for the example; it is not calculated from the displayed coordinates.

Later this entry appears among 64 sampled transitions. Suppose its current online row is still `[1,2,4,0]`; `gather` selects 4. Suppose the bootstrap next-state row is `[1,3,2,0]`; its maximum is 3. The target is `2+.99×3=4.97`. Error is `4−4.97=−.97`; this entry's Huber loss is `.5×.97²=.47045`. If all other 63 entries happened to have zero loss, mean batch loss would be `.47045/64=.00735078125`; this prediction's derivative would be `−.97/64=−.01515625`. Real batches usually have many nonzero errors.

Backpropagation distributes this signal through the chosen output and shared hidden parameters. Suppose the full gradient vector had norm 20: clipping to 10 scales it by approximately .5. Adam then updates parameters using its running statistics. You cannot infer the exact next Q-value from .97 and the learning rate alone: inputs, activations, all batch gradients and Adam history matter. For intuition only, a scalar Adam parameter's first step with a negative gradient and zero prior moments is approximately a positive learning-rate-sized change; the actual DQN update is multidimensional.

If the same transition were genuinely terminal, its target would be 2, regardless of the next-state maximum. The prediction error would be 2 and its Huber loss 1.5. If it were only truncated, target would remain 4.97. At step 1000 an update occurs and then the online weights are copied to the target. At step 1001 only storage occurs under defaults.

## 7. Exploration, seeds, ending episodes and evaluation

**Exploration** tries actions beyond the greedy choice. In **epsilon-greedy** selection, epsilon is the probability of taking the random branch. A random draw can still pick the greedy action, so epsilon is not the exact fraction of actions differing from greedy.

After k completed training episodes, `epsilon=max(.05,1−.95×min(k/200,1))`: initially 1; after 1 episode .99525; after 100 .525; after 200 and 300 .05. The CSV saves the post-episode value. Episode 1 itself uses 1.0. Since decay is by episode, policies with longer episodes experience different numbers of exploratory steps before the same epsilon value.

A **seed** initializes a pseudorandom sequence. Training episode i (zero-based) resets with training seed+i. Seeds affect initialization, exploration, replay sampling and simulator starts. Equal seed numbers do not force different architectures to have equal weights, action trajectories or replay contents. Global generators are reseeded by construction, so simultaneous interactive experiments in one process can interfere. The saved CPU reproducibility check supports repeatability in that setup, not a universal promise across hardware/library versions.

**Termination** means the simulated task actually ended, for example a crash, leaving bounds or resting on the surface. **Truncation** means an external time cutoff. Both stop the episode loop, but only termination suppresses future value. `entrenar_agente` already passes `terminado` alone to `aprender`. The 1,000-step training budget and environment limit bound attempts. Replay retains the actual final next observation rather than a reset observation.

**Training** changes parameters using replay, explores, and invokes `fin_episodio`. **Evaluation** tests a fixed policy: `explorar=False`, no `aprender`, no replay writes, no epsilon schedule changes. The `explorar and ...` short circuit avoids consuming exploration randomness during greedy selection. Evaluation stopping uses termination or truncation. These are final fixed-budget policies, not the best intermediate checkpoints selected retrospectively.

## 8. Person A: what A0, A1 and A2 actually compare

| Configuration | Hidden layers | Learning rate | Parameters |
|---|---|---|---|
| A0 | 64,64 | .001 | 4,996 |
| A1 | 128,128 | .0005 | 18,180 |
| A2 | 64,64 | .0005 | 4,996 |

A **controlled comparison** holds planned settings fixed while changing a chosen factor. A0 versus A1 changes two factors together, so it measures a configuration difference but cannot attribute that difference separately to architecture or learning rate. A0 versus A2 changes learning rate with architecture fixed. A2 versus A1 changes width with learning rate fixed. These latter comparisons isolate the configured factor conditional on these budgets, seeds and other settings; trajectories, replay samples and realized step counts still diverge as consequences of learning.

Missing is the wide network at lr=.001. Therefore this is not a complete two-by-two design and cannot determine all interactions between width and learning rate. It also does not establish an optimal learning rate, universal superiority, or a mechanism explaining failure. A **hypothesis** is a plausible explanation still requiring testing: lower learning rate may learn useful control too slowly within 300 episodes. More steps for A2 and weaker results are compatible with that hypothesis but do not prove it.

## 9. Saved measurements and how cautiously to interpret them

Every trained row below used 300 episodes, at most 1,000 steps each, no early stopping, and greedy evaluation on reset seeds 10000–10199. **Mean** is arithmetic average; **standard deviation** describes spread of episode returns, here population convention `ddof=0`. It is not uncertainty across independent training runs.

| Config / training seed | Eval mean | Eval std | Successes / 200 | Steps | Updates | Train / eval seconds |
|---|---:|---:|---:|---:|---:|---:|
| A0 / 42 | 167.22 | 103.12 | 113 (56.5%) | 152,500 | 37,876 | 31.57 / 6.77 |
| A0 / 1042 | 209.76 | 76.33 | 146 (73%) | 136,486 | 33,872 | 83.26 / 9.26 |
| A1 / 42 | 142.05 | 102.47 | 65 (32.5%) | 156,033 | 38,759 | 34.31 / 8.28 |
| A1 / 1042 | 33.69 | 112.97 | 33 (16.5%) | 144,640 | 35,911 | 56.92 / 11.42 |
| A2 / 42 | −7.43 | 23.87 | 0 (0%) | 185,876 | 46,220 | 85.97 / 14.30 |
| A2 / 1042 | 7.69 | 59.03 | 6 (3%) | 171,781 | 42,696 | 78.95 / 15.44 |
| Random / action seed 42 | −183.17 | 111.96 | 0 (0%) | no training | 0 | 0 / .65 |

Sources: [consolidated evidence](../results/a2_controlled_comparison/evidence.json), individual summaries linked through the source map below, and the original [random summary](../results/comparison_20261027T001236_958528Z/random_seed42/summary.json). The random policy has an action seed, not a learned training replicate.

**Observation:** A0 beats A1 and A2 on both matched training seeds in evaluation mean and success. A1 beats A2 on both. Averaging the two policy means gives A0 188.49, A1 87.87, A2 .13; these are descriptive averages of only two trained instances. A0 and A1 exceed the assignment's >10% rule on each saved seed; A2 does not. Exactly 20/200 would be 10%, so at least 21/200 is needed.

**Conclusion within scope:** A0 is the strongest provisional choice among these saved configurations. Neither a statistical significance claim nor a reliable ranking for all future training seeds follows. Two hundred evaluation episodes assess scenarios for one policy; they are not 200 independent training replicates. Shared evaluation seeds make comparisons more comparable, but policies still produce different trajectories. Development evaluations used to compare settings should not be described as untouched final tests.

### Reading the learning curves

The saved curves plot raw returns and causal moving averages, with both episode and cumulative-step axes. A **moving average** smooths a window of recent values; here up to 100 episodes, with shorter windows near the beginning. Smoothing can hide failures and lag recent changes. These are training returns under changing exploration, not repeated greedy evaluations of successive checkpoints.

To ground the curve discussion, these are means calculated from the saved CSVs in three nonoverlapping blocks, not new experiments:

| Config / seed | Episodes 1–100 | 101–200 | 201–300 |
|---|---:|---:|---:|
| A0 / 42 | −114.38 | 10.34 | 137.41 |
| A0 / 1042 | −108.75 | 10.10 | 179.83 |
| A1 / 42 | −124.86 | 38.22 | 87.36 |
| A1 / 1042 | −130.94 | 21.77 | 199.43 |
| A2 / 42 | −125.74 | −24.58 | −16.37 |
| A2 / 1042 | −116.11 | −27.39 | −8.95 |

**Observation:** every run's block averages improve over its initial block, but final evaluation differs substantially. A1/1042's high final training-block average and low final greedy evaluation mean illustrate why a training curve cannot replace evaluation. This discrepancy alone does not identify its cause.

A2's final-block mean episode lengths are 975.01 and 894.78 steps, close to the 1,000-step cap, despite low returns. **Conclusion:** longer episodes do not establish successful landing. **Hypothesis:** delayed landing or prolonged hovering may contribute; aggregate length/return alone cannot identify the motion.

### Behavior evidence, without making up a trajectory story

Saved greedy recordings and JSON traces exist for A0/A1 trained with 42 on environment seeds 20000, 20001 and 20002. [episodes.csv](../recordings/greedy_seed42_20261029T233216_825512Z/episodes.csv) records A0 returns 226.65, 258.72, 260.07 and lengths 583, 977, 341; A1 returns 287.62, 224.54, 143.67 and lengths 247, 693, 981. All six terminated; none truncated. A1's last episode misses the return threshold even though it terminated. A near-cap length is not automatically a time-limit termination.

These three predefined scenarios are illustrative, separate from the 200 evaluation scenarios. They do not supersede aggregate performance. A1 can outperform A0 on an individual scenario while losing on the aggregate. To make visual claims such as oscillation, tilt, hovering or engine chatter, inspect the saved GIF/contact sheet and corroborate with position/velocity/action traces. This guide does not assert those motions from return alone. The notebook's label “crashes (R<−50)” is a return-based proxy, not a physical crash detector; `terminated` alone also does not classify every failure cause.

### Runtime and uncertainty

Equal episode budgets are not equal interaction or optimization budgets. A2 takes more steps and updates. Runtime is wall-clock time and includes different workload contention: the A2 notes explicitly record concurrent processes and different conditions for older baselines. Thus the measured seconds cannot isolate hardware efficiency or architecture cost. Seed-42 A0/A1 throughput was about 4,831/4,548 steps/s; disposable pilot throughput near 9,000 steps/s did not predict full-run speed. Keep measured timing separate from a causal speed claim.

Two training seeds leave substantial uncertainty. Evaluation standard deviations measure scenario spread; they are not confidence intervals for the average performance of the training algorithm. Future statistical comparisons should respect training seed as the replicate, avoid pooling all episode returns as independent trained agents, and distinguish descriptive paired differences from evidence of generalization.

## 10. Target-off prototype: what exists

An **ablation** retrains a system after removing/changing a component to measure its contribution. `use_target=False` creates no `red_objetivo`; `_calcular_objetivos` uses `red_q` instead, still under `no_grad`. Gamma stays .99; replay, minibatches, loss and optimizer remain. It removes the separate frozen bootstrap network, not future reward and not detached targets.

The disposable 20-episode seed-42 pilot actually learned: target-on made 213 optimizer updates over 1,850 steps, target-off 211 over 1,842. Their 10-episode evaluation means were −134.11 and −226.71, both with zero successes. The synthetic correctness check also drives target-off through two updates and checks no target attribute exists and loss is finite.

**Conclusion:** the switch works through real learning. **Limitation:** this is not a full-budget, multi-seed performance ablation. It cannot establish how much a target network helps. Different interaction counts emerge even under the same short episode budget. Do not present the prototype as a completed final ablation study.

## 11. Correctness checks, artifacts and loading

`verificar_dqn` in notebook section 6 uses disposable agents. Saved executed notebooks show the checks passed. No checks or environment runs were rerun for this explanation.

The checks test `[64,4]` output, correct `gather` indexing, finite loss and online weight changes, equal-but-independent initial target storage, target freezing/no gradients and hard copies, known terminal/nonterminal targets, detached targets, replay observation copies, epsilon endpoints, and lack of greedy-action side effects on weights/counters/replay/RNGs. They check target-off updates and reproducibility of two short real CPU runs with identical seeds. These are semantic checks, not proof of convergence or comprehensive test coverage. Short reproducibility checks use altered warmup/sync settings to reach updates quickly.

The historical status records additional CSV counts, seed/step/update schedule checks and reloading both original seed-42 policies to reproduce their first five evaluation returns. That statement applies to the documented original A0/A1 reload check, not automatically every later checkpoint. The selected-run script reuses notebook definitions but skips the full notebook verification cell and demonstrations; a selected run's CSV evidence is not proof that the entire notebook was reexecuted for it.

| Saved artifact | What it tells you |
|---|---|
| `config.json` | Settings, seeds, environment, versions, CPU/device and hardware metadata |
| `training.csv` | Per-episode return, reset seed, length, cumulative steps, post-episode epsilon |
| `evaluation.csv` | Individual evaluation seeds, returns and threshold outcomes |
| `summary.json` | Aggregates, actual steps, update counts and timings |
| `online_weights.pt` | Online network's parameter state dictionary |
| PNG curves | Views of the saved numerical evidence |
| executed notebooks / selected-run logs | Outputs for the execution path used |
| recording GIF/contact PNG/JSON | Illustrative motion, trace and checkpoint identity |
| comparison manifest | SHA-256 file fingerprints for recorded evidence |

A **checkpoint** is saved model state. Here it stores policy weights only. Reconstruct the same architecture from `hidden_dims`, load weights strictly on CPU, and choose actions greedily with matching environment settings. The recording script demonstrates this without training. Target-off construction is sufficient for policy inference because actions use the online network only.

Exact training continuation would also require Adam moments, replay contents, target weights, counters, epsilon/episode schedule, RNG states and environment state as applicable. These are not all stored. Loading online weights into a fresh target-on agent also leaves its initially copied target stale unless explicitly synchronized; that does not affect online greedy inference but matters if training is resumed. A successful load proves shape compatibility, not equivalent environment behavior across versions.

### Source map for the six trained configurations

- Original A0/A1, seed 42: [comparison_20261027T001236_958528Z](../results/comparison_20261027T001236_958528Z/).
- A0, seed 1042: [comparison_20261029T233234_746501Z](../results/comparison_20261029T233234_746501Z/).
- A1, seed 1042: [comparison_20261029T233420_927075Z](../results/comparison_20261029T233420_927075Z/).
- A2, seed 42: [comparison_20261029T233249_251868Z](../results/comparison_20261029T233249_251868Z/).
- A2, seed 1042: [comparison_20261029T233253_026919Z](../results/comparison_20261029T233253_026919Z/).
- Consolidation and caveats: [A2 notes](../results/a2_controlled_comparison/NOTES.md), [evidence](../results/a2_controlled_comparison/evidence.json), [manifest](../results/a2_controlled_comparison/manifest.json).
- Disposable prototype: [pilot directory](../results/pilot_20261027T001204_313174Z/).

## 12. Unfinished work and combining A, B and C

Policy slices (views of chosen actions while varying selected observation dimensions and fixing the others) remain unimplemented in section 10. The full required ablation study remains unimplemented in section 11: at least three component variants, multiple training seeds, appropriate statistical comparisons. Enabling `RUN_FINAL_EXPERIMENTS` raises explicit errors. Architecture/rate comparisons and a target-off pilot do not by themselves complete that requirement. The team's own final report also remains separate from this study material.

Person A's focus is architecture and learning rate. Person B's replay focus connects to capacity, sampling, warmup, batch size and update frequency; those are related but distinct settings. Person C's exploration focus connects to epsilon shape, floor, decay unit and decay duration. Do not claim B or C experiments are complete based on this folder: their measured contributions were not established here.

For integration, first agree on one complete baseline and shared training/evaluation seeds and budgets. Preserve each person's actual settings and evidence. Compare one planned change at a time, retraining fresh agents. Record actual steps/updates as well as episodes. Replay changes affect data reuse; exploration changes affect the experience entering replay; architecture/rate changes affect how that experience becomes values. Therefore combining separately promising settings can introduce interactions and needs a fresh validation experiment rather than an assumption that gains add up. Decide jointly which three removals qualify as the required ablations. Use common evaluation scenarios and additional training seeds; reserve a later untouched evaluation set if doing further selection.

## 13. Questions to practice until you can answer without reading

1. **Why four outputs rather than one?** One pass estimates every discrete action value; `argmax` selects a decision, `gather` selects the action used in each replay example.
2. **Are Q-values probabilities?** No. They estimate discounted reward and need no normalization to sum to one.
3. **Why learn from old actions?** Q-learning can learn from actions generated by an older exploratory policy; replay reuses them. This is called off-policy learning. Old data can still be correlated or unhelpful.
4. **Where are the correct action labels?** There are none. Reward plus bootstrap estimates constructs a target for the action actually taken.
5. **Why not propagate through the target?** This update treats its bootstrap reference as fixed; allowing derivatives through it changes the learning rule, including when the online network supplies both branches.
6. **Frozen target versus detached target?** Frozen target parameters persist between copies; detachment prevents a derivative path in the current update. They solve related but distinct problems.
7. **Does target-off mean gamma zero?** No. It keeps future value and changes which network provides it.
8. **Does clipping replace Huber?** No. Huber shapes the error penalty; clipping limits the resulting full gradient norm before Adam.
9. **Why is 200 evaluation episodes not enough to settle the comparison?** They measure one fixed trained policy's scenarios. Training itself is random; there are only two saved trained instances per configuration.
10. **Did the larger network help?** A1 beats equally low-rate A2 on both saved seeds. A0 beats A1, but that comparison also changes learning rate. Neither statement proves width always helps or hurts.
11. **Why can a smaller rate perform worse?** Smaller steps can fail to reach useful behavior within a fixed budget, but that is a hypothesis here, not an identified cause.
12. **Why can long survival be poor?** Delay and engine costs can accumulate without successful landing; A2 has near-cap lengths and low returns. Length alone does not reveal the motion.
13. **Why not report only success?** Success is a threshold and loses information about how far failures fall below it; means and distributions add context. Neither metric alone tells the full story.
14. **What does seeding guarantee?** Repeatable pseudorandom sequences under controlled conditions, supported by a short local check; not identical trajectories between different algorithms or universal cross-platform equality.
15. **Can I resume from the checkpoint?** You can restore the greedy policy, but not exactly resume its training history from online weights alone.
16. **What is your strongest honest contribution statement?** I can explain the implemented DQN mechanisms, the architecture/rate comparisons and saved checks; A0 leads in the two available matched seeds, while final policy analysis and full ablations remain unfinished.

Self-check: trace `elegir_accion → env.step → aprender → _actualizar_red → _calcular_objetivos`, point to the terminal mask and `gather`, calculate 4.97 by hand, and explain why your evidence supports a provisional decision rather than a universal claim.
