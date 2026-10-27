# Start here: what exists and what to learn

This work lives in `TAREA2/` of
https://github.com/agustinocchiuzzi/laboratorioAprendAut1 on branch `thiago`.
`TAREA 1/` contains the previous assignment. The supplied LunarLander notebook and
PDF in this directory are unchanged. Our working notebook, evidence, scripts and
notes have been added alongside them.

Work in `LunarLander_DQN_A.ipynb`; read
`LunarLander_DQN_A_comparison_executed.ipynb` to inspect the completed run and plots.
Do not accidentally start from the unimplemented `LunarLander_DQN_Tarea.ipynb`.

## The problem, before the terminology

The simulator provides eight numbers: horizontal position, altitude, two velocities,
angle, angular velocity, and two leg-contact indicators. The agent chooses one of
four discrete actions: do nothing or fire one of the engines. The environment then
returns a new observation and a reward. Rewards are feedback from the simulator;
there is no dataset of labelled correct engine choices.

An **environment step** is one action followed by one simulator response. An
**episode** is one attempt from reset until the attempt ends or reaches its time
limit. **Return** is the sum of rewards in that episode. In this assignment an
episode counts as successful if return is at least 200. A crash and the evaluation
success threshold are different concepts: merely surviving does not guarantee 200.

A **policy** is the agent's rule for choosing actions from observations. Here its
learned rule is: choose the action with the largest predicted Q-value.

## What I implemented, with the exact terms

Everything specific to the agent stays inside `AgenteDQN` in section 6.

- **DQN (Deep Q-Network):** a neural network approximates `Q(s, a)`, the expected
  discounted future return from taking action `a` in state `s` and following the
  policy afterwards. It predicts four values, one per action. These are return
  estimates, not probabilities. The final layer has no softmax.
- **MLP (multilayer perceptron):** a fully connected neural network. A0 has layer
  sizes `8 → 64 → 64 → 4`; A1 has `8 → 128 → 128 → 4`. Hidden layers use ReLU,
  `max(0, x)`. **Weights/parameters** are the numbers learned in these layers.
- **Epsilon-greedy exploration:** during training `elegir_accion` chooses a random
  action with probability epsilon; otherwise it chooses the largest Q-value
  (**greedy action**, `argmax`). Epsilon drops linearly from 1.0 to 0.05 over 200
  completed episodes. At 0.05, about 5% of training decisions explore at random.
  Evaluation always chooses greedily.
- **Transition:** `(state, action, reward, next_state, terminated)`, the information
  from one interaction. `aprender` saves copies of observations, so reusing an
  observation array cannot corrupt stored experience.
- **Experience replay / replay buffer:** memory for the latest 50,000 transitions.
  Instead of updating from just the latest transition, learning samples 64 distinct
  transitions uniformly at random. This reuses experience and reduces the temporal
  correlation within training batches. A **minibatch** is that set of 64 samples.
- **Warmup:** collect 1,000 interactions before updates start. Then take one
  **optimizer update** every four environment steps. An optimizer update adjusts
  network weights; it is different from taking an action in the environment.
- **Bellman target:** the desired prediction for each sampled action is
  `y = reward + 0.99 * (1 - terminated) * max Q_bootstrap(next_state)`.
  **Gamma / discount factor**, here 0.99, reduces the contribution of later rewards.
  Example: reward=2 and maximum next-state Q=3 give `y=4.97` for a nonterminal
  transition; a terminal transition gives `y=2`.
- **Bootstrapping:** using another Q estimate to construct this target, rather than
  waiting for a complete future trajectory. The word “target” here means the value
  used for learning; it does not mean that we know the true optimal return.
- **Target network (`red_objetivo`):** a separate copy of the online/trainable
  network (`red_q`) supplies the next-state Q estimates. It starts with identical
  weights and is refreshed every 1,000 environment steps. Between copies it stays
  frozen, making the learning target less rapidly changing.
- **Target-off switch:** `use_target=False` uses `red_q` itself for bootstrapping
  and creates no target network. Future rewards remain included. This changes one
  component; setting gamma to zero would be a different experiment.
- **Detached target / `torch.no_grad()`:** treat `y` as fixed during each update.
  Gradients update the online prediction of the sampled action; they do not flow
  through the next-state estimate, including in target-off mode.
- **`gather`:** selects the Q-value of the action actually taken from the four
  predicted values for each sampled state. The loss compares those chosen values
  with their corresponding Bellman targets.
- **Loss:** Huber / SmoothL1 measures prediction error, behaving quadratically near
  zero and linearly for larger errors. **Backpropagation** computes gradients of
  that loss with respect to the network weights. **Adam** uses these gradients to
  adjust weights. **Learning rate** controls update scale: A0 uses 0.001; A1 uses
  0.0005. **Gradient clipping** limits their combined norm to 10.
- **Termination versus truncation:** a genuine terminal state has no future value
  in the target. A time-limit cutoff (**truncation**) still bootstraps from the
  actual next observation; the supplied trainer already passes the correct flag.
- **`fin_episodio`:** counts completed training episodes and updates epsilon.

Training and evaluation loops and the supplied plotting utilities were already
provided by the course; I preserved their definitions. I added experiment call
sites, checks, controls and saving around them.

## What ran and what the measurements mean

A **seed** initializes pseudo-random generators. Seed 42 was used to train both
models; each environment episode resets with the training seed plus its index.
Using one training seed is one trained instance per configuration, not evidence
about how stable training is across many independent runs.

Both models trained for 300 episodes, with at most 1,000 steps per episode and no
early stopping. Equal episodes are not equal steps: A0 used 152,500 steps; A1 used
156,033. They were then evaluated without exploration or learning over 200 complete
episodes with reset seeds 10000 through 10199. A fresh random baseline faced the
same reset seeds. These are development evaluations used for comparing settings.

- A0: average return 167.22; 113/200 successful episodes (56.5%).
- A1: average return 142.05; 65/200 successful episodes (32.5%).
- Random: average return −183.17; 0/200 successful episodes.

Both exceed the assignment's strictly-greater-than-10% success threshold on these
runs. A0 leads on the planned metric, greedy mean return. Architecture and learning
rate changed together, so these results cannot establish which individual change
caused the difference. Two hundred evaluation episodes measure different scenarios
for one trained policy, not 200 independently trained policies.

A separate 20-episode target-off pilot completed 211 optimizer updates. This verifies
that disabling the target network works through real learning. It does not establish
whether target networks improve performance; that requires a matched full ablation.

**Ablation** means retraining after removing one component and comparing it with the
complete agent, while holding other settings fixed. The final assignment needs at
least three such variants, several training seeds and statistical comparisons.

## What was checked and saved

Checks cover tensor shapes, chosen-action indexing, online weight changes, frozen
and independent target weights, target copies, detached Bellman targets, terminal
masking, observation copies, epsilon scheduling, greedy action side effects,
target-off updates, and a reproducible short real run. Both full saved models were
reloaded and reproduced the first five recorded evaluation returns.

Each trained run has `config.json` (settings/software/hardware), `training.csv`
(per-episode return, length, steps and epsilon), `evaluation.csv` (individual
returns/seeds), `summary.json` (aggregate measurements), `online_weights.pt`
(learned weights) and plots. These files are under `results/`; exact paths are in
`RUN_STATUS_A.md`. A **checkpoint** here restores the policy for evaluation. It
is not a complete snapshot for exact continuation of training.

## What you should do next, in order

1. Read notebook sections 2–5 to understand the environment, agent interface, and
   the supplied training/evaluation loops. Then read the DQN class in section 6
   using the definitions above. Identify the Bellman formula in the code.
2. Trace one transition by hand: observe state, select action, receive reward and
   next state, store it, and later use it in a sampled learning update. Explain
   why a terminal target is just the immediate reward.
3. Inspect the already-executed comparison notebook, especially both raw learning
   curves and greedy evaluations. A **moving average** smooths noisy returns;
   ours uses up to the latest 100 episodes, with smaller initial windows. It is
   not an additional experiment or an evaluation score.
4. Write your own meeting notes: what you see in the curves, which metric favors
   which model, limits of one seed and two changed settings, and your next test.
5. If there is time, set `SEMILLAS_ENTRENAMIENTO=[1042]` in the working notebook
   and run comparison mode to add one independent training run per configuration.
   Keep the seed-42 evidence. Running default pilot mode will not do this.
6. Coordinate with your teammate using branch `thiago`. The implemented files and
   recorded results are already together in `TAREA2/` on that branch.

To repeat the existing seed-42 comparison from `TAREA2/`:

```sh
.venv/bin/python scripts/run_notebook.py --mode comparison
```

You do not need to retrain to read the existing results. In an editor, choose
`.venv/bin/python` as the notebook kernel (the Python process running the cells).
The editable notebook defaults to pilot mode; choose `MODO="comparison"` for main
runs. Full reproduction commands are in `README_A.md`.

Still unfinished for the final assignment: policy slices (examining chosen actions
while varying selected state coordinates), the multi-seed ablation study and
statistical tests, and the team's own IEEE report. Sections 10/11 are explicitly
skipped for Thursday. The GIF in section 9 is optional and has not been generated.
