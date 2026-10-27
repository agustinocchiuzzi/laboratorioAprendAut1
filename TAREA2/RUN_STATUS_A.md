# Person A — implementation and measured runs

Completed locally on October 26, 2026 (America/Montevideo). Result directory names
use UTC, so their date is October 27.

The [executed comparison notebook](comparison_results.ipynb)
contains code, checks, learning curves, and evaluation outputs. The editable
[working notebook](lander.ipynb) defaults to pilot mode. The supplied
course notebook remains unchanged. See [README_A.md](README_A.md) for commands.

## Completed

- Self-contained `AgenteDQN`: replay, Adam/Huber updates, gradient clipping,
  terminal masking, epsilon-greedy actions and episode schedule, hard target copies.
- `use_target=False`: no separate target network; real learning updates verified.
- Correctness checks passed: batch/gather shapes, parameter updates, target storage
  independence/freezing/copying, terminal/nonterminal targets, detached targets,
  replay observation copies, epsilon schedule, evaluation side effects, and
  reproducibility of a short real environment run.
- Full notebook executed in a fresh Python 3.12 kernel without errors, including
  explicit skips for optional GIF and deferred sections 10/11.
- Main A0/A1 comparison: 300 episodes each, training seed 42, final checkpoints,
  no early stopping, and greedy evaluation on common seeds 10000–10199.
- Raw numerical results, configs, hardware/package versions, weights and curves
  saved. Both saved policies reloaded and reproduced their first five recorded
  evaluation returns. CSV counts, seeds, steps and optimizer-update schedules checked.

## Measured evidence (one training seed)

Main evidence: [comparison directory](results/comparison_20261027T001236_958528Z).
Evaluation standard deviations below use `ddof=0`.

- **A0**: mean return **167.22**, std **103.12**; **113/200 successes (56.5%)**.
  152,500 training steps; 37,876 optimizer updates; 31.57 s training, 6.77 s evaluation.
  [Summary](results/comparison_20261027T001236_958528Z/A0_seed42/summary.json),
  [learning curves](results/comparison_20261027T001236_958528Z/A0_seed42/learning_curves.png).
- **A1**: mean return **142.05**, std **102.47**; **65/200 successes (32.5%)**.
  156,033 training steps; 38,759 optimizer updates; 34.31 s training, 8.28 s evaluation.
  [Summary](results/comparison_20261027T001236_958528Z/A1_seed42/summary.json),
  [learning curves](results/comparison_20261027T001236_958528Z/A1_seed42/learning_curves.png).
- **Random**: mean return **−183.17**, std **111.96**; **0/200 successes**.
  No training; 0.65 s evaluation.
  [Summary](results/comparison_20261027T001236_958528Z/random_seed42/summary.json).

Both DQNs exceed the specified >10% success threshold on this development set.
A0 leads on the planned preliminary metric, greedy mean return. These are one-seed
measurements; they do not establish reliability across training seeds or isolate
architecture from learning rate. Equal episode counts produced different step counts.

Disposable [pilot evidence](results/pilot_20261027T001204_313174Z) is separate.
Each pilot used 20 training episodes and 10 evaluation episodes. Target-on A0 made
213 updates; target-off made 211 updates. This checks that the switch runs through
actual learning, not its effect in a controlled full-budget ablation. Neither pilot
had a successful evaluation episode. Initial pilot throughput was about 9,000
steps/s; full-run throughput fell to about 4,500–4,800 steps/s, illustrating why
pilot projections need rechecking.

## What you need to do

1. Read the working agent and checks; be ready to explain its Bellman update,
   replay, target mask, target synchronization, and exploration schedule.
2. Open the executed comparison notebook and saved curves. Write your **own**
   meeting observations: improvement/instability, success versus mean return,
   runtime, uncertainty, and the next single experiment. Do not treat this status
   record as the team's report or substitute for inspecting the learning curves.
3. Optional before Thursday: add 1042 to `SEMILLAS_ENTRENAMIENTO` and run the
   comparison again for both configurations. To run only the additional seed, use
   `[1042]`; keep the existing seed-42 evidence alongside the new directory.
4. Bring the executed notebook plus the complete main result directory.

Still required for November 4: policy slices, at least three proper ablations with
multiple training seeds and statistical comparisons, and the team's own IEEE report.
Sections 10/11 are explicitly deferred; this is a Thursday checkpoint implementation.
