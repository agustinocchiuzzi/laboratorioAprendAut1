# Thursday implementation — Person A

**Already executed:** both the pilot and the full seed-42 A0/A1 comparison.
Read [RUN_STATUS_A.md](RUN_STATUS_A.md) for measured results and your next steps.
The commands below let you repeat or extend those runs.

Work in `LunarLander_DQN_A.ipynb`; the course original is untouched. The DQN,
all agent helpers, and correctness checks are in the working notebook. Course
training/evaluation functions and plotting utilities are preserved.

## Run here

The implementation and evidence are versioned in this repository's `TAREA2/`.
From the repository root, set up a Python 3.12 environment once:

```sh
cd TAREA2
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For the exact package versions used in the recorded runs, use
`requirements-lock.txt` instead. That lock records the original macOS environment.
All remaining commands below run from `TAREA2/`:

```sh
.venv/bin/python scripts/run_notebook.py --mode pilot
```

This executes the notebook in a fresh kernel, checks DQN correctness, then trains
20-episode A0 and target-off pilots with short 10-episode evaluations. These are
**disposable pilots**, not the main comparison. The executed notebook is saved as
`LunarLander_DQN_A_pilot_executed.ipynb`, including outputs. No packages are
installed by running the notebook. Prior result directories are never overwritten.
The executed notebook file for a mode is replaced by the next execution of that mode.

After inspecting the pilot, run the actual Thursday comparison:

```sh
.venv/bin/python scripts/run_notebook.py --mode comparison
```

This creates fresh A0 and A1 agents with training seed 42, runs 300 episodes each
(maximum 1,000 steps, no early stopping), and evaluates each and a fresh random
agent on seeds 10000–10199. It saves the executed comparison notebook. Keep the
machine awake while it runs. The notebook runner prints cell progress; detailed
training output is captured in the executed notebook.

To use a notebook editor, select `.venv/bin/python`, open the working notebook,
and Run All. The controls cell defaults to `MODO="pilot"`. Change it to
`"comparison"` for the main runs. Add 1042 to `SEMILLAS_ENTRENAMIENTO` before
running the comparison if time permits a second seed for both configurations.
Avoid running the CLI and interactive notebook concurrently: both reseed global RNGs.

## Evidence

Each execution writes a unique `results/<mode>_<UTC timestamp>/` directory.
Each trained run has `config.json`, `training.csv`, `evaluation.csv`,
`summary.json`, `online_weights.pt`, and a plot against environment steps.
The random baseline has its config, evaluation CSV, and summary. Configs include
package versions, hardware, CPU threads, settings and seeds. Summary standard
deviation uses `ddof=0`. Epsilon in training CSV is the post-episode value.
Weights restore a policy for evaluation; they do not resume training exactly.

Rebuild both episode and step plots from saved numerical results:

```sh
.venv/bin/python scripts/plot_results.py results/<mode_timestamp>
```

There is no claim of equal compute: compare recorded steps and runtime. A0/A1
change architecture **and** learning rate, so this comparison cannot separate
their effects. Main evaluation requires at least 21/200 returns >=200 to exceed
10% success. Pilot evaluations are much smaller and should not select a policy.

## Your work for Thursday

1. Read the agent and checks until you can explain replay, `gather`, the terminal
   mask, detached Bellman targets, epsilon decay, and hard target copies.
2. Inspect the already-executed comparison notebook. Repeating it is optional. Both
   configurations used the same episode budget. If shortening it, edit `EPISODIOS` once before either
   main run; keep one common budget and record the reason.
3. Review both learning curves and evaluation summaries. Choose provisionally by
   greedy mean return, considering success count and runtime too. Write your own
   observations about weak/unstable performance and the limits of one training seed.
4. Bring the executed notebook and the complete result directory. If time permits,
   repeat both configurations with training seed 1042.

Section 9 GIF is optional (`RUN_GIF=False`). Sections 10 (policy slices) and 11
(the full ablation study) print an explicit deferral. Setting
`RUN_FINAL_EXPERIMENTS=True` raises an error until those sections are implemented.
These are required for the final submission, together with several training seeds,
statistical comparisons and the team's own IEEE report. This implementation does
not write the report or supply the team's scientific interpretation.

## Recreate the environment

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` records this machine's full installed environment;
`requirements.txt` lists the direct dependencies for a different platform.
On Colab, use the commented `%pip` line in the dependency cell instead. Run the
working notebook there; the CLI helper is optional.
