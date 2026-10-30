# Learning-rate extension: A3 (0.005)

User-requested exploratory experiment, executed October 29, 2026. A3 uses the unchanged DQN agent with hidden_dims=(64,64), lr=.005. It was added to the notebook configuration dictionary and selected-run CLI choices. Two fresh policies trained sequentially with seeds 42 and 1042: 300 episodes each, maximum 1000 steps, no early stopping. Greedy final-policy evaluation used the same 200 development reset seeds 10000–10199. Existing A0/A2 policies were not retrained.

| Learning rate | Seed 42 mean / success | Seed 1042 mean / success | Mean of two policy means |
|---|---|---|---|
| .0005 (A2) | -7.43 / 0% | 7.69 / 3% | .13 |
| .001 (A0) | 167.22 / 56.5% | 209.76 / 73% | 188.49 |
| .005 (A3) | 203.32 / 64% | 99.87 / 9% | 151.60 |

Observation: A3 beats A0 by 36.10 return points for seed 42 but loses by 109.89 for seed 1042. A3 training used 127997/134811 steps, 31750/33453 optimizer updates and 28.44/30.08 seconds; evaluation took 7.67/9.85 seconds. A3 does not exceed the >10% success requirement on seed 1042 (18/200). Its evaluation episode-return standard deviations are 78.31 and 86.84 (ddof=0), not uncertainty across training replicates.

Conclusion limited to these observations: .005 can outperform .001 for an individual training seed, but it is not a consistent improvement. A0 remains the provisional choice by average evaluation return and success across the two seeds. No statistical significance, global optimum or learning-rate boundary is established. Worse results at .0005 do not imply all smaller rates are worse, and rates above .001 need not improve monotonically. More sensitivity at a higher rate is a plausible hypothesis; these endpoints do not prove the training mechanism.

Controlled factor: configured learning rate only, at fixed architecture and other saved experimental settings. Actual trajectories, replay samples, episode lengths, steps and optimizer counts can differ as a consequence. These are equal episode budgets, not equal step or compute budgets. Historical runtime contention differs, so measured wall time does not isolate computational efficiency. The evaluation scenarios are development data already used for tuning, not an untouched final test.

Validation performed: all six A0/A2/A3 runs have 300 training and 200 evaluation rows; expected training/evaluation seed sequences; matching architecture and substantive settings except learning rate; consistent summed lengths/cumulative steps, update schedules, return mean/std and success counts. Raw configs, CSVs, final online weights and runner logs remain in unique source directories listed in evidence.json. A3 learning curves are saved in each run; learning_rate_curves.png compares causal moving averages by episodes and cumulative steps. manifest.json fingerprints source artifacts. The selected-run path does not execute the full notebook correctness-check cell; no claim that it was rerun here. Checkpoints remain policy weights only.

Reproduce with the project environment from the workspace root:

```
.venv/bin/python laboratorioAprendAut1/TAREA2/scripts/run_notebook.py --mode comparison --config A3 --seed 42
.venv/bin/python laboratorioAprendAut1/TAREA2/scripts/run_notebook.py --mode comparison --config A3 --seed 1042
```

Each invocation creates a new source directory. The saved analysis script documents the calculations and refuses to overwrite this comparison directory. The earlier study guide predates A3; use this addendum for the new measurements.
