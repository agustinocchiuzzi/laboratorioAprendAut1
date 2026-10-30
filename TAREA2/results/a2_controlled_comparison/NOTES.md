# A2 experiment evidence

A2 alone was added: hidden_dims=(64,64), lr=0.0005. Fresh agents, seeds 42 and 1042, 300 episodes, maximum 1000 steps, no early stopping, unchanged DQN defaults. Greedy final-policy evaluation uses reset seeds 10000–10199. Raw CSVs, weights, configs and summaries remain in their original unique directories; manifest.json records SHA-256 hashes.

A0 versus A2 isolates learning rate at (64,64). Both available matched training seeds favor A0 in evaluation return and success. A2 versus A1 isolates architecture at lr=0.0005: both matched seeds favor A1 in mean return and success, although A1 varies substantially across training seeds (142.05 versus 33.69). No additional baseline training was launched by this experiment.

A2 late training episodes approach the step cap while returns stay near zero: longer survival does not imply successful landing. Lower learning rate may slow useful learning within this fixed episode budget; trajectories and replay data also diverge, so this does not establish a mechanism. A2 used more actual steps than the seed-42 baselines despite identical episode budgets. Larger A1 capacity helped at seed 42 under this learning rate, but this is conditional evidence.

Only two A2 training seeds were observed; the 200 reset seeds measure each fixed policy, not 200 independent training replicates. Differences across training seeds and broad evaluation return distributions limit generalization. No statistical significance or universal superiority is claimed. Runtime is measured wall time: A2 ran concurrently with other processes and older baselines ran under different contention, so these timings do not isolate computational efficiency.

Reproduce training from repository root with `.venv/bin/python laboratorioAprendAut1/TAREA2/scripts/run_notebook.py --mode comparison --config A2 --seed 42` and repeat with 1042. Each invocation creates a unique output directory. Analysis: scripts/compare_a2.py creates a new timestamped evidence directory. Raw learning curves are unsmoothed, shown by episode and actual cumulative steps.
