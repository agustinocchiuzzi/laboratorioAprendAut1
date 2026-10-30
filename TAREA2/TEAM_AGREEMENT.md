# Prepare now, agree tomorrow

**Integration update (October 30):** A/B/C are now combined on
`integration/assignment-2`; see [INTEGRATION.md](INTEGRATION.md). All three component
switches are implemented and tested. The protocol remains a draft; the integration
does not imply that the team selected A0 or approved a final budget.

This is an editable proposal, not a team decision or a report. No new training was
started when preparing these tools. The existing agent and runner were preserved.

## 1. Reference configuration: what stays fixed

The reference is the complete agent against which each ablation is compared. It
includes the environment, architecture, learning rate, optimizer, replay settings,
update schedule, target schedule and exploration schedule. An ablation starts from
fresh random weights and changes only its named component. It does not continue
training from the reference checkpoint.

Propose A0 (64,64; lr=.001), LunarLander-v3 discrete with wind off, Adam, gamma=.99,
Huber loss, gradient norm cap 10, replay 50,000, batch 64, warmup 1,000, update every
4 steps, target copy every 1,000 steps, epsilon 1→.05 over 200 episodes. These are
our candidate settings; B and C may have evidence supporting a different choice.
Tomorrow select the complete reference using all three people's evidence and
record its source revision. Freeze it before the full ablation study.

The shared agent implements use_target, use_replay and use_exploration. The updated
manifest is preparation/integrated_job_manifest.json; team_job_manifest.json is
a historical snapshot. Current-transition-only replay-off changes effective batch
size to one while retaining warmup and update frequency; agree on that definition
explicitly and document the consequence.

## 2. Training budget: what a fair comparison spends

Propose 300 episodes per policy, at most 1,000 steps per episode, no early stopping,
and evaluation of the final checkpoint. This matches our existing calibration.
Equal episodes do not imply equal environment steps or equal compute: an agent
that stays airborne longer can use more steps. Save both steps and wall-clock
training/evaluation runtime. Do not select the best-looking checkpoint separately
for each variant. If the team chooses a shorter or longer budget, apply it to the
reference and every variant; old runs with a different budget remain calibration.

## 3. Training seeds: how often the whole training is repeated

A training seed initializes Python, NumPy, PyTorch, action sampling, and the
sequence of environment resets. Different seeds change initial weights and the
experience gathered. Each seed produces a fresh, independently trained policy.
Use the same planned seed list for every condition to create matched seed blocks.

Propose [42,1042,2042,3042,4042,5042,6042,7042] for the final study. Four conditions
(full plus three ablations) make 32 trained policies. Eight is a practical proposal,
not a guarantee of statistical power; set the seed budget after measuring compute
cost and before significance testing. A three-seed initial pilot can check the
pipeline but is not the final inferential study. Preserve failed or weak runs.
Seeds do not guarantee identical results across different package versions/devices.

## 4. Evaluation seeds: which scenarios every trained policy faces

Evaluation randomness is separate from training randomness. Run all policies on
identical reset seeds with greedy actions, no learning, and no exploration-schedule
changes. Propose the existing 200 development seeds 10000–10199. They are already
used for tuning and should be described as development evaluation.

Reserve 30000–30199 for an independent check after all settings are frozen. Those
seeds have not been evaluated by these preparation tools. Do not inspect them,
tune settings using their scores and continue calling them an untouched final test.
Record every return and success flag. Success is return>=200; to exceed 10% with
200 episodes requires at least 21 successes.

The statistical unit is a trained policy's evaluation mean. Two hundred episodes
of one policy are not 200 independently trained seeds. Decide the statistical test,
its assumptions and correction across three comparisons together before analysis.
The preparation aggregator performs descriptive calculations only.

## 5. Output format: how work can be combined and checked

Every run needs its own folder and these files:
- config.json: all settings, component switches, seed positions, package versions,
  hardware/device/threads and source revision when the reference is frozen.
- training.csv: per-episode return, length, reset seed, cumulative steps and epsilon.
- evaluation.csv: each reset seed, return and success flag.
- summary.json: completed episodes, total steps, optimizer updates, runtimes,
  evaluation mean/median/std and successes/count.
- online_weights.pt: final policy weights; architecture is in config.json.

The original saved-run schema is supported. Legacy missing use_replay and
use_exploration flags mean enabled for the existing full DQN only. Newly created
variants must document the flags and their effective settings. A weights file is
not a full snapshot for exact continuation of training.

The aggregator checks settings and raw-file consistency, rejects duplicate seeds,
reports missing policies, and distinguishes sample SD across training-seed means
from the ddof=0 SD across evaluation episodes in one policy's summary.

## Implemented preparation tools

From this workstation's TAREA2 directory (use your installed environment's Python
instead of ../../.venv/bin/python on another machine):

```sh
# Generate a reviewable list of jobs; never starts them.
../../.venv/bin/python scripts/study_tools.py plan \
  --protocol protocols/team_draft.json --output preparation/new_job_manifest.json

# Combine compatible saved runs, without retraining.
../../.venv/bin/python scripts/study_tools.py aggregate \
  --runs results/comparison_20261027T001236_958528Z/A0_seed42 \
         results/comparison_20261029T233234_746501Z/A0_seed1042 \
  --output preparation/new_A0_seed_summary.json

# Load existing weights and produce policy slices, without stepping an environment.
../../.venv/bin/python scripts/study_tools.py slices \
  --run results/comparison_20261027T001236_958528Z/A0_seed42 \
  --output preparation/new_A0_slices
```

Commands refuse to overwrite existing outputs. Example artifacts have already been
created under preparation/. Policy slices vary two dimensions while holding the
other six fixed; both chosen actions and max Q estimates are saved with numerical
arrays, fixed values and checkpoint hash. Such synthetic states may be unreachable;
use actual rollouts alongside them when interpreting behavior. Q is not a probability.

Next: edit protocols/team_draft.json together, record the agreed settings and
source revision, assign one full-reference owner and one owner per removal,
review the completed integration pilot, and start the shared full study.
