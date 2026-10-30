# Person A speaking outline — prompts, not a script

Aim for 5–7 minutes. Rephrase each prompt in your own words. Keep the detailed guide open for follow-up questions.

## 1. What problem are we solving? (40 seconds)

- Agent chooses among four engine actions from eight physical observations.
- One step gives a new observation and reward. One episode is an attempt. Return is its sum of rewards; success is return ≥200.
- We learn from feedback, without labels telling us the correct engine action.

## 2. What did the course give us? (30 seconds)

- Environment, interface, random reference, training/evaluation loops and plotting/GIF utilities.
- We filled the DQN agent and added checks, experiments and saved evidence.
- My focus: architecture and learning rate. Avoid claiming authorship of course utilities or library algorithms.

## 3. Explain the implementation (90 seconds)

- Q-values estimate discounted future reward; the network returns four values, not probabilities.
- `__init__`: online MLP, replay storage, Adam and optional independent target copy.
- `elegir_accion`: epsilon-greedy during training, greedy during evaluation.
- `aprender`: copy/store each transition; warm up 1,000 steps; update every four steps.
- `_actualizar_red`: sample 64 replay entries; `gather` picks the actions actually taken; mean Huber loss, backpropagation, norm clipping to 10, Adam update.
- `_calcular_objetivos`: reward + .99 × nonterminal mask × best next Q, with no gradient through the target.
- Copy online into target every 1,000 environment steps, after any update.
- `fin_episodio`: linear epsilon decay from 1 to .05 over 200 completed episodes.

## 4. Show you understand one update (40 seconds)

- Teaching example: chosen Q=4, reward=2, next maximum=3.
- Nonterminal target=2+.99×3=4.97; error=−.97; Huber contribution=.47045 before averaging over the batch.
- Terminal target=2. Time-limit truncation still uses 4.97.
- Adam changes weights using the entire batch and its history; this is not simply “replace 4 by 4.97.”

## 5. Explain my comparisons and measurements (90 seconds)

| Setting | Hidden widths | Learning rate | Eval means for training seeds 42 / 1042 |
|---|---|---|---|
| A0 | 64,64 | .001 | 167.22 / 209.76 |
| A1 | 128,128 | .0005 | 142.05 / 33.69 |
| A2 | 64,64 | .0005 | −7.43 / 7.69 |

- A0 versus A1 changes two factors. A0 versus A2 isolates the configured learning rate; A2 versus A1 isolates width at the lower rate.
- All six trained policies used 300 episodes and 200 shared greedy evaluation scenarios.
- Success counts: A0 113/146, A1 65/33, A2 0/6, each out of 200.
- A0 leads on both saved training seeds. Provisional preference, not universal superiority or significance.
- There are two independent training seeds per setting, not 200 training replicates.
- Curves improve overall; A2's late lengths approach the cap while returns stay low. Longer episodes do not imply landing success.
- Timings range from roughly 32 to 86 seconds of training under different contention and step counts. They do not isolate efficiency.

## 6. What checks and limitations should I mention? (45 seconds)

- Saved checks cover shapes, `gather`, masks, detached targets, weight updates, target independence/copies, replay copies, epsilon, evaluation side effects and short seeded reproducibility.
- Target-off is implemented and performed 211 updates in a 20-episode pilot. It is a working prototype, not a finished performance ablation.
- Saved configs, raw CSVs, weights, plots and recordings make findings inspectable. Online weights restore inference, not exact training continuation.
- Older status notes omit the newer A2 and seed-1042 evidence; use the actual saved run files.

## 7. How do we combine with B and C? (35 seconds)

- Agree on one baseline. B investigates replay; C investigates exploration; A investigates architecture/rate.
- Keep matched seeds, budgets, evaluation settings and complete artifacts. Record steps/updates too.
- Gains can interact: exploration changes replay data, replay changes updates, architecture/rate changes learning from it.
- Still unfinished: policy slices, the required full multi-seed component ablations/statistical comparisons, and our own final report.

## Quick follow-up answers

- **Why target network?** A slower-changing bootstrap reference; copied periodically.
- **Why no gradients through targets?** The current update learns toward a fixed reference, even when target-off uses the online network.
- **Why replay?** Reuse experience and reduce immediate temporal correlation.
- **Why evaluation without exploration?** Measure the fixed learned greedy policy without training side effects.
- **Did smaller learning rate fail inherently?** These two seeds and this budget performed worse; the mechanism and generality remain uncertain.
- **What can the recordings show?** Specific saved behaviors, not overall success rates. Do not infer crashes or hovering from return alone.

Practice from memory: define each term before using it, point to the method implementing it, label numbers as observations, proposed causes as hypotheses, and conclusions with their limits.

Full explanations and evidence links: [learning guide](LEARNING_GUIDE_A.md).
