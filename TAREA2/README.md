# LunarLander DQN — Thursday checkpoint

The implementation, experiments and results for Person A are on branch `thiago`.
The original course notebook and assignment PDF are unchanged.

- [Start here: terminology, implementation and learning order](START_HERE_A.md)
- [Measured results and remaining work](RUN_STATUS_A.md)
- [Environment setup and execution commands](README_A.md)
- [Editable notebook](LunarLander_DQN_A.ipynb)
- [Executed comparison notebook with outputs](LunarLander_DQN_A_comparison_executed.ipynb)

A0 and A1 each trained for 300 episodes with seed 42 and were evaluated over 200
shared reset seeds. Saved configs, CSVs, weights and plots are under `results/`.
The default editable-notebook mode is a short pilot; the main comparison has
already been executed and saved. See the linked instructions before rerunning.

Policy slices, the full multi-seed ablation study and statistical comparisons are
still deferred for the final submission. The team must write its own report.
