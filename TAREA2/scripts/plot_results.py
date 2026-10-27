"""Rebuild learning curves from saved CSVs without training a policy."""
import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='One timestamped results directory')
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    paths = sorted(args.results.glob('*/training.csv'))
    if not paths:
        parser.error('No training.csv files found in child run directories')
    for path in paths:
        with path.open() as source:
            rows = list(csv.DictReader(source))
        rewards = np.array([float(row['return']) for row in rows])
        smooth = np.array([rewards[max(0, i - 99):i + 1].mean() for i in range(len(rewards))])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, column, label in zip(axes, ['episode', 'cumulative_steps'], ['Episode', 'Cumulative environment steps']):
            x = [int(row[column]) for row in rows]
            ax.plot(x, rewards, alpha=.3, label='Raw training return')
            ax.plot(x, smooth, label='Causal moving average (up to 100 episodes)')
            ax.axhline(200, color='green', linestyle='--', linewidth=1)
            ax.set(xlabel=label, ylabel='Return', title=path.parent.name)
            ax.legend(fontsize=7)
        fig.tight_layout()
        output = path.parent / 'learning_curves.png'
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(output)


if __name__ == '__main__':
    main()
