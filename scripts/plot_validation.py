"""Regenerate readable IEEE-column figures from the recorded validation grid."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/validation"
OUTPUT = ROOT / "figuras"


def axes_pair():
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.35), layout="constrained")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Error (1 - accuracy)")
    axes[1].set_ylabel("Macro-F1 de validación")
    return fig, axes


def save(fig, name):
    fig.savefig(OUTPUT / name, dpi=300)
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7}):
        id3 = pd.read_csv(SOURCE / "id3.csv")
        positions = np.arange(len(id3))
        fig, axes = axes_pair()
        axes[0].errorbar(positions, id3.error_validacion,
                        yerr=id3.error_validacion_std, marker="o", markersize=3,
                        capsize=2, label="Validación")
        axes[0].plot(positions, id3.error_train, "s--", markersize=3, label="Train")
        axes[0].legend()
        axes[1].errorbar(positions, id3.macro_f1, yerr=id3.macro_f1_std,
                        marker="o", markersize=3, capsize=2)
        for axis in axes:
            axis.set_xticks(positions, [f"{value:g}" for value in id3.min_info_gain])
            axis.set_xlabel("min_info_gain (valores de la grilla)")
        save(fig, "id3.png")

        nb = pd.read_csv(SOURCE / "nb.csv")
        fig, axes = axes_pair()
        axes[0].errorbar(nb.m, nb.error_validacion, yerr=nb.error_validacion_std,
                        marker="o", markersize=3, capsize=2, label="NB propio: validación")
        axes[0].plot(nb.m, nb.error_validacion_sklearn, "x--", markersize=4,
                     label="CategoricalNB: validación")
        axes[0].plot(nb.m, nb.error_train, color="0.4", linestyle=":", label="Train (ambos NB)")
        axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                       ncol=2, columnspacing=0.8)
        axes[1].errorbar(nb.m, nb.macro_f1, yerr=nb.macro_f1_std,
                        marker="o", markersize=3, capsize=2, label="NB propio")
        axes[1].plot(nb.m, nb.macro_f1_sklearn, "x--", markersize=4, label="CategoricalNB")
        axes[1].legend()
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlabel("m (escala logarítmica); alpha = m/4")
        save(fig, "nb_propio.png")

        rf = pd.read_csv(SOURCE / "random_forest.csv")
        fig, axes = axes_pair()
        axes[0].set_title("Línea continua: validación; discontinua: train", fontsize=7)
        for depth, part in rf.groupby("profundidad", sort=False):
            part = part.sort_values("min_samples_leaf")
            line = axes[0].errorbar(part.min_samples_leaf, part.error_validacion,
                                   yerr=part.error_validacion_std, marker="o", markersize=3,
                                   capsize=2, label=f"Prof. {depth}")
            color = line.lines[0].get_color()
            axes[0].plot(part.min_samples_leaf, part.error_train, "--", color=color)
            axes[1].errorbar(part.min_samples_leaf, part.macro_f1, yerr=part.macro_f1_std,
                            marker="o", markersize=3, capsize=2, color=color)
        axes[0].legend(ncol=3, fontsize=6.5, loc="lower right", columnspacing=0.7)
        for axis in axes:
            axis.set_xticks([1, 5, 20, 50])
            axis.set_xlabel("min_samples_leaf")
        save(fig, "random_forest.png")


if __name__ == "__main__":
    main()
