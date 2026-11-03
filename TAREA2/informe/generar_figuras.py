"""Regenera las figuras del informe con CSV y pesos guardados, sin entrenar.

Desde TAREA2: python informe/generar_figuras.py
Requiere NumPy, Matplotlib y PyTorch del entorno del proyecto.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tarea2-informe-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figuras"
VARIANTS = ["full", "no_target", "no_replay", "no_exploration"]
LABELS = ["Completo", "Sin red objetivo", "Sin replay", "Sin exploración"]


def variant(config):
    for switch, name in [("use_target", "no_target"),
                         ("use_replay", "no_replay"),
                         ("use_exploration", "no_exploration")]:
        if not config[switch]:
            return name
    return "full"


def main():
    manifest = json.loads((ROOT / "analysis/team_study/manifest.json").read_text())
    assert len(manifest["selected_runs"]) == 32
    runs = {}
    for rel in manifest["selected_runs"]:
        folder = ROOT / rel
        for filename in ["config.json", "training.csv", "online_weights.pt"]:
            actual = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
            assert actual == manifest["source_hashes"][rel][filename]
        config = json.loads((folder / "config.json").read_text())
        key = (variant(config), config["training_seed"])
        assert key not in runs
        runs[key] = folder
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6,
        "lines.linewidth": 1.1, "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(3.49, 2.35), layout="constrained")
    styles = ["-", "--", "-.", ":"]
    for name, label, style in zip(VARIANTS, LABELS, styles):
        curves = []
        for (v, seed), folder in sorted(runs.items()):
            if v != name:
                continue
            with (folder / "training.csv").open() as handle:
                values = np.array([float(r["return"]) for r in csv.DictReader(handle)])
            assert len(values) == 300
            curves.append([values[max(0, i - 19):i + 1].mean() for i in range(300)])
        array = np.array(curves)
        assert array.shape == (8, 300)
        mean, sd = array.mean(axis=0), array.std(axis=0, ddof=1)
        line, = ax.plot(np.arange(1, 301), mean, style, label=label)
        ax.fill_between(np.arange(1, 301), mean - sd, mean + sd,
                        color=line.get_color(), alpha=0.10, linewidth=0)
    ax.set(xlabel="Episodio de entrenamiento", ylabel="Retorno suavizado", xlim=(1, 300))
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "curvas_aprendizaje.pdf")
    plt.close(fig)

    torch.set_num_threads(1)
    folder = runs[("full", 42)]
    net = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64),
                        nn.ReLU(), nn.Linear(64, 4))
    net.load_state_dict(torch.load(folder / "online_weights.pt", map_location="cpu", weights_only=True))
    net.eval()
    reference = np.array([0, 0.8, 0, -0.3, 0, 0, 0, 0], dtype=np.float32)
    cuts = [
        (0, 2, (-1, 1), (-1.5, 1.5), "$x$", "$v_x$", "Posición y velocidad horizontal"),
        (4, 5, (-0.7, 0.7), (-1.5, 1.5), r"$\theta$", r"$\omega$", "Ángulo y velocidad angular"),
        (1, 3, (0.1, 1.5), (-2, 1), "$y$", "$v_y$", "Altura y velocidad vertical"),
    ]
    colors = ListedColormap(["#bbbbbb", "#e69f00", "#009e73", "#0072b2"])
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), 4)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.45))
    fig.subplots_adjust(left=0.09, right=0.935, bottom=0.19, top=0.94,
                        wspace=0.42, hspace=0.53)
    for col, (ix, iy, rx, ry, lx, ly, title) in enumerate(cuts):
        x = np.linspace(*rx, 101)
        y = np.linspace(*ry, 101)
        xx, yy = np.meshgrid(x, y)
        states = np.tile(reference, (xx.size, 1))
        states[:, ix], states[:, iy] = xx.ravel(), yy.ravel()
        with torch.no_grad():
            q = net(torch.from_numpy(states)).numpy().reshape(101, 101, 4)
        axes[0, col].pcolormesh(x, y, q.argmax(axis=2), cmap=colors,
                               norm=norm, shading="auto", rasterized=True)
        im = axes[1, col].pcolormesh(x, y, q.max(axis=2), cmap="viridis",
                                    shading="auto", rasterized=True)
        fig.colorbar(im, ax=axes[1, col], fraction=0.045, pad=0.035)
        axes[0, col].set_title(title)
        for ax in axes[:, col]:
            ax.set(xlabel=lx, ylabel=ly, xlim=rx, ylim=ry)
            ax.set_xticks([rx[0], 0 if ix != 1 else 0.8, rx[1]])
            ax.set_yticks([ry[0], 0, ry[1]])
            ax.tick_params(length=2.5, pad=2)
    action_names = ["0: nada", "1: motor izquierdo", "2: motor principal", "3: motor derecho"]
    fig.legend(handles=[Patch(color=colors(i), label=label)
                        for i, label in enumerate(action_names)],
               loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False)
    fig.savefig(OUT / "cortes_politica.pdf")
    plt.close(fig)
    print("Figuras generadas desde las 32 políticas del manifiesto. No se ejecutó el entorno.")


if __name__ == "__main__":
    main()
