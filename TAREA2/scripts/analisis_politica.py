"""Análisis de la política aprendida (Agustín, plan del 1/11).

Carga pesos ya entrenados (no entrena nada) y genera:
  1. Cortes de la política del DQN completo (semilla 42): acción elegida y max Q en
     tres planos del estado, con los estados realmente visitados superpuestos.
  2. El mismo corte para las 8 semillas del completo (¿la política es estable entre semillas?).
  3. Completo contra sin exploración con la misma semilla.
  4. Episodios reales sin exploración: cómo termina cada episodio y qué acción usa
     según la altura.
  5. Acuerdo con el controlador heurístico que trae Gymnasium (`heuristic`).

Los episodios reales usan las semillas de evaluación de desarrollo 10000-10019 (ya
usadas en la evaluación). No se tocan las semillas reservadas 30000-30199.

Uso, desde TAREA2/:
    python scripts/analisis_politica.py --runs results/comparison_20261031*/A0*_seed* \
        results/comparison_20261102*/A0*_seed* --output analisis_politica/
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.envs.box2d.lunar_lander import heuristic
from matplotlib.colors import BoundaryNorm, ListedColormap
from torch import nn

torch.set_num_threads(1)

# Medido en el entorno: 1 empuja a la izquierda y gira antihorario (sube el ángulo);
# 3 empuja a la derecha y gira horario (baja el ángulo); 2 frena la caída.
ACCIONES = ['Nada', 'Motor izq. (gira +)', 'Motor principal', 'Motor der. (gira −)']
COLORES = ListedColormap(['#bbbbbb', '#e69f00', '#009e73', '#0072b2'])
NORMA = BoundaryNorm(np.arange(-.5, 4.5, 1), 4)
NOMBRES = {'full': 'Completo', 'no_exploration': 'Sin exploración',
           'no_replay': 'Sin replay', 'no_target': 'Sin red objetivo'}

# Mismos cortes y estado fijo que `study_tools.py slices`, para que sean comparables.
ESTADO_FIJO = np.array([0, .8, 0, -.3, 0, 0, 0, 0], dtype=np.float32)
CORTES = [  # (índice x, índice y, rango x, rango y, etiqueta x, etiqueta y)
    (0, 2, (-1, 1), (-1.5, 1.5), 'x (posición horizontal)', 'vx (velocidad horizontal)'),
    (4, 5, (-.7, .7), (-1.5, 1.5), 'θ (ángulo)', 'ω (velocidad angular)'),
    (1, 3, (.1, 1.5), (-2, 1), 'y (altura)', 'vy (velocidad vertical)'),
]
SEMILLA_EVAL, EPISODIOS = 10_000, 20
RESOLUCION = 101


def variante(config):
    """Nombre de la variante a partir de los interruptores guardados en config.json."""
    if not config.get('use_target', True):
        return 'no_target'
    if not config.get('use_replay', True):
        return 'no_replay'
    if not config.get('use_exploration', True):
        return 'no_exploration'
    return 'full'


def cargar_red(carpeta):
    """Reconstruye la red Q (8 -> 64 -> 64 -> 4) y carga los pesos finales."""
    config = json.loads((carpeta / 'config.json').read_text())
    h1, h2 = config['hidden_dims']
    red = nn.Sequential(nn.Linear(8, h1), nn.ReLU(), nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 4))
    red.load_state_dict(torch.load(carpeta / 'online_weights.pt', map_location='cpu', weights_only=True))
    red.eval()
    return red, config


def q_de(red, estados):
    with torch.no_grad():
        return red(torch.as_tensor(np.asarray(estados, dtype=np.float32))).numpy()


def corte(red, ix, iy, rx, ry):
    """Evalúa la red en una grilla que varía dos dimensiones y deja las otras seis fijas."""
    x, y = np.linspace(*rx, RESOLUCION), np.linspace(*ry, RESOLUCION)
    xx, yy = np.meshgrid(x, y)
    estados = np.tile(ESTADO_FIJO, (xx.size, 1))
    estados[:, ix], estados[:, iy] = xx.ravel(), yy.ravel()
    q = q_de(red, estados).reshape(RESOLUCION, RESOLUCION, 4)
    return x, y, q, estados


def jugar(red, episodios=EPISODIOS):
    """Episodios greedy reales. Devuelve estados, acciones y cómo terminó cada episodio."""
    env = gym.make('LunarLander-v3', continuous=False, enable_wind=False)
    estados, acciones, finales = [], [], []
    try:
        for i in range(episodios):
            s, _ = env.reset(seed=SEMILLA_EVAL + i)
            total, pasos = 0.0, 0
            while True:
                a = int(q_de(red, s[None])[0].argmax())
                estados.append(s)
                acciones.append(a)
                s, r, terminado, truncado, _ = env.step(a)
                total += r
                pasos += 1
                if terminado or truncado:
                    break
            # Gymnasium reemplaza la recompensa del último paso por +100 (posado) o −100 (choque/fuera).
            fin = 'posado' if terminado and r == 100 else 'choque' if terminado else 'flotando'
            finales.append(dict(semilla=SEMILLA_EVAL + i, recompensa=total, pasos=pasos, fin=fin))
    finally:
        env.close()
    return np.array(estados), np.array(acciones), finales


def acuerdo_heuristica(estados, acciones):
    """Fracción de estados donde la acción del DQN coincide con `heuristic` de Gymnasium."""
    env = gym.make('LunarLander-v3').unwrapped
    esperadas = np.array([heuristic(env, s) for s in estados])
    env.close()
    return float((esperadas == acciones).mean()), esperadas


def dibujar_corte(ax_a, ax_q, red, spec, visitados=None, titulo=''):
    ix, iy, rx, ry, lx, ly = spec
    x, y, q, _ = corte(red, ix, iy, rx, ry)
    ax_a.pcolormesh(x, y, q.argmax(axis=2), shading='auto', cmap=COLORES, norm=NORMA)
    if visitados is not None:
        ax_a.scatter(visitados[:, ix], visitados[:, iy], s=1, c='black', alpha=.15)
        ax_a.set_xlim(rx); ax_a.set_ylim(ry)
    ax_a.set(xlabel=lx, ylabel=ly, title=titulo)
    if ax_q is not None:
        im = ax_q.pcolormesh(x, y, q.max(axis=2), shading='auto', cmap='viridis')
        plt.colorbar(im, ax=ax_q, label='max Q')
        ax_q.set(xlabel=lx, ylabel=ly)


def leyenda_acciones(fig):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLORES(i), label=n) for i, n in enumerate(ACCIONES)],
               loc='lower center', ncol=4, fontsize=9, frameon=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    politicas = {}   # (variante, semilla) -> (red, carpeta)
    for carpeta in args.runs:
        red, config = cargar_red(carpeta)
        politicas[(variante(config), config['training_seed'])] = (red, carpeta)
    semillas = sorted(s for v, s in politicas if v == 'full')
    print(f"Políticas cargadas: {sorted(politicas)}")

    # --- 4 y 5: episodios reales y acuerdo con la heurística (se usan también en las figuras) ---
    rollouts, metricas = {}, []
    for (var, semilla), (red, carpeta) in sorted(politicas.items()):
        estados, acciones, finales = jugar(red)
        acuerdo, _ = acuerdo_heuristica(estados, acciones)
        _, _, q_grilla, est_grilla = corte(red, *CORTES[2][:4])
        acuerdo_grilla, _ = acuerdo_heuristica(est_grilla, q_grilla.reshape(-1, 4).argmax(1))
        rollouts[(var, semilla)] = (estados, acciones)
        fines = [f['fin'] for f in finales]
        metricas.append(dict(
            variante=var, semilla=semilla, carpeta=str(carpeta),
            recompensa_media=float(np.mean([f['recompensa'] for f in finales])),
            pasos_medios=float(np.mean([f['pasos'] for f in finales])),
            posado=fines.count('posado'), choque=fines.count('choque'), flotando=fines.count('flotando'),
            uso_acciones=[float((acciones == a).mean()) for a in range(4)],
            acuerdo_heuristica_visitados=acuerdo, acuerdo_heuristica_corte_altura=acuerdo_grilla,
            episodios=finales))
    (args.output / 'metricas_politica.json').write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f"\n{'variante':<16}{'semilla':>8}{'R media':>9}{'pasos':>7}{'posa':>6}{'choca':>6}{'flota':>6}"
          f"{'nada':>6}{'izq':>6}{'ppal':>6}{'der':>6}{'acuerdo':>9}")
    for m in metricas:
        u = m['uso_acciones']
        print(f"{NOMBRES[m['variante']]:<16}{m['semilla']:>8}{m['recompensa_media']:9.1f}{m['pasos_medios']:7.0f}"
              f"{m['posado']:6d}{m['choque']:6d}{m['flotando']:6d}"
              + ''.join(f"{v:6.0%}" for v in u) + f"{m['acuerdo_heuristica_visitados']:9.0%}")
    print("\n(20 episodios greedy por política, semillas 10000-10019; 'acuerdo' = acción igual a `heuristic`)")

    # --- 1: cortes del completo, semilla 42, con estados visitados ---
    red, _ = politicas[('full', semillas[0])]
    fig, ejes = plt.subplots(2, 3, figsize=(15, 8.5))
    for col, spec in enumerate(CORTES):
        dibujar_corte(ejes[0, col], ejes[1, col], red, spec, rollouts[('full', semillas[0])][0])
    fig.suptitle(f'DQN completo, semilla {semillas[0]}: acción greedy (arriba, puntos = estados visitados '
                 f'en 20 episodios) y max Q (abajo). Fijo: x=0, y=0.8, vx=0, vy=−0.3, θ=ω=0, sin contacto')
    leyenda_acciones(fig)
    fig.tight_layout(rect=(0, .04, 1, .97))
    fig.savefig(args.output / f'cortes_completo_s{semillas[0]}.png', dpi=150)
    plt.close(fig)

    # --- 2: el mismo corte para las 8 semillas del completo ---
    for k, nombre in [(1, 'angulo'), (2, 'altura')]:
        fig, ejes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True, sharey=True)
        for ax, semilla in zip(ejes.ravel(), semillas):
            dibujar_corte(ax, None, politicas[('full', semilla)][0], CORTES[k], titulo=f'semilla {semilla}')
        fig.suptitle(f'DQN completo, las {len(semillas)} semillas: corte {CORTES[k][4]} – {CORTES[k][5]}')
        leyenda_acciones(fig)
        fig.tight_layout(rect=(0, .05, 1, .96))
        fig.savefig(args.output / f'cortes_8_semillas_{nombre}.png', dpi=150)
        plt.close(fig)

    # --- 3: completo contra sin exploración, misma semilla ---
    comparables = [s for s in semillas if ('no_exploration', s) in politicas]
    if comparables:
        elegidas = [comparables[0], comparables[-1]]
        fig, ejes = plt.subplots(len(elegidas), 4, figsize=(18, 4.3 * len(elegidas)), squeeze=False)
        for fila, semilla in enumerate(elegidas):
            for j, var in enumerate(['full', 'no_exploration']):
                red_v = politicas[(var, semilla)][0]
                visit = rollouts[(var, semilla)][0]
                for k, col in [(1, 2 * j), (2, 2 * j + 1)]:
                    dibujar_corte(ejes[fila, col], None, red_v, CORTES[k], visit,
                                  titulo=f'{NOMBRES[var]}, semilla {semilla}')
        fig.suptitle('Completo contra sin exploración (puntos = estados visitados en 20 episodios greedy)')
        leyenda_acciones(fig)
        fig.tight_layout(rect=(0, .05, 1, .96))
        fig.savefig(args.output / 'cortes_completo_vs_sin_exploracion.png', dpi=150)
        plt.close(fig)

    # --- 4: uso de acciones según la altura, en episodios reales ---
    bordes = np.linspace(0, 1.5, 16)
    variantes = [v for v in NOMBRES if any(k[0] == v for k in rollouts)]
    fig, ejes = plt.subplots(1, len(variantes), figsize=(5.5 * len(variantes), 4), squeeze=False)
    for ax, var in zip(ejes[0], variantes):
        est = np.concatenate([e for (v, _), (e, a) in rollouts.items() if v == var])
        acc = np.concatenate([a for (v, _), (e, a) in rollouts.items() if v == var])
        tramo = np.clip(np.digitize(est[:, 1], bordes) - 1, 0, len(bordes) - 2)
        fracciones = np.array([[np.mean(acc[tramo == t] == a) if (tramo == t).any() else np.nan
                                for t in range(len(bordes) - 1)] for a in range(4)])
        abajo = np.zeros(len(bordes) - 1)
        centros = (bordes[:-1] + bordes[1:]) / 2
        for a in range(4):
            ax.barh(centros, fracciones[a], left=abajo, height=bordes[1] - bordes[0],
                    color=COLORES(a), label=ACCIONES[a])
            abajo += np.nan_to_num(fracciones[a])
        ax.set(xlabel='fracción de pasos', ylabel='altura y', xlim=(0, 1),
               title=f'{NOMBRES[var]} (todas las semillas)')
    ejes[0, 0].legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(args.output / 'acciones_por_altura.png', dpi=150)
    plt.close(fig)
    print(f"\nFiguras y métricas en {args.output.resolve()}")


if __name__ == '__main__':
    main()
