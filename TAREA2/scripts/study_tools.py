"""Preparation and inference tools. No command in this module trains an agent."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / 'protocols/team_draft.json'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def validate_protocol(protocol):
    seeds = protocol['training_seeds']
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Training seeds must be distinct nonnegative integers')
    budget, evaluation = protocol['budget'], protocol['evaluation']
    for value in [budget['episodes'], budget['max_steps'], evaluation['episodes'], evaluation['reserved_episodes']]:
        if type(value) is not int or value <= 0:
            raise ValueError('Episode counts and step limits must be positive integers')
    development = set(range(evaluation['development_seed_start'], evaluation['development_seed_start'] + evaluation['episodes']))
    reserved = set(range(evaluation['reserved_seed_start'], evaluation['reserved_seed_start'] + evaluation['reserved_episodes']))
    if development & reserved:
        raise ValueError('Development and reserved evaluation seeds overlap')
    if not evaluation['greedy'] or evaluation['learning']:
        raise ValueError('Evaluation must be greedy and learning disabled')
    if protocol['reference']['early_stopping'] is not None:
        raise ValueError('This matched fixed-budget study requires early stopping disabled')


def make_manifest(protocol):
    validate_protocol(protocol)
    source = read_json(ROOT / 'lander.ipynb')
    constructor = next(node for cell in source['cells'] if cell['cell_type'] == 'code'
        for node in ast.parse(''.join(cell['source'])).body
        if isinstance(node, ast.ClassDef) and node.name == 'AgenteDQN')
    init = next(node for node in constructor.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    supported = {a.arg for a in init.args.args + init.args.kwonlyargs}
    jobs = []
    for variant, spec in protocol['variants'].items():
        missing = [key for key in spec['changes'] if key.startswith('use_') and key not in supported]
        for seed in protocol['training_seeds']:
            jobs.append(dict(job_id=f'{variant}_seed{seed}', variant=variant,
                training_seed=seed, owner=spec['owner'], changes=spec['changes'],
                missing_agent_switches=missing))
    return dict(status=protocol['status'], protocol=protocol, jobs=jobs, job_count=len(jobs),
        maximum_training_steps=len(jobs)*protocol['budget']['episodes']*protocol['budget']['max_steps'],
        note='Planning only; no agents constructed or trained. Reserved evaluation seeds remain unused.')


def aggregate(protocol, folders):
    """Validate study comparability and summarize independently trained seeds."""
    validate_protocol(protocol)
    groups = {}
    sources = []
    seen = set()
    for folder in map(Path, folders):
        for required in protocol['output_files']:
            if not (folder / required).is_file():
                raise ValueError(f'{folder}: missing {required}')
        config = read_json(folder / 'config.json')
        summary = read_json(folder / 'summary.json')
        switches = {key: config.get(key, True) for key in ['use_target', 'use_replay', 'use_exploration']}
        if any(type(value) is not bool for value in switches.values()):
            raise ValueError(f'{folder}: component switches must be booleans')
        off = [key for key, value in switches.items() if value is False]
        names = {'use_target': 'no_target', 'use_replay': 'no_replay', 'use_exploration': 'no_exploration'}
        if len(off) > 1:
            raise ValueError(f'{folder}: more than one component removed')
        variant = names[off[0]] if off else 'full'
        if variant == 'no_replay' and config.get('effective_batch_size') != 1:
            raise ValueError(f'{folder}: replay-off effective batch size must be documented as 1')
        expected = {**protocol['reference'], **protocol['budget']}
        if variant == 'no_replay':
            expected['replay_sampling'] = 'current transition only'
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f'{folder}: {key} differs from protocol ({config.get(key)!r} != {value!r})')
        # Old A runs predate these fields. Their documented defaults remain valid;
        # new schedules/soft targets must not be silently pooled with that reference.
        legacy_defaults = dict(epsilon_schedule='linear_episodes', epsilon_decay_steps=50000,
            epsilon_decay_factor=.995, target_update='hard', tau=.001)
        for key, default in legacy_defaults.items():
            actual = config.get(key, default)
            if key == 'target_update' and actual == 'hard copy after optimizer step':
                actual = 'hard'
            if actual != protocol['reference'].get(key, default):
                raise ValueError(f'{folder}: {key} differs from protocol')
        seed = config['training_seed']
        if seed not in protocol['training_seeds'] or (variant, seed) in seen:
            raise ValueError(f'{folder}: unplanned or duplicate training seed for {variant}')
        seen.add((variant, seed))
        evaluation = protocol['evaluation']
        if (config['evaluation_seed'], config['evaluation_episodes']) != (evaluation['development_seed_start'], evaluation['episodes']):
            raise ValueError(f'{folder}: evaluation settings differ')
        if config['evaluation_exploration']:
            raise ValueError(f'{folder}: evaluation used exploration')
        with (folder / 'evaluation.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        wanted = list(range(evaluation['development_seed_start'], evaluation['development_seed_start'] + evaluation['episodes']))
        if [int(r['reset_seed']) for r in rows] != wanted:
            raise ValueError(f'{folder}: evaluation CSV seeds differ')
        returns = [float(r['return']) for r in rows]
        if not all(math.isfinite(x) for x in returns):
            raise ValueError(f'{folder}: nonfinite returns')
        mean = sum(returns)/len(returns)
        successes = sum(x >= evaluation['success_return'] for x in returns)
        if not math.isclose(mean, summary['mean_return'], abs_tol=1e-8) or successes != summary['successes']:
            raise ValueError(f'{folder}: summary disagrees with raw returns')
        with (folder / 'training.csv').open() as handle:
            training = list(csv.DictReader(handle))
        if len(training) != protocol['budget']['episodes'] or sum(int(r['length']) for r in training) != summary['total_steps']:
            raise ValueError(f'{folder}: training budget or step totals differ')
        record = dict(training_seed=seed, mean_return=mean, successes=successes,
            evaluation_episodes=len(returns), training_steps=summary['total_steps'], training_seconds=summary['training_seconds'])
        groups.setdefault(variant, []).append(record)
        sources.append(str(folder.resolve()))
    result = {}
    for variant, spec in protocol['variants'].items():
        rows = sorted(groups.get(variant, []), key=lambda r: r['training_seed'])
        values = [r['mean_return'] for r in rows]
        mean = sum(values)/len(values) if values else None
        sd = math.sqrt(sum((v-mean)**2 for v in values)/(len(values)-1)) if len(values)>1 else None
        result[variant] = dict(per_training_seed=rows, mean_across_training_seeds=mean,
            training_seed_sample_sd=sd, missing_training_seeds=sorted(set(protocol['training_seeds'])-{r['training_seed'] for r in rows}))
    return dict(protocol_status=protocol['status'], variants=result, sources=sources,
        complete=all(not row['missing_training_seeds'] for row in result.values()),
        statistical_tests='Not performed; choose test and assumptions with the team. Evaluation episodes are not independent trained policies.')


def policy_slices(folder, output):
    """Load a saved policy and generate three state slices. No environment stepping."""
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import random
    from abc import ABC, abstractmethod
    from collections import deque
    import numpy as np
    import gymnasium as gym
    import torch
    from torch import nn, optim
    from torch.nn import functional as F
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    folder, output = Path(folder), Path(output)
    config = read_json(folder / 'config.json')
    namespace = dict(ABC=ABC, abstractmethod=abstractmethod, deque=deque, random=random,
        np=np, torch=torch, nn=nn, optim=optim, F=F)
    for cell in read_json(ROOT / 'lander.ipynb')['cells']:
        if cell['cell_type'] != 'code': continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.ClassDef) and node.name in {'Agente', 'AgenteDQN'}:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / 'lander.ipynb'), 'exec'), namespace)
    torch.set_num_threads(1)
    agent = namespace['AgenteDQN'](gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
        gym.spaces.Discrete(4), semilla=0, hidden_dims=tuple(config['hidden_dims']), use_target=False)
    weights = folder / 'online_weights.pt'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    agent.red_q.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    agent.red_q.eval()
    output.mkdir(parents=True, exist_ok=False)
    specs = [(0, 2, (-1, 1), (-1.5, 1.5), 'x', 'vx'),
             (4, 5, (-.7, .7), (-1.5, 1.5), 'angle', 'angular velocity'),
             (1, 3, (.1, 1.5), (-2, 1), 'height', 'vy')]
    fixed = np.array([0, .8, 0, -.3, 0, 0, 0, 0], dtype=np.float32)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cmap = ListedColormap(['#777777', '#e69f00', '#009e73', '#0072b2'])
    norm = BoundaryNorm(np.arange(-.5, 4.5, 1), 4)
    metadata = []
    for column, (ix, iy, rx, ry, lx, ly) in enumerate(specs):
        x, y = np.linspace(*rx, 101), np.linspace(*ry, 101)
        xx, yy = np.meshgrid(x, y)
        states = np.tile(fixed, (xx.size, 1)); states[:, ix] = xx.ravel(); states[:, iy] = yy.ravel()
        with torch.no_grad():
            q = agent.red_q(torch.as_tensor(states)).numpy().reshape(101, 101, 4)
        actions = q.argmax(axis=2)
        np.savez_compressed(output / f'slice_{ix}_{iy}.npz', x=x, y=y, q=q, actions=actions, fixed_state=fixed)
        im = axes[0, column].pcolormesh(x, y, actions, shading='auto', cmap=cmap, norm=norm)
        cb = fig.colorbar(im, ax=axes[0, column], ticks=range(4))
        cb.ax.set_yticklabels(['No engine', 'Side action 1', 'Main engine', 'Side action 3'])
        im = axes[1, column].pcolormesh(x, y, q.max(axis=2), shading='auto')
        fig.colorbar(im, ax=axes[1, column], label='Maximum Q estimate')
        for row in range(2): axes[row, column].set(xlabel=lx, ylabel=ly)
        metadata.append(dict(x_index=ix, y_index=iy, x_range=rx, y_range=ry))
    fig.suptitle(f"{config['name']} training seed {config['training_seed']}: actions (top), max Q (bottom)")
    fig.tight_layout(); fig.savefig(output / 'policy_slices.png', dpi=150); plt.close(fig)
    assert agent.env_steps == agent.optimizer_updates == 0 and len(agent.memoria) == 0
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == digest
    write_new_json(output / 'metadata.json', dict(checkpoint=str(weights.resolve()), checkpoint_sha256=digest,
        source_notebook_sha256=hashlib.sha256((ROOT/'lander.ipynb').read_bytes()).hexdigest(),
        fixed_state=fixed.tolist(), slices=metadata, training_performed=False,
        caveat='Synthetic states may be unreachable or outside training experience. Q-values are estimates, not probabilities.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['plan', 'aggregate']:
        cmd = sub.add_parser(command)
        cmd.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
        cmd.add_argument('--output', type=Path, required=True)
        if command == 'aggregate': cmd.add_argument('--runs', type=Path, nargs='+', required=True)
    cmd = sub.add_parser('slices')
    cmd.add_argument('--run', type=Path, required=True)
    cmd.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'slices':
        policy_slices(args.run, args.output)
    elif args.command == 'plan':
        manifest = make_manifest(read_json(args.protocol))
        write_new_json(args.output, manifest)
        print(f"Planned {manifest['job_count']} jobs; zero training started.")
    else:
        write_new_json(args.output, aggregate(read_json(args.protocol), args.runs))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
