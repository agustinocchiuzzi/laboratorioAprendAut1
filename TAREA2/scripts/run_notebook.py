"""Execute the self-contained notebook with the current Python kernel."""
import argparse
import ast
import contextlib
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_selected(notebook, configuration, seed, variant="full", overrides=None):
    """Reuse notebook definitions, skipping demonstrations and other experiments."""
    namespace = {"__name__": "__runner__"}
    def execute(index, definitions_only=False):
        source = notebook.cells[index].source
        tree = ast.parse(source)
        if definitions_only:
            tree.body = [node for node in tree.body if isinstance(node,
                (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom,
                 ast.Assign, ast.AnnAssign))]
            # These cells also contain demonstration assignments. Keep only
            # definitions, imports, and the evaluation threshold constant.
            tree.body = [node for node in tree.body if not isinstance(node,
                (ast.Assign, ast.AnnAssign)) or
                (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                 and t.id == "UMBRAL_EXITO" for t in node.targets))]
        exec(compile(tree, f"lander.ipynb:cell{index}", "exec"), namespace)
    execute(3)
    execute(4)
    execute(5)
    for index in (11, 13, 15, 17):
        execute(index, definitions_only=True)
    execute(19)
    folder = namespace["RAIZ_RESULTADOS"]
    print(f"Selected run output: {folder.resolve()}", flush=True)
    parameters = {**namespace["CONFIGURACIONES"][configuration],
                  **namespace["VARIANTES"][variant], **(overrides or {})}
    name = configuration if variant == "full" else f"{configuration}_{variant}"
    if overrides:
        name += "_custom"
    with (folder / "runner.log").open("x") as log, contextlib.redirect_stdout(log):
        namespace["entrenar_y_guardar"](name, parameters, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['pilot', 'comparison', 'integration-pilot'], default='pilot')
    parser.add_argument("--config", choices=["A0", "A1", "A2", "A3"])
    parser.add_argument("--seed", type=int, help="Training seed (requires --config; default: 42)")
    parser.add_argument("--variant", choices=["full", "no_target", "no_replay", "no_exploration"])
    parser.add_argument("--epsilon-schedule", choices=["linear_episodes", "linear_steps", "exponential_episodes"])
    parser.add_argument("--replay-capacity", type=int)
    parser.add_argument("--target-update", choices=["hard", "soft"])
    args = parser.parse_args()
    overrides = {key: value for key, value in {
        "epsilon_schedule": args.epsilon_schedule, "replay_capacity": args.replay_capacity,
        "target_update": args.target_update}.items() if value is not None}
    if (args.seed is not None or args.variant is not None or overrides) and not args.config:
        parser.error("--seed, --variant and parameter overrides require --config")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.replay_capacity is not None and args.replay_capacity < 64:
        parser.error("--replay-capacity must be at least the batch size (64)")
    os.chdir(ROOT)
    os.environ['DQN_MODE'] = args.mode
    os.environ.setdefault('MPLBACKEND', 'module://matplotlib_inline.backend_inline')
    import tempfile
    runtime = tempfile.mkdtemp(prefix='dqn-jupyter-')
    os.environ.setdefault('IPYTHONDIR', runtime)
    os.environ.setdefault('JUPYTER_RUNTIME_DIR', runtime)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    notebook = nbformat.read(ROOT / 'lander.ipynb', as_version=4)
    if args.config:
        try:
            run_selected(notebook, args.config, 42 if args.seed is None else args.seed,
                         args.variant or "full", overrides)
        finally:
            shutil.rmtree(runtime, ignore_errors=True)
        return
    manager = KernelManager(kernel_name='python3')
    # Use this exact virtualenv, without installing a user-global kernelspec.
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']

    def report(cell_index, **kwargs):
        print(f'Cell {cell_index + 1}/{len(notebook.cells)}', flush=True)

    client = NotebookClient(notebook, km=manager, timeout=None, on_cell_start=report,
                            resources={'metadata': {'path': str(ROOT)}})
    output = ROOT / f'{args.mode}_results.ipynb'
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
        print(f'Notebook with outputs: {output}', flush=True)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        shutil.rmtree(runtime, ignore_errors=True)
    print('Completed. Read the experiment paths and measured results in the notebook.', flush=True)


if __name__ == '__main__':
    main()
