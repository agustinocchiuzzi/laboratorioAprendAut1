"""Execute the self-contained notebook with the current Python kernel."""
import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['pilot', 'comparison'], default='pilot')
    args = parser.parse_args()
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

    notebook = nbformat.read(ROOT / 'LunarLander_DQN_A.ipynb', as_version=4)
    manager = KernelManager(kernel_name='python3')
    # Use this exact virtualenv, without installing a user-global kernelspec.
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']

    def report(cell_index, **kwargs):
        print(f'Cell {cell_index + 1}/{len(notebook.cells)}', flush=True)

    client = NotebookClient(notebook, km=manager, timeout=None, on_cell_start=report,
                            resources={'metadata': {'path': str(ROOT)}})
    output = ROOT / f'LunarLander_DQN_A_{args.mode}_executed.ipynb'
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
