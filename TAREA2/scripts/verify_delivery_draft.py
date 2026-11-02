"""Execute the draft in a new kernel from an empty directory, with local runtime files."""
import base64
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'entrega_borrador.ipynb'
    notebook = nbformat.read(path, as_version=4)
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
    os.environ['MPLBACKEND'] = 'module://matplotlib_inline.backend_inline'
    with tempfile.TemporaryDirectory(prefix='thiago-delivery-') as runtime:
        os.environ['IPYTHONDIR'] = runtime
        os.environ['JUPYTER_RUNTIME_DIR'] = runtime
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, km=manager, timeout=180,
            resources={'metadata': {'path': runtime}},
            on_cell_start=lambda cell_index, **kw: print(f'Celda {cell_index+1}/{len(notebook.cells)}', flush=True))
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
        assert not any(output.output_type == 'error' for cell in notebook.cells if cell.cell_type == 'code'
            for output in cell.get('outputs', []))
        output = ROOT / 'analysis/team_study'
        output.mkdir(parents=True, exist_ok=True)
        plots = []
        for cell in notebook.cells:
            for item in cell.get('outputs', []):
                data = item.get('data', {})
                if 'image/png' in data:
                    dest = output / f'notebook_figure_{len(plots)+1}.png'
                    dest.write_bytes(base64.b64decode(data['image/png']))
                    plots.append(str(dest))
                if 'image/gif' in data:
                    (output / 'notebook_rollouts.gif').write_bytes(base64.b64decode(data['image/gif']))
        print('OK: execution in a fresh kernel from an empty temporary directory; embedded data/weights only.')
        print('Figures:', plots)


if __name__ == '__main__':
    main()
