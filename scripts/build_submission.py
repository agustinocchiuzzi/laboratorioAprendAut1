"""Build the self-contained submission from an executed notebook and current PDF."""

import ast
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "entrega"
PACKAGE = "Tarea1_Rivas_Duarte_Occhiuzzi"
FILES = (
    "Informe_final.pdf", "notebook.ipynb", "id3.py", "naive_bayes.py",
    "futbol_uruguayo.zip", "requirements.txt", "README.md",
)


def main():
    notebook = json.loads((DELIVERY / "notebook.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if [cell.get("execution_count") for cell in cells] != list(range(1, len(cells) + 1)):
        raise RuntimeError("Ejecutar y guardar todas las celdas desde un kernel nuevo.")
    if any(output["output_type"] == "error" for cell in cells for output in cell.get("outputs", [])):
        raise RuntimeError("El notebook contiene errores de ejecución.")
    flag = next(
        node.value for cell in cells for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RUN_FINAL_TEST" for target in node.targets)
    )
    if not isinstance(flag, ast.Constant) or flag.value is not True:
        raise RuntimeError("La entrega debe tener RUN_FINAL_TEST=True.")
    if not notebook["metadata"]["language_info"]["version"].startswith("3.12."):
        raise RuntimeError("Ejecutar el notebook con Python 3.12.")

    for source, target in [
        (ROOT / "Informe_final.pdf", DELIVERY / "Informe_final.pdf"),
        (ROOT / "data/raw/futbol_uruguayo.zip", DELIVERY / "futbol_uruguayo.zip"),
        (ROOT / "requirements.txt", DELIVERY / "requirements.txt"),
        (ROOT / "src/id3.py", DELIVERY / "id3.py"),
        (ROOT / "src/naive_bayes.py", DELIVERY / "naive_bayes.py"),
    ]:
        shutil.copy2(source, target)

    output = ROOT / "output/entrega" / (PACKAGE + ".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(DELIVERY / name, arcname=f"{PACKAGE}/{name}")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {f"{PACKAGE}/{name}" for name in FILES}
        for name in FILES:
            assert archive.read(f"{PACKAGE}/{name}") == (DELIVERY / name).read_bytes()
    print(output)


if __name__ == "__main__":
    main()
