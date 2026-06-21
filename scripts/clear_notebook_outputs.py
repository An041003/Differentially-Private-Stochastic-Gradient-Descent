"""Remove execution output from project notebooks in place.

Use this before committing notebooks so the repository contains source cells
only, not stale local outputs or execution counters.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def clear_outputs(path: Path) -> bool:
    notebook = nbformat.read(path, as_version=4)
    changed = False
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if cell.outputs or cell.execution_count is not None:
            cell.outputs = []
            cell.execution_count = None
            changed = True
    if changed:
        nbformat.write(notebook, path)
    return changed


def main() -> None:
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    changed_count = sum(clear_outputs(path) for path in notebooks)
    print(f"Checked {len(notebooks)} notebooks; cleaned {changed_count}.")


if __name__ == "__main__":
    main()
