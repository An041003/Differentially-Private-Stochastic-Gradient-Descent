"""Build a portable project ZIP without modifying the source notebooks.

The staging copy is intentionally restricted to files needed to reproduce,
inspect, and submit the DP-SGD project.  Notebook outputs are cleared only in
the ZIP so the working notebooks remain untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "dist" / "DPSGD_project_package_20260621.zip"

ROOT_FILES = (
    "Agent.md",
    "Architecture.md",
    "README.md",
    "requirements.txt",
    "run_project.bat",
    "data.md",
)
INCLUDED_DIRECTORIES = (
    "data",
    "src",
    "scripts",
    "notebooks",
    "docs",
    "results",
    "figures",
    "report",
    "slides",
)
IGNORED_PARTS = {"__pycache__", ".ipynb_checkpoints", ".venv", "dist"}


def is_included_file(path: Path) -> bool:
    return not any(part in IGNORED_PARTS for part in path.parts)


def copy_notebook_without_outputs(source: Path, target: Path) -> None:
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(notebook, target)


def copy_release_tree(staging_root: Path) -> list[str]:
    copied: list[str] = []
    for relative_name in ROOT_FILES:
        source = ROOT / relative_name
        if source.exists():
            destination = staging_root / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(destination.relative_to(staging_root).as_posix())

    for relative_name in INCLUDED_DIRECTORIES:
        source_dir = ROOT / relative_name
        if not source_dir.exists():
            continue
        for source in source_dir.rglob("*"):
            if not source.is_file() or not is_included_file(source):
                continue
            destination = staging_root / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".ipynb":
                copy_notebook_without_outputs(source, destination)
            else:
                shutil.copy2(source, destination)
            copied.append(destination.relative_to(staging_root).as_posix())
    return sorted(copied)


def verify_clean_notebooks(staging_root: Path) -> int:
    notebook_count = 0
    for notebook_path in staging_root.glob("notebooks/**/*.ipynb"):
        notebook_count += 1
        notebook = nbformat.read(notebook_path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code" and (cell.outputs or cell.execution_count is not None):
                raise RuntimeError(f"Notebook still has output: {notebook_path}")
    return notebook_count


def write_manifest(staging_root: Path, copied_files: list[str], notebook_count: int) -> None:
    manifest = {
        "package_name": "DPSGD reproducible project package",
        "created_utc": datetime.now(UTC).isoformat(),
        "notebook_output_policy": "All code-cell outputs and execution counts were removed in this package only.",
        "run_command": "run_project.bat",
        "file_count": len(copied_files),
        "notebook_count": notebook_count,
        "included_top_level_directories": list(INCLUDED_DIRECTORIES),
        "files": copied_files,
    }
    (staging_root / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def create_archive(staging_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staging_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(staging_root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean DP-SGD project release ZIP.")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARCHIVE, help="Destination ZIP path.")
    args = parser.parse_args()
    archive_path = args.output if args.output.is_absolute() else ROOT / args.output

    with tempfile.TemporaryDirectory(prefix="dpsgd_release_") as temp_dir:
        staging_root = Path(temp_dir) / "DPSGD_project"
        staging_root.mkdir()
        copied_files = copy_release_tree(staging_root)
        notebook_count = verify_clean_notebooks(staging_root)
        write_manifest(staging_root, copied_files, notebook_count)
        create_archive(staging_root, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        print(f"Created: {archive_path}")
        print(f"Files: {len(archive.namelist())}")
        print(f"Notebooks cleaned: {notebook_count}")


if __name__ == "__main__":
    main()
