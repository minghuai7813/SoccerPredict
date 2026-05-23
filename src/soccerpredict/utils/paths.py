"""Filesystem path helpers.

All paths are anchored at the repository root so the package behaves the
same whether invoked from a notebook, a CLI command, or a unit test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

DataLayer = Literal["raw", "processed", "external"]


def data_dir(layer: DataLayer = "raw") -> Path:
    """Return the absolute path to one of the project's data layers.

    The directory is created on first access so downstream code can write
    immediately without extra ``mkdir`` boilerplate.
    """
    path = PROJECT_ROOT / "data" / layer
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_dir() -> Path:
    """Return the absolute path to the YAML config directory."""
    return PROJECT_ROOT / "config"
