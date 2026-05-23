"""Project configuration.

Reads a YAML config file plus environment variables (via ``.env``) and
returns a typed :class:`AppConfig` instance that the rest of the codebase
can pass around without re-parsing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from soccerpredict.utils.paths import PROJECT_ROOT, config_dir

DEFAULT_CONFIG_FILE = "default.yaml"


@dataclass(slots=True)
class CompetitionConfig:
    """A single competition target.

    ``name`` follows the soccerdata convention (e.g. ``"FIFA World Cup"``
    or ``"ENG-Premier League"``). ``seasons`` is a list of season tags
    accepted by soccerdata (e.g. ``"2022"`` for World Cup, ``"23-24"`` for
    leagues).
    """

    name: str
    seasons: list[str] = field(default_factory=list)
    source: str = "FBref"


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    competitions: list[CompetitionConfig]
    soccerdata_dir: Path | None
    log_level: str
    random_seed: int

    @property
    def default_competition(self) -> CompetitionConfig:
        if not self.competitions:
            raise ValueError("No competitions configured.")
        return self.competitions[0]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML + environment.

    Resolution order (later overrides earlier):
        1. ``config/default.yaml``
        2. The file referenced by ``path`` (if any)
        3. Relevant environment variables loaded from ``.env``
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    raw = _read_yaml(config_dir() / DEFAULT_CONFIG_FILE)
    if path is not None:
        raw = {**raw, **_read_yaml(Path(path))}

    competitions_raw = raw.get("competitions") or []
    competitions = [
        CompetitionConfig(
            name=c["name"],
            seasons=list(c.get("seasons", [])),
            source=c.get("source", "FBref"),
        )
        for c in competitions_raw
    ]

    env_competition = os.getenv("DEFAULT_COMPETITION")
    env_season = os.getenv("DEFAULT_SEASON")
    if env_competition and not any(c.name == env_competition for c in competitions):
        competitions.insert(
            0,
            CompetitionConfig(
                name=env_competition,
                seasons=[env_season] if env_season else [],
            ),
        )

    soccerdata_dir_raw = os.getenv("SOCCERDATA_DIR") or raw.get("soccerdata_dir")
    soccerdata_dir = Path(soccerdata_dir_raw).expanduser() if soccerdata_dir_raw else None

    return AppConfig(
        competitions=competitions,
        soccerdata_dir=soccerdata_dir,
        log_level=os.getenv("LOG_LEVEL", raw.get("log_level", "INFO")),
        random_seed=int(raw.get("random_seed", 42)),
    )
