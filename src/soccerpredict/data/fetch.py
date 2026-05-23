"""Match-level data fetchers backed by `soccerdata`.

The functions here are intentionally thin so the rest of the codebase
never imports ``soccerdata`` directly. Swapping to another data source
later only requires changing this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from soccerpredict.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)


def _build_reader(
    source: str,
    leagues: str | list[str],
    seasons: str | int | list[str | int],
    data_dir: Path | None,
):
    """Construct a soccerdata reader instance.

    ``source`` is the soccerdata class name (e.g. ``"FBref"``,
    ``"FotMob"``, ``"FIFA"``). We import lazily so importing
    :mod:`soccerpredict.data` stays cheap when offline.
    """
    import soccerdata as sd

    try:
        reader_cls = getattr(sd, source)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown soccerdata source: {source!r}. "
            f"Try one of: FBref, FotMob, FIFA, Understat, ClubElo, MatchHistory."
        ) from exc

    kwargs: dict[str, object] = {"leagues": leagues, "seasons": seasons}
    if data_dir is not None:
        kwargs["data_dir"] = str(data_dir)

    log.info("Building {} reader: leagues={}, seasons={}", source, leagues, seasons)
    return reader_cls(**kwargs)


def fetch_schedule(
    competition: str = "FIFA World Cup",
    season: str | int = 2022,
    source: str = "FBref",
    data_dir: Path | None = None,
) -> "pd.DataFrame":
    """Return the match schedule (fixtures + results when available)."""
    reader = _build_reader(source, competition, season, data_dir)
    df = reader.read_schedule()
    log.info("Fetched schedule: {} rows", len(df))
    return df


def fetch_matches(
    competition: str = "FIFA World Cup",
    season: str | int = 2022,
    source: str = "FBref",
    data_dir: Path | None = None,
) -> "pd.DataFrame":
    """Return per-match summary stats (goals, xG, possession, etc.).

    Falls back to schedule data when the configured source does not
    expose richer per-match stats for the requested competition.
    """
    reader = _build_reader(source, competition, season, data_dir)

    if hasattr(reader, "read_team_match_stats"):
        df = reader.read_team_match_stats(stat_type="schedule")
    else:
        df = reader.read_schedule()

    log.info("Fetched matches: {} rows", len(df))
    return df
