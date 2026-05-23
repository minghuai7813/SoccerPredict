"""Player-level data fetchers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from soccerpredict.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)


def fetch_player_stats(
    competition: str = "FIFA World Cup",
    season: str | int = 2022,
    stat_type: str = "standard",
    source: str = "FBref",
    data_dir: Path | None = None,
) -> "pd.DataFrame":
    """Fetch season-aggregated player stats for a competition."""
    import soccerdata as sd

    reader_cls = getattr(sd, source)
    kwargs: dict[str, object] = {"leagues": competition, "seasons": season}
    if data_dir is not None:
        kwargs["data_dir"] = str(data_dir)
    reader = reader_cls(**kwargs)

    if not hasattr(reader, "read_player_season_stats"):
        raise NotImplementedError(
            f"{source} reader does not expose `read_player_season_stats`."
        )

    df = reader.read_player_season_stats(stat_type=stat_type)
    log.info(
        "Fetched player stats ({}): {} rows for {} {}",
        stat_type,
        len(df),
        competition,
        season,
    )
    return df
