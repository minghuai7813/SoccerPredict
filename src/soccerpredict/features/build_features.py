"""Match-level feature engineering.

The goal of this module is to turn a raw match schedule + side stats into
a model-ready table with one row per (match, home/away perspective).
Initial features are deliberately simple — recent form, rest days, goal
differential. Heavier features (ELO, xG-weighted form, squad strength)
will be layered on incrementally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from soccerpredict.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)


def compute_recent_form(
    matches: pd.DataFrame,
    team_col: str = "team",
    date_col: str = "date",
    points_col: str = "points",
    window: int = 5,
) -> pd.DataFrame:
    """Add a rolling ``form_{window}`` column = sum of points in the last
    ``window`` matches *before* the current one.

    The input is expected in long format: one row per (match, team), with
    ``points`` already encoded (3 for win, 1 for draw, 0 for loss).
    """
    import pandas as pd  # noqa: F401  (imported for runtime type compatibility)

    df = matches.sort_values([team_col, date_col]).copy()
    df[f"form_{window}"] = (
        df.groupby(team_col)[points_col]
        .shift(1)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return df


def add_rest_days(
    matches: pd.DataFrame,
    team_col: str = "team",
    date_col: str = "date",
) -> pd.DataFrame:
    """Add a ``rest_days`` column = days since the team's previous match."""
    df = matches.sort_values([team_col, date_col]).copy()
    df["rest_days"] = (
        df.groupby(team_col)[date_col]
        .diff()
        .dt.days
        .fillna(value=14)
    )
    return df


def build_match_features(matches_long: pd.DataFrame) -> pd.DataFrame:
    """Compose the default feature set for downstream modeling.

    Parameters
    ----------
    matches_long
        Long-format match table with columns:
        ``match_id, date, team, opponent, is_home, goals_for, goals_against, points``.

    Returns
    -------
    pd.DataFrame
        Same rows as input, enriched with form/rest features.
    """
    df = compute_recent_form(matches_long)
    df = add_rest_days(df)
    log.info("Built match features: {} rows, {} cols", len(df), df.shape[1])
    return df
