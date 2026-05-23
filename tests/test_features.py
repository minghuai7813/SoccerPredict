"""Tests for feature engineering helpers."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def sample_long_matches() -> pd.DataFrame:
    """Toy long-format match table: two teams, four matches, alternating wins."""
    return pd.DataFrame(
        {
            "match_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "date": pd.to_datetime(
                [
                    "2022-11-20",
                    "2022-11-20",
                    "2022-11-25",
                    "2022-11-25",
                    "2022-11-29",
                    "2022-11-29",
                    "2022-12-03",
                    "2022-12-03",
                ]
            ),
            "team": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "opponent": ["B", "A", "B", "A", "B", "A", "B", "A"],
            "is_home": [True, False, False, True, True, False, False, True],
            "goals_for": [2, 1, 0, 2, 3, 1, 1, 1],
            "goals_against": [1, 2, 2, 0, 1, 3, 1, 1],
            "points": [3, 0, 0, 3, 3, 0, 1, 1],
        }
    )


def test_compute_recent_form_shifts_correctly(sample_long_matches: pd.DataFrame) -> None:
    from soccerpredict.features import compute_recent_form

    out = compute_recent_form(sample_long_matches, window=3)
    assert "form_3" in out.columns
    first_row_a = out[(out["team"] == "A")].sort_values("date").iloc[0]
    assert pd.isna(first_row_a["form_3"]) or first_row_a["form_3"] == 0


def test_add_rest_days(sample_long_matches: pd.DataFrame) -> None:
    from soccerpredict.features import add_rest_days

    out = add_rest_days(sample_long_matches)
    assert "rest_days" in out.columns
    a_rows = out[out["team"] == "A"].sort_values("date")
    assert int(a_rows["rest_days"].iloc[1]) == 5


def test_build_match_features_runs(sample_long_matches: pd.DataFrame) -> None:
    from soccerpredict.features import build_match_features

    out = build_match_features(sample_long_matches)
    for col in ("form_5", "rest_days"):
        assert col in out.columns
