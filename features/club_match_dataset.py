"""
Club match dataset — domestic league results + roster-based profiles.
俱乐部比赛数据集：五大联赛赛果 + 基于名单的球员组成特征。

Usage / 用法:
    from features.club_match_dataset import build_club_match_dataset
    X, y, meta = build_club_match_dataset()
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extractors.football_data_loader import load_club_matches
from features.club_elo import ClubEloTracker
from features.club_names import league_seasons_for_match
from features.match_features import build_match_feature_row
from features.team_profile import build_club_profile, normalize_club_name

# All Big-5 domestic leagues for club training.
DEFAULT_LEAGUES = ["EPL", "Ligue1", "Bundesliga", "LaLiga", "SerieA"]
DEFAULT_SEASONS = ["2122", "2223", "2324", "2425", "2526"]


@lru_cache(maxsize=512)
def _cached_club_profile(club: str, latest_league_season: str, home_elo: float, away_elo_rank_proxy: int) -> dict:
    """
    LRU cache for club profiles keyed by (club, feature season, elo).
    Elo discretized via int(home_elo) for cache key stability.
    """
    league_seasons = league_seasons_for_match(latest_league_season)
    return build_club_profile(
        club,
        club_elo=home_elo,
        club_elo_rank=away_elo_rank_proxy,
        league_seasons=league_seasons,
        latest_league_season=latest_league_season,
    )


def _profile_for_side(
    club: str,
    latest_league_season: str,
    elo: float,
    elo_rank: int,
) -> dict[str, float]:
    canonical = normalize_club_name(club)
    key_elo = int(round(elo))
    return _cached_club_profile(canonical, latest_league_season, float(key_elo), elo_rank)


def build_club_match_dataset(
    seasons: list[str] | None = None,
    leagues: list[str] | None = None,
    max_matches: int | None = None,
    download: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build feature matrix from domestic club matches.
    从国内联赛赛果构建俱乐部训练集。
    """
    _cached_club_profile.cache_clear()

    matches = load_club_matches(
        seasons=seasons or DEFAULT_SEASONS,
        leagues=leagues or DEFAULT_LEAGUES,
        download=download,
    )
    if matches.empty:
        print("[ClubDataset] No matches loaded.")
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    if max_matches is not None and len(matches) > max_matches:
        matches = matches.iloc[:max_matches].copy()

    tracker = ClubEloTracker()
    rows: list[dict] = []
    meta_rows: list[dict] = []
    base_columns: list[str] | None = None
    skipped = 0

    print(f"\n[ClubDataset] Building features for {len(matches)} matches...")
    for i, m in matches.iterrows():
        home, away = m["home_team"], m["away_team"]
        h_elo, a_elo = tracker.snapshot(home, away)
        h_rank = tracker.rank(home)
        a_rank = tracker.rank(away)

        try:
            home_f = _profile_for_side(home, m["latest_league_season"], h_elo, h_rank)
            away_f = _profile_for_side(away, m["latest_league_season"], a_elo, a_rank)
        except Exception:
            skipped += 1
            tracker.apply_result(home, away, int(m["home_score"]), int(m["away_score"]))
            continue

        if not home_f or not away_f:
            skipped += 1
            tracker.apply_result(home, away, int(m["home_score"]), int(m["away_score"]))
            continue

        row = build_match_feature_row(home_f, away_f, base_columns)
        if base_columns is None:
            base_columns = list(row.keys())

        rows.append(row)
        meta_rows.append({
            **m.to_dict(),
            "home_elo_pre": h_elo,
            "away_elo_pre": a_elo,
            "competition": f"{m['league']} {m['season_code']}",
            "tournament": "club_domestic",
        })

        tracker.apply_result(home, away, int(m["home_score"]), int(m["away_score"]))

        if (len(rows) % 200) == 0 and len(rows) > 0:
            print(f"  ... {len(rows)} feature rows built")

    if not rows:
        print(f"[ClubDataset] No feature rows (skipped {skipped}).")
        return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

    X = pd.DataFrame(rows, columns=base_columns)
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    y = meta["result"].astype(int)

    n_nan = X.isna().sum().sum()
    n_total = max(X.shape[0] * X.shape[1], 1)
    pct = (1 - n_nan / n_total) * 100

    print(f"\n[ClubDataset] {len(X)} matches x {X.shape[1]} features")
    print(f"  Skipped: {skipped}")
    print(f"  Data availability: {pct:.1f}%")
    print(f"  Label distribution: {dict(y.value_counts())}")
    print(f"  Elo teams tracked: {tracker.team_count}")
    print(f"  Top Elo: {tracker.top(5)}")

    return X, y, meta


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    X, y, meta = build_club_match_dataset(max_matches=500)
    print(f"\nShape: {X.shape}")
