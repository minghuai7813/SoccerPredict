"""
Multi-tournament match dataset builder.
多赛事比赛数据集构造器。

Combines matches from 2018 WC, Euro 2020, and 2022 WC into a single
training dataset, with tournament-appropriate features for each.
将 2018 世界杯、2020 欧洲杯、2022 世界杯的比赛合并为统一训练集，
每个赛事使用对应的赛前特征。

Usage / 用法:
    from features.match_dataset import build_match_dataset
    X, y, meta = build_match_dataset()
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsbombpy import sb

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.team_features import build_team_feature_matrix
from features.player_aggregation import positional_matchup_features

# Tournament registry: each entry defines the data sources for one tournament.
# 赛事注册表：每项定义该赛事的数据源。
TOURNAMENTS = [
    {
        "name": "FIFA World Cup 2018",
        "short": "WC2018",
        "competition_id": 43,
        "season_id": 3,
        "elo_key": "wc2018",
        "league_seasons": {
            "2016-2017": 0.6,
            "2017-2018": 1.0,
        },
        "latest_league_season": "2017-2018",
        "prior_national_comps": [],
    },
    {
        "name": "UEFA Euro 2020",
        "short": "EURO2020",
        "competition_id": 55,
        "season_id": 43,
        "elo_key": "euro2020",
        "league_seasons": {
            "2018-2019": 0.5,
            "2019-2020": 0.75,
            "2020-2021": 1.0,
        },
        "latest_league_season": "2020-2021",
        "prior_national_comps": ["FIFA World Cup 2018"],
    },
    {
        "name": "FIFA World Cup 2022",
        "short": "WC2022",
        "competition_id": 43,
        "season_id": 106,
        "elo_key": "wc2022",
        "league_seasons": {
            "2018-2019": 0.4,
            "2019-2020": 0.6,
            "2020-2021": 0.8,
            "2021-2022": 1.0,
        },
        "latest_league_season": "2021-2022",
        "prior_national_comps": ["FIFA World Cup 2018", "UEFA Euro 2020"],
    },
]

KNOCKOUT_STAGES = {
    "Round of 16", "Quarter-finals", "Semi-finals",
    "Final", "3rd Place Final",
}


def _extract_90min_score(match_id: int) -> dict:
    """
    Extract 90-min regulation score from StatsBomb event data.
    从 StatsBomb 事件数据中提取 90 分钟常规时间比分。
    """
    import warnings
    warnings.filterwarnings("ignore")

    events = sb.events(match_id=match_id)
    if events.empty:
        return {}

    goals = events[events["type"] == "Shot"]
    if "shot_outcome" in goals.columns:
        goals = goals[goals["shot_outcome"] == "Goal"]

    periods = sorted(events["period"].unique())
    has_et = any(p > 2 for p in periods)
    has_pen = 5 in periods

    def _count_goals_in_periods(goal_df, period_list):
        subset = goal_df[goal_df["period"].isin(period_list)]
        counts: dict[str, int] = {}
        for _, g in subset.iterrows():
            t = g.get("team", "")
            if t:
                counts[t] = counts.get(t, 0) + 1
        return counts

    return {
        "team_reg_goals": _count_goals_in_periods(goals, [1, 2]),
        "team_et_goals": _count_goals_in_periods(goals, [3, 4]),
        "team_pen_goals": _count_goals_in_periods(goals, [5]),
        "has_extra_time": has_et,
        "has_penalties": has_pen,
    }


def _load_tournament_matches(
    competition_id: int, season_id: int, tournament_name: str,
) -> pd.DataFrame:
    """
    Load matches for a single tournament with 90-min score extraction.
    加载单个赛事的比赛并提取 90 分钟比分。
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)

    records = []
    knockout_ids = []

    for _, row in matches.iterrows():
        stage = row.get("competition_stage", "Group Stage")
        is_ko = stage in KNOCKOUT_STAGES
        mid = row["match_id"]
        if is_ko:
            knockout_ids.append((mid, row))

        records.append({
            "match_id": mid,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score_full": int(row["home_score"]),
            "away_score_full": int(row["away_score"]),
            "stage": stage,
            "is_knockout": is_ko,
            "tournament": tournament_name,
        })

    df = pd.DataFrame(records)
    df["home_score_90"] = df["home_score_full"]
    df["away_score_90"] = df["away_score_full"]
    df["went_to_et"] = False
    df["went_to_penalties"] = False
    df["qualifier"] = ""

    if knockout_ids:
        print(f"    Extracting 90-min scores for {len(knockout_ids)} knockout matches...")
        for mid, row in knockout_ids:
            home, away = row["home_team"], row["away_team"]
            ev = _extract_90min_score(mid)
            if not ev:
                continue

            mask = df["match_id"] == mid
            df.loc[mask, "home_score_90"] = ev["team_reg_goals"].get(home, 0)
            df.loc[mask, "away_score_90"] = ev["team_reg_goals"].get(away, 0)
            df.loc[mask, "went_to_et"] = ev["has_extra_time"]
            df.loc[mask, "went_to_penalties"] = ev["has_penalties"]

            h_full, a_full = int(row["home_score"]), int(row["away_score"])
            if h_full > a_full:
                qual = home
            elif a_full > h_full:
                qual = away
            else:
                h_pen = ev["team_pen_goals"].get(home, 0)
                a_pen = ev["team_pen_goals"].get(away, 0)
                qual = home if h_pen > a_pen else away
            df.loc[mask, "qualifier"] = qual

    # Labels based on 90-min score.
    df["result"] = df.apply(
        lambda r: 1 if r["home_score_90"] > r["away_score_90"]
        else (-1 if r["home_score_90"] < r["away_score_90"] else 0),
        axis=1,
    )
    df["goal_diff"] = df["home_score_90"] - df["away_score_90"]
    df["home_score"] = df["home_score_90"]
    df["away_score"] = df["away_score_90"]

    return df


def build_match_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build combined multi-tournament dataset.
    构建多赛事合并数据集。

    Returns
    -------
    X : pd.DataFrame   — Feature matrix (NaN for missing features)
    y : pd.Series       — Labels (1/0/-1 based on 90-min score)
    meta : pd.DataFrame — Match metadata
    """
    all_X = []
    all_meta = []

    for t in TOURNAMENTS:
        print(f"\n[Dataset] === {t['name']} ===")
        print(f"  Loading matches...")
        match_df = _load_tournament_matches(
            t["competition_id"], t["season_id"], t["name"],
        )
        print(f"  {len(match_df)} matches loaded.")

        print(f"  Building team features...")
        team_features = build_team_feature_matrix(
            competition_id=t["competition_id"],
            season_id=t["season_id"],
            elo_key=t["elo_key"],
            league_seasons=t["league_seasons"] or None,
            latest_league_season=t["latest_league_season"],
            prior_national_comps=t["prior_national_comps"] or None,
        )
        feature_cols = team_features.columns.tolist()

        rows = []
        valid_matches = []

        for _, match in match_df.iterrows():
            home, away = match["home_team"], match["away_team"]
            if home not in team_features.index or away not in team_features.index:
                continue

            home_f = team_features.loc[home]
            away_f = team_features.loc[away]
            home_d = home_f.to_dict()
            away_d = away_f.to_dict()
            row = {}
            for col in feature_cols:
                row[f"home_{col}"] = home_f[col]
                row[f"away_{col}"] = away_f[col]
                hv = home_f[col] if pd.notna(home_f[col]) else np.nan
                av = away_f[col] if pd.notna(away_f[col]) else np.nan
                row[f"diff_{col}"] = hv - av if pd.notna(hv) and pd.notna(av) else np.nan

            for mk, mv in positional_matchup_features(home_d, away_d).items():
                row[mk] = mv

            rows.append(row)
            valid_matches.append(match)

        if rows:
            X_t = pd.DataFrame(rows)
            meta_t = pd.DataFrame(valid_matches).reset_index(drop=True)
            all_X.append(X_t)
            all_meta.append(meta_t)
            print(f"  -> {len(X_t)} matches with features.")

    X = pd.concat(all_X, ignore_index=True)
    meta = pd.concat(all_meta, ignore_index=True)
    y = meta["result"]

    n_nan = X.isna().sum().sum()
    n_total = X.shape[0] * X.shape[1]
    pct_available = (1 - n_nan / n_total) * 100

    print(f"\n[Dataset] Combined: {len(X)} matches x {X.shape[1]} features")
    print(f"  Data availability: {pct_available:.1f}%")
    for t_name in meta["tournament"].unique():
        t_mask = meta["tournament"] == t_name
        print(f"  {t_name}: {t_mask.sum()} matches")
    print(f"  Label distribution: {dict(y.value_counts())}")

    return X, y, meta


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    X, y, meta = build_match_dataset()
    print("\n" + "=" * 70)
    print(f"Final shape: {X.shape}")
    print(f"NaN per column (top 10):")
    nan_counts = X.isna().sum().sort_values(ascending=False)
    print(nan_counts.head(10).to_string())
