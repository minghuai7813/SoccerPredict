"""
Match-level dataset builder for the 2022 World Cup.
2022 世界杯比赛级别数据集构造器。

This module combines:
  1. Match results from StatsBomb (who played whom, what was the score).
  2. Team-level features from team_features.py (aggregated player stats).
To produce a single flat DataFrame where each row is one match, with
features for both home and away teams plus the actual result.
本模块将 StatsBomb 的比赛结果与 team_features 的队级特征合并，
生成每行代表一场比赛的扁平 DataFrame。

This is the direct input to the prediction model.
这是预测模型的直接输入。

Usage / 用法:
    from features.match_dataset import build_match_dataset
    X, y, meta = build_match_dataset()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from statsbombpy import sb

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.team_features import build_team_feature_matrix

COMPETITION_ID = 43
SEASON_ID = 106


def _load_match_results() -> pd.DataFrame:
    """
    Pull all 2022 World Cup match results from StatsBomb.
    从 StatsBomb 拉取 2022 世界杯所有比赛结果。

    Returns a DataFrame with columns:
      match_id, home_team, away_team, home_score, away_score, result.
    result encoding: 1 = home win, 0 = draw, -1 = away win.
    result 编码：1 = 主队赢, 0 = 平局, -1 = 客队赢。
    """
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)

    records = []
    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        h_score = int(row["home_score"])
        a_score = int(row["away_score"])

        if h_score > a_score:
            result = 1
        elif h_score < a_score:
            result = -1
        else:
            result = 0

        records.append({
            "match_id": row["match_id"],
            "home_team": home,
            "away_team": away,
            "home_score": h_score,
            "away_score": a_score,
            "result": result,
            "goal_diff": h_score - a_score,
        })

    return pd.DataFrame(records)


def build_match_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build the full match dataset: features + labels.
    构建完整的比赛数据集：特征 + 标签。

    For each match, we create a feature vector by concatenating the home
    team's features and the away team's features (prefixed with "home_"
    and "away_"). We also add a "diff_" feature set = home - away for
    each numeric feature, which helps the model learn relative strength.
    每场比赛的特征向量 = 主队特征 + 客队特征 + 差值特征。
    差值特征 = 主队 - 客队，帮助模型学习相对实力。

    Returns
    -------
    X : pd.DataFrame
        Feature matrix, one row per match.
    y : pd.Series
        Labels: 1 (home win), 0 (draw), -1 (away win).
    meta : pd.DataFrame
        Match metadata (match_id, team names, scores) for reference.
    """
    print("[Dataset] Loading match results from StatsBomb...")
    match_df = _load_match_results()
    print(f"  Found {len(match_df)} matches.")

    print("[Dataset] Building team feature matrix...")
    team_features = build_team_feature_matrix()
    feature_cols = team_features.columns.tolist()
    print(f"  {len(feature_cols)} features per team.")

    # For each match, look up home and away team features.
    # 对每场比赛，查找主队和客队的特征。
    rows = []
    valid_matches = []

    for _, match in match_df.iterrows():
        home = match["home_team"]
        away = match["away_team"]

        if home not in team_features.index or away not in team_features.index:
            print(f"  [SKIP] Missing features for {home} vs {away}")
            continue

        home_feats = team_features.loc[home]
        away_feats = team_features.loc[away]

        row = {}
        for col in feature_cols:
            row[f"home_{col}"] = home_feats[col]
            row[f"away_{col}"] = away_feats[col]
            row[f"diff_{col}"] = home_feats[col] - away_feats[col]

        rows.append(row)
        valid_matches.append(match)

    X = pd.DataFrame(rows)
    meta = pd.DataFrame(valid_matches).reset_index(drop=True)
    y = meta["result"]

    print(f"\n[Dataset] Final dataset: {len(X)} matches × {X.shape[1]} features.")
    print(f"  Label distribution: {dict(y.value_counts())}")

    return X, y, meta


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    X, y, meta = build_match_dataset()
    print("\n" + "=" * 70)
    print("Sample matches:")
    print(meta[["home_team", "away_team", "home_score", "away_score", "result"]].head(10).to_string())
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {list(X.columns[:10])} ...")
