"""
Match-level feature row from two team profiles.
从主客队画像构建单场比赛特征行。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.player_aggregation import positional_matchup_features


def build_match_feature_row(
    home_feats: dict[str, float],
    away_feats: dict[str, float],
    base_columns: list[str] | None = None,
) -> dict[str, float]:
    """
    Combine home/away/diff + positional matchup into one feature dict.
    合并 home/away/diff 与位置对位特征。
    """
    all_keys = set(home_feats) | set(away_feats)
    row: dict[str, float] = {}

    for col in sorted(all_keys):
        hv = home_feats.get(col, np.nan)
        av = away_feats.get(col, np.nan)
        row[f"home_{col}"] = hv
        row[f"away_{col}"] = av
        if pd.notna(hv) and pd.notna(av):
            row[f"diff_{col}"] = float(hv) - float(av)
        else:
            row[f"diff_{col}"] = np.nan

    for mk, mv in positional_matchup_features(home_feats, away_feats).items():
        row[mk] = mv

    if base_columns is not None:
        return {c: row.get(c, np.nan) for c in base_columns}
    return row


def profiles_to_dataframe(
    home_feats: dict[str, float],
    away_feats: dict[str, float],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Single-row DataFrame aligned to training columns."""
    row = build_match_feature_row(home_feats, away_feats, feature_columns)
    return pd.DataFrame([row], columns=feature_columns)
