"""
Upset (dark horse) dataset: lopsided matches only, binary upset label.
冷门数据集：仅悬殊局，标签 = 热门未赢（含平局）。

Usage / 用法:
    from features.upset_dataset import build_upset_dataset, MIN_ELO_GAP
    X, y, meta = build_upset_dataset()
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extractors.elo_scraper import get_pre_tournament_elo
from features.match_dataset import build_match_dataset

MIN_ELO_GAP = 200

ELO_MAP = {
    "FIFA World Cup 2018": "wc2018",
    "UEFA Euro 2020": "euro2020",
    "FIFA World Cup 2022": "wc2022",
}


def _favorite_win(result: int, home: str, away: str, favorite: str) -> bool:
    """True if 90-min result is a win for the Elo favorite."""
    return (result == 1 and favorite == home) or (result == -1 and favorite == away)


def annotate_elo_and_upset(
    meta: pd.DataFrame,
    y: pd.Series,
    *,
    min_elo_gap: int = MIN_ELO_GAP,
) -> pd.DataFrame:
    """
    Add elo_gap, favorite, underdog, upset (1 = favorite did not win).
    为 meta 增加 Elo 差、热门、冷门标签。
    """
    caches = {k: get_pre_tournament_elo(v) for k, v in ELO_MAP.items()}
    extras = []
    for i in range(len(meta)):
        home = meta.iloc[i]["home_team"]
        away = meta.iloc[i]["away_team"]
        t = meta.iloc[i]["tournament"]
        ed = caches[t]
        eh = ed.get(home, {}).get("elo", 1500)
        ea = ed.get(away, {}).get("elo", 1500)
        gap = abs(eh - ea)
        if eh >= ea:
            favorite, underdog = home, away
        else:
            favorite, underdog = away, home
        result = int(y.iloc[i])
        fav_win = _favorite_win(result, home, away, favorite)
        extras.append({
            "elo_home": eh,
            "elo_away": ea,
            "elo_gap": gap,
            "favorite": favorite,
            "underdog": underdog,
            "favorite_win": fav_win,
            "upset": int(not fav_win),
            "is_lopsided": gap >= min_elo_gap,
        })
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(extras)], axis=1)


def build_upset_dataset(
    min_elo_gap: int = MIN_ELO_GAP,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build feature matrix for lopsided matches with binary upset label.

    Returns
    -------
    X, y_upset, meta — only rows with elo_gap >= min_elo_gap.
    y_upset : 1 = upset (draw or underdog win), 0 = favorite win.
    """
    X, y, meta = build_match_dataset()
    enriched = annotate_elo_and_upset(meta, y, min_elo_gap=min_elo_gap)
    mask = enriched["is_lopsided"].values
    X_u = X.loc[mask].reset_index(drop=True)
    meta_u = enriched.loc[mask].reset_index(drop=True)
    y_upset = meta_u["upset"].astype(int)

    if verbose:
        n = len(y_upset)
        n_up = int(y_upset.sum())
        print(f"\n[Upset Dataset] Elo gap >= {min_elo_gap}: {n} matches")
        print(f"  Upsets (fav did not win): {n_up} ({n_up/n:.1%})")
        print(f"  Favorite wins: {n - n_up} ({(n-n_up)/n:.1%})")
        for t in meta_u["tournament"].unique():
            m = meta_u["tournament"] == t
            print(f"  {t}: {m.sum()} matches, upsets {y_upset[m].sum()}")

    return X_u, y_upset, meta_u
