"""
Load and normalize match 1X2 odds (OddsPortal CSV + Elo fallback).
加载并标准化比赛胜平负赔率（OddsPortal CSV + Elo 回退）。

Usage / 用法:
    from extractors.odds_loader import load_match_odds, attach_odds_to_meta
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.market_odds import LABEL_AWAY, LABEL_DRAW, LABEL_HOME, elo_1x2_probs

ODDS_DIR = _PROJECT_ROOT / "data" / "odds"
UNIFIED_PATH = ODDS_DIR / "match_odds_unified.csv"

# StatsBomb name -> OddsPortal / common bookmaker spelling
TEAM_ALIASES: dict[str, str] = {
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "ir iran": "iran",
    "iran": "iran",
    "usa": "united states",
    "united states": "united states",
    "côte d'ivoire": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "czechia": "czech republic",
    "türkiye": "turkey",
    "turkiye": "turkey",
    "bosnia and herzegovina": "bosnia & herzegovina",
}


def normalize_team(name: str) -> str:
    """Lowercase canonical key for fuzzy joins."""
    if not name or not isinstance(name, str):
        return ""
    n = name.strip().lower()
    n = re.sub(r"\s+", " ", n)
    return TEAM_ALIASES.get(n, n)


def _parse_1x2_market(cell: str) -> dict[str, float] | None:
    """Extract bet365 (or average) decimal odds from OddsPortal cell."""
    if not cell or (isinstance(cell, float) and np.isnan(cell)):
        return None
    try:
        markets = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except (ValueError, SyntaxError):
        try:
            markets = json.loads(cell.replace("'", '"'))
        except Exception:
            return None
    if not markets:
        return None

    prefer = ("bet365", "Betway", "888sport", "1xBet")
    chosen = None
    for bk in prefer:
        for m in markets:
            if str(m.get("bookmaker_name", "")).lower() == bk.lower():
                chosen = m
                break
        if chosen:
            break
    if chosen is None:
        chosen = markets[0]

    try:
        return {
            "home": float(chosen["1"]),
            "draw": float(chosen["X"]),
            "away": float(chosen["2"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def odds_to_probs(oh: float, od: float, oa: float) -> dict[int, float]:
    """De-vig implied probabilities from decimal odds."""
    inv = np.array([1.0 / max(oh, 1.01), 1.0 / max(od, 1.01), 1.0 / max(oa, 1.01)])
    inv /= inv.sum()
    return {LABEL_HOME: float(inv[0]), LABEL_DRAW: float(inv[1]), LABEL_AWAY: float(inv[2])}


def _load_oddsportal_csv(path: Path, tournament: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        odds = _parse_1x2_market(r.get("1x2_market", ""))
        if odds is None:
            continue
        home = str(r.get("home_team", "")).strip()
        away = str(r.get("away_team", "")).strip()
        probs = odds_to_probs(odds["home"], odds["draw"], odds["away"])
        rows.append({
            "tournament": tournament,
            "home_team": home,
            "away_team": away,
            "home_key": normalize_team(home),
            "away_key": normalize_team(away),
            "match_date": pd.to_datetime(r.get("match_date"), utc=True, errors="coerce"),
            "odds_home": odds["home"],
            "odds_draw": odds["draw"],
            "odds_away": odds["away"],
            "prob_home": probs[LABEL_HOME],
            "prob_draw": probs[LABEL_DRAW],
            "prob_away": probs[LABEL_AWAY],
            "odds_source": "oddsportal",
            "bookmaker": "bet365_or_avg",
        })
    return pd.DataFrame(rows)


def load_raw_odds_files() -> pd.DataFrame:
    """Load all OddsPortal CSVs in data/odds/."""
    files = [
        (ODDS_DIR / "wc2018_oddsportal.csv", "FIFA World Cup 2018"),
        (ODDS_DIR / "wc2022_oddsportal.csv", "FIFA World Cup 2022"),
        (ODDS_DIR / "euro2020_oddsportal.csv", "UEFA Euro 2020"),
    ]
    parts = [_load_oddsportal_csv(p, t) for p, t in files]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_match_odds(*, rebuild: bool = False) -> pd.DataFrame:
    """
    Unified odds table; rebuild from raw CSVs when requested.
    统一赔率表；rebuild=True 时从原始 CSV 重建。
    """
    if UNIFIED_PATH.exists() and not rebuild:
        return pd.read_csv(UNIFIED_PATH, parse_dates=["match_date"])

    raw = load_raw_odds_files()
    if raw.empty:
        return raw
    raw = raw.drop_duplicates(subset=["tournament", "home_key", "away_key"], keep="first")
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(UNIFIED_PATH, index=False)
    return raw


def attach_odds_to_meta(
    meta: pd.DataFrame,
    odds_df: pd.DataFrame,
    elo_caches: dict[str, dict],
) -> pd.DataFrame:
    """
    Join odds onto match metadata; fill gaps with Elo-implied probs.
    将赔率合并到比赛元数据；缺失用 Elo 补齐。
    """
    out = meta.copy()
    out["home_key"] = out["home_team"].map(normalize_team)
    out["away_key"] = out["away_team"].map(normalize_team)

    if odds_df.empty:
        odds_df = pd.DataFrame()

    if not odds_df.empty:
        odds_df = odds_df.copy()
        if "home_key" not in odds_df.columns:
            odds_df["home_key"] = odds_df["home_team"].map(normalize_team)
            odds_df["away_key"] = odds_df["away_team"].map(normalize_team)

    prob_cols = ["prob_home", "prob_draw", "prob_away", "odds_home", "odds_draw", "odds_away"]
    for c in prob_cols + ["odds_source"]:
        out[c] = np.nan
    out["odds_source"] = ""

    for i, row in out.iterrows():
        tournament = row["tournament"]
        hk, ak = row["home_key"], row["away_key"]

        hit = None
        if not odds_df.empty:
            mask = (
                (odds_df["tournament"] == tournament)
                & (odds_df["home_key"] == hk)
                & (odds_df["away_key"] == ak)
            )
            hits = odds_df[mask]
            if hits.empty:
                # try swapped home/away (OddsPortal sometimes flips)
                mask2 = (
                    (odds_df["tournament"] == tournament)
                    & (odds_df["home_key"] == ak)
                    & (odds_df["away_key"] == hk)
                )
                hits = odds_df[mask2]
                if not hits.empty:
                    h = hits.iloc[0]
                    out.at[i, "prob_home"] = h["prob_away"]
                    out.at[i, "prob_draw"] = h["prob_draw"]
                    out.at[i, "prob_away"] = h["prob_home"]
                    out.at[i, "odds_home"] = h["odds_away"]
                    out.at[i, "odds_draw"] = h["odds_draw"]
                    out.at[i, "odds_away"] = h["odds_home"]
                    out.at[i, "odds_source"] = "oddsportal_swapped"
                    continue
            if not hits.empty:
                hit = hits.iloc[0]

        if hit is not None:
            out.at[i, "prob_home"] = hit["prob_home"]
            out.at[i, "prob_draw"] = hit["prob_draw"]
            out.at[i, "prob_away"] = hit["prob_away"]
            out.at[i, "odds_home"] = hit["odds_home"]
            out.at[i, "odds_draw"] = hit["odds_draw"]
            out.at[i, "odds_away"] = hit["odds_away"]
            out.at[i, "odds_source"] = hit.get("odds_source", "oddsportal")
        else:
            ed = elo_caches.get(tournament, {})
            eh = ed.get(row["home_team"], {}).get("elo", 1500)
            ea = ed.get(row["away_team"], {}).get("elo", 1500)
            ep = elo_1x2_probs(eh, ea)
            out.at[i, "prob_home"] = ep[LABEL_HOME]
            out.at[i, "prob_draw"] = ep[LABEL_DRAW]
            out.at[i, "prob_away"] = ep[LABEL_AWAY]
            out.at[i, "odds_home"] = 1.0 / ep[LABEL_HOME]
            out.at[i, "odds_draw"] = 1.0 / ep[LABEL_DRAW]
            out.at[i, "odds_away"] = 1.0 / ep[LABEL_AWAY]
            out.at[i, "odds_source"] = "elo_fallback"

    return out
