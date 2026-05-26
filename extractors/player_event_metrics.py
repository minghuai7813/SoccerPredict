"""
Per-player StatsBomb event aggregation (expanded metrics).
StatsBomb 事件级球员指标聚合（扩展版）。

Shared by WC 2018 / Euro 2020 / WC 2022 extractors.
供各届大赛提取脚本共用。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

# Keys accumulated per player per match (excluding player_name).
# 每场每球员累加的指标键（不含 player_name）。
METRIC_KEYS = [
    "xg",
    "goals",
    "shots",
    "shots_on_target",
    "passes_completed",
    "passes_attempted",
    "key_passes",
    "through_balls",
    "crosses",
    "progressive_passes",
    "dribble_attempts",
    "dribble_success",
    "carries_count",
    "progressive_carry_distance",
    "interceptions",
    "blocks",
    "clearances",
    "aerial_duels_won",
    "aerial_duels_lost",
    "ball_recoveries",
    "fouls_committed",
    "fouls_won",
    "tackle_attempts",
    "tackles_won",
    "pressures",
    "counter_pressures",
    "actions_under_pressure",
    "yellow_cards",
    "red_cards",
    "goalkeeper_saves",
]


def empty_metrics() -> dict[str, float | int]:
    """Return zeroed metric dict / 返回归零指标字典。"""
    return {k: 0 if k != "xg" and k != "progressive_carry_distance" else 0.0 for k in METRIC_KEYS}


def merge_metrics(acc: dict[str, float | int], row: dict[str, float | int]) -> None:
    """In-place merge match stats into accumulator / 就地合并单场统计。"""
    for k in METRIC_KEYS:
        acc[k] = acc[k] + row[k]


def _xy(loc: Any) -> tuple[float | None, float | None]:
    """Parse StatsBomb location [x, y] / 解析坐标。"""
    if loc is None or (isinstance(loc, float) and pd.isna(loc)):
        return None, None
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None, None


def _progressive_distance(start: Any, end: Any, min_dist: float = 10.0) -> float:
    """
    Euclidean carry/pass progression proxy when direction unknown.
    未知进攻方向时用欧氏距离近似 progressive distance。
    """
    x0, y0 = _xy(start)
    x1, y1 = _xy(end)
    if x0 is None or x1 is None:
        return 0.0
    d = math.hypot(x1 - x0, y1 - y0)
    return d if d >= min_dist else 0.0


def _is_true(val: Any) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return bool(val)


def _aggregate_player_events(match_id: int, sb_module: Any) -> list[dict]:
    """
    Aggregate per-player stats from one match event stream.
    从单场比赛事件流聚合每名球员统计。
    """
    import warnings
    warnings.filterwarnings("ignore")

    events = sb_module.events(match_id=match_id)
    if events.empty:
        return []

    players_in_match = events["player"].dropna().unique()
    results: list[dict] = []

    for player_name in players_in_match:
        pe = events[events["player"] == player_name]
        m = empty_metrics()

        # --- Shots / 射门 ---
        shots = pe[pe["type"] == "Shot"]
        m["shots"] = len(shots)
        if not shots.empty:
            if "shot_statsbomb_xg" in shots.columns:
                m["xg"] = round(float(shots["shot_statsbomb_xg"].sum()), 4)
            if "shot_outcome" in shots.columns:
                sot = {"Goal", "Saved", "Saved to Post"}
                m["shots_on_target"] = int(shots["shot_outcome"].isin(sot).sum())
                m["goals"] = int((shots["shot_outcome"] == "Goal").sum())

        # --- Passes / 传球 ---
        passes = pe[pe["type"] == "Pass"]
        m["passes_attempted"] = len(passes)
        if m["passes_attempted"] > 0:
            if "pass_outcome" in passes.columns:
                m["passes_completed"] = int(passes["pass_outcome"].isna().sum())
            if "pass_shot_assist" in passes.columns:
                m["key_passes"] = int(passes["pass_shot_assist"].apply(_is_true).sum())
            if "pass_through_ball" in passes.columns:
                m["through_balls"] = int(passes["pass_through_ball"].apply(_is_true).sum())
            elif "pass_type" in passes.columns:
                m["through_balls"] = int(
                    passes["pass_type"].astype(str).str.contains("Through", case=False, na=False).sum()
                )
            if "pass_cross" in passes.columns:
                m["crosses"] = int(passes["pass_cross"].apply(_is_true).sum())
            has_plen = "pass_length" in passes.columns
            for _, prow in passes.iterrows():
                if has_plen and pd.notna(prow.get("pass_length")):
                    if float(prow["pass_length"]) >= 25:
                        m["progressive_passes"] += 1
                else:
                    m["progressive_passes"] += int(
                        _progressive_distance(prow.get("location"), prow.get("pass_end_location")) > 0
                    )

        # --- Dribbles / 带球 ---
        dribbles = pe[pe["type"] == "Dribble"]
        m["dribble_attempts"] = len(dribbles)
        if not dribbles.empty and "dribble_outcome" in dribbles.columns:
            m["dribble_success"] = int(
                dribbles["dribble_outcome"].isin(["Complete", "Success"]).sum()
            )

        # --- Carries / 推进 ---
        carries = pe[pe["type"] == "Carry"]
        m["carries_count"] = len(carries)
        for _, crow in carries.iterrows():
            dist = _progressive_distance(crow.get("location"), crow.get("carry_end_location"))
            m["progressive_carry_distance"] += dist

        # --- Defensive / 防守 ---
        m["interceptions"] = len(pe[pe["type"] == "Interception"])
        m["blocks"] = len(pe[pe["type"] == "Block"])
        m["clearances"] = len(pe[pe["type"] == "Clearance"])
        m["ball_recoveries"] = len(pe[pe["type"] == "Ball Recovery"])

        duels = pe[pe["type"] == "Duel"]
        if not duels.empty and "duel_type" in duels.columns:
            aerial = duels[duels["duel_type"].astype(str).str.contains("Aerial", na=False)]
            if not aerial.empty and "duel_outcome" in aerial.columns:
                won = aerial["duel_outcome"].isin(["Won", "Success"])
                m["aerial_duels_won"] = int(won.sum())
                m["aerial_duels_lost"] = int((~won).sum())

            tackle_duels = duels[duels["duel_type"] == "Tackle"]
            m["tackle_attempts"] = len(tackle_duels)
            if not tackle_duels.empty and "duel_outcome" in tackle_duels.columns:
                won_out = ["Won", "Success In Play", "Success Out"]
                m["tackles_won"] = int(tackle_duels["duel_outcome"].isin(won_out).sum())

        # --- Fouls & cards / 犯规与牌 ---
        m["fouls_committed"] = len(pe[pe["type"] == "Foul Committed"])
        m["fouls_won"] = len(pe[pe["type"] == "Foul Won"])

        fouls = pe[pe["type"] == "Foul Committed"]
        if not fouls.empty and "foul_committed_card" in fouls.columns:
            for card in fouls["foul_committed_card"].dropna().astype(str):
                cl = card.lower()
                if "yellow" in cl:
                    m["yellow_cards"] += 1
                if "red" in cl:
                    m["red_cards"] += 1

        bad = pe[pe["type"] == "Bad Behaviour"]
        if not bad.empty and "bad_behaviour_card" in bad.columns:
            for card in bad["bad_behaviour_card"].dropna().astype(str):
                cl = card.lower()
                if "yellow" in cl:
                    m["yellow_cards"] += 1
                if "red" in cl:
                    m["red_cards"] += 1

        # --- Pressing / 压迫 ---
        pressures = pe[pe["type"] == "Pressure"]
        m["pressures"] = len(pressures)
        if not pressures.empty and "counterpress" in pressures.columns:
            m["counter_pressures"] = int(pressures["counterpress"].apply(_is_true).sum())

        if "under_pressure" in pe.columns:
            m["actions_under_pressure"] = int(pe["under_pressure"].apply(_is_true).sum())

        # --- Goalkeeper / 门将 ---
        gk = pe[pe["type"] == "Goal Keeper"]
        if not gk.empty and "goalkeeper_outcome" in gk.columns:
            m["goalkeeper_saves"] = int(
                gk["goalkeeper_outcome"].astype(str).str.contains("Saved", na=False).sum()
            )

        m["xg"] = round(float(m["xg"]), 4)
        m["progressive_carry_distance"] = round(float(m["progressive_carry_distance"]), 2)
        m["player_name"] = player_name
        results.append(m)

    return results
