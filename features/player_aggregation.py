"""
Player-level aggregation into team features (star, positional, weighted).
球员级别聚合为队伍特征（明星球员、位置质量、加权）。

Replaces naive sum across all squad members.
替代对全队球员一视同仁的简单求和。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# National stat fields used in aggregation / 聚合用的国家队统计字段
NATIONAL_FLOAT_FIELDS = [
    "xg", "goals", "shots", "shots_on_target", "key_passes", "through_balls",
    "crosses", "progressive_passes", "dribble_attempts", "dribble_success",
    "carries_count", "progressive_carry_distance", "interceptions", "blocks",
    "clearances", "pressures", "counter_pressures", "tackle_attempts",
    "tackles_won", "goalkeeper_saves",
]

MIN_TOURNAMENT_MINUTES = 90


def _gini(values: list[float]) -> float:
    """Gini coefficient; 0 = equal, 1 = one player dominates."""
    if not values or sum(values) <= 0:
        return 0.0
    arr = np.sort(np.array(values, dtype=float))
    n = len(arr)
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _primary_position(pos: str | None) -> str:
    if not pos:
        return ""
    return pos.split(",")[0].strip().upper()


def _per90(value: float, minutes: float) -> float:
    if minutes <= 0:
        return 0.0
    return value / minutes * 90.0


def aggregate_team_from_players(
    player_rows: list[dict],
) -> dict[str, float]:
    """
    Build team-level features from list of player stat dicts.
    从球员统计列表构建队伍级特征。

    Each row: position, minutes_played, is_starter, + NATIONAL_FLOAT_FIELDS
    """
    feats: dict[str, float] = {}

    squad = [p for p in player_rows if (p.get("minutes_played") or 0) >= MIN_TOURNAMENT_MINUTES]
    if not squad:
        squad = player_rows

    if not squad:
        return feats

    total_mins = sum(p.get("minutes_played") or 0 for p in squad) or 1

    # Minutes-weighted sums / 按出场时间加权求和
    for field in NATIONAL_FLOAT_FIELDS:
        wsum = 0.0
        for p in squad:
            mins = p.get("minutes_played") or 0
            w = mins / total_mins
            wsum += (p.get(field) or 0) * w
        feats[f"nt_w_{field}"] = wsum

    # Star player features / 明星球员（数据驱动）
    xg_list = [float(p.get("xg") or 0) for p in squad]
    goals_list = [float(p.get("goals") or 0) for p in squad]
    drib_list = [float(p.get("dribble_success") or 0) for p in squad]

    xg_sorted = sorted(xg_list, reverse=True)
    feats["star_top3_xg"] = sum(xg_sorted[:3])
    total_goals = sum(goals_list)
    feats["star_top_scorer_share"] = max(goals_list) / total_goals if total_goals > 0 else 0.0
    feats["star_xg_gini"] = _gini(xg_list)
    feats["star_max_dribbles"] = max(drib_list) if drib_list else 0.0
    feats["squad_starters"] = sum(1 for p in squad if p.get("is_starter"))
    feats["squad_depth"] = len(squad)

    # Positional quality (per90) / 位置质量 per90
    pos_buckets: dict[str, list[dict]] = {"GK": [], "DF": [], "MF": [], "FW": []}
    for p in squad:
        pos = _primary_position(p.get("position"))
        if pos in pos_buckets:
            pos_buckets[pos].append(p)

    fw_mins = sum(p.get("minutes_played") or 0 for p in pos_buckets["FW"])
    df_mins = sum(p.get("minutes_played") or 0 for p in pos_buckets["DF"])
    mf_mins = sum(p.get("minutes_played") or 0 for p in pos_buckets["MF"])

    fw_xg = sum(p.get("xg") or 0 for p in pos_buckets["FW"])
    fw_drib = sum(p.get("dribble_success") or 0 for p in pos_buckets["FW"])
    feats["pos_fw_xg_per90"] = _per90(fw_xg, fw_mins)
    feats["pos_fw_dribbles_per90"] = _per90(fw_drib, fw_mins)
    feats["pos_fw_attack_score"] = feats["pos_fw_xg_per90"] + 0.15 * feats["pos_fw_dribbles_per90"]

    df_tkl = sum((p.get("tackles_won") or 0) + (p.get("interceptions") or 0) + (p.get("blocks") or 0)
                 for p in pos_buckets["DF"])
    feats["pos_df_defense_per90"] = _per90(df_tkl, df_mins)

    mf_kp = sum((p.get("key_passes") or 0) + (p.get("progressive_passes") or 0)
                for p in pos_buckets["MF"])
    mf_carries = sum(p.get("carries_count") or 0 for p in pos_buckets["MF"])
    feats["pos_mf_creative_per90"] = _per90(mf_kp, mf_mins)
    feats["pos_mf_carries_per90"] = _per90(mf_carries, mf_mins)

    feats["pos_gk_saves"] = sum(p.get("goalkeeper_saves") or 0 for p in pos_buckets["GK"])

    team_press = sum(p.get("pressures") or 0 for p in squad)
    team_carry = sum(p.get("carries_count") or 0 for p in squad)
    feats["team_pressures_per90"] = _per90(team_press, total_mins)
    feats["team_carries_per90"] = _per90(team_carry, total_mins)

    # Club chemistry: squad members from same club (pre-tournament familiarity proxy).
    # 俱乐部化学反应：同俱乐部队友越多，国家队磨合成本越低（代理变量）。
    clubs = [
        (p.get("current_club") or "").strip()
        for p in squad
        if (p.get("current_club") or "").strip()
    ]
    n_squad = len(squad)
    if clubs and n_squad >= 2:
        from collections import Counter
        club_counts = Counter(clubs)
        max_cluster = max(club_counts.values())
        pair_count = sum(c * (c - 1) // 2 for c in club_counts.values() if c >= 2)
        max_pairs = n_squad * (n_squad - 1) // 2
        feats["club_max_cluster_ratio"] = max_cluster / n_squad
        feats["club_pair_density"] = pair_count / max_pairs if max_pairs else 0.0
        feats["club_distinct_clubs"] = len(club_counts)
    else:
        feats["club_max_cluster_ratio"] = np.nan
        feats["club_pair_density"] = np.nan
        feats["club_distinct_clubs"] = np.nan

    return feats


def positional_matchup_features(home_feats: dict, away_feats: dict) -> dict[str, float]:
    """
    Head-to-head positional matchup deltas for one match.
    单场比赛的位置对位差值特征。
    """
    h_fw = home_feats.get("pos_fw_attack_score", 0) or 0
    a_df = away_feats.get("pos_df_defense_per90", 0) or 0
    a_fw = away_feats.get("pos_fw_attack_score", 0) or 0
    h_df = home_feats.get("pos_df_defense_per90", 0) or 0

    h_mf = home_feats.get("pos_mf_creative_per90", 0) or 0
    a_mf = away_feats.get("pos_mf_creative_per90", 0) or 0

    return {
        "matchup_home_fw_vs_away_df": h_fw - a_df,
        "matchup_away_fw_vs_home_df": a_fw - h_df,
        "matchup_mf_creative_diff": h_mf - a_mf,
        "matchup_press_vs_carry": (
            (home_feats.get("team_pressures_per90") or 0)
            - (away_feats.get("team_carries_per90") or 0)
        ),
        "matchup_carry_vs_press": (
            (home_feats.get("team_carries_per90") or 0)
            - (away_feats.get("team_pressures_per90") or 0)
        ),
    }
