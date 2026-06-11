"""
Roster-centric team profile builder (club or national squad).
基于名单/预计上场的队伍画像构建器（俱乐部与国家队共用）。

The model should depend on *who is playing*, not on the entity type.
模型应依赖「上场球员组成」，而非国家队/俱乐部标签。

Usage / 用法:
    from features.team_profile import PlayerSlot, build_club_profile, build_national_profile
    profile = build_club_profile("Paris Saint-Germain", roster, club_elo=1964)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Player, PlayerStatsLeague, PlayerStatsNational
from features.player_aggregation import NATIONAL_FLOAT_FIELDS, aggregate_team_from_players

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"

from features.club_names import FOOTBALL_DATA_TO_CANONICAL

# Common ClubElo / media aliases → canonical club name for DB lookup.
CLUB_ALIASES: dict[str, str] = {
    **{k.lower(): v for k, v in FOOTBALL_DATA_TO_CANONICAL.items()},
    "psg": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
    "paris saint-germain": "Paris Saint-Germain",
    "paris saint germain": "Paris Saint-Germain",
    "arsenal fc": "Arsenal",
    "arsenal": "Arsenal",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "real madrid": "Real Madrid",
    "barcelona": "Barcelona",
    "fc barcelona": "Barcelona",
    "bayern": "Bayern Munich",
    "bayern munich": "Bayern Munich",
}

# League stat columns mapped into aggregate_team_from_players field names.
# 联赛字段映射到聚合函数使用的统一字段名。
LEAGUE_TO_AGGREGATION: dict[str, str] = {
    "xg": "xg",
    "goals": "goals",
    "assists": "assists",
    "interceptions": "interceptions",
    "tackles_won": "tackles_won",
    "passes_completed": "passes_completed",
}

# Position-based per-90 baselines derived from Big-5 league real data (2024-25).
# These drive proxy estimates for missing advanced stats.
# 基于五大联赛真实数据得出的各位置 per-90 基准值，用于反推缺失的高级指标。
_POS_BASELINES_P90: dict[str, dict[str, float]] = {
    "GK": {
        "shots": 0.00, "shots_on_target": 0.00, "key_passes": 0.04,
        "through_balls": 0.00, "crosses": 0.00, "progressive_passes": 0.10,
        "dribble_attempts": 0.02, "dribble_success": 0.01,
        "carries_count": 1.0, "progressive_carry_distance": 5.0,
        "blocks": 0.05, "clearances": 0.50,
        "pressures": 0.30, "counter_pressures": 0.05,
        "tackle_attempts": 0.06, "goalkeeper_saves": 2.5,
    },
    "DF": {
        "shots": 0.50, "shots_on_target": 0.18, "key_passes": 0.35,
        "through_balls": 0.02, "crosses": 0.30, "progressive_passes": 2.0,
        "dribble_attempts": 0.40, "dribble_success": 0.25,
        "carries_count": 3.0, "progressive_carry_distance": 30.0,
        "blocks": 0.70, "clearances": 2.5,
        "pressures": 4.0, "counter_pressures": 0.50,
        "tackle_attempts": 1.80, "goalkeeper_saves": 0.0,
    },
    "MF": {
        "shots": 1.20, "shots_on_target": 0.45, "key_passes": 1.0,
        "through_balls": 0.10, "crosses": 0.50, "progressive_passes": 3.5,
        "dribble_attempts": 1.5, "dribble_success": 0.90,
        "carries_count": 5.0, "progressive_carry_distance": 60.0,
        "blocks": 0.50, "clearances": 0.80,
        "pressures": 7.0, "counter_pressures": 0.80,
        "tackle_attempts": 1.70, "goalkeeper_saves": 0.0,
    },
    "FW": {
        "shots": 2.50, "shots_on_target": 1.10, "key_passes": 0.80,
        "through_balls": 0.08, "crosses": 0.35, "progressive_passes": 1.0,
        "dribble_attempts": 2.5, "dribble_success": 1.30,
        "carries_count": 4.0, "progressive_carry_distance": 40.0,
        "blocks": 0.20, "clearances": 0.15,
        "pressures": 5.5, "counter_pressures": 0.60,
        "tackle_attempts": 0.80, "goalkeeper_saves": 0.0,
    },
}


@dataclass
class PlayerSlot:
    """
    One roster slot with expected minutes (starter or sub).
    单个名单位：球员 + 预计出场时间（首发/替补权重）。
    """

    name: str | None = None
    player_id: str | None = None
    position: str | None = None
    expected_minutes: float = 90.0
    is_starter: bool = True


@dataclass
class SquadContext:
    """
    Shared DB lookups when building one side's profile.
    构建单方画像时的共享数据库上下文。
    """

    pid_to_pos: dict[str, str] = field(default_factory=dict)
    pid_to_club: dict[str, str] = field(default_factory=dict)
    pid_to_name: dict[str, str] = field(default_factory=dict)


def normalize_club_name(name: str) -> str:
    """Normalize club alias to canonical form."""
    key = re.sub(r"\s+", " ", name.strip().lower())
    return CLUB_ALIASES.get(key, name.strip())


def _compute_trend(values: list[float]) -> float:
    """Simple linear slope across seasons (positive = improving)."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    if np.std(y) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _elo_conf_features(
    elo_rating: float,
    elo_rank: int,
    confederation: str = "UEFA",
) -> dict[str, float]:
    feats: dict[str, float] = {
        "elo_rating": elo_rating,
        "elo_rank": float(elo_rank),
    }
    for c in ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC"]:
        feats[f"conf_{c}"] = 1.0 if confederation == c else 0.0
    return feats


def _league_features_for_squad(
    team_pids: set[str],
    season_data: dict[str, list[dict]],
    league_seasons: dict[str, float],
    latest_league_season: str | None,
) -> dict[str, float]:
    """Recency-weighted league aggregates for a squad subset."""
    feats: dict[str, float] = {}
    if not league_seasons or not season_data:
        for k in [
            "lg_goals_weighted", "lg_assists_weighted", "lg_minutes_weighted",
            "lg_goal_involvement_weighted", "lg_player_count", "lg_seasons_available",
            "lg_goals_per90", "lg_goals_max", "lg_goal_trend", "lg_consistency",
        ]:
            feats[k] = np.nan
        return feats

    weighted_goals = weighted_assists = weighted_minutes = 0.0
    total_weight = 0.0
    season_goal_avgs: list[float] = []

    for season in sorted(league_seasons.keys()):
        w = league_seasons[season]
        players_in = season_data.get(season, [])
        if not players_in:
            continue
        s_goals = sum(p["goals"] for p in players_in)
        s_assists = sum(p["assists"] for p in players_in)
        s_minutes = sum(p["minutes"] for p in players_in)
        n = len(players_in)
        weighted_goals += s_goals * w
        weighted_assists += s_assists * w
        weighted_minutes += s_minutes * w
        total_weight += w
        season_goal_avgs.append(s_goals / n if n > 0 else 0)

    if total_weight > 0:
        feats["lg_goals_weighted"] = weighted_goals / total_weight
        feats["lg_assists_weighted"] = weighted_assists / total_weight
        feats["lg_minutes_weighted"] = weighted_minutes / total_weight
        feats["lg_goal_involvement_weighted"] = (weighted_goals + weighted_assists) / total_weight
    else:
        for k in ["lg_goals_weighted", "lg_assists_weighted", "lg_minutes_weighted",
                  "lg_goal_involvement_weighted"]:
            feats[k] = np.nan

    latest = season_data.get(latest_league_season, []) if latest_league_season else []
    feats["lg_player_count"] = float(len(latest)) if latest else np.nan
    feats["lg_seasons_available"] = float(len(season_data))

    total_g = sum(p["goals"] for p in latest) if latest else 0
    total_m = sum(p["minutes"] for p in latest) if latest else 0
    feats["lg_goals_per90"] = (total_g / total_m * 90) if total_m > 0 else np.nan

    all_pg = [p["goals"] for s in season_data.values() for p in s]
    feats["lg_goals_max"] = float(max(all_pg)) if all_pg else np.nan
    feats["lg_goal_trend"] = _compute_trend(season_goal_avgs)
    if len(season_goal_avgs) >= 2:
        mg = np.mean(season_goal_avgs)
        feats["lg_consistency"] = float(np.std(season_goal_avgs) / mg) if mg > 0 else 0.0
    else:
        feats["lg_consistency"] = np.nan
    return feats


def _positional_composition_features(
    team_pids: set[str],
    pid_to_pos: dict[str, str],
    season_data: dict[str, list[dict]],
    latest_league_season: str | None,
) -> dict[str, float]:
    feats: dict[str, float] = {}
    if not latest_league_season or not season_data.get(latest_league_season):
        for k in ["pos_fw_ratio", "pos_mf_ratio", "pos_df_ratio", "pos_fw_goals", "pos_mf_goals"]:
            feats[k] = np.nan
        return feats

    latest_data = season_data[latest_league_season]
    pos_counts = {"GK": 0, "DF": 0, "MF": 0, "FW": 0}
    pos_goals = {"DF": 0, "MF": 0, "FW": 0}
    for pid in team_pids:
        pos = pid_to_pos.get(pid, "")
        if pos in pos_counts:
            pos_counts[pos] += 1
        if pos in pos_goals:
            for p in latest_data:
                if p["pid"] == pid:
                    pos_goals[pos] += p["goals"]

    total_outfield = pos_counts["DF"] + pos_counts["MF"] + pos_counts["FW"]
    feats["pos_fw_ratio"] = pos_counts["FW"] / total_outfield if total_outfield > 0 else np.nan
    feats["pos_mf_ratio"] = pos_counts["MF"] / total_outfield if total_outfield > 0 else np.nan
    feats["pos_df_ratio"] = pos_counts["DF"] / total_outfield if total_outfield > 0 else np.nan
    feats["pos_fw_goals"] = float(pos_goals["FW"])
    feats["pos_mf_goals"] = float(pos_goals["MF"])
    return feats


def _national_aggregation_features(
    team_pids: set[str],
    national_orm_rows: list[PlayerStatsNational],
    pid_to_pos: dict[str, str],
    pid_to_club: dict[str, str],
    prior_national_comps_count: int,
) -> dict[str, float]:
    player_combined: dict[str, dict] = {}
    for nrow in national_orm_rows:
        pid = nrow.internal_player_id
        if pid not in team_pids:
            continue
        if pid not in player_combined:
            player_combined[pid] = {
                "position": pid_to_pos.get(pid, ""),
                "current_club": pid_to_club.get(pid, ""),
                "minutes_played": 0,
                "is_starter": 0,
            }
            for f in NATIONAL_FLOAT_FIELDS:
                player_combined[pid][f] = 0
        for f in NATIONAL_FLOAT_FIELDS:
            player_combined[pid][f] += getattr(nrow, f, None) or 0
        player_combined[pid]["minutes_played"] += nrow.minutes_played or 0
        player_combined[pid]["is_starter"] = max(
            player_combined[pid]["is_starter"], nrow.is_starter or 0,
        )

    feats: dict[str, float] = {"nt_competitions_count": float(prior_national_comps_count)}
    if player_combined:
        feats.update(aggregate_team_from_players(list(player_combined.values())))
    return feats


def _league_rows_to_player_dicts(
    slots: list[PlayerSlot],
    league_stat_rows: dict[str, dict],
    ctx: SquadContext,
    same_club: str | None = None,
) -> list[dict]:
    """
    Convert league per-player stats + expected minutes into aggregation rows.
    将联赛 per-player 统计与预计分钟转为 aggregate_team_from_players 输入。

    Uses position-aware proxy estimation for missing advanced metrics:
    当高级指标缺失时，根据球员位置和已有数据（goals/xG/assists/tackles）
    推算合理的 shots / key_passes / dribbles / pressures 等。
    """
    rows: list[dict] = []
    for slot in slots:
        pid = slot.player_id
        if not pid or pid not in league_stat_rows:
            continue
        stat = league_stat_rows[pid]
        mins = max(float(slot.expected_minutes), 1.0)
        pos_raw = slot.position or ctx.pid_to_pos.get(pid, "")
        pos_key = pos_raw.split(",")[0].strip().upper() if pos_raw else ""
        if pos_key not in _POS_BASELINES_P90:
            pos_key = "MF"

        row: dict = {
            "position": pos_raw,
            "current_club": same_club or ctx.pid_to_club.get(pid, ""),
            "minutes_played": mins,
            "is_starter": 1 if slot.is_starter else 0,
        }
        for f in NATIONAL_FLOAT_FIELDS:
            row[f] = 0.0

        season_mins = max(float(stat.get("minutes_played") or 0), 90.0)
        scale = mins / season_mins

        # Step 1: copy over directly mapped DB columns.
        # 第一步：直接映射 DB 已有字段。
        for league_col, agg_col in LEAGUE_TO_AGGREGATION.items():
            raw = stat.get(league_col)
            if raw is not None:
                row[agg_col] = float(raw) * scale

        # Step 2: xG proxy from goals when missing.
        # 第二步：缺少 xG 时用进球数反推。
        if (row.get("xg") or 0) == 0 and stat.get("goals"):
            row["xg"] = float(stat["goals"]) * 0.65 * scale
        if row.get("goals", 0) == 0 and stat.get("goals"):
            row["goals"] = float(stat["goals"]) * scale

        # Step 3: position-aware proxy for missing advanced stats.
        # 第三步：基于位置 + 已有数据推算缺失的高级指标。
        # Uses a blend: if the player has real data for a related metric,
        # derive from it; otherwise fall back to positional p90 baselines.
        base = _POS_BASELINES_P90[pos_key]

        xg_val = row.get("xg") or 0.0
        goals_val = row.get("goals") or 0.0
        assists_val = row.get("assists") or 0.0
        tackles_val = row.get("tackles_won") or 0.0
        intercept_val = row.get("interceptions") or 0.0

        # Per-90 rates for this player (based on expected tournament minutes).
        xg_p90 = xg_val / mins * 90 if mins > 0 else 0
        goals_p90 = goals_val / mins * 90 if mins > 0 else 0
        tackles_p90 = tackles_val / mins * 90 if mins > 0 else 0
        intercept_p90 = intercept_val / mins * 90 if mins > 0 else 0

        # Quality factor: how this player compares to Big-5 positional median.
        # Scales proxy baselines up/down so strong players get higher estimates
        # and weaker-league players get appropriately lower values.
        # 质量因子：用球员实际数据相对 Big-5 中位数的比例来缩放估算值，
        # 使得强力球员得到更高估算，弱联赛球员得到合理折扣。
        _POS_QUALITY_REF = {
            "GK": {"def_ref": 0.30},
            "DF": {"def_ref": 2.21, "atk_ref": 0.08},
            "MF": {"atk_ref": 0.21, "def_ref": 1.08},
            "FW": {"atk_ref": 0.38, "def_ref": 0.55},
        }
        ref = _POS_QUALITY_REF[pos_key]
        if pos_key in ("FW", "MF"):
            q_atk = min(xg_p90 / ref["atk_ref"], 2.0) if ref.get("atk_ref") and xg_p90 > 0 else 0.5
            q_def = min((tackles_p90 + intercept_p90) / ref["def_ref"], 2.0) if (tackles_p90 + intercept_p90) > 0 else 0.5
        elif pos_key == "DF":
            q_def = min((tackles_p90 + intercept_p90) / ref["def_ref"], 2.0) if (tackles_p90 + intercept_p90) > 0 else 0.5
            q_atk = min(xg_p90 / ref["atk_ref"], 2.0) if ref.get("atk_ref") and xg_p90 > 0 else 0.5
        else:
            q_atk = 0.5
            q_def = min((tackles_p90 + intercept_p90) / ref.get("def_ref", 1.0), 2.0) if (tackles_p90 + intercept_p90) > 0 else 0.5

        q_atk = max(q_atk, 0.25)
        q_def = max(q_def, 0.25)

        # --- Attacking proxies (shots, SoT) ---
        if row.get("shots", 0) == 0:
            if xg_p90 > 0:
                shots_p90 = xg_p90 / 0.10
            else:
                shots_p90 = base["shots"] * q_atk
            row["shots"] = shots_p90 * mins / 90

        if row.get("shots_on_target", 0) == 0:
            row["shots_on_target"] = row["shots"] * 0.42

        # --- Creative proxies (key_passes, through_balls, crosses, prog passes) ---
        # Assists-based estimate uses Big-5 ratio: ~6 KP per assist, ~20 prog passes.
        # When a player has real assists, derive key_passes from assists but use
        # quality-scaled baseline for progressive_passes (they aren't tied 1:1).
        # 有真实助攻时用助攻推算关键传球，无助攻时用质量因子 × 基准。
        if row.get("key_passes", 0) == 0:
            if assists_val > 0:
                row["key_passes"] = max(assists_val * 5.5, base["key_passes"] * q_atk * mins / 90)
            else:
                row["key_passes"] = base["key_passes"] * q_atk * mins / 90

        if row.get("through_balls", 0) == 0:
            row["through_balls"] = base["through_balls"] * q_atk * mins / 90

        if row.get("crosses", 0) == 0:
            row["crosses"] = base["crosses"] * q_atk * mins / 90

        if row.get("progressive_passes", 0) == 0:
            row["progressive_passes"] = base["progressive_passes"] * q_atk * mins / 90

        # --- Dribbling proxies ---
        if row.get("dribble_attempts", 0) == 0:
            row["dribble_attempts"] = base["dribble_attempts"] * q_atk * mins / 90
        if row.get("dribble_success", 0) == 0:
            row["dribble_success"] = base["dribble_success"] * q_atk * mins / 90

        # --- Carrying proxies ---
        if row.get("carries_count", 0) == 0:
            row["carries_count"] = base["carries_count"] * q_atk * mins / 90
        if row.get("progressive_carry_distance", 0) == 0:
            row["progressive_carry_distance"] = base["progressive_carry_distance"] * q_atk * mins / 90

        # --- Defensive proxies (blocks, clearances, pressures, tackle attempts) ---
        if row.get("blocks", 0) == 0:
            if intercept_val > 0:
                row["blocks"] = intercept_val * 0.6
            else:
                row["blocks"] = base["blocks"] * q_def * mins / 90

        if row.get("clearances", 0) == 0:
            if intercept_val > 0 and pos_key == "DF":
                row["clearances"] = intercept_val * 2.0
            else:
                row["clearances"] = base["clearances"] * q_def * mins / 90

        if row.get("pressures", 0) == 0:
            if tackles_val > 0:
                row["pressures"] = tackles_val * 4.0
            else:
                row["pressures"] = base["pressures"] * q_def * mins / 90

        if row.get("counter_pressures", 0) == 0:
            row["counter_pressures"] = base["counter_pressures"] * q_def * mins / 90

        if row.get("tackle_attempts", 0) == 0:
            if tackles_val > 0:
                row["tackle_attempts"] = tackles_val / 0.60
            else:
                row["tackle_attempts"] = base["tackle_attempts"] * q_def * mins / 90

        # --- GK saves ---
        if row.get("goalkeeper_saves", 0) == 0 and pos_key == "GK":
            row["goalkeeper_saves"] = base["goalkeeper_saves"] * mins / 90

        rows.append(row)
    return rows


def build_team_profile(
    team_pids: set[str],
    *,
    elo_rating: float,
    elo_rank: int,
    confederation: str = "UEFA",
    season_data: dict[str, list[dict]] | None = None,
    league_seasons: dict[str, float] | None = None,
    latest_league_season: str | None = None,
    national_orm_rows: list[PlayerStatsNational] | None = None,
    pid_to_pos: dict[str, str] | None = None,
    pid_to_club: dict[str, str] | None = None,
    prior_national_comps_count: int = 0,
    league_player_rows: list[dict] | None = None,
    squad_chemistry_same_club: bool = False,
) -> dict[str, float]:
    """
    Build one team profile dict from squad player IDs and optional stat sources.
    从名单球员 ID 及统计源构建单方队伍特征字典。
    """
    season_data = season_data or {}
    league_seasons = league_seasons or {}
    pid_to_pos = pid_to_pos or {}
    pid_to_club = pid_to_club or {}
    national_orm_rows = national_orm_rows or []

    feats = _elo_conf_features(elo_rating, elo_rank, confederation)
    feats.update(_league_features_for_squad(
        team_pids, season_data, league_seasons, latest_league_season,
    ))
    feats.update(_positional_composition_features(
        team_pids, pid_to_pos, season_data, latest_league_season,
    ))

    if league_player_rows:
        agg = aggregate_team_from_players(league_player_rows)
        feats.update(agg)
        feats["nt_competitions_count"] = 0.0
        if squad_chemistry_same_club and league_player_rows:
            n = len(league_player_rows)
            feats["club_max_cluster_ratio"] = 1.0
            feats["club_pair_density"] = 1.0 if n >= 2 else 0.0
            feats["club_distinct_clubs"] = 1.0
    else:
        feats.update(_national_aggregation_features(
            team_pids, national_orm_rows, pid_to_pos, pid_to_club,
            prior_national_comps_count,
        ))

    return feats


def load_squad_context(session: Session) -> SquadContext:
    """Load player id → position/club/name maps from DB."""
    ctx = SquadContext()
    for pid, name, pos, club in session.query(
        Player.internal_player_id,
        Player.full_name,
        Player.position,
        Player.current_club,
    ):
        ctx.pid_to_name[pid] = name
        if pos:
            ctx.pid_to_pos[pid] = pos.split(",")[0].strip()
        if club:
            ctx.pid_to_club[pid] = club.strip()
    return ctx


def resolve_roster_slots(
    roster: list[PlayerSlot],
    session: Session,
    ctx: SquadContext | None = None,
) -> tuple[list[PlayerSlot], SquadContext]:
    """
    Resolve player names to internal IDs via fuzzy matching.
    通过模糊匹配将球员姓名解析为 internal_player_id。
    """
    from utils.entity_resolution import PlayerMatcher

    ctx = ctx or load_squad_context(session)
    matcher = PlayerMatcher(ctx.pid_to_name)
    resolved: list[PlayerSlot] = []

    for slot in roster:
        if slot.player_id:
            resolved.append(slot)
            continue
        if not slot.name:
            continue
        result = matcher.match_name(slot.name, threshold=78)
        if result.is_new:
            continue
        pos = slot.position or ctx.pid_to_pos.get(result.internal_player_id, "")
        resolved.append(PlayerSlot(
            player_id=result.internal_player_id,
            name=slot.name,
            position=pos,
            expected_minutes=slot.expected_minutes,
            is_starter=slot.is_starter,
        ))
    return resolved, ctx


def roster_from_club(club_name: str, session: Session, ctx: SquadContext | None = None) -> list[PlayerSlot]:
    """
    Build roster from all players whose current_club matches (fallback when XI unknown).
    当无首发名单时，从 DB 中 current_club 匹配的球员构建名单。
    """
    ctx = ctx or load_squad_context(session)
    canonical = normalize_club_name(club_name)
    needle = canonical.lower()
    slots: list[PlayerSlot] = []
    for pid, club in ctx.pid_to_club.items():
        if needle in club.lower() or club.lower() in needle:
            slots.append(PlayerSlot(
                player_id=pid,
                name=ctx.pid_to_name.get(pid),
                position=ctx.pid_to_pos.get(pid),
                expected_minutes=90.0 if len(slots) < 11 else 30.0,
                is_starter=len(slots) < 11,
            ))
    return slots


def load_league_stats_for_pids(
    pids: set[str],
    league_seasons: dict[str, float],
    session: Session,
    latest_league_season: str | None = None,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """
    Return season_data subset and latest per-player league stat dicts.
    返回 squad 的联赛 season_data 及最近赛季的球员统计。
    """
    if not league_seasons:
        return {}, {}

    season_list = list(league_seasons.keys())
    rows = session.query(
        PlayerStatsLeague.internal_player_id,
        PlayerStatsLeague.season,
        PlayerStatsLeague.goals,
        PlayerStatsLeague.assists,
        PlayerStatsLeague.minutes_played,
        PlayerStatsLeague.xg,
        PlayerStatsLeague.interceptions,
        PlayerStatsLeague.tackles_won,
        PlayerStatsLeague.passes_completed,
    ).filter(
        PlayerStatsLeague.internal_player_id.in_(list(pids)),
        PlayerStatsLeague.season.in_(season_list),
    ).all()

    season_data: dict[str, list[dict]] = {}
    latest_by_pid: dict[str, dict] = {}
    # Track which season each pid was last seen in for fallback.
    # 记录每个球员最近出现的赛季，用于 fallback 逻辑。
    pid_best_season: dict[str, str] = {}

    season_order = sorted(league_seasons.keys())

    for pid, season, goals, assists, minutes, xg, interceptions, tackles_won, passes_completed in rows:
        if season not in season_data:
            season_data[season] = []
        season_data[season].append({
            "goals": goals or 0,
            "assists": assists or 0,
            "minutes": minutes or 0,
            "pid": pid,
        })
        stat_dict = {
            "goals": goals or 0,
            "assists": assists or 0,
            "minutes_played": minutes or 0,
            "xg": xg or 0.0,
            "interceptions": interceptions or 0,
            "tackles_won": tackles_won or 0,
            "passes_completed": passes_completed or 0,
        }
        if latest_league_season and season == latest_league_season:
            latest_by_pid[pid] = stat_dict
        else:
            prev = pid_best_season.get(pid)
            if prev is None or season_order.index(season) > season_order.index(prev):
                pid_best_season[pid] = season
                if pid not in latest_by_pid:
                    latest_by_pid[pid] = stat_dict

    return season_data, latest_by_pid


def build_club_profile(
    club_name: str,
    roster: list[PlayerSlot] | None = None,
    *,
    club_elo: float = 1500.0,
    club_elo_rank: int = 50,
    league_seasons: dict[str, float] | None = None,
    latest_league_season: str = "2024-2025",
    session: Session | None = None,
) -> dict[str, float]:
    """
    Build club profile from explicit XI or DB club roster + league stats.
    从指定首发或 DB 俱乐部名单 + 联赛统计构建俱乐部画像。
    """
    if league_seasons is None:
        league_seasons = {
            "2021-2022": 0.15,
            "2022-2023": 0.25,
            "2023-2024": 0.45,
            "2024-2025": 1.0,
        }

    # Query all seasons we might have in DB (fbref scrape may be partial).
    load_seasons = dict(league_seasons)
    for s in ("2020-2021", "2019-2020", "2018-2019"):
        load_seasons.setdefault(s, 0.1)

    engine = create_engine(DB_URL, echo=False)
    own_session = session is None
    if own_session:
        session = Session(engine)

    try:
        ctx = load_squad_context(session)
        if roster:
            slots, ctx = resolve_roster_slots(roster, session, ctx)
        else:
            slots = roster_from_club(club_name, session, ctx)

        pids = {s.player_id for s in slots if s.player_id}
        if not pids:
            return build_team_profile(
                set(),
                elo_rating=club_elo,
                elo_rank=club_elo_rank,
                league_seasons=league_seasons,
                latest_league_season=latest_league_season,
                league_player_rows=[],
                squad_chemistry_same_club=True,
            )

        season_data, latest_by_pid = load_league_stats_for_pids(
            pids, load_seasons, session, latest_league_season,
        )
        effective_latest = latest_league_season
        if not latest_by_pid:
            for fallback in (
                "2024-2025", "2023-2024", "2022-2023", "2021-2022",
                "2020-2021", "2019-2020",
            ):
                if fallback == latest_league_season:
                    continue
                season_data, latest_by_pid = load_league_stats_for_pids(
                    pids, load_seasons, session, fallback,
                )
                if latest_by_pid:
                    effective_latest = fallback
                    break
        canonical = normalize_club_name(club_name)
        league_rows = _league_rows_to_player_dicts(
            slots, latest_by_pid, ctx, same_club=canonical,
        )

        return build_team_profile(
            pids,
            elo_rating=club_elo,
            elo_rank=club_elo_rank,
            confederation="UEFA",
            season_data=season_data,
            league_seasons=league_seasons,
            latest_league_season=effective_latest,
            pid_to_pos=ctx.pid_to_pos,
            pid_to_club=ctx.pid_to_club,
            league_player_rows=league_rows,
            squad_chemistry_same_club=True,
        )
    finally:
        if own_session:
            session.close()


# Default league season weights per elo_key for auto-loading club stats.
# 各 elo_key 对应的默认联赛赛季权重，用于自动加载球员俱乐部统计。
_ELO_KEY_LEAGUE_DEFAULTS: dict[str, tuple[dict[str, float], str]] = {
    "wc2018": ({"2016-2017": 0.6, "2017-2018": 1.0}, "2017-2018"),
    "euro2020": (
        {"2018-2019": 0.5, "2019-2020": 0.75, "2020-2021": 1.0},
        "2020-2021",
    ),
    "wc2022": (
        {"2020-2021": 0.6, "2021-2022": 1.0},
        "2021-2022",
    ),
    "wc2026": (
        {"2022-2023": 0.3, "2023-2024": 0.5, "2024-2025": 0.8, "2025-2026": 1.0},
        "2025-2026",
    ),
}


def build_national_profile(
    team_name: str,
    roster: list[PlayerSlot] | None = None,
    *,
    elo_key: str = "wc2022",
    league_seasons: dict[str, float] | None = None,
    latest_league_season: str | None = None,
    prior_national_comps: list[str] | None = None,
    team_pids: set[str] | None = None,
    pid_to_pos: dict[str, str] | None = None,
    pid_to_club: dict[str, str] | None = None,
    season_data: dict[str, list[dict]] | None = None,
    national_orm_rows: list[PlayerStatsNational] | None = None,
) -> dict[str, float]:
    """
    Build national-team profile from roster or precomputed tournament maps.
    Auto-loads club league stats when league_seasons is not provided.
    从名单或已有大赛映射构建国家队画像。未指定联赛赛季时自动加载俱乐部统计。
    """
    from extractors.elo_scraper import get_pre_tournament_elo

    # Auto-populate league seasons from tournament config when caller omits them.
    # 当调用者未指定联赛赛季时，根据赛事配置自动填充。
    if league_seasons is None:
        defaults = _ELO_KEY_LEAGUE_DEFAULTS.get(elo_key)
        if defaults:
            league_seasons, latest_league_season = defaults
        else:
            league_seasons = {}
    if latest_league_season is None:
        latest_league_season = "2021-2022"
    if prior_national_comps is None:
        prior_national_comps = []

    elo_data = get_pre_tournament_elo(elo_key)
    elo_info = elo_data.get(team_name, {})
    elo_rating = float(elo_info.get("elo", 1500))
    elo_rank = int(elo_info.get("rank", len(elo_data)))
    conf = elo_info.get("confederation", "UEFA")

    pids = team_pids or set()
    engine = create_engine(DB_URL, echo=False)

    if roster and not pids:
        with Session(engine) as session:
            slots, ctx = resolve_roster_slots(roster, session)
            pids = {s.player_id for s in slots if s.player_id}
            pid_to_pos = ctx.pid_to_pos
            pid_to_club = ctx.pid_to_club

    # Auto-load league stats from DB when we have player IDs and league seasons.
    # 有球员 ID + 联赛赛季配置时，自动从 DB 加载联赛统计。
    league_player_rows = None
    if pids and league_seasons and season_data is None:
        with Session(engine) as session:
            if pid_to_pos is None or pid_to_club is None:
                ctx = load_squad_context(session)
                pid_to_pos = pid_to_pos or ctx.pid_to_pos
                pid_to_club = pid_to_club or ctx.pid_to_club

            sd, latest_by_pid = load_league_stats_for_pids(
                pids, league_seasons, session, latest_league_season,
            )
            season_data = sd

            if latest_by_pid:
                dummy_slots = [
                    PlayerSlot(
                        player_id=pid,
                        position=pid_to_pos.get(pid, ""),
                        expected_minutes=90.0,
                        is_starter=True,
                    )
                    for pid in pids
                    if pid in latest_by_pid
                ]
                league_player_rows = _league_rows_to_player_dicts(
                    dummy_slots, latest_by_pid,
                    SquadContext(
                        pid_to_pos=pid_to_pos or {},
                        pid_to_club=pid_to_club or {},
                    ),
                )

    # Auto-load international career stats when caller did not supply them.
    # 调用者未传入国家队统计时，从 DB 自动加载国际比赛 caps/goals。
    if national_orm_rows is None and pids:
        with Session(engine) as session:
            national_orm_rows = session.query(PlayerStatsNational).filter(
                PlayerStatsNational.internal_player_id.in_(list(pids)),
            ).all()

    return build_team_profile(
        pids,
        elo_rating=elo_rating,
        elo_rank=elo_rank,
        confederation=conf,
        season_data=season_data or {},
        league_seasons=league_seasons,
        latest_league_season=latest_league_season,
        national_orm_rows=national_orm_rows or [],
        pid_to_pos=pid_to_pos or {},
        pid_to_club=pid_to_club or {},
        prior_national_comps_count=len(prior_national_comps),
        league_player_rows=league_player_rows,
    )
