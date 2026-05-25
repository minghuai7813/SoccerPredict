"""
Team-level feature engineering from player statistics.
从球员统计数据聚合出国家队级别特征。

Feature design philosophy / 特征设计思路:
  - Multi-season league data: captures form, trend, consistency.
    多赛季联赛数据：捕捉状态、趋势、稳定性。
  - Recency weighting: recent seasons matter more.
    近期加权：最近的赛季权重更高。
  - National team data: 2018 WC + Euro 2020 event stats.
    国家队数据：2018 世界杯 + Euro 2020 事件级统计。
  - Positional breakdown: team composition by GK/DF/MF/FW.
    位置拆分：按守门员/后卫/中场/前锋分析球队构成。
  - Elo ratings: strongest single predictor of match outcomes.
    Elo 评分：比赛结果最强的单一预测因子。
  - Confederation: captures regional playing style and familiarity.
    洲际联盟：捕捉区域比赛风格和熟悉度。

Usage / 用法:
    from features.team_features import build_team_feature_matrix
    df = build_team_feature_matrix()
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Player, PlayerStatsLeague, PlayerStatsNational
from extractors.elo_scraper import get_pre_wc_elo

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"

# Recency weights: more recent seasons get higher weight.
# 近因权重：越近的赛季权重越高。
SEASON_WEIGHTS = {
    "2018-2019": 0.4,
    "2019-2020": 0.6,
    "2020-2021": 0.8,
    "2021-2022": 1.0,
}


def _build_player_team_map() -> dict[str, str]:
    """
    Build player_name → team_name mapping from 2022 WC StatsBomb lineups.
    从 2022 世界杯 StatsBomb 阵容数据构建球员→球队映射。
    """
    from statsbombpy import sb
    import warnings
    warnings.filterwarnings("ignore")

    matches = sb.matches(competition_id=43, season_id=106)
    player_team_map: dict[str, str] = {}

    for _, match_row in matches.iterrows():
        mid = match_row["match_id"]
        try:
            lineups = sb.lineups(match_id=mid)
        except Exception:
            continue
        for team_name, roster_df in lineups.items():
            for _, p in roster_df.iterrows():
                pname = p.get("player_name", "")
                if pname and pname not in player_team_map:
                    player_team_map[pname] = team_name

    return player_team_map


def _resolve_pid_to_team(
    player_team_map: dict[str, str],
    players: list[tuple[str, str]],
) -> dict[str, str]:
    """
    Match internal player IDs to team names via fuzzy matching.
    通过模糊匹配将数据库球员 ID 映射到球队名。
    """
    from utils.entity_resolution import PlayerMatcher

    matcher = PlayerMatcher({name: pid for pid, name in players})
    pid_to_team: dict[str, str] = {}

    for sb_name, team in player_team_map.items():
        result = matcher.match_name(sb_name, threshold=80)
        if not result.is_new:
            pid_to_team[result.internal_player_id] = team

    return pid_to_team


def _compute_trend(values: list[float]) -> float:
    """
    Simple linear slope across seasons (positive = improving).
    简单线性斜率（正值 = 进步中）。
    Returns 0 if fewer than 2 data points.
    """
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    if np.std(y) == 0:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def build_team_feature_matrix() -> pd.DataFrame:
    """
    Build a comprehensive team-level feature matrix for all 2022 WC teams.
    为所有 2022 世界杯参赛队构建综合队级特征矩阵。

    Feature groups / 特征组:
      1. League stats (recency-weighted aggregate across 4 seasons)
      2. League trend features (slope across seasons)
      3. National team stats (2018 WC + Euro 2020)
      4. Positional composition (GK/DF/MF/FW distribution)
      5. Elo rating + rank
      6. Confederation encoding
    """
    engine = create_engine(DB_URL, echo=False)

    print("[Features] Building player -> team mapping from StatsBomb...")
    player_team_map = _build_player_team_map()
    print(f"  Mapped {len(player_team_map)} players to teams.")

    with Session(engine) as session:
        players = session.query(
            Player.internal_player_id,
            Player.full_name,
        ).all()

        pid_to_team = _resolve_pid_to_team(player_team_map, players)
        print(f"  Resolved {len(pid_to_team)} players to national teams.")

        # Load player positions.
        pid_to_pos: dict[str, str] = {}
        for pid, pos in session.query(Player.internal_player_id, Player.position).all():
            if pos:
                primary = pos.split(",")[0].strip()
                pid_to_pos[pid] = primary

        # --- League stats (multi-season) ---
        league_rows = session.query(
            PlayerStatsLeague.internal_player_id,
            PlayerStatsLeague.season,
            PlayerStatsLeague.goals,
            PlayerStatsLeague.assists,
            PlayerStatsLeague.minutes_played,
        ).all()

        # Organize: team → season → list of player stats.
        team_season_data: dict[str, dict[str, list[dict]]] = {}
        for pid, season, goals, assists, minutes in league_rows:
            team = pid_to_team.get(pid)
            if not team or not season:
                continue
            if team not in team_season_data:
                team_season_data[team] = {}
            if season not in team_season_data[team]:
                team_season_data[team][season] = []
            team_season_data[team][season].append({
                "goals": goals or 0,
                "assists": assists or 0,
                "minutes": minutes or 0,
                "pid": pid,
            })

        # --- National stats (all competitions: WC 2018 + Euro 2020) ---
        national_rows = session.query(
            PlayerStatsNational.internal_player_id,
            PlayerStatsNational.competition,
            PlayerStatsNational.xg,
            PlayerStatsNational.passes_completed,
            PlayerStatsNational.passes_attempted,
            PlayerStatsNational.interceptions,
            PlayerStatsNational.tackles_won,
            PlayerStatsNational.caps,
        ).all()

    # --- Build features per team ---
    all_teams = set(pid_to_team.values())
    elo_data = get_pre_wc_elo()

    feature_rows = {}

    for team in sorted(all_teams):
        feats: dict[str, float] = {}

        # ====== GROUP 1: League stats (recency-weighted) ======
        season_data = team_season_data.get(team, {})
        weighted_goals = 0.0
        weighted_assists = 0.0
        weighted_minutes = 0.0
        total_weight = 0.0
        player_count_latest = 0

        season_goal_avgs = []

        for season in sorted(SEASON_WEIGHTS.keys()):
            w = SEASON_WEIGHTS[season]
            players_in_season = season_data.get(season, [])
            if not players_in_season:
                continue

            s_goals = sum(p["goals"] for p in players_in_season)
            s_assists = sum(p["assists"] for p in players_in_season)
            s_minutes = sum(p["minutes"] for p in players_in_season)
            n = len(players_in_season)

            weighted_goals += s_goals * w
            weighted_assists += s_assists * w
            weighted_minutes += s_minutes * w
            total_weight += w

            season_goal_avgs.append(s_goals / n if n > 0 else 0)

            if season == "2021-2022":
                player_count_latest = n

        if total_weight > 0:
            feats["lg_goals_weighted"] = weighted_goals / total_weight
            feats["lg_assists_weighted"] = weighted_assists / total_weight
            feats["lg_minutes_weighted"] = weighted_minutes / total_weight
            feats["lg_goal_involvement_weighted"] = (weighted_goals + weighted_assists) / total_weight
        else:
            feats["lg_goals_weighted"] = 0
            feats["lg_assists_weighted"] = 0
            feats["lg_minutes_weighted"] = 0
            feats["lg_goal_involvement_weighted"] = 0

        feats["lg_player_count"] = player_count_latest
        feats["lg_seasons_available"] = len(season_data)

        # Per-90 efficiency from latest season.
        latest = season_data.get("2021-2022", [])
        total_goals_latest = sum(p["goals"] for p in latest) if latest else 0
        total_min_latest = sum(p["minutes"] for p in latest) if latest else 0
        feats["lg_goals_per90"] = (total_goals_latest / total_min_latest * 90) if total_min_latest > 0 else 0

        # Max individual scorer across all seasons.
        all_player_goals = [p["goals"] for s in season_data.values() for p in s]
        feats["lg_goals_max"] = max(all_player_goals) if all_player_goals else 0

        # ====== GROUP 2: League trend features ======
        feats["lg_goal_trend"] = _compute_trend(season_goal_avgs)

        # Consistency: coefficient of variation across seasons.
        if len(season_goal_avgs) >= 2:
            mean_g = np.mean(season_goal_avgs)
            feats["lg_consistency"] = np.std(season_goal_avgs) / mean_g if mean_g > 0 else 0
        else:
            feats["lg_consistency"] = 0

        # ====== GROUP 3: National team stats ======
        team_pids = {pid for pid, t in pid_to_team.items() if t == team}
        nt_stats = [
            r for r in national_rows if r[0] in team_pids
        ]

        if nt_stats:
            nt_xg = sum((r[2] or 0) for r in nt_stats)
            nt_pc = sum((r[3] or 0) for r in nt_stats)
            nt_pa = sum((r[4] or 0) for r in nt_stats)
            nt_ints = sum((r[5] or 0) for r in nt_stats)
            nt_tkl = sum((r[6] or 0) for r in nt_stats)
            nt_caps = [r[7] or 0 for r in nt_stats]
            n_nt = len(nt_stats)

            feats["nt_xg_sum"] = nt_xg
            feats["nt_xg_mean"] = nt_xg / n_nt
            feats["nt_passes_completed_sum"] = nt_pc
            feats["nt_pass_completion_rate"] = nt_pc / nt_pa if nt_pa > 0 else 0
            feats["nt_interceptions_mean"] = nt_ints / n_nt
            feats["nt_tackles_mean"] = nt_tkl / n_nt
            feats["nt_defensive_actions_mean"] = (nt_ints + nt_tkl) / n_nt
            feats["nt_caps_mean"] = np.mean(nt_caps)
            feats["nt_caps_max"] = max(nt_caps)
            feats["nt_player_count"] = n_nt

            # How many competitions does this team have data for?
            competitions = set(r[1] for r in nt_stats if r[1])
            feats["nt_competitions_count"] = len(competitions)
        else:
            for k in ["nt_xg_sum", "nt_xg_mean", "nt_passes_completed_sum",
                       "nt_pass_completion_rate", "nt_interceptions_mean",
                       "nt_tackles_mean", "nt_defensive_actions_mean",
                       "nt_caps_mean", "nt_caps_max", "nt_player_count",
                       "nt_competitions_count"]:
                feats[k] = 0

        # ====== GROUP 4: Positional composition ======
        pos_counts = {"GK": 0, "DF": 0, "MF": 0, "FW": 0}
        pos_goals = {"DF": 0, "MF": 0, "FW": 0}
        pos_minutes = {"DF": 0, "MF": 0, "FW": 0}

        for pid in team_pids:
            pos = pid_to_pos.get(pid, "")
            if pos in pos_counts:
                pos_counts[pos] += 1
            # Attribute latest-season goals by position.
            if pos in pos_goals:
                for p in latest:
                    if p["pid"] == pid:
                        pos_goals[pos] += p["goals"]
                        pos_minutes[pos] += p["minutes"]

        total_outfield = pos_counts["DF"] + pos_counts["MF"] + pos_counts["FW"]
        feats["pos_fw_ratio"] = pos_counts["FW"] / total_outfield if total_outfield > 0 else 0
        feats["pos_mf_ratio"] = pos_counts["MF"] / total_outfield if total_outfield > 0 else 0
        feats["pos_df_ratio"] = pos_counts["DF"] / total_outfield if total_outfield > 0 else 0
        feats["pos_fw_goals"] = pos_goals["FW"]
        feats["pos_mf_goals"] = pos_goals["MF"]

        # ====== GROUP 5: Elo rating ======
        elo_info = elo_data.get(team, {})
        feats["elo_rating"] = elo_info.get("elo", 1500)
        feats["elo_rank"] = elo_info.get("rank", 32)

        # ====== GROUP 6: Confederation encoding ======
        conf = elo_info.get("confederation", "OTHER")
        for c in ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC"]:
            feats[f"conf_{c}"] = 1.0 if conf == c else 0.0

        feature_rows[team] = feats

    team_df = pd.DataFrame(feature_rows).T
    team_df = team_df.fillna(0)

    print(f"\n[Features] Built feature matrix: {team_df.shape[0]} teams x {team_df.shape[1]} features.")
    feature_groups = {
        "League (weighted)": [c for c in team_df.columns if c.startswith("lg_")],
        "National team": [c for c in team_df.columns if c.startswith("nt_")],
        "Position": [c for c in team_df.columns if c.startswith("pos_")],
        "Elo": [c for c in team_df.columns if c.startswith("elo_")],
        "Confederation": [c for c in team_df.columns if c.startswith("conf_")],
    }
    for group, cols in feature_groups.items():
        print(f"  {group}: {len(cols)} features")

    return team_df


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    df = build_team_feature_matrix()
    print("\n" + "=" * 70)
    print(df.sort_values("elo_rating", ascending=False).to_string())
