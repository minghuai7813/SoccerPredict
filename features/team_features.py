"""
Team-level feature engineering — tournament-agnostic version.
队伍级别特征工程——赛事通用版本。

Supports building features for any tournament (2018 WC, Euro 2020, 2022 WC)
using available data. Missing data is left as NaN for XGBoost compatibility.
支持为任意赛事构建特征，缺失数据保留 NaN 供 XGBoost 处理。

Usage / 用法:
    from features.team_features import build_team_feature_matrix
    df = build_team_feature_matrix(competition_id=43, season_id=106,
                                    elo_key="wc2022")
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
from extractors.elo_scraper import get_pre_tournament_elo
from features.player_aggregation import NATIONAL_FLOAT_FIELDS, aggregate_team_from_players

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"


def _build_player_team_map(competition_id: int, season_id: int) -> dict[str, str]:
    """
    Build player_name -> team_name mapping from StatsBomb lineups.
    从 StatsBomb 阵容数据构建球员→球队映射。
    """
    from statsbombpy import sb
    import warnings
    warnings.filterwarnings("ignore")

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
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
    """Simple linear slope across seasons (positive = improving)."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.array(values, dtype=float)
    if np.std(y) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def build_team_feature_matrix(
    competition_id: int = 43,
    season_id: int = 106,
    elo_key: str = "wc2022",
    league_seasons: dict[str, float] | None = None,
    latest_league_season: str | None = "2021-2022",
    prior_national_comps: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build team-level feature matrix for a given tournament.
    为指定赛事构建队伍级别特征矩阵。

    Parameters
    ----------
    competition_id, season_id : int
        StatsBomb competition/season identifiers.
    elo_key : str
        Key for pre-tournament Elo data ("wc2018", "euro2020", "wc2022").
    league_seasons : dict
        Season string -> recency weight. None means no league data.
    latest_league_season : str or None
        Which season to use for per-90 and positional stats.
    prior_national_comps : list
        Competition names to use for national team features.
    """
    if league_seasons is None:
        league_seasons = {}
    if prior_national_comps is None:
        prior_national_comps = []

    engine = create_engine(DB_URL, echo=False)

    print(f"[Features] Building player->team map (comp={competition_id}, season={season_id})...")
    player_team_map = _build_player_team_map(competition_id, season_id)
    print(f"  Mapped {len(player_team_map)} players to teams.")

    with Session(engine) as session:
        players = session.query(
            Player.internal_player_id, Player.full_name,
        ).all()

        pid_to_team = _resolve_pid_to_team(player_team_map, players)
        print(f"  Resolved {len(pid_to_team)} players to DB records.")

        pid_to_pos: dict[str, str] = {}
        pid_to_club: dict[str, str] = {}
        for pid, pos, club in session.query(
            Player.internal_player_id, Player.position, Player.current_club,
        ).all():
            if pos:
                pid_to_pos[pid] = pos.split(",")[0].strip()
            if club:
                pid_to_club[pid] = club.strip()

        # League stats from DB (only if we have relevant seasons).
        league_rows = []
        if league_seasons:
            season_list = list(league_seasons.keys())
            league_rows = session.query(
                PlayerStatsLeague.internal_player_id,
                PlayerStatsLeague.season,
                PlayerStatsLeague.goals,
                PlayerStatsLeague.assists,
                PlayerStatsLeague.minutes_played,
            ).filter(
                PlayerStatsLeague.season.in_(season_list),
            ).all()

        # National team stats (prior competitions only — no leakage).
        # 国家队统计（仅赛前已有赛事，避免泄漏）。
        national_orm_rows: list[PlayerStatsNational] = []
        if prior_national_comps:
            national_orm_rows = (
                session.query(PlayerStatsNational)
                .filter(PlayerStatsNational.competition.in_(prior_national_comps))
                .all()
            )

    # Organize league data: team -> season -> player stats.
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
            "goals": goals or 0, "assists": assists or 0,
            "minutes": minutes or 0, "pid": pid,
        })

    # Build features per team.
    all_teams = set(pid_to_team.values()) | set(player_team_map.values())
    elo_data = get_pre_tournament_elo(elo_key)
    feature_rows = {}

    for team in sorted(all_teams):
        feats: dict[str, float] = {}

        # ====== Elo + Confederation (always available) ======
        elo_info = elo_data.get(team, {})
        feats["elo_rating"] = elo_info.get("elo", 1500)
        feats["elo_rank"] = elo_info.get("rank", len(elo_data))

        conf = elo_info.get("confederation", "OTHER")
        for c in ["UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC"]:
            feats[f"conf_{c}"] = 1.0 if conf == c else 0.0

        # ====== League stats (recency-weighted) ======
        season_data = team_season_data.get(team, {})

        if league_seasons and season_data:
            weighted_goals = 0.0
            weighted_assists = 0.0
            weighted_minutes = 0.0
            total_weight = 0.0
            season_goal_avgs = []

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
                feats["lg_goals_weighted"] = np.nan
                feats["lg_assists_weighted"] = np.nan
                feats["lg_minutes_weighted"] = np.nan
                feats["lg_goal_involvement_weighted"] = np.nan

            team_pids_in_db = {pid for pid, t in pid_to_team.items() if t == team}
            latest = season_data.get(latest_league_season, []) if latest_league_season else []
            feats["lg_player_count"] = len(latest) if latest else np.nan
            feats["lg_seasons_available"] = len(season_data)

            total_g = sum(p["goals"] for p in latest) if latest else 0
            total_m = sum(p["minutes"] for p in latest) if latest else 0
            feats["lg_goals_per90"] = (total_g / total_m * 90) if total_m > 0 else np.nan

            all_pg = [p["goals"] for s in season_data.values() for p in s]
            feats["lg_goals_max"] = max(all_pg) if all_pg else np.nan

            feats["lg_goal_trend"] = _compute_trend(season_goal_avgs)
            if len(season_goal_avgs) >= 2:
                mg = np.mean(season_goal_avgs)
                feats["lg_consistency"] = np.std(season_goal_avgs) / mg if mg > 0 else 0
            else:
                feats["lg_consistency"] = np.nan
        else:
            for k in ["lg_goals_weighted", "lg_assists_weighted", "lg_minutes_weighted",
                       "lg_goal_involvement_weighted", "lg_player_count",
                       "lg_seasons_available", "lg_goals_per90", "lg_goals_max",
                       "lg_goal_trend", "lg_consistency"]:
                feats[k] = np.nan

        # ====== National team stats (player-level aggregation) ======
        team_pids = {pid for pid, t in pid_to_team.items() if t == team}
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

        if player_combined:
            feats.update(aggregate_team_from_players(list(player_combined.values())))
            feats["nt_competitions_count"] = len(prior_national_comps)
        else:
            feats["nt_competitions_count"] = 0

        # ====== Positional composition ======
        if latest_league_season and season_data.get(latest_league_season):
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
            feats["pos_fw_goals"] = pos_goals["FW"]
            feats["pos_mf_goals"] = pos_goals["MF"]
        else:
            for k in ["pos_fw_ratio", "pos_mf_ratio", "pos_df_ratio",
                       "pos_fw_goals", "pos_mf_goals"]:
                feats[k] = np.nan

        feature_rows[team] = feats

    team_df = pd.DataFrame(feature_rows).T

    # Count how many features have data vs NaN.
    n_features = team_df.shape[1]
    n_available = team_df.notna().sum(axis=1).mean()
    print(f"\n[Features] Built: {team_df.shape[0]} teams x {n_features} features.")
    print(f"  Avg features with data: {n_available:.0f}/{n_features}")

    return team_df


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    df = build_team_feature_matrix()
    print("\n" + "=" * 70)
    print(df.sort_values("elo_rating", ascending=False).head(10).to_string())
