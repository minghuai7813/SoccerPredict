"""
Team-level feature engineering — tournament-agnostic version.
队伍级别特征工程——赛事通用版本。

National-team matrix builder; per-team logic lives in features.team_profile.
国家队特征矩阵构建器；单方画像逻辑在 team_profile 模块。

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
from features.team_profile import build_team_profile, load_squad_context

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

        ctx = load_squad_context(session)

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

        national_orm_rows: list[PlayerStatsNational] = []
        if prior_national_comps:
            national_orm_rows = (
                session.query(PlayerStatsNational)
                .filter(PlayerStatsNational.competition.in_(prior_national_comps))
                .all()
            )

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

    all_teams = set(pid_to_team.values()) | set(player_team_map.values())
    elo_data = get_pre_tournament_elo(elo_key)
    feature_rows = {}

    for team in sorted(all_teams):
        team_pids = {pid for pid, t in pid_to_team.items() if t == team}
        elo_info = elo_data.get(team, {})
        feature_rows[team] = build_team_profile(
            team_pids,
            elo_rating=float(elo_info.get("elo", 1500)),
            elo_rank=int(elo_info.get("rank", len(elo_data))),
            confederation=elo_info.get("confederation", "OTHER"),
            season_data=team_season_data.get(team, {}),
            league_seasons=league_seasons,
            latest_league_season=latest_league_season,
            national_orm_rows=national_orm_rows,
            pid_to_pos=ctx.pid_to_pos,
            pid_to_club=ctx.pid_to_club,
            prior_national_comps_count=len(prior_national_comps),
        )

    team_df = pd.DataFrame(feature_rows).T
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
