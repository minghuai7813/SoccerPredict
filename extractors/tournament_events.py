"""
Generic StatsBomb tournament event extractor.
通用 StatsBomb 赛事事件提取器。

Aggregates expanded per-player metrics + lineup minutes into player_stats_national.
将扩展球员指标与 lineup 出场时间写入 player_stats_national。
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

from statsbombpy import sb

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session

from db.models import Player, PlayerStatsNational
from extractors.player_event_metrics import (
    METRIC_KEYS,
    empty_metrics,
    merge_metrics,
    _aggregate_player_events,
)
from features.lineup_minutes import aggregate_lineup_minutes_for_match
from utils.entity_resolution import PlayerMatcher

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"


def _apply_metrics(row: PlayerStatsNational, totals: dict[str, Any]) -> None:
    """Copy metric dict onto ORM row / 将指标写入 ORM 行。"""
    for k in METRIC_KEYS:
        setattr(row, k, totals[k])


def extract_tournament_national_stats(
    competition_id: int,
    season_id: int,
    competition_label: str,
    as_of_date: date,
    *,
    skip_if_competition_exists: bool = False,
) -> dict[str, int]:
    """
  Extract and upsert national stats for one tournament.
  提取并写入单届赛事的国家队统计。
    """
    import warnings
    warnings.filterwarnings("ignore")

    engine = create_engine(DB_URL, echo=False)
    stats = {
        "matches_processed": 0,
        "players_aggregated": 0,
        "rows_updated": 0,
        "rows_inserted": 0,
        "skipped_existing": 0,
    }

    print(f"\n[Tournament] {competition_label} (comp={competition_id}, season={season_id})")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].tolist()
    print(f"  {len(match_ids)} matches")

    player_totals: dict[str, dict[str, Any]] = {}
    player_minutes: dict[str, int] = {}
    player_starts: dict[str, int] = {}
    player_apps: dict[str, int] = {}
    team_match_counts: dict[str, int] = {}

    for i, mid in enumerate(match_ids, 1):
        print(f"  Match {i}/{len(match_ids)} id={mid}...", end=" ")
        try:
            match_stats = _aggregate_player_events(mid, sb)
            lineup_mins = aggregate_lineup_minutes_for_match(mid, sb)
        except Exception as exc:
            print(f"FAIL: {exc}")
            continue

        for ps in match_stats:
            name = ps["player_name"]
            if name not in player_totals:
                player_totals[name] = empty_metrics()
                player_totals[name]["matches"] = 0
            merge_metrics(player_totals[name], ps)
            player_totals[name]["matches"] += 1

        for pname, info in lineup_mins.items():
            player_minutes[pname] = player_minutes.get(pname, 0) + info["minutes"]
            player_starts[pname] = player_starts.get(pname, 0) + info["starts"]
            player_apps[pname] = player_apps.get(pname, 0) + 1
            team = info.get("team")
            if team:
                team_match_counts[team] = team_match_counts.get(team, 0) + 1

        stats["matches_processed"] += 1
        print(f"OK ({len(match_stats)} players)")

    stats["players_aggregated"] = len(player_totals)

    # is_starter: started >= 50% of team matches they appeared for
    # 首发：在所属球队比赛中首发比例 >= 50%
    starter_flags: dict[str, int] = {}
    for pname, starts in player_starts.items():
        apps = player_apps.get(pname, 1)
        starter_flags[pname] = 1 if starts / max(apps, 1) >= 0.5 else 0

    with Session(engine) as session:
        all_players = {
            r.full_name: r.internal_player_id
            for r in session.query(Player.full_name, Player.internal_player_id).all()
        }
        matcher = PlayerMatcher(all_players)

        existing_pids = set()
        if skip_if_competition_exists:
            existing_pids = {
                r[0]
                for r in session.query(PlayerStatsNational.internal_player_id)
                .filter(PlayerStatsNational.competition == competition_label)
                .all()
            }

        for sb_name, totals in player_totals.items():
            result = matcher.match_name(sb_name, threshold=80)
            if result.is_new:
                continue

            pid = result.internal_player_id
            if pid in existing_pids:
                stats["skipped_existing"] += 1
                continue

            row = (
                session.query(PlayerStatsNational)
                .filter_by(internal_player_id=pid, competition=competition_label)
                .first()
            )
            if row is None and competition_label == "FIFA World Cup 2022":
                row = (
                    session.query(PlayerStatsNational)
                    .filter_by(internal_player_id=pid)
                    .filter(
                        or_(
                            PlayerStatsNational.competition == competition_label,
                            PlayerStatsNational.competition.is_(None),
                        )
                    )
                    .first()
                )

            mins = int(player_minutes.get(sb_name, 0))
            caps = int(totals.get("matches", 0))
            starter = starter_flags.get(sb_name, 0)

            if row:
                _apply_metrics(row, totals)
                row.competition = competition_label
                row.as_of_date = as_of_date
                row.minutes_played = mins or row.minutes_played
                row.caps = caps
                row.is_starter = starter
                stats["rows_updated"] += 1
            else:
                new_row = PlayerStatsNational(
                    internal_player_id=pid,
                    competition=competition_label,
                    as_of_date=as_of_date,
                    minutes_played=mins,
                    caps=caps,
                    is_starter=starter,
                )
                _apply_metrics(new_row, totals)
                session.add(new_row)
                stats["rows_inserted"] += 1

        session.commit()

    print(f"  Updated: {stats['rows_updated']}, Inserted: {stats['rows_inserted']}")
    return stats
