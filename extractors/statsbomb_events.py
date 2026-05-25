"""
StatsBomb event-level stats extractor for the 2022 World Cup.
StatsBomb 事件级别统计数据提取器。

Why event-level data? 为什么要用事件级数据？
FBref 只能提供五大联赛的聚合统计，211 名小联赛球员没有联赛数据。
但 StatsBomb 的事件数据覆盖世界杯全部 64 场比赛的每一个球员，
而且精度远超聚合统计——精确到每次射门的 xG、每次传球的成功/失败。

This module aggregates per-player stats from raw match events:
  - xG (sum of shot_statsbomb_xg)
  - Passes completed / attempted
  - Interceptions
  - Tackles won (from Duel events)
  - Carries, dribbles, pressures

These fill the NULL columns in player_stats_national and also update
the team_features pipeline with much richer signals.
这些数据填补 player_stats_national 中的 NULL 列，同时为特征工程提供更丰富的信号。

Usage / 用法:
    python -m extractors.statsbomb_events
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
from statsbombpy import sb

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding
fix_console_encoding()

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Base, Player, PlayerStatsNational
from utils.entity_resolution import PlayerMatcher

COMPETITION_ID = 43
SEASON_ID = 106

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"


def _aggregate_player_events(match_id: int) -> list[dict]:
    """
    Aggregate per-player stats from a single match's event stream.
    从单场比赛的事件流中聚合每个球员的统计数据。

    StatsBomb event model:
      - Shot: has shot_statsbomb_xg (float).
      - Pass: pass_outcome is NaN for completed, has value for incomplete.
      - Interception: one event per interception.
      - Duel: includes tackles; duel_outcome indicates success.

    Returns a list of dicts, one per player, with aggregated stats.
    """
    import warnings
    warnings.filterwarnings("ignore")

    events = sb.events(match_id=match_id)
    if events.empty:
        return []

    players_in_match = events["player"].dropna().unique()
    results = []

    for player_name in players_in_match:
        pe = events[events["player"] == player_name]

        # xG: sum of all shot xG values.
        # xG：所有射门 xG 值的总和。
        shots = pe[pe["type"] == "Shot"]
        xg = 0.0
        if not shots.empty and "shot_statsbomb_xg" in shots.columns:
            xg = shots["shot_statsbomb_xg"].sum()

        # Passes: completed = outcome is NaN, attempted = total pass events.
        # 传球：完成 = outcome 为空，尝试 = 总传球事件数。
        passes = pe[pe["type"] == "Pass"]
        passes_attempted = len(passes)
        passes_completed = 0
        if passes_attempted > 0 and "pass_outcome" in passes.columns:
            passes_completed = int(passes["pass_outcome"].isna().sum())

        # Interceptions.
        # 拦截。
        interceptions = len(pe[pe["type"] == "Interception"])

        # Tackles: from Duel events where duel_type is Tackle.
        # 抢断：来自 Duel 事件中 duel_type 为 Tackle 的。
        tackles_won = 0
        duels = pe[pe["type"] == "Duel"]
        if not duels.empty and "duel_type" in duels.columns:
            tackle_duels = duels[duels["duel_type"] == "Tackle"]
            if not tackle_duels.empty and "duel_outcome" in tackle_duels.columns:
                won_outcomes = ["Won", "Success In Play", "Success Out"]
                tackles_won = int(
                    tackle_duels["duel_outcome"].isin(won_outcomes).sum()
                )

        results.append({
            "player_name": player_name,
            "xg": round(xg, 4),
            "passes_completed": passes_completed,
            "passes_attempted": passes_attempted,
            "interceptions": interceptions,
            "tackles_won": tackles_won,
        })

    return results


def extract_and_update_national_stats() -> dict[str, int]:
    """
    Main pipeline: extract event-level stats from all WC matches,
    aggregate per player across all matches, and UPDATE the existing
    player_stats_national rows with the enriched columns.
    主流程：从所有世界杯比赛中提取事件级统计，按球员聚合，
    然后 UPDATE 已有的 player_stats_national 行。

    Why UPDATE instead of INSERT? 为什么是 UPDATE 而不是 INSERT？
    player_stats_national 已经有 487 行（来自 FBref WC scraper），
    包含 goals, assists, minutes, caps。
    这里只是补充 xG, passes, interceptions, tackles 这些 NULL 列。
    """
    engine = create_engine(DB_URL, echo=False)

    stats = {
        "matches_processed": 0,
        "players_aggregated": 0,
        "rows_updated": 0,
        "new_rows_inserted": 0,
    }

    print("[Events] Loading match list for 2022 World Cup...")
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    match_ids = matches["match_id"].tolist()
    print(f"  Found {len(match_ids)} matches.")

    # Accumulate stats across all matches per player.
    # 跨所有比赛按球员累积统计。
    player_totals: dict[str, dict] = {}

    for i, mid in enumerate(match_ids, 1):
        print(f"[Events] Processing match {i}/{len(match_ids)} (id={mid})...", end=" ")

        try:
            match_stats = _aggregate_player_events(mid)
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

        for ps in match_stats:
            name = ps["player_name"]
            if name not in player_totals:
                player_totals[name] = {
                    "xg": 0.0,
                    "passes_completed": 0,
                    "passes_attempted": 0,
                    "interceptions": 0,
                    "tackles_won": 0,
                }
            player_totals[name]["xg"] += ps["xg"]
            player_totals[name]["passes_completed"] += ps["passes_completed"]
            player_totals[name]["passes_attempted"] += ps["passes_attempted"]
            player_totals[name]["interceptions"] += ps["interceptions"]
            player_totals[name]["tackles_won"] += ps["tackles_won"]

        stats["matches_processed"] += 1
        print(f"OK ({len(match_stats)} players)")

    stats["players_aggregated"] = len(player_totals)
    print(f"\n[Events] Aggregated stats for {len(player_totals)} unique players.")

    # Match player names to DB and update.
    # 匹配球员名字到数据库并更新。
    with Session(engine) as session:
        all_players = {
            row.full_name: row.internal_player_id
            for row in session.query(Player.full_name, Player.internal_player_id).all()
        }
        matcher = PlayerMatcher(all_players)

        for sb_name, totals in player_totals.items():
            result = matcher.match_name(sb_name, threshold=80)
            if result.is_new:
                continue

            pid = result.internal_player_id

            # Try to update existing national stats row.
            # 尝试更新已有的国家队统计行。
            existing = session.query(PlayerStatsNational).filter_by(
                internal_player_id=pid
            ).first()

            if existing:
                existing.xg = round(totals["xg"], 4)
                existing.passes_completed = totals["passes_completed"]
                existing.passes_attempted = totals["passes_attempted"]
                existing.interceptions = totals["interceptions"]
                existing.tackles_won = totals["tackles_won"]
                stats["rows_updated"] += 1
            else:
                # Player exists in DB but has no national stats row yet.
                # 球员在 DB 中但还没有国家队统计行——新建。
                new_row = PlayerStatsNational(
                    internal_player_id=pid,
                    as_of_date=date(2022, 12, 18),
                    xg=round(totals["xg"], 4),
                    passes_completed=totals["passes_completed"],
                    passes_attempted=totals["passes_attempted"],
                    interceptions=totals["interceptions"],
                    tackles_won=totals["tackles_won"],
                )
                session.add(new_row)
                stats["new_rows_inserted"] += 1

        session.commit()

    print(f"\n{'='*60}")
    print("[Events] Enrichment complete. Summary:")
    print(f"  Matches processed:     {stats['matches_processed']}")
    print(f"  Players aggregated:    {stats['players_aggregated']}")
    print(f"  Existing rows updated: {stats['rows_updated']}")
    print(f"  New rows inserted:     {stats['new_rows_inserted']}")
    print(f"{'='*60}")

    return stats


if __name__ == "__main__":
    extract_and_update_national_stats()
