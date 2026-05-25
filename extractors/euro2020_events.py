"""
Euro 2020 event-level stats extractor for pre-WC national team features.
Euro 2020 事件级统计提取器，用于赛前国家队特征。

Why Euro 2020? 为什么要 Euro 2020 数据？
Euro 2020（实际 2021 年举办）覆盖 12 支同时参加 2022 世界杯的欧洲球队，
提供了比 2018 WC 更近的国家队表现数据。
非欧洲球队仍然依赖 2018 WC 数据（StatsBomb 免费数据不含其他赛前洲际赛事）。

StatsBomb competition: id=55, season_id=43.
Overlapping teams with 2022 WC:
  Belgium, Croatia, Denmark, England, France, Germany,
  Netherlands, Poland, Portugal, Spain, Switzerland, Wales

Usage / 用法:
    python -m extractors.euro2020_events
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from statsbombpy import sb

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding
fix_console_encoding()

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Player, PlayerStatsNational
from utils.entity_resolution import PlayerMatcher
from extractors.statsbomb_events import _aggregate_player_events

COMPETITION_ID = 55
SEASON_ID = 43

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"


def extract_euro2020_national_stats() -> dict[str, int]:
    """
    Extract event-level stats from Euro 2020 matches and insert into
    player_stats_national with competition='UEFA Euro 2020'.
    从 Euro 2020 比赛中提取事件级统计，写入 player_stats_national。

    Only inserts for players already in our DB (2022 WC roster).
    Skips players that already have a Euro 2020 row.
    只为数据库中已有的球员（2022 WC 名单）插入数据。
    跳过已有 Euro 2020 数据的球员。
    """
    import warnings
    warnings.filterwarnings("ignore")

    engine = create_engine(DB_URL, echo=False)

    stats = {
        "matches_processed": 0,
        "players_aggregated": 0,
        "rows_inserted": 0,
        "players_skipped_not_in_db": 0,
    }

    print("[Euro 2020] Loading match list...")
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    match_ids = matches["match_id"].tolist()
    print(f"  Found {len(match_ids)} matches.")

    player_totals: dict[str, dict] = {}

    for i, mid in enumerate(match_ids, 1):
        print(f"  Match {i}/{len(match_ids)} (id={mid})...", end=" ")
        try:
            match_stats = _aggregate_player_events(mid)
        except Exception as exc:
            print(f"FAIL: {exc}")
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
                    "matches": 0,
                }
            player_totals[name]["xg"] += ps["xg"]
            player_totals[name]["passes_completed"] += ps["passes_completed"]
            player_totals[name]["passes_attempted"] += ps["passes_attempted"]
            player_totals[name]["interceptions"] += ps["interceptions"]
            player_totals[name]["tackles_won"] += ps["tackles_won"]
            player_totals[name]["matches"] += 1

        stats["matches_processed"] += 1
        print(f"OK ({len(match_stats)})")

    stats["players_aggregated"] = len(player_totals)
    print(f"\n[Euro 2020] {len(player_totals)} unique players aggregated.")

    with Session(engine) as session:
        all_players = {
            r.full_name: r.internal_player_id
            for r in session.query(Player.full_name, Player.internal_player_id).all()
        }
        matcher = PlayerMatcher(all_players)

        # Check which players already have Euro 2020 data.
        already_done = {
            r[0] for r in session.query(
                PlayerStatsNational.internal_player_id
            ).filter(
                PlayerStatsNational.competition == "UEFA Euro 2020"
            ).all()
        }

        for sb_name, totals in player_totals.items():
            result = matcher.match_name(sb_name, threshold=80)
            if result.is_new:
                stats["players_skipped_not_in_db"] += 1
                continue

            if result.internal_player_id in already_done:
                continue

            row = PlayerStatsNational(
                internal_player_id=result.internal_player_id,
                competition="UEFA Euro 2020",
                as_of_date=date(2021, 7, 11),
                xg=round(totals["xg"], 4),
                goals=None,
                assists=None,
                caps=totals["matches"],
                minutes_played=None,
                passes_completed=totals["passes_completed"],
                passes_attempted=totals["passes_attempted"],
                interceptions=totals["interceptions"],
                tackles_won=totals["tackles_won"],
            )
            session.add(row)
            stats["rows_inserted"] += 1

        session.commit()

    print(f"\n{'='*60}")
    print("[Euro 2020] Extraction complete. Summary:")
    print(f"  Matches processed:   {stats['matches_processed']}")
    print(f"  Players aggregated:  {stats['players_aggregated']}")
    print(f"  Rows inserted:       {stats['rows_inserted']}")
    print(f"  Not in DB (skipped): {stats['players_skipped_not_in_db']}")
    print(f"{'='*60}")

    return stats


if __name__ == "__main__":
    extract_euro2020_national_stats()
