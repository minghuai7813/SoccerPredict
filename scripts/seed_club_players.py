"""
Seed club squads from FBref league player lists into players table.
从 FBref 联赛球员列表向 players 表导入俱乐部球员。

Run before fbref_scraper for new seasons so league stats can attach.
在新赛季跑 fbref_scraper 前先执行，确保新球员能匹配入库。

Usage / 用法:
    python scripts/seed_club_players.py
    python scripts/seed_club_players.py --seasons 2024-2025 2025-2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding, to_ascii_name
fix_console_encoding()

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Base, Player
from extractors.fbref_scraper import _flatten_columns, _get_col, _rate_limit
from features.team_profile import CLUB_ALIASES
from utils.entity_resolution import PlayerMatcher

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"

SEED_LEAGUES = [
    "ENG-Premier League",
    "FRA-Ligue 1",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
]

# Most recent FBref season first in list; later seasons overwrite current_club.
DEFAULT_SEASONS = ["2024-2025", "2025-2026"]


def _normalize_fbref_club(raw: str) -> str:
    name = str(raw).lstrip("0123456789. ").strip()
    key = name.lower()
    return CLUB_ALIASES.get(key, name)


def seed_from_fbref(
    season: str,
    leagues: list[str] | None = None,
    limit: int | None = None,
    *,
    refresh_club: bool = True,
    matcher: PlayerMatcher | None = None,
    session: Session | None = None,
) -> dict[str, int]:
    """
    Pull one season of FBref standard stats; insert unknown players + club.
    拉取 FBref 标准统计；为未知球员创建记录并回填 current_club。
    """
    import soccerdata as sd

    leagues = leagues or SEED_LEAGUES
    stats = {
        "season": season,
        "leagues": 0,
        "rows_seen": 0,
        "players_new": 0,
        "players_matched": 0,
        "clubs_updated": 0,
        "positions_updated": 0,
    }

    own_session = session is None
    engine = create_engine(DB_URL, echo=False)
    if own_session:
        Base.metadata.create_all(engine)
        session = Session(engine)

    assert session is not None

    if matcher is None:
        existing = {
            name: pid
            for pid, name in session.query(
                Player.internal_player_id, Player.full_name,
            ).all()
        }
        matcher = PlayerMatcher(existing)

    new_count = 0
    try:
        for league in leagues:
            print(f"\n[Seed] {league} {season} ...")
            try:
                fb = sd.FBref(leagues=league, seasons=season)
                df = fb.read_player_season_stats(stat_type="standard")
            except Exception as exc:
                print(f"  [skip] {exc}")
                continue

            if df is None or df.empty:
                print("  [skip] empty dataframe")
                continue

            df = _flatten_columns(df.reset_index())
            stats["leagues"] += 1
            league_new = 0

            for _, row in df.iterrows():
                stats["rows_seen"] += 1
                player_name = str(row.get("player", row.get("Player", ""))).strip()
                if not player_name or player_name == "nan":
                    continue

                club_raw = _get_col(row, "Club", "club", "team")
                club_name = _normalize_fbref_club(club_raw) if club_raw else None
                pos_raw = _get_col(row, "Pos", "pos")
                pos_str = str(pos_raw).strip() if pos_raw and str(pos_raw) != "nan" else None

                result = matcher.match_name(player_name, threshold=78)
                if result.is_new:
                    player = Player(
                        internal_player_id=result.internal_player_id,
                        full_name=player_name,
                        full_name_ascii=to_ascii_name(player_name),
                        position=pos_str,
                        current_club=club_name,
                    )
                    session.add(player)
                    stats["players_new"] += 1
                    new_count += 1
                    league_new += 1
                else:
                    stats["players_matched"] += 1
                    obj = session.get(Player, result.internal_player_id)
                    if obj:
                        if club_name and refresh_club and obj.current_club != club_name:
                            obj.current_club = club_name
                            stats["clubs_updated"] += 1
                        elif club_name and not obj.current_club:
                            obj.current_club = club_name
                            stats["clubs_updated"] += 1
                        if pos_str and not obj.position:
                            obj.position = pos_str
                            stats["positions_updated"] += 1

                if limit and new_count >= limit:
                    session.commit()
                    print(f"[Seed] --limit {limit} reached.")
                    return stats

            session.commit()
            print(f"  -> {league_new} new players this league")
            _rate_limit()

    finally:
        if own_session:
            session.close()

    return stats


def seed_all_seasons(
    seasons: list[str],
    leagues: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Seed multiple seasons; last season wins for current_club refresh."""
    totals = {
        "seasons": 0,
        "leagues": 0,
        "rows_seen": 0,
        "players_new": 0,
        "players_matched": 0,
        "clubs_updated": 0,
        "positions_updated": 0,
    }

    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        existing = {
            name: pid
            for pid, name in session.query(
                Player.internal_player_id, Player.full_name,
            ).all()
        }
        matcher = PlayerMatcher(existing)

        for season in seasons:
            print(f"\n{'#' * 60}\n# SEASON {season}\n{'#' * 60}")
            part = seed_from_fbref(
                season,
                leagues=leagues,
                limit=limit,
                refresh_club=True,
                matcher=matcher,
                session=session,
            )
            totals["seasons"] += 1
            for k in totals:
                if k != "seasons" and k in part:
                    totals[k] += part[k]
            if limit and totals["players_new"] >= limit:
                break

    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed club players from FBref (Big-5)")
    parser.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help="FBref seasons, e.g. 2024-2025 2025-2026",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--leagues", nargs="+", default=SEED_LEAGUES)
    args = parser.parse_args()

    if len(args.seasons) == 1:
        stats = seed_from_fbref(args.seasons[0], args.leagues, args.limit)
    else:
        stats = seed_all_seasons(args.seasons, args.leagues, args.limit)

    print("\n" + "=" * 60)
    print("[Seed] Complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nNext:")
    for s in args.seasons:
        print(f"  python -m extractors.fbref_scraper --season {s}")
    print("=" * 60)


if __name__ == "__main__":
    main()
