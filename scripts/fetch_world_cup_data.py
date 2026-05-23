"""Fetch World Cup data via soccerdata and persist it under ./data/raw.

Usage:
    python scripts/fetch_world_cup_data.py --season 2022
    python scripts/fetch_world_cup_data.py --season 2022 --include-players
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccerpredict.data import (  # noqa: E402
    fetch_player_stats,
    fetch_schedule,
    fetch_team_stats,
)
from soccerpredict.utils import get_logger, load_config  # noqa: E402
from soccerpredict.utils.paths import data_dir  # noqa: E402

log = get_logger("scripts.fetch_world_cup_data")


def _safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("/", "-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2022", help="World Cup year, e.g. 2022.")
    parser.add_argument("--source", default="FBref", help="soccerdata source class name.")
    parser.add_argument(
        "--include-players",
        action="store_true",
        help="Also fetch player-level season stats (slower).",
    )
    args = parser.parse_args()

    cfg = load_config()
    competition = "FIFA World Cup"
    out_dir = data_dir("raw")

    log.info("Fetching schedule for {} {}...", competition, args.season)
    schedule = fetch_schedule(
        competition=competition,
        season=args.season,
        source=args.source,
        data_dir=cfg.soccerdata_dir,
    )
    schedule_path = out_dir / f"{_safe_name(competition)}_{args.season}_schedule.parquet"
    schedule.to_parquet(schedule_path)
    log.info("Saved schedule -> {} ({} rows)", schedule_path, len(schedule))

    try:
        log.info("Fetching team season stats...")
        team_stats = fetch_team_stats(
            competition=competition,
            season=args.season,
            stat_type="standard",
            source=args.source,
            data_dir=cfg.soccerdata_dir,
        )
        team_path = out_dir / f"{_safe_name(competition)}_{args.season}_team_stats.parquet"
        team_stats.to_parquet(team_path)
        log.info("Saved team stats -> {} ({} rows)", team_path, len(team_stats))
    except NotImplementedError as err:
        log.warning("Skipping team stats: {}", err)

    if args.include_players:
        try:
            log.info("Fetching player season stats (this can be slow)...")
            player_stats = fetch_player_stats(
                competition=competition,
                season=args.season,
                stat_type="standard",
                source=args.source,
                data_dir=cfg.soccerdata_dir,
            )
            player_path = (
                out_dir / f"{_safe_name(competition)}_{args.season}_player_stats.parquet"
            )
            player_stats.to_parquet(player_path)
            log.info("Saved player stats -> {} ({} rows)", player_path, len(player_stats))
        except NotImplementedError as err:
            log.warning("Skipping player stats: {}", err)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
