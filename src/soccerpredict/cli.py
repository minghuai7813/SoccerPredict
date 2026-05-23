"""Command-line entry point.

Installed by ``pip install -e .`` as the ``soccerpredict`` console
script. Run ``soccerpredict --help`` for the up-to-date usage.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from soccerpredict import __version__
from soccerpredict.utils import get_logger, load_config
from soccerpredict.utils.paths import data_dir

log = get_logger(__name__)


def _cmd_fetch(args: argparse.Namespace) -> int:
    from soccerpredict.data import fetch_schedule

    cfg = load_config()
    competition = args.competition or cfg.default_competition.name
    season = args.season

    out_dir = data_dir("raw")
    out_path = out_dir / f"{competition.replace(' ', '_')}_{season}_schedule.parquet"

    log.info("Fetching schedule for {} {}", competition, season)
    df = fetch_schedule(
        competition=competition,
        season=season,
        source=args.source,
        data_dir=cfg.soccerdata_dir,
    )

    df.to_parquet(out_path)
    log.info("Wrote {} rows to {}", len(df), out_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soccerpredict",
        description="Soccer match prediction toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"soccerpredict {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch match data for a competition/season.")
    p_fetch.add_argument("--competition", default=None, help="Competition name, e.g. 'FIFA World Cup'.")
    p_fetch.add_argument("--season", default="2022", help="Season tag, e.g. '2022' or '23-24'.")
    p_fetch.add_argument("--source", default="FBref", help="soccerdata source class name.")
    p_fetch.set_defaults(func=_cmd_fetch)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
