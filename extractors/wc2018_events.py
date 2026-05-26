"""
2018 World Cup event-level national stats extractor.
2018 世界杯事件级国家队统计提取器。

Usage / 用法:
    python -m extractors.wc2018_events
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding

fix_console_encoding()

from extractors.tournament_events import extract_tournament_national_stats

COMPETITION_ID = 43
SEASON_ID = 3
COMPETITION_LABEL = "FIFA World Cup 2018"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract 2018 WC national stats")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip players who already have FIFA World Cup 2018 rows",
    )
    args = parser.parse_args()
    extract_tournament_national_stats(
        COMPETITION_ID,
        SEASON_ID,
        COMPETITION_LABEL,
        date(2018, 7, 15),
        skip_if_competition_exists=args.skip_existing,
    )
