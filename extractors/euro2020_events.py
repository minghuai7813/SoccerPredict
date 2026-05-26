"""
Euro 2020 event-level stats extractor (expanded metrics).
Euro 2020 事件级统计提取器（扩展指标）。

Usage / 用法:
    python -m extractors.euro2020_events
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

COMPETITION_ID = 55
SEASON_ID = 43
COMPETITION_LABEL = "UEFA Euro 2020"


if __name__ == "__main__":
    extract_tournament_national_stats(
        COMPETITION_ID,
        SEASON_ID,
        COMPETITION_LABEL,
        date(2021, 7, 11),
        skip_if_competition_exists=False,
    )
