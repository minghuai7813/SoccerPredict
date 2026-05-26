"""
StatsBomb event extractor — 2022 World Cup (expanded metrics).
StatsBomb 事件提取器 — 2022 世界杯（扩展指标）。

Usage / 用法:
    python -m extractors.statsbomb_events
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

from extractors.player_event_metrics import _aggregate_player_events  # noqa: F401
from extractors.tournament_events import extract_tournament_national_stats

COMPETITION_ID = 43
SEASON_ID = 106
COMPETITION_LABEL = "FIFA World Cup 2022"


def extract_and_update_national_stats() -> dict[str, int]:
    """2022 WC pipeline / 2022 世界杯流程。"""
    return extract_tournament_national_stats(
        COMPETITION_ID,
        SEASON_ID,
        COMPETITION_LABEL,
        date(2022, 12, 18),
        skip_if_competition_exists=False,
    )


if __name__ == "__main__":
    extract_and_update_national_stats()
