"""
Download historical 1X2 odds from OddsPortal into data/odds/.
从 OddsPortal 下载历史 1X2 赔率到 data/odds/。

Requires: pip install oddsharvester && python -m playwright install chromium

Usage / 用法:
    python scripts/download_odds.py
    python scripts/download_odds.py --tournament wc2018
    python scripts/download_odds.py --rebuild-unified
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding

fix_console_encoding()

ODDS_DIR = _PROJECT_ROOT / "data" / "odds"

# OddsHarvester: league slug + season year on oddsportal.com
SCRAPE_JOBS = {
    "wc2018": ("world-cup", "2018", "wc2018_oddsportal.csv"),
    "wc2022": ("world-cup", "2022", "wc2022_oddsportal.csv"),
    "euro2020": None,  # dedicated scraper in extractors/oddsportal_euro2020.py
}


def _run_oddsharvester(league: str, season: str, outfile: Path) -> int:
    cmd = [
        sys.executable, "-m", "oddsharvester", "historic",
        "-s", "football",
        "-l", league,
        "-m", "1x2",
        "--season", season,
        "-f", "csv",
        "-o", str(outfile.with_suffix("")),
        "--headless",
    ]
    print(f"\n[Download] {' '.join(cmd)}")
    return subprocess.call(cmd)


def _scrape_euro2020_manual(outfile: Path) -> int:
    """
    Euro 2020 is not a built-in OddsHarvester league; scrape via results URLs.
    欧洲杯不在 oddsharvester 联赛列表中，用结果页链接批量抓取。
    """
    try:
        from extractors.oddsportal_euro2020 import scrape_euro2020_odds
    except ImportError:
        print("[Download] euro2020 module not available; skip.")
        return 1
    return 0 if scrape_euro2020_odds(outfile) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Download tournament betting odds")
    parser.add_argument(
        "--tournament",
        choices=list(SCRAPE_JOBS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--rebuild-unified", action="store_true")
    args = parser.parse_args()

    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = SCRAPE_JOBS if args.tournament == "all" else {args.tournament: SCRAPE_JOBS[args.tournament]}

    for key, job in jobs.items():
        if job is None:
            out = ODDS_DIR / "euro2020_oddsportal.csv"
            if out.exists() and args.tournament != "all":
                print(f"[Download] {out.name} exists, skip")
            else:
                print("[Download] Euro 2020: running dedicated scraper...")
                _scrape_euro2020_manual(out)
            continue
        league, season, fname = job
        out = ODDS_DIR / fname
        if out.exists() and args.tournament != "all":
            print(f"[Download] {out.name} exists, skip (delete to re-fetch)")
            continue
        rc = _run_oddsharvester(league, season, out)
        if rc != 0:
            print(f"[Download] WARNING: {key} exited {rc}")

    if args.rebuild_unified or True:  # always refresh unified after download run
        from extractors.odds_loader import load_match_odds
        df = load_match_odds(rebuild=True)
        print(f"\n[Download] Unified odds: {len(df)} rows -> data/odds/match_odds_unified.csv")
        if not df.empty:
            print(df.groupby("tournament").size().to_string())


if __name__ == "__main__":
    main()
