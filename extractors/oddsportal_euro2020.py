"""
Scrape UEFA Euro 2020 1X2 odds from OddsPortal (not in OddsHarvester league list).
抓取 2020 欧洲杯 1X2 赔率。

Usage / 用法:
    python -m extractors.oddsportal_euro2020
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

EURO_RESULTS_URL = "https://www.oddsportal.com/football/europe/euro-2020/results/"
OUT_PATH = _PROJECT_ROOT / "data" / "odds" / "euro2020_oddsportal.csv"


def _collect_match_links() -> list[str]:
    from playwright.sync_api import sync_playwright

    links: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(EURO_RESULTS_URL, wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(3000)
        html = page.content()
        browser.close()

    for m in re.finditer(r'href="(/football/h2h/[^"]+#[A-Za-z0-9]+)"', html):
        links.append("https://www.oddsportal.com" + m.group(1))
    return list(dict.fromkeys(links))


def scrape_euro2020_odds(out_path: Path | None = None) -> bool:
    out_path = out_path or OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    links = _collect_match_links()
    print(f"[Euro2020] Found {len(links)} match links")
    if not links:
        return False

    # Batch scrape via oddsharvester match-link mode (chunks of 10)
    all_rows: list[Path] = []
    chunk_size = 8
    for i in range(0, len(links), chunk_size):
        chunk = links[i : i + chunk_size]
        tmp = out_path.parent / f"_euro2020_chunk_{i // chunk_size}.csv"
        cmd = [
            sys.executable, "-m", "oddsharvester", "historic",
            "-s", "football",
            "--season", "2020",
            "-m", "1x2",
            "-f", "csv",
            "-o", str(tmp.with_suffix("")),
            "--headless",
        ]
        for link in chunk:
            cmd.extend(["--match-link", link])
        print(f"[Euro2020] Chunk {i // chunk_size + 1}, {len(chunk)} matches...")
        rc = subprocess.call(cmd)
        if rc == 0 and tmp.exists():
            all_rows.append(tmp)

    if not all_rows:
        return False

    import pandas as pd
    parts = [pd.read_csv(p) for p in all_rows if p.exists()]
    pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["home_team", "away_team", "match_date"],
    ).to_csv(out_path, index=False)
    for p in all_rows:
        p.unlink(missing_ok=True)
    print(f"[Euro2020] Saved {len(parts)} chunks -> {out_path}")
    return True


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    ok = scrape_euro2020_odds()
    sys.exit(0 if ok else 1)
