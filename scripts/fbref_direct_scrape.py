"""
Direct FBref scraper for non-Big-5 leagues using requests + pandas.
Lightweight alternative to soccerdata/Selenium.

Usage:
    python scripts/fbref_direct_scrape.py                    # all leagues
    python scripts/fbref_direct_scrape.py --league TUR       # single league
    python scripts/fbref_direct_scrape.py --dry-run          # preview only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding, to_ascii_name
fix_console_encoding()

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from db.models import Base, Player, PlayerStatsLeague
from thefuzz import fuzz

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"
ROSTERS_PATH = _PROJECT_ROOT / "data" / "wc2026_rosters.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://fbref.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# FBref comp_id -> (url_slug, display_name)
EXTRA_LEAGUES = {
    "TUR": (26, "Super-Lig", "Super Lig"),
    "NED": (23, "Eredivisie", "Eredivisie"),
    "ENG2": (10, "Championship", "Championship"),
    "POR": (32, "Primeira-Liga", "Primeira Liga"),
    "BEL": (37, "Belgian-Pro-League", "Belgian Pro League"),
    "USA": (22, "Major-League-Soccer", "MLS"),
    "SAU": (70, "Saudi-Professional-League", "Saudi Pro League"),
    "CZE": (66, "Czech-First-League", "Czech First League"),
    "SCO": (40, "Scottish-Premiership", "Scottish Premiership"),
    "BRA": (24, "Serie-A", "Serie A (Brazil)"),
    "MEX": (31, "Liga-MX", "Liga MX"),
    "DEN": (50, "Danish-Superliga", "Danish Superliga"),
    "ARG": (21, "Primera-Division", "Primera Division"),
    "SUI": (56, "Super-League", "Swiss Super League"),
    "NOR": (28, "Eliteserien", "Eliteserien"),
    "JPN": (25, "J1-League", "J1 League"),
    "KOR": (55, "K-League-1", "K League 1"),
    "GRE": (27, "Super-League-Greece", "Super League Greece"),
    "GER2": (33, "2-Bundesliga", "2. Bundesliga"),
}

DELAY_MIN = 5
DELAY_MAX = 9


def _fetch_fbref_stats(comp_id: int, url_slug: str) -> pd.DataFrame | None:
    url = f"https://fbref.com/en/comps/{comp_id}/stats/{url_slug}-Stats"
    print(f"  GET {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            print(f"  429 Rate Limited! Sleeping 600s...")
            time.sleep(600)
            resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  HTTP error: {e}")
        return None

    try:
        tables = pd.read_html(resp.text, attrs={"id": "stats_standard"})
        if tables:
            return tables[0]
    except Exception:
        pass

    try:
        tables = pd.read_html(resp.text)
        if tables:
            for t in tables:
                cols_str = " ".join(str(c) for c in t.columns)
                if "Gls" in cols_str or "Goals" in cols_str:
                    return t
            return tables[0]
    except Exception as e:
        print(f"  Parse error: {e}")
        return None

    return None


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        for col in df.columns:
            if c.lower() in str(col).lower():
                return col
    return None


def _get_val(row, df, candidates, default=None):
    col = _find_col(df, candidates)
    if col is None:
        return default
    val = row.get(col)
    if pd.isna(val):
        return default
    return val


def _match_player(name_ascii: str, roster_ascii_map: dict[str, str],
                  threshold: int = 75) -> str | None:
    best_pid = None
    best_score = 0
    for r_ascii, pid in roster_ascii_map.items():
        score = fuzz.token_set_ratio(name_ascii, r_ascii)
        if score > best_score:
            best_score = score
            best_pid = pid
    if best_score >= threshold:
        return best_pid
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None, help="Single league key (e.g. TUR)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rosters = json.loads(ROSTERS_PATH.read_text(encoding="utf-8"))
    engine = create_engine(DB_URL, echo=False)

    # Build ASCII lookup for all WC2026 roster players who lack real stats
    # (i.e., those with match_status != 'matched' or those in non-Big-5 clubs)
    all_roster_pids = {}
    for team, players in rosters.items():
        for p in players:
            ascii_name = to_ascii_name(p["name"])
            all_roster_pids[ascii_name] = p["player_id"]

    leagues_to_scrape = (
        {args.league: EXTRA_LEAGUES[args.league]}
        if args.league
        else EXTRA_LEAGUES
    )

    total_updated = 0

    with Session(engine) as session:
        for league_key, (comp_id, url_slug, display) in leagues_to_scrape.items():
            print(f"\n{'='*60}")
            print(f"[{league_key}] {display} (comp_id={comp_id})")
            print(f"{'='*60}")

            df = _fetch_fbref_stats(comp_id, url_slug)
            if df is None or df.empty:
                print("  No data retrieved, skip")
                delay = random.uniform(DELAY_MIN, DELAY_MAX)
                print(f"  Waiting {delay:.1f}s...")
                time.sleep(delay)
                continue

            df = _flatten_cols(df)
            player_col = _find_col(df, ["Player", "player"])
            if player_col is None:
                print("  Cannot find Player column, skip")
                continue

            matched_count = 0
            for _, row in df.iterrows():
                raw_name = str(row.get(player_col, ""))
                if not raw_name or raw_name == "nan":
                    continue
                name_ascii = to_ascii_name(raw_name)
                pid = _match_player(name_ascii, all_roster_pids, threshold=78)
                if pid is None:
                    continue

                goals = int(_get_val(row, df, ["Gls", "Goals"], 0) or 0)
                assists = int(_get_val(row, df, ["Ast", "Assists"], 0) or 0)
                minutes = int(
                    str(_get_val(row, df, ["Min", "Minutes"], 0) or 0)
                    .replace(",", "")
                )
                xg = float(_get_val(row, df, ["xG"], 0) or 0)
                passes_att = int(_get_val(row, df, ["Att_Passes", "PasTotAtt", "Passes_Att"], 0) or 0)
                passes_cmp = int(_get_val(row, df, ["Cmp_Passes", "PasTotCmp", "Passes_Cmp"], 0) or 0)
                tkl = int(_get_val(row, df, ["Tkl", "Tackles"], 0) or 0)
                interc = int(_get_val(row, df, ["Int", "Intercep"], 0) or 0)

                if minutes < 50:
                    continue

                existing = session.query(PlayerStatsLeague).filter_by(
                    internal_player_id=pid, season="2024-2025"
                ).first()

                if existing and existing.minutes_played and existing.minutes_played > minutes:
                    continue

                if existing:
                    existing.goals = goals
                    existing.assists = assists
                    existing.minutes_played = minutes
                    existing.xg = xg
                    existing.passes_attempted = passes_att or existing.passes_attempted
                    existing.passes_completed = passes_cmp or existing.passes_completed
                    existing.tackles_won = tkl or existing.tackles_won
                    existing.interceptions = interc or existing.interceptions
                    existing.as_of_date = date(2025, 5, 25)
                else:
                    session.add(PlayerStatsLeague(
                        internal_player_id=pid, season="2024-2025",
                        as_of_date=date(2025, 5, 25),
                        goals=goals, assists=assists, minutes_played=minutes,
                        xg=xg, passes_attempted=passes_att,
                        passes_completed=passes_cmp,
                        tackles_won=tkl, interceptions=interc,
                    ))

                matched_count += 1
                print(f"    {raw_name:30s} G={goals} A={assists} xG={xg:.1f} min={minutes}")

            total_updated += matched_count
            print(f"  >> {matched_count} WC2026 players updated from {display}")

            if not args.dry_run:
                session.commit()

            delay = random.uniform(DELAY_MIN, DELAY_MAX)
            print(f"  Waiting {delay:.1f}s before next league...")
            time.sleep(delay)

        if not args.dry_run:
            session.commit()

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_updated} players updated with real FBref data")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
