"""
Direct FBref scraper using Selenium + pandas.
Bypasses soccerdata's broken league routing for non-Big-5 leagues.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding, to_ascii_name
fix_console_encoding()

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from thefuzz import fuzz

from db.models import Player, PlayerStatsLeague

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"
ROSTERS_PATH = _PROJECT_ROOT / "data" / "wc2026_rosters.json"

LEAGUES_TO_SCRAPE = {
    "ENG-Championship":      (10,  "Championship",             "2024-2025"),
    "BEL-First Division A":  (37,  "Belgian-First-Division-A", "2024-2025"),
    "BRA-Serie A":           (24,  "Serie-A",                  "2025"),
    "ARG-Primera Division":  (21,  "Primera-Division",         "2025"),
    "SAU-Pro League":        (70,  "Saudi-Professional-League","2024-2025"),
    "GER-2. Bundesliga":     (33,  "2-Bundesliga",             "2024-2025"),
    "SUI-Super League":      (57,  "Swiss-Super-League",       "2024-2025"),
    "MEX-Liga MX":           (31,  "Liga-MX",                  "2024-2025"),
    "JPN-J1 League":         (25,  "J1-League",                "2025"),
    "KOR-K League 1":        (55,  "K-League-1",               "2025"),
    "CZE-First League":      (66,  "Czech-First-League",       "2024-2025"),
    "GRE-Super League":      (27,  "Super-League-Greece",      "2024-2025"),
}

# Club pattern -> league key (same as scrape_extra_leagues.py)
_CLUB_TO_LEAGUE = {
    "ENG-Championship": ["sheffield united", "burnley", "leeds", "hull", "stoke",
        "swansea", "derby", "middlesbrough", "millwall", "coventry",
        "peterborough", "wrexham", "rotherham", "barnsley", "norwich",
        "charlton", "watford", "portsmouth", "blackburn", "sunderland",
        "west brom", "cardiff", "qpr", "luton", "birmingham",
        "sheffield wednesday", "oxford united", "plymouth", "preston",
        "bristol city"],
    "BEL-First Division A": ["club brugge", "anderlecht", "gent", "genk",
        "standard liege", "charleroi", "union saint", "sint-truiden",
        "beveren", "mechelen", "dender", "cercle brugge", "antwerp",
        "kortrijk", "westerlo", "oud-heverlee"],
    "BRA-Serie A": ["flamengo", "palmeiras", "sao paulo", "gremio",
        "atletico mineiro", "botafogo", "internacional", "bragantino",
        "fluminense", "corinthians", "vasco da gama", "santos",
        "fortaleza", "bahia", "cruzeiro", "athletico paranaense"],
    "ARG-Primera Division": ["river plate", "boca juniors", "independiente",
        "racing", "san lorenzo", "lanus", "talleres", "velez sarsfield",
        "estudiantes", "huracan", "rosario central"],
    "SAU-Pro League": ["al hilal", "al nassr", "al ahli", "al ittihad",
        "al qadsiah", "al ettifaq", "al fayha", "al shabab", "al riyadh",
        "al akhdoud", "al khaleej", "al orubah", "al raed", "damac",
        "al fateh", "al tai"],
    "GER-2. Bundesliga": ["hamburg", "st pauli", "hannover", "karlsruher",
        "fortuna dusseldorf", "holstein kiel", "hertha berlin",
        "greuther furth", "darmstadt", "kaiserslautern", "paderborn",
        "braunschweig", "elversberg", "preussen munster", "schalke"],
    "SUI-Super League": ["fc zurich", "young boys", "servette", "lugano",
        "st gallen", "luzern", "sion", "grasshopper", "winterthur", "yverdon"],
    "MEX-Liga MX": ["club america", "unam", "pumas", "chivas", "cruz azul",
        "toluca", "atlas", "tijuana", "pachuca", "mazatlan", "leon",
        "santos laguna", "monterrey", "tigres"],
    "JPN-J1 League": ["kashima", "fc tokyo", "sanfrecce hiroshima",
        "albirex niigata", "machida zelvia", "yokohama", "kawasaki",
        "urawa", "vissel kobe"],
    "KOR-K League 1": ["ulsan", "jeonbuk", "daejeon", "gangwon", "fc seoul",
        "suwon", "pohang", "incheon"],
    "CZE-First League": ["slavia prague", "sparta prague", "viktoria plzen",
        "hradec kralove", "banik ostrava", "bohemians", "slovacko",
        "jablonec", "mlada boleslav"],
    "GRE-Super League": ["olympiacos", "panathinaikos", "aris", "aek athens",
        "paok", "volos", "atromitos"],
}


def _classify_club(club: str) -> str | None:
    cl = club.lower().strip()
    for league, patterns in _CLUB_TO_LEAGUE.items():
        for p in patterns:
            if p in cl:
                return league
    return None


def _load_targets() -> dict[str, dict[str, dict]]:
    rosters = json.loads(ROSTERS_PATH.read_text(encoding="utf-8"))
    by_league: dict[str, dict[str, dict]] = {}
    for team, players in rosters.items():
        for p in players:
            if p.get("match_status") == "matched":
                continue
            club = p.get("club", "")
            league = _classify_club(club)
            if league is None:
                continue
            if league not in by_league:
                by_league[league] = {}
            by_league[league][p["player_id"]] = p
    return by_league


def _get_driver():
    from seleniumbase import Driver
    driver = Driver(uc=True, headless=True)
    return driver


def _fbref_stats_url(comp_id: int, slug: str, season: str) -> str:
    return f"https://fbref.com/en/comps/{comp_id}/{season}/stats/{season}-{slug}-Stats"


def scrape_with_selenium(url: str, driver) -> pd.DataFrame | None:
    print(f"  Loading {url}")
    try:
        driver.get(url)
        time.sleep(random.uniform(3.0, 5.0))
        html = driver.page_source
        tables = pd.read_html(StringIO(html))
        if not tables:
            return None
        df = max(tables, key=len)
        return df
    except Exception as e:
        print(f"  Selenium error: {e}")
        return None


def _extract_stats_from_df(df: pd.DataFrame) -> list[dict]:
    cols = [str(c) for c in df.columns]

    def _find(*patterns):
        for pat in patterns:
            for c in cols:
                if pat.lower() in c.lower():
                    return c
        return None

    player_col = _find("Player", "player")
    goals_col = _find("Gls")
    assists_col = _find("Ast")
    xg_col = _find("xG")
    min_col = _find("Min")
    tkl_col = _find("Tkl")
    int_col = _find("Int")

    if player_col is None:
        return []

    results = []
    for _, row in df.iterrows():
        name = str(row.get(player_col, "")).strip()
        if not name or name == "nan" or "Player" in name:
            continue

        def _snum(col):
            if col is None:
                return None
            v = row.get(col)
            if pd.isna(v):
                return None
            try:
                return float(str(v).replace(",", ""))
            except (ValueError, TypeError):
                return None

        results.append({
            "name": name,
            "ascii": to_ascii_name(name),
            "goals": int(_snum(goals_col)) if _snum(goals_col) is not None else None,
            "assists": int(_snum(assists_col)) if _snum(assists_col) is not None else None,
            "xg": _snum(xg_col),
            "minutes_played": int(_snum(min_col)) if _snum(min_col) is not None else None,
            "tackles_won": int(_snum(tkl_col)) if _snum(tkl_col) is not None else None,
            "interceptions": int(_snum(int_col)) if _snum(int_col) is not None else None,
        })
    return results


def _update_db(stats: list[dict], targets: dict[str, dict], session, as_of: date) -> int:
    updated = 0
    for pid, entry in targets.items():
        name = entry["name"]
        ascii_name = to_ascii_name(name)
        best, best_score, best_stat = None, 0, None
        for s in stats:
            score = fuzz.token_set_ratio(ascii_name, s["ascii"])
            if score > best_score:
                best_score = score
                best = s["name"]
                best_stat = s
        if best_score >= 80 and best_stat:
            existing = session.query(PlayerStatsLeague).filter_by(
                internal_player_id=pid, season="2024-2025"
            ).first()
            if not existing:
                existing = session.query(PlayerStatsLeague).filter_by(
                    internal_player_id=pid
                ).order_by(PlayerStatsLeague.as_of_date.desc()).first()
            if existing:
                for k in ("goals", "assists", "xg", "minutes_played", "tackles_won", "interceptions"):
                    v = best_stat.get(k)
                    if v is not None:
                        setattr(existing, k, v)
                existing.as_of_date = as_of
            updated += 1
            print(f"    UPD {name:30s} <- {best:30s} (score={best_score}) "
                  f"G={best_stat.get('goals')} xG={best_stat.get('xg')}")
    return updated


def main():
    targets_by_league = _load_targets()
    for lg, pids in sorted(targets_by_league.items(), key=lambda x: -len(x[1])):
        print(f"  {lg:30s} {len(pids):3d}")

    driver = _get_driver()
    engine = create_engine(DB_URL, echo=False)
    total = 0

    try:
        with Session(engine) as session:
            for league_key, (comp_id, slug, season) in LEAGUES_TO_SCRAPE.items():
                targets = targets_by_league.get(league_key, {})
                if not targets:
                    continue

                print(f"\n{'='*60}")
                print(f"[{league_key}] {len(targets)} players (season={season})")

                url = _fbref_stats_url(comp_id, slug, season)
                df = scrape_with_selenium(url, driver)

                if df is None or df.empty:
                    print("  No data, skip")
                    continue

                stats = _extract_stats_from_df(df)
                print(f"  Parsed {len(stats)} players from table")

                n = _update_db(stats, targets, session, date(2025, 6, 1))
                total += n
                print(f"  Result: {n}/{len(targets)} updated")

                session.commit()
                time.sleep(random.uniform(4, 7))
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"TOTAL: {total} players updated with real FBref data")


if __name__ == "__main__":
    main()
