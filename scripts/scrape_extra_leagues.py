"""
Scrape real player stats from FBref for non-Big-5 leagues
and replace synthetic priors in DB for WC2026 roster players.

Reuses soccerdata infrastructure from fbref_scraper.py.

Usage:
    python scripts/scrape_extra_leagues.py
    python scripts/scrape_extra_leagues.py --league NED-Eredivisie
    python scripts/scrape_extra_leagues.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
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

from db.models import Base, Player, PlayerStatsLeague
from extractors.fbref_scraper import (
    _scrape_league_with_retry,
    _extract_league_stat_fields,
    _get_col,
    _rate_limit,
    SLEEP_MIN,
    SLEEP_MAX,
)

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"
ROSTERS_PATH = _PROJECT_ROOT / "data" / "wc2026_rosters.json"

# soccerdata league key -> (season, as_of_date)
EXTRA_LEAGUES: dict[str, tuple[str, date]] = {
    "NED-Eredivisie":        ("2024-2025", date(2025, 5, 25)),
    "TUR-Super Lig":         ("2024-2025", date(2025, 5, 25)),
    "POR-Primeira Liga":     ("2024-2025", date(2025, 5, 25)),
    "BEL-First Division A":  ("2024-2025", date(2025, 5, 25)),
    "ENG-Championship":      ("2024-2025", date(2025, 5, 25)),
    "SCO-Premiership":       ("2024-2025", date(2025, 5, 25)),
    "USA-MLS":               ("2025",      date(2025, 6, 1)),
    "ITA-Serie B":           ("2024-2025", date(2025, 5, 25)),
    "FRA-Ligue 2":           ("2024-2025", date(2025, 5, 25)),
    "DEN-Superliga":         ("2024-2025", date(2025, 5, 25)),
    "NOR-Eliteserien":       ("2025",      date(2025, 6, 1)),
    "SUI-Super League":      ("2024-2025", date(2025, 5, 25)),
    "BRA-Serie A":           ("2025",      date(2025, 6, 1)),
    "ARG-Primera Division":  ("2025",      date(2025, 6, 1)),
    "MEX-Liga MX":           ("2024-2025", date(2025, 5, 25)),
    "JPN-J1 League":         ("2025",      date(2025, 6, 1)),
    "KOR-K League 1":        ("2025",      date(2025, 6, 1)),
    "CZE-First League":      ("2024-2025", date(2025, 5, 25)),
    "GRE-Super League":      ("2024-2025", date(2025, 5, 25)),
    "SAU-Pro League":        ("2024-2025", date(2025, 5, 25)),
    "GER-2. Bundesliga":     ("2024-2025", date(2025, 5, 25)),
}

# Map WC2026 roster club patterns -> soccerdata league key
_CLUB_TO_LEAGUE: list[tuple[list[str], str]] = [
    (["psv", "ajax", "feyenoord", "az ", "twente", "utrecht", "nec",
      "heracles", "pec zwolle", "rkc", "sparta rotterdam", "volendam",
      "almere city", "heerenveen", "go ahead", "fortuna sittard",
      "waalwijk", "groningen", "cambuur"], "NED-Eredivisie"),
    (["galatasaray", "fenerbahce", "besiktas", "trabzonspor", "konyaspor",
      "basaksehir", "antalyaspor", "alanyaspor", "samsunspor", "sivasspor",
      "kayserispor", "rizespor", "gaziantep", "hatayspor", "kasimpasa",
      "adana demirspor", "pendikspor", "istanbulspor"], "TUR-Super Lig"),
    (["benfica", "sporting", "porto", "braga", "vitoria guimaraes",
      "farense", "chaves", "estrela", "casa pia", "gil vicente",
      "tondela", "vizela", "arouca", "boavista", "famalicao",
      "moreirense", "nacional", "rio ave"], "POR-Primeira Liga"),
    (["club brugge", "anderlecht", "gent", "genk", "standard liege",
      "charleroi", "union saint", "sint-truiden", "beveren", "mechelen",
      "dender", "zulte waregem", "cercle brugge", "antwerp",
      "kortrijk", "westerlo", "oud-heverlee"], "BEL-First Division A"),
    (["sheffield united", "burnley", "leeds", "hull", "stoke", "swansea",
      "derby", "middlesbrough", "millwall", "coventry", "peterborough",
      "wrexham", "rotherham", "barnsley", "norwich", "charlton", "watford",
      "portsmouth", "blackburn", "sunderland", "west brom", "cardiff",
      "qpr", "luton", "birmingham", "sheffield wednesday", "oxford united",
      "plymouth", "preston", "bristol city"], "ENG-Championship"),
    (["celtic", "rangers", "hearts", "hibernian", "motherwell",
      "dundee", "ross county", "st mirren", "kilmarnock",
      "aberdeen", "livingston", "st johnstone"], "SCO-Premiership"),
    (["inter miami", "lafc", "columbus crew", "nashville", "chicago fire",
      "philadelphia union", "new york city", "seattle", "minnesota",
      "portland timbers", "colorado rapids", "atlanta united",
      "toronto", "vancouver whitecaps", "orlando city", "dallas",
      "fc cincinnati", "austin fc", "charlotte fc", "new england",
      "san diego", "real salt lake", "houston", "montreal",
      "new york red", "la galaxy", "san jose", "sporting kc",
      "st. louis"], "USA-MLS"),
    (["sassuolo", "pisa", "cremonese", "frosinone", "sampdoria",
      "modena", "palermo", "bari", "catanzaro", "brescia",
      "reggiana", "sudtirol", "spezia", "cittadella"], "ITA-Serie B"),
    (["montpellier", "bastia", "sochaux", "nancy", "ajaccio",
      "caen", "guingamp", "lorient", "rodez", "pau",
      "dunkerque", "amiens", "troyes", "grenoble", "laval"], "FRA-Ligue 2"),
    (["copenhagen", "brondby", "midtjylland", "nordsjaelland",
      "silkeborg", "aarhus", "viborg", "lyngby", "randers"], "DEN-Superliga"),
    (["viking", "bodo/glimt", "molde", "sarpsborg",
      "rosenborg", "lillestrom", "stromsgodset", "brann",
      "haugesund", "stabaek", "tromso"], "NOR-Eliteserien"),
    (["fc zurich", "young boys", "servette", "lugano",
      "st gallen", "luzern", "sion", "grasshopper",
      "winterthur", "yverdon"], "SUI-Super League"),
    (["flamengo", "palmeiras", "sao paulo", "gremio",
      "atletico mineiro", "botafogo", "internacional", "bragantino",
      "fluminense", "corinthians", "vasco da gama", "santos",
      "fortaleza", "bahia", "cruzeiro", "athletico paranaense",
      "goias", "cuiaba", "coritiba", "america mineiro"], "BRA-Serie A"),
    (["river plate", "boca juniors", "independiente", "racing",
      "san lorenzo", "lanus", "talleres", "velez sarsfield",
      "estudiantes", "godoy cruz", "union", "argentinos juniors",
      "banfield", "defensa y justicia", "huracan",
      "rosario central", "platense", "tigre"], "ARG-Primera Division"),
    (["club america", "unam", "pumas", "chivas", "cruz azul", "toluca",
      "atlas", "tijuana", "pachuca", "mazatlan", "leon", "santos laguna",
      "monterrey", "tigres", "puebla", "necaxa", "queretaro",
      "juarez"], "MEX-Liga MX"),
    (["kashima", "fc tokyo", "sanfrecce hiroshima", "albirex niigata",
      "machida zelvia", "yokohama", "kawasaki", "urawa", "nagoya",
      "vissel kobe", "cerezo", "gamba", "consadole",
      "avispa"], "JPN-J1 League"),
    (["ulsan", "jeonbuk", "daejeon", "gangwon", "fc seoul",
      "suwon", "pohang", "incheon", "jeju"], "KOR-K League 1"),
    (["slavia prague", "sparta prague", "viktoria plzen",
      "hradec kralove", "banik ostrava", "bohemians",
      "slovacko", "jablonec", "mlada boleslav", "sigma olomouc",
      "teplice"], "CZE-First League"),
    (["olympiacos", "panathinaikos", "aris", "aek athens",
      "paok", "volos", "ofi", "atromitos", "asteras",
      "lamia", "giannina"], "GRE-Super League"),
    (["al hilal", "al nassr", "al ahli", "al ittihad", "al qadsiah",
      "al ettifaq", "al fayha", "al raed", "al fateh", "al tai",
      "damac", "al shabab", "al riyadh", "al akhdoud",
      "al khaleej", "al orubah"], "SAU-Pro League"),
    (["hamburg", "st pauli", "hannover", "karlsruher",
      "fortuna dusseldorf", "holstein kiel", "hertha berlin",
      "greuther furth", "darmstadt", "kaiserslautern",
      "paderborn", "braunschweig", "elversberg",
      "preussen munster", "schalke"], "GER-2. Bundesliga"),
]


def _classify_club_to_league(club: str) -> str | None:
    cl = club.lower().strip()
    for patterns, league in _CLUB_TO_LEAGUE:
        for p in patterns:
            if p in cl:
                return league
    return None


def _load_wc2026_targets() -> dict[str, dict[str, dict]]:
    """Group unmatched WC2026 roster players by soccerdata league key."""
    rosters = json.loads(ROSTERS_PATH.read_text(encoding="utf-8"))
    by_league: dict[str, dict[str, dict]] = {}

    for team, players in rosters.items():
        for p in players:
            if p.get("match_status") == "matched":
                continue
            club = p.get("club", "")
            league = _classify_club_to_league(club)
            if league is None:
                continue
            if league not in by_league:
                by_league[league] = {}
            by_league[league][p["player_id"]] = p

    return by_league


def _match_and_update(
    league_df: pd.DataFrame,
    targets: dict[str, dict],
    session: Session,
    as_of: date,
    dry_run: bool = False,
) -> int:
    """Match scraped FBref rows to WC2026 roster players, update DB."""
    updated = 0

    for pid, roster_entry in targets.items():
        target_name = roster_entry["name"]
        target_ascii = to_ascii_name(target_name)

        best_row = None
        best_score = 0
        best_fbref_name = ""

        for _, row in league_df.iterrows():
            fbref_name = str(row.get("player", "")).strip()
            if not fbref_name or fbref_name == "nan":
                continue
            fbref_ascii = to_ascii_name(fbref_name)
            score = fuzz.token_set_ratio(target_ascii, fbref_ascii)
            if score > best_score:
                best_score = score
                best_row = row
                best_fbref_name = fbref_name

        if best_score >= 80 and best_row is not None:
            fields = _extract_league_stat_fields(best_row)

            existing = session.query(PlayerStatsLeague).filter_by(
                internal_player_id=pid,
                season="2024-2025",
            ).first()
            if existing is None:
                existing = session.query(PlayerStatsLeague).filter_by(
                    internal_player_id=pid,
                ).order_by(PlayerStatsLeague.as_of_date.desc()).first()

            if existing and not dry_run:
                for k, v in fields.items():
                    if v is not None:
                        setattr(existing, k, v)
                existing.as_of_date = as_of
            elif not existing and not dry_run:
                stat_row = PlayerStatsLeague(
                    internal_player_id=pid,
                    season="2024-2025",
                    as_of_date=as_of,
                    **{k: v for k, v in fields.items()},
                )
                session.add(stat_row)

            # Also update position if available
            pos_raw = _get_col(best_row, "pos", "Pos")
            if pos_raw and not dry_run:
                pos_str = str(pos_raw).strip()
                if pos_str and pos_str != "nan":
                    player = session.get(Player, pid)
                    if player and not player.position:
                        player.position = pos_str

            updated += 1
            marker = "[DRY]" if dry_run else "[UPD]"
            print(
                f"  {marker} {target_name:30s} <- {best_fbref_name:30s} "
                f"(score={best_score}) G={fields.get('goals')} "
                f"xG={fields.get('xg')} Min={fields.get('minutes_played')}"
            )

    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets_by_league = _load_wc2026_targets()
    print(f"WC2026 targets across {len(targets_by_league)} leagues:")
    for lg, pids in sorted(targets_by_league.items(), key=lambda x: -len(x[1])):
        print(f"  {lg:30s} {len(pids):3d} players")

    engine = create_engine(DB_URL, echo=False)
    total_updated = 0
    total_failed = 0

    leagues_to_do = (
        {args.league: EXTRA_LEAGUES[args.league]}
        if args.league and args.league in EXTRA_LEAGUES
        else EXTRA_LEAGUES
    )

    with Session(engine) as session:
        for league_key, (season, as_of) in leagues_to_do.items():
            targets = targets_by_league.get(league_key, {})
            if not targets:
                continue

            print(f"\n{'='*60}")
            print(f"[{league_key}] {len(targets)} WC2026 players to find (season={season})")

            league_df = _scrape_league_with_retry(league_key, season)
            if league_df is None or league_df.empty:
                print(f"  FAILED - no data returned")
                total_failed += len(targets)
                continue

            print(f"  Scraped {len(league_df)} player rows from FBref")

            n = _match_and_update(league_df, targets, session, as_of, args.dry_run)
            total_updated += n
            print(f"  Result: {n}/{len(targets)} updated")

            if not args.dry_run:
                session.commit()
                print(f"  Committed to DB")

            _rate_limit()

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_updated} players updated with REAL FBref data")
    if total_failed:
        print(f"FAILED leagues: {total_failed} players not updated")


if __name__ == "__main__":
    main()
