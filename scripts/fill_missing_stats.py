"""
Fill missing WC2026 roster players with league-tier-scaled stat priors.

For each unmatched player we:
1. Classify their club into a league tier (1-5).
2. Look up position-based median stats from our DB (Big-5 baselines).
3. Scale by a tier factor and insert a synthetic PlayerStatsLeague row.

This ensures every WC2026 team gets a meaningful profile instead of NaN.

Usage:
    python scripts/fill_missing_stats.py              # all teams
    python scripts/fill_missing_stats.py --team Turkey # single team
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding
fix_console_encoding()

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from db.models import Base, Player, PlayerStatsLeague

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"
ROSTERS_PATH = _PROJECT_ROOT / "data" / "wc2026_rosters.json"

TIER_FACTOR = {
    1: 1.00,  # Big-5 (EPL, La Liga, BuLi, Serie A, Ligue 1)
    2: 0.82,  # Eredivisie, Liga Portugal, Super Lig, Belgian JPL, Championship
    3: 0.65,  # MLS, A-League, J/K-League, Saudi Pro, Czech/Polish/Scottish top flight
    4: 0.50,  # Lower European, Middle-East top, CONCACAF domestic
    5: 0.35,  # Semi-pro, lower divisions, small confederations
}

# club substring -> tier  (case-insensitive matching)
_TIER_2_CLUBS = {
    "psv", "ajax", "feyenoord", "az ", "nec", "twente", "utrecht",
    "heracles", "pec zwolle", "rkc", "volendam", "sparta rotterdam",
    "sc telstar", "den bosch", "vvv",  # Dutch lower
    "benfica", "sporting", "porto", "braga", "vitoria guimaraes",
    "farense", "chaves", "torreense", "fcsb", "estrela",  # Portugal
    "galatasaray", "fenerbahce", "besiktas", "trabzonspor",
    "konyaspor", "rizespor", "basaksehir", "kayserispor", # Turkey
    "club brugge", "anderlecht", "gent", "genk", "standard liege",
    "charleroi", "union saint", "sint-truiden", "beveren",
    "zulte waregem", "mechelen",  # Belgium
    "celtic", "rangers", "hearts", "hibernian",  # Scotland
    "sheffield united", "burnley", "leeds", "hull", "stoke",
    "swansea", "derby", "middlesbrough", "millwall", "coventry",
    "ipswich", "peterborough", "wrexham", "rotherham", "barnsley",
    "norwich", "port vale", "braintree", "charlton",  # Championship / EFL
    "nottingham forest", "watford", "leicester",  # borderline PL/Champ
    "copenhagen", "brondby", "midtjylland", "nordsjaelland", "silkeborg",
    "viking", "bodo/glimt", "molde",  # Scandinavia top
    "red star belgrade", "dinamo zagreb",  # East Europe top
    "olympiacos", "panathinaikos", "aris",  # Greece
    "fc zurich", "young boys", "servette", "lugano",
    "st gallen",  # Swiss Super League
    "freiburg", "augsburg", "mainz", "wolfsburg", "hoffenheim",
    "werder", "hamburg", "st pauli", "holstein kiel",  # Germany (some 2.BuLi)
}

_TIER_3_CLUBS = {
    "orlando pirates", "mamelodi sundowns", "kaizer chiefs",  # South Africa
    "slavia prague", "sparta prague", "viktoria plzen",  # Czech
    "inter miami", "lafc", "columbus crew", "nashville", "chicago fire",
    "philadelphia union", "new york city", "seattle", "minnesota",
    "portland timbers", "colorado rapids", "atlanta united",
    "toronto", "vancouver whitecaps", "orlando city", "dallas",
    "fc cincinnati", "austin fc", "charlotte fc", "new england",
    "san diego",  # MLS
    "yokohama", "kashima", "fc tokyo", "gamba osaka", "kawasaki",
    "sanfrecce hiroshima", "albirex niigata", "machida zelvia",  # J-League
    "ulsan", "jeonbuk", "daejeon", "gangwon",  # K-League
    "al hilal", "al nassr", "al ahli", "al ittihad", "al qadsiah",
    "al ettifaq", "al fayha", "al ula",  # Saudi Pro
    "al ain", "al wahda", "al jazira", "baniyas", "al bataeh",
    "dibba", "al dhafra",  # UAE
    "abha club", "al duhail", "al sadd", "al rayyan", "al wakrah",
    "al gharafa", "al arabi", "al shamal",  # Qatar/Saudi (move from tier 4 dupes)
    "melbourne city", "melbourne victory", "western sydney",
    "wellington phoenix", "sydney fc", "newcastle jets",
    "auckland",  # A-League / NZ
    "esteghlal", "persepolis", "tractor", "sepahan",
    "foolad", "malavan",  # Iran
    "al duhail", "al sadd", "al rayyan", "al wakrah",
    "al gharafa", "al arabi", "al shamal",  # Qatar
    "cracovia", "pogon szczecin", "jagiellonia",  # Poland
    "ferencvaros",  # Hungary
    "sassuolo", "pisa", "cremonese",  # Serie B
    "lechia gdansk",  # Poland
    "motherwell",  # Scotland lower
    "almere city", "randers",  # misc
    "sunderland",  # was in PL but context-dependent
    "pakhtakor", "neftchi", "nasaf", "navbahor", "bukhara",
    "agmk", "dinamo samarqand", "surkhon",  # Uzbekistan
    "al ahly", "zamalek", "pyramids", "enppi",  # Egypt
    "flamengo", "palmeiras", "sao paulo", "gremio",
    "atletico mineiro", "botafogo", "internacional",
    "bragantino", "fluminense", "corinthians", "vasco da gama",
    "santos",  # Brazil Serie A
    "river plate", "boca juniors", "independiente", "racing",
    "san lorenzo", "lanus", "talleres",  # Argentina top
    "cerro porteno", "olimpia", "club america", "unam",
    "pumas", "chivas", "cruz azul", "toluca", "atlas",
    "tijuana", "pachuca", "mazatlan", "leon",  # Mexico/Paraguay
    "accra hearts", "maribor", "young boys",  # misc
}

_TIER_4_CLUBS = {
    "al hussein", "al wehdat", "al faisaly", "al karma", "al quwa al jawiya",
    "al zawraa", "al shorta", "al talaba", "al sailiya", "al shabab",
    "qatar sc",  # Middle East lower
    "violette", "plaza amador", "marathon", "saprissa",
    "puerto cabello", "deportivo la guaira", "universidad catolica",
    "universidad de concepcion", "cobresal",  # CONCACAF / CONMEBOL lower
    "sochaux", "bastia", "nancy",  # France lower
    "el paso locomotive", "colorado springs",  # USL
    "tatran presov",  # Slovakia
    "fc cosmos koblenz",  # Germany amateur
    "zed", "al najma", "siwele", "polokwane city",
    "el gouna",  # misc small
    "hradec kralove",  # Czech lower
    "selangor", "fc seoul",  # SEA / Korea
    "apollon limassol", "pafos", "cultural leonesa",
    "maccabi haifa", "maccabi tel aviv", "ironi kiryat shmona",
    "igdir", "turan tovuz",  # misc
    "aek larnaca", "kifisia",  # Cyprus / Greece lower
    "vizela", "tondela", "casa pia",  # Portugal lower
    "persib",  # Indonesia
    "grazer ak",  # Austria lower
    "ross county",  # Scotland lower
    "noam", "raja casablanca", "lokomotiva zagreb",
    "rs berkane", "royal armed forces",  # Africa misc
    "larisa", "atromitos",  # Greece lower
    "spartak moscow", "rostov", "pari nizhny novgorod",
    "krasnodar", "dynamo makhachkala",  # Russia
    "abha club", "igdir",  # misc
    "terengganu",  # Malaysia
}


def _classify_tier(club: str) -> int:
    if not club:
        return 5
    cl = club.lower().strip()
    for pattern in _TIER_2_CLUBS:
        if pattern in cl:
            return 2
    for pattern in _TIER_3_CLUBS:
        if pattern in cl:
            return 3
    for pattern in _TIER_4_CLUBS:
        if pattern in cl:
            return 4
    return 5


def compute_position_medians(session: Session) -> dict[str, dict[str, float]]:
    """Compute per-position median stats from existing DB data (Big-5 baseline)."""
    from sqlalchemy import text
    positions = ["GK", "DF", "MF", "FW"]
    cols = ["xg", "goals", "assists", "passes_completed", "passes_attempted",
            "interceptions", "tackles_won", "minutes_played"]
    medians = {}
    for pos in positions:
        pos_data = {}
        for col in cols:
            rows = session.execute(text(f"""
                SELECT psl.{col}
                FROM player_stats_league psl
                JOIN players p ON p.internal_player_id = psl.internal_player_id
                WHERE p.position = :pos
                  AND psl.season IN ('2024-2025', '2025-2026')
                  AND psl.{col} IS NOT NULL
                  AND psl.minutes_played > 200
                ORDER BY psl.{col}
            """), {"pos": pos}).fetchall()
            if rows:
                vals = [r[0] for r in rows]
                mid = len(vals) // 2
                pos_data[col] = float(vals[mid])
            else:
                pos_data[col] = 0.0
        medians[pos] = pos_data
    return medians


def fill_team(team: str, roster: list[dict], medians: dict, session: Session,
              dry_run: bool = False) -> int:
    """Insert synthetic stats for unmatched players on one team. Returns count."""
    unmatched = [p for p in roster if p.get("match_status") != "matched"]
    if not unmatched:
        print(f"  {team}: all matched, skip")
        return 0

    existing_pids = set(
        r[0] for r in session.query(PlayerStatsLeague.internal_player_id)
        .filter(PlayerStatsLeague.internal_player_id.in_(
            [p["player_id"] for p in unmatched]
        )).all()
    )

    filled = 0
    for p in unmatched:
        pid = p["player_id"]
        if pid in existing_pids:
            continue
        pos = p.get("position", "MF")
        club = p.get("club", "")
        tier = _classify_tier(club)
        factor = TIER_FACTOR[tier]
        base = medians.get(pos, medians.get("MF", {}))

        row = PlayerStatsLeague(
            internal_player_id=pid,
            season="2024-2025",
            as_of_date=date(2025, 5, 25),
            xg=round(base.get("xg", 0) * factor, 2),
            goals=int(base.get("goals", 0) * factor),
            assists=int(base.get("assists", 0) * factor),
            passes_completed=int(base.get("passes_completed", 0) * factor),
            passes_attempted=int(base.get("passes_attempted", 0) * factor),
            interceptions=int(base.get("interceptions", 0) * factor),
            tackles_won=int(base.get("tackles_won", 0) * factor),
            minutes_played=int(base.get("minutes_played", 0) * factor),
        )
        if not dry_run:
            session.add(row)
        filled += 1
        print(f"    {p['name']:30s} {pos} tier={tier} factor={factor:.2f}  {club}")

    if not dry_run:
        session.flush()
    return filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default=None, help="Process single team only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rosters = json.loads(ROSTERS_PATH.read_text(encoding="utf-8"))
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        print("[1] Computing position medians from Big-5 data...")
        medians = compute_position_medians(session)
        for pos, vals in medians.items():
            print(f"  {pos}: goals={vals['goals']:.0f} xg={vals['xg']:.1f} "
                  f"assists={vals['assists']:.0f} mins={vals['minutes_played']:.0f} "
                  f"tackles={vals['tackles_won']:.0f} intercept={vals['interceptions']:.0f}")

        total_filled = 0
        teams_to_process = (
            {args.team: rosters[args.team]} if args.team else rosters
        )

        for team, roster in sorted(teams_to_process.items()):
            if not roster:
                continue
            t = len(roster)
            m = sum(1 for x in roster if x.get("match_status") == "matched")
            if m == t:
                continue
            print(f"\n[{team}] {m}/{t} matched, filling {t - m} gaps...")
            n = fill_team(team, roster, medians, session, dry_run=args.dry_run)
            total_filled += n

        if not args.dry_run:
            session.commit()
            print(f"\n=== COMMITTED {total_filled} synthetic stat rows ===")
        else:
            print(f"\n=== DRY RUN: would insert {total_filled} rows ===")


if __name__ == "__main__":
    main()
