"""
Parse WC2026 squad lists from Sky Sports text dump, then match/seed into DB.

Step 1: Parse raw text -> data/wc2026_squads.json
Step 2: Match each player to DB via fuzzy matching, seed missing players
Step 3: Save final roster mapping to data/wc2026_rosters.json

Usage:
    python scripts/parse_wc2026_squads.py
"""
import json
import re
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding, to_ascii_name
fix_console_encoding()

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from db.models import Base, Player
from utils.entity_resolution import PlayerMatcher

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"

SOURCE_FILE = (
    Path(r"C:\Users\mengt\.cursor\projects\d-CursorProjects-SoccerProject")
    / "agent-tools"
    / "a80f62b0-df9e-4935-a1d6-59a922b2dbaa.txt"
)

TEAM_RENAME = {
    "USA": "United States",
    "Ivory Coast": "C\u00f4te d'Ivoire",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

POS_MAP = {
    "Goalkeepers": "GK",
    "Defenders": "DF",
    "Midfielders": "MF",
    "Forwards": "FW",
}


def parse_player_segment(text, position):
    results = []
    entries = re.split(r",\s*(?=[A-Z\u00C0-\u017F])", text.strip().rstrip("."))
    for entry in entries:
        entry = entry.strip().rstrip(";").strip()
        if not entry or len(entry) < 3:
            continue
        m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", entry)
        if m:
            name = m.group(1).strip()
            club_raw = m.group(2).strip()
            if ", on loan" in club_raw:
                club_raw = club_raw.split(", on loan")[0]
            results.append({"name": name, "club": club_raw, "position": position})
        else:
            name = entry.strip()
            if name and len(name) > 2:
                results.append({"name": name, "club": "", "position": position})
    return results


def parse_squads_from_text(filepath):
    raw = open(filepath, "r", encoding="utf-16-le").read()
    lines = raw.split("\n")
    squads = {}
    current_team = None

    SKIP_HEADERS = {
        "Also See:",
        "Sign up for Sky Sports push notifications",
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("### "):
            team = line[4:].strip()
            if team in SKIP_HEADERS or "Sky Sports" in team or "Upgrade" in team:
                continue
            team = TEAM_RENAME.get(team, team)
            current_team = team
            squads[team] = []
            continue

        if current_team and ":" in line:
            for pos_label, pos_code in POS_MAP.items():
                if line.startswith(pos_label + ":"):
                    player_text = line[len(pos_label) + 1 :].strip()
                    players = parse_player_segment(player_text, pos_code)
                    squads[current_team].extend(players)
                    break

        if line.startswith("Manager:"):
            current_team = None

    return squads


def match_and_seed(squads):
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        name_to_pid = {}
        for pid, name in session.query(
            Player.internal_player_id, Player.full_name
        ).all():
            name_to_pid[name] = pid
        print(f"DB has {len(name_to_pid)} players for matching")

        matcher = PlayerMatcher(name_to_pid)
        rosters = {}
        stats = {"matched": 0, "new": 0, "total": 0}

        for team, players in sorted(squads.items()):
            team_roster = []
            matched_count = 0
            for p in players:
                stats["total"] += 1
                result = matcher.match_name(p["name"], threshold=78)
                if result.is_new:
                    new_pid = str(uuid.uuid4())
                    ascii_name = to_ascii_name(p["name"])
                    new_player = Player(
                        internal_player_id=new_pid,
                        full_name=p["name"],
                        full_name_ascii=ascii_name,
                        display_name=p["name"],
                        position=p["position"],
                        current_club=p["club"],
                    )
                    session.add(new_player)
                    name_to_pid[p["name"]] = new_pid
                    stats["new"] += 1
                    team_roster.append({
                        "player_id": new_pid,
                        "name": p["name"],
                        "club": p["club"],
                        "position": p["position"],
                        "match_status": "new",
                    })
                else:
                    stats["matched"] += 1
                    matched_count += 1
                    team_roster.append({
                        "player_id": result.internal_player_id,
                        "name": p["name"],
                        "matched_to": result.matched_name,
                        "score": result.score,
                        "club": p["club"],
                        "position": p["position"],
                        "match_status": "matched",
                    })
            rosters[team] = team_roster
            pct = matched_count / len(players) * 100 if players else 0
            print(
                f"  {team:<30s} {len(players):2d} players | "
                f"{matched_count:2d} matched ({pct:.0f}%)"
            )

        session.commit()
        print(
            f"\nTotal: {stats['total']} players | "
            f"{stats['matched']} matched | {stats['new']} new seeded"
        )

    return rosters


def main():
    print("=" * 60)
    print("  WC2026 Squad Parser + DB Seeder")
    print("=" * 60)

    if not SOURCE_FILE.exists():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        return

    print("\n[Step 1] Parsing squads from Sky Sports text...")
    squads = parse_squads_from_text(SOURCE_FILE)
    total_players = sum(len(v) for v in squads.values())
    print(f"  Parsed {len(squads)} teams, {total_players} players")

    squads_path = _PROJECT_ROOT / "data" / "wc2026_squads.json"
    with open(squads_path, "w", encoding="utf-8") as f:
        json.dump(squads, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {squads_path}")

    print(f"\n[Step 2] Matching players to DB + seeding new ones...")
    rosters = match_and_seed(squads)

    rosters_path = _PROJECT_ROOT / "data" / "wc2026_rosters.json"
    with open(rosters_path, "w", encoding="utf-8") as f:
        json.dump(rosters, f, indent=2, ensure_ascii=False)
    print(f"\n[Step 3] Saved rosters to {rosters_path}")

    matched = sum(
        1 for team in rosters.values()
        for p in team if p["match_status"] == "matched"
    )
    total = sum(len(team) for team in rosters.values())
    print(f"\nFinal: {matched}/{total} players matched to existing DB records")
    print(f"Coverage: {matched/total*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
