"""
Parse TM national team HTML file -> save JSON with caps/goals.
Usage: python scripts/_extract_teams.py <html_file> <output_json>
"""
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

POSITIONS = [
    "Goalkeeper", "Centre-Back", "Left-Back", "Right-Back",
    "Defensive Midfield", "Central Midfield", "Attacking Midfield",
    "Left Winger", "Right Winger", "Second Striker", "Centre-Forward",
]


def parse_tm_squad_html(html_path):
    html = open(html_path, "r", encoding="utf-8").read()
    tables = pd.read_html(io.StringIO(html))
    if len(tables) < 2:
        print("ERROR: no squad table found")
        return []
    df = tables[1]
    valid = df[df["International matches"].notna()].copy()
    valid = valid[valid["International matches"].apply(lambda x: str(x).strip().isdigit())]
    players = []
    for _, row in valid.iterrows():
        name = str(row["Player"])
        for pos in POSITIONS:
            name = name.split(pos)[0]
        name = name.strip()
        caps = int(row["International matches"])
        goals_raw = str(row["Goals"]).strip()
        goals = int(goals_raw) if goals_raw.isdigit() else 0
        players.append({"name": name, "caps": caps, "goals": goals})
    return players


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python _extract_teams.py <html_file> <output_json>")
        sys.exit(1)
    html_file = sys.argv[1]
    out_file = sys.argv[2]
    players = parse_tm_squad_html(html_file)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(players, f, indent=2, ensure_ascii=False)
    print("Saved {} players to {}".format(len(players), out_file))
