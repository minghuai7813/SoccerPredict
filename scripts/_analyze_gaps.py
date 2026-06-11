"""One-shot analysis of unmatched WC2026 roster players."""
import json, pathlib

data = json.loads(
    pathlib.Path(r"d:/CursorProjects/SoccerProject/data/wc2026_rosters.json")
    .read_text(encoding="utf-8")
)

low_teams = []
for team, players in data.items():
    t = len(players)
    m = sum(1 for x in players if x.get("match_status") == "matched")
    if t > 0 and m / t < 0.40:
        low_teams.append((team, t, m))

low_teams.sort(key=lambda x: x[2] / x[1])

for team, t, m in low_teams:
    unmatched = [p for p in data[team] if p.get("match_status") != "matched"]
    print(f"\n=== {team} ({m}/{t} matched, {t - m} missing) ===")
    for p in unmatched:
        nm = p["name"]
        pos = p["position"]
        cl = p.get("club", "?")
        print(f"  {nm:32s} {pos:3s}  {cl}")
