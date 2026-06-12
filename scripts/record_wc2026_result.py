"""Record WC2026 match results. Usage: python -m scripts.record_wc2026_result --list"""
from __future__ import annotations
import argparse, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.encoding import fix_console_encoding
fix_console_encoding()

RESULTS = ROOT / "data" / "wc2026_results.json"
SCHEDULE = ROOT / "data" / "wc2026_schedule.json"


def mk_key(d, g, h, a):
    return d + "|" + g + "|" + h + "|" + a


def res_code(hg, ag):
    if hg > ag:
        return 1
    if hg < ag:
        return -1
    return 0


def load_sched():
    return json.loads(SCHEDULE.read_text(encoding="utf-8"))


def load_res():
    if not RESULTS.exists():
        return {"_meta": {"tournament": "FIFA World Cup 2026"}, "results": []}
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def save_res(data):
    data.setdefault("_meta", {})
    data["_meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_fix(h, a, date=None, group=None):
    for m in load_sched().get("group_stage_matches", []):
        if m["home"] != h or m["away"] != a:
            continue
        if date and m.get("date") != date:
            continue
        if group and m.get("group") != group:
            continue
        return dict(m)
    return None


def parse_score(s):
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", s)
    if not m:
        raise ValueError("bad score: " + s)
    return int(m.group(1)), int(m.group(2))


def record(h, a, hg, ag, date=None, group=None, source=None, notes=None, overwrite=False):
    fx = find_fix(h, a, date, group)
    if fx:
        date, group = fx["date"], fx["group"]
        venue, md, ko = fx.get("venue"), fx.get("matchday"), fx.get("kickoff_et")
    else:
        if not date or not group:
            raise ValueError("fixture not in schedule")
        venue = md = ko = None
    key = mk_key(date, group, h, a)
    data = load_res()
    idx = next((i for i, r in enumerate(data["results"]) if r.get("match_key") == key), None)
    if idx is not None and not overwrite:
        raise ValueError("already recorded: " + key)
    row = {
        "match_key": key, "matchday": md, "date": date, "group": group,
        "home": h, "away": a, "venue": venue, "kickoff_et": ko,
        "status": "finished", "home_goals": hg, "away_goals": ag,
        "result": res_code(hg, ag), "goals": [], "red_cards": [],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "notes": notes,
    }
    if idx is not None:
        row["recorded_at"] = data["results"][idx].get("recorded_at", row["recorded_at"])
        data["results"][idx] = row
    else:
        data["results"].append(row)
    save_res(data)
    return row


def standings(g):
    sched = load_sched()
    teams = sched.get("groups", {}).get(g, [])
    t = {x: {"team": x, "pts": 0, "p": 0, "gf": 0, "ga": 0, "gd": 0} for x in teams}
    rec = {r["match_key"]: r for r in load_res().get("results", [])}
    for m in sched.get("group_stage_matches", []):
        if m.get("group") != g:
            continue
        r = rec.get(mk_key(m["date"], m["group"], m["home"], m["away"]))
        if not r:
            continue
        h, a, hg, ag = m["home"], m["away"], r["home_goals"], r["away_goals"]
        for s, gf, ga in ((h, hg, ag), (a, ag, hg)):
            t[s]["p"] += 1
            t[s]["gf"] += gf
            t[s]["ga"] += ga
        if hg > ag:
            t[h]["pts"] += 3
        elif hg < ag:
            t[a]["pts"] += 3
        else:
            t[h]["pts"] += 1
            t[a]["pts"] += 1
    for x in t:
        t[x]["gd"] = t[x]["gf"] - t[x]["ga"]
    return sorted(t.values(), key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--standings", metavar="G")
    p.add_argument("--home")
    p.add_argument("--away")
    p.add_argument("--score")
    p.add_argument("--date")
    p.add_argument("--group")
    p.add_argument("--source")
    p.add_argument("--notes")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.list:
        for r in sorted(load_res().get("results", []), key=lambda x: x["date"]):
            print(r["date"], "G" + r.get("group", "?"), r["home"], str(r["home_goals"]) + "-" + str(r["away_goals"]), r["away"])
        return
    if a.standings:
        for i, r in enumerate(standings(a.standings.upper()), 1):
            print(str(i) + ".", r["team"], str(r["pts"]) + "pts", str(r["gf"]) + "-" + str(r["ga"]))
        return
    if not (a.home and a.away and a.score):
        p.error("need --home --away --score")
    hg, ag = parse_score(a.score)
    row = record(a.home, a.away, hg, ag, date=a.date, group=a.group.upper() if a.group else None,
                 source=a.source, notes=a.notes, overwrite=a.overwrite)
    print("Recorded", row["home"], str(row["home_goals"]) + "-" + str(row["away_goals"]), row["away"])


if __name__ == "__main__":
    main()
