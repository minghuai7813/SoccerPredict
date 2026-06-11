"""
Batch group-stage predictions for FIFA World Cup 2026.
WC2026 group-stage batch prediction via Elo + Poisson.

Usage:
    python -m scripts.predict_wc2026_groups
    python -m scripts.predict_wc2026_groups --matchday 1
    python -m scripts.predict_wc2026_groups --group A
    python -m scripts.predict_wc2026_groups --standings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extractors.elo_scraper import get_pre_tournament_elo
from models.market_odds import (
    decimal_odds,
    elo_1x2_probs,
    expected_goals_from_elo,
    poisson_score_matrix,
)

SCHEDULE_PATH = _PROJECT_ROOT / "data" / "wc2026_schedule.json"
LABEL = {1: "Home", 0: "Draw", -1: "Away"}


def load_schedule() -> dict:
    with open(SCHEDULE_PATH, encoding="utf-8") as f:
        return json.load(f)


def predict_match_elo(
    home: str, away: str, elo_data: dict, *, home_adv: float = 0.0,
) -> dict:
    h_info = elo_data.get(home, {"elo": 1500, "rank": 99})
    a_info = elo_data.get(away, {"elo": 1500, "rank": 99})
    h_elo, a_elo = h_info["elo"], a_info["elo"]

    probs = elo_1x2_probs(h_elo, a_elo, home_adv=home_adv)
    lam_h, lam_a = expected_goals_from_elo(
        h_elo, a_elo, home_adv=home_adv, total_goals=2.55,
    )
    mat = poisson_score_matrix(lam_h, lam_a, max_g=6)

    scored = [
        (mat[i, j], i, j)
        for i in range(mat.shape[0])
        for j in range(mat.shape[1])
    ]
    scored.sort(reverse=True)
    top5 = [(h, a, float(p)) for p, h, a in scored[:5]]

    pick = max(probs, key=probs.get)
    return {
        "home": home,
        "away": away,
        "h_elo": h_elo,
        "a_elo": a_elo,
        "probs": probs,
        "pick": pick,
        "lambdas": (lam_h, lam_a),
        "top_scores": top5,
    }


def _sim_group(group_name, matches, elo_data):
    schedule = load_schedule()
    teams_in_group = schedule["groups"][group_name]
    table = {
        t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "w": 0, "d": 0, "l": 0}
        for t in teams_in_group
    }
    for m in matches:
        pred = predict_match_elo(m["home"], m["away"], elo_data)
        lh, la = pred["lambdas"]
        exp_h, exp_a = round(lh), round(la)
        home, away = m["home"], m["away"]
        if home not in table or away not in table:
            continue
        table[home]["gf"] += exp_h
        table[home]["ga"] += exp_a
        table[away]["gf"] += exp_a
        table[away]["ga"] += exp_h
        if pred["pick"] == 1:
            table[home]["pts"] += 3
            table[home]["w"] += 1
            table[away]["l"] += 1
        elif pred["pick"] == -1:
            table[away]["pts"] += 3
            table[away]["w"] += 1
            table[home]["l"] += 1
        else:
            table[home]["pts"] += 1
            table[away]["pts"] += 1
            table[home]["d"] += 1
            table[away]["d"] += 1

    for t in table:
        table[t]["gd"] = table[t]["gf"] - table[t]["ga"]

    standings = [
        {"team": t, **v, "elo": elo_data.get(t, {}).get("elo", 0)}
        for t, v in table.items()
    ]
    standings.sort(
        key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True,
    )
    return standings


def main():
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    parser = argparse.ArgumentParser(
        description="WC2026 group-stage predictions",
    )
    parser.add_argument(
        "--matchday", type=int, default=None, choices=[1, 2, 3],
    )
    parser.add_argument("--group", default=None)
    parser.add_argument(
        "--standings", action="store_true",
        help="Show predicted group standings",
    )
    args = parser.parse_args()

    schedule = load_schedule()
    elo_data = get_pre_tournament_elo("wc2026")
    matches = schedule["group_stage_matches"]

    if args.matchday:
        matches = [m for m in matches if m["matchday"] == args.matchday]
    if args.group:
        g = args.group.upper()
        matches = [m for m in matches if m["group"] == g]

    if not matches:
        print("No matches found for the given filter.")
        return

    current_date = ""
    for m in matches:
        if m["date"] != current_date:
            current_date = m["date"]
            print("")
            print("=" * 72)
            print("  " + current_date)
            print("=" * 72)

        pred = predict_match_elo(m["home"], m["away"], elo_data)
        prob_h = pred["probs"][1]
        prob_d = pred["probs"][0]
        prob_a = pred["probs"][-1]
        lh, la = pred["lambdas"]
        top1 = pred["top_scores"][0]
        pick = LABEL[pred["pick"]]

        elo_diff = pred["h_elo"] - pred["a_elo"]
        if abs(elo_diff) > 200:
            arrow = ">>>"
        elif abs(elo_diff) > 100:
            arrow = ">>"
        else:
            arrow = ">"

        print("")
        print(
            "  Group %s | %s vs %s  (%s)"
            % (m["group"], m["home"], m["away"], m["venue"])
        )
        print(
            "    Elo: %d vs %d  (diff %+d)"
            % (pred["h_elo"], pred["a_elo"], elo_diff)
        )
        print(
            "    W %5.1f%%  D %5.1f%%  L %5.1f%%  %s %s"
            % (prob_h * 100, prob_d * 100, prob_a * 100, arrow, pick)
        )
        print(
            "    xG: %.2f - %.2f  |  Most likely: %d-%d (%.1f%%)"
            % (lh, la, top1[0], top1[1], top1[2] * 100)
        )

    if args.standings or (args.group and not args.matchday):
        if args.group:
            groups_to_show = [args.group.upper()]
        else:
            groups_to_show = sorted(schedule["groups"].keys())
        all_matches = schedule["group_stage_matches"]

        print("")
        print("")
        print("=" * 72)
        print("  PREDICTED GROUP STANDINGS")
        print("=" * 72)

        for g in groups_to_show:
            g_matches = [m for m in all_matches if m["group"] == g]
            standings = _sim_group(g, g_matches, elo_data)
            print("")
            print("  Group " + g)
            print(
                "  %-28s %3s %2s %2s %2s %3s %5s"
                % ("Team", "Pts", "W", "D", "L", "GD", "Elo")
            )
            print(
                "  %s %s %s %s %s %s %s"
                % ("-" * 28, "-" * 3, "-" * 2, "-" * 2, "-" * 2, "-" * 3, "-" * 5)
            )
            for i, row in enumerate(standings):
                marker = " Q" if i < 2 else "  "
                print(
                    "%s%-27s %3d %2d %2d %2d %+3d %5d"
                    % (
                        marker, row["team"], row["pts"],
                        row["w"], row["d"], row["l"],
                        row["gd"], row["elo"],
                    )
                )


if __name__ == "__main__":
    main()
