"""
Underdog upset capture: when Elo gap is large, did the model pick the weaker side to WIN?

黑马捕捉：悬殊局里，模型是否押中弱队「赢球」（非平局）。

Usage:
    python scripts/underdog_upset_analysis.py
    python scripts/underdog_upset_analysis.py --min-gap 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features.match_dataset import build_match_dataset
from extractors.elo_scraper import get_pre_tournament_elo
from models.match_predictor import LABEL_MAP, _elo_predict


def _build_frame() -> pd.DataFrame:
    X, y, meta = build_match_dataset()
    fn = X.columns.tolist()
    Xs = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=fn)
    Xs = Xs.replace([np.inf, -np.inf], 0)
    Xs = pd.DataFrame(StandardScaler().fit_transform(Xs), columns=fn)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    y_pred = cross_val_predict(rf, Xs, y, cv=LeaveOneOut())

    elo_map = {
        "FIFA World Cup 2018": "wc2018",
        "UEFA Euro 2020": "euro2020",
        "FIFA World Cup 2022": "wc2022",
    }
    caches = {k: get_pre_tournament_elo(v) for k, v in elo_map.items()}

    rows = []
    for i in range(len(y)):
        h = meta.iloc[i]["home_team"]
        a = meta.iloc[i]["away_team"]
        t = meta.iloc[i]["tournament"]
        ed = caches[t]
        eh = ed.get(h, {}).get("elo", 1500)
        ea = ed.get(a, {}).get("elo", 1500)
        gap = abs(eh - ea)
        if eh >= ea:
            favorite, underdog = h, a
            underdog_side = "away"
        else:
            favorite, underdog = a, h
            underdog_side = "home"

        act = int(y.iloc[i])
        pr = int(y_pred[i])
        elo_p, _ = _elo_predict(h, a, ed)

        # Underdog WINS in 90 min (strict upset — not draw)
        underdog_won = (act == 1 and underdog_side == "home") or (
            act == -1 and underdog_side == "away"
        )
        model_picks_underdog_win = (pr == 1 and underdog_side == "home") or (
            pr == -1 and underdog_side == "away"
        )
        elo_picks_underdog_win = (elo_p == 1 and underdog_side == "home") or (
            elo_p == -1 and underdog_side == "away"
        )
        model_picks_favorite_win = (pr == 1 and favorite == h) or (
            pr == -1 and favorite == a
        )

        rows.append({
            "match": f"{h} vs {a}",
            "tournament": t,
            "stage": meta.iloc[i].get("stage", ""),
            "elo_gap": gap,
            "favorite": favorite,
            "underdog": underdog,
            "score": f"{int(meta.iloc[i]['home_score'])}-{int(meta.iloc[i]['away_score'])}",
            "actual": LABEL_MAP[act],
            "model_pred": LABEL_MAP[pr],
            "elo_pred": LABEL_MAP[elo_p],
            "underdog_won": underdog_won,
            "model_picks_underdog_win": model_picks_underdog_win,
            "elo_picks_underdog_win": elo_picks_underdog_win,
            "dark_horse_hit": underdog_won and model_picks_underdog_win,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gap", type=int, default=200, help="Min |Elo_home - Elo_away|")
    args = parser.parse_args()

    df = _build_frame()
    lopsided = df[df["elo_gap"] >= args.min_gap].copy()

    n_lop = len(lopsided)
    n_upset = int(lopsided["underdog_won"].sum())
    n_hit = int(lopsided["dark_horse_hit"].sum())
    n_model_pick_dog = int(lopsided["model_picks_underdog_win"].sum())
    n_elo_pick_dog = int(lopsided["elo_picks_underdog_win"].sum())

    print("=" * 72)
    print(f"  Underdog WIN capture (Elo gap >= {args.min_gap})")
    print("  Definition: weaker team wins in 90 min — draws do NOT count as upsets")
    print("=" * 72)

    print(f"\n  Lopsided matches: {n_lop}")
    print(f"  Underdog actually won: {n_upset}")

    if n_upset:
        recall = n_hit / n_upset
        print(
            f"\n  [A] Upset recall — when underdog won, model also picked underdog win:\n"
            f"      {n_hit} / {n_upset} = {recall:.1%}  <<< true dark-horse capture"
        )
    else:
        print("\n  [A] No underdog wins in this tier.")

    if n_model_pick_dog:
        precision = n_hit / n_model_pick_dog
        print(
            f"\n  [B] Upset precision — when model picked underdog win (lopsided only):\n"
            f"      {n_hit} / {n_model_pick_dog} = {precision:.1%}"
        )
    else:
        print("\n  [B] Model never picked underdog to win in lopsided matches.")

    if n_elo_pick_dog:
        elo_hits = int(lopsided["elo_picks_underdog_win"].sum())
        print(f"\n  [C] Elo baseline picked underdog win: {elo_hits} times (lopsided)")

    # Contrast: picking favorite on lopsided
    fav_won = lopsided.apply(
        lambda r: (r["actual"] == "Home Win" and r["favorite"] == r["match"].split(" vs ")[0])
        or (r["actual"] == "Away Win" and r["favorite"] == r["match"].split(" vs ")[1]),
        axis=1,
    )
    # simpler from stored logic
    fav_won_mask = []
    for _, r in lopsided.iterrows():
        h, a = r["match"].split(" vs ")
        fw = (r["actual"] == "Home Win" and r["favorite"] == h) or (
            r["actual"] == "Away Win" and r["favorite"] == a
        )
        fav_won_mask.append(fw)
    fav_won_n = sum(fav_won_mask)
    model_fav_when_fav_won = 0
    for (_, r), fw in zip(lopsided.iterrows(), fav_won_mask):
        if not fw:
            continue
        h, a = r["match"].split(" vs ")
        mp = (r["model_pred"] == "Home Win" and r["favorite"] == h) or (
            r["model_pred"] == "Away Win" and r["favorite"] == a
        )
        if mp:
            model_fav_when_fav_won += 1
    print(
        f"\n  [Contrast] Favorite won {fav_won_n} times; model picked favorite win "
        f"{model_fav_when_fav_won}/{fav_won_n} ({model_fav_when_fav_won/max(fav_won_n,1):.0%})"
        " — this is 'following the market', not dark horses"
    )

    print("\n" + "-" * 72)
    print("  TRUE DARK HORSES — lopsided + model picked underdog WIN + underdog won")
    print("-" * 72)
    hits = lopsided[lopsided["dark_horse_hit"]].sort_values("elo_gap", ascending=False)
    if hits.empty:
        print("  (none)")
    else:
        for _, r in hits.iterrows():
            print(
                f"  gap={r['elo_gap']:.0f} | {r['match']} ({r['score']}) | "
                f"underdog={r['underdog']} | {r['tournament']} {r['stage']}"
            )

    print("\n" + "-" * 72)
    print("  MISSED UPSETS — underdog won but model did NOT pick underdog win")
    print("-" * 72)
    missed = lopsided[lopsided["underdog_won"] & ~lopsided["model_picks_underdog_win"]]
    missed = missed.sort_values("elo_gap", ascending=False)
    for _, r in missed.iterrows():
        print(
            f"  gap={r['elo_gap']:.0f} | {r['match']} ({r['score']}) | "
            f"underdog={r['underdog']} won | model={r['model_pred']} | "
            f"{r['tournament']} {r['stage']}"
        )

    print("\n" + "-" * 72)
    print("  FALSE ALARMS — lopsided + model picked underdog win but underdog lost/drew")
    print("-" * 72)
    false_alarms = lopsided[
        lopsided["model_picks_underdog_win"] & ~lopsided["underdog_won"]
    ].sort_values("elo_gap", ascending=False)
    for _, r in false_alarms.iterrows():
        print(
            f"  gap={r['elo_gap']:.0f} | {r['match']} ({r['score']}) | "
            f"underdog={r['underdog']} | actual={r['actual']} | "
            f"{r['tournament']} {r['stage']}"
        )

    print("\n" + "=" * 72)


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    main()
