"""
ROI-focused backtest for lopsided matches (Elo gap >= 200).
以盈利为目标：悬殊局押冷门 / 平局 / 弱队赢 / 波胆（泊松近似赔率）。

Usage:
    python -m models.upset_roi
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extractors.elo_scraper import get_pre_tournament_elo
from extractors.odds_loader import attach_odds_to_meta, load_match_odds
from features.match_dataset import build_match_dataset
from features.upset_dataset import MIN_ELO_GAP, annotate_elo_and_upset, build_upset_dataset


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    fn = X.columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=fn).replace([np.inf, -np.inf], 0)
    return pd.DataFrame(StandardScaler().fit_transform(X_imp), columns=fn)


def _fav_code(home: str, away: str, favorite: str) -> int:
    return 1 if favorite == home else -1


def _dog_code(home: str, away: str, favorite: str) -> int:
    return -1 if favorite == home else 1


def _odds(row: pd.Series, code: int) -> float:
    if code == 1:
        return float(row["odds_home"])
    if code == 0:
        return float(row["odds_draw"])
    return float(row["odds_away"])


def _actual_code(h: int, a: int) -> int:
    if h > a:
        return 1
    if h < a:
        return -1
    return 0


def _pnl(odds: float, won: bool) -> float:
    return (odds - 1.0) if won else -1.0


def _best_non_fav(proba_row: np.ndarray, classes: list, home: str, away: str, fav: str) -> int:
    idx = {int(c): i for i, c in enumerate(classes)}
    dog = _dog_code(home, away, fav)
    cands = []
    if 0 in idx:
        cands.append((proba_row[idx[0]], 0))
    if dog in idx:
        cands.append((proba_row[idx[dog]], dog))
    return max(cands, key=lambda x: x[0])[1]


def _cs_odds(lam_h: float, lam_a: float, sh: int, sa: int) -> float:
    p = max(poisson.pmf(sh, lam_h) * poisson.pmf(sa, lam_a), 1e-4)
    return min((1.0 - 0.12) / p, 250.0)


def _stats(rows: list[dict]) -> dict:
    if not rows:
        return {"bets": 0, "wins": 0, "win_pct": 0.0, "roi_pct": 0.0, "pnl": 0.0, "avg_odds": 0.0}
    pnl = sum(r["pnl"] for r in rows)
    w = sum(1 for r in rows if r["won"])
    return {
        "bets": len(rows),
        "wins": w,
        "win_pct": w / len(rows),
        "roi_pct": pnl / len(rows) * 100,
        "pnl": pnl,
        "avg_odds": float(np.mean([r["odds"] for r in rows])),
    }


def _print_row(name: str, s: dict) -> None:
    if s["bets"] == 0:
        print(f"  {name:<42s}  (no bets)")
        return
    print(
        f"  {name:<42s}  {s['bets']:>4d} bets | win {s['win_pct']:>5.1%} | "
        f"avg odds {s['avg_odds']:>5.1f}x | ROI {s['roi_pct']:>+7.1f}% | P&L {s['pnl']:>+6.1f}u"
    )


def run() -> None:
    print("=" * 72)
    print("  PROJECT ORACLE — Upset ROI (profit-first, lopsided only)")
    print(f"  Elo gap >= {MIN_ELO_GAP} | flat 1u | LOO | odds from data/odds/")
    print("=" * 72)

    X, y, meta = build_match_dataset()
    enriched = annotate_elo_and_upset(meta, y)
    mask = enriched["is_lopsided"].values

    elo_map = {
        "FIFA World Cup 2018": "wc2018",
        "UEFA Euro 2020": "euro2020",
        "FIFA World Cup 2022": "wc2022",
    }
    caches = {k: get_pre_tournament_elo(v) for k, v in elo_map.items()}
    meta_o = attach_odds_to_meta(meta, load_match_odds(), caches)

    Xs = _prep(X)
    loo = LeaveOneOut()

    rf_main = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    rf_main.fit(Xs, y)
    classes = list(rf_main.classes_)
    main_proba = cross_val_predict(rf_main, Xs, y, cv=loo, method="predict_proba")

    X_u, y_u, _ = build_upset_dataset(verbose=False)
    rf_up = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=3,
        random_state=42, class_weight="balanced",
    )
    Xsu = _prep(X_u)
    p_upset_u = cross_val_predict(rf_up, Xsu, y_u, cv=loo, method="predict_proba")[:, 1]

    fn = X.columns.tolist()
    X_pos = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X), columns=fn,
    ).replace([np.inf, -np.inf], 0).clip(lower=0)
    pred_h = np.clip(
        cross_val_predict(PoissonRegressor(alpha=1.0, max_iter=1000), X_pos, meta["home_score"], cv=loo),
        0.05, 6.0,
    )
    pred_a = np.clip(
        cross_val_predict(PoissonRegressor(alpha=1.0, max_iter=1000), X_pos, meta["away_score"], cv=loo),
        0.05, 6.0,
    )

    p_upset = np.zeros(len(y))
    j = 0
    for i in range(len(y)):
        if mask[i]:
            p_upset[i] = p_upset_u[j]
            j += 1

    # Collect lopsided match rows
    lop_idx = [i for i in range(len(y)) if mask[i]]
    print(f"\n  Lopsided matches: {len(lop_idx)} | real upsets: {int(enriched.loc[mask, 'upset'].sum())}")

    strategies: dict[str, list[dict]] = {
        "A: Always bet FAVORITE (low odds)": [],
        "B: Always bet UNDERDOG win only": [],
        "C: Always bet DRAW only": [],
        "D: Always bet best non-fav (draw vs dog)": [],
    }
    for tau in (0.40, 0.45, 0.50, 0.55):
        strategies[f"E: P(upset)>={tau:.2f} -> best non-fav"] = []
        strategies[f"F: P(upset)>={tau:.2f} -> UNDERDOG only"] = []
        strategies[f"G: P(upset)>={tau:.2f} -> DRAW only"] = []
    strategies["H: P(upset)>=0.45 + value edge>=10%"] = []
    strategies["I: Correct score (Poisson top score, all lop)"] = []
    strategies["J: CS when P(upset)>=0.45 & model CS prob>=6%"] = []

    for i in lop_idx:
        row = meta_o.iloc[i]
        enr = enriched.iloc[i]
        home, away = row["home_team"], row["away_team"]
        fav = enr["favorite"]
        h, a = int(row["home_score"]), int(row["away_score"])
        actual = _actual_code(h, a)
        fav_c = _fav_code(home, away, fav)
        dog_c = _dog_code(home, away, fav)
        pr = main_proba[i]
        best_nf = _best_non_fav(pr, classes, home, away, fav)

        # A favorite
        o = _odds(row, fav_c)
        strategies["A: Always bet FAVORITE (low odds)"].append({
            "pnl": _pnl(o, actual == fav_c), "won": actual == fav_c, "odds": o,
        })

        # B underdog
        o = _odds(row, dog_c)
        strategies["B: Always bet UNDERDOG win only"].append({
            "pnl": _pnl(o, actual == dog_c), "won": actual == dog_c, "odds": o,
        })

        # C draw
        o = _odds(row, 0)
        strategies["C: Always bet DRAW only"].append({
            "pnl": _pnl(o, actual == 0), "won": actual == 0, "odds": o,
        })

        # D best non-fav always
        o = _odds(row, best_nf)
        strategies["D: Always bet best non-fav (draw vs dog)"].append({
            "pnl": _pnl(o, actual == best_nf), "won": actual == best_nf, "odds": o,
        })

        sh = int(round(pred_h[i]))
        sa = int(round(pred_a[i]))
        cs_o = _cs_odds(pred_h[i], pred_a[i], sh, sa)
        strategies["I: Correct score (Poisson top score, all lop)"].append({
            "pnl": _pnl(cs_o, h == sh and a == sa),
            "won": h == sh and a == sa,
            "odds": cs_o,
        })

        for tau in (0.40, 0.45, 0.50, 0.55):
            if p_upset[i] < tau:
                continue
            o_nf = _odds(row, best_nf)
            strategies[f"E: P(upset)>={tau:.2f} -> best non-fav"].append({
                "pnl": _pnl(o_nf, actual == best_nf), "won": actual == best_nf, "odds": o_nf,
            })
            o_d = _odds(row, dog_c)
            strategies[f"F: P(upset)>={tau:.2f} -> UNDERDOG only"].append({
                "pnl": _pnl(o_d, actual == dog_c), "won": actual == dog_c, "odds": o_d,
            })
            o_dr = _odds(row, 0)
            strategies[f"G: P(upset)>={tau:.2f} -> DRAW only"].append({
                "pnl": _pnl(o_dr, actual == 0), "won": actual == 0, "odds": o_dr,
            })

        # Value edge: model prob vs implied for best non-fav
        if p_upset[i] >= 0.45:
            idx = {int(c): i for i, c in enumerate(classes)}
            model_p_nf = pr[idx[best_nf]]
            implied_nf = 1.0 / _odds(row, best_nf)
            if model_p_nf >= implied_nf * 1.10:
                o = _odds(row, best_nf)
                strategies["H: P(upset)>=0.45 + value edge>=10%"].append({
                    "pnl": _pnl(o, actual == best_nf),
                    "won": actual == best_nf,
                    "odds": o,
                })

        if p_upset[i] >= 0.45:
            p_cs = poisson.pmf(sh, pred_h[i]) * poisson.pmf(sa, pred_a[i])
            if p_cs >= 0.06 and (actual != fav_c):
                strategies["J: CS when P(upset)>=0.45 & model CS prob>=6%"].append({
                    "pnl": _pnl(cs_o, h == sh and a == sa),
                    "won": h == sh and a == sa,
                    "odds": cs_o,
                })

    print("\n" + "-" * 72)
    print("[ROI by strategy] (higher ROI = more units per bet on average)")
    print("-" * 72)
    for name in strategies:
        _print_row(name, _stats(strategies[name]))

    # Highlight big wins
    print("\n" + "-" * 72)
    print("[Big hits] E strategy tau=0.45, won bets with odds >= 4.0")
    print("-" * 72)
    key = "E: P(upset)>=0.45 -> best non-fav"
    for i in lop_idx:
        if p_upset[i] < 0.45:
            continue
        row = meta_o.iloc[i]
        enr = enriched.iloc[i]
        home, away = row["home_team"], row["away_team"]
        fav = enr["favorite"]
        actual = _actual_code(int(row["home_score"]), int(row["away_score"]))
        pick = _best_non_fav(main_proba[i], classes, home, away, fav)
        if actual != pick:
            continue
        o = _odds(row, pick)
        if o >= 4.0:
            print(
                f"  +{o-1:.1f}u  {home} vs {away} ({row['home_score']}-{row['away_score']}) "
                f"bet {('Draw' if pick==0 else 'Underdog')} @ {o:.1f}x"
            )

    print("\n" + "-" * 72)
    print("[Notes]")
    print("  - Favorite bets: low avg odds (~1.3-1.8), need high win% to profit.")
    print("  - CS odds are Poisson-fair estimates; real books differ (often lower).")
    print("  - Past LOO sim != future; use as relative comparison between strategies.")
    print("  - Your goal: prefer strategies with positive ROI + acceptable bet count.")
    print("=" * 72)


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    run()
