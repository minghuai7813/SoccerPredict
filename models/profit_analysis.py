"""
Profitability analysis: upset capture rate + simulated betting ROI.
盈利分析：爆冷捕捉率 + 模拟投注 ROI。

Key insight / 核心洞察:
  Betting is profitable when you bet on outcomes where the market
  UNDERESTIMATES the probability. Specifically:
  - Underdog wins: odds are high (3x-10x payout)
  - Underdog draws vs favorite: odds are moderate (2x-4x)
  Even 30% accuracy on these is profitable if odds are 4:1+.
  投注盈利的关键不在于整体准确率，而在于：
  在市场低估的场次下注。弱队胜赔率高（3-10倍），
  即使只猜中30%也能盈利。

Usage / 用法:
    python models/profit_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LABEL_MAP = {1: "Home Win", 0: "Draw", -1: "Away Win"}


def _elo_win_prob(elo_a: int, elo_b: int) -> float:
    """P(A wins) based on Elo difference."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _implied_odds(prob: float) -> float:
    """Convert probability to decimal odds (European format)."""
    return 1.0 / max(prob, 0.01)


def run():
    from features.match_dataset import build_match_dataset
    from extractors.elo_scraper import get_pre_tournament_elo

    print("=" * 70)
    print("  PROJECT ORACLE — Profitability & Upset Analysis")
    print("=" * 70)

    X, y, meta = build_match_dataset()
    feature_names = X.columns.tolist()
    n = len(X)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)
    X_imp = X_imp.replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=feature_names)

    loo = LeaveOneOut()
    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                min_samples_leaf=4, random_state=42,
                                class_weight="balanced")
    y_pred = cross_val_predict(rf, X_scaled, y, cv=loo)
    y_proba = cross_val_predict(rf, X_scaled, y, cv=loo, method="predict_proba")

    # Elo data per tournament
    elo_map = {"FIFA World Cup 2018": "wc2018", "UEFA Euro 2020": "euro2020",
               "FIFA World Cup 2022": "wc2022"}
    elo_caches = {k: get_pre_tournament_elo(v) for k, v in elo_map.items()}

    # Build analysis dataframe
    rows = []
    for i in range(n):
        home = meta.iloc[i]["home_team"]
        away = meta.iloc[i]["away_team"]
        tournament = meta.iloc[i]["tournament"]
        elo_data = elo_caches[tournament]
        elo_h = elo_data.get(home, {}).get("elo", 1500)
        elo_a = elo_data.get(away, {}).get("elo", 1500)
        elo_diff = elo_h - elo_a

        # Determine favorite and underdog
        if elo_h > elo_a:
            favorite, underdog = home, away
            fav_elo, dog_elo = elo_h, elo_a
        else:
            favorite, underdog = away, home
            fav_elo, dog_elo = elo_a, elo_h

        fav_prob = _elo_win_prob(fav_elo, dog_elo)

        actual = y.iloc[i]
        actual_is_fav_win = (actual == 1 and favorite == home) or (actual == -1 and favorite == away)
        actual_is_draw = (actual == 0)
        actual_is_upset = not actual_is_fav_win and not actual_is_draw

        pred = y_pred[i]
        pred_is_fav_win = (pred == 1 and favorite == home) or (pred == -1 and favorite == away)
        pred_is_draw = (pred == 0)
        pred_is_upset = not pred_is_fav_win and not pred_is_draw

        # RF class probabilities (classes are sorted: -1, 0, 1)
        rf_classes = rf.classes_ if hasattr(rf, 'classes_') else [-1, 0, 1]

        rows.append({
            "match": f"{home} vs {away}",
            "home": home, "away": away,
            "tournament": tournament.split()[-1],
            "favorite": favorite, "underdog": underdog,
            "elo_diff": abs(elo_diff),
            "fav_prob": fav_prob,
            "actual": LABEL_MAP[actual],
            "actual_code": actual,
            "actual_is_fav_win": actual_is_fav_win,
            "actual_is_draw": actual_is_draw,
            "actual_is_upset": actual_is_upset,
            "pred": LABEL_MAP[pred],
            "pred_is_upset": pred_is_upset,
            "pred_is_draw": pred_is_draw,
            "correct": actual == pred,
        })

    df = pd.DataFrame(rows)

    # ================================================================
    # 1. UPSET BREAKDOWN
    # ================================================================
    print("\n" + "=" * 70)
    print("[1] Match Outcome Breakdown (by Elo favorite)")
    print("=" * 70)

    n_fav = df["actual_is_fav_win"].sum()
    n_draw = df["actual_is_draw"].sum()
    n_upset = df["actual_is_upset"].sum()

    print(f"\n  Favorite wins: {n_fav}/{n} ({n_fav/n:.1%})")
    print(f"  Draws:         {n_draw}/{n} ({n_draw/n:.1%})")
    print(f"  Upsets:        {n_upset}/{n} ({n_upset/n:.1%})")

    # ================================================================
    # 2. UPSET CAPTURE by Elo gap tier
    # ================================================================
    print("\n" + "=" * 70)
    print("[2] Upset Capture Rate by Elo Gap")
    print("=" * 70)

    tiers = [
        ("Small gap (Elo <100)", df["elo_diff"] < 100),
        ("Medium gap (100-200)", (df["elo_diff"] >= 100) & (df["elo_diff"] < 200)),
        ("Large gap (200+)", df["elo_diff"] >= 200),
    ]

    print(f"\n  {'Tier':<30s} {'Matches':>8s} {'Upsets':>8s} {'We caught':>10s} {'Rate':>6s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10} {'-'*6}")

    for tier_name, mask in tiers:
        tier_df = df[mask]
        t_total = len(tier_df)
        t_upsets = tier_df["actual_is_upset"].sum()
        t_caught = ((tier_df["actual_is_upset"]) & (tier_df["correct"])).sum()
        t_rate = t_caught / t_upsets if t_upsets > 0 else 0
        print(f"  {tier_name:<30s} {t_total:>8d} {t_upsets:>8d} {t_caught:>10d} {t_rate:>5.0%}")

    # Also: draw capture when favorite expected to win
    print(f"\n  Draw capture (favorite expected to win, actual was draw):")
    for tier_name, mask in tiers:
        tier_df = df[mask]
        t_draws = tier_df["actual_is_draw"].sum()
        t_pred_draw = ((tier_df["actual_is_draw"]) & (tier_df["pred_is_draw"])).sum()
        rate = t_pred_draw / t_draws if t_draws > 0 else 0
        print(f"    {tier_name:<30s} {t_draws} draws, caught {t_pred_draw} ({rate:.0%})")

    # ================================================================
    # 3. SPECIFIC UPSETS LIST
    # ================================================================
    print("\n" + "=" * 70)
    print("[3] All Actual Upsets — Did We Predict Them?")
    print("=" * 70)

    upsets = df[df["actual_is_upset"]].sort_values("elo_diff", ascending=False)
    print(f"\n  {'Match':<35s} {'Elo gap':>8s} {'Fav%':>6s} {'Actual':<10s} {'Our pred':<10s} {'Hit':>4s}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*4}")

    for _, r in upsets.iterrows():
        hit = "Y" if r["correct"] else ""
        print(f"  {r['match']:<35s} {r['elo_diff']:>8.0f} {r['fav_prob']:>5.0%} {r['actual']:<10s} {r['pred']:<10s} {hit:>4s}")

    # All draws where a clear favorite existed
    print(f"\n  Draws (with clear favorite, Elo gap > 50):")
    draws = df[(df["actual_is_draw"]) & (df["elo_diff"] > 50)].sort_values("elo_diff", ascending=False)
    print(f"  {'Match':<35s} {'Elo gap':>8s} {'Fav%':>6s} {'Our pred':<10s} {'Hit':>4s}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*10} {'-'*4}")
    for _, r in draws.iterrows():
        hit = "Y" if r["correct"] else ""
        print(f"  {r['match']:<35s} {r['elo_diff']:>8.0f} {r['fav_prob']:>5.0%} {r['pred']:<10s} {hit:>4s}")

    # ================================================================
    # 4. SIMULATED BETTING (using Elo-implied fair odds)
    # ================================================================
    print("\n" + "=" * 70)
    print("[4] Simulated Betting (Elo-implied odds)")
    print("=" * 70)

    # Use Elo to generate "market odds" (simplified).
    # Real bookmaker odds include margin, but this is a fair approximation.
    # 用 Elo 生成"市场赔率"（简化版，不含庄家利润率）。

    strategies = []

    # Strategy A: Bet on underdog when model predicts upset
    pnl_a = 0
    bets_a = 0
    wins_a = 0
    details_a = []
    for _, r in df.iterrows():
        if r["pred_is_upset"]:
            bets_a += 1
            dog_win_prob = 1 - r["fav_prob"]
            odds = _implied_odds(dog_win_prob)
            if r["actual_is_upset"]:
                profit = odds - 1
                pnl_a += profit
                wins_a += 1
                details_a.append((r["match"], f"+{profit:.1f}", odds, r["tournament"]))
            else:
                pnl_a -= 1
                details_a.append((r["match"], "-1.0", odds, r["tournament"]))

    roi_a = pnl_a / bets_a * 100 if bets_a > 0 else 0
    strategies.append(("A: Bet underdog win", bets_a, wins_a, pnl_a, roi_a))

    # Strategy B: Bet on draw when model predicts draw AND favorite exists
    pnl_b = 0
    bets_b = 0
    wins_b = 0
    for _, r in df.iterrows():
        if r["pred_is_draw"] and r["elo_diff"] > 50:
            bets_b += 1
            draw_prob_est = 0.25
            odds = _implied_odds(draw_prob_est)
            if r["actual_is_draw"]:
                profit = odds - 1
                pnl_b += profit
                wins_b += 1
            else:
                pnl_b -= 1

    roi_b = pnl_b / bets_b * 100 if bets_b > 0 else 0
    strategies.append(("B: Bet draw (fav exists)", bets_b, wins_b, pnl_b, roi_b))

    # Strategy C: Bet underdog OR draw when model disagrees with Elo favorite
    pnl_c = 0
    bets_c = 0
    wins_c = 0
    for _, r in df.iterrows():
        if r["pred_is_upset"] or (r["pred_is_draw"] and r["elo_diff"] > 80):
            bets_c += 1
            if r["pred_is_upset"]:
                dog_prob = 1 - r["fav_prob"]
                odds = _implied_odds(dog_prob)
                won = r["actual_is_upset"]
            else:
                odds = _implied_odds(0.25)
                won = r["actual_is_draw"]

            if won:
                pnl_c += odds - 1
                wins_c += 1
            else:
                pnl_c -= 1

    roi_c = pnl_c / bets_c * 100 if bets_c > 0 else 0
    strategies.append(("C: Upset OR draw", bets_c, wins_c, pnl_c, roi_c))

    print(f"\n  {'Strategy':<30s} {'Bets':>6s} {'Wins':>6s} {'Win%':>7s} {'P&L':>8s} {'ROI':>8s}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*8}")
    for name, bets, wins, pnl, roi in strategies:
        w_pct = wins / bets * 100 if bets > 0 else 0
        print(f"  {name:<30s} {bets:>6d} {wins:>6d} {w_pct:>6.1f}% {pnl:>+7.1f} {roi:>+7.1f}%")

    # Show details for strategy A
    if details_a:
        print(f"\n  Strategy A detail (underdog win bets):")
        print(f"  {'Match':<35s} {'Odds':>6s} {'P&L':>7s} {'Tourn':>6s}")
        print(f"  {'-'*35} {'-'*6} {'-'*7} {'-'*6}")
        for match, pl, odds, tourn in sorted(details_a, key=lambda x: -float(x[1])):
            print(f"  {match:<35s} {odds:>5.1f}x {pl:>7s} {tourn:>6s}")

    # ================================================================
    # 5. KEY TAKEAWAY
    # ================================================================
    print("\n" + "=" * 70)
    print("[5] Key Takeaway")
    print("=" * 70)

    best = max(strategies, key=lambda x: x[4])
    print(f"\n  Best strategy: {best[0]}")
    print(f"  {best[1]} bets, {best[2]} wins ({best[2]/best[1]*100:.1f}%), ROI: {best[4]:+.1f}%")

    print(f"\n  To be profitable in real betting:")
    print(f"    1. Need actual bookmaker odds (not Elo-implied)")
    print(f"    2. Need to beat the ~5-8% bookmaker margin")
    print(f"    3. Focus on high-value bets: large Elo gap upsets")
    print(f"    4. Consider Asian Handicap and Over/Under markets")

    print("\n" + "=" * 70)
    print("  Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    run()
