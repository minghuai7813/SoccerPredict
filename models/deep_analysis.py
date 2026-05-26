"""
Deep analysis: upset detection, error severity, over/under prediction.
深度分析：黑马检测、错误严重度、大小球预测。

Answers:
  1. Which model catches more upsets (dark horses)?
     哪个模型更能捕捉黑马？
  2. When wrong, how wrong? (near-miss vs total-miss)
     错了的话，错了多少？（差一点 vs 完全错）
  3. Can we predict over/under (total goals)?
     能不能预测大小球？

Usage / 用法:
    python models/deep_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LABEL_MAP = {1: "Home Win", 0: "Draw", -1: "Away Win"}
LABEL_TO_XGB = {-1: 0, 0: 1, 1: 2}
XGB_TO_LABEL = {0: -1, 1: 0, 2: 1}


def _elo_predict(home, away, elo_data):
    ra = elo_data.get(home, {}).get("elo", 1500)
    rb = elo_data.get(away, {}).get("elo", 1500)
    prob = 1.0 / (1.0 + 10 ** ((rb - ra - 60) / 400.0))
    if prob > 0.45:
        return 1, prob
    elif prob < 0.35:
        return -1, prob
    return 0, prob


def run():
    from features.match_dataset import build_match_dataset
    from extractors.elo_scraper import get_pre_tournament_elo

    print("=" * 70)
    print("  PROJECT ORACLE — Deep Analysis")
    print("=" * 70)

    X, y, meta = build_match_dataset()
    feature_names = X.columns.tolist()
    n = len(X)

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)
    X_imp = X_imp.replace([np.inf, -np.inf], 0)
    X_nan = X.replace([np.inf, -np.inf], np.nan)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=feature_names)

    loo = LeaveOneOut()

    # --- Build all predictions ---
    print("\n[1] Training models...")

    # RF classification
    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                min_samples_leaf=4, random_state=42,
                                class_weight="balanced")
    y_rf = cross_val_predict(rf, X_scaled, y, cv=loo)

    # XGBoost classification
    y_xgb_enc = y.map(LABEL_TO_XGB)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.7,
                         min_child_weight=4, reg_alpha=1.5, reg_lambda=3.0,
                         num_class=3, eval_metric="mlogloss",
                         random_state=42, verbosity=0)
    y_xgb = pd.Series(cross_val_predict(xgb, X_nan, y_xgb_enc, cv=loo)).map(XGB_TO_LABEL).values

    # Goal predictions (RF regressor)
    y_home_g = meta["home_score"].values.astype(float)
    y_away_g = meta["away_score"].values.astype(float)

    rf_reg_h = RandomForestRegressor(n_estimators=300, max_depth=5,
                                      min_samples_leaf=4, random_state=42)
    rf_reg_a = RandomForestRegressor(n_estimators=300, max_depth=5,
                                      min_samples_leaf=4, random_state=42)
    pred_rf_h = cross_val_predict(rf_reg_h, X_scaled, y_home_g, cv=loo)
    pred_rf_a = cross_val_predict(rf_reg_a, X_scaled, y_away_g, cv=loo)

    # XGBoost regressor for goals
    xgb_reg_h = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7,
                              min_child_weight=4, reg_alpha=1.5, reg_lambda=3.0,
                              random_state=42, verbosity=0)
    xgb_reg_a = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7,
                              min_child_weight=4, reg_alpha=1.5, reg_lambda=3.0,
                              random_state=42, verbosity=0)
    pred_xgb_h = cross_val_predict(xgb_reg_h, X_nan, y_home_g, cv=loo)
    pred_xgb_a = cross_val_predict(xgb_reg_a, X_nan, y_away_g, cv=loo)

    # Poisson regressor for goals
    X_pos = X_imp.clip(lower=0)
    poi_h = PoissonRegressor(alpha=1.0, max_iter=1000)
    poi_a = PoissonRegressor(alpha=1.0, max_iter=1000)
    pred_poi_h = np.clip(cross_val_predict(poi_h, X_pos, y_home_g, cv=loo), 0.1, 6.0)
    pred_poi_a = np.clip(cross_val_predict(poi_a, X_pos, y_away_g, cv=loo), 0.1, 6.0)

    # Elo predictions
    elo_map = {"FIFA World Cup 2018": "wc2018", "UEFA Euro 2020": "euro2020",
               "FIFA World Cup 2022": "wc2022"}
    elo_caches = {k: get_pre_tournament_elo(v) for k, v in elo_map.items()}
    elo_preds = np.array([
        _elo_predict(meta.iloc[i]["home_team"], meta.iloc[i]["away_team"],
                     elo_caches[meta.iloc[i]["tournament"]])[0]
        for i in range(n)
    ])

    actual_total = y_home_g + y_away_g

    # ==========================================================
    # ANALYSIS 1: UPSET DETECTION (which model catches dark horses?)
    # ==========================================================
    print("\n" + "=" * 70)
    print("[Analysis 1] Upset Detection — Who catches the dark horses?")
    print("=" * 70)

    # "Upset" = Elo favorite didn't win in 90 min
    elo_wrong = (y.values != elo_preds)
    n_upsets = elo_wrong.sum()

    models_clf = {
        "Random Forest": y_rf,
        "XGBoost": y_xgb,
    }

    print(f"\n  Total matches: {n}")
    print(f"  Elo wrong (upsets): {n_upsets} ({n_upsets/n:.1%})")

    for name, preds in models_clf.items():
        caught = ((y.values == preds) & elo_wrong).sum()
        print(f"  {name} caught: {caught}/{n_upsets} ({caught/n_upsets:.1%})")

    # Show specific upsets each model caught
    print(f"\n  Notable upsets detected:")
    print(f"  {'Match':<40s} {'Actual':<10s} {'Elo':<10s} {'RF':<10s} {'XGB':<10s} {'Tournament'}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")

    for i in range(n):
        if not elo_wrong[i]:
            continue
        home = meta.iloc[i]["home_team"]
        away = meta.iloc[i]["away_team"]
        match_str = f"{home} vs {away}"
        actual = LABEL_MAP[y.iloc[i]]
        elo_p = LABEL_MAP[elo_preds[i]]
        rf_p = LABEL_MAP[y_rf[i]]
        xgb_p = LABEL_MAP[y_xgb[i]]
        t = meta.iloc[i]["tournament"].split()[-1]

        rf_ok = "✓" if y.iloc[i] == y_rf[i] else ""
        xgb_ok = "✓" if y.iloc[i] == y_xgb[i] else ""

        if rf_ok or xgb_ok:
            print(f"  {match_str:<40s} {actual:<10s} {elo_p:<10s} {rf_p:<10s}{rf_ok:>1s} {xgb_p:<10s}{xgb_ok:>1s} {t}")

    # ==========================================================
    # ANALYSIS 2: ERROR SEVERITY — How wrong when wrong?
    # ==========================================================
    print("\n" + "=" * 70)
    print("[Analysis 2] Error Severity — How wrong when wrong?")
    print("=" * 70)

    # For classification: off-by-1 (e.g. predicted Draw, was Home Win) vs
    # off-by-2 (predicted Home Win, was Away Win)
    print("\n  --- Classification error severity ---")
    for name, preds in [("Elo", elo_preds), ("RF", y_rf), ("XGB", y_xgb)]:
        wrong = (y.values != preds)
        n_wrong = wrong.sum()
        if n_wrong == 0:
            continue
        diffs = np.abs(y.values[wrong] - preds[wrong])
        off_by_1 = (diffs == 1).sum()
        off_by_2 = (diffs == 2).sum()
        print(f"  {name:>5s}: {n_wrong} wrong — {off_by_1} off-by-1 (near miss), {off_by_2} off-by-2 (total miss)")

    # For goal prediction: MAE and error distribution
    print(f"\n  --- Goal prediction accuracy ---")
    print(f"  {'Model':<15s} {'MAE(H)':>8s} {'MAE(A)':>8s} {'MAE(Tot)':>8s} {'Exact H':>8s} {'Exact A':>8s} {'Exact Both':>10s}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    goal_models = [
        ("RF Regressor", pred_rf_h, pred_rf_a),
        ("XGB Regressor", pred_xgb_h, pred_xgb_a),
        ("Poisson", pred_poi_h, pred_poi_a),
    ]

    for name, ph, pa in goal_models:
        ph_r = np.round(ph).astype(int)
        pa_r = np.round(pa).astype(int)
        mae_h = mean_absolute_error(y_home_g, ph)
        mae_a = mean_absolute_error(y_away_g, pa)
        mae_t = mean_absolute_error(actual_total, ph + pa)
        exact_h = (ph_r == y_home_g).sum()
        exact_a = (pa_r == y_away_g).sum()
        exact_both = ((ph_r == y_home_g) & (pa_r == y_away_g)).sum()
        print(f"  {name:<15s} {mae_h:>8.2f} {mae_a:>8.2f} {mae_t:>8.2f} {exact_h:>7d} {exact_a:>7d} {exact_both:>9d}")

    # Show example predictions with error
    print(f"\n  Score prediction examples (RF Regressor):")
    print(f"  {'Match':<35s} {'Actual':>7s} {'Pred':>7s} {'Error':>7s}")
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    for i in indices[:25]:
        home = meta.iloc[i]["home_team"]
        away = meta.iloc[i]["away_team"]
        ah, aa = int(y_home_g[i]), int(y_away_g[i])
        ph_r, pa_r = round(pred_rf_h[i], 1), round(pred_rf_a[i], 1)
        err = abs(pred_rf_h[i] - ah) + abs(pred_rf_a[i] - aa)
        print(f"  {home+' v '+away:<35s} {ah}-{aa:>4d} {ph_r:.1f}-{pa_r:.1f} {err:>6.1f}")

    # ==========================================================
    # ANALYSIS 3: OVER/UNDER (Total Goals)
    # ==========================================================
    print("\n" + "=" * 70)
    print("[Analysis 3] Over/Under — Total Goals Prediction")
    print("=" * 70)

    # Standard betting line: Over/Under 2.5
    actual_over25 = (actual_total > 2.5)
    print(f"\n  Actual: {actual_over25.sum()}/{n} matches had Over 2.5 goals ({actual_over25.mean():.1%})")
    print(f"  Avg total goals: {actual_total.mean():.2f}")

    print(f"\n  --- Over/Under 2.5 prediction accuracy ---")
    for name, ph, pa in goal_models:
        pred_total = ph + pa
        pred_over25 = (pred_total > 2.5)
        correct = (pred_over25 == actual_over25).sum()
        print(f"  {name:<15s}: {correct}/{n} ({correct/n:.1%})")

    # Also try Over/Under 1.5 and 3.5
    for line in [1.5, 2.5, 3.5]:
        actual_over = (actual_total > line)
        print(f"\n  Over/Under {line}:")
        print(f"    Actual Over: {actual_over.sum()}/{n} ({actual_over.mean():.1%})")
        for name, ph, pa in goal_models:
            pred_over = ((ph + pa) > line)
            correct = (pred_over == actual_over).sum()
            print(f"    {name:<15s}: {correct}/{n} ({correct/n:.1%})")

    # ==========================================================
    # ANALYSIS 4: WHICH MODEL MAKES MONEY?
    # ==========================================================
    print("\n" + "=" * 70)
    print("[Analysis 4] Profitability Simulation (Simplified)")
    print("=" * 70)

    # Simplified betting: bet 1 unit on every match where model disagrees with Elo
    # If correct: win 1 unit (simplification; real payout varies)
    # If wrong: lose 1 unit
    # This approximates "contrarian betting" — only bet when you see something the market doesn't.
    # 简化博彩模拟：只在模型与 Elo 意见不同时下注。
    # 押对赚 1，押错亏 1。这模拟"逆势投注"策略。

    print(f"\n  Strategy: Bet 1 unit when model disagrees with Elo")
    print(f"  {'Model':<15s} {'Bets':>6s} {'Wins':>6s} {'Win%':>7s} {'P&L':>7s} {'ROI':>7s}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")

    for name, preds in [("RF", y_rf), ("XGB", y_xgb)]:
        disagree = (preds != elo_preds)
        n_bets = disagree.sum()
        wins = ((preds == y.values) & disagree).sum()
        losses = n_bets - wins
        pnl = wins - losses
        roi = pnl / n_bets * 100 if n_bets > 0 else 0
        print(f"  {name:<15s} {n_bets:>6d} {wins:>6d} {wins/n_bets:.1%} {pnl:>+6d} {roi:>+6.1f}%")

    # Over/Under betting
    print(f"\n  Strategy: Bet Over/Under 2.5 (contrarian vs naive baseline)")
    print(f"  {'Model':<15s} {'Bets':>6s} {'Wins':>6s} {'Win%':>7s}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*7}")

    # Naive baseline: always bet Over (since avg goals > 2.5 in WC)
    naive_over = actual_over25.sum()
    print(f"  {'Always Over':<15s} {n:>6d} {naive_over:>6d} {naive_over/n:.1%}")

    for name, ph, pa in goal_models:
        pred_over = ((ph + pa) > 2.5)
        correct = (pred_over == actual_over25).sum()
        print(f"  {name:<15s} {n:>6d} {correct:>6d} {correct/n:.1%}")

    print("\n" + "=" * 70)
    print("  Deep analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    run()
