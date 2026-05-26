"""
Feature selection (A) + interaction discovery (B) pipeline.
特征选择 + 特征交互发现流水线。

A: SHAP analysis → select top features → retrain with slim set
   SHAP 分析 → 精选特征 → 用精简集重新训练
B: Analyze correct vs incorrect predictions → derive interaction features
   分析判对/判错比赛的差异 → 构造交互特征

Usage / 用法:
    python models/feature_engineering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LABEL_MAP = {1: "Home Win", 0: "Draw", -1: "Away Win"}
LABEL_TO_XGB = {-1: 0, 0: 1, 1: 2}
XGB_TO_LABEL = {0: -1, 1: 0, 2: 1}


def run():
    from features.match_dataset import build_match_dataset
    from extractors.elo_scraper import get_pre_tournament_elo

    print("=" * 70)
    print("  Feature Engineering Pipeline (A + B)")
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

    # Elo baseline
    elo_map = {"FIFA World Cup 2018": "wc2018", "UEFA Euro 2020": "euro2020",
               "FIFA World Cup 2022": "wc2022"}
    elo_caches = {k: get_pre_tournament_elo(v) for k, v in elo_map.items()}

    def _elo_pred(i):
        h, a = meta.iloc[i]["home_team"], meta.iloc[i]["away_team"]
        ed = elo_caches[meta.iloc[i]["tournament"]]
        ra = ed.get(h, {}).get("elo", 1500)
        rb = ed.get(a, {}).get("elo", 1500)
        p = 1.0 / (1.0 + 10 ** ((rb - ra - 60) / 400.0))
        return 1 if p > 0.45 else (-1 if p < 0.35 else 0)

    elo_preds = np.array([_elo_pred(i) for i in range(n)])
    acc_elo = accuracy_score(y, elo_preds)

    # ================================================================
    # STEP A: SHAP Analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("[Step A] SHAP Feature Importance Analysis")
    print("=" * 70)

    rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                min_samples_leaf=4, random_state=42,
                                class_weight="balanced")
    rf.fit(X_scaled, y)

    print("  Computing SHAP values (this takes a moment)...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    # shap_values shape varies: list of (n,f) or (n,f,c) array.
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.Series(
        mean_abs_shap, index=feature_names,
    ).sort_values(ascending=False)

    print(f"\n  Top 30 features by SHAP importance:")
    print(f"  {'Rank':<5s} {'Feature':<45s} {'SHAP':>8s}")
    print(f"  {'-'*5} {'-'*45} {'-'*8}")
    for rank, (feat, val) in enumerate(feature_importance.head(30).items(), 1):
        bar = "\u2588" * int(val * 300)
        print(f"  {rank:<5d} {feat:<45s} {val:>8.4f}  {bar}")

    # Bottom 30 (noise candidates)
    print(f"\n  Bottom 20 features (noise candidates to remove):")
    for feat, val in feature_importance.tail(20).items():
        print(f"    {feat:<45s} {val:.6f}")

    # ================================================================
    # STEP A2: Feature Selection — keep top-N
    # ================================================================
    print("\n" + "=" * 70)
    print("[Step A2] Feature Selection — Testing different cutoffs")
    print("=" * 70)

    loo = LeaveOneOut()
    results = []

    for top_n in [15, 20, 25, 30, 40, 50, 99]:
        selected = feature_importance.head(top_n).index.tolist()
        X_sel = X_scaled[selected]

        rf_sel = RandomForestClassifier(n_estimators=300, max_depth=6,
                                        min_samples_leaf=4, random_state=42,
                                        class_weight="balanced")
        y_pred = cross_val_predict(rf_sel, X_sel, y, cv=loo)
        acc = accuracy_score(y, y_pred)
        results.append((top_n, acc))
        print(f"  Top {top_n:>3d} features: RF accuracy = {acc:.1%}")

    best_n, best_acc_rf = max(results, key=lambda x: x[1])
    print(f"\n  >>> Best cutoff: Top {best_n} features ({best_acc_rf:.1%})")

    selected_features = feature_importance.head(best_n).index.tolist()

    # Also test XGBoost with selected features
    X_sel_nan = X_nan[selected_features]
    y_xgb_enc = y.map(LABEL_TO_XGB)

    xgb_sel = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.7,
                             min_child_weight=4, reg_alpha=1.5, reg_lambda=3.0,
                             num_class=3, eval_metric="mlogloss",
                             random_state=42, verbosity=0)
    y_xgb_sel = pd.Series(
        cross_val_predict(xgb_sel, X_sel_nan, y_xgb_enc, cv=loo)
    ).map(XGB_TO_LABEL).values
    acc_xgb_sel = accuracy_score(y, y_xgb_sel)
    print(f"  XGBoost with top {best_n}: {acc_xgb_sel:.1%}")

    # ================================================================
    # STEP B: Feature Interaction Discovery
    # ================================================================
    print("\n" + "=" * 70)
    print("[Step B] Interaction Discovery — Correct vs Incorrect")
    print("=" * 70)

    X_sel_scaled = X_scaled[selected_features]
    rf_best = RandomForestClassifier(n_estimators=300, max_depth=6,
                                      min_samples_leaf=4, random_state=42,
                                      class_weight="balanced")
    y_pred_best = cross_val_predict(rf_best, X_sel_scaled, y, cv=loo)

    correct_mask = (y.values == y_pred_best)
    incorrect_mask = ~correct_mask

    print(f"\n  Correct predictions: {correct_mask.sum()}")
    print(f"  Incorrect predictions: {incorrect_mask.sum()}")

    # Compare mean feature values between correct and incorrect groups.
    X_analysis = X_imp[selected_features].copy()
    X_analysis["correct"] = correct_mask

    print(f"\n  Features where correct/incorrect groups differ most:")
    print(f"  {'Feature':<45s} {'Correct':>10s} {'Incorrect':>10s} {'Diff':>10s} {'Ratio':>8s}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    diffs = []
    for col in selected_features:
        mean_c = X_analysis.loc[correct_mask, col].mean()
        mean_i = X_analysis.loc[incorrect_mask, col].mean()
        std_all = X_analysis[col].std()
        if std_all > 0:
            effect_size = abs(mean_c - mean_i) / std_all
        else:
            effect_size = 0
        diffs.append((col, mean_c, mean_i, mean_c - mean_i, effect_size))

    diffs.sort(key=lambda x: x[4], reverse=True)
    for col, mc, mi, diff, es in diffs[:15]:
        print(f"  {col:<45s} {mc:>10.3f} {mi:>10.3f} {diff:>+10.3f} {es:>7.3f}")

    # ================================================================
    # STEP B2: Construct Interaction Features
    # ================================================================
    print("\n" + "=" * 70)
    print("[Step B2] Constructing Interaction Features")
    print("=" * 70)

    # Key interactions based on domain knowledge + SHAP findings.
    # Elo difference × league goal involvement → "strong team with in-form players"
    # Elo rank difference × positional balance → "top team with balanced squad"
    # 基于领域知识 + SHAP 的关键交互特征。

    X_enhanced = X_imp[selected_features].copy()

    interactions_added = []

    # Only add interactions if the base features exist.
    def _safe_interact(name, col_a, col_b, op="multiply"):
        if col_a in X_enhanced.columns and col_b in X_enhanced.columns:
            if op == "multiply":
                X_enhanced[name] = X_enhanced[col_a] * X_enhanced[col_b]
            elif op == "abs_diff":
                X_enhanced[name] = np.abs(X_enhanced[col_a] - X_enhanced[col_b])
            elif op == "ratio":
                denom = X_enhanced[col_b].replace(0, 0.001)
                X_enhanced[name] = X_enhanced[col_a] / denom
            interactions_added.append(name)
            return True
        return False

    _safe_interact("ix_elo_x_lg_goals", "diff_elo_rating", "diff_lg_goals_weighted")
    _safe_interact("ix_elo_x_lg_involve", "diff_elo_rating", "diff_lg_goal_involvement_weighted")
    _safe_interact("ix_elo_x_nt_xg", "diff_elo_rating", "diff_nt_xg_sum")
    _safe_interact("ix_lg_goals_x_nt_xg", "diff_lg_goals_weighted", "diff_nt_xg_sum")
    _safe_interact("ix_elo_rank_spread", "home_elo_rank", "away_elo_rank", "abs_diff")
    _safe_interact("ix_elo_x_conf", "diff_elo_rating", "diff_conf_UEFA")

    # Strength asymmetry: when one team is much stronger on paper.
    if "diff_elo_rating" in X_enhanced.columns:
        X_enhanced["ix_elo_gap_sq"] = X_enhanced["diff_elo_rating"] ** 2
        interactions_added.append("ix_elo_gap_sq")

    # "Dark horse signal": low Elo rank but high league stats.
    _safe_interact("ix_underdog_form", "away_elo_rank", "away_lg_goals_weighted")

    print(f"  Added {len(interactions_added)} interaction features:")
    for name in interactions_added:
        print(f"    {name}")

    # ================================================================
    # STEP C: Retrain with enhanced feature set
    # ================================================================
    print("\n" + "=" * 70)
    print("[Step C] Retrain with Enhanced Features")
    print("=" * 70)

    # Scale the enhanced set
    scaler2 = StandardScaler()
    X_enh_scaled = pd.DataFrame(
        scaler2.fit_transform(X_enhanced),
        columns=X_enhanced.columns,
    )

    # RF with enhanced features
    rf_enh = RandomForestClassifier(n_estimators=300, max_depth=6,
                                     min_samples_leaf=4, random_state=42,
                                     class_weight="balanced")
    y_pred_enh_rf = cross_val_predict(rf_enh, X_enh_scaled, y, cv=loo)
    acc_enh_rf = accuracy_score(y, y_pred_enh_rf)

    # XGBoost with enhanced features (use NaN version for base + imputed interactions)
    X_enh_xgb = X_nan[selected_features].copy()
    for col in interactions_added:
        X_enh_xgb[col] = X_enhanced[col]

    xgb_enh = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7,
                              min_child_weight=4, reg_alpha=1.5, reg_lambda=3.0,
                              num_class=3, eval_metric="mlogloss",
                              random_state=42, verbosity=0)
    y_xgb_enh = pd.Series(
        cross_val_predict(xgb_enh, X_enh_xgb, y_xgb_enc, cv=loo)
    ).map(XGB_TO_LABEL).values
    acc_enh_xgb = accuracy_score(y, y_xgb_enh)

    # ================================================================
    # FINAL COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("[Final Comparison]")
    print("=" * 70)

    print(f"\n  {'Model':<40s} {'Accuracy':>10s} {'vs Elo':>8s}")
    print(f"  {'-'*40} {'-'*10} {'-'*8}")
    print(f"  {'Elo baseline':<40s} {acc_elo:>9.1%}   {'---':>5s}")
    print(f"  {'RF (all 99 features)':<40s} {best_acc_rf if best_n == 99 else 'N/A':>9}   {'':>5s}")

    # Get original RF accuracy with all 99 features from results list
    all99 = [r for r in results if r[0] == 99]
    if all99:
        acc_99 = all99[0][1]
        print(f"  {'RF (all 99 features)':<40s} {acc_99:>9.1%} {acc_99-acc_elo:>+7.1%}")
    print(f"  {'RF (top {0} selected)'.format(best_n):<40s} {best_acc_rf:>9.1%} {best_acc_rf-acc_elo:>+7.1%}")
    print(f"  {'RF (top {0} + interactions)'.format(best_n):<40s} {acc_enh_rf:>9.1%} {acc_enh_rf-acc_elo:>+7.1%}")
    print(f"  {'XGBoost (top {0} selected)'.format(best_n):<40s} {acc_xgb_sel:>9.1%} {acc_xgb_sel-acc_elo:>+7.1%}")
    print(f"  {'XGBoost (top {0} + interactions)'.format(best_n):<40s} {acc_enh_xgb:>9.1%} {acc_enh_xgb-acc_elo:>+7.1%}")

    # Upset detection comparison
    elo_wrong = (y.values != elo_preds)
    n_upsets = elo_wrong.sum()

    print(f"\n  Upset detection ({n_upsets} upsets):")
    for name, preds in [("RF (selected)", y_pred_best),
                         ("RF (selected+interact)", y_pred_enh_rf),
                         ("XGB (selected+interact)", y_xgb_enh)]:
        caught = ((y.values == preds) & elo_wrong).sum()
        print(f"    {name:<35s}: {caught}/{n_upsets} ({caught/n_upsets:.1%})")

    # Error severity
    print(f"\n  Error severity:")
    for name, preds in [("RF (selected+interact)", y_pred_enh_rf),
                         ("XGB (selected+interact)", y_xgb_enh)]:
        wrong = (y.values != preds)
        n_wrong = wrong.sum()
        diffs_err = np.abs(y.values[wrong] - preds[wrong])
        off1 = (diffs_err == 1).sum()
        off2 = (diffs_err == 2).sum()
        print(f"    {name:<35s}: {n_wrong} wrong — {off1} off-by-1, {off2} off-by-2")

    # Classification reports for best models
    print(f"\n  --- RF (selected + interactions) ---")
    print(classification_report(
        y, y_pred_enh_rf,
        target_names=[LABEL_MAP[l] for l in sorted(y.unique())],
        zero_division=0,
    ))

    print(f"  --- XGBoost (selected + interactions) ---")
    print(classification_report(
        y, y_xgb_enh,
        target_names=[LABEL_MAP[l] for l in sorted(y.unique())],
        zero_division=0,
    ))

    # Save selected features for future use
    selected_path = _PROJECT_ROOT / "models" / "selected_features.txt"
    with open(selected_path, "w") as f:
        f.write("# Selected features from SHAP analysis\n")
        f.write(f"# Top {best_n} + {len(interactions_added)} interactions\n\n")
        f.write("# Base features:\n")
        for feat in selected_features:
            f.write(f"{feat}\n")
        f.write("\n# Interaction features:\n")
        for feat in interactions_added:
            f.write(f"{feat}\n")
    print(f"\n  Selected features saved to {selected_path}")

    print("\n" + "=" * 70)
    print("  Feature engineering complete.")
    print("=" * 70)


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    run()
