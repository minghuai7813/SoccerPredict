"""
Upset predictor — binary model for lopsided matches (Elo gap >= 200).
冷门预测器：悬殊局内预测「热门未赢」（含平局）。

Usage / 用法:
    python -m models.upset_predictor
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.upset_dataset import MIN_ELO_GAP, build_upset_dataset


def _prep_matrices(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fn = X.columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=fn)
    X_imp = X_imp.replace([np.inf, -np.inf], 0)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_imp), columns=fn)
    X_nan = X.replace([np.inf, -np.inf], np.nan)
    return X_scaled, X_nan, fn


def train_and_evaluate(min_elo_gap: int = MIN_ELO_GAP) -> dict:
    print("=" * 72)
    print("  PROJECT ORACLE — Upset Predictor (Lopsided Matches)")
    print(f"  Cohort: Elo gap >= {min_elo_gap} | Label: upset = favorite did NOT win")
    print("=" * 72)

    X, y, meta = build_upset_dataset(min_elo_gap=min_elo_gap)
    if len(X) < 10:
        print("[Upset] Too few matches. Aborting.")
        return {}

    X_scaled, X_nan, fn = _prep_matrices(X)
    loo = LeaveOneOut()
    n = len(y)
    n_upset = int(y.sum())

    # Baselines — always predict "no upset" (favorite wins)
    base0 = np.zeros(n, dtype=int)
    acc_base0 = accuracy_score(y, base0)
    # Always predict upset
    base1 = np.ones(n, dtype=int)
    acc_base1 = accuracy_score(y, base1)

    print("\n" + "-" * 72)
    print("[Baselines]")
    print(f"  Always 'favorite wins' (no upset):  acc={acc_base0:.1%}  "
          f"(recall upset=0%)")
    print(f"  Always 'upset':                    acc={acc_base1:.1%}  "
          f"(precision on upsets={n_upset/n:.1%})")

    # Random Forest
    print("\n" + "-" * 72)
    print("[Model A] Random Forest (class_weight=balanced)")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=3,
        random_state=42, class_weight="balanced",
    )
    y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=loo)
    y_proba_rf = cross_val_predict(rf, X_scaled, y, cv=loo, method="predict_proba")
    _print_binary_metrics(y, y_pred_rf, y_proba_rf[:, 1], "RF")

    rf.fit(X_scaled, y)
    imp = pd.Series(rf.feature_importances_, index=fn).sort_values(ascending=False)
    print("\n  Top 12 features (RF, full fit):")
    for feat, v in imp.head(12).items():
        print(f"    {feat:<42s} {v:.4f}")

    # XGBoost
    print("\n" + "-" * 72)
    print("[Model B] XGBoost (scale_pos_weight)")
    spw = (n - n_upset) / max(n_upset, 1)
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        reg_alpha=1.5,
        reg_lambda=3.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    y_pred_xgb = cross_val_predict(xgb, X_nan, y, cv=loo)
    y_proba_xgb = cross_val_predict(xgb, X_nan, y, cv=loo, method="predict_proba")
    _print_binary_metrics(y, y_pred_xgb, y_proba_xgb[:, 1], "XGB")

    xgb.fit(X_nan, y)
    imp_x = pd.Series(xgb.feature_importances_, index=fn).sort_values(ascending=False)
    print("\n  Top 12 features (XGB, full fit):")
    for feat, v in imp_x.head(12).items():
        print(f"    {feat:<42s} {v:.4f}")

    # Threshold sweep on RF proba (upset recall vs precision)
    print("\n" + "-" * 72)
    print("[Threshold sweep] RF P(upset) — LOO")
    print(f"  {'tau':>6s} {'pred+':>6s} {'prec':>7s} {'rec':>7s} {'f1':>7s}")
    best_f1, best_tau = 0.0, 0.5
    for tau in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        pred = (y_proba_rf[:, 1] >= tau).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", pos_label=1, zero_division=0,
        )
        n_pos = pred.sum()
        print(f"  {tau:>6.2f} {n_pos:>6d} {p:>6.1%} {r:>6.1%} {f1:>6.3f}")
        if f1 >= best_f1:
            best_f1, best_tau = f1, tau

    print(f"\n  Best F1 tau (RF): {best_tau:.2f} (F1={best_f1:.3f})")

    # List LOO hits/misses at best tau
    pred_best = (y_proba_rf[:, 1] >= best_tau).astype(int)
    print("\n" + "-" * 72)
    print(f"[LOO] True upsets caught at tau={best_tau:.2f} (RF)")
    print("-" * 72)
    for i in range(n):
        if y.iloc[i] != 1:
            continue
        hit = pred_best[i] == 1
        h, a = meta.iloc[i]["home_team"], meta.iloc[i]["away_team"]
        sc = f"{int(meta.iloc[i]['home_score'])}-{int(meta.iloc[i]['away_score'])}"
        mark = "HIT" if hit else "MISS"
        print(
            f"  [{mark}] gap={meta.iloc[i]['elo_gap']:.0f} "
            f"P={y_proba_rf[i,1]:.2f} | {h} vs {a} ({sc}) "
            f"| fav={meta.iloc[i]['favorite']}"
        )

    print("\n" + "=" * 72)
    print("  Next: python -m models.upset_gating")
    print("=" * 72)

    return {
        "n_lopsided": n,
        "n_upset": n_upset,
        "rf_accuracy": accuracy_score(y, y_pred_rf),
        "xgb_accuracy": accuracy_score(y, y_pred_xgb),
        "best_tau": best_tau,
    }


def _print_binary_metrics(
    y: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    name: str,
) -> None:
    acc = accuracy_score(y, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", pos_label=1, zero_division=0,
    )
    print(f"  Accuracy: {acc:.1%}")
    print(f"  Upset precision: {p:.1%} | recall: {r:.1%} | F1: {f1:.3f}")
    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion (rows=actual, cols=pred) [fav_win, upset]:")
    print(f"    {cm}")
    print(classification_report(
        y, y_pred, target_names=["Favorite win", "Upset"], zero_division=0,
    ))


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    train_and_evaluate()
