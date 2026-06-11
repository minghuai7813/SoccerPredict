"""
Multi-tournament match predictor with model comparison.
多赛事比赛预测器——多模型对比版。

Trains on 2018 WC + Euro 2020 + 2022 WC combined (179 matches).
基于 2018 世界杯 + 2020 欧洲杯 + 2022 世界杯合并数据（179 场）训练。

Usage / 用法:
    python -m models.match_predictor
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LABEL_MAP = {1: "Home Win", 0: "Draw", -1: "Away Win"}
LABEL_TO_XGB = {-1: 0, 0: 1, 1: 2}
XGB_TO_LABEL = {0: -1, 1: 0, 2: 1}


def _elo_predict(home_team: str, away_team: str, elo_data: dict) -> tuple[int, float]:
    """Elo prediction with ~60 neutral-venue advantage."""
    ra = elo_data.get(home_team, {}).get("elo", 1500)
    rb = elo_data.get(away_team, {}).get("elo", 1500)
    prob = 1.0 / (1.0 + 10 ** ((rb - ra - 60) / 400.0))
    if prob > 0.45:
        return 1, prob
    elif prob < 0.35:
        return -1, prob
    return 0, prob


def _poisson_result_probs(lh: float, la: float, max_g: int = 7) -> dict[int, float]:
    """P(home win), P(draw), P(away win) from Poisson parameters."""
    from scipy.stats import poisson
    mat = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            mat[i, j] = poisson.pmf(i, lh) * poisson.pmf(j, la)
    return {1: np.triu(mat, k=1).sum(), 0: np.trace(mat), -1: np.tril(mat, k=-1).sum()}


def train_and_evaluate() -> dict:
    from features.match_dataset import build_match_dataset
    from extractors.elo_scraper import get_pre_tournament_elo

    print("=" * 65)
    print("  PROJECT ORACLE — Match Predictor v0.4")
    print("  (Multi-tournament: 2018 WC + Euro 2020 + 2022 WC)")
    print("=" * 65)

    X, y, meta = build_match_dataset()
    if X.empty:
        print("[Model] No data. Aborting.")
        return {}

    feature_names = X.columns.tolist()

    # Prepare two versions of X:
    # X_imp: NaN -> median (for RF, Poisson)
    # X_nan: keep NaN (XGBoost handles natively)
    # 两版特征矩阵：填充版（RF/Poisson）和原始版（XGBoost 原生处理 NaN）。
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X), columns=feature_names, index=X.index,
    )
    X_imp = X_imp.replace([np.inf, -np.inf], 0)

    X_nan = X.replace([np.inf, -np.inf], np.nan)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imp), columns=feature_names, index=X.index,
    )

    loo = LeaveOneOut()
    n_matches = len(X)

    # ================================================================
    # MODEL A: Random Forest
    # ================================================================
    print("\n" + "=" * 65)
    print("[Model A] Random Forest")
    print("=" * 65)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=loo)
    acc_rf = accuracy_score(y, y_pred_rf)
    print(f"  LOO-CV Accuracy: {acc_rf:.1%}")

    # ================================================================
    # MODEL B: XGBoost (handles NaN natively)
    # ================================================================
    print("\n" + "=" * 65)
    print("[Model B] XGBoost")
    print("=" * 65)

    y_xgb = y.map(LABEL_TO_XGB)

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=4,
        reg_alpha=1.5,
        reg_lambda=3.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    y_pred_xgb_raw = cross_val_predict(xgb, X_nan, y_xgb, cv=loo)
    y_pred_xgb = pd.Series(y_pred_xgb_raw).map(XGB_TO_LABEL).values
    acc_xgb = accuracy_score(y, y_pred_xgb)
    print(f"  LOO-CV Accuracy: {acc_xgb:.1%}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y, y_pred_xgb,
        target_names=[LABEL_MAP[l] for l in sorted(y.unique())],
        zero_division=0,
    ))

    # ================================================================
    # MODEL C: Poisson Regression
    # ================================================================
    print("=" * 65)
    print("[Model C] Poisson Regression")
    print("=" * 65)

    y_home_g = meta["home_score"].values.astype(float)
    y_away_g = meta["away_score"].values.astype(float)
    X_pos = X_imp.clip(lower=0)

    ph = PoissonRegressor(alpha=1.0, max_iter=1000)
    pa = PoissonRegressor(alpha=1.0, max_iter=1000)
    pred_hg = np.clip(cross_val_predict(ph, X_pos, y_home_g, cv=loo), 0.1, 6.0)
    pred_ag = np.clip(cross_val_predict(pa, X_pos, y_away_g, cv=loo), 0.1, 6.0)

    y_pred_poi = np.array([
        max(_poisson_result_probs(pred_hg[i], pred_ag[i]), key=lambda k: _poisson_result_probs(pred_hg[i], pred_ag[i])[k])
        for i in range(n_matches)
    ])
    acc_poi = accuracy_score(y, y_pred_poi)
    print(f"  LOO-CV Accuracy (W/D/L): {acc_poi:.1%}")
    print(f"  MAE home goals: {mean_absolute_error(y_home_g, pred_hg):.2f}")
    print(f"  MAE away goals: {mean_absolute_error(y_away_g, pred_ag):.2f}")

    # ================================================================
    # Elo baseline (per-tournament Elo)
    # ================================================================
    elo_map = {
        "FIFA World Cup 2018": "wc2018",
        "UEFA Euro 2020": "euro2020",
        "FIFA World Cup 2022": "wc2022",
        "FIFA World Cup 2026": "wc2026",
    }
    elo_caches = {k: get_pre_tournament_elo(v) for k, v in elo_map.items()}

    elo_preds = np.array([
        _elo_predict(
            meta.iloc[i]["home_team"],
            meta.iloc[i]["away_team"],
            elo_caches[meta.iloc[i]["tournament"]],
        )[0]
        for i in range(n_matches)
    ])
    acc_elo = accuracy_score(y, elo_preds)

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print("\n" + "=" * 65)
    print("[Comparison] All Models — LOO-CV Accuracy")
    print("=" * 65)

    models = [
        ("Elo (market)", acc_elo, elo_preds),
        ("Random Forest", acc_rf, y_pred_rf),
        ("XGBoost", acc_xgb, y_pred_xgb),
        ("Poisson (W/D/L)", acc_poi, y_pred_poi),
    ]

    gs_mask = ~meta["is_knockout"].values
    ko_mask = meta["is_knockout"].values

    # Per-tournament breakdown.
    tournaments = meta["tournament"].unique()

    header = f"  {'Model':<20s} {'Overall':>8s}"
    sep = f"  {'-'*20} {'-'*8}"
    for t in tournaments:
        short = t.split()[-1][:6]
        header += f" {short:>8s}"
        sep += f" {'-'*8}"
    header += f" {'Group':>8s} {'KO':>8s}"
    sep += f" {'-'*8} {'-'*8}"

    print(f"\n{header}")
    print(sep)

    for name, overall, preds in models:
        line = f"  {name:<20s} {overall:>7.1%}"
        for t in tournaments:
            t_mask = (meta["tournament"] == t).values
            t_acc = accuracy_score(y[t_mask], preds[t_mask]) if t_mask.sum() else 0
            line += f" {t_acc:>7.1%}"
        gs_acc = accuracy_score(y[gs_mask], preds[gs_mask]) if gs_mask.sum() else 0
        ko_acc = accuracy_score(y[ko_mask], preds[ko_mask]) if ko_mask.sum() else 0
        line += f" {gs_acc:>7.1%} {ko_acc:>7.1%}"
        print(line)

    best_name, best_acc, best_preds = max(models[1:], key=lambda x: x[1])
    print(f"\n  >>> Best model: {best_name} ({best_acc:.1%})")

    # ================================================================
    # ALPHA ANALYSIS (best model vs Elo)
    # ================================================================
    print("\n" + "=" * 65)
    print(f"[Alpha] {best_name} vs Elo Market")
    print("=" * 65)

    edge = best_acc - acc_elo
    sign = "+" if edge > 0 else ""
    print(f"\n  Overall: Model {best_acc:.1%} vs Elo {acc_elo:.1%} = {sign}{edge:.1%}")

    agree = (elo_preds == best_preds)
    disagree_mask = ~agree
    disagree_n = disagree_mask.sum()

    if disagree_n > 0:
        model_wins = ((y.values == best_preds) & disagree_mask).sum()
        elo_wins = ((y.values == elo_preds) & disagree_mask).sum()
        both_wrong = disagree_n - model_wins - elo_wins
        print(f"\n  Agreement: {agree.sum()}/{n_matches}")
        print(f"  Disagreements: {disagree_n}/{n_matches}")
        print(f"    Model right: {model_wins} | Elo right: {elo_wins} | Both wrong: {both_wrong}")
        if model_wins > elo_wins:
            print(f"    >>> ALPHA confirmed ({model_wins}:{elo_wins})")

    # Upset detection.
    elo_wrong = (y.values != elo_preds)
    caught = ((y.values == best_preds) & elo_wrong)
    print(f"\n  Upsets (Elo wrong): {elo_wrong.sum()}/{n_matches}")
    print(f"  Model caught: {caught.sum()}/{elo_wrong.sum()}")

    # ================================================================
    # KNOCKOUT QUALIFIER (2022 WC only)
    # ================================================================
    wc22_ko = meta[(meta["tournament"] == "FIFA World Cup 2022") & meta["is_knockout"]]
    if len(wc22_ko) > 0:
        print("\n" + "=" * 65)
        print("[2022 WC Knockout] Qualifier Prediction")
        print("=" * 65)

        elo22 = elo_caches["FIFA World Cup 2022"]
        qual_ours, qual_elo = [], []
        for idx in wc22_ko.index:
            home, away = meta.loc[idx, "home_team"], meta.loc[idx, "away_team"]
            elo_h = elo22.get(home, {}).get("elo", 1500)
            elo_a = elo22.get(away, {}).get("elo", 1500)

            p = best_preds[idx]
            if p == 1:
                qual_ours.append(home)
            elif p == -1:
                qual_ours.append(away)
            else:
                qual_ours.append(home if elo_h >= elo_a else away)

            qual_elo.append(home if elo_h >= elo_a else away)

        actual_q = wc22_ko["qualifier"].values
        our_c = sum(p == a for p, a in zip(qual_ours, actual_q))
        elo_c = sum(p == a for p, a in zip(qual_elo, actual_q))
        n_ko = len(qual_ours)
        print(f"  Model: {our_c}/{n_ko} ({our_c/n_ko:.1%}) | Elo: {elo_c}/{n_ko} ({elo_c/n_ko:.1%})")

    # ================================================================
    # Feature importance
    # ================================================================
    print("\n" + "=" * 65)
    print("[Feature Importance] Top 15")
    print("=" * 65)

    xgb.fit(X_nan, y_xgb)
    importances = pd.Series(xgb.feature_importances_, index=feature_names)
    top15 = importances.sort_values(ascending=False).head(15)
    for feat, imp in top15.items():
        bar = "\u2588" * int(imp * 200)
        print(f"  {feat:<42s} {imp:.4f}  {bar}")

    print("\n" + "=" * 65)
    print("  Analysis complete.")
    print("=" * 65)

    return {
        "accuracy_rf": acc_rf,
        "accuracy_xgb": acc_xgb,
        "accuracy_poisson": acc_poi,
        "accuracy_elo": acc_elo,
        "best_model": best_name,
        "n_matches": n_matches,
    }


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    train_and_evaluate()
