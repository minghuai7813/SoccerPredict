"""
Gated main + upset models with threshold sweep and Poisson scorelines.
门控主模型/冷门模型 + 阈值扫描 + 泊松比分预测。

Rule:
  Elo gap < 200  -> main RF 1X2
  Elo gap >= 200 -> upset RF; if P(upset) >= tau -> non-favorite 1X2 else favorite win

Usage:
    python -m models.combined_predictor
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
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.match_dataset import build_match_dataset
from features.upset_dataset import MIN_ELO_GAP, annotate_elo_and_upset, build_upset_dataset
from models.match_predictor import LABEL_MAP


def _prep_X(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fn = X.columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=fn).replace([np.inf, -np.inf], 0)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_imp), columns=fn)
    return X_scaled, X.replace([np.inf, -np.inf], np.nan), fn


def _loo_main_1x2(X_scaled: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray, list]:
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    rf.fit(X_scaled, y)
    classes = list(rf.classes_)
    pred = cross_val_predict(rf, X_scaled, y, cv=LeaveOneOut())
    proba = cross_val_predict(rf, X_scaled, y, cv=LeaveOneOut(), method="predict_proba")
    return pred, proba, classes


def _loo_upset_proba(X_u: pd.DataFrame, y_u: pd.Series) -> np.ndarray:
    Xs, _, _ = _prep_X(X_u)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=3,
        random_state=42, class_weight="balanced",
    )
    return cross_val_predict(
        rf, Xs, y_u, cv=LeaveOneOut(), method="predict_proba",
    )[:, 1]


def _loo_poisson_goals(X: pd.DataFrame, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fn = X.columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_pos = pd.DataFrame(imp.fit_transform(X), columns=fn).replace([np.inf, -np.inf], 0).clip(lower=0)
    y_h = meta["home_score"].values.astype(float)
    y_a = meta["away_score"].values.astype(float)
    loo = LeaveOneOut()
    ph = PoissonRegressor(alpha=1.0, max_iter=1000)
    pa = PoissonRegressor(alpha=1.0, max_iter=1000)
    pred_h = np.clip(cross_val_predict(ph, X_pos, y_h, cv=loo), 0.05, 6.0)
    pred_a = np.clip(cross_val_predict(pa, X_pos, y_a, cv=loo), 0.05, 6.0)
    return pred_h, pred_a


def _fav_code(meta_row, enriched_row) -> int:
    h, a = meta_row["home_team"], meta_row["away_team"]
    fav = enriched_row["favorite"]
    return 1 if fav == h else -1


def _non_fav_pick(proba_row, classes, meta_row, enriched_row) -> int:
    idx = {int(c): i for i, c in enumerate(classes)}
    h, a = meta_row["home_team"], meta_row["away_team"]
    fav = enriched_row["favorite"]
    cands = []
    if fav == h:
        if -1 in idx:
            cands.append((proba_row[idx[-1]], -1))
        if 0 in idx:
            cands.append((proba_row[idx[0]], 0))
    else:
        if 1 in idx:
            cands.append((proba_row[idx[1]], 1))
        if 0 in idx:
            cands.append((proba_row[idx[0]], 0))
    return max(cands, key=lambda x: x[0])[1] if cands else 0


def _gated_pred(
    i: int,
    tau: float,
    mask_lop: np.ndarray,
    main_pred: np.ndarray,
    main_proba: np.ndarray,
    classes: list,
    p_upset: np.ndarray,
    meta: pd.DataFrame,
    enriched: pd.DataFrame,
) -> int:
    if not mask_lop[i]:
        return int(main_pred[i])
    if p_upset[i] >= tau:
        return _non_fav_pick(main_proba[i], classes, meta.iloc[i], enriched.iloc[i])
    return _fav_code(meta.iloc[i], enriched.iloc[i])


def _most_likely_score(lam_h: float, lam_a: float, max_g: int = 6) -> tuple[int, int]:
    best_p, best = 0.0, (0, 0)
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            if p > best_p:
                best_p, best = p, (i, j)
    return best


def run() -> None:
    print("=" * 72)
    print("  PROJECT ORACLE — Combined Predictor (1X2 + Scores)")
    print(f"  gap < {MIN_ELO_GAP}: main RF | gap >= {MIN_ELO_GAP}: upset RF + tau")
    print("=" * 72)

    X, y, meta = build_match_dataset()
    enriched = annotate_elo_and_upset(meta, y)
    mask_lop = enriched["is_lopsided"].values
    y_true = y.values.astype(int)

    X_scaled, _, _ = _prep_X(X)
    main_pred, main_proba, classes = _loo_main_1x2(X_scaled, y)
    acc_main = accuracy_score(y_true, main_pred)

    X_u, y_u, _ = build_upset_dataset(verbose=False)
    up_proba_u = _loo_upset_proba(X_u, y_u)
    p_upset = np.zeros(len(y))
    j = 0
    for i in range(len(y)):
        if mask_lop[i]:
            p_upset[i] = up_proba_u[j]
            j += 1

    upset_true = enriched["upset"].values.astype(int)

    # --- Threshold sweep ---
    print("\n" + "-" * 72)
    print("[1] Threshold sweep (LOO)")
    print(f"  {'tau':>5s} {'acc_all':>8s} {'acc_lo':>8s} {'acc_hi':>8s} "
          f"{'up_rec':>8s} {'up_pre':>8s}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    best_acc_row = None
    best_rec_row = None
    rows = []
    for tau in np.arange(0.25, 0.66, 0.05):
        gated = np.array([
            _gated_pred(i, tau, mask_lop, main_pred, main_proba, classes,
                        p_upset, meta, enriched)
            for i in range(len(y))
        ])
        acc = accuracy_score(y_true, gated)
        norm_i = ~mask_lop
        acc_lo = accuracy_score(y_true[norm_i], gated[norm_i])
        acc_hi = accuracy_score(y_true[mask_lop], gated[mask_lop])
        pred_up = np.zeros(len(y), dtype=int)
        for i in range(len(y)):
            if mask_lop[i] and gated[i] != _fav_code(meta.iloc[i], enriched.iloc[i]):
                pred_up[i] = 1
        pre, rec, _, _ = precision_recall_fscore_support(
            upset_true[mask_lop], pred_up[mask_lop],
            average="binary", pos_label=1, zero_division=0,
        )
        rows.append((tau, acc, acc_lo, acc_hi, rec, pre))
        print(f"  {tau:>5.2f} {acc:>7.1%} {acc_lo:>7.1%} {acc_hi:>7.1%} {rec:>7.1%} {pre:>7.1%}")
        if best_acc_row is None or acc > best_acc_row[1]:
            best_acc_row = (tau, acc, acc_lo, acc_hi, rec, pre)
        if best_rec_row is None or rec > best_rec_row[4]:
            best_rec_row = (tau, acc, acc_lo, acc_hi, rec, pre)

    print(f"\n  Best overall accuracy: tau={best_acc_row[0]:.2f} -> {best_acc_row[1]:.1%} "
          f"(main alone {acc_main:.1%})")
    print(f"  Best upset recall (lopsided): tau={best_rec_row[0]:.2f} -> "
          f"recall {best_rec_row[4]:.1%}, acc_all {best_rec_row[1]:.1%}")

    # Use tau=0.45 as balanced default (from prior upset_predictor F1)
    tau_bal = 0.45
    gated_bal = np.array([
        _gated_pred(i, tau_bal, mask_lop, main_pred, main_proba, classes,
                    p_upset, meta, enriched)
        for i in range(len(y))
    ])

    # --- Scores (Poisson, all matches; same features) ---
    print("\n" + "-" * 72)
    print("[2] Score prediction (Poisson LOO, all matches)")
    pred_h, pred_a = _loo_poisson_goals(X, meta)
    y_h = meta["home_score"].values.astype(int)
    y_a = meta["away_score"].values.astype(int)
    mae_h = mean_absolute_error(y_h, pred_h)
    mae_a = mean_absolute_error(y_a, pred_a)
    exact = 0
    wdl_from_score = 0
    for i in range(len(y)):
        sh, sa = _most_likely_score(pred_h[i], pred_a[i])
        if sh == y_h[i] and sa == y_a[i]:
            exact += 1
        res = 1 if sh > sa else (-1 if sh < sa else 0)
        if res == y_true[i]:
            wdl_from_score += 1

    print(f"  MAE home goals: {mae_h:.2f}")
    print(f"  MAE away goals: {mae_a:.2f}")
    print(f"  Exact scoreline: {exact}/{len(y)} ({exact/len(y):.1%})")
    print(f"  W/D/L from predicted score: {wdl_from_score}/{len(y)} ({wdl_from_score/len(y):.1%})")

    # Gated 1X2 + same Poisson lambdas (scores independent of tau)
    print("\n" + "-" * 72)
    print(f"[3] Combined summary at balanced tau={tau_bal:.2f}")
    acc_g = accuracy_score(y_true, gated_bal)
    print(f"  1X2 accuracy (gated): {acc_g:.1%}  |  main only: {acc_main:.1%}")
    print(f"  Score MAE (H/A): {mae_h:.2f} / {mae_a:.2f}")
    print(f"  (1X2 and scores are separate models; lambdas do not use upset gate)")

    print("\n" + "-" * 72)
    print("[4] Feature catalog: python scripts/list_features.py")
    print("=" * 72)


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    run()
