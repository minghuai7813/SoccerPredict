"""
Combine main 1X2 model with upset model via Elo-gap gating.
主模型 + 冷门模型门控组合。

When Elo gap >= 200 and P(upset) high: pick draw/underdog win via main model proba.
当悬殊且 P(冷门) 高：在主模型概率中于「非热门」结果间选取。

Usage / 用法:
    python -m models.upset_gating
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.upset_dataset import MIN_ELO_GAP, annotate_elo_and_upset
from features.match_dataset import build_match_dataset


def _main_loo_predictions(
    X: pd.DataFrame, y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """LOO RF 1X2 predictions and class probabilities (classes -1,0,1)."""
    fn = X.columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=fn).replace([np.inf, -np.inf], 0)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_imp), columns=fn)
    loo = LeaveOneOut()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    rf.fit(X_scaled, y)
    classes = list(rf.classes_)
    y_pred = cross_val_predict(rf, X_scaled, y, cv=loo)
    proba = cross_val_predict(rf, X_scaled, y, cv=loo, method="predict_proba")
    return y_pred, proba, classes


def _upset_loo_proba(X_u: pd.DataFrame, y_u: pd.Series) -> np.ndarray:
    fn = X_u.columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X_u), columns=fn).replace([np.inf, -np.inf], 0)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_imp), columns=fn)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=3,
        random_state=42, class_weight="balanced",
    )
    return cross_val_predict(
        rf, X_scaled, y_u, cv=LeaveOneOut(), method="predict_proba",
    )[:, 1]


def _non_favorite_pick(
    proba_row: np.ndarray,
    classes: list,
    home: str,
    away: str,
    favorite: str,
) -> int:
    """Among draw + underdog win, pick class with highest RF probability."""
    idx = {int(c): i for i, c in enumerate(classes)}
    candidates = []
    if favorite == home:
        # underdog win = away
        if -1 in idx:
            candidates.append((proba_row[idx[-1]], -1))
        if 0 in idx:
            candidates.append((proba_row[idx[0]], 0))
    else:
        if 1 in idx:
            candidates.append((proba_row[idx[1]], 1))
        if 0 in idx:
            candidates.append((proba_row[idx[0]], 0))
    if not candidates:
        return 0
    return max(candidates, key=lambda x: x[0])[1]


def _favorite_result_code(home: str, away: str, favorite: str) -> int:
    return 1 if favorite == home else -1


def run(min_elo_gap: int = MIN_ELO_GAP) -> None:
    from features.upset_dataset import build_upset_dataset

    print("=" * 72)
    print("  PROJECT ORACLE — Gated Main + Upset Models")
    print(f"  Lopsided if Elo gap >= {min_elo_gap}")
    print("=" * 72)

    X, y, meta = build_match_dataset()
    enriched = annotate_elo_and_upset(meta, y, min_elo_gap=min_elo_gap)
    y_main_pred, main_proba, classes = _main_loo_predictions(X, y)

    mask = enriched["is_lopsided"].values
    X_u, y_u, meta_u = build_upset_dataset(min_elo_gap=min_elo_gap, verbose=False)
    upset_proba_u = _upset_loo_proba(X_u, y_u)
    # Map lopsided rows back to full index
    p_upset_full = np.zeros(len(y))
    j = 0
    for i in range(len(y)):
        if mask[i]:
            p_upset_full[i] = upset_proba_u[j]
            j += 1

    y_true = y.values.astype(int)
    main_acc = accuracy_score(y_true, y_main_pred)

    print(f"\n  Main RF alone (all {len(y)} matches): {main_acc:.1%}")

    best = {"tau": 0.5, "acc": 0.0, "rec": 0.0, "prec": 0.0}
    print(f"\n  {'tau':>6s} {'all_acc':>8s} {'lop_acc':>8s} {'upset_rec':>10s} {'upset_prec':>11s}")
    for tau in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        gated = []
        for i in range(len(y)):
            if not mask[i]:
                gated.append(int(y_main_pred[i]))
                continue
            if p_upset_full[i] >= tau:
                gated.append(_non_favorite_pick(
                    main_proba[i], classes,
                    meta.iloc[i]["home_team"],
                    meta.iloc[i]["away_team"],
                    enriched.iloc[i]["favorite"],
                ))
            else:
                gated.append(_favorite_result_code(
                    meta.iloc[i]["home_team"],
                    meta.iloc[i]["away_team"],
                    enriched.iloc[i]["favorite"],
                ))
        gated = np.array(gated)
        acc = accuracy_score(y_true, gated)
        lop_idx = np.where(mask)[0]
        lop_acc = accuracy_score(y_true[lop_idx], gated[lop_idx])
        upset_true = enriched["upset"].values.astype(int)
        pred_upset = np.zeros(len(y), dtype=int)
        for i in lop_idx:
            if gated[i] != _favorite_result_code(
                meta.iloc[i]["home_team"], meta.iloc[i]["away_team"],
                enriched.iloc[i]["favorite"],
            ):
                pred_upset[i] = 1
        p, r, _, _ = precision_recall_fscore_support(
            upset_true[mask], pred_upset[mask],
            average="binary", pos_label=1, zero_division=0,
        )
        print(f"  {tau:>6.2f} {acc:>7.1%} {lop_acc:>7.1%} {r:>9.1%} {p:>10.1%}")
        if acc > best["acc"]:
            best = {"tau": tau, "acc": acc, "rec": r, "prec": p, "lop_acc": lop_acc}

    tau = best["tau"]
    print(f"\n  Best overall accuracy at tau={tau:.2f}: {best['acc']:.1%} "
          f"(vs main {main_acc:.1%}, delta {best['acc']-main_acc:+.1%})")
    print(f"  Lopsided-only acc: {best['lop_acc']:.1%} | "
          f"upset recall {best['rec']:.1%} | precision {best['prec']:.1%}")

    # Build final gated preds at best tau
    gated = []
    for i in range(len(y)):
        if not mask[i]:
            gated.append(int(y_main_pred[i]))
        elif p_upset_full[i] >= tau:
            gated.append(_non_favorite_pick(
                main_proba[i], classes,
                meta.iloc[i]["home_team"], meta.iloc[i]["away_team"],
                enriched.iloc[i]["favorite"],
            ))
        else:
            gated.append(_favorite_result_code(
                meta.iloc[i]["home_team"], meta.iloc[i]["away_team"],
                enriched.iloc[i]["favorite"],
            ))
    gated = np.array(gated)

    # Non-lopsided breakdown
    norm_idx = np.where(~mask)[0]
    print(f"\n  Non-lopsided ({len(norm_idx)}): gated==main, acc="
          f"{accuracy_score(y_true[norm_idx], gated[norm_idx]):.1%}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    run()
