"""
Club match predictor — RF + Poisson on domestic league training set.
俱乐部比赛预测器：在国内联赛训练集上训练 RF + 泊松。

Usage / 用法:
    python -m models.club_predictor
    python -m models.club_predictor --max-matches 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.club_match_dataset import build_club_match_dataset
from models.match_predictor import LABEL_MAP


def train_and_evaluate(max_matches: int | None = None, cv_folds: int | None = None) -> dict:
    print("=" * 65)
    print("  PROJECT ORACLE — Club Predictor")
    print("  Training: domestic leagues (football-data.co.uk)")
    print("=" * 65)

    X, y, meta = build_club_match_dataset(max_matches=max_matches)
    if X.empty:
        print("[Club] No data.")
        return {}

    fn = X.columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=fn)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=fn)

    n = len(y)
    if cv_folds is None:
        cv = LeaveOneOut() if n <= 1000 else StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42,
        )
        cv_label = "LOO" if n <= 1000 else "5-fold"
    elif cv_folds <= 1:
        cv = LeaveOneOut()
        cv_label = "LOO"
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_label = f"{cv_folds}-fold"

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    y_pred = cross_val_predict(rf, X_scaled, y, cv=cv)
    acc = accuracy_score(y, y_pred)
    print(f"\n[RF] {cv_label} accuracy: {acc:.1%}  ({n} matches)")

    X_pos = X_imp.clip(lower=0)
    ph = PoissonRegressor(alpha=1.0, max_iter=1000)
    pa = PoissonRegressor(alpha=1.0, max_iter=1000)
    pred_h = np.clip(cross_val_predict(ph, X_pos, meta["home_score"], cv=cv), 0.05, 6.0)
    pred_a = np.clip(cross_val_predict(pa, X_pos, meta["away_score"], cv=cv), 0.05, 6.0)
    mae_h = mean_absolute_error(meta["home_score"], pred_h)
    mae_a = mean_absolute_error(meta["away_score"], pred_a)
    print(f"[Poisson] MAE home/away: {mae_h:.2f} / {mae_a:.2f}")

    rf.fit(X_scaled, y)
    return {
        "accuracy": acc,
        "n_matches": len(y),
        "mae_home": mae_h,
        "mae_away": mae_a,
        "model": rf,
        "imputer": imp,
        "scaler": scaler,
        "poisson_home": ph.fit(X_pos, meta["home_score"]),
        "poisson_away": pa.fit(X_pos, meta["away_score"]),
        "feature_names": fn,
    }


def main() -> None:
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None, help="Default: LOO if n<=1000 else 5")
    args = parser.parse_args()
    train_and_evaluate(max_matches=args.max_matches, cv_folds=args.cv_folds)


if __name__ == "__main__":
    main()
