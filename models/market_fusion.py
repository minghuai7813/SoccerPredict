"""
Model B: blend RF predictions with market (bookmaker / Elo) probabilities.
模型 B：随机森林预测与盘口（博彩/Elo）概率融合。

Usage / 用法:
    python -m models.market_fusion
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extractors.odds_loader import attach_odds_to_meta, load_match_odds
from models.match_predictor import LABEL_MAP

LABELS = [-1, 0, 1]


def _proba_row(classes: list, row: np.ndarray) -> dict[int, float]:
    return {int(classes[j]): float(row[j]) for j in range(len(classes))}


def _blend(p_model: dict[int, float], p_market: dict[int, float], alpha: float) -> dict[int, float]:
    out = {}
    for k in LABELS:
        out[k] = alpha * p_model.get(k, 0) + (1 - alpha) * p_market.get(k, 0)
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def _predict_class(probs: dict[int, float]) -> int:
    return max(probs, key=probs.get)


def run() -> None:
    from features.match_dataset import build_match_dataset
    from extractors.elo_scraper import get_pre_tournament_elo

    print("=" * 72)
    print("  PROJECT ORACLE — Market Fusion (B)")
    print("=" * 72)

    X, y, meta = build_match_dataset()
    if X.empty:
        print("[Fusion] No data.")
        return

    elo_map = {
        "FIFA World Cup 2018": "wc2018",
        "UEFA Euro 2020": "euro2020",
        "FIFA World Cup 2022": "wc2022",
    }
    elo_caches = {k: get_pre_tournament_elo(v) for k, v in elo_map.items()}

    odds_df = load_match_odds()
    meta_odds = attach_odds_to_meta(meta, odds_df, elo_caches)

    n_book = (meta_odds["odds_source"] == "oddsportal").sum()
    n_swap = (meta_odds["odds_source"] == "oddsportal_swapped").sum()
    n_elo = meta_odds["odds_source"].str.contains("elo", na=False).sum()
    print(f"\n[Odds coverage] bookmaker: {n_book} | swapped-fix: {n_swap} | elo fallback: {n_elo} / {len(meta)}")

    feature_names = X.columns.tolist()
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)
    X_imp = X_imp.replace([np.inf, -np.inf], 0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=feature_names)

    loo = LeaveOneOut()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    rf.fit(X_scaled, y)
    classes = list(rf.classes_)
    y_proba = cross_val_predict(rf, X_scaled, y, cv=loo, method="predict_proba")
    y_pred_model = cross_val_predict(rf, X_scaled, y, cv=loo)

    # Market-only predictions
    y_pred_market = []
    p_market_rows = []
    for i in range(len(y)):
        pm = {
            1: meta_odds.iloc[i]["prob_home"],
            0: meta_odds.iloc[i]["prob_draw"],
            -1: meta_odds.iloc[i]["prob_away"],
        }
        p_market_rows.append(pm)
        y_pred_market.append(_predict_class(pm))
    y_pred_market = np.array(y_pred_market)

    acc_model = accuracy_score(y, y_pred_model)
    acc_market = accuracy_score(y, y_pred_market)

    # Tune alpha (LOO blend — same-fold note: light leakage on alpha grid; acceptable for screening)
    best_alpha, best_acc, best_preds = 1.0, acc_model, y_pred_model
    alphas = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0]
    alpha_results = []

    for alpha in alphas:
        preds = []
        for i in range(len(y)):
            p_model = _proba_row(classes, y_proba[i])
            p_fused = _blend(p_model, p_market_rows[i], alpha)
            preds.append(_predict_class(p_fused))
        preds = np.array(preds)
        acc = accuracy_score(y, preds)
        alpha_results.append((alpha, acc))
        if acc > best_acc:
            best_acc, best_alpha, best_preds = acc, alpha, preds

    print("\n" + "-" * 72)
    print("[Accuracy] LOO-CV 90-min 1X2")
    print(f"  {'Model only (RF)':<28s} {acc_model:>6.1%}")
    print(f"  {'Market only (odds/Elo)':<28s} {acc_market:>6.1%}")
    print(f"  {'Fused (best alpha)':<28s} {best_acc:>6.1%}  alpha={best_alpha:.2f}")

    print("\n  Alpha sweep:")
    for a, acc in alpha_results:
        bar = "*" if a == best_alpha else " "
        print(f"    alpha={a:.2f}  {acc:>6.1%} {bar}")

    # ROI on fused picks (use attached decimal odds for predicted class)
    def _roi(preds: np.ndarray) -> tuple[float, int]:
        pnl = 0.0
        for i, pred in enumerate(preds):
            if pred == 1:
                od = meta_odds.iloc[i]["odds_home"]
            elif pred == 0:
                od = meta_odds.iloc[i]["odds_draw"]
            else:
                od = meta_odds.iloc[i]["odds_away"]
            if pd.isna(od):
                continue
            pnl += (od - 1) if y.iloc[i] == pred else -1.0
        return pnl, len(preds)

    pnl_m, _ = _roi(y_pred_model)
    pnl_k, _ = _roi(y_pred_market)
    pnl_f, _ = _roi(best_preds)
    print("\n" + "-" * 72)
    print("[Simulated ROI] 1 unit per match on predicted outcome (attached odds)")
    print(f"  RF only:     {pnl_m/len(y)*100:+.1f}%")
    print(f"  Market only: {pnl_k/len(y)*100:+.1f}%")
    print(f"  Fused:       {pnl_f/len(y)*100:+.1f}%")

    # Subset: only real bookmaker odds
    book_mask = meta_odds["odds_source"].str.startswith("oddsportal", na=False)
    if book_mask.sum() >= 20:
        acc_b = accuracy_score(y[book_mask], best_preds[book_mask.values])
        acc_m_b = accuracy_score(y[book_mask], y_pred_model[book_mask.values])
        print(f"\n  Bookmaker-odds subset ({book_mask.sum()} matches):")
        print(f"    RF {acc_m_b:.1%} | Fused {acc_b:.1%}")

    # Log loss (calibration)
    y_true_idx = [{-1: 0, 0: 1, 1: 2}[int(v)] for v in y]
    proba_model = np.zeros((len(y), 3))
    for i in range(len(y)):
        pr = _proba_row(classes, y_proba[i])
        for j, lab in enumerate(sorted(classes)):
            proba_model[i, j] = pr[int(lab)]
    ll_model = log_loss(y_true_idx, proba_model, labels=[0, 1, 2])

    proba_fused = np.zeros((len(y), 3))
    for i in range(len(y)):
        pr = _blend(_proba_row(classes, y_proba[i]), p_market_rows[i], best_alpha)
        for j, lab in enumerate(sorted(classes)):
            proba_fused[i, j] = pr[int(lab)]
    ll_fused = log_loss(y_true_idx, proba_fused, labels=[0, 1, 2])

    print(f"\n  Log-loss: RF {ll_model:.3f} | Fused {ll_fused:.3f} (lower is better)")

    print("\n" + "=" * 72)
    print(f"  Recommended blend: p = {best_alpha:.2f}*p_model + {1-best_alpha:.2f}*p_market")
    print("  Re-download odds: python scripts/download_odds.py")
    print("=" * 72)


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    run()
