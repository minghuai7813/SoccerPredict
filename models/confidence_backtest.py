"""
Confidence-based selective betting backtest (LOO-CV).
置信度筛选 + 模拟 ROI 回测（留一交叉验证）。

Uses RF class probabilities; settles 90-min 1X2 vs Elo-implied odds proxy.
使用随机森林概率；按 90 分钟胜平负结算，赔率用 Elo 代理。

Usage / 用法:
    python -m models.confidence_backtest
    python -m models.confidence_backtest --export data/confidence_loo.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.market_odds import LABEL_AWAY, LABEL_DRAW, LABEL_HOME, decimal_odds, elo_1x2_probs
from models.match_predictor import LABEL_MAP, _elo_predict

BOOK_MARGIN = 0.05


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 1e-12]
    return float(-(p * np.log(p)).sum())


def _norm_entropy(probs: np.ndarray) -> float:
    """0 = certain, 1 = uniform over 3 classes."""
    h = _entropy(probs)
    h_max = np.log(3)
    return h / h_max if h_max > 0 else 0.0


def build_loo_frame() -> pd.DataFrame:
    from features.match_dataset import build_match_dataset
    from extractors.elo_scraper import get_pre_tournament_elo

    X, y, meta = build_match_dataset()
    feature_names = X.columns.tolist()
    n = len(X)

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
    y_pred = cross_val_predict(rf, X_scaled, y, cv=loo)
    y_proba = cross_val_predict(rf, X_scaled, y, cv=loo, method="predict_proba")

    elo_map = {
        "FIFA World Cup 2018": "wc2018",
        "UEFA Euro 2020": "euro2020",
        "FIFA World Cup 2022": "wc2022",
    }
    elo_caches = {k: get_pre_tournament_elo(v) for k, v in elo_map.items()}

    rows = []
    for i in range(n):
        home = meta.iloc[i]["home_team"]
        away = meta.iloc[i]["away_team"]
        tournament = meta.iloc[i]["tournament"]
        ed = elo_caches[tournament]
        eh = ed.get(home, {}).get("elo", 1500)
        ea = ed.get(away, {}).get("elo", 1500)

        elo_probs = elo_1x2_probs(eh, ea)
        elo_pred, _ = _elo_predict(home, away, ed)

        proba_row = {classes[j]: y_proba[i, j] for j in range(len(classes))}
        prob_vec = np.array([proba_row.get(LABEL_HOME, 0), proba_row.get(LABEL_DRAW, 0), proba_row.get(LABEL_AWAY, 0)])
        pred = int(y_pred[i])
        actual = int(y.iloc[i])

        max_prob = float(prob_vec.max())
        sorted_p = np.sort(prob_vec)[::-1]
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else max_prob
        ent = _norm_entropy(prob_vec)

        elo_fav = max(elo_probs, key=elo_probs.get)
        model_fav = max(proba_row, key=proba_row.get)

        odds_pred = decimal_odds(elo_probs[pred], BOOK_MARGIN)
        pnl = (odds_pred - 1.0) if actual == pred else -1.0

        edge = proba_row[pred] - elo_probs[pred]

        rows.append({
            "match": f"{home} vs {away}",
            "tournament": tournament,
            "short_tournament": tournament.split()[-1],
            "stage": meta.iloc[i].get("stage", ""),
            "is_knockout": bool(meta.iloc[i]["is_knockout"]),
            "went_to_et": bool(meta.iloc[i].get("went_to_et", False)),
            "home_score": int(meta.iloc[i]["home_score"]),
            "away_score": int(meta.iloc[i]["away_score"]),
            "total_goals": int(meta.iloc[i]["home_score"]) + int(meta.iloc[i]["away_score"]),
            "actual": actual,
            "actual_label": LABEL_MAP[actual],
            "pred": pred,
            "pred_label": LABEL_MAP[pred],
            "correct": actual == pred,
            "max_prob": max_prob,
            "prob_margin": margin,
            "norm_entropy": ent,
            "pred_prob": proba_row[pred],
            "elo_prob_pred": elo_probs[pred],
            "edge_vs_elo": edge,
            "agree_elo_pred": pred == elo_pred,
            "agree_elo_favorite": model_fav == elo_fav,
            "elo_pred": elo_pred,
            "elo_favorite": elo_fav,
            "odds_decimal": odds_pred,
            "pnl_unit": pnl,
        })

    return pd.DataFrame(rows)


def _summarize_subset(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {"label": label, "n": 0, "accuracy": np.nan, "roi_pct": np.nan, "pnl": 0.0}
    acc = df["correct"].mean()
    pnl = df["pnl_unit"].sum()
    roi = pnl / len(df) * 100
    return {"label": label, "n": len(df), "accuracy": acc, "roi_pct": roi, "pnl": pnl}


def sweep_max_prob(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = [0.33, 0.36, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55]
    rows = []
    for t in thresholds:
        sub = df[df["max_prob"] >= t]
        r = _summarize_subset(sub, f"max_prob>={t:.2f}")
        r["threshold"] = t
        rows.append(r)
    return pd.DataFrame(rows)


def sweep_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Rules combining confidence + Elo agreement."""
    rules = [
        ("baseline_all", df),
        ("max_prob>=0.42", df[df["max_prob"] >= 0.42]),
        ("max_prob>=0.45", df[df["max_prob"] >= 0.45]),
        ("max_prob>=0.45 & margin>=0.12", df[(df["max_prob"] >= 0.45) & (df["prob_margin"] >= 0.12)]),
        ("max_prob>=0.45 & entropy<=0.85", df[(df["max_prob"] >= 0.45) & (df["norm_entropy"] <= 0.85)]),
        ("max_prob>=0.42 & agree_elo", df[(df["max_prob"] >= 0.42) & df["agree_elo_pred"]]),
        ("max_prob>=0.42 & agree_elo_fav", df[(df["max_prob"] >= 0.42) & df["agree_elo_favorite"]]),
        ("edge>=0.05 & max_prob>=0.40", df[(df["edge_vs_elo"] >= 0.05) & (df["max_prob"] >= 0.40)]),
        ("edge>=0.08 & max_prob>=0.38", df[(df["edge_vs_elo"] >= 0.08) & (df["max_prob"] >= 0.38)]),
        ("disagree_elo & max_prob>=0.45", df[(~df["agree_elo_pred"]) & (df["max_prob"] >= 0.45)]),
    ]
    return pd.DataFrame([_summarize_subset(sub, name) for name, sub in rules])


def print_report(df: pd.DataFrame) -> None:
    n = len(df)
    base_acc = df["correct"].mean()
    base_pnl = df["pnl_unit"].sum()
    base_roi = base_pnl / n * 100

    print("=" * 72)
    print("  PROJECT ORACLE — Confidence Backtest (LOO-CV, RF)")
    print("  Settlement: 90-min 1X2 | Odds: Elo proxy + 5% margin")
    print("=" * 72)
    print(f"\n  All matches: {n} | Accuracy: {base_acc:.1%} | "
          f"P&L: {base_pnl:+.1f}u | ROI: {base_roi:+.1f}%")

    # By tournament
    print("\n" + "-" * 72)
    print("[By tournament] (bet every match at Elo odds on model pick)")
    print(f"  {'Tournament':<22s} {'N':>5s} {'Acc':>7s} {'ROI':>8s}")
    for t, g in df.groupby("short_tournament"):
        s = _summarize_subset(g, t)
        print(f"  {t:<22s} {s['n']:>5d} {s['accuracy']:>6.1%} {s['roi_pct']:>+7.1f}%")

    # Group vs knockout (90-min market)
    print("\n" + "-" * 72)
    print("[By phase] 90-min 1X2 (not 'to qualify')")
    for phase, mask in [("Group stage", ~df["is_knockout"]), ("Knockout", df["is_knockout"])]:
        s = _summarize_subset(df[mask], phase)
        print(f"  {phase:<18s} {s['n']:>5d} {s['accuracy']:>6.1%} {s['roi_pct']:>+7.1f}%")

    # max_prob sweep
    print("\n" + "-" * 72)
    print("[Sweep] min max_prob (higher = fewer bets)")
    sweep = sweep_max_prob(df)
    print(f"  {'Threshold':>12s} {'Bets':>6s} {'Acc':>7s} {'ROI':>8s} {'P&L':>8s}")
    for _, r in sweep.iterrows():
        if r["n"] == 0:
            continue
        print(f"  >={r['threshold']:.2f}       {int(r['n']):>6d} {r['accuracy']:>6.1%} "
              f"{r['roi_pct']:>+7.1f}% {r['pnl']:>+7.1f}")

    # Combined rules
    print("\n" + "-" * 72)
    print("[Selective rules] recommended filters")
    combo = sweep_combined(df)
    print(f"  {'Rule':<38s} {'Bets':>5s} {'Acc':>7s} {'ROI':>8s}")
    for _, r in combo.iterrows():
        if r["n"] == 0:
            continue
        print(f"  {r['label']:<38s} {int(r['n']):>5d} {r['accuracy']:>6.1%} {r['roi_pct']:>+7.1f}%")

    best = combo[combo["n"] >= 15].sort_values("roi_pct", ascending=False).head(1)
    if not best.empty:
        b = best.iloc[0]
        print(f"\n  Best rule (n>=15): {b['label']} — ROI {b['roi_pct']:+.1f}%, "
              f"acc {b['accuracy']:.1%} on {int(b['n'])} bets")

    # Draw-heavy note
    n_draw = (df["actual"] == LABEL_DRAW).sum()
    pred_draw = (df["pred"] == LABEL_DRAW).sum()
    print("\n" + "-" * 72)
    print("[Draw note] Model vs market on draws")
    print(f"  Actual draws: {n_draw}/{n} ({n_draw/n:.1%})")
    print(f"  Predicted draws: {pred_draw}/{n}")
    draw_bets = df[df["pred"] == LABEL_DRAW]
    if len(draw_bets):
        print(f"  When betting draw: acc {draw_bets['correct'].mean():.1%}, "
              f"ROI {draw_bets['pnl_unit'].sum()/len(draw_bets)*100:+.1f}%")

    # Knockout qualifier (separate market — informational)
    ko = df[df["is_knockout"]]
    if ko["went_to_et"].any():
        et_n = ko["went_to_et"].sum()
        print(f"\n  Knockout matches with ET: {et_n} — 90-min labels used for 1X2 above;")
        print("  'To qualify' / ET winner markets need separate models (future).")

    print("\n" + "=" * 72)
    print("  Odds without API: Elo 1X2 proxy in models/market_odds.py")
    print("  Free historical odds: football-data.co.uk (add loader in B)")
    print("  Next: B fusion, C club chemistry, O/U from Poisson lambdas")
    print("=" * 72)


def run(export_path: str | None = None) -> pd.DataFrame:
    print("\n[Confidence] Building LOO predictions (this may take ~3 min)...")
    df = build_loo_frame()
    print_report(df)
    if export_path:
        out = Path(export_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n  Exported: {out}")
    return df


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding

    fix_console_encoding()
    parser = argparse.ArgumentParser(description="Confidence selective betting backtest")
    parser.add_argument("--export", type=str, default=None, help="CSV path for per-match LOO rows")
    args = parser.parse_args()
    run(export_path=args.export)
