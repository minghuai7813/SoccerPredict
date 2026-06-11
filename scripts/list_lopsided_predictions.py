"""List LOO predictions for large Elo-gap matches."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features.match_dataset import build_match_dataset
from extractors.elo_scraper import get_pre_tournament_elo
from models.match_predictor import LABEL_MAP, _elo_predict
from models.profit_analysis import _elo_win_prob


def main() -> None:
    X, y, meta = build_match_dataset()
    fn = X.columns.tolist()
    Xs = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=fn)
    Xs = Xs.replace([np.inf, -np.inf], 0)
    Xs = pd.DataFrame(StandardScaler().fit_transform(Xs), columns=fn)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    y_pred = cross_val_predict(rf, Xs, y, cv=LeaveOneOut())

    elo_map = {
        "FIFA World Cup 2018": "wc2018",
        "UEFA Euro 2020": "euro2020",
        "FIFA World Cup 2022": "wc2022",
    }
    caches = {k: get_pre_tournament_elo(v) for k, v in elo_map.items()}

    rows = []
    for i in range(len(y)):
        h = meta.iloc[i]["home_team"]
        a = meta.iloc[i]["away_team"]
        t = meta.iloc[i]["tournament"]
        ed = caches[t]
        eh = ed.get(h, {}).get("elo", 1500)
        ea = ed.get(a, {}).get("elo", 1500)
        gap = abs(eh - ea)
        fav = h if eh >= ea else a
        fav_elo, dog_elo = max(eh, ea), min(eh, ea)
        fp = _elo_win_prob(fav_elo, dog_elo)
        elo_p, _ = _elo_predict(h, a, ed)
        act, pr = int(y.iloc[i]), int(y_pred[i])
        rows.append({
            "match": f"{h} vs {a}",
            "tournament": t,
            "stage": meta.iloc[i].get("stage", ""),
            "elo_gap": gap,
            "fav_prob": fp,
            "favorite": fav,
            "score": f"{int(meta.iloc[i]['home_score'])}-{int(meta.iloc[i]['away_score'])}",
            "actual": LABEL_MAP[act],
            "model_pred": LABEL_MAP[pr],
            "elo_pred": LABEL_MAP[elo_p],
            "model_ok": act == pr,
            "elo_ok": act == elo_p,
            "fav_won": (act == 1 and fav == h) or (act == -1 and fav == a),
        })

    df = pd.DataFrame(rows)
    for thresh in (200, 150):
        sub = df[df["elo_gap"] >= thresh].sort_values("elo_gap", ascending=False)
        if sub.empty:
            continue
        hit = int(sub["model_ok"].sum())
        print(f"\n=== Elo gap >= {thresh}: {len(sub)} matches | RF correct {hit}/{len(sub)} ===\n")
        for _, r in sub.iterrows():
            mark = "OK" if r["model_ok"] else "MISS"
            tag = "" if r["fav_won"] else " UPSET"
            print(
                f"[{mark}] gap={r['elo_gap']:.0f} fav_win~{r['fav_prob']:.0%} | "
                f"{r['match']} ({r['score']}) | actual={r['actual']} model={r['model_pred']} "
                f"fav={r['favorite']}{tag} | {r['tournament']} {r['stage']}"
            )


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    main()
