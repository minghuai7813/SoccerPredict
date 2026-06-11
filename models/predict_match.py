"""
Single-match prediction API — roster-based, club or national.
单场比赛预测：基于名单，支持俱乐部与国家队。

Trains on historical international matches (179); club predictions use
league-weighted roster profiles + optional ClubElo. RF output is
best-effort out-of-domain for clubs until club match labels are added.

Usage / 用法:
    python -m models.predict_match --home-type club --home "Paris SG" \\
        --away-type club --away Arsenal --home-elo 1964 --away-elo 2053

    python -m models.predict_match --home-type national --home France \\
        --away-type national --away England --elo-key wc2022
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.match_features import profiles_to_dataframe
from features.team_profile import PlayerSlot, build_club_profile, build_national_profile
from models.market_odds import (
    decimal_odds,
    elo_1x2_probs,
    expected_goals_from_elo,
    poisson_score_matrix,
)

LABEL_NAMES = {1: "主胜", 0: "平局", -1: "客胜"}


def _top_scorelines(lam_h: float, lam_a: float, n: int = 8) -> list[tuple[int, int, float]]:
    mat = poisson_score_matrix(lam_h, lam_a, max_g=8)
    scored = [(mat[i, j], i, j) for i in range(mat.shape[0]) for j in range(mat.shape[1])]
    scored.sort(reverse=True)
    return [(h, a, float(p)) for p, h, a in scored[:n]]


def _poisson_wdl(lam_h: float, lam_a: float) -> dict[int, float]:
    mat = poisson_score_matrix(lam_h, lam_a, max_g=8)
    hw = sum(mat[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i > j)
    dr = sum(mat[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i == j)
    aw = sum(mat[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i < j)
    return {1: float(hw), 0: float(dr), -1: float(aw)}


def _parse_roster(raw: str | None) -> list[PlayerSlot] | None:
    """Parse 'Name:90,Name2:45' into PlayerSlot list."""
    if not raw:
        return None
    slots: list[PlayerSlot] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, mins = part.rsplit(":", 1)
            slots.append(PlayerSlot(
                name=name.strip(),
                expected_minutes=float(mins),
                is_starter=float(mins) >= 60,
            ))
        else:
            slots.append(PlayerSlot(name=part, expected_minutes=90.0, is_starter=True))
    return slots or None


_NATIONAL_ELO_KEYS = frozenset({"wc2018", "euro2020", "wc2022", "wc2026"})
_ml_model_cache: dict[tuple, tuple] = {}


def _maybe_infer_national_type(name: str, side_type: str, elo_key: str) -> str:
    """
    When elo_key targets a national tournament, infer national side type
    from WC2026 roster membership if caller left the default 'club'.
    大赛 Elo 键 + 默认 club 时，若队名在 WC2026 名单中则自动切为 national。
    """
    if side_type != "club" or elo_key not in _NATIONAL_ELO_KEYS:
        return side_type
    if elo_key == "wc2026" and name in _get_wc2026_rosters():
        return "national"
    return side_type


_WC2026_ROSTERS_PATH = _PROJECT_ROOT / "data" / "wc2026_rosters.json"
_wc2026_rosters_cache: dict | None = None


def _get_wc2026_rosters() -> dict:
    """Lazy-load WC2026 roster mapping. 延迟加载 WC2026 名单映射。"""
    global _wc2026_rosters_cache
    if _wc2026_rosters_cache is None:
        import json
        if _WC2026_ROSTERS_PATH.exists():
            with open(_WC2026_ROSTERS_PATH, encoding="utf-8") as f:
                _wc2026_rosters_cache = json.load(f)
        else:
            _wc2026_rosters_cache = {}
    return _wc2026_rosters_cache


def _load_side_profile(
    side_type: str,
    name: str,
    roster: list[PlayerSlot] | None,
    elo: float | None,
    elo_rank: int,
    elo_key: str,
) -> dict[str, float]:
    if side_type == "club":
        return build_club_profile(
            name,
            roster=roster,
            club_elo=elo if elo is not None else 1500.0,
            club_elo_rank=elo_rank,
        )
    if side_type == "national":
        team_pids = None
        if roster is None and elo_key == "wc2026":
            rosters = _get_wc2026_rosters()
            team_data = rosters.get(name, [])
            if team_data:
                team_pids = set(
                    p["player_id"] for p in team_data if p.get("player_id")
                )
        return build_national_profile(
            name, roster=roster, elo_key=elo_key, team_pids=team_pids,
        )
    raise ValueError(f"Unknown side type: {side_type}")


def _train_domain(home_type: str, away_type: str, domain: str) -> str:
    if domain in ("club", "national"):
        return domain
    if home_type == "club" and away_type == "club":
        return "club"
    if home_type == "national" and away_type == "national":
        return "national"
    return "national"


def _fit_ml_models(
    domain: str,
    max_train_matches: int | None,
) -> tuple[list[str], SimpleImputer, StandardScaler, RandomForestClassifier, pd.DataFrame, dict]:
    """Load training data and fit imputer/scaler/RF; return artifacts + meta."""
    cache_key = (domain, max_train_matches)
    if cache_key in _ml_model_cache:
        return _ml_model_cache[cache_key]

    if domain == "club":
        from features.club_match_dataset import build_club_match_dataset
        X, y, meta = build_club_match_dataset(max_matches=max_train_matches)
        label = "club domestic (EPL+Ligue1)"
    else:
        from features.match_dataset import build_match_dataset
        X, y, meta = build_match_dataset()
        label = "international (WC/Euro)"

    if X.empty:
        raise ValueError(f"No training data for domain={domain}")

    fn = X.columns.tolist()
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=fn)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp), columns=fn)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=4,
        random_state=42, class_weight="balanced",
    )
    rf.fit(X_scaled, y)
    print(f"\n[Train] Domain: {label} | {len(y)} matches | {len(fn)} features")
    artifacts = (fn, imp, scaler, rf, X_imp, meta)
    _ml_model_cache[cache_key] = artifacts
    return artifacts


def predict_match(
    home_name: str,
    away_name: str,
    *,
    home_type: str = "club",
    away_type: str = "club",
    home_roster: list[PlayerSlot] | None = None,
    away_roster: list[PlayerSlot] | None = None,
    home_elo: float | None = None,
    away_elo: float | None = None,
    home_elo_rank: int = 50,
    away_elo_rank: int = 50,
    elo_key: str = "wc2022",
    venue: str = "neutral",
    train_model: bool = True,
    domain: str = "auto",
    max_train_matches: int | None = None,
) -> dict:
    """
    Predict W/D/L and scorelines for one match.
    预测单场胜平负与比分分布。
    """
    home_type = _maybe_infer_national_type(home_name, home_type, elo_key)
    away_type = _maybe_infer_national_type(away_name, away_type, elo_key)

    home_adv = 0.0 if venue == "neutral" else 60.0

    home_feats = _load_side_profile(
        home_type, home_name, home_roster, home_elo, home_elo_rank, elo_key,
    )
    away_feats = _load_side_profile(
        away_type, away_name, away_roster, away_elo, away_elo_rank, elo_key,
    )

    h_elo = float(home_elo if home_elo is not None else home_feats.get("elo_rating", 1500))
    a_elo = float(away_elo if away_elo is not None else away_feats.get("elo_rating", 1500))

    elo_probs = elo_1x2_probs(h_elo, a_elo, home_adv=home_adv)
    lam_h, lam_a = expected_goals_from_elo(h_elo, a_elo, home_adv=home_adv, total_goals=2.65)
    poi_wdl = _poisson_wdl(lam_h, lam_a)
    scores = _top_scorelines(lam_h, lam_a)

    result: dict = {
        "home": home_name,
        "away": away_name,
        "home_type": home_type,
        "away_type": away_type,
        "home_elo": h_elo,
        "away_elo": a_elo,
        "elo_1x2": elo_probs,
        "poisson_lambdas": (lam_h, lam_a),
        "poisson_1x2": poi_wdl,
        "top_scorelines": scores,
        "home_profile_dims": {
            "attack": home_feats.get("pos_fw_attack_score"),
            "defense": home_feats.get("pos_df_defense_per90"),
            "creative": home_feats.get("pos_mf_creative_per90"),
            "press": home_feats.get("team_pressures_per90"),
        },
        "away_profile_dims": {
            "attack": away_feats.get("pos_fw_attack_score"),
            "defense": away_feats.get("pos_df_defense_per90"),
            "creative": away_feats.get("pos_mf_creative_per90"),
            "press": away_feats.get("team_pressures_per90"),
        },
        "model_1x2": None,
        "model_pick": None,
        "model_poisson_lambdas": None,
        "model_top_scorelines": None,
        "train_domain": None,
        "note": None,
    }

    train_dom = _train_domain(home_type, away_type, domain)
    result["train_domain"] = train_dom

    if not train_model:
        return result

    try:
        fn, imp, scaler, rf, X_imp, train_meta = _fit_ml_models(train_dom, max_train_matches)
    except ValueError:
        result["note"] = "训练数据为空，仅返回 Elo/泊松结果。"
        return result

    if train_dom == "club":
        result["note"] = "RF/Poisson 在俱乐部国内联赛赛果上训练（roster+联赛特征）。"
    else:
        result["note"] = "RF/Poisson 在国家队大赛赛果上训练。"

    classes = [int(c) for c in rf.classes_]
    idx = {c: i for i, c in enumerate(classes)}

    X_match = profiles_to_dataframe(home_feats, away_feats, fn)
    X_match_imp = pd.DataFrame(imp.transform(X_match), columns=fn)
    X_match_s = pd.DataFrame(scaler.transform(X_match_imp), columns=fn)
    proba = rf.predict_proba(X_match_s)[0]
    pick = int(rf.predict(X_match_s)[0])
    model_probs = {c: float(proba[idx[c]]) for c in classes if c in idx}

    X_pos = X_imp.clip(lower=0)
    ph = PoissonRegressor(alpha=1.0, max_iter=1000).fit(X_pos, train_meta["home_score"])
    pa = PoissonRegressor(alpha=1.0, max_iter=1000).fit(X_pos, train_meta["away_score"])
    X_m_pos = X_match_imp.clip(lower=0)
    ml_h = float(np.clip(ph.predict(X_m_pos)[0], 0.05, 6.0))
    ml_a = float(np.clip(pa.predict(X_m_pos)[0], 0.05, 6.0))

    result["model_1x2"] = model_probs
    result["model_pick"] = pick
    result["model_poisson_lambdas"] = (ml_h, ml_a)
    result["model_top_scorelines"] = _top_scorelines(ml_h, ml_a)

    return result


def _print_report(r: dict) -> None:
    print("=" * 72)
    print(f"  {r['home']} vs {r['away']}  ({r['home_type']} / {r['away_type']})")
    print(f"  Elo: {r['home_elo']:.0f} vs {r['away_elo']:.0f}")
    print("=" * 72)

    print("\n[画像维度] attack / defense / creative / press")
    for side in ("home", "away"):
        d = r[f"{side}_profile_dims"]
        print(f"  {side}: {d}")

    print("\n[Elo 胜平负]")
    for k in (1, 0, -1):
        p = r["elo_1x2"][k]
        print(f"  {LABEL_NAMES[k]:4s}  {p:6.1%}  (~{decimal_odds(p):.2f})")

    lh, la = r["poisson_lambdas"]
    print(f"\n[Elo 泊松] λ主={lh:.2f}  λ客={la:.2f}")
    print("[Elo 泊松 胜平负]")
    for k in (1, 0, -1):
        print(f"  {LABEL_NAMES[k]:4s}  {r['poisson_1x2'][k]:6.1%}")

    print("\n[比分 Top 8 — Elo 泊松]")
    for i, (h, a, p) in enumerate(r["top_scorelines"], 1):
        print(f"  {i:2d}. {h}-{a}  {p:5.1%}")

    if r.get("model_1x2"):
        dom = r.get("train_domain", "?")
        print(f"\n[RF 模型 — {dom} 训练域]")
        for k, p in sorted(r["model_1x2"].items(), key=lambda x: -x[1]):
            print(f"  {LABEL_NAMES.get(k, str(k)):4s}  {p:6.1%}")
        print(f"  >>> 首选: {LABEL_NAMES.get(r['model_pick'], r['model_pick'])}")
        ml_h, ml_a = r["model_poisson_lambdas"]
        print(f"\n[RF 泊松] λ主={ml_h:.2f}  λ客={ml_a:.2f}")
        print("[比分 Top 8 — RF 泊松]")
        for i, (h, a, p) in enumerate(r["model_top_scorelines"], 1):
            print(f"  {i:2d}. {h}-{a}  {p:5.1%}")

    if r.get("note"):
        print(f"\n[说明] {r['note']}")
    print("=" * 72)


def main() -> None:
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    parser = argparse.ArgumentParser(description="Roster-based match prediction")
    parser.add_argument("--home", required=True, help="Home entity name")
    parser.add_argument("--away", required=True, help="Away entity name")
    parser.add_argument("--home-type", choices=["club", "national"], default="club")
    parser.add_argument("--away-type", choices=["club", "national"], default="club")
    parser.add_argument("--home-roster", default=None, help="Comma names, optional :minutes")
    parser.add_argument("--away-roster", default=None, help="Comma names, optional :minutes")
    parser.add_argument("--home-elo", type=float, default=None)
    parser.add_argument("--away-elo", type=float, default=None)
    parser.add_argument("--elo-key", default="wc2026", choices=["wc2018", "euro2020", "wc2022", "wc2026"])
    parser.add_argument("--venue", choices=["neutral", "home"], default="neutral")
    parser.add_argument("--domain", choices=["auto", "club", "national"], default="auto")
    parser.add_argument("--max-train-matches", type=int, default=None)
    parser.add_argument("--no-train", action="store_true", help="Skip RF (Elo/Poisson only)")

    args = parser.parse_args()
    out = predict_match(
        args.home,
        args.away,
        home_type=args.home_type,
        away_type=args.away_type,
        home_roster=_parse_roster(args.home_roster),
        away_roster=_parse_roster(args.away_roster),
        home_elo=args.home_elo,
        away_elo=args.away_elo,
        elo_key=args.elo_key,
        venue=args.venue,
        train_model=not args.no_train,
        domain=args.domain,
        max_train_matches=args.max_train_matches,
    )
    _print_report(out)


if __name__ == "__main__":
    main()
