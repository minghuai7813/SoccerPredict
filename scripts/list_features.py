"""
List all match-level features used by the models.
列出当前模型使用的全部比赛级特征。

Usage:
    python scripts/list_features.py
    python scripts/list_features.py --export data/feature_catalog.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Base team-level feature names (before home_/away_/diff_ prefix).
TEAM_FEATURE_GROUPS = {
    "Elo / 大洲": [
        "elo_rating", "elo_rank",
        "conf_UEFA", "conf_CONMEBOL", "conf_CONCACAF", "conf_CAF", "conf_AFC",
    ],
    "联赛 (FBref, 加权)": [
        "lg_goals_weighted", "lg_assists_weighted", "lg_minutes_weighted",
        "lg_goal_involvement_weighted", "lg_player_count", "lg_seasons_available",
        "lg_goals_per90", "lg_goals_max", "lg_goal_trend", "lg_consistency",
    ],
    "联赛阵容结构": [
        "pos_fw_ratio", "pos_mf_ratio", "pos_df_ratio",
        "pos_fw_goals", "pos_mf_goals",
    ],
    "国家队聚合 (StatsBomb 事件, 加权)": [
        "nt_competitions_count",
        *[f"nt_w_{f}" for f in [
            "xg", "goals", "shots", "shots_on_target", "key_passes", "through_balls",
            "crosses", "progressive_passes", "dribble_attempts", "dribble_success",
            "carries_count", "progressive_carry_distance", "interceptions", "blocks",
            "clearances", "pressures", "counter_pressures", "tackle_attempts",
            "tackles_won", "goalkeeper_saves",
        ]],
    ],
    "明星 / 阵容深度": [
        "star_top3_xg", "star_top_scorer_share", "star_xg_gini",
        "star_max_dribbles", "squad_starters", "squad_depth",
    ],
    "位置质量 (per90)": [
        "pos_fw_xg_per90", "pos_fw_dribbles_per90", "pos_fw_attack_score",
        "pos_df_defense_per90", "pos_mf_creative_per90", "pos_mf_carries_per90",
        "pos_gk_saves",
    ],
    "球队风格 (per90)": [
        "team_pressures_per90", "team_carries_per90",
    ],
    "俱乐部化学反应 (同俱乐部队友)": [
        "club_max_cluster_ratio", "club_pair_density", "club_distinct_clubs",
    ],
    "对位 (仅比赛级, 无 home/away 前缀)": [
        "matchup_home_fw_vs_away_df",
        "matchup_away_fw_vs_home_df",
        "matchup_mf_creative_diff",
        "matchup_press_vs_carry",
        "matchup_carry_vs_press",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=str, default=None)
    args = parser.parse_args()

    from features.match_dataset import build_match_dataset

    print("Building dataset to verify column names...")
    X, _, _ = build_match_dataset()

    lines = []
    lines.append("PROJECT ORACLE — Feature Catalog")
    lines.append("=" * 60)
    lines.append(f"Match matrix: {X.shape[0]} rows x {X.shape[1]} columns")
    lines.append("")
    lines.append("Naming: most team stats appear as home_*, away_*, diff_*")
    lines.append("(diff = home - away). Matchup_* are match-level only.")
    lines.append("")

    catalog_bases: set[str] = set()
    for bases in TEAM_FEATURE_GROUPS.values():
        catalog_bases.update(bases)

    for group, bases in TEAM_FEATURE_GROUPS.items():
        lines.append(f"## {group} ({len(bases)} base names)")
        for b in bases:
            for prefix in ("home", "away", "diff"):
                col = f"{prefix}_{b}"
                ok = "ok" if col in X.columns else "—"
                if b.startswith("matchup_"):
                    break
            if b.startswith("matchup_"):
                ok = "ok" if b in X.columns else "—"
                lines.append(f"  {b}  [{ok}]")
            else:
                lines.append(f"  {b}")
                lines.append(f"    -> home_{b}, away_{b}, diff_{b}")
        lines.append("")

    # Verify all X columns accounted for
    expected = set()
    for b in catalog_bases:
        if b.startswith("matchup_"):
            expected.add(b)
        else:
            expected.update(f"{p}_{b}" for p in ("home", "away", "diff"))

    extra = sorted(set(X.columns) - expected)
    missing = sorted(expected - set(X.columns))
    if extra:
        lines.append("## Extra columns in matrix (not in catalog)")
        for c in extra:
            lines.append(f"  {c}")
    if missing:
        lines.append("## Catalog entries missing from matrix")
        for c in missing:
            lines.append(f"  {c}")

    lines.append("")
    lines.append(f"Total columns in X: {len(X.columns)}")

    text = "\n".join(lines)
    print("\n" + text)

    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\nExported: {out}")


if __name__ == "__main__":
    from utils.encoding import fix_console_encoding
    fix_console_encoding()
    main()
