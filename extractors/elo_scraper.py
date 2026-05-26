"""
Pre-tournament Elo rating loader using historical CSV data.
基于历史 CSV 数据的赛前 Elo 评分加载器。

Source: JGravier/soccer-elo (eloratings.net year-end snapshots 1901-2023).
数据来源：JGravier/soccer-elo（eloratings.net 年终快照 1901-2023）。

We use the year-end snapshot BEFORE each tournament as the pre-tournament Elo:
  - 2018 WC (Jun 2018) → use 2017 year-end ratings
  - Euro 2020 (Jun 2021) → use 2020 year-end ratings
  - 2022 WC (Nov 2022) → use 2021 year-end ratings
每个赛事使用其开赛前的年终 Elo 快照：
  2018 世界杯 → 2017 年终；欧洲杯 2020 → 2020 年终；2022 世界杯 → 2021 年终。

Usage / 用法:
    from extractors.elo_scraper import get_pre_tournament_elo
    elo = get_pre_tournament_elo("wc2022")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CSV_PATH = _PROJECT_ROOT / "data" / "elo_ratings_1901_2023.csv"

# Tournament key → (snapshot_year, participating_team_list_or_None)
# If team list is None, return all teams for that year.
# 赛事 key → (快照年份, 参赛队列表或 None)
_TOURNAMENT_CONFIG = {
    "wc2018": {"year": 2017},
    "euro2020": {"year": 2020},
    "wc2022": {"year": 2021},
}

# StatsBomb uses specific team names; CSV may use different ones.
# This mapping normalizes CSV names to StatsBomb names.
# StatsBomb 用特定队名；CSV 可能不同。此映射统一到 StatsBomb 命名。
_NAME_MAP = {
    "USA": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Ivory Coast": "Côte d'Ivoire",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "IR Iran": "Iran",
    "Korea DPR": "North Korea",
    "China PR": "China",
    "Cape Verde Islands": "Cape Verde",
    "Curaçao": "Curacao",
}

# Confederation assignment based on common knowledge.
# 基于常识的洲际联盟分配。
_CONFEDERATION_BY_TEAM = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Peru": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Venezuela": "CONMEBOL",
    "Bolivia": "CONMEBOL",

    "Mexico": "CONCACAF", "United States": "CONCACAF", "Costa Rica": "CONCACAF",
    "Canada": "CONCACAF", "Panama": "CONCACAF", "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "El Salvador": "CONCACAF", "Curacao": "CONCACAF",

    "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC",
    "Australia": "AFC", "Saudi Arabia": "AFC", "Qatar": "AFC",
    "Iraq": "AFC", "Uzbekistan": "AFC", "China": "AFC",
    "United Arab Emirates": "AFC", "Oman": "AFC", "Bahrain": "AFC",
    "Jordan": "AFC", "Syria": "AFC", "Palestine": "AFC",
    "India": "AFC", "Thailand": "AFC", "Vietnam": "AFC",
    "Indonesia": "AFC",

    "Nigeria": "CAF", "Cameroon": "CAF", "Ghana": "CAF",
    "Senegal": "CAF", "Morocco": "CAF", "Tunisia": "CAF",
    "Egypt": "CAF", "Algeria": "CAF", "South Africa": "CAF",
    "Côte d'Ivoire": "CAF", "Mali": "CAF", "Burkina Faso": "CAF",
    "DR Congo": "CAF", "Congo": "CAF", "Guinea": "CAF",
    "Cape Verde": "CAF", "Gabon": "CAF", "Equatorial Guinea": "CAF",
    "Zambia": "CAF", "Zimbabwe": "CAF", "Uganda": "CAF",
    "Kenya": "CAF", "Tanzania": "CAF", "Mozambique": "CAF",
    "Namibia": "CAF", "Angola": "CAF", "Benin": "CAF",
    "Togo": "CAF", "Niger": "CAF",

    "New Zealand": "OFC",
}


def _load_csv() -> pd.DataFrame:
    """Load and cache the Elo CSV."""
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"Elo CSV not found at {_CSV_PATH}. "
            "Run: python -c \"import urllib.request; "
            "urllib.request.urlretrieve("
            "'https://raw.githubusercontent.com/JGravier/soccer-elo/main/"
            "csv/ranking_soccer_1901-2023.csv', 'data/elo_ratings_1901_2023.csv')\""
        )
    return pd.read_csv(_CSV_PATH)


def _normalize_name(name: str) -> str:
    """Normalize team name to match StatsBomb conventions."""
    return _NAME_MAP.get(name, name)


def _guess_confederation(team: str) -> str:
    """Guess confederation from team name. Default to UEFA for European teams."""
    if team in _CONFEDERATION_BY_TEAM:
        return _CONFEDERATION_BY_TEAM[team]
    return "UEFA"


def get_pre_tournament_elo(tournament_key: str = "wc2022") -> dict[str, dict]:
    """
    Return pre-tournament Elo ratings from CSV data.
    从 CSV 数据返回赛前 Elo 评分。

    Parameters
    ----------
    tournament_key : str
        One of "wc2018", "euro2020", "wc2022".

    Returns
    -------
    dict: team_name → {"elo": int, "rank": int, "confederation": str}
    """
    config = _TOURNAMENT_CONFIG.get(tournament_key)
    if config is None:
        raise ValueError(f"Unknown tournament: {tournament_key}. "
                         f"Available: {list(_TOURNAMENT_CONFIG.keys())}")

    df = _load_csv()
    year_data = df[df["year"] == config["year"]].copy()

    if year_data.empty:
        raise ValueError(f"No data for year {config['year']} in CSV.")

    year_data["team"] = year_data["team"].apply(_normalize_name)
    year_data = year_data.sort_values("rating", ascending=False).reset_index(drop=True)

    result = {}
    for idx, row in year_data.iterrows():
        team = row["team"]
        result[team] = {
            "elo": int(row["rating"]),
            "rank": int(row["rank"]),
            "confederation": _guess_confederation(team),
        }

    return result


def get_pre_wc_elo() -> dict[str, dict]:
    """Backward-compatible alias for 2022 WC Elo."""
    return get_pre_tournament_elo("wc2022")


if __name__ == "__main__":
    for key in ["wc2018", "euro2020", "wc2022"]:
        elo = get_pre_tournament_elo(key)
        config = _TOURNAMENT_CONFIG[key]
        print(f"\n{'='*55}")
        print(f" {key.upper()} (year-end {config['year']}) — {len(elo)} teams")
        print(f"{'='*55}")
        sorted_teams = sorted(elo.items(), key=lambda x: x[1]["elo"], reverse=True)
        for team, data in sorted_teams[:15]:
            print(f"  {data['rank']:<4} {team:<25} {data['elo']:<6} {data['confederation']}")
        print(f"  ... ({len(elo)} total)")
