"""
Club name normalization across data sources.
跨数据源的俱乐部名称归一化。

football-data.co.uk, FBref, ClubElo, and our DB use different spellings.
"""

from __future__ import annotations

# football-data.co.uk HomeTeam/AwayTeam -> canonical name for DB / profiles.
FOOTBALL_DATA_TO_CANONICAL: dict[str, str] = {
    "Paris SG": "Paris Saint-Germain",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion",
    "Newcastle": "Newcastle United",
    "Tottenham": "Tottenham Hotspur",
    "Spurs": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Leicester": "Leicester City",
    "Norwich": "Norwich City",
    "Sheffield United": "Sheffield Utd",
    "Sheffield Utd": "Sheffield Utd",
    "Bayern Munich": "Bayern Munich",
    "Bayern": "Bayern Munich",
    "Inter": "Inter Milan",
    "Inter Milan": "Inter Milan",
    "AC Milan": "AC Milan",
    "Milan": "AC Milan",
    "Atletico Madrid": "Atlético Madrid",
    "Ath Madrid": "Atlético Madrid",
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona",
    "Dortmund": "Borussia Dortmund",
    "M'gladbach": "Borussia M'gladbach",
    "M Gladbach": "Borussia M'gladbach",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Leverkusen": "Bayer Leverkusen",
    "Marseille": "Olympique Marseille",
    "Lyon": "Olympique Lyon",
    "Monaco": "AS Monaco",
    "Lille": "Lille OSC",
    "Lens": "RC Lens",
    "Rennes": "Stade Rennais",
    "Nice": "OGC Nice",
}

# Canonical -> football-data (reverse lookup for display).
CANONICAL_TO_FOOTBALL_DATA: dict[str, str] = {}
for fd, canon in FOOTBALL_DATA_TO_CANONICAL.items():
    CANONICAL_TO_FOOTBALL_DATA.setdefault(canon, fd)


def normalize_football_data_team(name: str) -> str:
    """Map football-data team label to canonical club name."""
    n = (name or "").strip()
    return FOOTBALL_DATA_TO_CANONICAL.get(n, n)


def season_code_to_latest_league_season(season_code: str) -> str:
    """
    Map football-data folder code (e.g. '2324') to last completed FBref season
    before that campaign starts (pre-match features, no leakage).
    """
    if len(season_code) != 4 or not season_code.isdigit():
        return "2021-2022"
    start_yy = int(season_code[:2])
    prev_start = start_yy - 1
    prev_end = start_yy
    return f"20{prev_start:02d}-20{prev_end:02d}"


def league_seasons_for_match(latest: str) -> dict[str, float]:
    """Three-season recency weights ending at `latest` (YYYY-YYYY)."""
    parts = latest.split("-")
    if len(parts) != 2:
        return {"2019-2020": 0.3, "2020-2021": 0.5, "2021-2022": 1.0}
    end_year = int(parts[1])
    seasons = [
        f"{end_year - 3}-{end_year - 2}",
        f"{end_year - 2}-{end_year - 1}",
        f"{end_year - 1}-{end_year}",
    ]
    return {seasons[0]: 0.3, seasons[1]: 0.5, seasons[2]: 1.0}
