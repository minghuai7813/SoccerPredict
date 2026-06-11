"""
Download and parse domestic league results from football-data.co.uk.
下载并解析 football-data.co.uk 五大联赛赛果 CSV。

Free CSVs cover EPL, Ligue 1, Bundesliga, La Liga, Serie A — used as club
training labels until dedicated UCL CSVs are added.

Usage / 用法:
    python -m extractors.football_data_loader
    python -m extractors.football_data_loader --seasons 2324 2425
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from features.club_names import normalize_football_data_team, season_code_to_latest_league_season

DATA_DIR = _PROJECT_ROOT / "data" / "football_data"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# League file codes on football-data.co.uk
LEAGUE_FILES = {
    "EPL": "E0",
    "Ligue1": "F1",
    "Bundesliga": "D1",
    "LaLiga": "SP1",
    "SerieA": "I1",
}

DEFAULT_SEASONS = ["2122", "2223", "2324", "2425", "2526"]


def _parse_date(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def download_season_league(season_code: str, league_key: str, force: bool = False) -> Path:
    """Download one CSV to data/football_data/{season}_{league}.csv."""
    league_file = LEAGUE_FILES[league_key]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{season_code}_{league_file}.csv"
    if out.exists() and not force:
        return out

    url = BASE_URL.format(season=season_code, league=league_file)
    print(f"[football-data] GET {url}")
    try:
        raw = urllib.request.urlopen(url, timeout=30).read()
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"Cannot download {url}: {exc}") from exc
    out.write_bytes(raw)
    return out


def parse_match_csv(path: Path, season_code: str, league_key: str) -> pd.DataFrame:
    """Parse one football-data CSV into unified match rows."""
    text = path.read_bytes().decode("latin-1")
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "HomeTeam" not in df.columns:
        return pd.DataFrame()

    rows = []
    latest_league = season_code_to_latest_league_season(season_code)
    for _, r in df.iterrows():
        home = normalize_football_data_team(str(r.get("HomeTeam", "")))
        away = normalize_football_data_team(str(r.get("AwayTeam", "")))
        if not home or not away:
            continue
        try:
            hg = int(r["FTHG"])
            ag = int(r["FTAG"])
        except (ValueError, TypeError, KeyError):
            continue

        dt = _parse_date(str(r.get("Date", "")))
        result = 1 if hg > ag else (-1 if hg < ag else 0)

        rows.append({
            "date": dt,
            "season_code": season_code,
            "league": league_key,
            "home_team": home,
            "away_team": away,
            "home_score": hg,
            "away_score": ag,
            "result": result,
            "latest_league_season": latest_league,
            "source_file": path.name,
        })

    return pd.DataFrame(rows)


def load_club_matches(
    seasons: list[str] | None = None,
    leagues: list[str] | None = None,
    download: bool = True,
) -> pd.DataFrame:
    """
    Load (and optionally download) club match results across leagues/seasons.
    加载多联赛、多赛季俱乐部赛果。
    """
    seasons = seasons or DEFAULT_SEASONS
    leagues = leagues or list(LEAGUE_FILES.keys())
    frames: list[pd.DataFrame] = []

    for season_code in seasons:
        for league_key in leagues:
            path = DATA_DIR / f"{season_code}_{LEAGUE_FILES[league_key]}.csv"
            if download or not path.exists():
                try:
                    download_season_league(season_code, league_key, force=False)
                except FileNotFoundError as exc:
                    print(f"  [skip] {exc}")
                    continue
            if not path.exists():
                continue
            part = parse_match_csv(path, season_code, league_key)
            if not part.empty:
                frames.append(part)
                print(f"  {season_code} {league_key}: {len(part)} matches")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    from utils.encoding import fix_console_encoding
    fix_console_encoding()

    parser = argparse.ArgumentParser(description="Download football-data.co.uk CSVs")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--leagues", nargs="+", default=list(LEAGUE_FILES.keys()))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force:
        for s in args.seasons:
            for lk in args.leagues:
                download_season_league(s, lk, force=True)

    df = load_club_matches(seasons=args.seasons, leagues=args.leagues, download=True)
    print(f"\n[football-data] Total: {len(df)} matches")
    if not df.empty:
        print(df.groupby("league").size().to_string())


if __name__ == "__main__":
    main()
