"""Process soccerdata-cached HTML files for SUI/MEX/JPN + retry failed leagues."""
import json, sys
from datetime import date
from io import StringIO
from pathlib import Path
import pandas as pd
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding, to_ascii_name
fix_console_encoding()
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from thefuzz import fuzz
from db.models import Player, PlayerStatsLeague

DB_URL = f"sqlite:///{_PROJECT_ROOT / 'oracle_mvp.db'}"
ROSTERS_PATH = _PROJECT_ROOT / "data" / "wc2026_rosters.json"
CACHE_DIR = Path(r"C:\Users\mengt\soccerdata\data\FBref")

CACHED_LEAGUES = {
    "SUI-Super League": {
        "standard": CACHE_DIR / "players_SUI-Super League_2425_standard.html",
        "clubs": ["fc zurich", "young boys", "servette", "lugano",
                  "st gallen", "luzern", "sion", "grasshopper", "winterthur", "yverdon"],
    },
    "MEX-Liga MX": {
        "standard": CACHE_DIR / "players_MEX-Liga MX_2425_standard.html",
        "clubs": ["club america", "unam", "pumas", "chivas", "cruz azul", "toluca",
                  "atlas", "tijuana", "pachuca", "mazatlan", "leon", "santos laguna",
                  "monterrey", "tigres"],
    },
    "JPN-J1 League": {
        "standard": CACHE_DIR / "players_JPN-J1 League_2025_standard.html",
        "clubs": ["kashima", "fc tokyo", "sanfrecce hiroshima", "albirex niigata",
                  "machida zelvia", "yokohama", "kawasaki", "urawa", "vissel kobe"],
    },
}


def _classify(club, clubs):
    cl = club.lower()
    for p in clubs:
        if p in cl:
            return True
    return False


def _load_targets(clubs):
    rosters = json.loads(ROSTERS_PATH.read_text(encoding="utf-8"))
    targets = {}
    for team, players in rosters.items():
        for p in players:
            if p.get("match_status") == "matched":
                continue
            if _classify(p.get("club", ""), clubs):
                targets[p["player_id"]] = p
    return targets


def _parse_html(path):
    """Parse FBref HTML table from cached file."""
    html = path.read_text(encoding="utf-8")
    tables = pd.read_html(StringIO(html))
    if not tables:
        return None
    df = max(tables, key=len)
    return df


def _extract(df):
    cols = [str(c) for c in df.columns]
    def _find(*pats):
        for pat in pats:
            for c in cols:
                if pat.lower() in c.lower():
                    return c
        return None

    player_col = _find("Player", "player")
    gls = _find("Gls")
    ast = _find("Ast")
    xg = _find("xG")
    mn = _find("Min")
    if player_col is None:
        return []

    results = []
    for _, row in df.iterrows():
        name = str(row.get(player_col, "")).strip()
        if not name or name == "nan" or "Player" in name:
            continue
        def _n(c):
            if c is None: return None
            v = row.get(c)
            if pd.isna(v): return None
            try: return float(str(v).replace(",",""))
            except: return None
        results.append({
            "name": name,
            "ascii": to_ascii_name(name),
            "goals": int(_n(gls)) if _n(gls) is not None else None,
            "assists": int(_n(ast)) if _n(ast) is not None else None,
            "xg": _n(xg),
            "minutes_played": int(_n(mn)) if _n(mn) is not None else None,
        })
    return results


def _update(stats, targets, session, as_of):
    updated = 0
    for pid, entry in targets.items():
        name = entry["name"]
        asc = to_ascii_name(name)
        best, bs, bstat = None, 0, None
        for s in stats:
            sc = fuzz.token_set_ratio(asc, s["ascii"])
            if sc > bs:
                bs = sc
                best = s["name"]
                bstat = s
        if bs >= 80 and bstat:
            ex = session.query(PlayerStatsLeague).filter_by(
                internal_player_id=pid, season="2024-2025"
            ).first()
            if not ex:
                ex = session.query(PlayerStatsLeague).filter_by(
                    internal_player_id=pid
                ).order_by(PlayerStatsLeague.as_of_date.desc()).first()
            if ex:
                for k in ("goals", "assists", "xg", "minutes_played"):
                    v = bstat.get(k)
                    if v is not None:
                        setattr(ex, k, v)
                ex.as_of_date = as_of
            updated += 1
            print(f"  UPD {name:30s} <- {best:30s} (score={bs}) G={bstat.get('goals')} xG={bstat.get('xg')}")
    return updated


def main():
    engine = create_engine(DB_URL, echo=False)
    total = 0
    with Session(engine) as session:
        for league, cfg in CACHED_LEAGUES.items():
            targets = _load_targets(cfg["clubs"])
            if not targets:
                print(f"[{league}] no targets, skip")
                continue
            print(f"\n[{league}] {len(targets)} targets")
            html_path = cfg["standard"]
            if not html_path.exists():
                print(f"  Cache file not found: {html_path}")
                continue
            df = _parse_html(html_path)
            if df is None:
                print("  No table")
                continue
            stats = _extract(df)
            print(f"  Parsed {len(stats)} players")
            n = _update(stats, targets, session, date(2025, 6, 1))
            total += n
            print(f"  Result: {n}/{len(targets)}")
            session.commit()

    print(f"\nTOTAL: {total} updated from cache")


if __name__ == "__main__":
    main()
