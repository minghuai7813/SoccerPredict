"""
StatsBomb 2018 World Cup roster ingestion (all lineup players).
StatsBomb 2018 世界杯阵容入库（含替补，不仅首发）。

competition_id=43, season_id=3

Usage / 用法:
    python -m extractors.statsbomb_wc2018_parser
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding, to_ascii_name

fix_console_encoding()

from statsbombpy import sb
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Base, Player
from utils.entity_resolution import PlayerMatcher

COMPETITION_ID = 43
SEASON_ID = 3

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"


def _extract_lineup_players(match_id: int) -> list[dict]:
    """
    All players listed in match lineups (starters + subs).
    从 lineup 提取该场所有登记球员（含替补）。
    """
    import warnings
    warnings.filterwarnings("ignore")

    players: list[dict] = []
    seen: set[str] = set()

    try:
        lineups = sb.lineups(match_id=match_id)
    except Exception:
        return players

    for _, roster in lineups.items():
        if roster is None or roster.empty:
            continue
        for _, row in roster.iterrows():
            pname = row.get("player_name") or row.get("player_nickname")
            if not pname or str(pname) in seen:
                continue
            seen.add(str(pname))
            players.append({"player_name": str(pname)})

    return players


def ingest_wc2018_players() -> dict[str, int]:
    """Ingest 2018 WC squad players into players table."""
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)

    stats = {"matches_processed": 0, "players_new": 0, "players_existing": 0}

    with Session(engine) as session:
        existing = {
            r.full_name: r.internal_player_id
            for r in session.query(Player.full_name, Player.internal_player_id).all()
        }
        matcher = PlayerMatcher(existing)
        print(f"[WC2018] DB players loaded: {matcher.roster_size}")

        matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
        match_ids = matches["match_id"].tolist()
        print(f"[WC2018] {len(match_ids)} matches")

        for i, mid in enumerate(match_ids, 1):
            try:
                roster = _extract_lineup_players(mid)
            except Exception as exc:
                print(f"  match {mid} FAIL: {exc}")
                continue

            for p in roster:
                result = matcher.match_name(p["player_name"])
                if result.is_new:
                    session.add(Player(
                        internal_player_id=result.internal_player_id,
                        full_name=p["player_name"],
                        full_name_ascii=to_ascii_name(p["player_name"]),
                    ))
                    stats["players_new"] += 1
                else:
                    stats["players_existing"] += 1

            stats["matches_processed"] += 1
            if i % 10 == 0:
                print(f"  {i}/{len(match_ids)} matches...")

        session.commit()

    total = stats["players_new"] + stats["players_existing"]
    print(
        f"\n[WC2018] Done. Matches={stats['matches_processed']} "
        f"new={stats['players_new']} known={stats['players_existing']} "
        f"touch={total}"
    )
    return stats


if __name__ == "__main__":
    ingest_wc2018_players()
