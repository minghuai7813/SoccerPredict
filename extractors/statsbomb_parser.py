"""
StatsBomb data extractor for the 2022 FIFA World Cup.
StatsBomb 2022 世界杯数据提取器。

Why StatsBomb as the *first* data source? 为什么用 StatsBomb 作为第一个数据源？
StatsBomb 提供免费的 open-data，其中包含 2022 世界杯完整的比赛事件数据。
更重要的是，StatsBomb 的 Starting XI 事件包含每个球员的详细信息（名字、球衣号、位置），
非常适合用来建立我们的球员底表 (players table)。
以此为"锚"，后续再用 FBref 等来源补充联赛统计数据。

StatsBomb open data constants:
  - competition_id = 43 (FIFA World Cup)
  - season_id = 106 (2022)

Usage / 用法:
    python -m extractors.statsbomb_parser                       # full ingestion
    python -m extractors.statsbomb_parser --test-team Argentina  # grayscale test
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `db` and `utils` are importable.
# 确保项目根目录在 sys.path 中，以便导入 db 和 utils 模块。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Fix Windows console encoding BEFORE any print() calls.
# 在任何 print() 调用之前修复 Windows 终端编码。
from utils.encoding import fix_console_encoding, to_ascii_name
fix_console_encoding()

from statsbombpy import sb

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.models import Base, Player
from utils.entity_resolution import PlayerMatcher

# StatsBomb open-data identifiers for the 2022 World Cup.
# StatsBomb 开放数据中 2022 世界杯的固定标识。
COMPETITION_ID = 43
SEASON_ID = 106

# SQLite database path (same as init_db.py).
# SQLite 数据库路径（与 init_db.py 保持一致）。
DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"


def _load_existing_players(session: Session) -> dict[str, str]:
    """
    Load all existing players from the DB into a name->UUID mapping.
    从数据库加载所有已存在的球员，返回 {名字: UUID} 映射。

    Why preload? 为什么要预加载？
    避免每次遇到一个球员名字都去查询数据库，预加载到内存后
    PlayerMatcher 可以纯内存操作，速度快很多。
    """
    rows = session.query(Player.full_name, Player.internal_player_id).all()
    return {name: pid for name, pid in rows}


def _extract_starting_xi_players(match_id: int) -> list[dict]:
    """
    Extract player info from Starting XI events of a single match.
    从单场比赛的 Starting XI 事件中提取球员信息。

    StatsBomb Starting XI event structure (simplified):
    StatsBomb 首发事件的数据结构（简化版）：
      event["tactics"]["lineup"] = [
          {"player": {"id": 123, "name": "Lionel Messi"}, "position": {...}, "jersey_number": 10},
          ...
      ]

    Returns a list of dicts, each with keys: sb_player_id, player_name, jersey_number, position.
    返回字典列表，每个字典包含：sb_player_id, player_name, jersey_number, position。
    """
    events = sb.events(match_id=match_id)

    # Filter to Starting XI event type.
    # 筛选出 Starting XI 类型的事件。
    starting_xi_events = events[events["type"] == "Starting XI"]

    players_found: list[dict] = []

    for _, row in starting_xi_events.iterrows():
        tactics = row.get("tactics")
        if not tactics or not isinstance(tactics, dict):
            continue

        lineup = tactics.get("lineup", [])
        if not isinstance(lineup, list):
            continue

        for entry in lineup:
            player_info = entry.get("player", {})
            position_info = entry.get("position", {})

            player_name = player_info.get("name")
            if not player_name:
                continue

            players_found.append({
                "sb_player_id": player_info.get("id"),
                "player_name": player_name,
                "jersey_number": entry.get("jersey_number"),
                "position": position_info.get("name"),
            })

    return players_found


def _create_player_record(player_name: str, player_id: str) -> Player:
    """
    Build a Player ORM instance with both Unicode and ASCII names.
    构建同时包含 Unicode 原名和 ASCII 转写名的 Player ORM 实例。

    Why auto-generate full_name_ascii here? 为什么在这里自动生成 ASCII 名？
    这是球员首次入库的唯一入口（StatsBomb 是我们的"真相源"）。
    在入口处统一生成 ASCII 名，确保数据库中每个球员都有 ASCII 版本，
    下游的 FBref 匹配器可以直接使用。
    """
    return Player(
        internal_player_id=player_id,
        full_name=player_name,
        full_name_ascii=to_ascii_name(player_name),
    )


def ingest_world_cup_players() -> dict[str, int]:
    """
    Main ingestion pipeline: pull all 2022 WC matches from StatsBomb,
    extract Starting XI players, resolve names, and persist to DB.
    主入库流程：拉取 2022 世界杯所有比赛，提取首发球员，对齐名字，写入数据库。

    Returns
    -------
    dict[str, int]
        Summary stats: {"matches_processed", "players_new", "players_existing"}.
        汇总统计：处理了多少场比赛、新增了多少球员、匹配到了多少已有球员。
    """
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)

    stats = {"matches_processed": 0, "players_new": 0, "players_existing": 0}

    with Session(engine) as session:
        # Step 1: Preload existing players for fuzzy matching.
        # 第一步：预加载已有球员用于模糊匹配。
        existing = _load_existing_players(session)
        matcher = PlayerMatcher(existing)
        print(f"[StatsBomb] Preloaded {matcher.roster_size} existing players from DB.")

        # Step 2: Get all match IDs for the 2022 World Cup.
        # 第二步：获取 2022 世界杯所有比赛 ID。
        matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
        match_ids = matches["match_id"].tolist()
        print(f"[StatsBomb] Found {len(match_ids)} matches for 2022 World Cup.")

        # Step 3: Iterate over each match and extract Starting XI.
        # 第三步：遍历每场比赛，提取首发球员。
        for i, mid in enumerate(match_ids, start=1):
            print(f"[StatsBomb] Processing match {i}/{len(match_ids)} (match_id={mid})...")

            try:
                players_in_match = _extract_starting_xi_players(mid)
            except Exception as exc:
                # Graceful degradation: skip a match if its data is malformed.
                # 容错：如果某场比赛数据异常就跳过，不中断整个流程。
                print(f"  [WARN] Failed to parse match {mid}: {exc}")
                continue

            for p in players_in_match:
                result = matcher.match_name(p["player_name"])

                if result.is_new:
                    new_player = _create_player_record(
                        p["player_name"], result.internal_player_id
                    )
                    session.add(new_player)
                    stats["players_new"] += 1
                else:
                    stats["players_existing"] += 1

            stats["matches_processed"] += 1

        # Commit all new players in one transaction for efficiency.
        # 一次性提交所有新球员，减少数据库 I/O。
        session.commit()

    print(
        f"\n[StatsBomb] Done. "
        f"Matches: {stats['matches_processed']}, "
        f"New players: {stats['players_new']}, "
        f"Already known: {stats['players_existing']}"
    )
    return stats


def ingest_single_team_test(team_name: str = "Argentina") -> dict[str, int]:
    """
    Grayscale test: only process matches involving one specific team.
    灰度测试：只处理指定球队参与的比赛。

    Why test with a single team first? 为什么先拿单支球队测试？
    全量跑 64 场比赛之前，先用一支球队（~7 场）验证：
    1. StatsBomb API 调用是否正常
    2. Starting XI 解析逻辑是否正确
    3. 球员写入数据库时是否有主键冲突
    4. 重复运行时模糊匹配是否能正确识别已有球员

    Parameters
    ----------
    team_name : str
        Team to filter on. Default "Argentina".
        要筛选的球队名，默认"Argentina"。
    """
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)

    stats = {"matches_processed": 0, "players_new": 0, "players_existing": 0}

    with Session(engine) as session:
        existing = _load_existing_players(session)
        matcher = PlayerMatcher(existing)
        print(f"[TEST] Preloaded {matcher.roster_size} existing players from DB.")

        matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)

        # Filter to matches where the target team is either home or away.
        # 筛选出目标球队作为主队或客队的比赛。
        team_matches = matches[
            (matches["home_team"] == team_name)
            | (matches["away_team"] == team_name)
        ]
        match_ids = team_matches["match_id"].tolist()
        print(f"[TEST] Found {len(match_ids)} matches involving {team_name}.")

        if not match_ids:
            all_teams = sorted(
                set(matches["home_team"].tolist() + matches["away_team"].tolist())
            )
            print(f"[TEST] Available teams: {all_teams}")
            return stats

        for i, mid in enumerate(match_ids, start=1):
            home = team_matches.loc[team_matches["match_id"] == mid, "home_team"].iloc[0]
            away = team_matches.loc[team_matches["match_id"] == mid, "away_team"].iloc[0]
            print(f"[TEST] Match {i}/{len(match_ids)}: {home} vs {away} (id={mid})")

            try:
                players_in_match = _extract_starting_xi_players(mid)
            except Exception as exc:
                print(f"  [WARN] Failed to parse match {mid}: {exc}")
                continue

            for p in players_in_match:
                result = matcher.match_name(p["player_name"])
                ascii_name = to_ascii_name(p["player_name"])

                if result.is_new:
                    new_player = _create_player_record(
                        p["player_name"], result.internal_player_id
                    )
                    session.add(new_player)
                    stats["players_new"] += 1
                    print(
                        f"  [NEW]  {p['player_name']}  (ascii: {ascii_name})  "
                        f"->  {result.internal_player_id[:8]}..."
                    )
                else:
                    stats["players_existing"] += 1
                    print(
                        f"  [MATCH] {p['player_name']}  ~  {result.matched_name} "
                        f"(score={result.score}, id={result.internal_player_id[:8]}...)"
                    )

            stats["matches_processed"] += 1

        session.commit()

    print(
        f"\n[TEST] Done. "
        f"Matches: {stats['matches_processed']}, "
        f"New players: {stats['players_new']}, "
        f"Already known: {stats['players_existing']}"
    )

    # Verify: read back from DB to confirm data was persisted.
    # 验证：从数据库回读确认数据已落盘。
    with Session(engine) as session:
        total_in_db = session.query(Player).count()
        print(f"[TEST] Total players now in DB: {total_in_db}")

        # Show a sample of stored names (original vs ASCII) for verification.
        # 打印几条示例记录（原名 vs ASCII 名）供验证。
        sample = session.query(
            Player.full_name, Player.full_name_ascii
        ).limit(5).all()
        print("[TEST] Sample records (original -> ASCII):")
        for orig, ascii_n in sample:
            print(f"  {orig}  ->  {ascii_n}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StatsBomb World Cup 2022 ingestion")
    parser.add_argument(
        "--test-team",
        type=str,
        default=None,
        help="Run grayscale test for a single team (e.g. 'Argentina'). "
             "Omit to run full ingestion.",
    )
    args = parser.parse_args()

    if args.test_team:
        ingest_single_team_test(team_name=args.test_team)
    else:
        ingest_world_cup_players()
