"""
FBref league stats scraper — industrial-grade rewrite.
FBref 联赛统计数据爬虫——工业级重构版。

Architecture / 架构设计:
  ┌─────────────────────────────────────────────────────────────┐
  │  1. Checkpoint: query player_stats_league for already-done  │
  │     player IDs → compute "missing" set.                     │
  │  2. Scrape: pull league-level DataFrames from FBref          │
  │     (with exponential backoff on HTTP 429).                  │
  │  3. Match: fuzzy-match FBref names against DB roster (ASCII).│
  │  4. Insert: write matched rows in batches of BATCH_SIZE.     │
  └─────────────────────────────────────────────────────────────┘

Key guarantees / 四大保障:
  1. 断点续传 (Checkpointing):
     Startup reads player_stats_league → already-scraped IDs excluded.
     Script can be killed and restarted without re-scraping finished players.
     启动时读取 player_stats_league 中已有数据的球员 ID，跳过这些球员。
     脚本可以随时中断重启而不会重复抓取。

  2. 指数退避 & 限速 (Exponential Backoff & Rate Limiting):
     Normal requests: 3-5s random delay.
     HTTP 429 / rate-limit: sleep 600s then retry (up to MAX_RETRIES).
     Never crashes on transient network errors.
     常规请求间 3-5 秒随机延时；遇到 HTTP 429 强制休眠 600 秒后重试；
     绝不因网络瞬态错误导致整个程序崩溃。

  3. 批量提交 (Batch Commit):
     Every BATCH_SIZE (10) successfully matched players → session.commit().
     Prevents unbounded memory growth and ensures partial progress is persisted.
     每匹配成功 10 个球员就提交一次，防止内存无限增长并确保部分进度被持久化。

  4. 测试模式 (CLI Test Mode):
     `--limit N` caps the number of missing players processed.
     `--limit N` 限制最多处理 N 个缺失球员，方便灰度测试。

Usage / 用法:
    python -m extractors.fbref_scraper                # full run
    python -m extractors.fbref_scraper --limit 5      # grayscale test: 5 players max
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# 确保项目根目录在 sys.path 中，以便导入 db / utils 模块。
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding
fix_console_encoding()

from sqlalchemy import create_engine, distinct
from sqlalchemy.orm import Session

from db.models import Base, Player, PlayerStatsLeague, PlayerStatsNational
from utils.entity_resolution import PlayerMatcher

# ---------------------------------------------------------------------------
# Constants / 常量
# ---------------------------------------------------------------------------

DB_PATH = _PROJECT_ROOT / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"

TARGET_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# League seasons: pre-2018 WC through pre-2022 WC.
# 联赛赛季：2018 世界杯赛前 → 2022 世界杯赛前。
TARGET_SEASONS = [
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

# Season end dates for as_of_date (approximate end of each European season).
# 每个赛季的结束日期，用作 as_of_date 时间戳。
SEASON_END_DATES = {
    "2016-2017": date(2017, 6, 1),
    "2017-2018": date(2018, 6, 1),
    "2018-2019": date(2019, 6, 1),
    "2019-2020": date(2020, 8, 1),
    "2020-2021": date(2021, 6, 1),
    "2021-2022": date(2022, 6, 1),
    "2022-2023": date(2023, 6, 1),
    "2023-2024": date(2024, 6, 1),
    "2024-2025": date(2025, 6, 1),
    "2025-2026": date(2026, 6, 1),
}

# World Cup scraping target.
# 世界杯数据爬取目标。
WC_LEAGUE = "INT-World Cup"
WC_SEASON = "2022"

# Normal inter-request delay bounds (seconds).
# 常规请求间随机延时区间（秒）。
SLEEP_MIN = 3.0
SLEEP_MAX = 5.0

# Backoff duration when rate-limited (seconds) — 10 minutes.
# 被限速时的退避时长——10 分钟。
# Why 600s? FBref 的反爬策略通常在 5-10 分钟后重置计数器，
# 600s 是保守值，宁可多等也不要被封 IP。
RATE_LIMIT_BACKOFF = 600

# Max retries per league on rate-limit before giving up.
# 单个联赛因限速导致的最大重试次数。
MAX_RETRIES = 3

# Batch commit size: flush to DB every N successfully matched players.
# 批量提交大小：每成功匹配 N 个球员就落盘一次。
BATCH_SIZE = 10


# ---------------------------------------------------------------------------
# Helpers / 工具函数
# ---------------------------------------------------------------------------

def _rate_limit() -> None:
    """
    Sleep for a random duration between SLEEP_MIN and SLEEP_MAX seconds.
    在 SLEEP_MIN 到 SLEEP_MAX 秒之间随机休眠。

    Why random instead of fixed? 为什么用随机延时而不是固定延时？
    固定间隔的请求模式容易被反爬机制识别为机器行为，
    随机延时更接近人类浏览模式。
    """
    delay = random.uniform(SLEEP_MIN, SLEEP_MAX)
    print(f"  [Rate limit] sleeping {delay:.1f}s ...")
    time.sleep(delay)


def _safe_int(value) -> int | None:
    """Convert a value to int, returning None for NaN / non-numeric. / 安全转 int。"""
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_float(value) -> float | None:
    """Convert a value to float, returning None for NaN / non-numeric. / 安全转 float。"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def _get_col(row: pd.Series, *candidates: str | tuple) -> object:
    """
    Try multiple column key candidates and return the first non-null value.
    依次尝试多个列名候选项，返回第一个非空值。

    Why is this needed? 为什么需要这个函数？
    soccerdata 返回的 DataFrame 使用 MultiIndex 列（元组），例如：
      ('Performance', 'Gls')  而不是普通的 "Gls"
    不同的 stat_type 可能有不同的列名结构。
    这个函数做容错查找，避免因列名格式变化导致全部取到 None。
    """
    for key in candidates:
        try:
            val = row[key]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                return val
        except (KeyError, IndexError):
            continue
    return None


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns to single-level strings for easier access.
    将 MultiIndex 列扁平化为单层字符串，方便取值。
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    new_cols: list[str] = []
    seen: dict[str, int] = {}

    for col_tuple in df.columns:
        parts = [str(p).strip() for p in col_tuple if str(p).strip()]
        leaf = parts[-1] if parts else str(col_tuple)

        if leaf in seen:
            leaf = "_".join(parts)
        seen[leaf] = seen.get(leaf, 0) + 1
        new_cols.append(leaf)

    df_flat = df.copy()
    df_flat.columns = new_cols
    return df_flat


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Heuristic check whether an exception is caused by HTTP 429 / rate limiting.
    启发式判断异常是否由 HTTP 429 / 频率限制引起。

    Why heuristic? 为什么用启发式而不是精确匹配？
    soccerdata 内部封装了 requests/urllib，抛出的异常类型不固定，
    可能是 HTTPError(429)、ConnectionError、或者自定义 message 中包含 "429"。
    我们通过关键词匹配来捕获所有这些变体。
    """
    exc_str = str(exc).lower()
    markers = ["429", "rate limit", "too many requests", "ratelimit"]
    return any(m in exc_str for m in markers)


# ---------------------------------------------------------------------------
# Checkpoint logic / 断点续传逻辑
# ---------------------------------------------------------------------------

def _get_already_scraped_ids_for_season(session: Session, season: str) -> set[str]:
    """
    Query player_stats_league for player IDs that already have data for a
    specific season.
    查询 player_stats_league 中某个赛季已有数据的球员 ID 集合。
    """
    rows = session.query(
        distinct(PlayerStatsLeague.internal_player_id)
    ).filter(
        PlayerStatsLeague.season == season
    ).all()
    return {r[0] for r in rows}


def _get_missing_players_for_season(
    session: Session,
    season: str,
    limit: int | None = None,
) -> list[tuple[str, str, str]]:
    """
    Return players that do NOT yet have league stats for a given season.
    返回在指定赛季中还没有联赛统计数据的球员列表。
    """
    already_done = _get_already_scraped_ids_for_season(session, season)
    print(f"[Checkpoint] {season}: {len(already_done)} players already scraped.")

    all_players = session.query(
        Player.internal_player_id,
        Player.full_name,
        Player.full_name_ascii,
    ).all()

    missing = [
        (pid, name, ascii_name)
        for pid, name, ascii_name in all_players
        if pid not in already_done
    ]
    print(f"[Checkpoint] {season}: {len(missing)} players still need data.")

    if limit is not None and limit > 0:
        missing = missing[:limit]
        print(f"[Test mode] Capped to {len(missing)} players (--limit {limit}).")

    return missing


# ---------------------------------------------------------------------------
# FBref scraping with retry / FBref 爬取（带重试）
# ---------------------------------------------------------------------------

def _read_stat_type_with_retry(league: str, season: str, stat_type: str) -> pd.DataFrame | None:
    """Fetch one FBref player stat table (standard / shooting / misc)."""
    import soccerdata as sd

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fbref = sd.FBref(leagues=league, seasons=season)
            stats_df = fbref.read_player_season_stats(stat_type=stat_type)
            return _flatten_columns(stats_df.reset_index())
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(
                    f"  [HTTP 429] Rate-limited on {league} {stat_type} "
                    f"(attempt {attempt}/{MAX_RETRIES}). "
                    f"Sleeping {RATE_LIMIT_BACKOFF}s ..."
                )
                time.sleep(RATE_LIMIT_BACKOFF)
            else:
                print(f"  [ERROR] Failed {league} {season} {stat_type}: {exc}")
                return None
    print(f"  [ABORT] Gave up on {league} {stat_type} after {MAX_RETRIES} retries.")
    return None


def _estimate_xg_from_shooting(row: pd.Series) -> float | None:
    """
    xG proxy from shots — soccerdata no longer exposes FBref xG column.
    射门数据估算 xG（soccerdata 标准表已不含 xG 列）。
    """
    sh = _safe_float(_get_col(row, "Sh"))
    sot = _safe_float(_get_col(row, "SoT"))
    if sh is None and sot is None:
        return None
    sh = sh or 0.0
    sot = sot or 0.0
    off_target = max(sh - sot, 0.0)
    return 0.32 * sot + 0.04 * off_target


def _extract_league_stat_fields(row: pd.Series) -> dict[str, object]:
    """Map merged FBref row → player_stats_league columns."""
    xg_raw = _safe_float(_get_col(row, "xG", "npxG", "Expected_xG"))
    xg = xg_raw if xg_raw is not None else _estimate_xg_from_shooting(row)
    return {
        "goals": _safe_int(_get_col(row, "Gls", "goals")),
        "assists": _safe_int(_get_col(row, "Ast", "assists")),
        "minutes_played": _safe_int(_get_col(row, "Min", "minutes")),
        "xg": xg,
        "interceptions": _safe_int(_get_col(row, "Int", "Performance_Int")),
        "tackles_won": _safe_int(_get_col(row, "TklW", "Performance_TklW", "Tkl")),
    }


def _merge_league_stat_frames(
    standard_df: pd.DataFrame,
    shooting_df: pd.DataFrame | None,
    misc_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge standard + shooting + misc on player name."""
    base = standard_df.copy()
    if "player" not in base.columns:
        return base

    if shooting_df is not None and "player" in shooting_df.columns:
        sh_cols = [c for c in ["player", "Sh", "SoT", "SoT%"] if c in shooting_df.columns]
        base = base.merge(shooting_df[sh_cols], on="player", how="left", suffixes=("", "_sh"))

    if misc_df is not None and "player" in misc_df.columns:
        mi_cols = [c for c in ["player", "Int", "TklW"] if c in misc_df.columns]
        base = base.merge(misc_df[mi_cols], on="player", how="left", suffixes=("", "_mi"))

    return base


def _scrape_league_with_retry(league: str, season: str) -> pd.DataFrame | None:
    """
    Scrape standard + shooting + misc player stats and merge.
    爬取标准/射门/杂项表并合并。
    """
    standard = _read_stat_type_with_retry(league, season, "standard")
    if standard is None or standard.empty:
        return None
    shooting = _read_stat_type_with_retry(league, season, "shooting")
    misc = _read_stat_type_with_retry(league, season, "misc")
    return _merge_league_stat_frames(standard, shooting, misc)


# ---------------------------------------------------------------------------
# Core pipeline / 核心流程
# ---------------------------------------------------------------------------

def scrape_and_store_league_stats(
    limit: int | None = None,
    seasons: list[str] | None = None,
) -> dict[str, int]:
    """
    Main scraping pipeline: loops over multiple seasons, with checkpointing,
    batching, backoff, and position extraction.
    主爬取流程：循环多个赛季，带断点续传、批量提交、指数退避及位置提取。

    Parameters
    ----------
    limit : int | None
        Per-season cap on missing players (test mode).
    seasons : list[str] | None
        Seasons to scrape. Defaults to TARGET_SEASONS (4 seasons).
    """
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)

    if seasons is None:
        seasons = TARGET_SEASONS

    stats = {
        "seasons_processed": 0,
        "leagues_scraped": 0,
        "rows_inserted": 0,
        "players_matched": 0,
        "players_unmatched": 0,
        "players_skipped_checkpoint": 0,
        "batch_commits": 0,
        "positions_updated": 0,
    }

    with Session(engine) as session:
        # Build the PlayerMatcher once (reused across all seasons).
        # 只构建一次匹配器，所有赛季复用。
        all_players_map = {
            row.full_name: row.internal_player_id
            for row in session.query(Player.full_name, Player.internal_player_id).all()
        }
        matcher = PlayerMatcher(all_players_map)
        print(f"[FBref] Matcher loaded with {matcher.roster_size} DB players.")

        # Track which players already have position set.
        # 追踪哪些球员已有位置信息，避免重复更新。
        players_with_position: set[str] = {
            r[0] for r in session.query(Player.internal_player_id).filter(
                Player.position.isnot(None)
            ).all()
        }

        for season in seasons:
            print(f"\n{'#'*60}")
            print(f"# SEASON: {season}")
            print(f"{'#'*60}")

            missing = _get_missing_players_for_season(session, season, limit=limit)
            if not missing:
                print(f"[FBref] {season}: all players done. Skipping.")
                stats["seasons_processed"] += 1
                continue

            missing_ids: set[str] = {pid for pid, _, _ in missing}
            pending_in_batch = 0
            season_date = SEASON_END_DATES.get(season, date.today())

            for league in TARGET_LEAGUES:
                if not missing_ids:
                    print(f"\n[FBref] All targeted players filled for {season}. Skipping {league}.")
                    continue

                print(f"\n{'='*60}")
                print(f"[FBref] Scraping {league} ({season})...")
                print(f"  Remaining missing: {len(missing_ids)}")
                _rate_limit()

                league_df = _scrape_league_with_retry(league, season)
                if league_df is None or league_df.empty:
                    print(f"  [SKIP] No data returned for {league} {season}.")
                    continue

                stats["leagues_scraped"] += 1
                print(f"  Got {len(league_df)} player rows.")

                for idx, row in league_df.iterrows():
                    player_name = str(row.get("player", "")).strip()
                    if not player_name or player_name == "nan":
                        try:
                            player_name = str(idx[-1]) if isinstance(idx, tuple) else str(idx)
                        except Exception:
                            continue
                    if not player_name or player_name == "nan":
                        continue

                    result = matcher.match_name(player_name, threshold=80)
                    if result.is_new:
                        stats["players_unmatched"] += 1
                        continue
                    if result.internal_player_id not in missing_ids:
                        stats["players_skipped_checkpoint"] += 1
                        continue

                    if result.internal_player_id not in players_with_position:
                        pos_raw = _get_col(row, "pos", "Pos")
                        if pos_raw:
                            pos_str = str(pos_raw).strip()
                            if pos_str and pos_str != "nan":
                                player_obj = session.get(Player, result.internal_player_id)
                                if player_obj:
                                    player_obj.position = pos_str
                                    players_with_position.add(result.internal_player_id)
                                    stats["positions_updated"] += 1

                    fields = _extract_league_stat_fields(row)
                    stat_row = PlayerStatsLeague(
                        internal_player_id=result.internal_player_id,
                        season=season,
                        as_of_date=season_date,
                        xg=fields["xg"],
                        goals=fields["goals"],
                        assists=fields["assists"],
                        passes_completed=None,
                        passes_attempted=None,
                        interceptions=fields["interceptions"],
                        tackles_won=fields["tackles_won"],
                        minutes_played=fields["minutes_played"],
                    )
                    session.add(stat_row)
                    stats["players_matched"] += 1
                    stats["rows_inserted"] += 1
                    missing_ids.discard(result.internal_player_id)

                    pending_in_batch += 1
                    print(
                        f"  [+] {player_name} -> {result.matched_name} "
                        f"(score={result.score}) "
                        f"[batch {pending_in_batch}/{BATCH_SIZE}]"
                    )

                    if pending_in_batch >= BATCH_SIZE:
                        session.commit()
                        stats["batch_commits"] += 1
                        print(
                            f"  [COMMIT] Batch #{stats['batch_commits']} flushed "
                            f"({BATCH_SIZE} players). "
                            f"Total: {stats['rows_inserted']}"
                        )
                        pending_in_batch = 0

                _rate_limit()

            # Flush remaining for this season.
            if pending_in_batch > 0:
                session.commit()
                stats["batch_commits"] += 1
                print(
                    f"\n[COMMIT] Final batch for {season} "
                    f"({pending_in_batch} players)."
                )
                pending_in_batch = 0

            stats["seasons_processed"] += 1

    print(f"\n{'='*60}")
    print("[FBref] Multi-season scraping complete. Summary:")
    print(f"  Seasons processed:     {stats['seasons_processed']}")
    print(f"  Leagues scraped:       {stats['leagues_scraped']}")
    print(f"  Players matched:       {stats['players_matched']}")
    print(f"  Rows inserted:         {stats['rows_inserted']}")
    print(f"  Positions updated:     {stats['positions_updated']}")
    print(f"  Skipped (checkpoint):  {stats['players_skipped_checkpoint']}")
    print(f"  Unmatched (not in DB): {stats['players_unmatched']}")
    print(f"  Batch commits:         {stats['batch_commits']}")
    print(f"{'='*60}")

    return stats


def backfill_advanced_league_stats(
    seasons: list[str] | None = None,
) -> dict[str, int]:
    """
    Update xG / defensive fields on existing player_stats_league rows.
    为已有联赛 stat 行回填 xG 与防守字段。
    """
    if seasons is None:
        seasons = ["2024-2025", "2025-2026"]

    engine = create_engine(DB_URL, echo=False)
    stats = {
        "seasons_processed": 0,
        "leagues_scraped": 0,
        "rows_updated": 0,
        "rows_missing": 0,
        "players_unmatched": 0,
    }

    with Session(engine) as session:
        all_players_map = {
            row.full_name: row.internal_player_id
            for row in session.query(Player.full_name, Player.internal_player_id).all()
        }
        matcher = PlayerMatcher(all_players_map)
        print(f"[Backfill] Matcher loaded with {matcher.roster_size} DB players.")

        for season in seasons:
            print(f"\n{'#'*60}\n# BACKFILL {season}\n{'#'*60}")
            pending = 0

            for league in TARGET_LEAGUES:
                print(f"\n[Backfill] {league} ({season})...")
                _rate_limit()
                league_df = _scrape_league_with_retry(league, season)
                if league_df is None or league_df.empty:
                    print("  [skip] no data")
                    continue
                stats["leagues_scraped"] += 1

                for _, row in league_df.iterrows():
                    player_name = str(row.get("player", "")).strip()
                    if not player_name or player_name == "nan":
                        continue

                    result = matcher.match_name(player_name, threshold=80)
                    if result.is_new:
                        stats["players_unmatched"] += 1
                        continue

                    existing = (
                        session.query(PlayerStatsLeague)
                        .filter(
                            PlayerStatsLeague.internal_player_id == result.internal_player_id,
                            PlayerStatsLeague.season == season,
                        )
                        .first()
                    )
                    if existing is None:
                        stats["rows_missing"] += 1
                        continue

                    fields = _extract_league_stat_fields(row)
                    existing.xg = fields["xg"]
                    existing.interceptions = fields["interceptions"]
                    existing.tackles_won = fields["tackles_won"]
                    if fields["goals"] is not None:
                        existing.goals = fields["goals"]
                    if fields["assists"] is not None:
                        existing.assists = fields["assists"]
                    if fields["minutes_played"] is not None:
                        existing.minutes_played = fields["minutes_played"]
                    stats["rows_updated"] += 1
                    pending += 1
                    if pending >= BATCH_SIZE:
                        session.commit()
                        pending = 0

                if pending:
                    session.commit()
                    pending = 0
                _rate_limit()

            stats["seasons_processed"] += 1

    print(f"\n{'='*60}")
    print("[Backfill] Complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")
    return stats


# ---------------------------------------------------------------------------
# World Cup national-team stats / 世界杯国家队数据
# ---------------------------------------------------------------------------

def scrape_and_store_world_cup_stats() -> dict[str, int]:
    """
    Scrape 2022 World Cup player stats from FBref and store in
    player_stats_national. Also backfill players.current_club from
    the FBref 'Club' column.
    从 FBref 爬取 2022 世界杯球员统计，写入 player_stats_national 表。
    同时利用 FBref 返回的 Club 字段回填 players.current_club。

    Why separate from league scraping? 为什么和联赛爬取分开？
    世界杯是国家队赛事，数据应该写入 player_stats_national 而非
    player_stats_league，这样模型可以分别提取"俱乐部状态"和"国家队状态"。

    Checkpoint: skip players that already have national stats.
    断点续传：跳过已有国家队数据的球员。
    """
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)

    stats = {
        "players_matched": 0,
        "rows_inserted": 0,
        "clubs_updated": 0,
        "players_unmatched": 0,
        "batch_commits": 0,
    }

    with Session(engine) as session:
        # Checkpoint: find players that already have national stats.
        # 断点续传：找出已有国家队数据的球员。
        already_done = {
            r[0] for r in session.query(
                distinct(PlayerStatsNational.internal_player_id)
            ).all()
        }
        print(f"[World Cup] {len(already_done)} players already have national stats.")

        # Build matcher from all DB players.
        # 用全部 DB 球员构建匹配器。
        all_players_map = {
            row.full_name: row.internal_player_id
            for row in session.query(Player.full_name, Player.internal_player_id).all()
        }
        if not all_players_map:
            print("[World Cup] No players in DB. Run statsbomb_parser first.")
            return stats

        matcher = PlayerMatcher(all_players_map)
        print(f"[World Cup] Matcher loaded with {matcher.roster_size} DB players.")

        # Scrape World Cup data.
        # 爬取世界杯数据。
        print(f"\n[World Cup] Scraping {WC_LEAGUE} ({WC_SEASON})...")
        _rate_limit()

        wc_df = _scrape_league_with_retry(WC_LEAGUE, WC_SEASON)
        if wc_df is None or wc_df.empty:
            print("[World Cup] No data returned. Aborting.")
            return stats

        print(f"  Got {len(wc_df)} player rows from FBref World Cup data.")

        pending_in_batch = 0

        for idx, row in wc_df.iterrows():
            try:
                player_name = str(idx[-1]) if isinstance(idx, tuple) else str(idx)
                team_name = str(idx[2]) if isinstance(idx, tuple) and len(idx) > 2 else None
            except Exception:
                continue
            if not player_name or player_name == "nan":
                continue

            result = matcher.match_name(player_name, threshold=80)

            if result.is_new:
                stats["players_unmatched"] += 1
                continue

            # Checkpoint: skip if already has national stats.
            # 断点续传：已有国家队数据则跳过。
            if result.internal_player_id in already_done:
                continue

            # Backfill current_club from FBref's 'Club' column.
            # 利用 FBref 的 Club 字段回填 players.current_club。
            club_raw = _get_col(row, "Club")
            if club_raw:
                club_name = str(club_raw).lstrip("0123456789. ").strip()
                if club_name:
                    player_obj = session.get(Player, result.internal_player_id)
                    if player_obj and not player_obj.current_club:
                        player_obj.current_club = club_name
                        stats["clubs_updated"] += 1

            stat_row = PlayerStatsNational(
                internal_player_id=result.internal_player_id,
                as_of_date=date(2022, 12, 18),
                xg=None,
                goals=_safe_int(_get_col(row, "Gls", "goals")),
                assists=_safe_int(_get_col(row, "Ast", "assists")),
                passes_completed=None,
                passes_attempted=None,
                interceptions=None,
                tackles_won=None,
                caps=_safe_int(_get_col(row, "MP")),
                minutes_played=_safe_int(_get_col(row, "Min", "minutes")),
            )
            session.add(stat_row)
            stats["players_matched"] += 1
            stats["rows_inserted"] += 1
            already_done.add(result.internal_player_id)

            pending_in_batch += 1
            print(
                f"  [+] {player_name} -> {result.matched_name} "
                f"(score={result.score}) "
                f"[batch {pending_in_batch}/{BATCH_SIZE}]"
            )

            if pending_in_batch >= BATCH_SIZE:
                session.commit()
                stats["batch_commits"] += 1
                print(
                    f"  [COMMIT] Batch #{stats['batch_commits']} flushed. "
                    f"Total: {stats['rows_inserted']}"
                )
                pending_in_batch = 0

        if pending_in_batch > 0:
            session.commit()
            stats["batch_commits"] += 1

    print(f"\n{'='*60}")
    print("[World Cup] Scraping complete. Summary:")
    print(f"  Players matched:       {stats['players_matched']}")
    print(f"  Rows inserted:         {stats['rows_inserted']}")
    print(f"  Clubs backfilled:      {stats['clubs_updated']}")
    print(f"  Unmatched:             {stats['players_unmatched']}")
    print(f"  Batch commits:         {stats['batch_commits']}")
    print(f"{'='*60}")

    return stats


# ---------------------------------------------------------------------------
# CLI entry point / 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli = argparse.ArgumentParser(
        description=(
            "FBref scraper for Project Oracle. "
            "Scrapes Big-5 league (multi-season) + World Cup data."
        ),
    )
    cli.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max missing players to process per season (test mode).",
    )
    cli.add_argument(
        "--season",
        type=str,
        default=None,
        help="Scrape a single season only, e.g. '2022-2023'.",
    )
    cli.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Multiple seasons for scrape or backfill.",
    )
    cli.add_argument(
        "--backfill-advanced",
        action="store_true",
        help="Fill xG/interceptions/tackles on existing league stat rows.",
    )
    cli.add_argument(
        "--world-cup",
        action="store_true",
        help="Scrape 2022 World Cup national-team stats into player_stats_national.",
    )
    cli.add_argument(
        "--all",
        action="store_true",
        help="Run both league + World Cup scraping.",
    )
    args = cli.parse_args()

    if args.seasons:
        seasons_list = args.seasons
    elif args.season:
        seasons_list = [args.season]
    else:
        seasons_list = None

    if args.backfill_advanced:
        backfill_advanced_league_stats(seasons=seasons_list)
    elif args.all:
        scrape_and_store_world_cup_stats()
        scrape_and_store_league_stats(limit=args.limit, seasons=seasons_list)
    elif args.world_cup:
        scrape_and_store_world_cup_stats()
    else:
        scrape_and_store_league_stats(limit=args.limit, seasons=seasons_list)
