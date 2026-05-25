"""
ORM models for Project Oracle MVP database.
Project Oracle MVP 数据库的 ORM 模型定义。

Defines the four core tables specified in the architecture blueprint:
根据架构蓝图定义四张核心表：
  - players:              球员基本信息 / Player biographical data
  - player_stats_league:  联赛时序统计 / League-level time-series stats
  - player_stats_national:国家队时序统计 / National-team time-series stats
  - match_events:         比赛实时事件 / Live match events & state
"""

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base class.
    所有 ORM 模型的基类，用于自动建表。
    """
    pass


def _generate_uuid() -> str:
    """
    Generate a UUID4 string as the default player ID.
    生成 UUID4 字符串，用作球员内部唯一标识符。
    为什么用 UUID 而不是自增 ID？因为未来可能合并多个数据源，UUID 可以避免主键冲突。
    """
    return str(uuid.uuid4())


class Player(Base):
    """
    Core player biographical table.
    球员核心档案表——存储不随时间变化的基础信息（身高、生日等）。

    Why separate from stats? 为什么和统计数据分开？
    球员的身高体重等属性是相对固定的，而比赛统计是时间序列数据，
    分表存储符合数据库范式设计，也方便后续做 feature engineering。
    """

    __tablename__ = "players"

    # Primary key: UUID string to ensure uniqueness across data sources.
    # 主键使用 UUID 字符串，确保跨数据源合并时不会冲突。
    internal_player_id = Column(
        String(36),
        primary_key=True,
        default=_generate_uuid,
        comment="UUID v4 primary key",
    )
    full_name = Column(String(128), nullable=False, comment="Player full name")
    dob = Column(Date, nullable=True, comment="Date of birth")
    height_cm = Column(Float, nullable=True, comment="Height in centimeters")
    weight_kg = Column(Float, nullable=True, comment="Weight in kilograms")
    current_club = Column(String(128), nullable=True, comment="Current club name")
    league_tier = Column(
        Integer,
        nullable=True,
        comment="League tier level (1=top flight, 2=second division, etc.)",
    )

    # Relationships: one player -> many stat rows.
    # 关系映射：一个球员对应多条时序统计记录。
    league_stats = relationship(
        "PlayerStatsLeague", back_populates="player", cascade="all, delete-orphan"
    )
    national_stats = relationship(
        "PlayerStatsNational", back_populates="player", cascade="all, delete-orphan"
    )


class PlayerStatsLeague(Base):
    """
    Time-series league statistics per player.
    球员联赛级别的时序统计数据。

    Why time-series (as_of_date)? 为什么用时间序列？
    球员状态随赛季推进不断变化，我们需要保留历史快照，
    这样模型才能学到"近期状态"这类动态特征。
    """

    __tablename__ = "player_stats_league"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to players table.
    # 外键关联到 players 表。
    internal_player_id = Column(
        String(36),
        ForeignKey("players.internal_player_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="FK to players.internal_player_id",
    )

    # Snapshot timestamp: when this stat row was recorded / scraped.
    # 快照时间戳：这条统计数据对应的日期。
    # 用 Date 而非 DateTime，因为联赛统计通常以"天"为粒度。
    as_of_date = Column(
        Date, nullable=False, index=True, comment="Date this stat snapshot refers to"
    )

    # Core performance metrics from the architecture blueprint.
    # 架构蓝图要求的核心表现指标。
    xg = Column(Float, nullable=True, comment="Expected goals (xG)")
    goals = Column(Integer, nullable=True, comment="Goals scored")
    assists = Column(Integer, nullable=True, comment="Assists")
    passes_completed = Column(Integer, nullable=True, comment="Passes completed")
    passes_attempted = Column(Integer, nullable=True, comment="Passes attempted")
    interceptions = Column(Integer, nullable=True, comment="Interceptions")
    tackles_won = Column(Integer, nullable=True, comment="Tackles won")
    minutes_played = Column(Integer, nullable=True, comment="Minutes played")

    # Relationship back to player.
    # 反向关系映射回球员。
    player = relationship("Player", back_populates="league_stats")


class PlayerStatsNational(Base):
    """
    Time-series national team statistics per player.
    球员国家队级别的时序统计数据。

    Why separate from league stats? 为什么和联赛统计分开？
    国家队比赛频率、对手强度、战术体系和联赛完全不同，
    分开存储便于模型分别提取"俱乐部状态"和"国家队状态"两套特征。
    对于世界杯预测来说，国家队历史表现可能比联赛数据更有直接参考价值。
    """

    __tablename__ = "player_stats_national"

    id = Column(Integer, primary_key=True, autoincrement=True)

    internal_player_id = Column(
        String(36),
        ForeignKey("players.internal_player_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="FK to players.internal_player_id",
    )

    # Snapshot date for national team duty stats.
    # 国家队统计数据的快照日期。
    as_of_date = Column(
        Date, nullable=False, index=True, comment="Date this stat snapshot refers to"
    )

    # Metrics mirror the league table but in a national-team context.
    # 指标和联赛表类似，但上下文是国家队比赛。
    xg = Column(Float, nullable=True, comment="Expected goals (xG)")
    goals = Column(Integer, nullable=True, comment="Goals scored")
    assists = Column(Integer, nullable=True, comment="Assists")
    passes_completed = Column(Integer, nullable=True, comment="Passes completed")
    passes_attempted = Column(Integer, nullable=True, comment="Passes attempted")
    interceptions = Column(Integer, nullable=True, comment="Interceptions")
    tackles_won = Column(Integer, nullable=True, comment="Tackles won")
    caps = Column(Integer, nullable=True, comment="Total international caps as of date")
    minutes_played = Column(Integer, nullable=True, comment="Minutes played")

    player = relationship("Player", back_populates="national_stats")


class MatchEvent(Base):
    """
    Live match events and state tracking.
    比赛实时事件和状态追踪表。

    Why this table? 为什么需要这张表？
    比赛中的实时事件（红牌、伤病、换人、阵型变化）会直接影响比赛走向。
    记录这些事件，未来可以做 in-play 预测模型，也可以用作赛后复盘的特征。
    """

    __tablename__ = "match_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Match identification.
    # 比赛标识。
    match_id = Column(
        String(64), nullable=False, index=True, comment="External match identifier"
    )

    # Event timestamp: when this event happened during the match.
    # 事件时间戳：这个事件在比赛中的发生时间。
    event_timestamp = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        comment="UTC timestamp of the event",
    )

    # Match minute (e.g. 45, 90+3).
    # 比赛分钟数。
    match_minute = Column(Integer, nullable=True, comment="Minute of the match")

    # Event classification.
    # 事件分类：formation_change / red_card / injury / substitution / goal 等。
    event_type = Column(
        String(64),
        nullable=False,
        comment="Event type: formation_change, red_card, injury, substitution, goal, etc.",
    )

    # Current formation snapshot after this event.
    # 该事件发生后的当前阵型快照。
    formation = Column(
        String(16), nullable=True, comment="Formation string, e.g. '4-3-3'"
    )

    # Affected player, if applicable.
    # 涉及的球员（如果有的话）。
    player_id = Column(
        String(36), nullable=True, comment="Player involved in the event (UUID or external ID)"
    )

    # Free-form detail for extra context (JSON-friendly string).
    # 自由格式的额外上下文信息（可存 JSON 字符串）。
    detail = Column(
        Text, nullable=True, comment="Extra event detail, JSON-encoded if needed"
    )
