# Project Oracle

> A quantitative football prediction engine targeting the FIFA World Cup.
>
> 面向 FIFA 世界杯的量化足球预测引擎。

## Overview / 项目概述

Project Oracle is a multi-modal sports prediction system that combines player-level statistics, team dynamics, and live match events to forecast World Cup match outcomes.

Project Oracle 是一个多模态体育预测系统，结合球员级别统计数据、球队动态和比赛实时事件来预测世界杯比赛结果。

## Tech Stack / 技术栈

- **Language:** Python 3.10+
- **Database:** SQLite (MVP) via SQLAlchemy ORM
- **Data Sources:** `soccerdata`, `statsbombpy`

## Database Schema / 数据库结构

| Table | Description |
|-------|-------------|
| `players` | Player biographical data (UUID primary key) / 球员基础档案 |
| `player_stats_league` | Time-series league stats per player / 球员联赛时序统计 |
| `player_stats_national` | Time-series national team stats / 球员国家队时序统计 |
| `match_events` | Live match events (red cards, formations, injuries) / 比赛实时事件 |

## Quick Start / 快速开始

```bash
# Install dependencies / 安装依赖
pip install -r requirements.txt

# Initialize the database / 初始化数据库
python init_db.py
```

## Project Structure / 项目结构

```
SoccerProject/
├── .cursorrules        # AI coding standards & architecture blueprint
├── db/
│   ├── __init__.py
│   └── models.py       # SQLAlchemy ORM models (4 tables)
├── init_db.py          # Database initialization script
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT
└── README.md
```

## License

[MIT](LICENSE)
