# Project Oracle

> A quantitative football prediction engine targeting the FIFA World Cup.
>
> 面向 FIFA 世界杯的量化足球预测引擎。

## Overview / 项目概述

Project Oracle is a multi-modal sports prediction system that combines player-level statistics, team dynamics, Elo ratings, and historical national-team performance to forecast World Cup match outcomes — and measures whether it has an edge over the market consensus.

Project Oracle 是一个多模态体育预测系统，结合球员统计、球队动态、Elo 评分和历史国家队表现来预测世界杯比赛结果，并衡量模型是否具有超越市场共识的 alpha。

## Current Results / 当前成绩

| Model | Metric | Score |
|-------|--------|-------|
| Random Forest (3-class) | LOO-CV Accuracy | **57.8%** (baseline 33.3%) |
| Random Forest (regression) | MAE Goal Diff | **1.29 goals** |
| Elo-only baseline | Accuracy | 50.0% |
| **Model edge vs market** | | **+7.8%** |

**Alpha analysis**: On 28 matches where our model disagrees with Elo (market), we are correct 13 times vs market's 8 — demonstrating genuine alpha. The model correctly predicted 13/32 upsets that Elo missed, including Croatia-Brazil (draw), Spain-Germany (draw), and Qatar-Ecuador (away win).

Top predictive features: Elo rank, league goals (recency-weighted), Elo rating differential, league goal involvement.

## Tech Stack / 技术栈

- **Language:** Python 3.10+
- **Database:** SQLite (MVP) via SQLAlchemy ORM
- **Data Sources:** `statsbombpy` (2018 & 2022 WC + Euro 2020 events), `soccerdata` (FBref Big-5 league stats, 4 seasons)
- **ML:** scikit-learn (Random Forest baseline)
- **Entity Resolution:** `thefuzz` + `unidecode` (ASCII-first fuzzy matching)
- **External Signals:** Elo ratings, confederation encoding

## Database Schema / 数据库结构

| Table | Rows | Description |
|-------|------|-------------|
| `players` | 514 | Player biographical data + position + ASCII names (UUID PK) |
| `player_stats_league` | 1,038 | Big-5 league stats across 4 seasons (2018-2022) |
| `player_stats_national` | 340 | National team stats (2018 WC + Euro 2020 event data) |
| `match_events` | — | Live match events (red cards, formations, injuries) |

## Feature Engineering / 特征工程

33 features per team (99 per match with home/away/diff):

| Group | Count | Examples |
|-------|-------|---------|
| League (recency-weighted) | 10 | Weighted goals/assists/minutes, per-90, trend slope, consistency |
| National team | 11 | xG, pass completion rate, defensive actions, caps, competition count |
| Position | 5 | FW/MF/DF ratios, positional goal distribution |
| Elo | 2 | Pre-tournament Elo rating + rank |
| Confederation | 5 | One-hot: UEFA, CONMEBOL, CONCACAF, CAF, AFC |

## Project Structure / 项目结构

```
SoccerProject/
├── .cursorrules              # AI coding standards & architecture blueprint
├── db/
│   └── models.py             # SQLAlchemy ORM models (4 tables)
├── extractors/
│   ├── statsbomb_parser.py   # StatsBomb 2022 WC player ingestion
│   ├── statsbomb_events.py   # Event-level stats (xG, passes, tackles)
│   ├── euro2020_events.py    # Euro 2020 national team stats
│   ├── fbref_scraper.py      # FBref multi-season league stats (checkpoint, backoff)
│   └── elo_scraper.py        # Pre-WC Elo ratings for all 32 teams
├── utils/
│   ├── encoding.py           # UTF-8 console fix + ASCII transliteration
│   └── entity_resolution.py  # Fuzzy player name matching (token_set_ratio)
├── features/
│   ├── team_features.py      # Player stats → team-level feature matrix
│   └── match_dataset.py      # Match results + features → ML-ready dataset
├── models/
│   └── match_predictor.py    # Random Forest + alpha analysis vs Elo market
├── init_db.py                # Database initialization script
├── requirements.txt          # Python dependencies
└── README.md
```

## Quick Start / 快速开始

```bash
pip install -r requirements.txt
python init_db.py
python -m extractors.statsbomb_parser       # Ingest 2022 WC players
python -m extractors.fbref_scraper          # Scrape 4 seasons of Big-5 league stats
python -m extractors.euro2020_events        # Extract Euro 2020 national team data
python -m models.match_predictor            # Train model + alpha analysis
```

## License

[MIT](LICENSE)
