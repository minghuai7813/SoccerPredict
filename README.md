# SoccerPredict

> Soccer match prediction toolkit. Initial focus: **FIFA World Cup**. Designed to extend to club leagues (Premier League, La Liga, etc.) once the core pipeline is stable.
>
> 足球比赛预测工具包。当前聚焦 **FIFA 世界杯**，结构上预留向俱乐部联赛（英超、西甲等）扩展的能力。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-46a2f1.svg)](https://github.com/astral-sh/ruff)

---

## English

### Overview

SoccerPredict is a Python project for building, training, and evaluating machine-learning models that predict soccer match outcomes. It uses [`soccerdata`](https://soccerdata.readthedocs.io/) to scrape historical match, team, and player data from public sources (FBref, FotMob, FIFA, etc.) and provides a modular pipeline:

```
soccerdata  →  data fetch  →  feature engineering  →  model training  →  prediction
```

### Project structure

```
SoccerPredict/
├── src/soccerpredict/
│   ├── data/          # Data fetchers (soccerdata wrappers) for matches, teams, players
│   ├── features/      # Feature engineering (form, ELO, head-to-head, squad strength)
│   ├── models/        # Train / predict / evaluate
│   ├── utils/         # Config, logging, paths
│   └── cli.py         # `soccerpredict` command-line entry point
├── notebooks/         # Exploratory notebooks
├── scripts/           # One-shot scripts (e.g. fetch World Cup data)
├── config/            # YAML configuration (competitions, seasons, model hyperparams)
├── data/              # raw / processed / external (gitignored)
├── tests/             # Pytest tests
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── LICENSE
```

### Quick start

```bash
git clone https://github.com/minghuai7813/SoccerPredict.git
cd SoccerPredict

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements-dev.txt
pip install -e .

cp .env.example .env   # then edit .env

# Fetch World Cup data into ./data/raw
python scripts/fetch_world_cup_data.py --season 2022

# Or use the CLI installed by the package
soccerpredict fetch --competition "FIFA World Cup" --season 2022
```

### Roadmap

- [x] Project scaffolding
- [ ] World Cup match data ingestion via `soccerdata` (FBref / FotMob)
- [ ] Team-level features: rolling form, goal diff, xG, ELO
- [ ] Player-level features: squad strength aggregates
- [ ] Baseline models: logistic regression, XGBoost, LightGBM
- [ ] Calibrated probabilistic outputs (Win / Draw / Loss)
- [ ] Backtest harness over historical World Cups
- [ ] Extend to club leagues (EPL, La Liga, Bundesliga, Serie A, Ligue 1)
- [ ] Optional: REST API + lightweight web UI

### Contributing

This is currently a personal research project. Issues and PRs are welcome once a working baseline is published.

### License

[MIT](LICENSE)

---

## 中文

### 项目简介

SoccerPredict 是一个用 Python 构建、训练、评估足球比赛预测模型的项目。底层使用 [`soccerdata`](https://soccerdata.readthedocs.io/) 从 FBref、FotMob、FIFA 等公开数据源抓取历史比赛、球队、球员数据，整体流程是：

```
soccerdata  →  数据获取  →  特征工程  →  模型训练  →  预测输出
```

当前优先做 **FIFA 世界杯**，预测维度从国家队对战开始。基础打通后，会把 `competition` 这一层抽象出来，扩展到俱乐部联赛。

### 目录结构

见上方 English 部分的 `Project structure`。简要说明：

- `src/soccerpredict/data/`：数据获取层，封装 soccerdata 的 reader。
- `src/soccerpredict/features/`：特征工程，包括近期状态、ELO、历史交锋、阵容强度等。
- `src/soccerpredict/models/`：模型训练、预测、评估。
- `notebooks/`：探索性分析。
- `scripts/`：一次性脚本（例如拉世界杯历史数据）。
- `config/`：YAML 配置（比赛、赛季、模型超参）。
- `data/`：本地数据目录，按 `raw / processed / external` 分层，大文件已 gitignore。

### 快速开始

```bash
git clone https://github.com/minghuai7813/SoccerPredict.git
cd SoccerPredict

python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

pip install -r requirements-dev.txt
pip install -e .

copy .env.example .env             # Windows
# cp .env.example .env             # macOS / Linux

# 拉取 2022 世界杯数据到 ./data/raw
python scripts/fetch_world_cup_data.py --season 2022
```

### 路线图

- [x] 项目骨架
- [ ] 通过 `soccerdata` 拉取世界杯历史比赛数据（FBref / FotMob）
- [ ] 球队级特征：近期状态、净胜球、xG、ELO
- [ ] 球员级特征：阵容强度聚合
- [ ] 基线模型：逻辑回归、XGBoost、LightGBM
- [ ] 概率校准（胜 / 平 / 负）
- [ ] 历史世界杯回测
- [ ] 扩展到俱乐部联赛（英超、西甲、德甲、意甲、法甲）
- [ ] 可选：REST API + 简易前端

### 数据来源致谢

本项目通过 `soccerdata` 间接使用 FBref / FotMob / FIFA / Understat 等公开数据，遵守各自的使用条款。本仓库本身不分发任何抓取到的原始数据。

### 许可证

[MIT](LICENSE)
