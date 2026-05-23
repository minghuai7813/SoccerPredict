# Data Directory

This directory holds local datasets. Large files are gitignored — only `.gitkeep` placeholders are tracked.

## Layout

- `raw/` — Untouched data as fetched from sources (e.g. soccerdata cache exports, manual CSVs).
- `processed/` — Cleaned, joined, feature-engineered tables ready for modeling.
- `external/` — Third-party reference data (e.g. FIFA ranking snapshots, betting odds dumps).

## Conventions

- Use Parquet (`.parquet`) for processed tables — fast and typed.
- Use CSV only for human-inspectable snapshots.
- Never commit personal API keys, tokens, or proprietary feeds.

## 中文说明

`raw/` 放原始抓取数据，`processed/` 放清洗后的特征表，`external/` 放第三方参考数据（FIFA 排名、博彩赔率等）。大文件均被 gitignore，仓库里只保留 `.gitkeep` 占位。
