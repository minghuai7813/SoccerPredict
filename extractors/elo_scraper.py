"""
Pre-tournament Elo rating loader for 2022 World Cup teams.
2022 世界杯参赛队赛前 Elo 评分加载器。

Why Elo? 为什么用 Elo？
Elo 是量化国家队历史实力的最有效单一指标。
它综合了数十年的比赛结果，远比单届赛事统计可靠。
单独使用 Elo 就能预测 55-60% 的世界杯比赛结果。

Source: eloratings.net pre-tournament ratings (Nov 2022).
数据来源：eloratings.net 赛前评分（2022 年 11 月）。

Since eloratings.net may be hard to scrape reliably, we hardcode
the official pre-WC 2022 Elo ratings here. These are well-documented
public data that doesn't change.
由于 eloratings.net 可能不稳定，我们直接硬编码赛前 Elo。
这些是公开的历史数据，不会变化。

Usage / 用法:
    from extractors.elo_scraper import get_pre_wc_elo
    elo = get_pre_wc_elo()
    # Returns dict: team_name → {"elo": int, "rank": int, "confederation": str}
"""

from __future__ import annotations

# Pre-tournament Elo ratings as of Nov 20, 2022 (day before WC started).
# Confederation codes: UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC.
# 2022 年 11 月 20 日（世界杯开幕前一天）的 Elo 评分。
# 赛区代码：欧足联/南美/中北美/非洲/亚洲/大洋洲。
_PRE_WC_2022_ELO: dict[str, dict] = {
    "Brazil": {"elo": 2169, "confederation": "CONMEBOL"},
    "Argentina": {"elo": 2143, "confederation": "CONMEBOL"},
    "France": {"elo": 2048, "confederation": "UEFA"},
    "Belgium": {"elo": 2007, "confederation": "UEFA"},
    "England": {"elo": 1969, "confederation": "UEFA"},
    "Netherlands": {"elo": 1964, "confederation": "UEFA"},
    "Spain": {"elo": 1945, "confederation": "UEFA"},
    "Portugal": {"elo": 1935, "confederation": "UEFA"},
    "Denmark": {"elo": 1921, "confederation": "UEFA"},
    "Germany": {"elo": 1900, "confederation": "UEFA"},
    "Croatia": {"elo": 1883, "confederation": "UEFA"},
    "Uruguay": {"elo": 1878, "confederation": "CONMEBOL"},
    "Switzerland": {"elo": 1868, "confederation": "UEFA"},
    "Mexico": {"elo": 1861, "confederation": "CONCACAF"},
    "United States": {"elo": 1843, "confederation": "CONCACAF"},
    "Senegal": {"elo": 1837, "confederation": "CAF"},
    "Serbia": {"elo": 1825, "confederation": "UEFA"},
    "Poland": {"elo": 1814, "confederation": "UEFA"},
    "Morocco": {"elo": 1808, "confederation": "CAF"},
    "Japan": {"elo": 1798, "confederation": "AFC"},
    "South Korea": {"elo": 1786, "confederation": "AFC"},
    "Australia": {"elo": 1756, "confederation": "AFC"},
    "Canada": {"elo": 1745, "confederation": "CONCACAF"},
    "Tunisia": {"elo": 1738, "confederation": "CAF"},
    "Ecuador": {"elo": 1736, "confederation": "CONMEBOL"},
    "Iran": {"elo": 1735, "confederation": "AFC"},
    "Wales": {"elo": 1717, "confederation": "UEFA"},
    "Ghana": {"elo": 1703, "confederation": "CAF"},
    "Cameroon": {"elo": 1698, "confederation": "CAF"},
    "Costa Rica": {"elo": 1691, "confederation": "CONCACAF"},
    "Qatar": {"elo": 1662, "confederation": "AFC"},
    "Saudi Arabia": {"elo": 1654, "confederation": "AFC"},
}


def get_pre_wc_elo() -> dict[str, dict]:
    """
    Return pre-WC 2022 Elo ratings with rank computed from rating order.
    返回 2022 世界杯赛前 Elo 评分，排名由评分从高到低计算。

    Returns dict: team_name → {"elo": int, "rank": int, "confederation": str}
    """
    sorted_teams = sorted(
        _PRE_WC_2022_ELO.items(),
        key=lambda x: x[1]["elo"],
        reverse=True,
    )
    result = {}
    for rank, (team, data) in enumerate(sorted_teams, 1):
        result[team] = {
            "elo": data["elo"],
            "rank": rank,
            "confederation": data["confederation"],
        }
    return result


if __name__ == "__main__":
    elo = get_pre_wc_elo()
    print(f"{'Rank':<5} {'Team':<20} {'Elo':<6} {'Confederation'}")
    print("-" * 50)
    for team, data in sorted(elo.items(), key=lambda x: x[1]["rank"]):
        print(f"{data['rank']:<5} {team:<20} {data['elo']:<6} {data['confederation']}")
