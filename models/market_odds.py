"""
Elo-based market proxy for 1X2, totals, and scorelines (no bookmaker data required).
基于 Elo 的盘口代理：胜平负、大小球、比分概率（无需真实赔率）。

Real odds sources (for future B):
  - football-data.co.uk (free CSV, many leagues + some internationals)
  - The Odds API, Betfair historical (paid)
  - oddsportal scrapes (ToS risk)

Label convention in this project: 90-minute regulation (group + KO).
淘汰赛「晋级」与 90 分钟 1X2 是不同市场，见 confidence_backtest 分表。

Usage / 用法:
    from models.market_odds import elo_1x2_probs, decimal_odds
"""

from __future__ import annotations

import numpy as np

LABEL_HOME = 1
LABEL_DRAW = 0
LABEL_AWAY = -1

# Empirical draw rate in our 179-match set (~23%); international tournaments ~25–28%.
BASE_DRAW_RATE = 0.26
HOME_ADVANTAGE = 60.0
ELO_SCALE = 400.0


def elo_home_win_prob(home_elo: float, away_elo: float, home_adv: float = HOME_ADVANTAGE) -> float:
    """P(home wins in a decisive result, ignoring draw mass)."""
    diff = home_elo + home_adv - away_elo
    return 1.0 / (1.0 + 10 ** (-diff / ELO_SCALE))


def elo_1x2_probs(
    home_elo: float,
    away_elo: float,
    *,
    home_adv: float = HOME_ADVANTAGE,
    base_draw: float = BASE_DRAW_RATE,
) -> dict[int, float]:
    """
    Three-way probabilities (90-min W/D/L) from Elo gap + draw prior.
    胜平负隐含概率，和为 1。
    """
    diff = home_elo + home_adv - away_elo
    p_hw = elo_home_win_prob(home_elo, away_elo, home_adv)
    closeness = float(np.exp(-abs(diff) / 150.0))
    p_draw = base_draw * (0.55 + 0.45 * closeness)
    p_draw = float(np.clip(p_draw, 0.08, 0.38))
    p_dec = 1.0 - p_draw
    p_home = p_dec * p_hw
    p_away = p_dec * (1.0 - p_hw)
    s = p_home + p_draw + p_away
    return {LABEL_HOME: p_home / s, LABEL_DRAW: p_draw / s, LABEL_AWAY: p_away / s}


def decimal_odds(prob: float, margin: float = 0.05) -> float:
    """Fair-ish decimal odds with bookmaker margin on implied prob."""
    p = max(prob, 0.02)
    return (1.0 - margin) / p


def expected_goals_from_elo(
    home_elo: float,
    away_elo: float,
    *,
    total_goals: float = 2.55,
    home_adv: float = HOME_ADVANTAGE,
) -> tuple[float, float]:
    """
    Split expected total goals by Elo strength (for Poisson markets).
    将预期总进球按实力差分给主客队。
    """
    diff = home_elo + home_adv - away_elo
    share_home = 1.0 / (1.0 + 10 ** (-diff / ELO_SCALE))
    lam_h = total_goals * share_home
    lam_a = total_goals * (1.0 - share_home)
    return max(lam_h, 0.15), max(lam_a, 0.15)


def poisson_score_matrix(
    lam_home: float,
    lam_away: float,
    max_g: int = 8,
) -> np.ndarray:
    from scipy.stats import poisson

    mat = np.zeros((max_g + 1, max_g + 1))
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            mat[i, j] = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
    return mat / mat.sum()


def over_under_probs(
    lam_home: float,
    lam_away: float,
    line: float = 2.5,
    max_g: int = 8,
) -> tuple[float, float]:
    """P(total goals > line), P(under). 大小球概率。"""
    mat = poisson_score_matrix(lam_home, lam_away, max_g)
    over = under = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            t = i + j
            if t > line:
                over += mat[i, j]
            elif t < line:
                under += mat[i, j]
            # push on integer line (e.g. exactly 2.0) ignored for 2.5
    s = over + under
    if s <= 0:
        return 0.5, 0.5
    return over / s, under / s


def result_probs_from_matrix(mat: np.ndarray) -> dict[int, float]:
    """W/D/L from score probability matrix."""
    p_home = float(np.triu(mat, k=1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.tril(mat, k=-1).sum())
    return {LABEL_HOME: p_home, LABEL_DRAW: p_draw, LABEL_AWAY: p_away}
