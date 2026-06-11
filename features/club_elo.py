"""
Running Club Elo from match results (pre-match ratings, no leakage).
基于赛果滚动更新的俱乐部 Elo（赛前评分，无泄漏）。
"""

from __future__ import annotations

import math

HOME_ADV = 60.0
ELO_SCALE = 400.0
K_FACTOR = 22.0
DEFAULT_ELO = 1500.0


def expected_score(rating_a: float, rating_b: float, home_adv: float = 0.0) -> float:
    diff = rating_a + home_adv - rating_b
    return 1.0 / (1.0 + 10 ** (-diff / ELO_SCALE))


def update_elo(
    rating_home: float,
    rating_away: float,
    home_goals: int,
    away_goals: int,
    *,
    home_adv: float = HOME_ADV,
    k: float = K_FACTOR,
) -> tuple[float, float]:
    """Update ratings after one match (draw = 0.5 / 0.5)."""
    exp_h = expected_score(rating_home, rating_away, home_adv)
    if home_goals > away_goals:
        act_h, act_a = 1.0, 0.0
    elif home_goals < away_goals:
        act_h, act_a = 0.0, 1.0
    else:
        act_h, act_a = 0.5, 0.5
    margin = abs(home_goals - away_goals)
    k_eff = k * (1.0 + 0.15 * min(margin, 3))
    new_h = rating_home + k_eff * (act_h - exp_h)
    new_a = rating_away + k_eff * (act_a - (1.0 - exp_h))
    return new_h, new_a


class ClubEloTracker:
    """
    Chronological Elo state; call snapshot before each match.
    按时间顺序维护 Elo；每场比赛前取 snapshot。
    """

    def __init__(self, default: float = DEFAULT_ELO) -> None:
        self._ratings: dict[str, float] = {}
        self._default = default
        self._games: dict[str, int] = {}

    def get(self, team: str) -> float:
        return self._ratings.get(team, self._default)

    def snapshot(self, home: str, away: str) -> tuple[float, float]:
        return self.get(home), self.get(away)

    def apply_result(self, home: str, away: str, hg: int, ag: int) -> None:
        rh, ra = self.get(home), self.get(away)
        nh, na = update_elo(rh, ra, hg, ag)
        self._ratings[home] = nh
        self._ratings[away] = na
        self._games[home] = self._games.get(home, 0) + 1
        self._games[away] = self._games.get(away, 0) + 1

    def rank(self, team: str) -> int:
        """Rank among known teams (1 = highest)."""
        sorted_teams = sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)
        for i, (t, _) in enumerate(sorted_teams, 1):
            if t == team:
                return i
        return len(sorted_teams) + 1

    def seed(self, team: str, elo: float) -> None:
        self._ratings[team] = elo

    @property
    def team_count(self) -> int:
        return len(self._ratings)

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)[:n]
