"""Data layer: thin wrappers around `soccerdata` readers."""

from soccerpredict.data.fetch import fetch_matches, fetch_schedule
from soccerpredict.data.players import fetch_player_stats
from soccerpredict.data.teams import fetch_team_stats

__all__ = [
    "fetch_matches",
    "fetch_player_stats",
    "fetch_schedule",
    "fetch_team_stats",
]
