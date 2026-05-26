"""
Lineup minutes from StatsBomb lineups.
从 StatsBomb 阵容数据计算出场时间。

Used to distinguish starters vs bench and weight player contributions.
用于区分主力/替补并对球员贡献加权。
"""

from __future__ import annotations

from typing import Any


def _minutes_from_positions(positions: Any) -> tuple[int, int]:
    """
    Parse positions list [{from, to, position}, ...] -> (minutes, started).
    解析 positions 列表得到 (minutes, started_as_xi)。
    """
    if positions is None:
        return 0, 0
    if isinstance(positions, str):
        return 0, 0

    total_mins = 0
    started = 0
    try:
        items = list(positions)
    except TypeError:
        return 0, 0

    for stint in items:
        if not isinstance(stint, dict):
            continue
        frm = stint.get("from")
        to = stint.get("to")
        if frm is None or to is None:
            continue
        try:
            mins = int(float(to) - float(frm))
        except (TypeError, ValueError):
            continue
        if mins < 0:
            mins = 0
        total_mins += mins
        if float(frm) == 0.0:
            started = 1

    return total_mins, started


def aggregate_lineup_minutes_for_match(match_id: int, sb_module: Any) -> dict[str, dict]:
    """
    Return player_name -> {minutes, starts, team} for one match.
    返回单场比赛 player_name -> {minutes, starts, team}。
    """
    import warnings
    warnings.filterwarnings("ignore")

    result: dict[str, dict] = {}
    try:
        lineups = sb_module.lineups(match_id=match_id)
    except Exception:
        return result

    for team_name, roster in lineups.items():
        if roster is None or roster.empty:
            continue
        for _, row in roster.iterrows():
            pname = row.get("player_name") or row.get("player_nickname")
            if not pname or (isinstance(pname, float) and str(pname) == "nan"):
                continue
            pname = str(pname)

            positions = row.get("positions")
            mins, started = _minutes_from_positions(positions)

            # Fallback: listed in lineup with no stint data -> assume starter 90'
            if mins == 0 and positions is not None:
                mins = 90
                started = 1

            if pname not in result:
                result[pname] = {"minutes": 0, "starts": 0, "team": team_name}
            result[pname]["minutes"] += mins
            result[pname]["starts"] = max(result[pname]["starts"], started)
            result[pname]["team"] = team_name

    return result
