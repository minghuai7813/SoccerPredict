import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

POSITIONS = [
    'Goalkeeper', 'Centre-Back', 'Left-Back', 'Right-Back',
    'Defensive Midfield', 'Central Midfield', 'Attacking Midfield',
    'Left Winger', 'Right Winger', 'Right Midfield', 'Left Midfield',
    'Second Striker', 'Centre-Forward',
]

POS_PATTERN = re.compile(r'\s+(' + '|'.join(re.escape(p) for p in POSITIONS) + r')$')


def clean_player_name(raw):
    raw = raw.strip()
    raw = POS_PATTERN.sub('', raw)
    return raw.strip()


def parse_tm_squad_html(html):
    tables = pd.read_html(io.StringIO(html))
    if len(tables) < 2:
        raise ValueError(f'Expected at least 2 tables, got {len(tables)}')
    df = tables[1]
    cols = df.columns.tolist()
    caps_col = None
    goals_col = None
    player_col = None
    for c in cols:
        cl = str(c).lower()
        if 'international' in cl and 'match' in cl:
            caps_col = c
        elif 'goal' in cl:
            goals_col = c
        elif 'player' in cl:
            player_col = c
    if caps_col is None:
        for c in cols:
            if 'nderspiele' in str(c) or 'Internationa' in str(c):
                caps_col = c
    if goals_col is None:
        for c in cols:
            if 'Tore' in str(c) or 'Goal' in str(c):
                goals_col = c
    if player_col is None:
        player_col = cols[1] if len(cols) > 1 else cols[0]
    if caps_col is None or goals_col is None:
        print(f'WARNING: caps_col={caps_col}, goals_col={goals_col}')
        print(f'Available columns: {cols}')
        return []
    results = []
    for _, row in df.iterrows():
        name_raw = str(row[player_col])
        caps_raw = str(row[caps_col])
        goals_raw = str(row[goals_col])
        if name_raw in ('nan', 'None', ''):
            continue
        caps_clean = caps_raw.replace(',', '').strip()
        goals_clean = goals_raw.replace(',', '').strip()
        if not caps_clean.isdigit():
            continue
        name = clean_player_name(name_raw)
        caps = int(caps_clean)
        goals = int(goals_clean) if goals_clean.isdigit() else 0
        results.append({'name': name, 'caps': caps, 'goals': goals})
    return results


if __name__ == '__main__':
    html_file = sys.argv[1]
    out_file = sys.argv[2]
    html = Path(html_file).read_text(encoding='utf-8', errors='replace')
    players = parse_tm_squad_html(html)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(players, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Parsed {len(players)} players -> {out_file}')
