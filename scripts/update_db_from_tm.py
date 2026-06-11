import io
import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(r'd:\CursorProjects\SoccerProject')
sys.path.insert(0, str(_ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from db.models import Base, Player, PlayerStatsLeague, PlayerStatsNational
from utils.encoding import to_ascii_name
from thefuzz import fuzz

DB_PATH = _ROOT / 'oracle_mvp.db'
engine = create_engine(f'sqlite:///{DB_PATH}')

TEAM_MAP = {
    'algeria_national.json': 'Algeria',
    'australia_national.json': 'Australia',
    'bosnia_national.json': 'Bosnia and Herzegovina',
    'canada_national.json': 'Canada',
    'cape_verde_national.json': 'Cape Verde',
    'curacao_national.json': 'Curacao',
    'czech_republic_national.json': 'Czech Republic',
    'egypt_national.json': 'Egypt',
    'haiti_national.json': 'Haiti',
    'iran_national.json': 'Iran',
    'iraq_national.json': 'Iraq',
    'jordan_national.json': 'Jordan',
    'new_zealand_national.json': 'New Zealand',
    'panama_national.json': 'Panama',
    'paraguay_national.json': 'Paraguay',
    'qatar_national.json': 'Qatar',
    'saudi_arabia_national.json': 'Saudi Arabia',
    'south_africa_national.json': 'South Africa',
    'turkey_national.json': 'Turkey',
}

rosters = json.loads((_ROOT / 'data' / 'wc2026_rosters.json').read_text(encoding='utf-8'))

total_updated = 0
team_results = {}

with Session(engine) as session:
    for json_file, team_name in TEAM_MAP.items():
        json_path = _ROOT / 'data' / 'tm_raw' / json_file
        if not json_path.exists():
            team_results[team_name] = 'FILE_NOT_FOUND'
            continue

        tm_players = json.loads(json_path.read_text(encoding='utf-8'))
        roster_players = rosters.get(team_name, [])
        if not roster_players:
            team_results[team_name] = 'NO_ROSTER'
            continue

        matched_count = 0
        for rp in roster_players:
            pid = rp['player_id']
            rp_name = rp['name']
            rp_ascii = to_ascii_name(rp_name)

            best_score = 0
            best_tm = None
            for tm in tm_players:
                tm_ascii = to_ascii_name(tm['name'])
                score = fuzz.token_set_ratio(rp_ascii, tm_ascii)
                if score > best_score:
                    best_score = score
                    best_tm = tm

            if best_score >= 78 and best_tm is not None:
                caps_val = best_tm['caps']
                goals_val = best_tm['goals']

                existing_nat = session.query(PlayerStatsNational).filter_by(
                    internal_player_id=pid,
                    competition='Career International'
                ).first()

                if existing_nat:
                    existing_nat.caps = caps_val
                    existing_nat.goals = goals_val
                    existing_nat.as_of_date = date(2025, 6, 1)
                else:
                    new_nat = PlayerStatsNational(
                        internal_player_id=pid,
                        competition='Career International',
                        as_of_date=date(2025, 6, 1),
                        caps=caps_val,
                        goals=goals_val,
                    )
                    session.add(new_nat)

                if caps_val > 0:
                    min_goals_est = max(1, int(goals_val / caps_val * 30))
                    league_stat = session.query(PlayerStatsLeague).filter_by(
                        internal_player_id=pid,
                        season='2024-2025'
                    ).first()
                    if league_stat and (league_stat.goals is None or league_stat.goals < min_goals_est):
                        league_stat.goals = min_goals_est

                matched_count += 1

        team_results[team_name] = matched_count
        total_updated += matched_count
        print(f'{team_name}: {matched_count}/{len(roster_players)} matched')

    session.commit()

print(f'\n=== SUMMARY ===')
print(f'Total players updated: {total_updated}')
print(f'Teams processed: {len(team_results)}')
for t, r in sorted(team_results.items()):
    print(f'  {t}: {r}')
