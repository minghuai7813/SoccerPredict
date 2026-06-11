from __future__ import annotations

import argparse, json, random, re, sys, time
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.encoding import fix_console_encoding, to_ascii_name
fix_console_encoding()

from sqlalchemy import create_engine, distinct
from sqlalchemy.orm import Session
from db.models import Base, Player, PlayerStatsLeague
from utils.entity_resolution import PlayerMatcher

DB_PATH = _PROJECT_ROOT / 'oracle_mvp.db'
DB_URL = f'sqlite:///{DB_PATH}'
ROSTERS_PATH = _PROJECT_ROOT / 'data' / 'wc2026_rosters.json'

SLEEP_MIN, SLEEP_MAX = 3.5, 6.0
RATE_LIMIT_BACKOFF = 300
BATCH_SIZE = 10

EXTRA_LEAGUES = [
    'NED-Eredivisie', 'POR-Primeira Liga', 'TUR-Super Lig',
    'BEL-First Division A', 'BRA-Serie A', 'SCO-Premiership',
    'SUI-Super League', 'AUT-Bundesliga', 'USA-MLS',
]

SEASONS_TO_SCRAPE = ['2024-2025', '2025-2026']
SEASON_END = {'2024-2025': date(2025, 6, 1), '2025-2026': date(2026, 6, 1)}

FBREF_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}


def _sleep():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def _load_unmatched():
    rosters = json.load(open(ROSTERS_PATH, encoding='utf-8'))
    out = []
    for team, players in rosters.items():
        if 'Sky Sports' in team or 'Upgrade' in team:
            continue
        for p in players:
            if p.get('match_status') == 'new':
                out.append({'name': p['name'], 'club': p.get('club', ''),
                            'position': p.get('position', ''),
                            'player_id': p['player_id'], 'team': team})
    return out


def _get_already_scraped(session, season):
    rows = session.query(distinct(PlayerStatsLeague.internal_player_id)).filter(
        PlayerStatsLeague.season == season).all()
    return {r[0] for r in rows}


def _try_int(s):
    try: return int(s) if s else None
    except (ValueError, TypeError): return None


def _try_float(s):
    try: return float(s) if s else None
    except (ValueError, TypeError): return None


def phase1_bulk_leagues():
    print('\n' + '=' * 60)
    print('  Phase 1: Bulk league scrape')
    print('=' * 60)
    try:
        import soccerdata as sd
    except ImportError:
        print('soccerdata not installed, skipping Phase 1')
        return 0
    from extractors.fbref_scraper import (
        _flatten_columns, _get_col, _rate_limit,
        _extract_league_stat_fields, _merge_league_stat_frames,
    )
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)
    unmatched = _load_unmatched()
    unmatched_pids = {p['player_id'] for p in unmatched}
    print(f'  {len(unmatched)} unmatched WC2026 players to find\n')
    with Session(engine) as session:
        name_to_pid = {}
        for pid, name in session.query(Player.internal_player_id, Player.full_name).all():
            name_to_pid[name] = pid
        matcher = PlayerMatcher(name_to_pid)
        total_found = 0
        for league in EXTRA_LEAGUES:
            for season in SEASONS_TO_SCRAPE:
                already_done = _get_already_scraped(session, season)
                still_needed = unmatched_pids - already_done
                if not still_needed:
                    print(f'  [{league} {season}] All covered, skip')
                    continue
                print(f'\n  [{league} {season}] Scraping...')
                try:
                    fbref = sd.FBref(leagues=league, seasons=season)
                    standard = fbref.read_player_season_stats(stat_type='standard')
                    standard = _flatten_columns(standard.reset_index())
                    _rate_limit()
                except Exception as e:
                    print(f'    ERROR: {e}')
                    if '429' in str(e).lower() or 'rate' in str(e).lower():
                        time.sleep(RATE_LIMIT_BACKOFF)
                    continue
                try:
                    shooting = fbref.read_player_season_stats(stat_type='shooting')
                    shooting = _flatten_columns(shooting.reset_index())
                    _rate_limit()
                except Exception:
                    shooting = None
                merged = _merge_league_stat_frames(standard, shooting, None)
                if merged is None or merged.empty or 'player' not in merged.columns:
                    print(f'    No data from {league} {season}')
                    continue
                found_in_league = 0
                for _, row in merged.iterrows():
                    fbref_name = str(row.get('player', ''))
                    if not fbref_name or fbref_name == 'nan':
                        continue
                    result = matcher.match_name(fbref_name, threshold=82)
                    if result.is_new:
                        continue
                    pid = result.internal_player_id
                    if pid not in unmatched_pids or pid in already_done:
                        continue
                    fields = _extract_league_stat_fields(row)
                    stat = PlayerStatsLeague(
                        internal_player_id=pid, season=season, league=league,
                        as_of_date=SEASON_END.get(season, date(2026, 6, 1)),
                        **fields)
                    session.add(stat)
                    already_done.add(pid)
                    found_in_league += 1
                session.commit()
                total_found += found_in_league
                print(f'    Matched {found_in_league} WC players from {league} {season}')
    print(f'\n  Phase 1 done: {total_found} new player-season stats')
    return total_found


def _search_fbref(player_name):
    query = quote_plus(player_name)
    url = f'https://fbref.com/en/search/search.fcgi?search={query}'
    try:
        resp = requests.get(url, headers=FBREF_HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 429:
            print(f'    [429] Rate limited, sleeping {RATE_LIMIT_BACKOFF}s')
            time.sleep(RATE_LIMIT_BACKOFF)
            return None
        if resp.status_code != 200:
            return None
        if '/players/' in resp.url and '/search/' not in resp.url:
            return resp.url
        soup = BeautifulSoup(resp.text, 'html.parser')
        search_div = soup.find('div', {'id': 'players'})
        if not search_div:
            return None
        first_link = search_div.find('a', href=re.compile(r'/en/players/'))
        if first_link:
            href = first_link['href']
            return f'https://fbref.com{href}' if href.startswith('/') else href
    except Exception as e:
        print(f'    Search error for {player_name}: {e}')
    return None


def _scrape_player_page(url):
    try:
        resp = requests.get(url, headers=FBREF_HEADERS, timeout=15)
        if resp.status_code == 429:
            print(f'    [429] Rate limited, sleeping {RATE_LIMIT_BACKOFF}s')
            time.sleep(RATE_LIMIT_BACKOFF)
            return None
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, 'html.parser')
        stats_table = soup.find('table', {'id': 'stats_standard_dom_lg'})
        if not stats_table:
            stats_table = soup.find('table', {'id': re.compile(r'stats_standard')})
        if not stats_table:
            return None
        tbody = stats_table.find('tbody')
        if not tbody:
            return None
        seasons = {}
        for tr in tbody.find_all('tr', class_=lambda c: c != 'thead'):
            th = tr.find('th', {'data-stat': 'year_id'})
            if not th:
                continue
            season_text = th.get_text(strip=True)
            cells = {}
            for td in tr.find_all('td'):
                stat = td.get('data-stat', '')
                val = td.get_text(strip=True)
                cells[stat] = val
            goals = _try_int(cells.get('goals', ''))
            assists = _try_int(cells.get('assists', ''))
            minutes = _try_int(cells.get('minutes', '').replace(',', ''))
            xg = _try_float(cells.get('xg', ''))
            interceptions = _try_int(cells.get('interceptions', ''))
            tackles = _try_int(cells.get('tackles_won', ''))
            if minutes is not None and minutes > 0:
                seasons[season_text] = {
                    'goals': goals, 'assists': assists,
                    'minutes_played': minutes, 'xg': xg,
                    'interceptions': interceptions, 'tackles_won': tackles,
                }
        return seasons
    except Exception as e:
        print(f'    Scrape error: {e}')
        return None


SEASON_ALIASES = {
    '2024-2025': ['2024-2025', '2024'],
    '2025-2026': ['2025-2026', '2025', '2026'],
}
