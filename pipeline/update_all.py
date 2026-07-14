"""
Weekly data update for the QB LEAF Explorer. Run from the repo root:

    python pipeline/update_all.py

Steps:
  1. Refresh player metadata (birthdates, draft picks) from nflverse
  2. Rebuild the base game file from fresh play-by-play (2006 -> current)
  3. Re-run the LEAF v3 walk-forward engine (all tuning on 2006-2018)
  4. Export app-format ratings files, pruning superseded exports

The GitHub Action (.github/workflows/update-ratings.yml) runs this weekly and
commits the refreshed leaf_v3_* files, which triggers a Railway redeploy.
"""

import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))


def refresh_player_meta():
    print('[1/4] player metadata...')
    p = nfl.import_players()
    meta = p[['gsis_id', 'display_name', 'position', 'birth_date',
              'draft_year', 'draft_round', 'draft_pick']].dropna(subset=['gsis_id'])
    meta.to_csv(ROOT / 'data' / 'raw' / 'player_meta.csv', index=False)
    print(f'    {len(meta)} players')


def refresh_game_data():
    print('[2/4] base game data from pbp...')
    current = datetime.now().year
    # NFL season spans the year boundary; before September, the newest
    # complete-or-in-progress season is last year's.
    last_season = current if datetime.now().month >= 9 else current - 1
    frames = []
    for season in range(2006, last_season + 1):
        try:
            df = nfl.import_pbp_data([season], downcast=False, cache=False)
        except Exception as e:
            print(f'    [WARN] {season}: {e}')
            continue
        p = df[(df['qb_dropback'] == 1) & df['passer_player_id'].notna()][
            ['game_id', 'game_date', 'season', 'week', 'passer_player_id',
             'passer_player_name', 'posteam', 'defteam', 'qb_epa', 'cpoe',
             'success', 'pass_attempt']].dropna(subset=['qb_epa'])
        g = p.groupby(['game_id', 'game_date', 'season', 'week',
                       'passer_player_id', 'passer_player_name', 'posteam', 'defteam']).agg(
            qb_epa_mean=('qb_epa', 'mean'), qb_epa_sum=('qb_epa', 'sum'),
            qb_epa_count=('qb_epa', 'size'), cpoe_mean=('cpoe', 'mean'),
            success_mean=('success', 'mean'), attempts=('pass_attempt', 'sum'),
        ).reset_index()
        frames.append(g)
        print(f'    {season}: {len(g)} QB-games')
        del df, p
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(ROOT / 'data' / 'raw' / 'qb_games_base.csv', index=False)
    print(f'    total {len(out):,} QB-games')


def run_engine():
    print('[3/4] LEAF v3 engine...')
    import leaf_engine
    leaf_engine.main()
    # engine writes to <repo>/data/production/leaf_v3_ratings.csv + params


def export_app_files():
    print('[4/4] export app files...')
    stamp = date.today().strftime('%Y%m%d')
    prod = ROOT / 'data' / 'production'

    df = pd.read_csv(prod / 'leaf_v3_ratings.csv')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['passer_player_id', 'game_date']).reset_index(drop=True)
    career_plays = df.groupby('passer_player_id')['plays'].transform('sum')
    df = df[career_plays >= 100].copy()
    df['game_number'] = df.groupby('passer_player_id').cumcount() + 1

    games = pd.DataFrame({
        'season': df['season'], 'week': df['week'], 'game_date': df['game_date'],
        'passer_player_id': df['passer_player_id'], 'player_name': df['passer_player_name'],
        'posteam': df['posteam'], 'defteam': df['defteam'], 'game_number': df['game_number'],
        'leaf_rating': df['leaf_v3'],
        'opp_adj_base_epa_kalman': df['k_informed'],
        'opp_adj_base_epa_uncertainty': np.sqrt(df['k_informed_var']),
        'game_epa': df['adj_epa'],
        'plays': df['plays'], 'age': df['age'],
    })

    last = df.groupby('passer_player_id').last()
    agg = df.groupby('passer_player_id').agg(total_games=('game_id', 'nunique'),
                                             total_attempts=('plays', 'sum'))
    current = pd.DataFrame({
        'player_id': last.index, 'player_name': last['passer_player_name'].values,
        'last_season': last['season'].values, 'last_week': last['week'].values,
        'leaf_rating': last['leaf_v3'].values,
        'leaf_uncertainty': np.sqrt(last['k_informed_var'].values),
        'total_games': agg['total_games'].values,
        'total_attempts': agg['total_attempts'].values,
    })

    # prune superseded exports so the repo doesn't accumulate snapshots
    for old in list(prod.glob('leaf_v3_game_by_game_*.csv')) + list(prod.glob('leaf_v3_current_ratings_*.csv')):
        old.unlink()

    games.to_csv(prod / f'leaf_v3_game_by_game_{stamp}.csv', index=False)
    current.to_csv(prod / f'leaf_v3_current_ratings_{stamp}.csv', index=False)
    with open(prod / 'last_update.txt', 'w') as f:
        f.write(f'{datetime.now().isoformat()}\n{len(games)} QB-games, {len(current)} QBs\n')
    print(f'    {len(games):,} game rows, {len(current)} QBs (stamp {stamp})')


def main():
    refresh_player_meta()
    refresh_game_data()
    run_engine()
    export_app_files()
    print('[OK] update complete')


if __name__ == '__main__':
    main()
