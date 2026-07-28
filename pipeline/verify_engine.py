"""
Fast pandas-compatibility smoke test for the LEAF v3 engine.

Runs in ~1 second on synthetic data, BEFORE the weekly job spends ~20 minutes
downloading play-by-play. Exercises the engine's pandas-version-sensitive code
paths (groupby.apply shapes) plus the numeric core, so an API incompatibility
fails the job immediately with a clear message instead of deep inside the
rebuild.

Why this exists: the 2026-07-21 and 07-28 runs both died at
`add_baselines ... unexpected keyword argument 'include_groups'` — CI resolves
pandas 1.5.3 (pinned by nfl_data_py) while local dev runs pandas 2.x, where
that kwarg is required to silence a warning. Same code, different pandas,
20 minutes of download wasted each time.
"""

import sys

import numpy as np
import pandas as pd

import leaf_engine


def synthetic_games(n_qb=4, seasons=(2015, 2016, 2017, 2018, 2019), per_season=8):
    rng = np.random.default_rng(0)
    rows = []
    for q in range(n_qb):
        for s in seasons:
            for w in range(per_season):
                rows.append({
                    'passer_player_id': f'QB{q}', 'passer_player_name': f'Q.B{q}',
                    'game_id': f'{s}_{w}_{q}', 'game_date': f'{s}-09-{w + 1:02d}',
                    'season': s, 'week': w + 1, 'posteam': 'AAA', 'defteam': 'BBB',
                    'epa': float(rng.normal(0, 0.2)), 'adj_epa': float(rng.normal(0, 0.2)),
                    'plays': int(rng.integers(20, 45)),
                    'cpoe': float(rng.normal(0, 3)), 'success': float(rng.uniform(0.3, 0.6)),
                    'age': 24 + s - 2015 + q, 'log_pick': float(np.log(10 + q)),
                })
    df = pd.DataFrame(rows)
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df.sort_values(['passer_player_id', 'game_date']).reset_index(drop=True)


def check(label, fn):
    try:
        fn()
    except Exception as e:
        print(f'  [FAIL] {label}: {type(e).__name__}: {e}')
        return False
    print(f'  [ok]   {label}')
    return True


def main():
    print(f'pandas {pd.__version__}, numpy {np.__version__}')
    df = synthetic_games()
    ok = True

    def _baselines():
        out = leaf_engine.add_baselines(df.copy(), 5)
        missing = [c for c in ['b1_expanding', 'b2_prev_season', 'b3_ewma', 'b4_last12']
                   if c not in out.columns]
        assert not missing, f'missing baseline columns: {missing}'
        assert len(out) == len(df), 'add_baselines changed row count'
        assert out['b2_prev_season'].notna().any(), 'b2_prev_season all null'

    def _age_drift():
        d = leaf_engine.fit_age_drift(df)
        assert isinstance(d, dict) and {'u25', '25_32', 'o32'} <= set(d), \
            f'unexpected age-drift shape: {d}'

    def _kalman():
        v = np.array([0.1, -0.05, 0.2, 0.0])
        p = np.array([30.0, 25.0, 40.0, 35.0])
        pri, post, var = leaf_engine.kalman_pass(v, p, 5e-5, 1.0, 0.005, 0.0)
        assert len(pri) == len(post) == len(var) == len(v), 'kalman length mismatch'
        assert np.all(np.isfinite(post)), 'kalman produced non-finite state'
        assert np.all(var > 0), 'kalman variance non-positive'

    def _defense():
        ratings = leaf_engine.walkforward_defense(df.copy(), 365, 600)
        assert len(ratings) == len(df), 'walkforward_defense length mismatch'
        assert np.all(np.isfinite(ratings)), 'walkforward_defense produced non-finite ratings'

    for label, fn in [('add_baselines (groupby.apply)', _baselines),
                      ('fit_age_drift (groupby.apply)', _age_drift),
                      ('kalman_pass (numeric core)', _kalman),
                      ('walkforward_defense', _defense)]:
        ok &= check(label, fn)

    if not ok:
        print('\n[FAIL] engine is not compatible with this pandas/numpy build — '
              'aborting before the play-by-play download.')
        sys.exit(1)
    print('[OK] engine smoke test passed')


if __name__ == '__main__':
    main()
