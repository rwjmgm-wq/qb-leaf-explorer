"""
LEAF v3 walk-forward rating engine (repo-local copy for the weekly update
action). Canonical version + full audit trail: DARKO_NFL/scripts/v3_honest/
and DARKO_NFL/docs/SPEC_LEAF_V3.md.

Every quantity at game t is computed from information strictly before t
(the filtered state additionally includes game t itself, used only to predict
t+1 onward). All hyperparameters are tuned on TRAIN_ERA (2006-2018) one-step-
ahead performance; the 2019+ test era never influences a parameter.

Layers:
  L1  walk-forward opponent adjustment (decayed, shrunk defense ratings)
  L2  scalar Kalman on adjusted EPA, obs noise = r_play / attempts
  L3  parallel Kalman states for CPOE + success rate, linear fusion (train fit)
  L4  draft-pick-informed rookie prior + age drift (train fit)

Output: data/production/leaf_v3_ratings.csv (one row per QB-game) and
        data/production/leaf_v3_params.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
TRAIN_MAX_SEASON = 2018

# ------------------------------------------------------------------ data

def load_games():
    df = pd.read_csv(ROOT / 'data' / 'raw' / 'qb_games_base.csv')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['game_date', 'game_id']).reset_index(drop=True)
    df['plays'] = df['qb_epa_count'].clip(lower=1)
    df['epa'] = df['qb_epa_mean']
    df['cpoe'] = df['cpoe_mean'].fillna(0)
    df['success'] = df['success_mean']

    meta = pd.read_csv(ROOT / 'data' / 'raw' / 'player_meta.csv')
    meta['birth_date'] = pd.to_datetime(meta['birth_date'], errors='coerce')
    df = df.merge(meta[['gsis_id', 'birth_date', 'draft_pick']],
                  left_on='passer_player_id', right_on='gsis_id', how='left')
    df['age'] = (df['game_date'] - df['birth_date']).dt.days / 365.25
    df['log_pick'] = np.log(df['draft_pick'].fillna(263))
    return df


# ------------------------------------------------------------------ L1: defense ratings

def walkforward_defense(df, halflife_days, k_shrink):
    """Defense rating before each game: decayed, shrunk mean EPA allowed
    relative to the decayed league mean. Strictly past games only."""
    lam = np.log(2) / halflife_days
    state = {}          # defteam -> [decayed_weight_sum, decayed_weighted_dev_sum, last_date]
    lg_w, lg_s, lg_date = 0.0, 0.0, None

    ratings = np.zeros(len(df))
    dates = df['game_date'].values
    teams = df['defteam'].values
    epas = df['epa'].values
    plays = df['plays'].values

    for i in range(len(df)):
        d = dates[i]
        # decay league aggregates to today
        if lg_date is not None:
            dec = np.exp(-lam * (d - lg_date) / np.timedelta64(1, 'D'))
            lg_w *= dec; lg_s *= dec
        lg_date = d
        lg_mean = lg_s / lg_w if lg_w > 50 else 0.0

        t = teams[i]
        if t in state:
            w, s, last = state[t]
            dec = np.exp(-lam * (d - last) / np.timedelta64(1, 'D'))
            w *= dec; s *= dec
        else:
            w, s = 0.0, 0.0
        # shrunk deviation from league mean (positive = allows more EPA = soft defense)
        ratings[i] = s / (w + k_shrink)

        # update with THIS game (affects only future games)
        state[t] = (w + plays[i], s + plays[i] * (epas[i] - lg_mean), d)
        lg_w += plays[i]; lg_s += plays[i] * epas[i]

    return ratings


def tune_defense(df):
    """Pick (halflife, k) maximizing correlation between rating and the game's
    actual EPA allowed, train era only (defense-side one-step-ahead skill)."""
    train = df['season'] <= TRAIN_MAX_SEASON
    best = None
    print('  tuning defense layer (train era)...')
    for hl in [180, 365, 730]:
        for k in [100, 300, 600]:
            r = walkforward_defense(df, hl, k)
            c = np.corrcoef(r[train], df.loc[train, 'epa'])[0, 1]
            print(f'    halflife={hl:4d}d k={k:3d}: r(def rating, epa allowed) = {c:+.4f}')
            if best is None or c > best[0]:
                best = (c, hl, k, r)
    _, hl, k, ratings = best
    print(f'  -> chosen: halflife={hl}, k={k}')
    return hl, k, ratings


# ------------------------------------------------------------------ L2: scalar Kalman

def kalman_pass(values, plays, q, r_play, p0, prior_mean, drift=None):
    """One-dimensional walk-forward Kalman. Returns (prior_preds, filtered, variances).
    prior_preds[i] = state estimate BEFORE seeing game i (the honest prediction)."""
    n = len(values)
    pri = np.empty(n); post = np.empty(n); var = np.empty(n)
    m, P = prior_mean, p0
    for i in range(n):
        if drift is not None:
            m = m + drift[i]
        P = P + q
        pri[i] = m
        R = r_play / plays[i]
        K = P / (P + R)
        m = m + K * (values[i] - m)
        P = (1 - K) * P
        post[i] = m; var[i] = P
    return pri, post, var


def tune_kalman(df, col, prior_mean):
    """Grid-search q, r_play, p0 by one-step-ahead weighted MSE on train era."""
    train_mask = (df['season'] <= TRAIN_MAX_SEASON).values
    groups = df.groupby('passer_player_id', sort=False)
    best = None
    for q in [1e-5, 5e-5, 2e-4, 8e-4, 3e-3]:
        for r_play in [1.0, 2.0, 3.0]:
            for p0 in [0.005, 0.02, 0.06]:
                se, wsum = 0.0, 0.0
                for _, g in groups:
                    v = g[col].values; pl = g['plays'].values
                    tr = train_mask[g.index.values]
                    if not tr.any():
                        continue
                    pri, _, _ = kalman_pass(v, pl, q, r_play, p0, prior_mean)
                    w = pl[tr]
                    se += (w * (v[tr] - pri[tr]) ** 2).sum()
                    wsum += w.sum()
                mse = se / wsum
                if best is None or mse < best[0]:
                    best = (mse, q, r_play, p0)
    mse, q, r_play, p0 = best
    print(f'  {col}: q={q:g}, r_play={r_play}, p0={p0} (train 1-step wMSE={mse:.4f})')
    return q, r_play, p0


# ------------------------------------------------------------------ L4: priors + age

def fit_rookie_prior(df):
    """Train-era: first-season adjusted EPA vs log(draft pick)."""
    firsts = []
    for pid, g in df.groupby('passer_player_id'):
        g0 = g[g['season'] == g['season'].min()]
        if g0['season'].iloc[0] > TRAIN_MAX_SEASON or g0['plays'].sum() < 100:
            continue
        firsts.append({'log_pick': g0['log_pick'].iloc[0],
                       'epa': np.average(g0['adj_epa'], weights=g0['plays'])})
    f = pd.DataFrame(firsts)
    b, a = np.polyfit(f['log_pick'], f['epa'], 1)
    print(f'  rookie prior: adj_epa = {a:+.4f} + {b:+.4f} * log(pick)  (n={len(f)} train rookies)')
    return a, b


def fit_age_drift(df):
    """Train-era within-QB year-over-year change in weighted adj EPA, by age."""
    rows = []
    for pid, g in df[df['season'] <= TRAIN_MAX_SEASON].groupby('passer_player_id'):
        seas = g.groupby('season').apply(
            lambda s: pd.Series({'epa': np.average(s['adj_epa'], weights=s['plays']),
                                 'plays': s['plays'].sum(), 'age': s['age'].mean()}),
            include_groups=False)
        seas = seas[seas['plays'] >= 150]
        years = sorted(seas.index)
        for y0, y1 in zip(years, years[1:]):
            if y1 == y0 + 1:
                rows.append({'age': seas.loc[y0, 'age'], 'delta': seas.loc[y1, 'epa'] - seas.loc[y0, 'epa']})
    d = pd.DataFrame(rows)
    buckets = {'u25': d[d.age < 25], '25_32': d[(d.age >= 25) & (d.age < 32)], 'o32': d[d.age >= 32]}
    drift = {k: float(v['delta'].mean()) if len(v) >= 10 else 0.0 for k, v in buckets.items()}
    print(f'  age drift (EPA/season): {drift} (n={ {k: len(v) for k, v in buckets.items()} })')
    return drift


def age_drift_per_game(age, drift):
    if np.isnan(age):
        return 0.0
    if age < 25:
        return drift['u25'] / 17
    if age < 32:
        return drift['25_32'] / 17
    return drift['o32'] / 17


# ------------------------------------------------------------------ baselines

def add_baselines(df, ewma_halflife):
    out = {}
    for pid, g in df.groupby('passer_player_id', sort=False):
        idx = g.index.values
        epa = g['epa'].values; pl = g['plays'].values
        n = len(g)
        expand = np.full(n, np.nan); ewma = np.full(n, np.nan); last12 = np.full(n, np.nan)
        csum = cw = 0.0
        alpha = 1 - 0.5 ** (1 / ewma_halflife)
        e = None
        for i in range(n):
            if cw > 0:
                expand[i] = csum / cw
            if e is not None:
                ewma[i] = e
            if i >= 1:
                lo = max(0, i - 12)
                last12[i] = np.average(epa[lo:i], weights=pl[lo:i])
            csum += epa[i] * pl[i]; cw += pl[i]
            e = epa[i] if e is None else (1 - alpha) * e + alpha * epa[i]
        out[pid] = (idx, expand, ewma, last12)
    for name, j in [('b1_expanding', 1), ('b3_ewma', 2), ('b4_last12', 3)]:
        col = np.full(len(df), np.nan)
        for pid, tup in out.items():
            col[tup[0]] = tup[j]
        df[name] = col
    # B2: previous season mean
    seas = df.groupby(['passer_player_id', 'season']).apply(
        lambda s: np.average(s['epa'], weights=s['plays']), include_groups=False).rename('season_epa').reset_index()
    seas['season'] += 1
    df = df.merge(seas, on=['passer_player_id', 'season'], how='left').rename(columns={'season_epa': 'b2_prev_season'})
    return df


def tune_ewma(df):
    train = df['season'] <= TRAIN_MAX_SEASON
    best = None
    for hl in [3, 5, 8, 12, 20]:
        d2 = add_baselines(df.copy(), hl)
        m = train & d2['b3_ewma'].notna()
        mse = np.average((d2.loc[m, 'epa'] - d2.loc[m, 'b3_ewma']) ** 2, weights=d2.loc[m, 'plays'])
        if best is None or mse < best[0]:
            best = (mse, hl)
    print(f'  EWMA halflife tuned: {best[1]} games')
    return best[1]


# ------------------------------------------------------------------ main

def main():
    print('=' * 70)
    print('LEAF v3 WALK-FORWARD ENGINE (per docs/SPEC_LEAF_V3.md)')
    print('=' * 70)
    df = load_games()
    print(f'{len(df)} QB-games, {df.passer_player_id.nunique()} passers, '
          f'{df.season.min()}-{df.season.max()} | train era <= {TRAIN_MAX_SEASON}')

    print('\n[L1] Walk-forward defense ratings')
    hl_def, k_def, def_ratings = tune_defense(df)
    df['def_rating'] = def_ratings
    df['adj_epa'] = df['epa'] - df['def_rating']

    print('\n[baselines] tuning EWMA halflife on train era')
    hl_ewma = tune_ewma(df)
    df = add_baselines(df, hl_ewma)

    print('\n[L2] Kalman tuning (train era, one-step-ahead)')
    kp = {}
    kp['adj_epa'] = tune_kalman(df, 'adj_epa', prior_mean=-0.05)
    kp['epa'] = tune_kalman(df, 'epa', prior_mean=-0.05)
    kp['cpoe'] = tune_kalman(df, 'cpoe', prior_mean=-1.0)
    kp['success'] = tune_kalman(df, 'success', prior_mean=0.42)

    print('\n[L2/L3] Running filters (full timeline, walk-forward)')
    for col, prior in [('adj_epa', -0.05), ('epa', -0.05), ('cpoe', -1.0), ('success', 0.42)]:
        q, r_play, p0 = kp[col]
        pri = np.empty(len(df)); post = np.empty(len(df)); var = np.empty(len(df))
        for pid, g in df.groupby('passer_player_id', sort=False):
            i = g.index.values
            a, b, c = kalman_pass(g[col].values, g['plays'].values, q, r_play, p0, prior)
            pri[i], post[i], var[i] = a, b, c
        df[f'k_{col}_pre'] = pri; df[f'k_{col}'] = post; df[f'k_{col}_var'] = var

    print('\n[L4] Rookie prior + age drift (train era fits)')
    a_pr, b_pr = fit_rookie_prior(df)
    drift = fit_age_drift(df)
    q, r_play, p0 = kp['adj_epa']
    pri = np.empty(len(df)); post = np.empty(len(df)); var = np.empty(len(df))
    for pid, g in df.groupby('passer_player_id', sort=False):
        i = g.index.values
        prior_mean = a_pr + b_pr * g['log_pick'].iloc[0]
        dr = np.array([age_drift_per_game(x, drift) for x in g['age'].values])
        a, b, c = kalman_pass(g['adj_epa'].values, g['plays'].values, q, r_play, p0, prior_mean, drift=dr)
        pri[i], post[i], var[i] = a, b, c
    df['k_informed_pre'] = pri; df['k_informed'] = post; df['k_informed_var'] = var

    print('\n[L3] Fusion weights (train era, predict next-16-games EPA)')
    rows = []
    for pid, g in df.groupby('passer_player_id', sort=False):
        v = g.reset_index()
        for i in range(0, len(v) - 16, 8):
            if v.loc[i, 'season'] > TRAIN_MAX_SEASON:
                continue
            fut = v.iloc[i + 1:i + 17]
            rows.append({'k_epa': v.loc[i, 'k_adj_epa'], 'k_cpoe': v.loc[i, 'k_cpoe'],
                         'k_success': v.loc[i, 'k_success'], 'k_informed': v.loc[i, 'k_informed'],
                         'y': np.average(fut['epa'], weights=fut['plays'])})
    tr = pd.DataFrame(rows)
    X = np.column_stack([np.ones(len(tr)), tr['k_informed'], tr['k_cpoe'], tr['k_success']])
    beta = np.linalg.lstsq(X, tr['y'].values, rcond=None)[0]
    print(f'  fusion: y = {beta[0]:+.4f} + {beta[1]:+.3f}*k_epa(informed) + '
          f'{beta[2]:+.4f}*k_cpoe + {beta[3]:+.3f}*k_success (n={len(tr)})')
    df['leaf_v3'] = beta[0] + beta[1] * df['k_informed'] + beta[2] * df['k_cpoe'] + beta[3] * df['k_success']
    df['leaf_v3_pre'] = beta[0] + beta[1] * df['k_informed_pre'] + beta[2] * df['k_cpoe_pre'] + beta[3] * df['k_success_pre']

    out_cols = ['game_id', 'game_date', 'season', 'week', 'passer_player_id', 'passer_player_name',
                'posteam', 'defteam', 'plays', 'epa', 'cpoe', 'success', 'age', 'log_pick',
                'def_rating', 'adj_epa', 'b1_expanding', 'b2_prev_season', 'b3_ewma', 'b4_last12',
                'k_epa', 'k_adj_epa', 'k_adj_epa_var', 'k_informed', 'k_informed_var', 'leaf_v3', 'leaf_v3_pre']
    out = ROOT / 'data' / 'production' / 'leaf_v3_ratings.csv'
    df[out_cols].to_csv(out, index=False)

    params = {'defense': {'halflife_days': hl_def, 'k_shrink': k_def},
              'ewma_halflife': hl_ewma,
              'kalman': {k: {'q': v[0], 'r_play': v[1], 'p0': v[2]} for k, v in kp.items()},
              'rookie_prior': {'intercept': a_pr, 'log_pick_slope': b_pr},
              'age_drift': drift,
              'fusion_beta': list(map(float, beta)),
              'train_max_season': TRAIN_MAX_SEASON}
    with open(ROOT / 'data' / 'production' / 'leaf_v3_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    print(f'\n[OK] ratings -> {out}')


if __name__ == '__main__':
    main()
