"""
LEAF Rating Visualization App

Interactive web application for visualizing QB LEAF ratings, trajectories, and predictions.

Features:
- Player selection and search
- Current LEAF rating with uncertainty bands
- Historical game-by-game trajectory
- Future predictions (1, 2, 3 years forward)
- Player comparison mode

Usage:
    python visualize_leaf_ratings.py

    Then open browser to: http://localhost:8050
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
DATA_DIR = Path("data/production")

def load_ratings_data():
    """Load LEAF ratings from production files."""
    logger.info("Loading LEAF ratings data...")

    # Load game-by-game ratings (LEAF v3: walk-forward, leakage-free -
    # see DARKO_NFL/docs/LEAF_V3_RESULTS.md)
    game_files = sorted(DATA_DIR.glob("leaf_v3_game_by_game_*.csv"))
    if not game_files:
        raise FileNotFoundError("No game-by-game ratings file found")

    game_data = pd.read_csv(game_files[-1])  # Most recent
    logger.info(f"  Loaded {len(game_data):,} game records")

    # Load current ratings
    current_files = sorted(DATA_DIR.glob("leaf_v3_current_ratings_*.csv"))
    if not current_files:
        raise FileNotFoundError("No current ratings file found")

    current_data = pd.read_csv(current_files[-1])  # Most recent
    logger.info(f"  Loaded {len(current_data):,} player ratings")

    # Load WR composite ratings (if available)
    try:
        wr_files = sorted(Path("data/processed").glob("wr_composite_ratings_*.csv"))
        if wr_files:
            wr_data = pd.read_csv(wr_files[-1])
            logger.info(f"  Loaded {len(wr_data)} WR composite ratings")
        else:
            wr_data = None
            logger.warning("  No WR composite ratings found")
    except Exception as e:
        logger.warning(f"  Could not load WR data: {e}")
        wr_data = None

    # Load QB composite ratings (if available)
    try:
        qb_files = sorted(Path("data/processed").glob("qb_composite_ratings_*.csv"))
        if qb_files:
            qb_data = pd.read_csv(qb_files[-1])
            logger.info(f"  Loaded {len(qb_data)} QB composite ratings")
        else:
            qb_data = None
            logger.warning("  No QB composite ratings found")
    except Exception as e:
        logger.warning(f"  Could not load QB data: {e}")
        qb_data = None

    return game_data, current_data, wr_data, qb_data

def get_qb_composite_data(qb_composite_data, player_id, season):
    """
    Get QB composite rating data for a specific player and season.

    Args:
        qb_composite_data: QB composite ratings DataFrame
        player_id: Player ID
        season: Season year

    Returns:
        Dictionary with QB composite metrics or None
    """
    if qb_composite_data is None:
        return None

    # Filter to specific player and season
    qb_season = qb_composite_data[
        (qb_composite_data['passer_player_id'] == player_id) &
        (qb_composite_data['season'] == season)
    ]

    if len(qb_season) == 0:
        return None

    qb = qb_season.iloc[0]

    return {
        'composite_rating': qb['qb_composite_rating'],
        'percentile': qb['qb_composite_percentile'],
        'yards_per_game': qb['yards_per_game'],
        'completion_pct': qb['completion_pct'],
        'yards_per_attempt': qb['yards_per_attempt'],
        'attempts_per_game': qb['attempts_per_game'],
        'sack_rate': qb['sack_rate']
    }

def calculate_team_wr_quality(wr_data, team, season):
    """
    Calculate aggregate WR quality for a team.

    Args:
        wr_data: WR composite ratings DataFrame
        team: Team abbreviation
        season: Season year

    Returns:
        Dictionary with team WR quality metrics
    """
    if wr_data is None or 'team' not in wr_data.columns:
        return None

    # Filter to team's WRs in that season
    team_wrs = wr_data[
        (wr_data['team'] == team) &
        (wr_data['season'] == season)
    ].copy()

    if len(team_wrs) == 0:
        return None

    # Calculate aggregate metrics
    return {
        'n_wrs': len(team_wrs),
        'avg_composite': team_wrs['wr_composite_rating'].mean(),
        'total_targets': team_wrs['targets'].sum(),
        'total_yac_epa': team_wrs['total_yac_epa'].sum(),
        'top_wr': team_wrs.nlargest(1, 'wr_composite_rating').iloc[0]['receiver_player_name']
            if len(team_wrs) > 0 else None,
        'top_wr_rating': team_wrs.nlargest(1, 'wr_composite_rating').iloc[0]['wr_composite_rating']
            if len(team_wrs) > 0 else None
    }

def filter_to_qbs(current_ratings, min_attempts=50):
    """Filter to actual QBs (exclude trick plays)."""
    qbs = current_ratings[current_ratings['total_attempts'] >= min_attempts].copy()
    qbs = qbs.sort_values('leaf_rating', ascending=False)
    logger.info(f"  Filtered to {len(qbs)} QBs with {min_attempts}+ attempts")
    return qbs

def determine_player_status(game_data_df, player_id, current_season=2025):
    """
    Determine if a player is active or retired.

    Args:
        game_data_df: Full game-by-game DataFrame
        player_id: Player ID to check
        current_season: Current season (default 2025)

    Returns:
        'active' if player played recently, 'retired' otherwise
    """
    player_games = game_data_df[game_data_df['passer_player_id'] == player_id]
    if len(player_games) == 0:
        return 'retired'

    last_season = player_games['season'].max()

    # Consider retired if haven't played since 2023 or earlier
    # (gives 2-year buffer for current season 2025)
    if last_season < current_season - 1:
        return 'retired'
    return 'active'

def split_active_retired_qbs(qb_data_df, game_data_df):
    """
    Split QBs into active and retired lists.

    Args:
        qb_data_df: Current ratings DataFrame
        game_data_df: Game-by-game DataFrame

    Returns:
        Tuple of (active_qbs, retired_qbs) DataFrames
    """
    qb_data_df = qb_data_df.copy()
    qb_data_df['status'] = qb_data_df['player_id'].apply(
        lambda pid: determine_player_status(game_data_df, pid)
    )

    active_qbs = qb_data_df[qb_data_df['status'] == 'active'].copy()
    retired_qbs = qb_data_df[qb_data_df['status'] == 'retired'].copy()

    logger.info(f"  Active QBs: {len(active_qbs)}")
    logger.info(f"  Retired QBs: {len(retired_qbs)}")

    return active_qbs, retired_qbs

import json as _json

def _load_v3_params():
    try:
        with open(DATA_DIR / 'leaf_v3_params.json') as f:
            return _json.load(f)
    except Exception:
        return {}

V3_PARAMS = _load_v3_params()


def calculate_predictions(player_games, years_forward=[1, 2, 3]):
    """
    Calibrated multi-year projections from the LEAF v3 state-space model.

    Point projection: filtered state + train-era age drift (u25 improve,
    25-32 flat-ish, 33+ decline), advancing the QB's age each year.
    Uncertainty: state variance + skill-change variance x years + one-season
    sampling noise. These components were calibrated on 2006-2018 and
    validated on frozen 2019-2025 data (80% nominal -> 77% actual coverage).
    Replaces the previous trend-extrapolation + hand-tuned regression logic,
    whose career-stage rates were fit on data affected by the constant
    leaf_rating bug (docs/LEAF_DATA_BUG_INVESTIGATION.md).
    """
    if len(player_games) == 0:
        return pd.DataFrame()

    current_rating = player_games['opp_adj_base_epa_kalman'].iloc[-1]
    state_var = player_games['opp_adj_base_epa_uncertainty'].iloc[-1] ** 2
    age = player_games['age'].iloc[-1] if 'age' in player_games.columns else np.nan

    drift = V3_PARAMS.get('age_drift', {'u25': 0.009, '25_32': -0.007, 'o32': -0.019})
    pint = V3_PARAMS.get('predictive_interval', {})
    skill_change_var = pint.get('skill_change_var', 0.0009)
    play_noise_var = pint.get('play_noise_var', 4.24)
    season_plays = 500.0

    def yearly_drift(a):
        if np.isnan(a):
            return 0.0
        if a < 25:
            return drift['u25']
        if a < 32:
            return drift['25_32']
        return drift['o32']

    predictions = []
    last_season = int(player_games['season'].iloc[-1])
    for years in years_forward:
        # accumulate drift one year at a time so the age bracket can change
        m, a = current_rating, age
        for _ in range(int(years)):
            m = m + yearly_drift(a)
            a = a + 1 if not np.isnan(a) else a

        # observed-season variance at this horizon (skill walk + sampling noise)
        var = state_var + skill_change_var * years + play_noise_var / season_plays
        sd = np.sqrt(var)

        predictions.append({
            'years_forward': years,
            'predicted_season': last_season + years,
            'predicted_rating': m,
            'predicted_uncertainty': sd,
            'lower_bound': m - 1.28 * sd,   # 80% interval (validated coverage)
            'upper_bound': m + 1.28 * sd,
        })

    return pd.DataFrame(predictions)

def create_player_trajectory_figure(player_games, predictions, player_name, is_retired=False):
    """
    Create interactive figure showing historical trajectory and predictions.

    Args:
        player_games: Game-by-game records
        predictions: Future predictions DataFrame
        player_name: Player name for title
        is_retired: Whether the player is retired (affects labeling)

    Returns:
        Plotly figure
    """
    subtitle = f"{player_name} - LEAF Rating Trajectory"
    if is_retired:
        subtitle += " (Retired)"

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[subtitle]
    )

    # Historical trajectory - individual games
    fig.add_trace(
        go.Scatter(
            x=player_games.index,
            y=player_games['opp_adj_base_epa_kalman'],
            mode='markers',
            name='Game Performance',
            marker=dict(size=8, color='#1f77b4', opacity=0.6),
            hovertemplate=(
                '<b>Game %{x}</b><br>' +
                'Season: %{customdata[0]}<br>' +
                'Week: %{customdata[1]}<br>' +
                'Rating: %{y:.3f}<br>' +
                'Opponent: %{customdata[2]}<br>' +
                '<extra></extra>'
            ),
            customdata=player_games[['season', 'week', 'defteam']].values
        )
    )

    # Smoothed trendline (rolling 8-game average)
    window = min(8, len(player_games))
    smoothed = None
    trend_end_value = player_games['opp_adj_base_epa_kalman'].iloc[-1]  # Default to last game

    if len(player_games) >= 3:
        smoothed = player_games['opp_adj_base_epa_kalman'].rolling(window=window, min_periods=1, center=False).mean()
        rolling_std = player_games['opp_adj_base_epa_kalman'].rolling(window=window, min_periods=1, center=False).std()

        # Get the smoothed value at the end for prediction connection
        # This now matches the trend line since we're not using center=True
        trend_end_value = smoothed.iloc[-1]

        # Upper confidence band
        fig.add_trace(
            go.Scatter(
                x=player_games.index,
                y=smoothed + 1.5 * rolling_std,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )

        # Lower confidence band
        fig.add_trace(
            go.Scatter(
                x=player_games.index,
                y=smoothed - 1.5 * rolling_std,
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(31, 119, 180, 0.15)',
                fill='tonexty',
                name='Trend Range',
                hoverinfo='skip'
            )
        )

        # Smoothed trendline
        fig.add_trace(
            go.Scatter(
                x=player_games.index,
                y=smoothed,
                mode='lines',
                name='Trend (8-game avg)',
                line=dict(color='#1f77b4', width=3),
                hovertemplate=(
                    '<b>Game %{x}</b><br>' +
                    'Smoothed Rating: %{y:.3f}<br>' +
                    '<extra></extra>'
                )
            )
        )

    # Future predictions
    if len(predictions) > 0:
        last_game_idx = len(player_games) - 1

        # Create prediction x-values (extend beyond historical data)
        games_per_year = 16
        pred_x = [last_game_idx] + [last_game_idx + (years * games_per_year)
                                     for years in predictions['years_forward']]
        pred_y = [trend_end_value] + predictions['predicted_rating'].tolist()

        # Prediction line
        fig.add_trace(
            go.Scatter(
                x=pred_x,
                y=pred_y,
                mode='lines+markers',
                name='Predicted Rating',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate=(
                    '<b>Prediction</b><br>' +
                    'Season: %{customdata[0]}<br>' +
                    'Rating: %{y:.3f}<br>' +
                    'Uncertainty: ±%{customdata[1]:.3f}<br>' +
                    '<extra></extra>'
                ),
                customdata=np.column_stack([
                    predictions['predicted_season'],
                    predictions['predicted_uncertainty']
                ])
            )
        )

        # Prediction uncertainty bands
        fig.add_trace(
            go.Scatter(
                x=pred_x,
                y=[trend_end_value + 1.96 * player_games['opp_adj_base_epa_uncertainty'].iloc[-1]] +
                  predictions['upper_bound'].tolist(),
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pred_x,
                y=[trend_end_value - 1.96 * player_games['opp_adj_base_epa_uncertainty'].iloc[-1]] +
                  predictions['lower_bound'].tolist(),
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 127, 14, 0.2)',
                fill='tonexty',
                name='Prediction Confidence',
                hoverinfo='skip'
            )
        )

    # Add reference line at 0 (average QB)
    fig.add_hline(y=0, line_dash="dot", line_color="gray",
                  annotation_text="Average QB", annotation_position="right")

    # Add elite/poor reference lines
    fig.add_hline(y=0.15, line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="Elite", annotation_position="right")
    fig.add_hline(y=-0.10, line_dash="dot", line_color="red", opacity=0.5,
                  annotation_text="Poor", annotation_position="right")

    # For retired players, add career statistics annotation
    if is_retired and len(player_games) > 0:
        career_avg = player_games['opp_adj_base_epa_kalman'].mean()
        career_best = player_games['opp_adj_base_epa_kalman'].max()
        career_seasons = player_games['season'].nunique()
        first_season = int(player_games['season'].min())
        last_season = int(player_games['season'].max())

        annotation_text = (
            f"<b>Career Summary ({first_season}-{last_season})</b><br>"
            f"Career Average: {career_avg:+.3f}<br>"
            f"Career Best: {career_best:+.3f}<br>"
            f"Seasons: {career_seasons}"
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=annotation_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10)
        )

    # Layout
    fig.update_layout(
        height=600,
        hovermode='x unified',
        xaxis_title="Game Number (Career)",
        yaxis_title="LEAF Rating (EPA per Play)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        # Mobile responsive settings
        margin=dict(l=60, r=20, t=80, b=60),
        # Better for embedding in blog posts
        autosize=True
    )

    return fig

# Initialize Dash app with custom styling
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Lora:wght@400;500&display=swap'
    ],
    title="QB LEAF Rating Explorer | Duncan Drafts"
)

# Custom CSS to match duncan-drafts.ghost.io theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --accent-color: #e52673;
                --accent-hover: #d11c63;
                --text-primary: #15171A;
                --text-secondary: #738a94;
                --bg-white: #ffffff;
                --border-color: #e5eff5;
                --card-shadow: 0 2px 4px rgba(0,0,0,0.08);
                --card-shadow-hover: 0 4px 12px rgba(0,0,0,0.12);
            }

            body {
                font-family: 'Lora', Georgia, serif;
                color: var(--text-primary);
                background-color: #f7f8f9;
            }

            h1, h2, h3, h4, h5, h6, .fw-bold {
                font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                font-weight: 700;
            }

            .card {
                border: 1px solid var(--border-color);
                border-radius: 8px;
                box-shadow: var(--card-shadow);
                transition: all 0.2s ease;
            }

            .card:hover {
                box-shadow: var(--card-shadow-hover);
            }

            .card-header {
                background-color: var(--bg-white);
                border-bottom: 2px solid var(--accent-color);
                border-radius: 8px 8px 0 0 !important;
            }

            /* Accent color for interactive elements */
            .form-check-input:checked {
                background-color: var(--accent-color);
                border-color: var(--accent-color);
            }

            /* Dropdown styling */
            .Select-control {
                border-radius: 8px;
                border-color: var(--border-color);
            }

            /* Info section styling */
            .info-section {
                background: linear-gradient(135deg, #667eea 0%, var(--accent-color) 100%);
                color: white;
                border-radius: 8px;
                padding: 2rem;
                margin-bottom: 2rem;
            }

            .info-section h3 {
                color: white;
                margin-bottom: 1rem;
            }

            /* Stat badges */
            .stat-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 12px;
                font-size: 0.85rem;
                font-weight: 500;
                background-color: rgba(229, 38, 115, 0.1);
                color: var(--accent-color);
            }

            /* Tooltip styling */
            .tooltip-icon {
                cursor: help;
                color: var(--accent-color);
                font-size: 0.9rem;
                margin-left: 0.25rem;
            }

            /* Mobile responsiveness */
            @media (max-width: 768px) {
                h1 {
                    font-size: 1.75rem !important;
                }

                .info-section {
                    padding: 1.5rem;
                }

                .info-section h3 {
                    font-size: 1.25rem;
                }

                .card-body {
                    padding: 1rem;
                }

                /* Make stat cards stack better on mobile */
                .row > .col-4 {
                    margin-bottom: 1rem;
                }
            }

            /* Learn more link */
            #learn-more-link {
                transition: opacity 0.2s ease;
            }

            #learn-more-link:hover {
                opacity: 0.8;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load data
game_data, current_data, wr_data, qb_composite_data = load_ratings_data()
qb_data = filter_to_qbs(current_data, min_attempts=50)

# Split into active and retired QBs
active_qbs, retired_qbs = split_active_retired_qbs(qb_data, game_data)

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("QB LEAF Rating Explorer", className="text-center mb-2",
                   style={'fontWeight': '700', 'fontSize': '2.5rem'}),
            html.P(
                "Explore quarterback performance with data-driven ratings from 2006-2025",
                className="text-center text-muted mb-4",
                style={'fontSize': '1.1rem'}
            )
        ])
    ]),

    # Methodology info section
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("What is LEAF Rating?", style={'marginBottom': '1rem'}),
                html.P([
                    "LEAF v3 is a walk-forward state-space QB rating: opponent-adjusted EPA filtered ",
                    "game by game (Kalman), with draft-pick priors, age curves, and a fusion of EPA, ",
                    "CPOE, and success rate. Every rating uses only games played before it — no hindsight. ",
                    html.Span("Honest out-of-sample accuracy: r = 0.47 for next-season EPA "
                              "(frozen 2019–2025 test era) — at the measured information ceiling of public "
                              "play-by-play data. Projection bands are calibrated 80% intervals "
                              "(77% actual coverage on held-out seasons).",
                              style={'fontWeight': '600'}),
                ], style={'marginBottom': '0', 'fontSize': '1.05rem'})
            ], className="info-section")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Player Status:", className="fw-bold"),
            dcc.RadioItems(
                id='player-status-filter',
                options=[
                    {'label': ' Active Players', 'value': 'active'},
                    {'label': ' Retired Players', 'value': 'retired'}
                ],
                value='active',
                inline=True,
                className="mb-3"
            )
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select QB:", className="fw-bold"),
            dcc.Dropdown(
                id='player-dropdown',
                options=[
                    {'label': f"{row['player_name']} ({row['leaf_rating']:.3f})",
                     'value': row['player_id']}
                    for _, row in active_qbs.iterrows()
                ],
                value=active_qbs.iloc[0]['player_id'] if len(active_qbs) > 0 else None,
                clearable=False,
                className="mb-3"
            )
        ], width=12, lg=6),

        dbc.Col([
            html.Label("Prediction Years:", className="fw-bold"),
            dcc.Checklist(
                id='prediction-years',
                options=[
                    {'label': ' 1 Year', 'value': 1},
                    {'label': ' 2 Years', 'value': 2},
                    {'label': ' 3 Years', 'value': 3}
                ],
                value=[1],
                inline=True,
                className="mb-3"
            )
        ], width=12, lg=6)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Current Rating", className="mb-0")),
                dbc.CardBody([
                    html.H2(id='current-rating', className="text-center"),
                    html.P(id='rating-interpretation', className="text-center text-muted")
                ])
            ], className="mb-3")
        ], width=12, md=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Career Stats", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='career-stats')
                ])
            ], className="mb-3")
        ], width=12, md=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Next Season Prediction", className="mb-0")),
                dbc.CardBody([
                    html.Div(id='next-season-pred')
                ])
            ], className="mb-3")
        ], width=12, md=4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                dcc.Graph(id='trajectory-plot', config={'displayModeBar': True})
            )
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Hr(style={'marginTop': '3rem', 'marginBottom': '2rem'}),
            html.Div([
                html.P([
                    html.Strong("Data Source: "),
                    "nflfastR play-by-play data (2006-2025 seasons) • ",
                    html.Strong("Model: "),
                    "LEAF v2.0 Multi-Feature Rating System • ",
                    html.Strong("Validation: "),
                    "r=0.41 correlation to future performance"
                ], className="text-center text-muted", style={'marginBottom': '0.5rem'}),
                html.P([
                    "Created by ",
                    html.A("Duncan Drafts", href="https://duncan-drafts.ghost.io",
                          target="_blank", style={'color': '#e52673', 'textDecoration': 'none', 'fontWeight': '600'}),
                    " • Data-driven NFL analysis and prospect evaluation"
                ], className="text-center text-muted small")
            ], style={'paddingBottom': '2rem'})
        ])
    ])
], fluid=True, className="p-4")

@app.callback(
    [Output('player-dropdown', 'options'),
     Output('player-dropdown', 'value')],
    [Input('player-status-filter', 'value')]
)
def update_dropdown_options(player_status):
    """Update dropdown options based on active/retired filter."""
    if player_status == 'active':
        qb_list = active_qbs
        label_format = lambda row: f"{row['player_name']} ({row['leaf_rating']:.3f})"
    else:  # retired
        qb_list = retired_qbs
        # Show career LEAF for retired players
        label_format = lambda row: f"{row['player_name']} (Career: {row['leaf_rating']:.3f})"

    options = [
        {'label': label_format(row), 'value': row['player_id']}
        for _, row in qb_list.iterrows()
    ]

    # Set default value to first player in list
    default_value = qb_list.iloc[0]['player_id'] if len(qb_list) > 0 else None

    return options, default_value

@app.callback(
    [Output('current-rating', 'children'),
     Output('current-rating', 'style'),
     Output('rating-interpretation', 'children'),
     Output('career-stats', 'children'),
     Output('next-season-pred', 'children'),
     Output('trajectory-plot', 'figure')],
    [Input('player-dropdown', 'value'),
     Input('prediction-years', 'value'),
     Input('player-status-filter', 'value')]
)
def update_visualizations(player_id, prediction_years, player_status):
    """Update all visualizations when player or settings change."""

    # Get player data
    player_info = qb_data[qb_data['player_id'] == player_id].iloc[0]
    player_games = game_data[game_data['passer_player_id'] == player_id].copy()
    player_games = player_games.sort_values(['season', 'week']).reset_index(drop=True)

    player_name = player_info['player_name']
    current_rating = player_info['leaf_rating']
    uncertainty = player_info['leaf_uncertainty']

    # Get QB's most recent team and season
    recent_game = player_games.iloc[-1] if len(player_games) > 0 else None
    qb_team = recent_game['posteam'] if recent_game is not None else None
    qb_season = int(recent_game['season']) if recent_game is not None else None

    # Current rating display
    rating_text = f"{current_rating:.3f} ± {uncertainty:.3f}"

    # Color based on rating
    if current_rating >= 0.15:
        rating_color = {'color': '#2ecc71'}  # Green (elite)
        interpretation = "Elite QB"
    elif current_rating >= 0.05:
        rating_color = {'color': '#3498db'}  # Blue (above average)
        interpretation = "Above Average QB"
    elif current_rating >= -0.05:
        rating_color = {'color': '#95a5a6'}  # Gray (average)
        interpretation = "Average QB"
    else:
        rating_color = {'color': '#e74c3c'}  # Red (below average)
        interpretation = "Below Average QB"

    # Career stats
    total_games = int(player_info['total_games'])
    total_attempts = int(player_info['total_attempts'])
    seasons = player_games['season'].nunique()

    # Check if player is retired
    is_retired = (player_status == 'retired')

    if is_retired:
        # For retired players, show career span and additional stats
        first_season = int(player_games['season'].min())
        last_season = int(player_games['season'].max())

        career_stats_content = html.Div([
            html.P(f"Career: {first_season}-{last_season}", className="mb-1"),
            html.P(f"Total Games: {total_games}", className="mb-1"),
            html.P(f"Total Attempts: {total_attempts:,}", className="mb-1"),
            html.P(f"Seasons Played: {seasons}", className="mb-0")
        ])
    else:
        career_stats_content = html.Div([
            html.P(f"Total Games: {total_games}", className="mb-1"),
            html.P(f"Total Attempts: {total_attempts:,}", className="mb-1"),
            html.P(f"Seasons Played: {seasons}", className="mb-0")
        ])

    # Calculate predictions (skip for retired players)
    if is_retired:
        predictions = pd.DataFrame()
        pred_content = html.Div([
            html.H5("RETIRED", className="text-center text-muted mb-1"),
            html.P(f"Last played: {int(player_games['season'].max())}",
                  className="text-center text-muted small mb-0")
        ])
    elif prediction_years and len(prediction_years) > 0:
        predictions = calculate_predictions(player_games, sorted(prediction_years))

        # Next season prediction
        if len(predictions) > 0:
            next_pred = predictions.iloc[0]
            pred_content = html.Div([
                html.H4(f"{next_pred['predicted_rating']:.3f}",
                       className="text-center mb-1"),
                html.P(f"±{next_pred['predicted_uncertainty']:.3f}",
                      className="text-center text-muted mb-1"),
                html.P(f"({next_pred['predicted_season']:.0f} Season)",
                      className="text-center text-muted small mb-0")
            ])
        else:
            pred_content = html.P("No prediction available", className="text-muted")
    else:
        predictions = pd.DataFrame()
        pred_content = html.P("Predictions disabled", className="text-muted")

    # Create trajectory figure
    fig = create_player_trajectory_figure(player_games, predictions, player_name, is_retired=is_retired)

    return (rating_text, rating_color, interpretation,
            career_stats_content, pred_content, fig)

def main():
    """Run the visualization app."""
    logger.info("=" * 80)
    logger.info("LEAF Rating Visualizer")
    logger.info("=" * 80)
    logger.info(f"\nLoaded {len(qb_data)} QBs with 50+ attempts")
    logger.info(f"Game-by-game data: {len(game_data):,} records")
    logger.info("\nStarting web server...")
    logger.info("Open browser to: http://localhost:8050")
    logger.info("\nPress Ctrl+C to stop the server")

    app.run(debug=True, host='127.0.0.1', port=8050)

# Expose server for production deployment (required by gunicorn)
server = app.server

if __name__ == "__main__":
    main()
