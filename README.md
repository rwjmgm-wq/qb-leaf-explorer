# QB LEAF Rating Explorer

Interactive web application for exploring quarterback performance using the **LEAF (Latent Evaluation of Aggregated Fundamentals)** rating system.

## Features

- **20+ years of QB data** (2006-2025 NFL seasons)
- **Interactive trajectory plots** showing career performance trends
- **Future performance predictions** powered by ML models (r=0.41 correlation)
- **Active and retired QB filtering**
- **Mobile-responsive design** matching Duncan Drafts brand

## What is LEAF?

LEAF is a multi-feature quarterback rating system that:
- Combines 5 core efficiency metrics (EPA, success rate, CPOE, etc.)
- Adjusts for opponent strength using Kalman filtering
- Predicts future performance 16 games ahead
- Achieves 0.41 correlation with actual outcomes

## Live Demo

Coming soon at: `https://your-app.railway.app`

## Local Development

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qb-leaf-explorer.git
cd qb-leaf-explorer

# Install dependencies
pip install -r requirements.txt

# Run the app
python scripts/visualization/visualize_leaf_ratings.py
```

Visit http://localhost:8050 in your browser.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions on:
- Hosting on Railway.app (free tier)
- Setting up daily auto-updates with GitHub Actions
- Embedding in Ghost.io blog

## Data Sources

- **Play-by-Play Data**: nflfastR (2006-2025 seasons)
- **Model**: LEAF v2.0 Multi-Feature Rating System
- **Validation**: r=0.41 correlation to future performance (16 games ahead)

## Project Structure

```
qb-leaf-explorer/
├── scripts/
│   └── visualization/
│       └── visualize_leaf_ratings.py  # Main Dash application
├── src/
│   └── features/
│       └── multifeature_rating_calculator.py  # Rating calculations
├── data/
│   └── production/
│       └── *.csv  # QB performance data
├── requirements.txt
└── README.md
```

## Technology Stack

- **Backend**: Python, Dash, Plotly
- **Frontend**: Dash Bootstrap Components, custom CSS
- **Deployment**: Gunicorn, Railway.app
- **CI/CD**: GitHub Actions (daily data updates)

## Author

Created by [Duncan Drafts](https://duncan-drafts.ghost.io)
Data-driven NFL analysis and prospect evaluation

## License

MIT License - feel free to use for your own projects!

## Contributing

Found a bug or have a feature request? Please open an issue!
