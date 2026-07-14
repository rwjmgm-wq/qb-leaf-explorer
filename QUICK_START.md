# 🚀 Quick Start Guide

Your deployment-ready project is set up! Follow these steps to get it live.

## Project Location

```
C:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\qb-leaf-explorer\
```

## What's Included

✅ **LEAF visualization app** (`scripts/visualization/visualize_leaf_ratings.py`)
✅ **Production-ready server** (`server = app.server` for gunicorn)
✅ **QB data files** (2006-2025 seasons in `data/production/`)
✅ **Minimal dependencies** (`requirements.txt` with only what's needed)
✅ **Git repository initialized** (`.git/`)
✅ **Deployment guide** (`DEPLOYMENT.md`)
✅ **Professional README** (`README.md`)

---

## Step 1: Configure Git (One-Time Setup)

If you haven't set up git before:

```bash
cd "C:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\qb-leaf-explorer"

git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 2: Create Initial Commit

```bash
cd "C:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\qb-leaf-explorer"

git commit -m "Initial commit: QB LEAF Rating Explorer"
```

## Step 3: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `qb-leaf-explorer`
3. Description: "Interactive QB LEAF Rating visualization with ML-powered predictions"
4. **Public** (required for Railway free tier)
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

## Step 4: Push to GitHub

GitHub will show you commands like this (copy from YOUR page):

```bash
cd "C:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\qb-leaf-explorer"

git remote add origin https://github.com/YOUR_USERNAME/qb-leaf-explorer.git
git branch -M main
git push -u origin main
```

## Step 5: Deploy to Railway

1. Go to https://railway.app and sign up with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Select `qb-leaf-explorer`
4. Railway auto-detects Python
5. Go to Settings → Add start command:
   ```
   gunicorn scripts.visualization.visualize_leaf_ratings:server --bind 0.0.0.0:$PORT
   ```
6. Get your live URL from Railway dashboard!

## Step 6: Embed in Ghost.io

Use iframe in Ghost HTML card:

```html
<iframe
  src="https://your-app.railway.app"
  width="100%"
  height="1200px"
  frameborder="0"
></iframe>
```

---

## Optional: Daily Auto-Updates

To set up GitHub Actions for daily data updates, see [DEPLOYMENT.md](DEPLOYMENT.md#part-2-daily-auto-updates-with-github-actions).

---

## Test Locally First

Before deploying, test locally:

```bash
cd "C:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\qb-leaf-explorer"

pip install -r requirements.txt
python scripts/visualization/visualize_leaf_ratings.py
```

Visit http://localhost:8050

---

## Project Structure

```
qb-leaf-explorer/
├── data/
│   └── production/         # QB performance data (CSV files)
├── src/
│   └── features/
│       └── multifeature_rating_calculator.py
├── scripts/
│   └── visualization/
│       └── visualize_leaf_ratings.py  # Main app
├── .gitignore              # What not to commit
├── DEPLOYMENT.md           # Detailed deployment guide
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── QUICK_START.md         # This file!
```

---

## Need Help?

- **Deployment issues**: See [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting)
- **Railway docs**: https://docs.railway.app
- **GitHub docs**: https://docs.github.com

---

## What's Different from Main Project?

This is a **clean deployment package** with only what's needed to run the visualization:

- ❌ No research scripts
- ❌ No ML training code
- ❌ No analysis notebooks
- ✅ Just the visualization app + data
- ✅ Minimal dependencies
- ✅ Production-ready

The main `DARKO_NFL` folder still has all your research code - this is just for deployment!
