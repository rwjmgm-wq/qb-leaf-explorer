# 🚀 LEAF Visualization Deployment Guide

This guide will help you deploy the LEAF Rating visualization to the web and set up daily automatic updates.

## 📋 Overview

**What you'll have:**
- Live web app at `https://your-app.railway.app`
- Auto-updates daily with fresh NFL data
- Embeddable in your Ghost.io blog

**Time to set up:** ~30 minutes

---

## Part 1: Hosting on Railway.app (FREE)

### Step 1: Prepare Your GitHub Repository

1. **Create a new repository** on GitHub (or use existing)
   - Name it something like `leaf-ratings-viz`
   - Make it public (required for free tier)

2. **Push your code:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: LEAF visualization"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/leaf-ratings-viz.git
   git push -u origin main
   ```

### Step 2: Deploy to Railway

1. Go to [railway.app](https://railway.app) and sign up (use GitHub account)

2. Click **"New Project"** → **"Deploy from GitHub repo"**

3. Select your `leaf-ratings-viz` repository

4. Railway will auto-detect Python and install dependencies

5. **Add start command:**
   - Go to your project → **Settings** → **Start Command**
   - Enter: `gunicorn scripts.visualization.visualize_leaf_ratings:server --bind 0.0.0.0:$PORT`

6. **Set environment variables** (if needed):
   - Go to **Variables** tab
   - Add any API keys if your data pipeline needs them

7. **Get your URL:**
   - Railway will give you a URL like `https://leaf-ratings-production.up.railway.app`
   - This is your live visualization!

### Step 3: Update Visualization for Production

You need to modify [visualize_leaf_ratings.py](scripts/visualization/visualize_leaf_ratings.py) to expose the server:

Add these lines at the end of the file (before `if __name__ == "__main__":`):

```python
# Expose server for production deployment
server = app.server
```

---

## Part 2: Daily Auto-Updates with GitHub Actions

### Option A: Update Data Files Daily (Recommended)

Create `.github/workflows/update-data.yml`:

```yaml
name: Update LEAF Data Daily

on:
  schedule:
    - cron: '0 12 * * *'  # Runs daily at 12pm UTC (7am EST)
  workflow_dispatch:  # Allows manual trigger

jobs:
  update-data:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run LEAF data pipeline
      run: |
        python scripts/legacy/deploy_leaf_v2.py --seasons 2025

    - name: Commit updated data
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add data/production/*.csv
        git diff --quiet && git diff --staged --quiet || git commit -m "Auto-update LEAF data [$(date +'%Y-%m-%d')]"

    - name: Push changes
      uses: ad-m/github-push-action@master
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
```

**How it works:**
1. Every day at 12pm UTC, GitHub Actions runs your data pipeline
2. Updated CSV files are committed to your repo
3. Railway detects the commit and auto-redeploys
4. Your visualization updates automatically!

### Option B: Simpler - Manual Weekly Updates

If daily is overkill, just run this locally once a week:

```bash
# Update data
python scripts/legacy/deploy_leaf_v2.py --seasons 2025

# Commit and push
git add data/production/*.csv
git commit -m "Weekly data update"
git push
```

Railway will auto-deploy the updates.

---

## Part 3: Embed in Ghost.io Blog

### Method 1: Full iFrame Embed

In your Ghost post, use HTML card:

```html
<iframe
  src="https://your-app.railway.app"
  width="100%"
  height="1200px"
  frameborder="0"
  style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);"
></iframe>
```

### Method 2: Link Button (Cleaner)

```html
<a href="https://your-app.railway.app"
   target="_blank"
   style="display: inline-block; padding: 12px 24px; background: #e52673; color: white; border-radius: 6px; text-decoration: none; font-weight: 600;">
  Explore QB LEAF Ratings →
</a>
```

### Method 3: Screenshot with Link

1. Take a nice screenshot of the visualization
2. Upload to Ghost
3. Link the image to your Railway URL

---

## 🎯 Quick Checklist

- [ ] Code pushed to GitHub
- [ ] Railway project created and deployed
- [ ] Added `server = app.server` to visualize_leaf_ratings.py
- [ ] Visualization loads at Railway URL
- [ ] GitHub Actions workflow created (optional)
- [ ] First data update runs successfully
- [ ] Embedded in Ghost.io blog post

---

## 🔧 Troubleshooting

### "Application failed to respond"
- Check that you added `server = app.server` to the file
- Verify start command in Railway settings

### "Module not found"
- Make sure `requirements.txt` is in the root directory
- Check Railway build logs

### Data not updating
- Check GitHub Actions logs
- Make sure workflow file is in `.github/workflows/`
- Verify cron schedule is correct (use [crontab.guru](https://crontab.guru))

### Ghost.io iframe not showing
- Check your Ghost theme allows iframes
- Try Method 2 (link button) instead

---

## 💰 Cost Estimate

- **Railway Free Tier**: 500 hours/month (plenty for this app)
- **GitHub Actions**: 2000 minutes/month free
- **Ghost.io**: Your existing plan

**Total additional cost: $0/month** ✅

---

## 🚀 Alternative Hosting Options

If you outgrow Railway or want alternatives:

### Render.com
- Similar to Railway
- 750 hours/month free
- Setup: Same as Railway

### Heroku
- $5/month for basic dyno
- Very stable, more features
- Setup: Similar to Railway, use Procfile

### PythonAnywhere
- $5/month for web app
- Good for smaller traffic
- Direct Python hosting

---

## 📝 Notes

- **Data freshness**: With daily updates, your visualization always shows current season data
- **Performance**: Railway free tier handles ~50-100 concurrent users easily
- **Scaling**: If you get more traffic, upgrade to Railway Pro ($5/month)
- **Custom domain**: Railway allows custom domains on free tier!

---

## Need Help?

- Railway docs: https://docs.railway.app
- GitHub Actions docs: https://docs.github.com/actions
- Ghost.io HTML embedding: https://ghost.org/help/

