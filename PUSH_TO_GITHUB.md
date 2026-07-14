# 🚀 Push to GitHub - Copy/Paste Instructions

✅ Git is configured and committed! Here's what you need to do:

---

## Step 1: Create GitHub Repository

Go to: **https://github.com/new**

Fill in:
- **Repository name**: `qb-leaf-explorer`
- **Description**: `Interactive QB LEAF Rating visualization with ML-powered predictions (2006-2025)`
- **Visibility**: **Public** (required for Railway free tier)
- ❌ **DO NOT** check "Add a README file"
- ❌ **DO NOT** check "Add .gitignore"
- ❌ **DO NOT** check "Choose a license"

Click **"Create repository"**

---

## Step 2: Push Your Code

GitHub will show you commands - **IGNORE THEM** and use these instead:

### Copy and paste these commands one at a time:

```bash
cd "C:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\qb-leaf-explorer"
```

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/qb-leaf-explorer.git
```
**⚠️ IMPORTANT:** Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username!

```bash
git branch -M main
```

```bash
git push -u origin main
```

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a [Personal Access Token](https://github.com/settings/tokens) (NOT your password)

---

## Step 3: Verify

After pushing, go to:
```
https://github.com/YOUR_GITHUB_USERNAME/qb-leaf-explorer
```

You should see all your files!

---

## Step 4: Deploy to Railway

1. Go to **https://railway.app**
2. Sign up/log in with **GitHub**
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select **`qb-leaf-explorer`**
5. Railway auto-detects Python and installs dependencies
6. Go to **Settings** tab
7. Add **Start Command**:
   ```
   gunicorn scripts.visualization.visualize_leaf_ratings:server --bind 0.0.0.0:$PORT
   ```
8. Click **"Save"**
9. Wait 2-3 minutes for deployment
10. Click **"Deployments"** tab → Find your URL!

---

## Your Live URL

Railway will give you something like:
```
https://qb-leaf-explorer-production.up.railway.app
```

Test it, then embed in Ghost.io:

```html
<iframe
  src="https://qb-leaf-explorer-production.up.railway.app"
  width="100%"
  height="1200px"
  frameborder="0"
></iframe>
```

---

## Troubleshooting

**"Application failed to respond"** on Railway:
- Check Railway logs (Deployments → View Logs)
- Verify start command is correct
- Make sure `server = app.server` is in visualize_leaf_ratings.py (✅ already done)

**Git push authentication failed:**
- Create a Personal Access Token: https://github.com/settings/tokens
- Use token instead of password

**Need help?**
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed troubleshooting

---

## What's Already Done ✅

- ✅ Git repository initialized
- ✅ All files committed (24 files, 110,214 lines)
- ✅ Production server configured
- ✅ Requirements.txt optimized
- ✅ README and docs created

**You just need to:**
1. Create GitHub repo (2 minutes)
2. Push code (1 minute)
3. Deploy to Railway (3 minutes)

**Total time: ~6 minutes** 🎉
