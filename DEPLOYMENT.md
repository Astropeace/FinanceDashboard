# 🌐 Deployment Guide

Two ways to share your dashboards publicly while keeping your local AI model running.

---

## ⚡ Option A — ngrok (Share right now, no GitHub needed)

Everything runs on your Mac. ngrok punches a public hole through your firewall.
Your friends' browser → ngrok → your machine.

### Step 1 — Install & auth ngrok
```bash
brew install ngrok/ngrok/ngrok
# Create a free account at https://ngrok.com, then:
ngrok config add-authtoken YOUR_TOKEN_FROM_NGROK_DASHBOARD
```

### Step 2 — Expose your dashboards (2 terminal windows)
```bash
# Terminal 1 — Climate Dashboard
ngrok http 8501

# Terminal 2 — Quant Terminal
ngrok http 8502
```

Each will print a public URL like `https://abc123.ngrok-free.app`. Share those with your friends.
**Your LLM at localhost:1234 doesn't need any changes** — Streamlit still runs on your machine.

---

## 🌐 Option B — Streamlit Community Cloud (Always online, no Mac needed to be awake)

The app lives on Streamlit's servers 24/7. Your LLM still runs on your Mac via a Cloudflare Tunnel.

### Step 1 — Push to GitHub
```bash
cd /Users/willsnoel/.gemini/antigravity/scratch/equity_dashboard
git init
git add app.py quant_terminal.py requirements.txt .streamlit/config.toml
git commit -m "Initial deploy"
# Create a repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Community Cloud (free)
1. Go to **https://share.streamlit.io**
2. Click **"New app"**
3. Connect your GitHub repo
4. Set **Main file path**: `app.py`
5. Click **Deploy** — done. You get a permanent URL like `https://yourname-dashboard.streamlit.app`

### Step 3 — Expose your local LLM via Cloudflare Tunnel (free, no account required)
This makes `localhost:1234` reachable from the internet with a stable URL.

```bash
# Install cloudflared (one time)
brew install cloudflared

# Run the tunnel — keep this terminal open whenever you want AI features to work
cloudflared tunnel --url http://localhost:1234
```

It will print a URL like: `https://random-words.trycloudflare.com`

### Step 4 — Set the LLM URL in Streamlit Cloud
1. In your Streamlit Cloud dashboard, go to **Settings → Secrets**
2. Add this:
```toml
LLM_BASE_URL = "https://random-words.trycloudflare.com"
```
3. The app will automatically pick it up via the `LLM_BASE_URL` environment variable.

> **Note:** The free Cloudflare tunnel URL changes each time you restart `cloudflared`.
> For a stable URL, create a free Cloudflare account and set up a named tunnel.

---

## Summary

| | ngrok | Streamlit Cloud + Cloudflare |
|---|---|---|
| Setup time | 2 minutes | ~20 minutes |
| Always online | ❌ (Mac must be on) | ✅ (app) / ❌ (LLM requires Mac) |
| Cost | Free tier (URL changes) | Free |
| Best for | Showing friends today | Permanent link |
| LLM works when | Always (runs locally) | Only when cloudflared is running |
