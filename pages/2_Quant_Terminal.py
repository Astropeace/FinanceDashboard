"""
Jane Street-style Quantitative Analysis Terminal
─────────────────────────────────────────────────
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import xml.etree.ElementTree as ET
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from shared import (
    LLM_BASE_URL, WATCHLIST, ALL_TICKERS, TICKER_TO_SECTOR,
    BENCH_TICKER, VIX_TICKER, CLIMATE_IMPACT_MATRIX, SECTOR_ORDER,
)


st.markdown("""
<style>
    /* terminal mono feel */
    code, .stCode { font-family: 'JetBrains Mono', 'Courier New', monospace !important; }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700; font-family: monospace; }
    [data-testid="stMetricLabel"] { font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    .stTabs [data-baseweb="tab"]  { font-weight: 700; font-size: 0.88rem; letter-spacing: 0.03em; }
    hr { margin: 0.3rem 0 0.8rem 0 !important; }
    /* signal badges */
    .sig-strong-buy  { color:#00ff88; font-weight:800; }
    .sig-buy         { color:#00cc66; font-weight:600; }
    .sig-neutral     { color:#aaaaaa; font-weight:400; }
    .sig-sell        { color:#ff6644; font-weight:600; }
    .sig-strong-sell { color:#ff2222; font-weight:800; }
    .stat-block {
        background:#111827; border:1px solid #1f2937;
        border-radius:6px; padding:0.7rem 1rem; margin-bottom:0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# QUANT PARAMETERS
# ─────────────────────────────────────────────
LOOKBACK    = '1y'   # price history window
ZSCORE_DAYS = 20     # rolling z-score window
MOM_SHORT   = 21     # ~1 month momentum
MOM_LONG    = 63     # ~3 month momentum
GDACS_URL   = "https://gdacs.org/xml/rss.xml"


# ─────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_price_matrix() -> pd.DataFrame:
    """Returns a wide DataFrame of daily adjusted close prices for all tickers + bench + VIX."""
    all_syms = ALL_TICKERS + [BENCH_TICKER, VIX_TICKER]
    raw = yf.download(all_syms, period=LOOKBACK, auto_adjust=True, progress=False)
    closes = raw['Close'] if 'Close' in raw else raw
    return closes.dropna(how='all')


@st.cache_data(ttl=300, show_spinner=False)
def load_fundamentals() -> pd.DataFrame:
    rows = []
    for sym in ALL_TICKERS:
        try:
            info = yf.Ticker(sym).info
            rows.append({
                'Ticker':         sym,
                'Sector':         TICKER_TO_SECTOR.get(sym, ''),
                'Price':          info.get('currentPrice', info.get('regularMarketPrice')),
                'Forward P/E':    info.get('forwardPE'),
                'PEG':            info.get('pegRatio'),
                'EV/EBITDA':      info.get('enterpriseToEbitda'),
                'Gross Margin':   info.get('grossMargins'),
                'Rev Growth':     info.get('revenueGrowth'),
                'Short Float %':  info.get('shortPercentOfFloat'),
                'Beta':           info.get('beta'),
                'Market Cap (B)': info.get('marketCap', 0) / 1e9 if info.get('marketCap') else None,
                '52W High':       info.get('fiftyTwoWeekHigh'),
                '52W Low':        info.get('fiftyTwoWeekLow'),
                'Analyst Target': info.get('targetMeanPrice'),
            })
        except Exception:
            rows.append({'Ticker': sym, 'Sector': TICKER_TO_SECTOR.get(sym,'')})
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_climate_events() -> pd.DataFrame:
    events = []
    try:
        r    = requests.get(GDACS_URL, timeout=10)
        root = ET.fromstring(r.content)
        ns   = {'georss': 'http://www.georss.org/georss'}
        for item in root.findall('./channel/item')[:80]:
            title = getattr(item.find('title'),       'text', '') or ''
            desc  = getattr(item.find('description'), 'text', '') or ''
            pub   = getattr(item.find('pubDate'),     'text', '') or ''
            lat = lon = None
            pt = item.find('georss:point', ns)
            if pt is not None and pt.text:
                try: lat, lon = map(float, pt.text.split())
                except: pass
            text = (title + ' ' + desc).lower()
            if   'earthquake' in text:                                   etype = 'Earthquake'
            elif 'flood'      in text:                                   etype = 'Flood'
            elif any(w in text for w in ('cyclone','hurricane','typhoon')): etype = 'Cyclone'
            elif 'volcan'     in text:                                   etype = 'Volcano'
            elif 'drought'    in text:                                   etype = 'Drought'
            elif 'tsunami'    in text:                                   etype = 'Tsunami'
            else:                                                        etype = 'Other'
            severity = 'Red' if 'red' in text else ('Orange' if 'orange' in text else 'Green')
            sev_score = {'Red': 3, 'Orange': 2, 'Green': 1}.get(severity, 1)
            try:    dt = parsedate_to_datetime(pub)
            except: dt = datetime.now(timezone.utc)
            events.append({'Event': title[:70], 'Type': etype, 'Severity': severity,
                           'SevScore': sev_score, 'ParsedDate': dt, 'lat': lat, 'lon': lon})
        df = pd.DataFrame(events)
        if not df.empty:
            now    = datetime.now(timezone.utc)
            df['Hours Ago'] = (now - df['ParsedDate']).dt.total_seconds() / 3600
            # Recency weight: events in last 24h get full weight, decay over 7 days
            df['RecencyWeight'] = np.clip(1.0 - df['Hours Ago'] / 168, 0.05, 1.0)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# QUANT ENGINE
# ─────────────────────────────────────────────
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def compute_zscore(series: pd.Series, window: int = ZSCORE_DAYS) -> pd.Series:
    """Rolling z-score of a return series."""
    roll_mean = series.rolling(window).mean()
    roll_std  = series.rolling(window).std()
    return (series - roll_mean) / roll_std.replace(0, np.nan)

def compute_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """1-month and 3-month momentum scores."""
    rows = []
    for col in ALL_TICKERS:
        if col not in prices.columns: continue
        s = prices[col].dropna()
        if len(s) < MOM_LONG + 5: continue
        mom1  = (s.iloc[-1] / s.iloc[-MOM_SHORT] - 1) * 100  if len(s) > MOM_SHORT  else None
        mom3  = (s.iloc[-1] / s.iloc[-MOM_LONG]  - 1) * 100  if len(s) > MOM_LONG   else None
        rows.append({'Ticker': col, 'Mom 1M (%)': mom1, 'Mom 3M (%)': mom3})
    return pd.DataFrame(rows)

def compute_beta(returns: pd.DataFrame) -> pd.DataFrame:
    """Beta of each ticker vs SPY over the full lookback window."""
    rows = []
    bench = BENCH_TICKER
    if bench not in returns.columns:
        return pd.DataFrame()
    bench_r = returns[bench].dropna()
    for col in ALL_TICKERS:
        if col not in returns.columns: continue
        tk_r = returns[col].dropna()
        common = bench_r.index.intersection(tk_r.index)
        if len(common) < 60: continue
        b, a, r, p, se = stats.linregress(bench_r.loc[common], tk_r.loc[common])
        rows.append({'Ticker': col, 'Beta': round(b, 3), 'Alpha (Ann %)': round(a * 252 * 100, 2),
                     'R²': round(r**2, 3), 'Sharpe (Est)': round((tk_r.mean()*252) / (tk_r.std()*np.sqrt(252)), 3)})
    return pd.DataFrame(rows)

def compute_correlation_matrix(returns: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Pearson correlation matrix for the last ZSCORE_DAYS*3 days."""
    valid = [t for t in tickers if t in returns.columns]
    recent = returns[valid].tail(ZSCORE_DAYS * 3)
    return recent.corr().round(3)

def compute_current_zscore(returns: pd.DataFrame) -> pd.DataFrame:
    """Latest rolling z-score for each ticker's return."""
    rows = []
    for col in ALL_TICKERS:
        if col not in returns.columns: continue
        z_series = compute_zscore(returns[col])
        if z_series.dropna().empty: continue
        latest_z   = z_series.iloc[-1]
        vol_20d    = returns[col].tail(20).std() * np.sqrt(252) * 100
        rows.append({'Ticker': col, 'Return Z-Score': round(latest_z, 2), 'Realized Vol 20D (%)': round(vol_20d, 2)})
    return pd.DataFrame(rows)

def compute_pair_divergence(prices: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """
    Finds pairs within the same sector whose 30-day log-price spread
    has deviated > 1.5 std from its mean — a potential mean-reversion signal.
    """
    results = []
    for sector, tickers in WATCHLIST.items():
        valid = [t for t in tickers if t in prices.columns]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, b = valid[i], valid[j]
                spread = np.log(prices[a]) - np.log(prices[b])
                spread = spread.dropna().tail(60)
                if len(spread) < 30: continue
                mu, sigma = spread.mean(), spread.std()
                if sigma < 1e-6: continue
                z  = (spread.iloc[-1] - mu) / sigma
                results.append({
                    'Pair':         f"{a} / {b}",
                    'Sector':       sector,
                    'Spread Z':     round(z, 2),
                    'Signal':       '← Long A / Short B' if z < -1.5 else ('← Long B / Short A' if z > 1.5 else '— Neutral'),
                    'Actionable':   abs(z) > 1.5,
                })
    return pd.DataFrame(results).sort_values('Spread Z', key=abs, ascending=False)

def compute_climate_score(events: pd.DataFrame) -> pd.DataFrame:
    """
    Proprietary Climate Impact Score per sector:
    Sums (impact_weight × severity_score × recency_weight) across all recent events.
    Returns a score in [-10, +10].
    """
    if events.empty:
        return pd.DataFrame({'Sector': SECTOR_ORDER, 'Climate Impact Score': [0.0] * len(SECTOR_ORDER)})
    scores = {s: 0.0 for s in SECTOR_ORDER}
    for _, ev in events.iterrows():
        etype   = ev.get('Type', 'Other')
        weights = CLIMATE_IMPACT_MATRIX.get(etype, CLIMATE_IMPACT_MATRIX['Other'])['weights']
        sev     = ev.get('SevScore', 1)
        rec     = ev.get('RecencyWeight', 0.5)
        for idx, sector in enumerate(SECTOR_ORDER):
            scores[sector] += weights[idx] * sev * rec
    rows = [{'Sector': s, 'Climate Impact Score': round(v, 3)} for s, v in scores.items()]
    return pd.DataFrame(rows).sort_values('Climate Impact Score', ascending=False)

def signal_label(z: float) -> str:
    if   z >  2.0: return "STRONG BUY"
    elif z >  1.0: return "BUY"
    elif z < -2.0: return "STRONG SELL"
    elif z < -1.0: return "SELL"
    else:          return "NEUTRAL"


# ─────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────
with st.spinner("Initializing quant engine…"):
    prices       = load_price_matrix()
    fundamentals = load_fundamentals()
    climate_ev   = load_climate_events()

returns       = compute_returns(prices) if not prices.empty else pd.DataFrame()
momentum      = compute_momentum(prices)
beta_df       = compute_beta(returns) if not returns.empty else pd.DataFrame()
zscore_df     = compute_current_zscore(returns) if not returns.empty else pd.DataFrame()
corr_matrix   = compute_correlation_matrix(returns, ALL_TICKERS) if not returns.empty else pd.DataFrame()
pairs_df      = compute_pair_divergence(prices, fundamentals)
climate_score = compute_climate_score(climate_ev)

# Master table: join everything
master = fundamentals.copy()
if not momentum.empty:
    master = master.merge(momentum,  on='Ticker', how='left')
if not beta_df.empty:
    master = master.merge(beta_df,   on='Ticker', how='left')
if not zscore_df.empty:
    master = master.merge(zscore_df, on='Ticker', how='left')
if 'Return Z-Score' in master.columns:
    master['Signal'] = master['Return Z-Score'].apply(
        lambda z: signal_label(z) if pd.notna(z) else 'N/A'
    )

last_sync = datetime.now(timezone.utc)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📐 QT Controls")
    st.caption(f"Sync: {last_sync.strftime('%b %d · %H:%M UTC')}")
    if st.button("⟳ Refresh Engine", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Signal Thresholds")
    z_threshold   = st.slider("Z-Score Flag Threshold", 0.5, 3.0, 1.5, 0.1)
    pair_z_thresh = st.slider("Pair Divergence Threshold", 0.5, 3.0, 1.5, 0.1)

    st.divider()
    st.subheader("🧠 AI Engine")
    llm_endpoint = st.text_input(
        "LLM Endpoint",
        value=st.session_state.get("llm_endpoint", LLM_BASE_URL),
        help="Paste your Cloudflare tunnel URL here. Saved for this session.",
        placeholder="https://xxxx.trycloudflare.com",
    )
    st.session_state["llm_endpoint"] = llm_endpoint
    model_id    = st.text_input("Model ID", value="qwen2.5-coder-14b-instruct-mlx")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05,
                            help="Lower = more analytical/deterministic output")

    st.divider()
    st.subheader("🌍 Disaster Filter")
    hours_back = st.slider("Event window (hours)", 12, 168, 72, 12)
    min_sev    = st.selectbox("Min Severity", ['Green','Orange','Red'], index=0)
    sev_order  = {'Green': 1, 'Orange': 2, 'Red': 3}

    st.divider()
    st.subheader("Active Signals")
    if 'Signal' in master.columns:
        buys  = master[master['Signal'].str.contains('BUY',  na=False)]
        sells = master[master['Signal'].str.contains('SELL', na=False)]
        if not buys.empty:
            st.markdown("**Buys**")
            for _, r in buys.iterrows():
                st.markdown(f"`{r['Ticker']}` — {r['Signal']}")
        if not sells.empty:
            st.markdown("**Sells**")
            for _, r in sells.iterrows():
                st.markdown(f"`{r['Ticker']}` — {r['Signal']}")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📐 Quantitative Analysis Terminal")
st.caption(
    f"Universe: {len(ALL_TICKERS)} tickers · "
    f"Lookback: {LOOKBACK} · "
    f"Z-window: {ZSCORE_DAYS}D · "
    f"Benchmark: {BENCH_TICKER} · "
    f"Sync: {last_sync.strftime('%b %d %Y %H:%M UTC')}"
)
st.divider()


# ─────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────
vix_now  = prices[VIX_TICKER].iloc[-1]  if VIX_TICKER in prices.columns and not prices.empty else None
spy_1d   = returns[BENCH_TICKER].iloc[-1]*100 if BENCH_TICKER in returns.columns and not returns.empty else None
spy_mom  = ((prices[BENCH_TICKER].iloc[-1] / prices[BENCH_TICKER].iloc[-MOM_LONG] - 1)*100
            if BENCH_TICKER in prices.columns and len(prices) > MOM_LONG else None)
n_buys   = int(master['Signal'].str.contains('BUY',  na=False).sum())  if 'Signal' in master.columns else 0
n_sells  = int(master['Signal'].str.contains('SELL', na=False).sum())  if 'Signal' in master.columns else 0
n_flagged_pairs = int(pairs_df['Actionable'].sum()) if not pairs_df.empty else 0
cl_score_max = climate_score['Climate Impact Score'].max() if not climate_score.empty else 0

regime = "RISK-OFF 🔴" if (vix_now or 0) > 25 else ("ELEVATED ⚠️" if (vix_now or 0) > 18 else "RISK-ON 🟢")

k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
k1.metric("Market Regime",   regime)
k2.metric("VIX",             f"{vix_now:.1f}"     if vix_now  else "—")
k3.metric("SPY 1D",          f"{spy_1d:+.2f}%"    if spy_1d   else "—",  delta=f"{spy_1d:+.2f}%" if spy_1d else None)
k4.metric("SPY 3M Mom",      f"{spy_mom:+.1f}%"   if spy_mom  else "—")
k5.metric("Buy Signals",     n_buys,               delta=f"+{n_buys}" if n_buys else None, delta_color="normal")
k6.metric("Sell Signals",    n_sells,              delta=f"-{n_sells}" if n_sells else None, delta_color="inverse")
k7.metric("Active Pairs",    n_flagged_pairs,      delta="mean-revert" if n_flagged_pairs else None, delta_color="off")

st.divider()


# ─────────────────────────────────────────────
# ROW 1: Z-SCORE SCREENER + CLIMATE IMPACT
# ─────────────────────────────────────────────
col_z, col_cl = st.columns([1.2, 1])

with col_z:
    st.subheader("🔬 Return Z-Score Screener")
    st.caption(f"Rolling {ZSCORE_DAYS}-day z-score of daily returns. |z| > {z_threshold} = flagged.")
    if not zscore_df.empty:
        df_z = zscore_df.merge(master[['Ticker','Sector','Signal']], on='Ticker', how='left')
        df_z = df_z.sort_values('Return Z-Score')
        colors_z = []
        for v in df_z['Return Z-Score']:
            if   v >  2.0: colors_z.append('#00ff88')
            elif v >  1.0: colors_z.append('#00cc66')
            elif v < -2.0: colors_z.append('#ff2222')
            elif v < -1.0: colors_z.append('#ff6644')
            else:          colors_z.append('#555577')

        fig_z = go.Figure(go.Bar(
            x=df_z['Return Z-Score'], y=df_z['Ticker'],
            orientation='h', marker_color=colors_z,
            text=[f"{v:+.2f}σ  {s}" for v, s in zip(df_z['Return Z-Score'], df_z['Signal'])],
            textposition='outside',
            customdata=df_z[['Sector','Realized Vol 20D (%)']].values,
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Z: %{x:+.2f}σ<br>20D Vol: %{customdata[1]:.1f}%<extra></extra>",
        ))
        fig_z.add_vline(x= z_threshold, line_dash="dot", line_color="#00cc66", opacity=0.7)
        fig_z.add_vline(x=-z_threshold, line_dash="dot", line_color="#ff6644", opacity=0.7)
        fig_z.add_vline(x=0, line_color="white", opacity=0.2)
        fig_z.update_layout(
            template='plotly_dark', title=f"Return Z-Score (±{z_threshold}σ threshold shown)",
            xaxis=dict(title='Z-Score (σ)', zeroline=False),
            margin=dict(l=10, r=100, t=45, b=10), height=420,
        )
        st.plotly_chart(fig_z, use_container_width=True)

with col_cl:
    st.subheader("🌡️ Climate Impact Score by Sector")
    st.caption("Proprietary score: Σ(event_weight × severity × recency_decay). Range ~[-10, +10].")
    if not climate_score.empty:
        cs = climate_score.sort_values('Climate Impact Score')
        colors_cl = ['#00cc88' if v > 0.2 else ('#ff6644' if v < -0.2 else '#666688')
                     for v in cs['Climate Impact Score']]
        fig_cl = go.Figure(go.Bar(
            x=cs['Climate Impact Score'], y=cs['Sector'],
            orientation='h', marker_color=colors_cl,
            text=[f"{v:+.3f}" for v in cs['Climate Impact Score']],
            textposition='outside',
        ))
        fig_cl.add_vline(x=0, line_color="white", opacity=0.3)
        fig_cl.update_layout(
            template='plotly_dark',
            title='Net Climate Tailwind / Headwind (live GDACS events)',
            xaxis_title='Impact Score',
            margin=dict(l=10, r=80, t=45, b=10), height=420,
        )
        st.plotly_chart(fig_cl, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────
# ROW 2: CORRELATION MATRIX + PAIR DIVERGENCE
# ─────────────────────────────────────────────
col_corr, col_pairs = st.columns([1.1, 1])

with col_corr:
    st.subheader("📐 Correlation Matrix")
    st.caption(f"Pearson correlations — rolling {ZSCORE_DAYS*3}D daily returns.")
    if not corr_matrix.empty:
        tickers_in = corr_matrix.index.tolist()
        z_vals  = corr_matrix.values
        annots  = [[f"{v:.2f}" for v in row] for row in z_vals]
        fig_corr = go.Figure(go.Heatmap(
            z=z_vals, x=tickers_in, y=tickers_in,
            colorscale='RdBu', reversescale=True,
            zmin=-1, zmax=1,
            text=annots, texttemplate="%{text}",
            textfont=dict(size=9),
            hoverongaps=False,
            colorbar=dict(title='Corr'),
        ))
        fig_corr.update_layout(
            template='plotly_dark',
            title='Pairwise Return Correlation (red=+1, blue=-1)',
            margin=dict(l=10, r=10, t=50, b=80),
            height=440,
            xaxis=dict(side='bottom', tickangle=-45),
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

with col_pairs:
    st.subheader("⚖️ Intra-Sector Pair Divergence")
    st.caption(f"Log-price spread z-score vs 60-day mean. |z| > {pair_z_thresh}σ = mean-reversion candidate.")
    if not pairs_df.empty:
        df_p = pairs_df[pairs_df['Spread Z'].abs() >= pair_z_thresh] if pair_z_thresh > 0 else pairs_df
        if df_p.empty:
            df_p = pairs_df.head(8)
        colors_p = ['#00cc88' if z < 0 else '#ff6644' for z in df_p['Spread Z']]
        fig_p = go.Figure(go.Bar(
            x=df_p['Spread Z'], y=df_p['Pair'],
            orientation='h', marker_color=colors_p,
            text=df_p['Signal'],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Spread Z: %{x:+.2f}σ<br>%{text}<extra></extra>",
        ))
        fig_p.add_vline(x= pair_z_thresh, line_dash="dot", line_color="#00cc66", opacity=0.7)
        fig_p.add_vline(x=-pair_z_thresh, line_dash="dot", line_color="#ff6644", opacity=0.7)
        fig_p.add_vline(x=0, line_color="white", opacity=0.2)
        fig_p.update_layout(
            template='plotly_dark',
            title=f"Spread Z-Score — Candidates > {pair_z_thresh}σ",
            margin=dict(l=10, r=180, t=45, b=10), height=440,
        )
        st.plotly_chart(fig_p, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────
# ROW 3: MOMENTUM FACTOR + BETA / RISK
# ─────────────────────────────────────────────
col_mom, col_beta = st.columns(2)

with col_mom:
    st.subheader("🏃 Momentum Factor")
    st.caption(f"1M (~{MOM_SHORT}D) and 3M (~{MOM_LONG}D) price momentum. Sorted by 1M.")
    if not momentum.empty:
        df_m = momentum.merge(master[['Ticker','Sector']], on='Ticker', how='left').dropna(subset=['Mom 1M (%)'])
        df_m = df_m.sort_values('Mom 1M (%)')
        fig_mom = go.Figure()
        fig_mom.add_trace(go.Bar(
            name='1M Mom', x=df_m['Mom 1M (%)'], y=df_m['Ticker'],
            orientation='h',
            marker_color=['#00cc88' if v >= 0 else '#ff4444' for v in df_m['Mom 1M (%)']],
            text=[f"{v:+.1f}%" for v in df_m['Mom 1M (%)']],
            textposition='outside', opacity=0.95,
        ))
        if 'Mom 3M (%)' in df_m.columns:
            fig_mom.add_trace(go.Scatter(
                name='3M Mom', x=df_m['Mom 3M (%)'], y=df_m['Ticker'],
                mode='markers', marker=dict(symbol='diamond', size=9, color='#ffffff', line=dict(width=1, color='#aaaaaa')),
            ))
        fig_mom.add_vline(x=0, line_color="white", opacity=0.3)
        fig_mom.update_layout(
            template='plotly_dark', title='1M Bar / 3M Diamond',
            margin=dict(l=10, r=80, t=45, b=10), height=380,
            legend=dict(orientation='h', y=1.08),
        )
        st.plotly_chart(fig_mom, use_container_width=True)

with col_beta:
    st.subheader("📏 Beta & Risk Decomposition")
    st.caption(f"vs {BENCH_TICKER} — full {LOOKBACK} window regression. Sharpe = annualised.")
    if not beta_df.empty:
        df_b = beta_df.merge(master[['Ticker','Sector']], on='Ticker', how='left').sort_values('Beta')
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(
            name='Beta', x=df_b['Beta'], y=df_b['Ticker'],
            orientation='h',
            marker_color=['#aaaaff' if v <= 1 else '#ff9944' for v in df_b['Beta']],
            text=[f"β={v:.2f}" for v in df_b['Beta']],
            textposition='outside', opacity=0.9,
        ))
        fig_b.add_trace(go.Scatter(
            name='Sharpe', x=df_b['Sharpe (Est)'], y=df_b['Ticker'],
            mode='markers', marker=dict(symbol='circle', size=10, color='#00ffcc',
                                        line=dict(width=1, color='white')),
            xaxis='x2',
        ))
        fig_b.update_layout(
            template='plotly_dark', title='Beta (bar) + Estimated Sharpe (cyan dot)',
            xaxis=dict(title='Beta',            side='bottom'),
            xaxis2=dict(title='Sharpe (Est)',   side='top',   overlaying='x', showgrid=False),
            margin=dict(l=10, r=80, t=60, b=10), height=380,
            legend=dict(orientation='h', y=1.13),
        )
        fig_b.add_vline(x=1.0, line_dash="dash", line_color="white", opacity=0.3,
                        annotation_text="β=1", annotation_position="top left")
        st.plotly_chart(fig_b, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────
# ROW 4: MASTER DATA TABLE
# ─────────────────────────────────────────────
st.subheader("📋 Full Quant Ledger")
tab_master, tab_pairs, tab_events, tab_raw = st.tabs([
    "🧮 Master Table", "⚖️ Pair Candidates", "🌍 Climate Feed", "📈 Price History"
])

with tab_master:
    if not master.empty:
        show_cols = ['Ticker','Sector','Signal','Price','Return Z-Score','Realized Vol 20D (%)',
                     'Beta','Sharpe (Est)','Mom 1M (%)','Mom 3M (%)','Forward P/E','PEG',
                     'EV/EBITDA','Gross Margin','Rev Growth','Short Float %','Market Cap (B)']
        df_ms = master[[c for c in show_cols if c in master.columns]].copy()
        # Colour-code signal
        def signal_color(s):
            mp = {
                'STRONG BUY':  'background-color:#003322',
                'BUY':         'background-color:#001a11',
                'STRONG SELL': 'background-color:#330000',
                'SELL':        'background-color:#1a0000',
            }
            return [f"background-color:{'#003322' if v=='STRONG BUY' else '#001a11' if v=='BUY' else '#330000' if v=='STRONG SELL' else '#1a0000' if v=='SELL' else ''}" for v in s]

        st.dataframe(
            df_ms,
            column_config={
                'Ticker':                st.column_config.TextColumn('Ticker', width='small'),
                'Sector':                st.column_config.TextColumn('Cohort'),
                'Signal':                st.column_config.TextColumn('Signal'),
                'Price':                 st.column_config.NumberColumn('Price',        format='$%.2f'),
                'Return Z-Score':        st.column_config.NumberColumn('Z-Score',      format='%.2fσ'),
                'Realized Vol 20D (%)':  st.column_config.NumberColumn('RVol 20D',     format='%.1f%%'),
                'Beta':                  st.column_config.NumberColumn('Beta',          format='%.3f'),
                'Sharpe (Est)':          st.column_config.NumberColumn('Sharpe',        format='%.3f'),
                'Mom 1M (%)':            st.column_config.NumberColumn('Mom 1M',        format='%.1f%%'),
                'Mom 3M (%)':            st.column_config.NumberColumn('Mom 3M',        format='%.1f%%'),
                'Forward P/E':           st.column_config.NumberColumn('Fwd P/E',       format='%.1f×'),
                'PEG':                   st.column_config.NumberColumn('PEG',            format='%.2f'),
                'EV/EBITDA':             st.column_config.NumberColumn('EV/EBITDA',     format='%.1f×'),
                'Gross Margin':          st.column_config.NumberColumn('Gross Margin',  format='%.1f%%'),
                'Rev Growth':            st.column_config.NumberColumn('Rev Growth',    format='%.1f%%'),
                'Short Float %':         st.column_config.NumberColumn('Short Float',   format='%.1f%%'),
                'Market Cap (B)':        st.column_config.NumberColumn('Mkt Cap',       format='$%.1fB'),
            },
            hide_index=True, use_container_width=True, height=400,
        )

with tab_pairs:
    if not pairs_df.empty:
        st.markdown(f"**{n_flagged_pairs} actionable pair(s)** above {pair_z_thresh}σ threshold.")
        st.dataframe(
            pairs_df,
            column_config={
                'Pair':      st.column_config.TextColumn('Pair'),
                'Sector':    st.column_config.TextColumn('Sector'),
                'Spread Z':  st.column_config.NumberColumn('Spread Z', format='%.2fσ'),
                'Signal':    st.column_config.TextColumn('Direction'),
                'Actionable':st.column_config.CheckboxColumn('Flag'),
            },
            hide_index=True, use_container_width=True, height=400,
        )

with tab_events:
    if not climate_ev.empty:
        ev_filter = climate_ev[
            (climate_ev['Hours Ago'] <= hours_back) &
            (climate_ev['SevScore'] >= sev_order.get(min_sev, 1))
        ].sort_values('Hours Ago')
        st.markdown(f"**{len(ev_filter)} events** — last {hours_back}h, severity ≥ {min_sev}.")
        st.dataframe(
            ev_filter[['Hours Ago','Type','Severity','SevScore','RecencyWeight','Event']],
            column_config={
                'Hours Ago':     st.column_config.NumberColumn('Hours Ago',   format='%.1f h'),
                'Type':          st.column_config.TextColumn('Hazard',         width='small'),
                'Severity':      st.column_config.TextColumn('Severity',       width='small'),
                'SevScore':      st.column_config.NumberColumn('Sev Score',    format='%d'),
                'RecencyWeight': st.column_config.NumberColumn('Recency Wt',   format='%.3f'),
                'Event':         st.column_config.TextColumn('Event',          width='large'),
            },
            hide_index=True, use_container_width=True, height=400,
        )

with tab_raw:
    selected_t = st.selectbox("Select Ticker", ALL_TICKERS)
    if selected_t in prices.columns:
        df_price_show = prices[[selected_t, BENCH_TICKER]].dropna().tail(252)
        # Normalise to 100
        df_norm = (df_price_show / df_price_show.iloc[0] * 100)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=df_norm.index, y=df_norm[selected_t],
                                       name=selected_t, line=dict(color='#00ccff', width=2)))
        fig_price.add_trace(go.Scatter(x=df_norm.index, y=df_norm[BENCH_TICKER],
                                       name=BENCH_TICKER, line=dict(color='#ffffff', width=1.5, dash='dash')))
        fig_price.update_layout(
            template='plotly_dark',
            title=f'{selected_t} vs SPY — Indexed to 100 (1Y)',
            yaxis_title='Indexed Price (100 = start)',
            margin=dict(l=10, r=10, t=45, b=10), height=380,
        )
        st.plotly_chart(fig_price, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────
# AI SYNTHESIS ENGINE
# ─────────────────────────────────────────────
st.subheader("🧠 Quantitative AI Synthesis")
col_ai_l, col_ai_r = st.columns([1, 2])

with col_ai_l:
    st.markdown(f"**Model:** `{model_id}`  \n**Temp:** `{temperature}`  \n**Endpoint:** `localhost:1234`")
    mode = st.radio("Report Mode", ["Morning Alpha Brief", "Pair Trade Analysis", "Climate Risk Matrix"], index=0)
    run_ai = st.button("⚡ Run Synthesis", type="primary", use_container_width=True)

with col_ai_r:
    if run_ai:
        with st.spinner(f"Querying {model_id}…"):
            try:
                z_str  = zscore_df.to_string(index=False) if not zscore_df.empty else 'N/A'
                p_str  = pairs_df[pairs_df['Actionable']].to_string(index=False) if not pairs_df.empty else 'None'
                cl_str = climate_score.to_string(index=False)
                ev_str = (climate_ev[climate_ev['Hours Ago'] <= 48][['Hours Ago','Type','Severity','Event']]
                          .head(15).to_string(index=False)) if not climate_ev.empty else 'N/A'
                mom_str = momentum.to_string(index=False)
                beta_str = beta_df.to_string(index=False) if not beta_df.empty else 'N/A'

                vix_str = f"{vix_now:.1f}" if vix_now is not None else "N/A"
                spy_str = f"{spy_1d:+.2f}" if spy_1d is not None else "N/A"

                if mode == "Morning Alpha Brief":
                    prompt = f"""You are a senior quant analyst at Jane Street. Today is {last_sync.strftime('%A %B %d, %Y')}.

Produce a rigorous morning alpha brief. Use statistical language. No disclaimers.

SECTIONS:
1. REGIME CONTEXT: VIX={vix_str}, SPY 1D={spy_str}%. Characterize the risk environment in one sentence.

2. TOP Z-SCORE SIGNALS: From the z-score data below, identify the 2 most statistically significant signals (|z| > 1.5). State the ticker, z-score, and a one-sentence hypothesis for the driver.

3. PAIR TRADE: From the actionable pair divergence below, recommend the single best mean-reversion pair trade. Include entry rationale and what would invalidate the trade.

4. CLIMATE ALPHA: Using the climate impact scores and recent events, identify which sector has the strongest quantitative tailwind from physical risk events, and which ticker within it has the best technical setup.

5. WATCH LIST: 2 tickers. One sentence each. Be specific — cite z-score, momentum, or beta.

--- Z-SCORE SCREEN ---
{z_str}

--- MOMENTUM FACTORS ---
{mom_str}

--- PAIR DIVERGENCE ---
{p_str}

--- CLIMATE IMPACT SCORES ---
{cl_str}

--- RECENT EVENTS (48h) ---
{ev_str}
"""
                elif mode == "Pair Trade Analysis":
                    prompt = f"""You are a stat-arb specialist at a quant fund. Analyze the following pair divergences and recommend the top 2 mean-reversion trades.

For each trade provide: Long/Short legs, Entry rationale (cite the spread z-score), Risk (what breaks the thesis), Target (what does mean-reversion look like).

--- PAIR DIVERGENCE DATA ---
{p_str}

--- MOMENTUM CONTEXT ---
{mom_str}

--- BETA DATA ---
{beta_str}
"""
                else:  # Climate Risk Matrix
                    prompt = f"""You are a macro risk analyst. Using the quantitative climate impact scores and live GDACS events, produce a physical risk attribution memo.

For each sector with |Climate Impact Score| > 0.3, state:
- The specific event types driving the score
- The primary ticker exposure and direction (long/short implication)
- A confidence level (Low/Medium/High) for the signal based on event recency

Then rank the top 3 actionable sector positions implied by current physical risk.

--- CLIMATE IMPACT SCORES ---
{cl_str}

--- RECENT EVENTS ---
{ev_str}
"""

                resp = requests.post(
                    f"{llm_endpoint}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"model": model_id,
                                     "messages": [{"role": "user", "content": prompt}],
                                     "temperature": temperature}),
                    timeout=90,
                )
                if resp.status_code == 200:
                    out = resp.json().get('choices',[{}])[0].get('message',{}).get('content','')
                    st.success(f"Synthesis complete — {last_sync.strftime('%H:%M UTC')}")
                    st.markdown(out)
                else:
                    st.error(f"LLM Error ({resp.status_code}): {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Timeout — model may be cold-loading, retry in 30s.")
            except Exception as e:
                st.error(f"Engine error: {e}")
    else:
        st.info("Select a report mode and click **⚡ Run Synthesis** to generate analysis.")
