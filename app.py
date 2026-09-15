import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import xml.etree.ElementTree as ET
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

# LLM endpoint — override with LLM_BASE_URL env var for cloud deployment
# e.g. export LLM_BASE_URL="https://your-tunnel.trycloudflare.com"
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Climate Terminal",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Authentication gate ─────────────────────────────
from auth import require_login
require_login()
# ────────────────────────────────────────────────────

st.markdown("""
<style>
    [data-testid="stMetricValue"]  { font-size: 1.35rem !important; font-weight: 700; }
    [data-testid="stMetricDelta"]  { font-size: 0.80rem !important; }
    [data-testid="stMetricLabel"]  { font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.04em; }
    .stTabs [data-baseweb="tab"]   { font-size: 0.92rem; font-weight: 600; }
    hr { margin: 0.4rem 0 0.9rem 0 !important; }
    /* Morning brief banner */
    .brief-box {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a1a2e 100%);
        border-left: 4px solid #00ccff;
        border-radius: 6px;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
    }
    /* Mover card */
    .mover-up   { color: #00cc88; font-weight: 700; }
    .mover-down { color: #ff4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
WATCHLIST = {
    'Pharma & Biotech':            ['LLY', 'NVO', 'VKTX'],
    'AI Energy Infrastructure':    ['GEV', 'VST', 'CEG', 'BE'],
    'Power Grid & Hardware':       ['PWR', 'ETN', 'POWL', 'MTZ'],
    'Optical & Data Interconnect': ['COHR', 'GLW', 'LITE'],
    'Cybersecurity':               ['CRWD', 'PANW', 'ZS', 'S'],
    'Climate Resilience':          ['XYL', 'FSLR', 'VRT'],
}

COLOR_MAP = {
    "Earthquake": "#FF9900",
    "Flood":      "#3399FF",
    "Cyclone":    "#00FFCC",
    "Volcano":    "#FF3333",
    "Drought":    "#FFD700",
    "Tsunami":    "#CC66FF",
    "Other":      "#AAAAAA",
}

IMPACT_MAP = {
    "Earthquake": "⚠️ Infrastructure Risk — Tailwind for PWR, ETN (grid rebuild demand)",
    "Flood":      "🌊 Supply Chain Stress — Bullish XYL (water mgmt), pressure on GLW logistics",
    "Cyclone":    "🌀 Grid Outage Risk — Tailwind for PWR; monitor FSLR (solar panel damage)",
    "Volcano":    "🌋 Airspace/Logistics Disruption — Watch fiber/tech supply chains (GLW, LITE)",
    "Drought":    "🏜️ Ag & Water Stress — Bullish XYL; elevated utility infrastructure demand",
    "Tsunami":    "🌊 Coastal Infrastructure Collapse — Broad infra demand; ETN, PWR upside",
    "Other":      "📡 Monitor for localized disruption signals",
}


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_equity_data(ticker_symbol: str, sector: str) -> dict:
    data = {
        'Ticker':          ticker_symbol,
        'Sector':          sector,
        'Current Price':   None,
        'Daily Chg (%)':   None,
        'Volume':          None,
        'Avg Volume':      None,
        'Vol Spike':       None,   # volume vs avg ratio
        'vs 50D MA':       None,
        '50-Day MA':       None,
        '52W High':        None,
        '52W Low':         None,
        'vs 52W High (%)': None,
        'Forward P/E':     None,
        'Market Cap (B)':  None,
        '30-Day Trend':    None,
    }
    try:
        ticker = yf.Ticker(ticker_symbol)
        info   = ticker.info

        hist = ticker.history(period="1mo")
        if not hist.empty:
            closes = hist['Close'].tolist()
            data['30-Day Trend'] = closes
            if len(closes) >= 2:
                data['Daily Chg (%)'] = ((closes[-1] - closes[-2]) / closes[-2]) * 100
            if 'Volume' in hist.columns and len(hist) >= 2:
                data['Volume']    = int(hist['Volume'].iloc[-1])
                data['Avg Volume'] = int(hist['Volume'].mean())
                avg = hist['Volume'].mean()
                if avg > 0:
                    data['Vol Spike'] = hist['Volume'].iloc[-1] / avg

        price = info.get('currentPrice', info.get('regularMarketPrice'))
        ma50  = info.get('fiftyDayAverage')
        hi52  = info.get('fiftyTwoWeekHigh')
        lo52  = info.get('fiftyTwoWeekLow')

        data['Current Price']   = price
        data['50-Day MA']       = ma50
        data['52W High']        = hi52
        data['52W Low']         = lo52
        data['Forward P/E']     = info.get('forwardPE')
        data['Market Cap (B)']  = info.get('marketCap', 0) / 1e9 if info.get('marketCap') else None

        if price and ma50 and ma50 > 0:
            data['vs 50D MA'] = ((price - ma50) / ma50) * 100
        if price and hi52 and hi52 > 0:
            data['vs 52W High (%)'] = ((price - hi52) / hi52) * 100

    except Exception:
        pass
    return data


@st.cache_data(ttl=300, show_spinner=False)
def build_pipeline() -> pd.DataFrame:
    rows = []
    for sector, tickers in WATCHLIST.items():
        for sym in tickers:
            rows.append(fetch_equity_data(sym, sector))
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather_events() -> pd.DataFrame:
    url = "https://gdacs.org/xml/rss.xml"
    events = []
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns   = {'georss': 'http://www.georss.org/georss'}

        for item in root.findall('./channel/item')[:80]:
            title = getattr(item.find('title'),       'text', 'Unknown') or 'Unknown'
            desc  = getattr(item.find('description'), 'text', '')        or ''
            pub   = getattr(item.find('pubDate'),     'text', '')        or ''

            lat, lon = None, None
            pt = item.find('georss:point', ns)
            if pt is not None and pt.text:
                try:
                    lat, lon = map(float, pt.text.split())
                except ValueError:
                    pass

            text = (title + ' ' + desc).lower()
            if 'earthquake' in text:                                  etype = 'Earthquake'
            elif 'flood' in text:                                     etype = 'Flood'
            elif any(w in text for w in ('cyclone','hurricane','typhoon')): etype = 'Cyclone'
            elif 'volcan' in text:                                    etype = 'Volcano'
            elif 'drought' in text:                                   etype = 'Drought'
            elif 'tsunami' in text:                                   etype = 'Tsunami'
            else:                                                     etype = 'Other'

            # GDACS severity from title
            severity = 'Green'
            if 'orange' in text: severity = 'Orange'
            elif 'red'   in text: severity = 'Red'

            try:
                dt = parsedate_to_datetime(pub)
            except Exception:
                dt = datetime.now(timezone.utc)

            clean = title.split(')')[0] + ')' if ')' in title else title[:65]
            events.append({
                'Event':         clean,
                'ParsedDate':    dt,
                'Date':          pub,
                'Type':          etype,
                'Severity':      severity,
                'Quant Signal':  IMPACT_MAP.get(etype, IMPACT_MAP['Other']),
                'Details':       (desc[:230] + '…') if len(desc) > 230 else desc,
                'lat':           lat,
                'lon':           lon,
            })

        df = pd.DataFrame(events)
        if not df.empty:
            max_date = df['ParsedDate'].max()
            age_s    = (max_date - df['ParsedDate']).dt.total_seconds()
            max_age  = max(age_s.max(), 1)
            df['Opacity']  = (1.0 - 0.75 * (age_s / max_age)).clip(0.2, 1.0)
            df['Hours Ago'] = (age_s / 3600).round(1)
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading data streams…"):
    df_equity = build_pipeline()
    df_events = fetch_weather_events()

last_refresh = datetime.now(timezone.utc)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🛰️ Terminal Controls")
    st.caption(f"Synced: {last_refresh.strftime('%b %d, %H:%M UTC')}")

    st.divider()
    if st.button("🔄 Force Refresh All Streams", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("🧠 AI Configuration")
    model_id    = st.text_input("Local Model ID", value="qwen2.5-coder-14b-instruct-mlx",
                                help="Exact ID from http://localhost:1234/v1/models")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    st.divider()
    st.subheader("🗺️ Disaster Map Filters")
    all_types = sorted(df_events['Type'].unique().tolist()) if not df_events.empty else []
    selected_types = st.multiselect("Hazard Types", options=all_types, default=all_types)
    hours_filter = st.slider("Show events from last N hours", 12, 168, 72, 12)

    st.divider()
    st.subheader("📊 Disaster Breakdown")
    if not df_events.empty:
        for etype, cnt in df_events['Type'].value_counts().items():
            recent = len(df_events[(df_events['Type'] == etype) & (df_events['Hours Ago'] <= 24)])
            st.metric(etype, cnt, delta=f"{recent} in last 24h", delta_color="off")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🛰️ Global Climate & Quant Equity Terminal")
st.caption(f"Data as of {last_refresh.strftime('%A, %B %d %Y — %H:%M UTC')}  ·  Prices refresh every 5 min  ·  Disasters refresh every 10 min")
st.divider()


# ─────────────────────────────────────────────
# MORNING BRIEF BANNER
# ─────────────────────────────────────────────
if not df_equity.empty and not df_events.empty:
    top_mover_up   = df_equity.dropna(subset=['Daily Chg (%)']).nlargest(1, 'Daily Chg (%)')
    top_mover_down = df_equity.dropna(subset=['Daily Chg (%)']).nsmallest(1, 'Daily Chg (%)')
    recent_events  = df_events[df_events['Hours Ago'] <= 24] if 'Hours Ago' in df_events else df_events
    vol_spikes     = df_equity[df_equity['Vol Spike'].fillna(0) > 1.5].sort_values('Vol Spike', ascending=False)

    up_row   = top_mover_up.iloc[0]   if not top_mover_up.empty   else None
    down_row = top_mover_down.iloc[0] if not top_mover_down.empty else None

    brief_parts = []
    if up_row is not None:
        brief_parts.append(f"📈 **{up_row['Ticker']}** leading up **{up_row['Daily Chg (%)']:+.2f}%**")
    if down_row is not None:
        brief_parts.append(f"📉 **{down_row['Ticker']}** lagging **{down_row['Daily Chg (%)']:+.2f}%**")
    if not vol_spikes.empty:
        vs = vol_spikes.iloc[0]
        brief_parts.append(f"🔊 **{vs['Ticker']}** volume spike **{vs['Vol Spike']:.1f}×** avg")
    if not recent_events.empty:
        high_sev = recent_events[recent_events['Severity'].isin(['Orange', 'Red'])]
        if not high_sev.empty:
            ev = high_sev.iloc[0]
            brief_parts.append(f"🚨 High-severity **{ev['Type']}** — {ev['Event'][:55]}…")
        else:
            brief_parts.append(f"🌍 **{len(recent_events)}** new disaster events in last 24h")

    st.markdown(
        f'<div class="brief-box"><b>☀️ Morning Brief</b> &nbsp;·&nbsp; '
        + "  &nbsp;|&nbsp;  ".join(brief_parts) +
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
total_cap   = df_equity['Market Cap (B)'].sum()              if 'Market Cap (B)'  in df_equity.columns else 0
avg_pe      = df_equity['Forward P/E'].dropna().mean()        if 'Forward P/E'     in df_equity.columns else 0
avg_chg     = df_equity['Daily Chg (%)'].dropna().mean()      if 'Daily Chg (%)'   in df_equity.columns else 0
n_events    = len(df_events)
n_recent    = len(df_events[df_events['Hours Ago'] <= 24])    if 'Hours Ago'       in df_events.columns else n_events
n_high_sev  = len(df_events[df_events['Severity'].isin(['Orange','Red'])]) if not df_events.empty else 0
above_ma    = int(df_equity['vs 50D MA'].dropna().gt(0).sum()) if 'vs 50D MA'      in df_equity.columns else 0
below_ma    = int(df_equity['vs 50D MA'].dropna().le(0).sum())

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Tracked Equities",    len(df_equity))
c2.metric("Cohort Market Cap",   f"${total_cap:,.0f}B")
c3.metric("Avg Forward P/E",     f"{avg_pe:.1f}×")
c4.metric("Portfolio Avg 1D",    f"{avg_chg:+.2f}%",  delta=f"{avg_chg:+.2f}%")
c5.metric("Above 50D MA",        f"{above_ma} / {above_ma+below_ma}")
c6.metric("Active Disasters",    n_events, delta=f"{n_recent} last 24h  ·  {n_high_sev} high-sev", delta_color="inverse")

st.divider()


# ─────────────────────────────────────────────
# TODAY'S MOVERS
# ─────────────────────────────────────────────
st.subheader("⚡ Today's Movers & Volume Anomalies")
col_mv1, col_mv2 = st.columns(2)

with col_mv1:
    if 'Daily Chg (%)' in df_equity.columns:
        df_sorted_chg = df_equity.dropna(subset=['Daily Chg (%)']).sort_values('Daily Chg (%)')
        colors_chg    = ['#ff4444' if v < 0 else '#00cc88' for v in df_sorted_chg['Daily Chg (%)']]
        fig_movers = go.Figure(go.Bar(
            x=df_sorted_chg['Daily Chg (%)'],
            y=df_sorted_chg['Ticker'],
            orientation='h',
            marker_color=colors_chg,
            text=[f"{v:+.2f}%" for v in df_sorted_chg['Daily Chg (%)']],
            textposition='outside',
            customdata=df_sorted_chg[['Sector', 'Current Price']].values,
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Price: $%{customdata[1]:.2f}<br>Change: %{x:+.2f}%<extra></extra>"
        ))
        fig_movers.update_layout(
            template='plotly_dark', title='1-Day Price Change (%)',
            xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1.5),
            margin=dict(l=10, r=50, t=40, b=10), height=370
        )
        st.plotly_chart(fig_movers, use_container_width=True)

with col_mv2:
    if 'Vol Spike' in df_equity.columns:
        df_vol = df_equity.dropna(subset=['Vol Spike']).sort_values('Vol Spike')
        colors_vol = ['#ffaa00' if v > 2.0 else '#888888' for v in df_vol['Vol Spike']]
        fig_vol = go.Figure(go.Bar(
            x=df_vol['Vol Spike'],
            y=df_vol['Ticker'],
            orientation='h',
            marker_color=colors_vol,
            text=[f"{v:.1f}×" for v in df_vol['Vol Spike']],
            textposition='outside',
            hovertemplate="<b>%{y}</b><br>Volume: %{x:.2f}× avg<extra></extra>"
        ))
        fig_vol.add_vline(x=1.0, line_dash="dash", line_color="white", opacity=0.4,
                          annotation_text="avg", annotation_position="top")
        fig_vol.add_vline(x=2.0, line_dash="dot", line_color="#ffaa00", opacity=0.6,
                          annotation_text="2× spike", annotation_position="top")
        fig_vol.update_layout(
            template='plotly_dark', title='Volume vs 30-Day Average (anomaly = gold)',
            margin=dict(l=10, r=50, t=40, b=10), height=370
        )
        st.plotly_chart(fig_vol, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────
# ROW 2: VALUATION SCATTER + MAP
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("📊 Valuation Matrix")
    df_valid = df_equity.dropna(subset=['Forward P/E', 'Market Cap (B)', 'vs 50D MA'])
    if not df_valid.empty:
        df_valid = df_valid.copy()
        df_valid['MA Signal'] = df_valid['vs 50D MA'].apply(
            lambda x: "Above 50D MA 🟢" if x >= 0 else "Below 50D MA 🔴"
        )
        fig_sc = px.scatter(
            df_valid, x="Forward P/E", y="Market Cap (B)",
            color="Sector", symbol="MA Signal",
            hover_name="Ticker",
            hover_data={"Current Price": ":.2f", "vs 50D MA": ":.2f",
                        "Daily Chg (%)": ":.2f", "Market Cap (B)": ":.1f", "MA Signal": False},
            log_y=True, template="plotly_dark",
            title="Forward P/E vs Market Cap — shape = 50D MA position",
        )
        fig_sc.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
        fig_sc.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=400)
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.warning("Insufficient data.")

with col_right:
    st.subheader("🌍 Live GDACS Disaster Map")
    if not df_events.empty:
        df_map = df_events[
            (df_events['Type'].isin(selected_types)) &
            (df_events['Hours Ago'] <= hours_filter)
        ].dropna(subset=['lat', 'lon'])

        if not df_map.empty:
            fig_map = px.scatter_geo(
                df_map, lat="lat", lon="lon", color="Type",
                hover_name="Event",
                hover_data={"lat": False, "lon": False, "Type": True,
                            "Hours Ago": ":.1f", "Severity": True,
                            "Quant Signal": True, "Details": True},
                color_discrete_map=COLOR_MAP, template="plotly_dark",
                title=f"Active Events — last {hours_filter}h (brighter = more recent)",
            )
            for trace in fig_map.data:
                mask = df_map['Type'] == trace.name
                if mask.any():
                    trace.marker.opacity = df_map.loc[mask, 'Opacity'].tolist()
                trace.marker.size = 11
                trace.marker.line = dict(width=1, color='rgba(255,255,255,0.35)')
            fig_map.update_geos(
                showcoastlines=True, coastlinecolor="#555",
                showland=True, landcolor="#1a1a2e",
                showocean=True, oceancolor="#0d1b2a",
                showframe=False, projection_type="natural earth", resolution=110,
            )
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=50, b=0), height=400,
                legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info(f"No events in the last {hours_filter}h matching the selected filters.")
    else:
        st.warning("Could not fetch GDACS data.")

st.divider()


# ─────────────────────────────────────────────
# ROW 3: 52W RANGE + 50D MA SIGNAL
# ─────────────────────────────────────────────
col_52, col_50 = st.columns(2)

with col_52:
    st.subheader("📐 52-Week Range Position")
    df_52 = df_equity.dropna(subset=['52W High', '52W Low', 'Current Price'])
    if not df_52.empty:
        fig_52 = go.Figure()
        for _, row in df_52.iterrows():
            lo, hi, px_ = row['52W Low'], row['52W High'], row['Current Price']
            pct = ((px_ - lo) / (hi - lo) * 100) if (hi - lo) > 0 else 50
            color = '#00cc88' if pct > 70 else ('#ffaa00' if pct > 35 else '#ff4444')
            fig_52.add_trace(go.Bar(
                x=[pct], y=[row['Ticker']],
                orientation='h', marker_color=color,
                text=f"${px_:.2f}  ({pct:.0f}%)",
                textposition='inside',
                hovertemplate=f"<b>{row['Ticker']}</b><br>Price: ${px_:.2f}<br>52W Low: ${lo:.2f}<br>52W High: ${hi:.2f}<br>Range Position: {pct:.1f}%<extra></extra>",
                showlegend=False,
                name=row['Ticker']
            ))
        fig_52.update_layout(
            template='plotly_dark',
            title='Position in 52-Week Range (0% = 52W Low, 100% = 52W High)',
            xaxis=dict(title='% of 52-Week Range', range=[0, 110]),
            margin=dict(l=10, r=10, t=50, b=10), height=370, barmode='overlay'
        )
        fig_52.add_vline(x=100, line_dash="dash", line_color="#ffffff", opacity=0.3)
        st.plotly_chart(fig_52, use_container_width=True)

with col_50:
    st.subheader("📡 50-Day MA Signal")
    df_ma = df_equity.dropna(subset=['vs 50D MA']).sort_values('vs 50D MA')
    if not df_ma.empty:
        colors_ma = ['#ff4444' if v < 0 else '#00cc88' for v in df_ma['vs 50D MA']]
        fig_ma = go.Figure(go.Bar(
            x=df_ma['vs 50D MA'], y=df_ma['Ticker'],
            orientation='h', marker_color=colors_ma,
            text=[f"{v:+.1f}%" for v in df_ma['vs 50D MA']],
            textposition='outside',
            customdata=df_ma[['Current Price', '50-Day MA']].values,
            hovertemplate="<b>%{y}</b><br>Price: $%{customdata[0]:.2f}<br>50D MA: $%{customdata[1]:.2f}<br>Delta: %{x:+.2f}%<extra></extra>",
        ))
        fig_ma.update_layout(
            template='plotly_dark', title='% vs 50-Day Moving Average',
            xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1.5),
            margin=dict(l=10, r=50, t=40, b=10), height=370,
        )
        st.plotly_chart(fig_ma, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────
# ROW 4: SECTOR HEATMAP + MARKET CAP BAR
# ─────────────────────────────────────────────
col_heat, col_mcap = st.columns(2)

with col_heat:
    st.subheader("🌡️ Sector Performance Heatmap")
    if 'Daily Chg (%)' in df_equity.columns and 'Sector' in df_equity.columns:
        sector_perf = df_equity.groupby('Sector')['Daily Chg (%)'].mean().reset_index()
        sector_perf.columns = ['Sector', 'Avg 1D Chg (%)']
        sector_perf = sector_perf.sort_values('Avg 1D Chg (%)', ascending=False)
        colors_heat = ['#00cc88' if v >= 0 else '#ff4444' for v in sector_perf['Avg 1D Chg (%)']]
        fig_heat = go.Figure(go.Bar(
            x=sector_perf['Avg 1D Chg (%)'],
            y=sector_perf['Sector'],
            orientation='h',
            marker_color=colors_heat,
            text=[f"{v:+.2f}%" for v in sector_perf['Avg 1D Chg (%)']],
            textposition='outside',
        ))
        fig_heat.update_layout(
            template='plotly_dark', title='Average 1D Change by Thematic Cohort',
            xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1.5),
            margin=dict(l=10, r=60, t=40, b=10), height=280,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

with col_mcap:
    st.subheader("💰 Market Cap by Ticker")
    df_bar = df_equity.dropna(subset=['Market Cap (B)']).sort_values('Market Cap (B)', ascending=True)
    if not df_bar.empty:
        fig_bar = px.bar(
            df_bar, x='Market Cap (B)', y='Ticker', color='Sector',
            orientation='h', template='plotly_dark',
            title='Market Cap ($B)', text_auto='.0f',
        )
        fig_bar.update_layout(
            margin=dict(l=10, r=10, t=40, b=10), height=280, showlegend=False
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────
# ROW 5: DATA TABLES
# ─────────────────────────────────────────────
st.subheader("🗃️ Data Ledger")
tab_equity, tab_events, tab_recent = st.tabs(["📋 Equity Ledger", "🌩️ All Disasters", "🚨 Last 24h Events"])

with tab_equity:
    if not df_equity.empty:
        disp_cols = ['Ticker','Sector','Current Price','Daily Chg (%)','Vol Spike',
                     'vs 50D MA','vs 52W High (%)','30-Day Trend','Forward P/E','Market Cap (B)']
        df_show = df_equity[[c for c in disp_cols if c in df_equity.columns]].copy()
        st.dataframe(
            df_show,
            column_config={
                "Ticker":           st.column_config.TextColumn("Ticker",        width="small"),
                "Sector":           st.column_config.TextColumn("Cohort"),
                "Current Price":    st.column_config.NumberColumn("Price",        format="$%.2f"),
                "Daily Chg (%)":    st.column_config.NumberColumn("1D Chg",       format="%.2f%%"),
                "Vol Spike":        st.column_config.NumberColumn("Vol ×Avg",     format="%.2f×"),
                "vs 50D MA":        st.column_config.NumberColumn("vs 50D MA",    format="%.2f%%"),
                "vs 52W High (%)":  st.column_config.NumberColumn("vs 52W High",  format="%.1f%%"),
                "30-Day Trend":     st.column_config.LineChartColumn("30D Trend"),
                "Forward P/E":      st.column_config.NumberColumn("Fwd P/E",      format="%.1f×"),
                "Market Cap (B)":   st.column_config.NumberColumn("Mkt Cap",      format="$%.1fB"),
            },
            hide_index=True, use_container_width=True, height=370,
        )

with tab_events:
    if not df_events.empty:
        df_ev = df_events[['ParsedDate','Hours Ago','Type','Severity','Event','Quant Signal','Details']].copy()
        st.dataframe(
            df_ev,
            column_config={
                "ParsedDate":   st.column_config.DatetimeColumn("Date (UTC)",       format="MMM D, HH:mm"),
                "Hours Ago":    st.column_config.NumberColumn("Hours Ago",          format="%.1f h"),
                "Type":         st.column_config.TextColumn("Hazard",               width="small"),
                "Severity":     st.column_config.TextColumn("Severity",             width="small"),
                "Event":        st.column_config.TextColumn("Event Overview",       width="medium"),
                "Quant Signal": st.column_config.TextColumn("Investment Signal",    width="large"),
                "Details":      st.column_config.TextColumn("On-Ground Details",    width="large"),
            },
            hide_index=True, use_container_width=True, height=370,
        )

with tab_recent:
    if not df_events.empty and 'Hours Ago' in df_events.columns:
        df_24 = df_events[df_events['Hours Ago'] <= 24].sort_values('ParsedDate', ascending=False)
        if not df_24.empty:
            df_24_show = df_24[['ParsedDate','Hours Ago','Type','Severity','Event','Quant Signal']].copy()
            st.dataframe(
                df_24_show,
                column_config={
                    "ParsedDate":   st.column_config.DatetimeColumn("Date (UTC)",    format="MMM D, HH:mm"),
                    "Hours Ago":    st.column_config.NumberColumn("Hours Ago",       format="%.1f h"),
                    "Type":         st.column_config.TextColumn("Hazard",            width="small"),
                    "Severity":     st.column_config.TextColumn("Severity",          width="small"),
                    "Event":        st.column_config.TextColumn("Event Overview",    width="large"),
                    "Quant Signal": st.column_config.TextColumn("Investment Signal", width="large"),
                },
                hide_index=True, use_container_width=True, height=370,
            )
        else:
            st.info("No new high-severity events in the last 24 hours.")

st.divider()


# ─────────────────────────────────────────────
# ROW 6: AI ALPHA ENGINE
# ─────────────────────────────────────────────
st.subheader("🧠 AI Alpha Engine")
col_ai_l, col_ai_r = st.columns([1, 2])

with col_ai_l:
    st.markdown(f"**Model:** `{model_id}`")
    st.caption(f"Temp: `{temperature}` · Endpoint: `localhost:1234`")
    run_ai = st.button("⚡ Generate Morning Alpha Report", type="primary", use_container_width=True)

with col_ai_r:
    if run_ai:
        with st.spinner(f"Querying {model_id}…"):
            try:
                eq_cols   = ['Ticker','Sector','Current Price','Daily Chg (%)','Vol Spike','vs 50D MA','Forward P/E','Market Cap (B)']
                eq_str    = df_equity[[c for c in eq_cols if c in df_equity.columns]].to_string(index=False)
                recent_ev = df_events[df_events['Hours Ago'] <= 48][['Type','Severity','Event','Quant Signal']].head(20) if not df_events.empty else pd.DataFrame()
                ev_str    = recent_ev.to_string(index=False) if not recent_ev.empty else "No recent events."

                prompt = f"""You are a senior analyst at a global macro hedge fund preparing the morning briefing for the portfolio manager. Today is {last_refresh.strftime('%A, %B %d, %Y')}.

Given the live thematic equity data and recent global climate/disaster events, write a sharp, concise morning alpha memo with EXACTLY these 4 sections:

1. 🔥 TOP RISK (2-3 sentences): The single most important risk from today's disaster feed. Name the specific ticker(s) most exposed.

2. 💹 TOP OPPORTUNITY (2-3 sentences): The strongest alpha opportunity from the intersection of the events and our portfolio. Be specific about tickers and the mechanism.

3. 📊 TECHNICAL FLAGS (bullet list): Note any tickers showing unusual volume spikes or notable 50D MA crossings today.

4. 📌 WATCH LIST (2 tickers, one sentence each): The 2 names to watch most closely today.

Rules: Use ticker symbols. Be specific. No disclaimers. No generic statements.

--- EQUITY DATA ---
{eq_str}

--- RECENT CLIMATE/DISASTER EVENTS (last 48h) ---
{ev_str}
"""
                resp = requests.post(
                    f"{LLM_BASE_URL}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"model": model_id, "messages": [{"role": "user", "content": prompt}],
                                     "temperature": temperature}),
                    timeout=90,
                )
                if resp.status_code == 200:
                    content = resp.json().get('choices',[{}])[0].get('message',{}).get('content','No output.')
                    st.success(f"Analysis complete — {last_refresh.strftime('%H:%M UTC')}")
                    st.markdown(content)
                else:
                    st.error(f"LLM Error ({resp.status_code}): {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Model may still be loading — try again in 30 seconds.")
            except Exception as e:
                st.error(f"Failed to reach local LLM: {e}")
    else:
        st.info("Click **⚡ Generate Morning Alpha Report** to run cross-analysis against today's live data.")
