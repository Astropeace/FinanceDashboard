import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import xml.etree.ElementTree as ET
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from shared import (
    LLM_BASE_URL, WATCHLIST, ALL_TICKERS, TICKER_TO_SECTOR,
    COLOR_MAP, IMPACT_MAP, BENCH_TICKER, render_nav,
)
from auth import require_login

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Climate Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_login()

st.markdown("""
<style>
    [data-testid="stMetricValue"]  { font-size:1.35rem !important; font-weight:700; }
    [data-testid="stMetricLabel"]  { font-size:0.75rem !important; text-transform:uppercase; letter-spacing:.05em; }
    .stTabs [data-baseweb="tab"]   { font-weight:700; font-size:.9rem; }
    hr { margin:.3rem 0 .8rem 0 !important; }
    .brief-box {
        background:linear-gradient(135deg,#0d1b2a 0%,#1a1a2e 100%);
        border-left:4px solid #00ccff; border-radius:6px;
        padding:.9rem 1.3rem; margin-bottom:1rem;
    }
    /* Conviction table */
    .conv-table { width:100%; border-collapse:collapse; font-size:.82rem; }
    .conv-table th { background:#1f2937; color:#9ca3af; text-transform:uppercase;
                     font-size:.7rem; letter-spacing:.06em; padding:.5rem .8rem; text-align:left; }
    .conv-table td { padding:.45rem .8rem; border-bottom:1px solid #1f2937; vertical-align:top; }
    .conv-table tr:hover td { background:#111827; }
    .tag { display:inline-block; padding:.15rem .5rem; border-radius:4px;
           font-size:.7rem; font-weight:700; margin-right:.3rem; }
    .tag-buy  { background:#064e3b; color:#6ee7b7; }
    .tag-watch{ background:#1e3a5f; color:#93c5fd; }
    .tag-risk { background:#4c1d1d; color:#fca5a5; }
</style>
""", unsafe_allow_html=True)

render_nav("climate")

# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_equity_data(sym: str, sector: str) -> dict:
    data = dict(Ticker=sym, Sector=sector, **{k: None for k in [
        'Current Price','Daily Chg (%)','Vol Spike','vs 50D MA',
        '50-Day MA','52W High','52W Low','vs 52W High (%)','Forward P/E',
        'Market Cap (B)','30-Day Trend',
    ]})
    try:
        t    = yf.Ticker(sym)
        info = t.info
        hist = t.history(period="1mo")
        if not hist.empty:
            closes = hist['Close'].tolist()
            data['30-Day Trend'] = closes
            if len(closes) >= 2:
                data['Daily Chg (%)'] = (closes[-1] - closes[-2]) / closes[-2] * 100
            avg = hist['Volume'].mean()
            if avg > 0:
                data['Vol Spike'] = hist['Volume'].iloc[-1] / avg
        price = info.get('currentPrice', info.get('regularMarketPrice'))
        ma50  = info.get('fiftyDayAverage')
        hi52  = info.get('fiftyTwoWeekHigh')
        data['Current Price']   = price
        data['50-Day MA']       = ma50
        data['52W High']        = hi52
        data['52W Low']         = info.get('fiftyTwoWeekLow')
        data['Forward P/E']     = info.get('forwardPE')
        data['Market Cap (B)']  = info.get('marketCap', 0) / 1e9 if info.get('marketCap') else None
        if price and ma50 and ma50 > 0:
            data['vs 50D MA'] = (price - ma50) / ma50 * 100
        if price and hi52 and hi52 > 0:
            data['vs 52W High (%)'] = (price - hi52) / hi52 * 100
    except Exception:
        pass
    return data

@st.cache_data(ttl=300, show_spinner=False)
def build_pipeline() -> pd.DataFrame:
    return pd.DataFrame([fetch_equity_data(sym, sec)
                         for sec, tickers in WATCHLIST.items() for sym in tickers])

@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather_events() -> pd.DataFrame:
    events = []
    try:
        r    = requests.get("https://gdacs.org/xml/rss.xml", timeout=10)
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
            if   'earthquake' in text:                                     etype = 'Earthquake'
            elif 'flood'      in text:                                     etype = 'Flood'
            elif any(w in text for w in ('cyclone','hurricane','typhoon')): etype = 'Cyclone'
            elif 'volcan'     in text:                                     etype = 'Volcano'
            elif 'drought'    in text:                                     etype = 'Drought'
            elif 'tsunami'    in text:                                     etype = 'Tsunami'
            else:                                                          etype = 'Other'
            sev = 'Red' if 'red' in text else ('Orange' if 'orange' in text else 'Green')
            try:    dt = parsedate_to_datetime(pub)
            except: dt = datetime.now(timezone.utc)
            clean = title.split(')')[0] + ')' if ')' in title else title[:65]
            events.append({'Event': clean, 'ParsedDate': dt, 'Date': pub,
                           'Type': etype, 'Severity': sev,
                           'Quant Signal': IMPACT_MAP.get(etype, IMPACT_MAP['Other']),
                           'Details': (desc[:230]+'…') if len(desc)>230 else desc,
                           'lat': lat, 'lon': lon})
        df = pd.DataFrame(events)
        if not df.empty:
            now   = datetime.now(timezone.utc)
            age_s = (now - df['ParsedDate']).dt.total_seconds()
            df['Hours Ago'] = (age_s / 3600).round(1)
            max_age = max(age_s.max(), 1)
            df['Opacity'] = (1.0 - 0.75 * (age_s / max_age)).clip(0.2, 1.0)
        return df
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────
# CONVICTION WATCHLIST  (Jane Street framing)
# ─────────────────────────────────────────────
CONVICTION = [
    # ticker  tag       one-line thesis
    ('LLY',  'buy',   'GLP-1 market leader. ~60% obesity share. Dual-agonist pipeline + oral formulation catalyst.'),
    ('VKTX', 'watch', 'Phase-2 challenger. High-upside asymmetry vs incumbents. Binary catalyst risk.'),
    ('CEG',  'buy',   'Pure-play nuclear. Signed hyperscaler PPAs. Trades at discount to replacement cost.'),
    ('VST',  'buy',   'Nuclear + gas portfolio. Best-in-class FCF. Direct data-center power contracts.'),
    ('GEV',  'buy',   'GE Vernova: turbines, GridOS software, nuclear services. Core "AI power stack" name.'),
    ('PWR',  'buy',   'Quanta Services: builds the grid. Backlog at record highs. Multi-year visibility.'),
    ('POWL', 'watch', 'Powell Industries: pure-play switchgear/MV equipment. Small cap, high conviction.'),
    ('COHR', 'buy',   'Coherent: #1 in 800G/1.6T transceivers. AI data-center traffic = optical demand.'),
    ('CIEN', 'watch', 'Ciena: long-haul optical backbone. Direct beneficiary of data-center interconnect boom.'),
    ('CRWD', 'buy',   'CrowdStrike: AI endpoint security leader. Platform sticky, recurring ARR. Premium justified.'),
    ('PANW', 'buy',   'Palo Alto: platform consolidator. Largest installed base. Best margin profile in cyber.'),
    ('ZS',   'watch', 'Zscaler: cloud-native zero-trust. High growth but richly valued. Watch for margin expansion.'),
    ('VRT',  'buy',   'Vertiv: liquid cooling pick-and-shovel. AI chip heat = structural demand. Strong backlog.'),
    ('XYL',  'watch', 'Xylem: water infrastructure. Climate-driven tailwind. Steady, lower-beta position.'),
    ('RXRX', 'risk',  'Recursion: pre-revenue AI drug discovery. High optionality, high burn. Speculative position only.'),
    ('TLN',  'watch', 'Talen Energy: nuclear cashflow, early in data-center PPA cycle. Less coverage = opportunity.'),
]

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading data streams…"):
    df_equity = build_pipeline()
    df_events = fetch_weather_events()

last_refresh = datetime.now(timezone.utc)

# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.caption(f"Synced: {last_refresh.strftime('%b %d · %H:%M UTC')}")
    if st.button("🔄 Force Refresh", use_container_width=True, type="primary"):
        st.cache_data.clear(); st.rerun()
    st.divider()
    st.subheader("🧠 AI Config")
    model_id    = st.text_input("Model ID", value="qwen2.5-coder-14b-instruct-mlx")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.divider()
    st.subheader("🗺️ Map Filters")
    all_types      = sorted(df_events['Type'].unique().tolist()) if not df_events.empty else []
    selected_types = st.multiselect("Hazard Types", options=all_types, default=all_types)
    hours_filter   = st.slider("Event window (h)", 12, 168, 72, 12)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🌍 Global Climate & Quant Equity Dashboard")
st.caption(f"GDACS · yFinance · {last_refresh.strftime('%A %B %d %Y — %H:%M UTC')}")
st.divider()

# ─────────────────────────────────────────────
# MORNING BRIEF
# ─────────────────────────────────────────────
if not df_equity.empty and not df_events.empty:
    up   = df_equity.dropna(subset=['Daily Chg (%)']).nlargest(1,  'Daily Chg (%)').iloc[0]
    dn   = df_equity.dropna(subset=['Daily Chg (%)']).nsmallest(1, 'Daily Chg (%)').iloc[0]
    spk  = df_equity[df_equity['Vol Spike'].fillna(0) > 2.0].sort_values('Vol Spike', ascending=False)
    ev24 = df_events[df_events['Hours Ago'] <= 24] if 'Hours Ago' in df_events else df_events
    parts = [
        f"📈 **{up['Ticker']}** leading **{up['Daily Chg (%)']:+.2f}%**",
        f"📉 **{dn['Ticker']}** lagging **{dn['Daily Chg (%)']:+.2f}%**",
    ]
    if not spk.empty:
        s = spk.iloc[0]
        parts.append(f"🔊 **{s['Ticker']}** volume **{s['Vol Spike']:.1f}×** avg")
    hi_sev = ev24[ev24['Severity'].isin(['Orange','Red'])] if not ev24.empty else pd.DataFrame()
    if not hi_sev.empty:
        ev = hi_sev.iloc[0]
        parts.append(f"🚨 **{ev['Type']}** ({ev['Severity']}) — {ev['Event'][:50]}…")
    st.markdown(
        '<div class="brief-box"><b>☀️ Morning Brief</b> &nbsp;·&nbsp; ' +
        '  &nbsp;|&nbsp;  '.join(parts) + '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────
total_cap  = df_equity['Market Cap (B)'].sum()               if 'Market Cap (B)'  in df_equity.columns else 0
avg_pe     = df_equity['Forward P/E'].dropna().mean()         if 'Forward P/E'     in df_equity.columns else 0
avg_chg    = df_equity['Daily Chg (%)'].dropna().mean()       if 'Daily Chg (%)'   in df_equity.columns else 0
n_events   = len(df_events)
n_recent   = int((df_events['Hours Ago'] <= 24).sum())        if 'Hours Ago'       in df_events.columns else 0
above_ma   = int(df_equity['vs 50D MA'].dropna().gt(0).sum()) if 'vs 50D MA'       in df_equity.columns else 0
below_ma   = int(df_equity['vs 50D MA'].dropna().le(0).sum())

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Universe",          f"{len(df_equity)} names")
c2.metric("Cohort Mkt Cap",    f"${total_cap:,.0f}B")
c3.metric("Avg Fwd P/E",       f"{avg_pe:.1f}×")
c4.metric("Portfolio 1D Avg",  f"{avg_chg:+.2f}%",  delta=f"{avg_chg:+.2f}%")
c5.metric("Above 50D MA",      f"{above_ma}/{above_ma+below_ma}")
c6.metric("Disasters (24h)",   n_events, delta=f"{n_recent} new", delta_color="inverse")

st.divider()

# ─────────────────────────────────────────────
# CONVICTION WATCHLIST TABLE
# ─────────────────────────────────────────────
with st.expander("📋 Jane Street Conviction Watchlist — click to expand", expanded=True):
    tag_html = {'buy': '<span class="tag tag-buy">BUY</span>',
                'watch': '<span class="tag tag-watch">WATCH</span>',
                'risk': '<span class="tag tag-risk">RISK</span>'}
    rows = "".join(
        f"<tr><td><b>{t}</b></td><td>{TICKER_TO_SECTOR.get(t,'')}</td>"
        f"<td>{tag_html.get(tag,'')}</td><td>{thesis}</td></tr>"
        for t, tag, thesis in CONVICTION
    )
    st.markdown(
        f'<table class="conv-table"><thead><tr>'
        f'<th>Ticker</th><th>Theme</th><th>Signal</th><th>Thesis</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>',
        unsafe_allow_html=True
    )

st.divider()

# ─────────────────────────────────────────────
# ROW 1: MOVERS + VOLUME
# ─────────────────────────────────────────────
st.subheader("⚡ Today's Movers & Volume Anomalies")
col_mv, col_vol = st.columns(2)

with col_mv:
    df_chg = df_equity.dropna(subset=['Daily Chg (%)']).sort_values('Daily Chg (%)')
    fig = go.Figure(go.Bar(
        x=df_chg['Daily Chg (%)'], y=df_chg['Ticker'], orientation='h',
        marker_color=['#ff4444' if v < 0 else '#00cc88' for v in df_chg['Daily Chg (%)']],
        text=[f"{v:+.2f}%" for v in df_chg['Daily Chg (%)']],
        textposition='outside',
        customdata=df_chg[['Sector','Current Price']].values,
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>$%{customdata[1]:.2f} · %{x:+.2f}%<extra></extra>",
    ))
    fig.update_layout(template='plotly_dark', title='1-Day Price Change',
                      xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1),
                      margin=dict(l=10,r=60,t=40,b=10), height=400)
    st.plotly_chart(fig, use_container_width=True)

with col_vol:
    df_vol = df_equity.dropna(subset=['Vol Spike']).sort_values('Vol Spike')
    fig_v = go.Figure(go.Bar(
        x=df_vol['Vol Spike'], y=df_vol['Ticker'], orientation='h',
        marker_color=['#ffaa00' if v > 2 else '#445566' for v in df_vol['Vol Spike']],
        text=[f"{v:.1f}×" for v in df_vol['Vol Spike']], textposition='outside',
        hovertemplate="<b>%{y}</b> · %{x:.2f}× avg volume<extra></extra>",
    ))
    fig_v.add_vline(x=2.0, line_dash="dot", line_color="#ffaa00", opacity=0.6,
                    annotation_text="2× spike", annotation_position="top")
    fig_v.update_layout(template='plotly_dark', title='Volume vs 30-Day Average',
                        margin=dict(l=10,r=60,t=40,b=10), height=400)
    st.plotly_chart(fig_v, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 2: VALUATION MATRIX + MAP
# ─────────────────────────────────────────────
col_l, col_r = st.columns([1.1, 1])

with col_l:
    st.subheader("📊 Valuation Matrix")
    df_v = df_equity.dropna(subset=['Forward P/E','Market Cap (B)','vs 50D MA']).copy()
    if not df_v.empty:
        df_v['MA Signal'] = df_v['vs 50D MA'].apply(lambda x: "Above 50D MA 🟢" if x>=0 else "Below 50D MA 🔴")
        fig_s = px.scatter(df_v, x='Forward P/E', y='Market Cap (B)',
                           color='Sector', symbol='MA Signal', hover_name='Ticker',
                           hover_data={'Current Price':':.2f','vs 50D MA':':.2f','Daily Chg (%)':':.2f',
                                       'Market Cap (B)':':.1f','MA Signal':False},
                           log_y=True, template='plotly_dark',
                           title='Fwd P/E vs Market Cap — shape = 50D MA')
        fig_s.update_traces(marker=dict(size=12, line=dict(width=1,color='white')))
        fig_s.update_layout(margin=dict(l=10,r=10,t=50,b=10), height=420)
        st.plotly_chart(fig_s, use_container_width=True)

with col_r:
    st.subheader("🌍 Live GDACS Disaster Map")
    if not df_events.empty:
        df_map = df_events[
            df_events['Type'].isin(selected_types) &
            (df_events['Hours Ago'] <= hours_filter)
        ].dropna(subset=['lat','lon'])
        if not df_map.empty:
            fig_m = px.scatter_geo(df_map, lat='lat', lon='lon', color='Type',
                                   hover_name='Event',
                                   hover_data={'lat':False,'lon':False,'Type':True,
                                               'Hours Ago':':.1f','Severity':True,
                                               'Quant Signal':True,'Details':True},
                                   color_discrete_map=COLOR_MAP, template='plotly_dark',
                                   title=f'Active Events — last {hours_filter}h')
            for trace in fig_m.data:
                mask = df_map['Type'] == trace.name
                if mask.any():
                    trace.marker.opacity = df_map.loc[mask,'Opacity'].tolist()
                trace.marker.size = 11
                trace.marker.line = dict(width=1, color='rgba(255,255,255,0.3)')
            fig_m.update_geos(showcoastlines=True, coastlinecolor='#555',
                              showland=True, landcolor='#1a1a2e',
                              showocean=True, oceancolor='#0d1b2a',
                              showframe=False, projection_type='natural earth')
            fig_m.update_layout(margin=dict(l=0,r=0,t=50,b=0), height=420,
                                legend=dict(orientation='h',y=-0.15,x=0.5,xanchor='center'))
            st.plotly_chart(fig_m, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 3: 52W RANGE + 50D MA SIGNAL
# ─────────────────────────────────────────────
col_52, col_50 = st.columns(2)

with col_52:
    st.subheader("📐 52-Week Range Position")
    df_52 = df_equity.dropna(subset=['52W High','52W Low','Current Price'])
    if not df_52.empty:
        fig_52 = go.Figure()
        for _, row in df_52.iterrows():
            lo, hi, px_ = row['52W Low'], row['52W High'], row['Current Price']
            pct = ((px_ - lo) / (hi - lo) * 100) if (hi - lo) > 0 else 50
            color = '#00cc88' if pct > 70 else ('#ffaa00' if pct > 35 else '#ff4444')
            fig_52.add_trace(go.Bar(
                x=[pct], y=[row['Ticker']], orientation='h', marker_color=color,
                text=f"${px_:.2f}  ({pct:.0f}%)", textposition='inside',
                showlegend=False, name=row['Ticker'],
                hovertemplate=f"<b>{row['Ticker']}</b><br>52W Low: ${lo:.2f} · High: ${hi:.2f}<br>Position: {pct:.1f}%<extra></extra>",
            ))
        fig_52.update_layout(template='plotly_dark',
                             title='Position in 52-Week Range (0%=Low · 100%=High)',
                             xaxis=dict(title='% of Range', range=[0,115]),
                             margin=dict(l=10,r=10,t=50,b=10), height=380, barmode='overlay')
        st.plotly_chart(fig_52, use_container_width=True)

with col_50:
    st.subheader("📡 50-Day MA Signal")
    df_ma = df_equity.dropna(subset=['vs 50D MA']).sort_values('vs 50D MA')
    if not df_ma.empty:
        fig_ma = go.Figure(go.Bar(
            x=df_ma['vs 50D MA'], y=df_ma['Ticker'], orientation='h',
            marker_color=['#ff4444' if v<0 else '#00cc88' for v in df_ma['vs 50D MA']],
            text=[f"{v:+.1f}%" for v in df_ma['vs 50D MA']], textposition='outside',
            customdata=df_ma[['Current Price','50-Day MA']].values,
            hovertemplate="<b>%{y}</b><br>Price: $%{customdata[0]:.2f} · 50D MA: $%{customdata[1]:.2f}<br>Delta: %{x:+.2f}%<extra></extra>",
        ))
        fig_ma.update_layout(template='plotly_dark', title='% vs 50-Day Moving Average',
                             xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1.5),
                             margin=dict(l=10,r=60,t=40,b=10), height=380)
        st.plotly_chart(fig_ma, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 4: SECTOR HEATMAP + MARKET CAP
# ─────────────────────────────────────────────
col_h, col_b = st.columns(2)

with col_h:
    st.subheader("🌡️ Sector Performance")
    sp = df_equity.groupby('Sector')['Daily Chg (%)'].mean().reset_index()
    sp.columns = ['Sector','Avg 1D (%)']
    sp = sp.sort_values('Avg 1D (%)')
    fig_h = go.Figure(go.Bar(
        x=sp['Avg 1D (%)'], y=sp['Sector'], orientation='h',
        marker_color=['#00cc88' if v>=0 else '#ff4444' for v in sp['Avg 1D (%)']],
        text=[f"{v:+.2f}%" for v in sp['Avg 1D (%)']], textposition='outside',
    ))
    fig_h.update_layout(template='plotly_dark', title='Avg 1D Chg by Theme',
                        xaxis=dict(zeroline=True,zerolinecolor='white',zerolinewidth=1),
                        margin=dict(l=10,r=70,t=40,b=10), height=280)
    st.plotly_chart(fig_h, use_container_width=True)

with col_b:
    st.subheader("💰 Market Cap by Ticker")
    df_bar = df_equity.dropna(subset=['Market Cap (B)']).sort_values('Market Cap (B)')
    fig_b = px.bar(df_bar, x='Market Cap (B)', y='Ticker', color='Sector',
                   orientation='h', template='plotly_dark',
                   title='Market Cap ($B)', text_auto='.0f')
    fig_b.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=280, showlegend=False)
    fig_b.update_traces(textposition='outside')
    st.plotly_chart(fig_b, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# DATA TABLES
# ─────────────────────────────────────────────
st.subheader("🗃️ Data Ledger")
te, tv, t24 = st.tabs(["📋 Equity Ledger", "🌩️ All Disasters", "🚨 Last 24h"])

with te:
    cols = ['Ticker','Sector','Current Price','Daily Chg (%)','Vol Spike',
            'vs 50D MA','vs 52W High (%)','30-Day Trend','Forward P/E','Market Cap (B)']
    st.dataframe(
        df_equity[[c for c in cols if c in df_equity.columns]],
        column_config={
            'Current Price':   st.column_config.NumberColumn('Price',     format='$%.2f'),
            'Daily Chg (%)':   st.column_config.NumberColumn('1D Chg',    format='%.2f%%'),
            'Vol Spike':       st.column_config.NumberColumn('Vol ×Avg',  format='%.2f×'),
            'vs 50D MA':       st.column_config.NumberColumn('vs 50D MA', format='%.2f%%'),
            'vs 52W High (%)': st.column_config.NumberColumn('vs 52W Hi', format='%.1f%%'),
            '30-Day Trend':    st.column_config.LineChartColumn('30D Trend'),
            'Forward P/E':     st.column_config.NumberColumn('Fwd P/E',   format='%.1f×'),
            'Market Cap (B)':  st.column_config.NumberColumn('Mkt Cap',   format='$%.1fB'),
        },
        hide_index=True, use_container_width=True, height=370,
    )

with tv:
    if not df_events.empty:
        st.dataframe(
            df_events[['ParsedDate','Hours Ago','Type','Severity','Event','Quant Signal','Details']],
            column_config={
                'ParsedDate': st.column_config.DatetimeColumn('Date (UTC)', format='MMM D, HH:mm'),
                'Hours Ago':  st.column_config.NumberColumn('Hrs Ago',      format='%.1f h'),
            },
            hide_index=True, use_container_width=True, height=370,
        )

with t24:
    if not df_events.empty and 'Hours Ago' in df_events.columns:
        df_24 = df_events[df_events['Hours Ago'] <= 24].sort_values('ParsedDate', ascending=False)
        if not df_24.empty:
            st.dataframe(
                df_24[['ParsedDate','Hours Ago','Type','Severity','Event','Quant Signal']],
                column_config={
                    'ParsedDate': st.column_config.DatetimeColumn('Date (UTC)', format='MMM D, HH:mm'),
                    'Hours Ago':  st.column_config.NumberColumn('Hrs Ago',      format='%.1f h'),
                },
                hide_index=True, use_container_width=True, height=370,
            )
        else:
            st.info("No new events in the last 24 hours.")

st.divider()

# ─────────────────────────────────────────────
# AI ALPHA ENGINE
# ─────────────────────────────────────────────
st.subheader("🧠 AI Alpha Engine")
col_al, col_ar = st.columns([1, 2])

with col_al:
    st.markdown(f"**Model:** `{model_id}`\n\n**Temp:** `{temperature}` · **Endpoint:** `{LLM_BASE_URL}`")
    run_ai = st.button("⚡ Generate Morning Alpha Report", type="primary", use_container_width=True)

with col_ar:
    if run_ai:
        with st.spinner(f"Querying {model_id}…"):
            try:
                eq_cols = ['Ticker','Sector','Current Price','Daily Chg (%)','Vol Spike','vs 50D MA','Forward P/E','Market Cap (B)']
                eq_str  = df_equity[[c for c in eq_cols if c in df_equity.columns]].to_string(index=False)
                ev_str  = (df_events[df_events['Hours Ago'] <= 48][['Type','Severity','Event','Quant Signal']].head(20).to_string(index=False)
                           if not df_events.empty else "No recent events.")
                prompt = f"""You are a senior analyst at Jane Street preparing the morning briefing. Today: {last_refresh.strftime('%A %B %d, %Y')}.

Write a sharp morning alpha memo with EXACTLY these 4 sections:

1. 🔥 TOP RISK (2-3 sentences): Most important risk from today's disaster feed. Name specific tickers exposed.
2. 💹 TOP OPPORTUNITY (2-3 sentences): Strongest alpha from events × portfolio intersection. Be specific.
3. 📊 TECHNICAL FLAGS (bullets): Tickers with unusual volume or notable 50D MA crossings.
4. 📌 WATCH LIST: 2 tickers. One sentence each. No disclaimers.

--- EQUITY DATA ---
{eq_str}

--- CLIMATE/DISASTER EVENTS (48h) ---
{ev_str}
"""
                resp = requests.post(
                    f"{LLM_BASE_URL}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"model": model_id,
                                     "messages": [{"role": "user", "content": prompt}],
                                     "temperature": temperature}),
                    timeout=90,
                )
                if resp.status_code == 200:
                    content = resp.json().get('choices',[{}])[0].get('message',{}).get('content','')
                    st.success(f"Analysis complete — {last_refresh.strftime('%H:%M UTC')}")
                    st.markdown(content)
                else:
                    st.error(f"LLM Error ({resp.status_code}): {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Timeout — model may be cold-loading, retry in 30s.")
            except Exception as e:
                st.error(f"Failed to reach local LLM: {e}")
    else:
        st.info("Click **⚡ Generate Morning Alpha Report** to run AI cross-analysis on today's live data.")
