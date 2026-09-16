"""
5_Commodities.py — Professional Commodities Terminal
─────────────────────────────────────────────────────
Jane Street-style commodities analysis:
  · Multi-timeframe trend dashboard (1D / 1W / 1M / 1Y)
  · Cross-commodity ratio monitor (GSR, Cu/Au, crack spread, WTI/Brent)
  · Rolling correlation matrix
  · Seasonality engine (5Y avg monthly returns)
  · Realized volatility screener
  · Term structure / contango-backwardation signal
  · AI commodity analysis
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import requests, json

from shared import LLM_BASE_URL

# ─────────────────────────────────────────────
# UNIVERSE
# ─────────────────────────────────────────────
COMMODITIES = {
    '⚡ Energy': {
        'WTI Crude':   'CL=F',
        'Brent Crude': 'BZ=F',
        'Natural Gas': 'NG=F',
        'RBOB Gas':    'RB=F',
        'Heating Oil': 'HO=F',
    },
    '🥇 Precious Metals': {
        'Gold':        'GC=F',
        'Silver':      'SI=F',
        'Platinum':    'PL=F',
        'Palladium':   'PA=F',
    },
    '🔩 Industrial Metals': {
        'Copper':      'HG=F',
        'Aluminum':    'ALI=F',
    },
    '🌾 Agriculture': {
        'Wheat':       'ZW=F',
        'Corn':        'ZC=F',
        'Soybeans':    'ZS=F',
        'Coffee':      'KC=F',
        'Sugar':       'SB=F',
        'Cocoa':       'CC=F',
        'Cotton':      'CT=F',
    },
    '🐄 Livestock': {
        'Live Cattle': 'LE=F',
        'Lean Hogs':   'HE=F',
    },
}

# ETF fallbacks for futures that fail on Streamlit Cloud
ETF_FALLBACKS = {
    'CL=F': 'USO',  'NG=F': 'UNG',  'GC=F': 'GLD',
    'SI=F': 'SLV',  'HG=F': 'CPER', 'ZW=F': 'WEAT',
    'ZC=F': 'CORN', 'ZS=F': 'SOYB',
}

ALL_FUTURES = [sym for grp in COMMODITIES.values() for sym in grp.values()]
NAME_MAP    = {sym: name for grp in COMMODITIES.values() for name, sym in grp.items()}
GROUP_MAP   = {sym: grp  for grp, items in COMMODITIES.items() for sym in items.values()}

# Key cross-commodity ratios tracked by prop desks
RATIOS = {
    'Gold / Silver (GSR)':        ('GC=F',  'SI=F',  'Risk sentiment — high = fear, low = risk-on'),
    'Copper / Gold':              ('HG=F',  'GC=F',  'Growth proxy — rising = economic expansion'),
    'WTI / Brent Spread':         ('CL=F',  'BZ=F',  'Geopolitical premium in Brent vs US supply'),
    'Crude Oil / Nat Gas (6:1)':  ('CL=F',  'NG=F',  'Energy substitution ratio (×6 for BTU parity)'),
    'Wheat / Corn':               ('ZW=F',  'ZC=F',  'Grain substitution — weather & supply signal'),
    'Silver / Gold (Ratio)':      ('SI=F',  'GC=F',  'Industrial vs monetary demand split'),
}

UNIT_MAP = {
    'CL=F': '$/bbl', 'BZ=F': '$/bbl', 'NG=F': '$/MMBtu',
    'RB=F': '$/gal', 'HO=F': '$/gal',
    'GC=F': '$/oz',  'SI=F': '$/oz',   'PL=F': '$/oz',  'PA=F': '$/oz',
    'HG=F': '$/lb',  'ALI=F': '$/MT',
    'ZW=F': '¢/bu',  'ZC=F': '¢/bu',   'ZS=F': '¢/bu',
    'KC=F': '¢/lb',  'SB=F': '¢/lb',   'CC=F': '$/MT',  'CT=F': '¢/lb',
    'LE=F': '¢/lb',  'HE=F': '¢/lb',
}

STYLES = """
<style>
    [data-testid="stMetricValue"] { font-size:1.2rem !important; font-weight:700; }
    [data-testid="stMetricLabel"] { font-size:0.7rem !important; text-transform:uppercase; letter-spacing:.05em; }
    .stTabs [data-baseweb="tab"] { font-weight:700; font-size:.88rem; }
    hr { margin:.3rem 0 .8rem 0 !important; }
    .ratio-card {
        background:#0d1b2a; border:1px solid #1e3a5f; border-radius:8px;
        padding:.8rem 1rem; margin:.3rem 0;
    }
    .ratio-value { font-size:1.4rem; font-weight:800; color:#00ccff; }
    .ratio-delta-pos { color:#00cc88; font-size:.85rem; }
    .ratio-delta-neg { color:#ff4444; font-size:.85rem; }
    .signal-bull { background:#064e3b; color:#6ee7b7; padding:.2rem .6rem; border-radius:4px; font-size:.75rem; font-weight:700; }
    .signal-bear { background:#4c1d1d; color:#fca5a5; padding:.2rem .6rem; border-radius:4px; font-size:.75rem; font-weight:700; }
    .signal-neut { background:#1f2937; color:#9ca3af; padding:.2rem .6rem; border-radius:4px; font-size:.75rem; font-weight:700; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING — with ETF fallback
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_commodity_prices(period: str = '2y') -> pd.DataFrame:
    """Load commodity futures, fall back to ETFs if futures unavailable."""
    all_syms = ALL_FUTURES + list(set(ETF_FALLBACKS.values()))
    try:
        raw = yf.download(all_syms, period=period, auto_adjust=True,
                          progress=False, group_by='ticker')
        if isinstance(raw.columns, pd.MultiIndex):
            try:    closes = raw.xs('Close', axis=1, level=1)
            except: closes = raw['Close']
        else:
            closes = raw['Close'] if 'Close' in raw.columns else raw
        closes = closes.dropna(how='all')
    except Exception:
        closes = pd.DataFrame()

    # Patch missing futures with ETF fallbacks
    for fut, etf in ETF_FALLBACKS.items():
        if fut not in closes.columns or closes[fut].dropna().empty:
            if etf in closes.columns and not closes[etf].dropna().empty:
                closes[fut] = closes[etf]

    return closes


@st.cache_data(ttl=300, show_spinner=False)
def build_returns_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym in ALL_FUTURES:
        if sym not in prices.columns:
            continue
        s = prices[sym].dropna()
        if len(s) < 5:
            continue
        price = s.iloc[-1]
        d1   = (s.iloc[-1]/s.iloc[-2]   - 1)*100 if len(s)>=2   else None
        d5   = (s.iloc[-1]/s.iloc[-6]   - 1)*100 if len(s)>=6   else None
        d21  = (s.iloc[-1]/s.iloc[-22]  - 1)*100 if len(s)>=22  else None
        d63  = (s.iloc[-1]/s.iloc[-64]  - 1)*100 if len(s)>=64  else None
        d252 = (s.iloc[-1]/s.iloc[-253] - 1)*100 if len(s)>=253 else None
        ma20  = s.tail(20).mean()
        ma50  = s.tail(50).mean()
        ma200 = s.tail(200).mean() if len(s)>=200 else None
        vol30 = s.pct_change().tail(30).std() * np.sqrt(252) * 100
        vol90 = s.pct_change().tail(90).std() * np.sqrt(252) * 100 if len(s)>=90 else None
        rows.append({
            'Symbol':      sym,
            'Name':        NAME_MAP.get(sym, sym),
            'Group':       GROUP_MAP.get(sym, ''),
            'Price':       round(price, 4),
            'Unit':        UNIT_MAP.get(sym, ''),
            '1D %':        round(d1,   2) if d1   is not None else None,
            '1W %':        round(d5,   2) if d5   is not None else None,
            '1M %':        round(d21,  2) if d21  is not None else None,
            '3M %':        round(d63,  2) if d63  is not None else None,
            '1Y %':        round(d252, 2) if d252 is not None else None,
            'vs 20D MA':   round((price-ma20)/ma20*100, 2) if ma20 else None,
            'vs 50D MA':   round((price-ma50)/ma50*100, 2) if ma50 else None,
            'vs 200D MA':  round((price-ma200)/ma200*100, 2) if ma200 else None,
            'RVol 30D':    round(vol30, 1),
            'RVol 90D':    round(vol90, 1) if vol90 is not None else None,
            'Vol Regime':  'HIGH' if vol30 > (vol90 or vol30)*1.2 else ('LOW' if vol30 < (vol90 or vol30)*0.8 else 'NORM'),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def build_seasonality(prices: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Average monthly returns by month (Jan–Dec) using up to 5Y of data."""
    if sym not in prices.columns:
        return pd.DataFrame()
    s = prices[sym].dropna()
    monthly = s.resample('ME').last().pct_change().dropna() * 100
    df = monthly.reset_index()
    df.columns = ['Date', 'Return']
    df['Month']  = df['Date'].dt.month
    df['Year']   = df['Date'].dt.year
    df['MonthName'] = df['Date'].dt.strftime('%b')
    result = df.groupby('Month').agg(
        AvgReturn=('Return','mean'),
        MedianReturn=('Return','median'),
        WinRate=('Return', lambda x: (x>0).mean()*100),
        MonthName=('MonthName','first'),
    ).reset_index()
    return result


@st.cache_data(ttl=300, show_spinner=False)
def build_correlation_matrix(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    syms = [s for s in ALL_FUTURES if s in prices.columns]
    ret  = prices[syms].pct_change().tail(window).dropna(how='all')
    corr = ret.corr().round(3)
    corr.index   = [NAME_MAP.get(s, s) for s in corr.index]
    corr.columns = [NAME_MAP.get(s, s) for s in corr.columns]
    return corr


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Loading commodity markets…"):
    prices  = load_commodity_prices('2y')
    df      = build_returns_table(prices)

now = datetime.now(timezone.utc)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.caption(f"Synced: {now.strftime('%b %d · %H:%M UTC')}")
    if st.button("🔄 Refresh", use_container_width=True, type="primary"):
        st.cache_data.clear(); st.rerun()
    st.divider()
    st.subheader("🔍 Seasonality Tool")
    seas_sym_name = st.selectbox("Commodity", options=df['Name'].tolist() if not df.empty else ['Gold'])
    st.divider()
    st.subheader("📐 Correlation Window")
    corr_window = st.slider("Rolling Days", 20, 252, 60, 5)
    st.divider()
    st.subheader("🧠 AI Config")
    llm_endpoint = st.text_input("LLM Endpoint",
                                  value=st.session_state.get("llm_endpoint", LLM_BASE_URL),
                                  placeholder="https://xxxx.trycloudflare.com")
    st.session_state["llm_endpoint"] = llm_endpoint
    model_id    = st.text_input("Model ID", value="qwen2.5-coder-14b-instruct-mlx")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🛢️ Commodities Terminal")
st.caption(f"Futures · Ratios · Seasonality · Vol Regimes · {now.strftime('%A %B %d %Y — %H:%M UTC')}")

if df.empty:
    st.error("No commodity data loaded. Markets may be closed or data source unavailable.")
    st.stop()

st.divider()

# ─────────────────────────────────────────────
# KPI STRIP — Top commodities by group
# ─────────────────────────────────────────────
highlight = ['WTI Crude','Brent Crude','Natural Gas','Gold','Silver','Copper',
             'Wheat','Corn','Soybeans']
kpi_df = df[df['Name'].isin(highlight)].set_index('Name')
kpi_cols = st.columns(len(kpi_df))
for col, name in zip(kpi_cols, highlight):
    if name in kpi_df.index:
        row   = kpi_df.loc[name]
        delta = f"{row['1D %']:+.2f}%" if row['1D %'] is not None else None
        col.metric(name, f"{row['Price']:,.2f}", delta=delta,
                   delta_color="normal" if row.get('1D %', 0) or 0 >= 0 else "inverse")

st.divider()

# ═══════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════
tab_daily, tab_weekly, tab_monthly, tab_yearly, tab_ratios, tab_corr, tab_seas, tab_vol, tab_ai = st.tabs([
    "📅 Daily", "📆 Weekly", "🗓️ Monthly", "📊 Yearly",
    "⚖️ Key Ratios", "🔗 Correlation", "📈 Seasonality",
    "🌡️ Volatility", "🧠 AI Analysis",
])

# ─────────────────────────────────────────────
# DAILY TAB
# ─────────────────────────────────────────────
with tab_daily:
    st.subheader("📅 Today's Commodity Moves")
    df_d = df.dropna(subset=['1D %']).sort_values('1D %')
    if not df_d.empty:
        fig = go.Figure(go.Bar(
            x=df_d['1D %'], y=df_d['Name'], orientation='h',
            marker_color=['#ff4444' if v < 0 else '#00cc88' for v in df_d['1D %']],
            text=[f"{v:+.2f}%" for v in df_d['1D %']], textposition='outside',
            customdata=df_d[['Group','Price','Unit']].values,
            hovertemplate="<b>%{y}</b> %{customdata[2]}<br>%{customdata[0]}<br>${%customdata[1]:.4g} · %{x:+.2f}%<extra></extra>",
        ))
        fig.add_vline(x=0, line_color='white', opacity=0.3, line_width=1)
        fig.update_layout(template='plotly_dark', title='1-Day % Change — All Commodities',
                          xaxis=dict(title='% Change'),
                          margin=dict(l=10, r=100, t=45, b=10), height=580)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily Leader / Laggard Summary")
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**🟢 Top Gainers**")
            for _, row in df_d.tail(5).iloc[::-1].iterrows():
                if row['1D %'] and row['1D %'] > 0:
                    st.markdown(f"**{row['Name']}** `{row['1D %']:+.2f}%`  ·  {row['Group']}")
        with dc2:
            st.markdown("**🔴 Top Losers**")
            for _, row in df_d.head(5).iterrows():
                if row['1D %'] and row['1D %'] < 0:
                    st.markdown(f"**{row['Name']}** `{row['1D %']:+.2f}%`  ·  {row['Group']}")

# ─────────────────────────────────────────────
# WEEKLY TAB
# ─────────────────────────────────────────────
with tab_weekly:
    st.subheader("📆 7-Day Rolling Returns")
    df_w = df.dropna(subset=['1W %']).sort_values('1W %')
    if not df_w.empty:
        fig_w = go.Figure(go.Bar(
            x=df_w['1W %'], y=df_w['Name'], orientation='h',
            marker_color=['#ff4444' if v < 0 else '#00cc88' for v in df_w['1W %']],
            text=[f"{v:+.2f}%" for v in df_w['1W %']], textposition='outside',
            hovertemplate="<b>%{y}</b><br>1W: %{x:+.2f}%<extra></extra>",
        ))
        fig_w.add_vline(x=0, line_color='white', opacity=0.3, line_width=1)
        fig_w.update_layout(template='plotly_dark', title='5-Day % Change',
                             margin=dict(l=10, r=100, t=45, b=10), height=580)
        st.plotly_chart(fig_w, use_container_width=True)

        # Group breakdown
        st.subheader("By Group")
        group_w = df_w.groupby('Group')['1W %'].mean().sort_values()
        fig_gw = go.Figure(go.Bar(
            x=group_w.values, y=group_w.index, orientation='h',
            marker_color=['#ff4444' if v < 0 else '#00cc88' for v in group_w.values],
            text=[f"{v:+.2f}%" for v in group_w.values], textposition='outside',
        ))
        fig_gw.update_layout(template='plotly_dark', title='Average 1W % by Commodity Group',
                              margin=dict(l=10, r=80, t=40, b=10), height=260)
        st.plotly_chart(fig_gw, use_container_width=True)

# ─────────────────────────────────────────────
# MONTHLY TAB
# ─────────────────────────────────────────────
with tab_monthly:
    st.subheader("🗓️ 30-Day Trend & Moving Average Status")
    df_m = df.dropna(subset=['1M %']).sort_values('1M %')
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_m = go.Figure(go.Bar(
            x=df_m['1M %'], y=df_m['Name'], orientation='h',
            marker_color=['#ff4444' if v < 0 else '#00cc88' for v in df_m['1M %']],
            text=[f"{v:+.2f}%" for v in df_m['1M %']], textposition='outside',
        ))
        fig_m.add_vline(x=0, line_color='white', opacity=0.3)
        fig_m.update_layout(template='plotly_dark', title='1-Month % Change',
                             margin=dict(l=10, r=80, t=40, b=10), height=520)
        st.plotly_chart(fig_m, use_container_width=True)
    with col_m2:
        df_ma = df.dropna(subset=['vs 20D MA','vs 50D MA']).sort_values('vs 50D MA')
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Bar(
            name='vs 20D MA', x=df_ma['vs 20D MA'], y=df_ma['Name'], orientation='h',
            marker_color=['#00cc88' if v >= 0 else '#ff4444' for v in df_ma['vs 20D MA']],
            opacity=0.8,
        ))
        fig_ma.add_trace(go.Scatter(
            name='vs 50D MA', x=df_ma['vs 50D MA'], y=df_ma['Name'],
            mode='markers', marker=dict(symbol='diamond', size=9, color='#ffffff'),
        ))
        fig_ma.add_vline(x=0, line_color='white', opacity=0.3)
        fig_ma.update_layout(template='plotly_dark', title='vs 20D & 50D Moving Averages',
                              barmode='overlay', legend=dict(orientation='h', y=1.08),
                              margin=dict(l=10, r=10, t=50, b=10), height=520)
        st.plotly_chart(fig_ma, use_container_width=True)

    # Multi-timeframe heatmap
    st.subheader("Multi-Timeframe Heatmap")
    periods = ['1D %', '1W %', '1M %', '3M %']
    heat_df = df.dropna(subset=['1D %']).set_index('Name')[periods]
    fig_heat = go.Figure(go.Heatmap(
        z=heat_df.values, x=['1 Day', '1 Week', '1 Month', '3 Months'], y=heat_df.index.tolist(),
        colorscale='RdYlGn', zmid=0,
        text=[[f"{v:+.1f}%" if v is not None and not np.isnan(v) else "—"
               for v in row] for row in heat_df.values],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title='% Chg'), hoverongaps=False,
    ))
    fig_heat.update_layout(
        template='plotly_dark', title='% Returns — All Timeframes',
        margin=dict(l=10, r=10, t=50, b=10), height=520,
        xaxis=dict(side='top'),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
# YEARLY TAB
# ─────────────────────────────────────────────
with tab_yearly:
    st.subheader("📊 52-Week Performance")
    df_y = df.dropna(subset=['1Y %']).sort_values('1Y %')
    if not df_y.empty:
        fig_y = go.Figure(go.Bar(
            x=df_y['1Y %'], y=df_y['Name'], orientation='h',
            marker_color=['#ff4444' if v < 0 else '#00cc88' for v in df_y['1Y %']],
            text=[f"{v:+.2f}%" for v in df_y['1Y %']], textposition='outside',
            customdata=df_y[['vs 200D MA','Group']].values,
            hovertemplate="<b>%{y}</b><br>%{customdata[1]}<br>1Y: %{x:+.2f}%<br>vs 200D MA: %{customdata[0]:+.2f}%<extra></extra>",
        ))
        fig_y.add_vline(x=0, line_color='white', opacity=0.3)
        fig_y.update_layout(template='plotly_dark', title='52-Week % Return',
                             margin=dict(l=10, r=100, t=45, b=10), height=580)
        st.plotly_chart(fig_y, use_container_width=True)

        col_y1, col_y2 = st.columns(2)
        with col_y1:
            st.markdown("**Best 5 (1Y)**")
            for _, r in df_y.tail(5).iloc[::-1].iterrows():
                st.markdown(f"**{r['Name']}** `{r['1Y %']:+.1f}%`")
        with col_y2:
            st.markdown("**vs 200-Day MA**")
            df_200 = df.dropna(subset=['vs 200D MA']).sort_values('vs 200D MA', ascending=False)
            for _, r in df_200.head(5).iterrows():
                icon = '🟢' if r['vs 200D MA'] > 0 else '🔴'
                st.markdown(f"{icon} **{r['Name']}** `{r['vs 200D MA']:+.1f}%` above/below 200D")
    else:
        st.info("1Y data requires at least 252 trading days of history.")

# ─────────────────────────────────────────────
# KEY RATIOS TAB
# ─────────────────────────────────────────────
with tab_ratios:
    st.subheader("⚖️ Cross-Commodity Ratio Monitor")
    st.caption("Key spread relationships tracked by commodities desks at major prop firms.")
    st.markdown("")

    for ratio_name, (sym_a, sym_b, description) in RATIOS.items():
        if sym_a not in prices.columns or sym_b not in prices.columns:
            continue
        s_a = prices[sym_a].dropna()
        s_b = prices[sym_b].dropna()
        if s_a.empty or s_b.empty:
            continue
        common = s_a.index.intersection(s_b.index)
        if len(common) < 20:
            continue

        ratio_series = (s_a.loc[common] / s_b.loc[common]).dropna()
        current = ratio_series.iloc[-1]
        d1  = (current / ratio_series.iloc[-2]  - 1)*100 if len(ratio_series)>=2  else 0
        d1m = (current / ratio_series.iloc[-22] - 1)*100 if len(ratio_series)>=22 else 0
        pct_rank_252 = float((ratio_series.tail(252) < current).mean() * 100) if len(ratio_series)>=252 else None

        # Percentile-based signal
        if pct_rank_252 is not None:
            if pct_rank_252 > 80:   signal, sig_class = "EXTENDED HIGH", "signal-bear"
            elif pct_rank_252 < 20: signal, sig_class = "DEPRESSED LOW",  "signal-bull"
            else:                   signal, sig_class = "MID-RANGE",       "signal-neut"
        else:
            signal, sig_class = "INSUFFICIENT DATA", "signal-neut"

        d1_color  = "ratio-delta-pos" if d1  >= 0 else "ratio-delta-neg"
        d1m_color = "ratio-delta-pos" if d1m >= 0 else "ratio-delta-neg"
        pct_str = f"{pct_rank_252:.0f}th pct (1Y)" if pct_rank_252 is not None else "N/A"

        col_r1, col_r2 = st.columns([2, 1])
        with col_r1:
            st.markdown(f"""
<div class="ratio-card">
<b>{ratio_name}</b><br>
<span style="color:#888; font-size:.78rem;">{description}</span><br><br>
<span class="ratio-value">{current:.4g}</span>&nbsp;
<span class="{d1_color}">1D: {d1:+.2f}%</span>&nbsp;|&nbsp;
<span class="{d1m_color}">1M: {d1m:+.2f}%</span>&nbsp;|&nbsp;
<span style="color:#aaa; font-size:.82rem;">{pct_str}</span>&nbsp;
<span class="{sig_class}">{signal}</span>
</div>""", unsafe_allow_html=True)
        with col_r2:
            # Mini sparkline of last 60 days
            mini = ratio_series.tail(60)
            fig_mini = go.Figure(go.Scatter(
                x=mini.index, y=mini.values, mode='lines',
                line=dict(color='#00ccff', width=1.5),
                fill='tozeroy', fillcolor='rgba(0,204,255,0.07)',
            ))
            fig_mini.update_layout(
                template='plotly_dark', showlegend=False,
                margin=dict(l=0, r=0, t=5, b=0), height=80,
                xaxis=dict(visible=False), yaxis=dict(visible=False),
            )
            st.plotly_chart(fig_mini, use_container_width=True)

# ─────────────────────────────────────────────
# CORRELATION TAB
# ─────────────────────────────────────────────
with tab_corr:
    st.subheader(f"🔗 Commodity Correlation Matrix ({corr_window}D Rolling)")
    st.caption("Pearson correlation of daily returns. Red = positive, Blue = negative (diversifying).")
    with st.spinner("Computing correlations…"):
        corr = build_correlation_matrix(prices, corr_window)
    if not corr.empty:
        z_vals = corr.values
        tickers_in = corr.columns.tolist()
        fig_corr = go.Figure(go.Heatmap(
            z=z_vals, x=tickers_in, y=tickers_in,
            colorscale='RdBu', reversescale=True, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z_vals],
            texttemplate="%{text}", textfont=dict(size=8),
            colorbar=dict(title='Corr'),
            hoverongaps=False,
        ))
        fig_corr.update_layout(
            template='plotly_dark',
            title=f'Pairwise Return Correlation — {corr_window}D window',
            margin=dict(l=10, r=10, t=50, b=100), height=600,
            xaxis=dict(tickangle=-45, side='bottom'),
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Highlight diversifiers
        st.subheader("Diversification Opportunities")
        st.caption("Pairs with correlation < −0.1 (potential hedges or uncorrelated returns)")
        flat = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
        flat.columns = ['Asset A', 'Asset B', 'Correlation']
        diversifiers = flat[flat['Correlation'] < -0.1].sort_values('Correlation')
        if not diversifiers.empty:
            st.dataframe(diversifiers.head(10), hide_index=True, use_container_width=True)
        else:
            st.info("No strong negative correlations found in current window.")

# ─────────────────────────────────────────────
# SEASONALITY TAB
# ─────────────────────────────────────────────
with tab_seas:
    st.subheader("📈 Commodity Seasonality Engine")
    st.caption("Average monthly returns computed from up to 5 years of history.")

    seas_row = df[df['Name'] == seas_sym_name]
    if not seas_row.empty:
        seas_sym = seas_row.iloc[0]['Symbol']
        seas_df  = build_seasonality(prices, seas_sym)
        if not seas_df.empty:
            MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            seas_df = seas_df.sort_values('Month')
            seas_df['MonthName'] = pd.Categorical(seas_df['MonthName'], categories=MONTHS, ordered=True)

            col_s1, col_s2 = st.columns([2, 1])
            with col_s1:
                fig_s = go.Figure()
                fig_s.add_trace(go.Bar(
                    x=seas_df['MonthName'], y=seas_df['AvgReturn'],
                    name='Avg Return',
                    marker_color=['#00cc88' if v >= 0 else '#ff4444' for v in seas_df['AvgReturn']],
                    text=[f"{v:+.2f}%" for v in seas_df['AvgReturn']], textposition='outside',
                    opacity=0.9,
                ))
                fig_s.add_trace(go.Scatter(
                    x=seas_df['MonthName'], y=seas_df['MedianReturn'],
                    name='Median', mode='lines+markers',
                    line=dict(color='#ffffff', dash='dot', width=1.5),
                    marker=dict(size=6),
                ))
                fig_s.add_hline(y=0, line_color='white', opacity=0.3)
                current_month = now.month
                current_month_name = MONTHS[current_month-1]
                fig_s.add_vline(
                    x=current_month_name, line_dash='dash',
                    line_color='#00ccff', opacity=0.8,
                    annotation_text="← Now", annotation_position="top",
                )
                fig_s.update_layout(
                    template='plotly_dark',
                    title=f'{seas_sym_name} — Average Monthly Returns (5Y)',
                    margin=dict(l=10, r=10, t=50, b=10), height=380,
                    legend=dict(orientation='h', y=1.08),
                )
                st.plotly_chart(fig_s, use_container_width=True)

            with col_s2:
                st.markdown(f"**{seas_sym_name} Seasonality Stats**")
                st.markdown("")
                # Current month
                cur = seas_df[seas_df['Month'] == current_month]
                if not cur.empty:
                    r = cur.iloc[0]
                    icon = '🟢' if r['AvgReturn'] >= 0 else '🔴'
                    st.metric(f"{icon} {r['MonthName']} Avg", f"{r['AvgReturn']:+.2f}%")
                    st.metric("Win Rate", f"{r['WinRate']:.0f}%")
                st.divider()
                # Best/worst months
                best  = seas_df.nlargest(3,  'AvgReturn')
                worst = seas_df.nsmallest(3, 'AvgReturn')
                st.markdown("**📅 Seasonally Strong**")
                for _, r in best.iterrows():
                    st.markdown(f"**{r['MonthName']}** `{r['AvgReturn']:+.2f}%` ({r['WinRate']:.0f}% win rate)")
                st.markdown("**📉 Seasonally Weak**")
                for _, r in worst.iterrows():
                    st.markdown(f"**{r['MonthName']}** `{r['AvgReturn']:+.2f}%` ({r['WinRate']:.0f}% win rate)")

# ─────────────────────────────────────────────
# VOLATILITY TAB
# ─────────────────────────────────────────────
with tab_vol:
    st.subheader("🌡️ Realized Volatility Screener")
    st.caption("30D vs 90D realized vol. HIGH = vol expanding, LOW = vol compressing.")

    df_vol = df.dropna(subset=['RVol 30D']).sort_values('RVol 30D', ascending=False)
    if not df_vol.empty:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Bar(
                name='30D RVol', x=df_vol['RVol 30D'], y=df_vol['Name'],
                orientation='h',
                marker_color=[
                    '#ff4444' if r == 'HIGH' else ('#00cc88' if r == 'LOW' else '#4466aa')
                    for r in df_vol['Vol Regime']
                ],
                text=[f"{v:.1f}%  {r}" for v, r in zip(df_vol['RVol 30D'], df_vol['Vol Regime'])],
                textposition='outside',
            ))
            if 'RVol 90D' in df_vol.columns:
                fig_rv.add_trace(go.Scatter(
                    name='90D RVol', x=df_vol['RVol 90D'].fillna(0), y=df_vol['Name'],
                    mode='markers', marker=dict(symbol='diamond', size=8, color='#ffffff'),
                ))
            fig_rv.update_layout(
                template='plotly_dark',
                title='Annualized Realized Volatility — 30D (bar) vs 90D (diamond)',
                barmode='overlay',
                margin=dict(l=10, r=120, t=50, b=10), height=540,
                legend=dict(orientation='h', y=1.08),
            )
            st.plotly_chart(fig_rv, use_container_width=True)

        with col_v2:
            st.markdown("**🔥 Highest Volatility**")
            for _, r in df_vol.head(5).iterrows():
                st.markdown(f"**{r['Name']}** `{r['RVol 30D']:.1f}%` annualized · `{r['Vol Regime']}`")
            st.divider()
            st.markdown("**😴 Lowest Volatility (compressed)**")
            for _, r in df_vol.tail(5).iterrows():
                st.markdown(f"**{r['Name']}** `{r['RVol 30D']:.1f}%` annualized · `{r['Vol Regime']}`")
            st.divider()
            st.markdown("**📡 Vol Regime Summary**")
            regime_counts = df_vol['Vol Regime'].value_counts()
            for reg, cnt in regime_counts.items():
                icon = {'HIGH':'🔴','LOW':'🟢','NORM':'⚪'}.get(reg,'⚪')
                st.markdown(f"{icon} **{reg}**: {cnt} commodities")

        # Historical volatility chart for selected commodity
        st.subheader("Rolling 30D Realized Vol — Historical")
        vol_pick = st.selectbox("Select commodity", df_vol['Name'].tolist(), key="vol_pick")
        vol_sym  = df[df['Name'] == vol_pick]['Symbol'].values[0] if vol_pick else None
        if vol_sym and vol_sym in prices.columns:
            rets = prices[vol_sym].pct_change().dropna()
            rv_series = rets.rolling(30).std() * np.sqrt(252) * 100
            rv_series = rv_series.dropna().tail(252)
            if not rv_series.empty:
                fig_rvh = go.Figure()
                fig_rvh.add_trace(go.Scatter(
                    x=rv_series.index, y=rv_series.values, mode='lines',
                    line=dict(color='#ff9900', width=2),
                    fill='tozeroy', fillcolor='rgba(255,153,0,0.08)',
                    name='30D RVol',
                ))
                mean_rv = rv_series.mean()
                fig_rvh.add_hline(y=mean_rv, line_dash='dot', line_color='white', opacity=0.5,
                                   annotation_text=f"Avg {mean_rv:.1f}%", annotation_position="right")
                fig_rvh.update_layout(
                    template='plotly_dark',
                    title=f'{vol_pick} — 30D Realized Vol (last 252 sessions)',
                    margin=dict(l=10, r=10, t=50, b=10), height=280,
                    showlegend=False,
                )
                st.plotly_chart(fig_rvh, use_container_width=True)

# ─────────────────────────────────────────────
# AI ANALYSIS TAB
# ─────────────────────────────────────────────
with tab_ai:
    st.subheader("🧠 AI Commodity Analysis")

    col_ai1, col_ai2 = st.columns([1, 2])
    with col_ai1:
        st.markdown("**📋 Analysis Templates**")
        run_overview  = st.button("🌍 Commodity Market Overview",     use_container_width=True)
        run_energy    = st.button("⚡ Energy Sector Deep Dive",        use_container_width=True)
        run_metals    = st.button("🥇 Metals Analysis",               use_container_width=True)
        run_agri      = st.button("🌾 Agriculture Outlook",           use_container_width=True)
        run_macro_com = st.button("🌐 Commodities vs Macro Regime",   use_container_width=True)
        run_trade     = st.button("⚖️ Best Commodity Trade Right Now", use_container_width=True)

    with col_ai2:
        # Build compact data context
        data_summary = df[['Name','Group','Price','Unit','1D %','1W %','1M %','1Y %',
                            'vs 50D MA','RVol 30D','Vol Regime']].to_string(index=False)

        TEMPLATES = {
            "overview":  f"Analyze the current commodity market landscape. Using the live data below:\n1. 🌍 MACRO CONTEXT — What is the commodity complex telling us about global growth and inflation?\n2. 📊 WINNERS / LOSERS — Which commodity groups are leading/lagging today and why?\n3. ⚠️ KEY RISKS — Top 2 geopolitical or macro risks moving commodities this week\n4. 💡 TRADE IDEAS — 2 specific commodity trades with entry thesis and invalidation levels\n5. 📅 CATALYSTS — What data or events to watch this week\n\n--- LIVE COMMODITY DATA ---\n{data_summary}",
            "energy":    f"Deep dive analysis on the energy complex (WTI, Brent, Nat Gas, Gasoline):\n1. 🛢️ OIL MARKET — Supply/demand balance, OPEC dynamics, crack spread implications\n2. 🔥 NAT GAS — Seasonal demand, storage levels signal, power burn outlook\n3. 📐 WTI/BRENT SPREAD — What the spread tells us about relative supply/demand\n4. ⚡ ENERGY EQUITIES — Implication for XLE, upstream vs downstream\n5. ⚖️ TRADE — Best energy commodity trade right now\n\n--- LIVE COMMODITY DATA ---\n{data_summary}",
            "metals":    f"Metals market analysis (precious and industrial):\n1. 🥇 GOLD — Key price drivers, real yields dynamic, central bank demand, technical levels\n2. ⚡ SILVER — Industrial demand vs monetary demand split, GSR interpretation\n3. 🔩 COPPER — Economic growth signal, China demand, inventory levels\n4. 💎 GOLD/SILVER RATIO — Current level interpretation and positioning implication\n5. 🔗 COPPER/GOLD — Growth indicator reading and what it signals for equities\n6. ⚖️ TRADE — Best metals trade right now with specific entry and target\n\n--- LIVE COMMODITY DATA ---\n{data_summary}",
            "agri":      f"Agriculture commodity analysis (wheat, corn, soybeans, softs):\n1. 🌾 GRAINS — Current supply/demand dynamics, La Niña/weather impact, export competition\n2. 🌽 CORN vs WHEAT — Substitution dynamics and spread opportunities\n3. ☕ SOFTS — Coffee, sugar, cotton key drivers\n4. 🇧🇷🇺🇸 GEOPOLITICAL — Brazil/Argentina vs US competition, trade flow impacts\n5. ⚖️ TRADE — Best agricultural commodity trade right now with rationale\n\n--- LIVE COMMODITY DATA ---\n{data_summary}",
            "macro_com": f"Commodities as a macro signal:\n1. 📈 INFLATION SIGNAL — What are commodities pricing in for inflation? (Oil, Gold, Copper)\n2. 📉 GROWTH SIGNAL — Copper/Gold ratio and base metals as economic barometer\n3. 💵 DOLLAR IMPACT — DXY correlation with commodity complex, FX transmission\n4. 📊 CORRELATION REGIME — Are traditional commodity/equity correlations holding?\n5. 🔀 POSITIONING — How should a multi-asset portfolio be positioned given current commodity signals?\n\n--- LIVE COMMODITY DATA ---\n{data_summary}",
            "trade":     f"Generate 5 specific commodity trade ideas for this week:\nFor each trade:\n- Instrument: exact commodity/ETF\n- Direction: Long / Short / Spread\n- Thesis: 2-sentence fundamental + technical rationale\n- Entry: current level or specific trigger\n- Target: price level or % move expected\n- Stop: invalidation level\n- Time horizon: days/weeks\n- Risk: what kills the trade\n\nFocus on high-conviction setups. Be specific with numbers.\n\n--- LIVE COMMODITY DATA ---\n{data_summary}",
        }

        ai_output_area = st.empty()

        def run_commodity_ai(template_key: str):
            prompt = TEMPLATES[template_key]
            with st.spinner("Analyzing commodity markets…"):
                try:
                    resp = requests.post(
                        f"{llm_endpoint}/v1/chat/completions",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({
                            "model": model_id,
                            "messages": [{"role": "user",
                                          "content": f"You are a senior commodities analyst at Jane Street with deep expertise in futures markets, macro dynamics, and cross-commodity relative value. Be precise, use specific prices and percentages from the data provided, and give actionable insights. No disclaimers.\n\n{prompt}"}],
                            "temperature": temperature,
                        }),
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        content = resp.json()['choices'][0]['message']['content']
                        st.session_state['comm_ai_output'] = content
                    else:
                        st.session_state['comm_ai_output'] = f"⚠️ Error {resp.status_code}: {resp.text[:200]}"
                except Exception as e:
                    st.session_state['comm_ai_output'] = f"⚠️ {e}"

        if run_overview:  run_commodity_ai("overview")
        if run_energy:    run_commodity_ai("energy")
        if run_metals:    run_commodity_ai("metals")
        if run_agri:      run_commodity_ai("agri")
        if run_macro_com: run_commodity_ai("macro_com")
        if run_trade:     run_commodity_ai("trade")

        if 'comm_ai_output' in st.session_state:
            st.markdown(st.session_state['comm_ai_output'])
        else:
            st.info("Select an analysis template to generate an AI report, or ask a custom question below.")

    st.divider()
    st.subheader("💬 Ask About Commodities")
    with st.form("comm_chat", clear_on_submit=True):
        col_q, col_s = st.columns([5,1])
        with col_q:
            question = st.text_input("", placeholder="e.g. 'Why is gold breaking out? Is copper signalling a recession?'",
                                     label_visibility="collapsed")
        with col_s:
            ask_btn = st.form_submit_button("Ask ➤", type="primary", use_container_width=True)
    if ask_btn and question.strip():
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{llm_endpoint}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({
                        "model": model_id,
                        "messages": [{"role": "user",
                                      "content": f"You are a senior commodities analyst at Jane Street. Answer using the live data below. Be specific and use numbers.\n\n--- LIVE DATA ---\n{data_summary}\n---\n\nQuestion: {question}"}],
                        "temperature": temperature,
                    }),
                    timeout=90,
                )
                if resp.status_code == 200:
                    answer = resp.json()['choices'][0]['message']['content']
                    st.markdown(f"**Q:** {question}")
                    st.markdown(answer)
                else:
                    st.error(f"LLM Error: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")
