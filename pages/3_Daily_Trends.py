"""
3_Daily_Trends.py — Broad Market Daily Trends
──────────────────────────────────────────────
General market pulse: indices, sectors, macro, rates, FX, commodities.
Not thematic — this is the wider tape.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone

from shared import LLM_BASE_URL

# ─────────────────────────────────────────────
# UNIVERSE
# ─────────────────────────────────────────────
INDICES = {
    'S&P 500':      'SPY',
    'Nasdaq 100':   'QQQ',
    'Russell 2000': 'IWM',
    'Dow Jones':    'DIA',
    'VIX':          '^VIX',
}

SECTORS = {
    'Tech':           'XLK',
    'Healthcare':     'XLV',
    'Financials':     'XLF',
    'Energy':         'XLE',
    'Industrials':    'XLI',
    'Comm Services':  'XLC',
    'Consumer Disc':  'XLY',
    'Consumer Stapl': 'XLP',
    'Utilities':      'XLU',
    'Materials':      'XLB',
}

MACRO = {
    'Gold':           'GLD',
    'Oil (WTI)':      'USO',
    'US Dollar':      'UUP',
    'Long Bonds':     'TLT',
    'Short Bonds':    'SHY',
    'High Yield':     'HYG',
    'Intl Dev':       'EFA',
    'Emerging Mkts':  'EEM',
}

ALL_SYMS = list(INDICES.values()) + list(SECTORS.values()) + list(MACRO.values())
SYM_TO_NAME = {v: k for d in [INDICES, SECTORS, MACRO] for k, v in d.items()}

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data() -> pd.DataFrame:
    rows = []
    try:
        raw = yf.download(ALL_SYMS, period='3mo', auto_adjust=True,
                          progress=False, group_by='ticker')
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                closes = raw.xs('Close', axis=1, level=1)
                volumes = raw.xs('Volume', axis=1, level=1)
            except KeyError:
                closes = raw['Close']
                volumes = raw['Volume']
        else:
            closes = raw[['Close']] if 'Close' in raw.columns else raw
            volumes = raw[['Volume']] if 'Volume' in raw.columns else pd.DataFrame()

        closes = closes.dropna(how='all')

        for sym in ALL_SYMS:
            if sym not in closes.columns:
                continue
            s = closes[sym].dropna()
            if len(s) < 2:
                continue
            price   = s.iloc[-1]
            d1      = (s.iloc[-1] / s.iloc[-2]  - 1) * 100  if len(s) >= 2  else None
            d5      = (s.iloc[-1] / s.iloc[-6]  - 1) * 100  if len(s) >= 6  else None
            d21     = (s.iloc[-1] / s.iloc[-22] - 1) * 100  if len(s) >= 22 else None
            d63     = (s.iloc[-1] / s.iloc[-64] - 1) * 100  if len(s) >= 64 else None
            ma20    = s.tail(20).mean()
            ma50    = s.tail(50).mean()
            vol_avg = None
            if sym in (volumes.columns if hasattr(volumes,'columns') else []):
                v = volumes[sym].dropna()
                if len(v) >= 2:
                    vol_avg = v.iloc[-1] / v.tail(30).mean() if v.tail(30).mean() > 0 else None
            rows.append({
                'Symbol':       sym,
                'Name':         SYM_TO_NAME.get(sym, sym),
                'Category':     ('Index' if sym in INDICES.values()
                                 else 'Sector ETF' if sym in SECTORS.values()
                                 else 'Macro'),
                'Price':        round(price, 2),
                '1D %':         round(d1,  2) if d1  is not None else None,
                '1W %':         round(d5,  2) if d5  is not None else None,
                '1M %':         round(d21, 2) if d21 is not None else None,
                '3M %':         round(d63, 2) if d63 is not None else None,
                'vs 20D MA':    round((price - ma20)/ma20*100, 2) if ma20 else None,
                'vs 50D MA':    round((price - ma50)/ma50*100, 2) if ma50 else None,
                'Vol Spike':    round(vol_avg, 2) if vol_avg is not None else None,
                'Sparkline':    s.tail(30).tolist(),
            })
    except Exception as e:
        st.warning(f"Data load error: {e}")
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_vix_history() -> pd.Series:
    try:
        raw = yf.Ticker('^VIX').history(period='3mo')
        return raw['Close'].dropna()
    except Exception:
        return pd.Series(dtype=float)

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size:1.3rem !important; font-weight:700; }
    [data-testid="stMetricLabel"] { font-size:0.73rem !important; text-transform:uppercase; letter-spacing:.05em; }
    hr { margin:.3rem 0 .8rem 0 !important; }
</style>
""", unsafe_allow_html=True)

with st.spinner("Loading market data…"):
    df = fetch_market_data()
    vix_hist = fetch_vix_history()

now = datetime.now(timezone.utc)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.caption(f"Synced: {now.strftime('%b %d · %H:%M UTC')}")
    if st.button("🔄 Refresh", use_container_width=True, type="primary"):
        st.cache_data.clear(); st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📊 Daily Market Trends")
st.caption(f"Broad tape — indices, sectors, macro · {now.strftime('%A %B %d %Y — %H:%M UTC')}")
st.divider()

if df.empty:
    st.error("No market data loaded. Check your internet connection.")
    st.stop()

# ─────────────────────────────────────────────
# KPI STRIP — INDICES
# ─────────────────────────────────────────────
idx_df = df[df['Category'] == 'Index']
cols = st.columns(len(idx_df))
for col, (_, row) in zip(cols, idx_df.iterrows()):
    delta = f"{row['1D %']:+.2f}%" if row['1D %'] is not None else None
    col.metric(
        row['Name'],
        f"${row['Price']:.2f}" if row['Symbol'] != '^VIX' else f"{row['Price']:.1f}",
        delta=delta,
        delta_color="inverse" if row['Symbol'] == '^VIX' else "normal",
    )

st.divider()

# ─────────────────────────────────────────────
# ROW 1: 1D MOVERS + VIX CHART
# ─────────────────────────────────────────────
col_mv, col_vix = st.columns([1.5, 1])

with col_mv:
    st.subheader("⚡ Today's 1D Movers — All Assets")
    df_mv = df.dropna(subset=['1D %']).sort_values('1D %')
    fig = go.Figure(go.Bar(
        x=df_mv['1D %'], y=df_mv['Name'], orientation='h',
        marker_color=['#ff4444' if v < 0 else '#00cc88' for v in df_mv['1D %']],
        text=[f"{v:+.2f}%" for v in df_mv['1D %']], textposition='outside',
        customdata=df_mv[['Category','Price']].values,
        hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>$%{customdata[1]:.2f} · %{x:+.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template='plotly_dark', title='1-Day % Change',
        xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1),
        margin=dict(l=10, r=80, t=40, b=10), height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_vix:
    st.subheader("😨 VIX — Fear Gauge (90D)")
    if not vix_hist.empty:
        vix_now = vix_hist.iloc[-1]
        regime_color = '#ff2222' if vix_now > 25 else ('#ffaa00' if vix_now > 18 else '#00cc88')
        fig_v = go.Figure()
        fig_v.add_hrect(y0=25, y1=vix_hist.max()+2, fillcolor='rgba(255,34,34,0.08)', line_width=0)
        fig_v.add_hrect(y0=18, y1=25,               fillcolor='rgba(255,170,0,0.08)',  line_width=0)
        fig_v.add_hrect(y0=0,  y1=18,               fillcolor='rgba(0,204,136,0.06)', line_width=0)
        fig_v.add_trace(go.Scatter(
            x=vix_hist.index, y=vix_hist.values, mode='lines',
            line=dict(color=regime_color, width=2),
            fill='tozeroy', fillcolor='rgba(100,100,255,0.08)',
        ))
        fig_v.add_hline(y=25, line_dash='dot', line_color='#ff4444', opacity=0.6,
                        annotation_text="Risk-Off >25", annotation_position="right")
        fig_v.add_hline(y=18, line_dash='dot', line_color='#ffaa00', opacity=0.6,
                        annotation_text="Elevated >18", annotation_position="right")
        regime = "🔴 RISK-OFF" if vix_now > 25 else ("⚠️ ELEVATED" if vix_now > 18 else "🟢 RISK-ON")
        fig_v.update_layout(
            template='plotly_dark',
            title=f'VIX = {vix_now:.1f}  ·  {regime}',
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10), height=500,
        )
        st.plotly_chart(fig_v, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 2: SECTOR ROTATION HEATMAP
# ─────────────────────────────────────────────
st.subheader("🌡️ Sector Rotation Heatmap")
sec_df = df[df['Category'] == 'Sector ETF'].copy()
if not sec_df.empty:
    periods = ['1D %', '1W %', '1M %', '3M %']
    heat_data = sec_df.set_index('Name')[periods].T
    fig_h = go.Figure(go.Heatmap(
        z=heat_data.values,
        x=heat_data.columns.tolist(),
        y=['1 Day', '1 Week', '1 Month', '3 Months'],
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{v:+.1f}%" if v is not None and not np.isnan(v) else "—"
               for v in row] for row in heat_data.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title='% Chg'),
        hoverongaps=False,
    ))
    fig_h.update_layout(
        template='plotly_dark',
        title='Sector ETF % Returns — Rolling Windows',
        margin=dict(l=10, r=10, t=50, b=10), height=250,
        xaxis=dict(side='top'),
    )
    st.plotly_chart(fig_h, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 3: MACRO DASHBOARD + BREADTH
# ─────────────────────────────────────────────
col_mac, col_bread = st.columns(2)

with col_mac:
    st.subheader("🌐 Macro Pulse")
    mac_df = df[df['Category'] == 'Macro'].dropna(subset=['1D %']).sort_values('1D %')
    fig_mac = go.Figure(go.Bar(
        x=mac_df['1D %'], y=mac_df['Name'], orientation='h',
        marker_color=['#ff4444' if v < 0 else '#00cc88' for v in mac_df['1D %']],
        text=[f"{v:+.2f}%" for v in mac_df['1D %']], textposition='outside',
        customdata=mac_df['Price'].values,
        hovertemplate="<b>%{y}</b><br>$%{customdata:.2f} · %{x:+.2f}%<extra></extra>",
    ))
    fig_mac.update_layout(
        template='plotly_dark', title='Macro ETF 1D Change',
        xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1),
        margin=dict(l=10, r=80, t=40, b=10), height=350,
    )
    st.plotly_chart(fig_mac, use_container_width=True)

with col_bread:
    st.subheader("📡 Market Breadth — % Above Moving Averages")
    ab20 = df.dropna(subset=['vs 20D MA'])
    ab50 = df.dropna(subset=['vs 50D MA'])
    above20 = (ab20['vs 20D MA'] > 0).mean() * 100
    above50 = (ab50['vs 50D MA'] > 0).mean() * 100
    ma_sectors = df[df['Category']=='Sector ETF'].dropna(subset=['vs 20D MA','vs 50D MA'])

    fig_br = go.Figure()
    fig_br.add_trace(go.Bar(
        name='vs 20D MA', x=ma_sectors['vs 20D MA'], y=ma_sectors['Name'],
        orientation='h',
        marker_color=['#00cc88' if v >= 0 else '#ff4444' for v in ma_sectors['vs 20D MA']],
        opacity=0.85,
    ))
    fig_br.add_trace(go.Scatter(
        name='vs 50D MA', x=ma_sectors['vs 50D MA'], y=ma_sectors['Name'],
        mode='markers', marker=dict(symbol='diamond', size=10, color='#ffffff'),
    ))
    fig_br.add_vline(x=0, line_color='white', opacity=0.3)
    fig_br.update_layout(
        template='plotly_dark',
        title=f'Sectors vs MAs  ·  {above20:.0f}% assets above 20D  ·  {above50:.0f}% above 50D',
        barmode='overlay', margin=dict(l=10, r=10, t=50, b=10), height=350,
        legend=dict(orientation='h', y=1.1),
    )
    st.plotly_chart(fig_br, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# ROW 4: RELATIVE PERFORMANCE CHART
# ─────────────────────────────────────────────
st.subheader("📈 Relative Performance — Rebased to 100 (90D)")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_rebased(syms: list) -> pd.DataFrame:
    try:
        raw = yf.download(syms, period='3mo', auto_adjust=True,
                          progress=False, group_by='ticker')
        if isinstance(raw.columns, pd.MultiIndex):
            try:    closes = raw.xs('Close', axis=1, level=1)
            except: closes = raw['Close']
        else:
            closes = raw['Close'] if 'Close' in raw.columns else raw
        closes = closes.dropna(how='all')
        return (closes / closes.iloc[0] * 100).round(3)
    except Exception:
        return pd.DataFrame()

perf_syms = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'USO']
perf = fetch_rebased(perf_syms)
if not perf.empty:
    fig_rel = go.Figure()
    colors = ['#00ccff','#ff9900','#cc44ff','#ffdd44','#44ffcc','#ff4444']
    labels = {'SPY':'S&P 500','QQQ':'Nasdaq','IWM':'Russell 2k','GLD':'Gold','TLT':'20Y Bond','USO':'Oil'}
    for sym, color in zip(perf_syms, colors):
        if sym in perf.columns:
            fig_rel.add_trace(go.Scatter(
                x=perf.index, y=perf[sym], mode='lines', name=labels.get(sym, sym),
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{labels.get(sym,sym)}</b>: %{{y:.1f}}<extra></extra>",
            ))
    fig_rel.add_hline(y=100, line_dash='dot', line_color='white', opacity=0.3)
    fig_rel.update_layout(
        template='plotly_dark', title='Rebased 90D — SPY vs QQQ vs IWM vs Gold vs Bonds vs Oil',
        margin=dict(l=10, r=10, t=50, b=10), height=360,
        legend=dict(orientation='h', y=1.05),
        hovermode='x unified',
    )
    st.plotly_chart(fig_rel, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────
# FULL DATA TABLE
# ─────────────────────────────────────────────
st.subheader("🗃️ Full Market Ledger")
display_cols = ['Name','Category','Price','1D %','1W %','1M %','3M %','vs 20D MA','vs 50D MA','Vol Spike','Sparkline']
st.dataframe(
    df[[c for c in display_cols if c in df.columns]].sort_values('1D %', ascending=False),
    column_config={
        'Price':      st.column_config.NumberColumn('Price',      format='$%.2f'),
        '1D %':       st.column_config.NumberColumn('1D %',       format='%+.2f%%'),
        '1W %':       st.column_config.NumberColumn('1W %',       format='%+.2f%%'),
        '1M %':       st.column_config.NumberColumn('1M %',       format='%+.2f%%'),
        '3M %':       st.column_config.NumberColumn('3M %',       format='%+.2f%%'),
        'vs 20D MA':  st.column_config.NumberColumn('vs 20D MA',  format='%+.2f%%'),
        'vs 50D MA':  st.column_config.NumberColumn('vs 50D MA',  format='%+.2f%%'),
        'Vol Spike':  st.column_config.NumberColumn('Vol Spike',  format='%.2f×'),
        'Sparkline':  st.column_config.LineChartColumn('30D Trend'),
    },
    hide_index=True, use_container_width=True, height=420,
)
