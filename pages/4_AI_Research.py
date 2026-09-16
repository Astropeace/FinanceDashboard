"""
4_AI_Research.py — AI Research Assistant
─────────────────────────────────────────
Free-form chat interface + one-click report templates.
Context is automatically populated with today's live market data.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timezone

from shared import LLM_BASE_URL, WATCHLIST, ALL_TICKERS, TICKER_TO_SECTOR

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .user-bubble {
        background: #1e3a5f; border-radius: 12px 12px 4px 12px;
        padding: .7rem 1rem; margin: .4rem 0 .4rem 15%;
        font-size: .9rem;
    }
    .ai-bubble {
        background: #0d1f0d; border: 1px solid #1f4d1f; border-radius: 12px 12px 12px 4px;
        padding: .7rem 1rem; margin: .4rem 15% .4rem 0;
        font-size: .9rem;
    }
    .report-box {
        background: #111827; border-left: 4px solid #00ccff;
        border-radius: 6px; padding: 1rem 1.2rem; margin: .5rem 0;
        font-size: .88rem;
    }
    hr { margin: .3rem 0 .8rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": "user"|"assistant", "content": str}
if "market_context" not in st.session_state:
    st.session_state.market_context = ""

# ─────────────────────────────────────────────
# MARKET CONTEXT BUILDER
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def build_market_context() -> str:
    """Builds a compact market snapshot string for the AI system prompt."""
    lines = [f"Today: {datetime.now(timezone.utc).strftime('%A %B %d %Y %H:%M UTC')}", ""]

    # Macro snapshot
    macro_syms = {'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', '^VIX': 'VIX',
                  'GLD': 'Gold', 'TLT': '20Y Bond', 'USO': 'Oil', 'UUP': 'USD'}
    macro_lines = []
    for sym, name in macro_syms.items():
        try:
            hist = yf.Ticker(sym).history(period='5d')
            if len(hist) >= 2:
                c  = hist['Close'].iloc[-1]
                d1 = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
                macro_lines.append(f"  {name}: {c:.2f} ({d1:+.2f}%)")
        except Exception:
            pass
    lines.append("MACRO SNAPSHOT:")
    lines.extend(macro_lines)
    lines.append("")

    # Watchlist
    lines.append("THEMATIC WATCHLIST (today's moves):")
    for sector, tickers in WATCHLIST.items():
        parts = []
        for sym in tickers:
            try:
                hist = yf.Ticker(sym).history(period='5d')
                if len(hist) >= 2:
                    d1 = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
                    parts.append(f"{sym} {d1:+.2f}%")
            except Exception:
                pass
        if parts:
            lines.append(f"  [{sector}]: {', '.join(parts)}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────
def call_llm(messages: list, model: str, temperature: float, endpoint: str) -> str:
    try:
        resp = requests.post(
            f"{endpoint}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }),
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        return f"⚠️ LLM Error ({resp.status_code}): {resp.text[:300]}"
    except requests.exceptions.Timeout:
        return "⚠️ Timeout — model may be loading. Try again in 30 seconds."
    except Exception as e:
        return f"⚠️ Connection error: {e}"

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ AI Config")
    llm_endpoint = st.text_input(
        "LLM Endpoint",
        value=st.session_state.get("llm_endpoint", LLM_BASE_URL),
        placeholder="https://xxxx.trycloudflare.com",
    )
    st.session_state["llm_endpoint"] = llm_endpoint
    model_id    = st.text_input("Model ID", value="qwen2.5-coder-14b-instruct-mlx")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.05)
    st.divider()
    st.subheader("📋 Report Templates")
    st.caption("Click to instantly generate a structured report:")
    run_morning  = st.button("☀️ Morning Alpha Brief",    use_container_width=True)
    run_risk     = st.button("🚨 Risk Report",             use_container_width=True)
    run_sector   = st.button("🌡️ Sector Rotation Report", use_container_width=True)
    run_pair     = st.button("⚖️ Pair Trade Ideas",        use_container_width=True)
    run_macro    = st.button("🌐 Macro Regime Report",     use_container_width=True)
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if st.button("🔄 Refresh Market Data", use_container_width=True, type="primary"):
        st.cache_data.clear(); st.session_state.market_context = ""; st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🧠 AI Research Assistant")
st.caption("Ask anything about the market, or click a report template in the sidebar.")
st.divider()

# Build context once per session
if not st.session_state.market_context:
    with st.spinner("Building market context…"):
        st.session_state.market_context = build_market_context()

ctx = st.session_state.market_context

SYSTEM_PROMPT = f"""You are a senior quantitative analyst with deep expertise in equities, macro, and derivatives. You have access to today's live market data shown below. Answer concisely and precisely — use numbers, percentages, and specific ticker names. Never give generic disclaimers. Structure your responses with clear headers when producing reports.

--- LIVE MARKET CONTEXT ---
{ctx}
--- END CONTEXT ---"""

# ─────────────────────────────────────────────
# REPORT TEMPLATE TRIGGERS
# ─────────────────────────────────────────────
REPORT_PROMPTS = {
    "morning":  "Generate a complete morning alpha brief with these exact sections:\n1. 🏛️ MACRO REGIME — characterize today's risk environment (VIX level, SPY move, bond direction)\n2. 🔥 TOP 3 OPPORTUNITIES — specific tickers with entry thesis, each in 2 sentences\n3. 🚨 TOP 2 RISKS — what could go wrong today, specific catalysts\n4. 📊 SECTOR LEADERS / LAGGARDS — which themes are working, which aren't\n5. 📌 WATCH LIST — 3 names to monitor today with price levels\nUse the live data provided. Be specific.",

    "risk":     "Generate a structured risk report:\n1. 🌡️ VOLATILITY REGIME — VIX interpretation, expected move for SPY today\n2. ⚠️ CONCENTRATION RISKS — which sectors/names are most exposed to downside\n3. 🔗 CORRELATION BREAKDOWN — are normal correlations holding? (equities/bonds/gold)\n4. 🌍 MACRO TAIL RISKS — top 3 external risks that could move markets this week\n5. 🛡️ HEDGING IDEAS — specific ETF or options strategies for current regime\nBe direct and actionable.",

    "sector":   "Generate a sector rotation analysis:\n1. 📈 MOMENTUM LEADERS — which sectors are showing strongest price momentum (1D and 1M)\n2. 📉 ROTATION CANDIDATES — sectors that look stretched or due for mean-reversion\n3. 🏆 BEST IN CLASS — single best ticker in each leading sector and why\n4. 🔄 ROTATION THESIS — what macro/earnings catalyst would rotate money between sectors\n5. ⚡ TRADE IDEA — one specific sector pair trade (long X / short Y) with rationale\nBase everything on the live data.",

    "pair":     "Identify 5 pair trade opportunities from our watchlist:\nFor each pair, provide:\n- Long / Short: specific tickers\n- Sector: same-sector or cross-sector\n- Thesis: one-sentence fundamental or technical basis\n- Catalyst: what triggers the convergence\n- Risk: what invalidates the trade\n- Time Horizon: days/weeks/months\nFocus on pairs within the same themes (Nuclear Power, Grid, Optical, Cyber, Pharma, Climate). Be concrete.",

    "macro":    "Generate a macro regime assessment:\n1. 📊 CURRENT REGIME — Risk-On / Risk-Off / Transitional, with evidence\n2. 💵 DOLLAR DYNAMICS — USD direction and impact on equities vs commodities\n3. 📈 RATES ENVIRONMENT — What bond yields/prices signal for equity multiples\n4. 🛢️ COMMODITY PULSE — Oil and gold moves and what they signal\n5. 🌍 GLOBAL RISK — any EM/DM divergences worth watching\n6. 💡 POSITIONING RECOMMENDATION — how to position a long/short equity book in this regime\nBe quantitative and specific.",
}

def trigger_report(key: str):
    prompt = REPORT_PROMPTS[key]
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt}]
    with st.spinner("Generating report…"):
        reply = call_llm(messages, model_id, temperature, llm_endpoint)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

if run_morning: trigger_report("morning")
if run_risk:    trigger_report("risk")
if run_sector:  trigger_report("sector")
if run_pair:    trigger_report("pair")
if run_macro:   trigger_report("macro")

# ─────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
<div style="text-align:center; color:#555; padding:3rem 0;">
    <h3>👋 Start a conversation</h3>
    <p>Ask anything about today's market, or use a report template from the sidebar.</p>
    <p style="font-size:.85rem; margin-top:1rem;">
    Examples:<br>
    <em>"What's the best nuclear energy trade right now?"</em><br>
    <em>"Why is QQQ underperforming IWM today?"</em><br>
    <em>"Summarize the macro risks for this week"</em><br>
    <em>"Give me 3 high-conviction ideas in cybersecurity"</em>
    </p>
</div>
""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                # Shorten template prompts for display
                display = msg["content"]
                if len(display) > 120:
                    display = display.split('\n')[0][:100] + "…"
                st.markdown(f'<div class="user-bubble">🧑 {display}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_input(
            "Ask the AI anything…",
            placeholder="e.g. 'What's the best trade in AI power infrastructure today?'",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

if submitted and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    # Build messages with full context in system turn
    messages = [{"role": "user", "content": SYSTEM_PROMPT}]
    # Add conversation history (last 8 turns to avoid token overflow)
    for h in st.session_state.chat_history[-8:]:
        messages.append({"role": h["role"], "content": h["content"]})
    with st.spinner("Thinking…"):
        reply = call_llm(messages, model_id, temperature, llm_endpoint)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()

# ─────────────────────────────────────────────
# MARKET CONTEXT EXPANDER
# ─────────────────────────────────────────────
with st.expander("📡 View live market context being sent to AI"):
    st.code(ctx, language="text")
