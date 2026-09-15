"""
shared.py — Constants and shared utilities used across all pages.
Edit the watchlist and impact map here — both pages pick them up automatically.
"""

import os

# ─────────────────────────────────────────────
# LLM ENDPOINT
# ─────────────────────────────────────────────
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234")

# ─────────────────────────────────────────────
# WATCHLIST  (Jane Street conviction names, Sep 2026)
# ─────────────────────────────────────────────
# Rules applied:
#   • No REITs under any circumstances
#   • Minimum ~$2B market cap for liquidity
#   • Each name must have a clear, defensible thesis tied to the theme
#   • Prefer pure-plays over diversified conglomerates where possible
# ─────────────────────────────────────────────
WATCHLIST = {

    # GLP-1 commercial arms race (LLY ~60% market share) + agentic AI
    # compressing drug discovery timelines from years to months.
    # VKTX = high-upside Phase-2 challenger. RXRX = AI-native platform.
    'Pharma & AI Drug Discovery': ['LLY', 'NVO', 'VKTX', 'RXRX'],

    # The defining macro trade of 2026: hyperscalers need always-on,
    # carbon-free power. CEG & VST hold nuclear fleets with signed PPAs.
    # TLN (Talen Energy) = pure-play nuclear cashflow. GEV = turbines + grid OS.
    'Nuclear & AI Power':         ['CEG', 'VST', 'TLN', 'GEV'],

    # Picks-and-shovels of the AI power buildout. PWR & MTZ build the
    # physical transmission lines and data-center campuses. ETN & POWL
    # supply the switchgear and power distribution hardware.
    'Grid Infrastructure':        ['PWR', 'ETN', 'POWL', 'MTZ'],

    # AI data centers run on photons, not copper. COHR dominates
    # 800G/1.6T transceivers. CIEN (Ciena) owns the long-haul optical
    # backbone. LITE = specialty photonics exposure.
    'Optical Interconnect':       ['COHR', 'CIEN', 'LITE'],

    # AI is the dual-edged sword: expands the attack surface while
    # enabling faster threat response. CRWD & PANW = platform consolidators.
    # ZS = cloud-native zero-trust. S = AI-native endpoint.
    'Cybersecurity':              ['CRWD', 'PANW', 'ZS', 'S'],

    # Physical climate events drive infrastructure demand.
    # VRT = #1 liquid cooling / thermal mgmt for AI chips (direct climate link).
    # XYL = water management. FSLR = domestic solar manufacturing.
    # GTLS (Chart Industries) = industrial gas / LNG infrastructure.
    'Climate Resilience':         ['VRT', 'XYL', 'FSLR', 'GTLS'],
}

ALL_TICKERS   = [t for ts in WATCHLIST.values() for t in ts]
TICKER_TO_SECTOR = {t: s for s, ts in WATCHLIST.items() for t in ts}
BENCH_TICKER  = 'SPY'
VIX_TICKER    = '^VIX'

# ─────────────────────────────────────────────
# DISASTER → MARKET IMPACT
# ─────────────────────────────────────────────
COLOR_MAP = {
    "Earthquake": "#FF9900",
    "Flood":      "#3399FF",
    "Cyclone":    "#00FFCC",
    "Volcano":    "#FF3333",
    "Drought":    "#FFD700",
    "Tsunami":    "#CC66FF",
    "Other":      "#888888",
}

IMPACT_MAP = {
    "Earthquake": "⚠️ Infrastructure Risk — Tailwind for PWR, ETN (grid rebuild demand)",
    "Flood":      "🌊 Supply Chain Stress — Bullish XYL (water mgmt), pressure on COHR logistics",
    "Cyclone":    "🌀 Grid Outage Risk — Tailwind for PWR; monitor FSLR (solar panel damage)",
    "Volcano":    "🌋 Airspace/Logistics Disruption — Watch optical supply chains (COHR, LITE)",
    "Drought":    "🏜️ Ag & Water Stress — Bullish XYL, GTLS; elevated utility infrastructure demand",
    "Tsunami":    "🌊 Coastal Infrastructure Collapse — Broad infra demand; ETN, PWR upside",
    "Other":      "📡 Monitor for localized disruption signals",
}

# Climate impact weight matrix  [Pharma, Nuclear/AI Power, Grid, Optical, Cyber, Climate]
# Scale: -2 (headwind) → +2 (tailwind)
CLIMATE_IMPACT_MATRIX = {
    'Earthquake': {'weights': [ 0.0,  0.3,  1.5, -0.5,  0.2,  0.5]},
    'Flood':      {'weights': [-0.2,  0.2,  0.8, -0.8,  0.1,  1.8]},
    'Cyclone':    {'weights': [-0.1,  0.5,  1.2, -0.6,  0.3,  1.5]},
    'Volcano':    {'weights': [ 0.0,  0.1,  0.2, -1.2,  0.0,  0.3]},
    'Drought':    {'weights': [ 0.1,  0.8,  0.4,  0.0,  0.0,  1.6]},
    'Tsunami':    {'weights': [-0.3,  0.4,  1.0, -0.8,  0.2,  0.8]},
    'Other':      {'weights': [ 0.0,  0.0,  0.1,  0.0,  0.0,  0.1]},
}
SECTOR_ORDER = list(WATCHLIST.keys())

# ─────────────────────────────────────────────
# NAV HELPER  (call once at top of each page)
# ─────────────────────────────────────────────
def render_nav(current: str) -> None:
    """Renders a top navigation bar and sidebar links between pages."""
    import streamlit as st

    pages = {
        "🌍 Climate Dashboard": "app.py",
        "📐 Quant Terminal":    "pages/2_📐_Quant_Terminal.py",
    }

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🛰️ Navigation")
        for label, path in pages.items():
            st.page_link(path, label=label)
        st.divider()
