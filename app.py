"""
app.py — Navigation shell.
Handles auth once. All page content lives in pages/.
"""
import streamlit as st
from auth import require_login

st.set_page_config(
    page_title="Finance Terminal",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

require_login()

pg = st.navigation([
    st.Page("pages/1_Climate_Dashboard.py", title="Climate Dashboard", icon="🌍"),
    st.Page("pages/2_Quant_Terminal.py",    title="Quant Terminal",    icon="📐"),
    st.Page("pages/3_Daily_Trends.py",      title="Daily Trends",      icon="📊"),
    st.Page("pages/4_AI_Research.py",       title="AI Research",       icon="🧠"),
    st.Page("pages/5_Commodities.py",       title="Commodities",       icon="🛢️"),
])
pg.run()
