"""
auth.py — Shared authentication gate for both dashboards.

Reads credentials from st.secrets (locally: .streamlit/secrets.toml,
on Streamlit Cloud: injected via the Secrets UI).

Usage:
    from auth import require_login
    require_login()   # call at the top of any page — halts if not authenticated
"""

import streamlit as st
import bcrypt
import hmac


def _check_password(entered: str, hashed: str) -> bool:
    """Constant-time bcrypt comparison to prevent timing attacks."""
    try:
        return bcrypt.checkpw(entered.encode(), hashed.encode())
    except Exception:
        return False


def require_login() -> None:
    """
    Renders a password gate. If not authenticated the rest of the
    page is blocked. Session persists until the browser tab is closed.
    """
    if st.session_state.get("authenticated"):
        return  # already logged in this session

    # Pull config from secrets
    try:
        title    = st.secrets.get("app_title",    "🛰️ Quant Terminal")
        password_hash = st.secrets["password_hash"]
    except (KeyError, FileNotFoundError):
        st.error(
            "🔐 `password_hash` not found in `st.secrets`.\n\n"
            "Add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets:\n"
            "```\npassword_hash = \"$2b$12$...\"\n```"
        )
        st.stop()

    # Login form
    st.markdown(f"## {title}")
    st.caption("Enter your access password to continue.")

    with st.form("login_form", clear_on_submit=True):
        entered = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Unlock →", use_container_width=True, type="primary")

    if submitted:
        if _check_password(entered, password_hash):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Try again.")

    # Block the rest of the page from rendering
    st.stop()
