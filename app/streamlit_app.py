"""
Learning Community Analytics: interactive dashboard.

Run from the project folder:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

st.set_page_config(page_title="Learning Community Analytics", page_icon="🎓", layout="wide")

import ui  # noqa: E402
from data import COMMUNITIES, available_sites, data_source  # noqa: E402
from views import about, assistant, compare, database, diffusion, home, network, topics  # noqa: E402

ui.inject_css()
SITES = available_sites()
if not SITES:
    st.error("No results found. Run  python src/run_pipeline.py  first.")
    st.stop()

# Community switcher: remembered in the URL (?community=ai) so links and refreshes keep it.
requested = st.query_params.get("community", SITES[0])
if "community" not in st.session_state:
    st.session_state.community = requested if requested in SITES else SITES[0]

brand, switch = st.columns([3, 2], vertical_alignment="center")
with brand:
    st.markdown(f"<div class='lca-brand'>🎓 Learning Community <span>Analytics</span></div>"
                f"<div style='color:{ui.MUTED};font-size:.85rem'>Data source: <b>{data_source()}</b></div>",
                unsafe_allow_html=True)
with switch:
    choice = st.segmented_control("Community", SITES, key="community", format_func=lambda s: COMMUNITIES[s],
                                  label_visibility="collapsed")
site = choice or SITES[0]
st.query_params["community"] = site

# One named function per page (Streamlit identifies pages by function name).
def home_page(): home.render(site)  # noqa: E704
def topics_page(): topics.render(site)
def network_page(): network.render(site)
def diffusion_page(): diffusion.render(site)
def compare_page(): compare.render(SITES)
def assistant_page(): assistant.render(site)
def database_page(): database.render(site)
def about_page(): about.render(site)


pages = [
    st.Page(home_page, title="Home", icon="🏠", default=True),
    st.Page(topics_page, title="Topics", icon="🧠", url_path="topics"),
    st.Page(network_page, title="Network", icon="🕸️", url_path="network"),
    st.Page(diffusion_page, title="Diffusion", icon="🌊", url_path="diffusion"),
    st.Page(compare_page, title="DS vs AI", icon="⚖️", url_path="compare"),
    st.Page(assistant_page, title="Question assistant", icon="🤖", url_path="assistant"),
    st.Page(database_page, title="Database", icon="🗄️", url_path="database"),
    st.Page(about_page, title="About", icon="ℹ️", url_path="about"),
]
st.navigation(pages, position="top").run()
ui.footer()
