"""Design system for the dashboard: dark 'neon purple' theme, reusable components and the chart theme."""

import html

import plotly.graph_objects as go
import streamlit as st

# Brand: near-black canvas, glassy cards, violet -> magenta accents
CANVAS, SURFACE, SURFACE_2 = "#0B0913", "#16121F", "#1E1830"
INK, MUTED, BORDER = "#F4F1FF", "#A6A0C0", "rgba(167,139,250,0.18)"
VIOLET, MAGENTA, LAVENDER = "#8B5CF6", "#D946EF", "#C4B5FD"
TEAL = VIOLET          # primary accent (name kept for the page modules)
AMBER = "#F5B544"

# Validated categorical palette for the dark surface (fixed order, never cycled).
# Checked with the dataviz validator: lightness band, chroma, contrast and colour-blind separation.
SERIES = ["#8B5CF6", "#EC4899", "#0891B2", "#D97706", "#059669", "#EA580C", "#3B82F6", "#DC2626"]
OTHER = "#4A4560"
# Sequential ramp (dark -> bright violet) for heatmaps on the dark surface
BLUES = ["#1B1629", "#2C1F52", "#44287F", "#5E33AE", "#7C45DB", "#A078F2", "#D9CCFF"]
# Each community keeps one colour everywhere
COMMUNITY_COLOR = {"datascience": SERIES[0], "ai": SERIES[1]}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{
    background: radial-gradient(1200px 600px at 85% -10%, rgba(139,92,246,0.18), transparent 60%),
                radial-gradient(900px 500px at -10% 20%, rgba(217,70,239,0.10), transparent 60%), {CANVAS};
}}
header[data-testid="stHeader"] {{ background: rgba(11,9,19,0.85); backdrop-filter: blur(10px);
    border-bottom: 1px solid {BORDER}; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.35rem !important; font-weight: 700 !important; margin-top: 1.6rem !important; }}
a {{ color: {LAVENDER} !important; }}

/* hero */
.lca-hero {{ position: relative; overflow: hidden; border-radius: 22px; padding: 38px 40px; color: #fff; margin: 4px 0 22px;
    background: linear-gradient(135deg, #1A1030 0%, #3B1A73 55%, #7E22CE 100%);
    border: 1px solid rgba(196,181,253,0.25); box-shadow: 0 20px 60px rgba(124,58,237,0.25); }}
.lca-hero::after {{ content: ""; position: absolute; right: -80px; top: -80px; width: 320px; height: 320px; border-radius: 50%;
    border: 3px solid rgba(217,70,239,0.55); box-shadow: 0 0 60px rgba(217,70,239,0.45), inset 0 0 40px rgba(217,70,239,0.25); }}
.lca-hero .eyebrow {{ color: {LAVENDER}; font-weight: 700; font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; }}
.lca-hero h1 {{ color: #fff; font-size: 2.35rem; font-weight: 800; margin: 8px 0 6px; line-height: 1.15; max-width: 900px; }}
.lca-hero p {{ color: #DDD6FE; font-size: 1.02rem; max-width: 820px; margin: 0; }}
.lca-hero .chips {{ margin-top: 16px; }}
.lca-chip {{ display: inline-block; background: rgba(255,255,255,0.10); color: #fff; border-radius: 999px;
             padding: 5px 13px; font-size: .8rem; margin: 0 6px 6px 0; border: 1px solid rgba(255,255,255,0.15); }}

/* page header */
.lca-page-title {{ font-size: 2rem; font-weight: 800; margin: 6px 0 2px;
    background: linear-gradient(90deg, #FFFFFF 0%, {LAVENDER} 60%, {MAGENTA} 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent; }}
.lca-page-sub {{ color: {MUTED}; font-size: 1rem; margin-bottom: 18px; }}

/* glass cards */
.lca-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 6px 0 10px; }}
.lca-kpi, .lca-card, .lca-step {{ background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border: 1px solid {BORDER}; border-radius: 16px; backdrop-filter: blur(6px); }}
.lca-kpi {{ padding: 16px 18px; }}
.lca-kpi .label {{ color: {MUTED}; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }}
.lca-kpi .value {{ color: {INK}; font-size: 1.75rem; font-weight: 800; margin-top: 4px; line-height: 1.1; }}
.lca-kpi .note {{ color: {MUTED}; font-size: .8rem; margin-top: 4px; }}
.lca-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin: 8px 0; }}
.lca-card {{ padding: 18px 20px; transition: border-color .2s; }}
.lca-card:hover {{ border-color: rgba(217,70,239,0.5); }}
.lca-card .icon {{ font-size: 1.4rem; }}
.lca-card .title {{ color: {INK}; font-weight: 700; font-size: 1rem; margin: 6px 0 4px; }}
.lca-card .big {{ font-weight: 800; font-size: 1.6rem;
    background: linear-gradient(90deg, {LAVENDER}, {MAGENTA}); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.lca-card .body {{ color: {MUTED}; font-size: .9rem; line-height: 1.45; }}

/* pipeline steps */
.lca-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
.lca-step {{ padding: 14px; }}
.lca-step .n {{ display: inline-flex; width: 26px; height: 26px; border-radius: 50%; color: #fff; font-weight: 700; font-size: .8rem;
    align-items: center; justify-content: center; background: linear-gradient(135deg, {VIOLET}, {MAGENTA});
    box-shadow: 0 0 14px rgba(217,70,239,0.45); }}
.lca-step .t {{ font-weight: 700; color: {INK}; margin-top: 8px; font-size: .92rem; }}
.lca-step .d {{ color: {MUTED}; font-size: .8rem; }}

/* words, callouts, footer */
.lca-word {{ display: inline-block; background: rgba(139,92,246,0.16); color: {LAVENDER}; border-radius: 8px; padding: 3px 10px;
             margin: 0 6px 6px 0; font-size: .85rem; font-weight: 600; border: 1px solid rgba(139,92,246,0.3); }}
.lca-callout {{ background: rgba(139,92,246,0.12); border: 1px solid rgba(139,92,246,0.3); border-radius: 14px;
    padding: 14px 18px; color: {INK}; margin: 8px 0 14px; }}
.lca-footer {{ color: {MUTED}; font-size: .8rem; text-align: center; margin-top: 40px; padding-top: 16px; border-top: 1px solid {BORDER}; }}
.lca-brand {{ font-weight: 800; font-size: 1.15rem; color: {INK}; }}
.lca-brand span {{ background: linear-gradient(90deg, {LAVENDER}, {MAGENTA}); -webkit-background-clip: text; background-clip: text; color: transparent; }}

/* Streamlit widgets */
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"], div[data-testid="stForm"], iframe {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 6px;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button, div[data-testid="stFormSubmitButton"] button {{
    background: linear-gradient(90deg, {VIOLET}, {MAGENTA}) !important; border: 0 !important; color: #fff !important;
    border-radius: 12px !important; font-weight: 700 !important; box-shadow: 0 8px 24px rgba(217,70,239,0.35);
}}
div[data-testid="stExpander"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="lca-page-title">{esc(title)}</div>'
                + (f'<div class="lca-page-sub">{esc(subtitle)}</div>' if subtitle else ""), unsafe_allow_html=True)


def hero(eyebrow: str, title: str, text: str, chips=()):
    chip_html = "".join(f'<span class="lca-chip">{esc(c)}</span>' for c in chips)
    st.markdown(f'<div class="lca-hero"><div class="eyebrow">{esc(eyebrow)}</div><h1>{esc(title)}</h1>'
                f'<p>{esc(text)}</p><div class="chips">{chip_html}</div></div>', unsafe_allow_html=True)


def kpis(items):
    """items: list of (label, value, note)"""
    cells = "".join(f'<div class="lca-kpi"><div class="label">{esc(l)}</div><div class="value">{esc(v)}</div>'
                    f'<div class="note">{esc(n)}</div></div>' for l, v, n in items)
    st.markdown(f'<div class="lca-kpis">{cells}</div>', unsafe_allow_html=True)


def cards(items):
    """items: list of (icon, title, big, body)"""
    cells = "".join(f'<div class="lca-card"><div class="icon">{i}</div><div class="title">{esc(t)}</div>'
                    f'<div class="big">{esc(b)}</div><div class="body">{esc(body)}</div></div>'
                    for i, t, b, body in items)
    st.markdown(f'<div class="lca-cards">{cells}</div>', unsafe_allow_html=True)


def steps(items):
    cells = "".join(f'<div class="lca-step"><span class="n">{k}</span><div class="t">{esc(t)}</div>'
                    f'<div class="d">{esc(d)}</div></div>' for k, (t, d) in enumerate(items, 1))
    st.markdown(f'<div class="lca-steps">{cells}</div>', unsafe_allow_html=True)


def words(ws):
    st.markdown("".join(f'<span class="lca-word">{esc(w)}</span>' for w in ws), unsafe_allow_html=True)


def callout(text: str):
    st.markdown(f'<div class="lca-callout">{esc(text)}</div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="lca-footer">Learning Community Analytics · NLP topics and knowledge-diffusion networks · '
                'Data: Stack Exchange (CC BY-SA) · Capstone project by Prateek Chauhan</div>', unsafe_allow_html=True)


def chart(fig: go.Figure, height=None, title=None):
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=INK),
        title=dict(text=title, x=0.01, font=dict(size=15, color=INK)) if title else None,
        margin=dict(l=10, r=10, t=48 if title else 16, b=10), legend_title_text="",
        hoverlabel=dict(font_size=13, bgcolor=SURFACE_2, bordercolor=VIOLET, font_color=INK),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, legend=dict(font=dict(color=INK)),
    )
    # automargin makes room for long tick labels such as topic names.
    fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.12)", tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.15)",
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED), automargin=True)
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})


def fmt_pct(x, d=0):
    return f"{100 * x:.{d}f}%"
