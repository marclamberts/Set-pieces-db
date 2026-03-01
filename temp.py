import os
import pandas as pd
import numpy as np
import streamlit as st

# Optional interactive charts
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Set Pieces — Corners (Allsvenskan 2025)", layout="wide")

DEFAULT_FILE = "Allsvenskan - Corners 2025.xlsx"
DEFAULT_SHEET = "Sheet 1"

# ---------- UI / UX (platform-style) ----------
st.markdown(
    """
<style>
/* App background */
html, body, [class*="css"] {
  background: #f5f7fb;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}

/* Main container width + padding */
.block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 2rem; }

/* Sidebar: clean filter rail */
section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid #e9eef6;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* Hide Streamlit menu/footer vibe */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Page header */
.app-header {
  background: #ffffff;
  border: 1px solid #e9eef6;
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
  margin-bottom: 14px;
}
.app-title { font-size: 28px; font-weight: 780; letter-spacing: -0.02em; margin: 0; }
.app-sub { color: #6b7280; margin: 4px 0 0 0; font-size: 14px; }

/* Cards */
.card {
  background: #ffffff;
  border: 1px solid #e9eef6;
  border-radius: 18px;
  padding: 16px 16px 12px 16px;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
  margin-bottom: 12px;
}
.card-tight { padding: 12px 14px; }
.card h3 { margin: 0 0 10px 0; font-size: 16px; letter-spacing: -0.01em; }

/* KPI grid */
.kpi-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
.kpi {
  background: #ffffff;
  border: 1px solid #e9eef6;
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
}
.kpi .label { color: #6b7280; font-size: 12px; margin-bottom: 6px; }
.kpi .value { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; }
.kpi .hint { color: #94a3b8; font-size: 11px; margin-top: 4px; }

/* Tabs spacing */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background: #ffffff;
  border: 1px solid #e9eef6;
  border-radius: 12px;
  padding: 8px 12px;
}
.stTabs [aria-selected="true"] {
  border-color: #c7d2fe;
  box-shadow: 0 10px 25px rgba(99, 102, 241, 0.10);
}

/* Dataframe container */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid #e9eef6;
}

/* Buttons */
.stDownloadButton button, .stButton button {
  background-color: #111827 !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  border: none !important;
  padding: 0.55em 1.05em !important;
  box-shadow: 0 10px 22px rgba(17, 24, 39, 0.18);
}
.stDownloadButton button:hover, .stButton button:hover {
  background-color: #0b1220 !important;
}
.small-note { color: #6b7280; font-size: 12px; }
.hr { border: none; border-top: 1px solid #e9eef6; margin: 10px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Data ----------
@st.cache_data(show_spinner=False)
def load_data_from_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=DEFAULT_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    df["Minute_num"] = pd.to_numeric(df["Minute"], errors="coerce") if "Minute" in df.columns else np.nan
    df["Second_num"] = pd.to_numeric(df["Second"], errors="coerce") if "Second" in df.columns else np.nan

    shot_ts = df["shot_timestamp"].notna() if "shot_timestamp" in df.columns else pd.Series(False, index=df.index)
    shot_out = df["shot.outcome.name"].notna() if "shot.outcome.name" in df.columns else pd.Series(False, index=df.index)
    df["is_shot"] = shot_ts | shot_out

    if "shot.statsbomb_xg" in df.columns:
        df["xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce").fillna(0.0)
    else:
        df["xg"] = 0.0

    # Normalized keys used across UI
    df["team"] = df["pass_team_name"] if "pass_team_name" in df.columns else None
    df["match"] = df["Match"] if "Match" in df.columns else None
    df["taker"] = df["Taker"] if "Taker" in df.columns else None
    df["technique"] = df["pass.technique.name"] if "pass.technique.name" in df.columns else None
    df["height"] = df["pass.height.name"] if "pass.height.name" in df.columns else None
    df["shot_outcome"] = df["shot.outcome.name"] if "shot.outcome.name" in df.columns else None
    df["sp_outcome"] = df["SP_outcome"] if "SP_outcome" in df.columns else None

    return df


# ---------- Sidebar: filter rail ----------
with st.sidebar:
    st.markdown("### Data source")
    uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        data_file = uploaded
        st.success("Loaded uploaded file")
    else:
        if os.path.exists(DEFAULT_FILE):
            data_file = DEFAULT_FILE
            st.info(f"Using: {DEFAULT_FILE}")
        else:
            st.error(f"Place `{DEFAULT_FILE}` next to this script or upload it.")
            st.stop()

df = load_data_from_excel(data_file)

# Global filters (rail)
with st.sidebar:
    st.markdown("### Filter rail")
    st.caption("Narrow the dataset. Views on the right update instantly.")

    # Quick search (match/team/taker)
    q = st.text_input("Quick search", value="", placeholder="Type team, match or taker…")

    teams = sorted(pd.Series(df["team"]).dropna().unique().tolist())
    sel_teams = st.multiselect("Teams", teams, default=teams)

    df_team = df[df["team"].isin(sel_teams)] if sel_teams else df

    matches = sorted(pd.Series(df_team["match"]).dropna().unique().tolist())
    sel_matches = st.multiselect("Matches", matches, default=matches)

    df_match = df_team[df_team["match"].isin(sel_matches)] if sel_matches else df_team

    takers = sorted(pd.Series(df_match["taker"]).dropna().unique().tolist())
    sel_takers = st.multiselect("Takers", takers, default=takers)

    techniques = sorted(pd.Series(df_match["technique"]).dropna().unique().tolist())
    sel_techniques = st.multiselect("Technique", techniques, default=techniques)

    heights = sorted(pd.Series(df_match["height"]).dropna().unique().tolist())
    sel_heights = st.multiselect("Delivery height", heights, default=heights)

    # SAFE minute control
    minute_series = pd.to_numeric(df_match.get("Minute_num", pd.Series(dtype=float)), errors="coerce").dropna()
    minute_unique = np.sort(minute_series.unique()) if len(minute_series) else np.array([])

    minute_mode = "disabled"
    minute_range = None
    minute_single_value = None
    apply_single_minute = False

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    st.markdown("#### Time window")
    if len(minute_unique) == 0:
        st.caption("No minute values → disabled")
    elif len(minute_unique) == 1:
        minute_single_value = int(minute_unique[0])
        apply_single_minute = st.checkbox(f"Only minute {minute_single_value}", value=False)
        minute_mode = "single"
    else:
        minute_range = st.slider(
            "Minutes",
            min_value=int(minute_unique.min()),
            max_value=int(minute_unique.max()),
            value=(int(minute_unique.min()), int(minute_unique.max())),
        )
        minute_mode = "range"

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    only_shots = st.toggle("Only corners → shot", value=False)
    show_raw = st.toggle("Show raw table", value=True)

# ---------- Apply filters ----------
f = df.copy()

if sel_teams:
    f = f[f["team"].isin(sel_teams)]
if sel_matches:
    f = f[f["match"].isin(sel_matches)]
if sel_takers:
    f = f[f["taker"].isin(sel_takers)]
if sel_techniques:
    f = f[f["technique"].isin(sel_techniques)]
if sel_heights:
    f = f[f["height"].isin(sel_heights)]

# Quick search across a few fields
if q.strip():
    qq = q.strip().lower()
    for col in ["team", "match", "taker"]:
        if col not in f.columns:
            f[col] = None
    mask = (
        f["team"].fillna("").astype(str).str.lower().str.contains(qq)
        | f["match"].fillna("").astype(str).str.lower().str.contains(qq)
        | f["taker"].fillna("").astype(str).str.lower().str.contains(qq)
    )
    f = f[mask]

# Minute filter
if "Minute_num" in f.columns:
    f = f.copy()
    f["Minute_num"] = pd.to_numeric(f["Minute_num"], errors="coerce")
    if minute_mode == "range" and minute_range is not None:
        f = f[f["Minute_num"].between(minute_range[0], minute_range[1], inclusive="both")]
    elif minute_mode == "single" and apply_single_minute and minute_single_value is not None:
        f = f[f["Minute_num"] == minute_single_value]

if only_shots and "is_shot" in f.columns:
    f = f[f["is_shot"]]

# ---------- Header ----------
st.markdown(
    """
<div class="app-header">
  <div class="app-title">Set Pieces • Corners</div>
  <div class="app-sub">Allsvenskan 2025 • Platform-style analysis view (SkillCorner / Opta / StatsBomb-inspired UX)</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- KPI layer (platform tiles) ----------
total_corners = int(len(f))
n_matches = int(f["match_id"].nunique()) if "match_id" in f.columns else int(pd.Series(f["match"]).nunique())
corners_per_match = (total_corners / n_matches) if n_matches else 0.0

shot_corners = int(f["is_shot"].sum()) if "is_shot" in f.columns else 0
shot_rate = (shot_corners / total_corners) if total_corners else 0.0
total_xg = float(pd.Series(f.get("xg", 0)).sum())

sp_txt = pd.Series(f.get("sp_outcome", "")).fillna("").astype(str)
shots_3s = int(sp_txt.str.contains("shot within 3 seconds", case=False, na=False).sum())

st.markdown(
    f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">Corner events</div><div class="value">{total_corners:,}</div><div class="hint">Rows in filtered set</div></div>
  <div class="kpi"><div class="label">Matches</div><div class="value">{n_matches:,}</div><div class="hint">Unique match_id / Match</div></div>
  <div class="kpi"><div class="label">Corners / match</div><div class="value">{corners_per_match:.2f}</div><div class="hint">Volume intensity</div></div>
  <div class="kpi"><div class="label">Shot rate</div><div class="value">{shot_rate*100:.1f}%</div><div class="hint">Corners leading to shots</div></div>
  <div class="kpi"><div class="label">Total xG</div><div class="value">{total_xg:.3f}</div><div class="hint">From resulting shots</div></div>
  <div class="kpi"><div class="label">Shots ≤ 3s</div><div class="value">{shots_3s:,}</div><div class="hint">From SP_outcome tag</div></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<hr class='hr'>", unsafe_allow_html=True)

# ---------- Views (like product navigation) ----------
tab_league, tab_team, tab_match, tab_player = st.tabs(["League View", "Team View", "Match View", "Player View"])

def bar_or_table(df_, x, y, title):
    st.markdown(f"<div class='card'><h3>{title}</h3>", unsafe_allow_html=True)
    if len(df_) == 0:
        st.info("No data for current filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    if PLOTLY_OK:
        fig = px.bar(df_, x=x, y=y)
        fig.update_layout(xaxis_title="", yaxis_title="", height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df_, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

def pie_or_table(df_, names, values, title):
    st.markdown(f"<div class='card'><h3>{title}</h3>", unsafe_allow_html=True)
    if len(df_) == 0:
        st.info("No data for current filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    if PLOTLY_OK:
        fig = px.pie(df_, names=names, values=values, hole=0.52)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df_, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_league:
    c1, c2 = st.columns(2)

    team_counts = (
        f.groupby("team", dropna=False)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="corners")
    )
    tech_counts = (
        f.groupby("technique", dropna=False)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="corners")
    )

    with c1:
        bar_or_table(team_counts, "team", "corners", "Corner volume by team")
    with c2:
        pie_or_table(tech_counts, "technique", "corners", "Delivery technique mix")

    c3, c4 = st.columns(2)

    shot_outcomes = (
        f[f.get("is_shot", False)]
        .groupby("shot_outcome", dropna=False)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="shots")
    )
    xg_by_team = (
        f.groupby("team", dropna=False)["xg"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    with c3:
        bar_or_table(shot_outcomes, "shot_outcome", "shots", "Shot outcomes from corners")
    with c4:
        bar_or_table(xg_by_team, "team", "xg", "xG from corners by team")

with tab_team:
    st.markdown("<div class='card card-tight'><h3>Team drilldown</h3>", unsafe_allow_html=True)

    team_list = sorted(pd.Series(f["team"]).dropna().unique().tolist())
    team_pick = st.selectbox("Select team", team_list if team_list else ["(no teams)"])
    st.markdown("</div>", unsafe_allow_html=True)

    if team_list:
        ft = f[f["team"] == team_pick].copy()

        a, b, c = st.columns(3)
        with a:
            taker_counts = ft.groupby("taker", dropna=False).size().sort_values(ascending=False).head(12).reset_index(name="corners")
            bar_or_table(taker_counts, "taker", "corners", "Primary takers (volume)")
        with b:
            tech = ft.groupby("technique", dropna=False).size().sort_values(ascending=False).reset_index(name="corners")
            pie_or_table(tech, "technique", "corners", "Technique mix")
        with c:
            xg_takers = ft.groupby("taker", dropna=False)["xg"].sum().sort_values(ascending=False).head(12).reset_index()
            bar_or_table(xg_takers, "taker", "xg", "xG by taker (danger creation)")

        st.markdown("<div class='card'><h3>Insight snapshots</h3>", unsafe_allow_html=True)
        shot_rate_team = (int(ft["is_shot"].sum()) / len(ft)) if len(ft) else 0.0
        st.markdown(
            f"""
- **Shot rate:** {shot_rate_team*100:.1f}%
- **Total xG:** {float(ft["xg"].sum()):.3f}
- **Most common technique:** {ft["technique"].fillna("Unknown").value_counts().idxmax() if len(ft) else "—"}
- **Top taker by volume:** {ft["taker"].fillna("Unknown").value_counts().idxmax() if len(ft) else "—"}
""")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_match:
    st.markdown("<div class='card card-tight'><h3>Match drilldown</h3>", unsafe_allow_html=True)

    match_list = sorted(pd.Series(f["match"]).dropna().unique().tolist())
    match_pick = st.selectbox("Select match", match_list if match_list else ["(no matches)"])
    st.markdown("</div>", unsafe_allow_html=True)

    if match_list:
        fm = f[f["match"] == match_pick].copy()

        c1, c2 = st.columns(2)
        with c1:
            by_team = fm.groupby("team", dropna=False).size().sort_values(ascending=False).reset_index(name="corners")
            bar_or_table(by_team, "team", "corners", "Corners by team in match")
        with c2:
            by_min = fm[["Minute_num"]].copy()
            by_min["Minute_num"] = pd.to_numeric(by_min["Minute_num"], errors="coerce")
            by_min = by_min.dropna()
            st.markdown("<div class='card'><h3>Corner timing</h3>", unsafe_allow_html=True)
            if len(by_min) == 0:
                st.info("No minute data for this selection.")
            else:
                if PLOTLY_OK:
                    fig = px.histogram(by_min, x="Minute_num", nbins=24)
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(by_min["Minute_num"].value_counts().sort_index())
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h3>Set-piece outcomes (table)</h3>", unsafe_allow_html=True)
        cols = [c for c in ["team", "taker", "technique", "height", "Minute_num", "sp_outcome", "is_shot", "shot_outcome", "xg"] if c in fm.columns]
        st.dataframe(fm[cols].sort_values(["Minute_num"], na_position="last"), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_player:
    st.markdown("<div class='card card-tight'><h3>Player drilldown</h3>", unsafe_allow_html=True)

    player_list = sorted(pd.Series(f["taker"]).dropna().unique().tolist())
    player_pick = st.selectbox("Select taker", player_list if player_list else ["(no takers)"])
    st.markdown("</div>", unsafe_allow_html=True)

    if player_list:
        fp = f[f["taker"] == player_pick].copy()
        c1, c2 = st.columns(2)
        with c1:
            tech = fp.groupby("technique", dropna=False).size().sort_values(ascending=False).reset_index(name="corners")
            pie_or_table(tech, "technique", "corners", "Technique profile")
        with c2:
            by_team = fp.groupby("team", dropna=False).size().sort_values(ascending=False).reset_index(name="corners")
            bar_or_table(by_team, "team", "corners", "Corners taken by team")

        st.markdown("<div class='card'><h3>Player summary</h3>", unsafe_allow_html=True)
        sr = (int(fp["is_shot"].sum()) / len(fp)) if len(fp) else 0.0
        st.markdown(
            f"""
- **Corners:** {len(fp):,}
- **Shot rate:** {sr*100:.1f}%
- **Total xG created:** {float(fp["xg"].sum()):.3f}
- **Top delivery height:** {fp["height"].fillna("Unknown").value_counts().idxmax() if len(fp) else "—"}
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Raw table + export ----------
if show_raw:
    st.markdown("<div class='card'><h3>Data export</h3>", unsafe_allow_html=True)
    st.caption("Export the current filtered dataset (what you see in all views).")

    # Keep columns sane (show important first)
    preferred = ["match", "team", "taker", "Minute_num", "Second_num", "technique", "height", "sp_outcome", "is_shot", "shot_outcome", "xg"]
    cols = [c for c in preferred if c in f.columns] + [c for c in f.columns if c not in preferred]
    st.dataframe(f[cols], use_container_width=True, hide_index=True)

    csv = f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="allsvenskan_corners_filtered.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Tip: Use the Filter rail + tabs like a scouting platform. The UX is optimized for fast drilldowns.")
