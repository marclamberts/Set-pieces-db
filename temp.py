import streamlit as st
import pandas as pd
import numpy as np

from utils import (
    load_data,
    inject_css,
    _to_num,
    styled_bar,
    styled_donut,
    styled_histogram,
)

# ─────────────────────────────────────────────────────────────
# App config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Corners · Allsvenskan 2025",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
df = load_data()

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-dot"></div>
          <div>
            <div class="sidebar-title">Corner Analytics</div>
            <div class="sidebar-sub">Allsvenskan 2025</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")
    try:
        st.page_link("streamlit_app.py", label="🏠 Home", icon="🏠")
        st.page_link("pages/1_League_Overview.py", label="🏟️ League Overview", icon="🏟️")
        st.page_link("pages/2_Team_Analysis.py", label="🧭 Team Analysis", icon="🧭")
        st.page_link("pages/3_Match_View.py", label="🗓️ Match View", icon="🗓️")
        st.page_link("pages/4_Player_Profiles.py", label="🧑‍💼 Player Profiles", icon="🧑‍💼")
        st.page_link("pages/5_Data_Explorer.py", label="📋 Data Explorer", icon="📋")
    except Exception:
        st.markdown(
            """
            - [🏟️ League Overview](./League_Overview)
            - [🧭 Team Analysis](./Team_Analysis)
            - [🗓️ Match View](./Match_View)
            - [🧑‍💼 Player Profiles](./Player_Profiles)
            - [📋 Data Explorer](./Data_Explorer)
            """
        )

    st.markdown("---")
    st.markdown("### Quick Filters")

    teams_all = sorted(df["team"].dropna().astype(str).unique().tolist())
    techniques_all = sorted(df["technique"].dropna().astype(str).unique().tolist())

    team_sel = st.multiselect("Teams", teams_all, default=teams_all)
    tech_sel = st.multiselect("Technique", techniques_all, default=techniques_all)

    # ✅ SAFE minute slider (won’t crash if min == max)
    mins = _to_num(df["Minute_num"]).dropna()
    minute_range = None

    if len(mins) > 0:
        min_val = int(mins.min())
        max_val = int(mins.max())

        if min_val < max_val:
            minute_range = st.slider(
                "Minute range",
                min_val,
                max_val,
                (min_val, max_val),
            )
        else:
            st.caption(f"Minute range: all events at minute {min_val}")

    only_shots = st.toggle("Only corners → shot", value=False)

# ─────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────
f = df.copy()
if team_sel:
    f = f[f["team"].isin(team_sel)]
if tech_sel:
    f = f[f["technique"].isin(tech_sel)]
if minute_range:
    f = f[_to_num(f["Minute_num"]).between(minute_range[0], minute_range[1])]
if only_shots:
    f = f[f["is_shot"] == True]

# ─────────────────────────────────────────────────────────────
# KPIs (robust)
# ─────────────────────────────────────────────────────────────
total = int(len(f))
teams = int(f.get("team", pd.Series(dtype=object)).nunique())

match_series = (
    f.get("match", pd.Series(dtype=object))
    .astype(str)
    .replace("nan", np.nan)
    .dropna()
)
matches = int(match_series.nunique())

is_shot = f.get("is_shot", pd.Series([False] * len(f))).fillna(False).astype(bool)
shots = int(is_shot.sum())
shot_rate = shots / total if total else 0.0

xg = float(f.get("xg", pd.Series([0.0] * len(f))).fillna(0).sum())
cpm = total / matches if matches else 0.0

shot_outcome = f.get("shot_outcome", pd.Series(dtype=object)).fillna("").astype(str)
goals = int(shot_outcome.str.contains("goal", case=False, na=False).sum())

# ─────────────────────────────────────────────────────────────
# Hero + KPIs
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-eyebrow">Allsvenskan 2025 · Corner Kick Events</div>
      <div class="hero-title">Corner Kick<br/>Analytics</div>
      <div class="hero-sub">
        A clean set-piece dashboard for delivery style, takers, outcomes and xG created from corners.
        <br/><span style="color: var(--muted2);">Filters applied:</span>
        <span class="badge">Teams: {len(team_sel) if team_sel else 0}</span>
        <span class="badge">Techniques: {len(tech_sel) if tech_sel else 0}</span>
        <span class="badge">Only shots: {"Yes" if only_shots else "No"}</span>
      </div>
      <div class="hero-badges" style="margin-top:10px;">
        <span class="badge">⚽ Live filters</span>
        <span class="badge">xG powered</span>
        <span class="badge">All matches</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-label">Corners</div>
        <div class="kpi-hint">Current selection</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{matches:,}</div>
        <div class="kpi-label">Matches</div>
        <div class="kpi-hint">Unique fixtures</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{teams:,}</div>
        <div class="kpi-label">Teams</div>
        <div class="kpi-hint">Unique clubs</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{cpm:.1f}</div>
        <div class="kpi-label">Corners / match</div>
        <div class="kpi-hint">Average pace</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{shot_rate*100:.1f}%</div>
        <div class="kpi-label">Shot rate</div>
        <div class="kpi-hint">Corner → shot</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{xg:.2f}</div>
        <div class="kpi-label">Total xG</div>
        <div class="kpi-hint">From corner shots</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1.35, 1], gap="large")

with left:
    st.markdown(
        "<div class='section-title'>Top Teams (Current Filters)</div>"
        "<div class='hero-sub'>Corners taken + xG created</div>",
        unsafe_allow_html=True,
    )

    if len(f) == 0:
        st.info("No events match the current filters.")
    else:
        by_team = (
            f.groupby("team", dropna=False)
            .agg(corners=("team", "size"), xg=("xg", "sum"), shots=("is_shot", "sum"))
            .reset_index()
        )
        by_team["xg_per_corner"] = by_team["xg"] / by_team["corners"].replace(0, np.nan)
        by_team["shot_rate"] = by_team["shots"] / by_team["corners"].replace(0, np.nan)
        by_team = by_team.sort_values(["corners", "xg"], ascending=[False, False])

        c1, c2 = st.columns(2, gap="medium")

        with c1:
            top_corners = by_team.head(10).sort_values("corners", ascending=True)
            styled_bar(top_corners, x="corners", y="team", orientation="h", height=360)

        with c2:
            top_xg = by_team.head(10).sort_values("xg", ascending=True)
            styled_bar(top_xg, x="xg", y="team", orientation="h", height=360)

        st.markdown(
            "<div class='section-title'>Timing</div>"
            "<div class='hero-sub'>When corners happen (minute distribution)</div>",
            unsafe_allow_html=True,
        )
        styled_histogram(_to_num(f["Minute_num"]), nbins=28, height=260)

with right:
    st.markdown(
        "<div class='section-title'>Delivery Profile</div>"
        "<div class='hero-sub'>Technique distribution (current selection)</div>",
        unsafe_allow_html=True,
    )
    tech = f.groupby("technique", dropna=False).size().reset_index(name="n")
    styled_donut(tech, "technique", "n", height=330)

    st.markdown(
        "<div class='section-title'>Featured Match</div>"
        "<div class='hero-sub'>Quick look at one fixture</div>",
        unsafe_allow_html=True,
    )

    matches_all = sorted(df["match"].dropna().astype(str).unique().tolist())
    match_counts = (
        df.groupby("match", dropna=False)
        .size()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )
    default_match = match_counts[0] if match_counts else (matches_all[0] if matches_all else "")
    match_pick = st.selectbox(
        "Match",
        matches_all,
        index=matches_all.index(default_match) if default_match in matches_all else 0,
    )

    mf = df[df["match"].astype(str) == str(match_pick)].copy()
    m_total = len(mf)
    m_shots = int(mf["is_shot"].fillna(False).sum()) if m_total else 0
    m_xg = float(mf["xg"].fillna(0).sum()) if m_total else 0.0
    m_sr = (m_shots / m_total) if m_total else 0.0

    st.markdown(
        f"""
        <div class="hero" style="padding:12px 14px 10px 14px;">
          <div class="hero-eyebrow">Match snapshot</div>
          <div style="font-weight:900;font-size:16px;line-height:1.15;margin-top:4px;">
            {match_pick}
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:12px;">
            <div class="kpi" style="padding:10px 10px 8px 10px;">
              <div class="kpi-value" style="font-size:18px;">{m_total:,}</div>
              <div class="kpi-label">Corners</div>
              <div class="kpi-hint">This match</div>
            </div>
            <div class="kpi" style="padding:10px 10px 8px 10px;">
              <div class="kpi-value" style="font-size:18px;">{m_sr*100:.1f}%</div>
              <div class="kpi-label">Shot rate</div>
              <div class="kpi-hint">Corner → shot</div>
            </div>
            <div class="kpi" style="padding:10px 10px 8px 10px;">
              <div class="kpi-value" style="font-size:18px;">{m_xg:.2f}</div>
              <div class="kpi-label">xG</div>
              <div class="kpi-hint">From shots</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if m_total:
        mt = (
            mf.groupby("team", dropna=False)
            .size()
            .reset_index(name="corners")
            .sort_values("corners", ascending=True)
        )
        styled_bar(mt, x="corners", y="team", orientation="h", height=240)
    else:
        st.info("No events found for this match.")

# ─────────────────────────────────────────────────────────────
# Explore cards
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="section-title">Explore</div>
    <div class="card-grid">
      <a class="navcard" href="./League_Overview">
        <div class="navcard-title">League Overview</div>
        <div class="navcard-sub">League-wide volume, technique mix, xG, outcomes and timing.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Team_Analysis">
        <div class="navcard-title">Team Analysis</div>
        <div class="navcard-sub">Pick a club and deep dive into takers, deliveries and results.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Match_View">
        <div class="navcard-title">Match View</div>
        <div class="navcard-sub">Select a fixture and inspect corner events and outcomes.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Player_Profiles">
        <div class="navcard-title">Player Profiles</div>
        <div class="navcard-sub">Compare takers by volume, style preferences, and xG impact.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Data_Explorer">
        <div class="navcard-title">Data Explorer</div>
        <div class="navcard-sub">Filter/search the full table and export your current view to CSV.</div>
        <div class="navcard-cta">→</div>
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="footer">
      Showing <b>{total:,}</b> corner events (filtered). · Goals (direct): <b>{goals}</b>
      <br/>Excel expected: <code>Allsvenskan - Corners 2025.xlsx</code>.
    </div>
    """,
    unsafe_allow_html=True,
)