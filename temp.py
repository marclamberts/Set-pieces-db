import streamlit as st
import pandas as pd
import os
import ast
import plotly.graph_objects as go
from mplsoccer import Pitch, VerticalPitch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# -------------------- Config --------------------
st.set_page_config(page_title="Set Piece Goals Dashboard", layout="wide")

PASSWORD = "PrincessWay2526"

# Session state for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter password to continue:", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
        st.experimental_rerun()
    else:
        st.stop()

if st.session_state.authenticated:
    ECONOMIST_COLORS = {
        "background": "#f5f5f5",
        "primary": "#3d6e70",
        "secondary": "#e3120b",
        "text": "#121212"
    }

    @st.cache_data
    def load_data():
        base_path = os.path.dirname(__file__)
        df1 = pd.read_excel(os.path.join(base_path, "events2.xlsx"))
        try:
            df2 = pd.read_excel(os.path.join(base_path, "merged_output.xlsx"))
            return pd.concat([df1, df2], ignore_index=True)
        except:
            return df1

    df = load_data()

    def parse_location(loc):
        try:
            if isinstance(loc, str):
                return ast.literal_eval(loc)
            elif isinstance(loc, (list, tuple)):
                return loc
            else:
                return [None, None, None]
        except:
            return [None, None, None]

    loc_df = df['location'].apply(parse_location).apply(pd.Series)
    loc_df.columns = ['location_x', 'location_y', 'location_z']
    df = pd.concat([df, loc_df], axis=1).copy()

    df = df[df["location_x"].notna() & df["shot.statsbomb_xg"].notna()]
    df_goals = df[(df["shot.outcome.name"] == "Goal") & (df["location_x"] >= 60)].copy()

    # -------------------- Filters --------------------
    with st.sidebar:
        st.header("Filters")
        col1, col2 = st.columns(2)

        with col1:
            filters = {}
            filters["Set Piece Type"] = st.selectbox("Set Piece", ["All"] + df["play_pattern.name"].dropna().unique().tolist())
            filters["Team"] = st.selectbox("Team", ["All"] + df["team.name"].dropna().unique().tolist())
            filters["Position"] = st.selectbox("Position", ["All"] + df["position.name"].dropna().unique().tolist())
            filters["Nation"] = st.selectbox("Nation", ["All"] + df["competition.country_name"].dropna().unique().tolist())

        with col2:
            filters["Match"] = st.selectbox("Match", ["All"] + df["Match"].dropna().unique().tolist())
            filters["Body Part"] = st.selectbox("Body Part", ["All"] + df["shot.body_part.name"].dropna().unique().tolist())
            filters["League"] = st.selectbox("League", ["All"] + df["competition.competition_name"].dropna().unique().tolist())
            filters["Season"] = st.selectbox("Season", ["All"] + df["season.season_name"].dropna().unique().tolist())

        filters["First-Time"] = st.selectbox("First-Time Shot", ["All", "True", "False"])
        xg_range = st.slider("xG Range", float(df["shot.statsbomb_xg"].min()), float(df["shot.statsbomb_xg"].max()), (0.0, 1.0), 0.01)

    # -------------------- Apply Filters --------------------
    filtered = df_goals.copy()
    for key, col in [
        ("Set Piece Type", "play_pattern.name"),
        ("Team", "team.name"),
        ("Match", "Match"),
        ("Position", "position.name"),
        ("Body Part", "shot.body_part.name"),
        ("Nation", "competition.country_name"),
        ("League", "competition.competition_name"),
        ("Season", "season.season_name")
    ]:
        if filters[key] != "All":
            filtered = filtered[filtered[col] == filters[key]]
    if filters["First-Time"] != "All":
        filtered = filtered[filtered["shot.first_time"] == (filters["First-Time"] == "True")]
    filtered = filtered[filtered["shot.statsbomb_xg"].between(*xg_range)]

    if filtered.empty:
        st.warning("No goals found for this filter.")
        st.stop()

    # -------------------- Tabs --------------------
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Goal Map", "📋 Data Table", "🧪 Test", "🌀 Throw ins"])

    with tab1:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[60, 120], showgrid=False, zeroline=False, visible=False),
            height=600
        )
        fig.add_trace(go.Scatter(
            x=filtered["location_y"],
            y=filtered["location_x"],
            mode='markers',
            marker=dict(
                size=filtered["shot.statsbomb_xg"] * 40 + 6,
                color='crimson',
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=filtered.apply(lambda row: f"Team: {row['team.name']}<br>Player: {row.get('player.name', 'N/A')}<br>Body Part: {row['shot.body_part.name']}<br>Match: {row['Match']}<br>xG: {row['shot.statsbomb_xg']:.2f}", axis=1),
            hoverinfo='text'
        ))
        shapes = [
            dict(type='rect', x0=0, x1=80, y0=60, y1=120, line=dict(color='black', width=2)),
            dict(type='rect', x0=18, x1=62, y0=96, y1=120, line=dict(color='black')),
            dict(type='rect', x0=30, x1=50, y0=114, y1=120, line=dict(color='black')),
            dict(type='line', x0=40, x1=40, y0=60, y1=120, line=dict(color='gray', dash='dash'))
        ]
        fig.update_layout(shapes=shapes)
        st.plotly_chart(fig, use_container_width=True)

        # KPIs below
        st.subheader("📊 Summary Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Filtered Goals", len(filtered))
        col2.metric("Average xG", round(filtered["shot.statsbomb_xg"].mean(), 3))
        col3.metric("Most Frequent Set Piece", filtered["play_pattern.name"].mode()[0] if not filtered.empty else "N/A")

    with tab2:
        st.dataframe(filtered)

    with tab3:
        st.subheader("xG Distribution")
        st.bar_chart(filtered["shot.statsbomb_xg"])

    with tab4:
        st.subheader("Throw-ins Leading to Shots")

        @st.cache_data
        def load_ti_data():
            base_path = os.path.dirname(__file__)
            df_ti = pd.read_excel(os.path.join(base_path, "TI.xlsx"))
            return df_ti

        ti = load_ti_data()

        def extract_xy(loc):
            try:
                if isinstance(loc, str):
                    loc = ast.literal_eval(loc)
                if isinstance(loc, (list, tuple)):
                    return [loc[0], loc[1]]
            except:
                pass
            return [None, None]

        # Parse locations
        ti[["location_x", "location_y"]] = pd.DataFrame(ti["location"].apply(extract_xy).tolist(), index=ti.index)
        ti[["pass.end_location_x", "pass.end_location_y"]] = pd.DataFrame(ti["pass.end_location"].apply(extract_xy).tolist(), index=ti.index)

        passes = ti[(ti["type.name"] == "Pass") & (ti["play_pattern.name"] == "From Throw In")]
        shots = ti[ti["type.name"] == "Shot"]

        throwins = passes[passes["possession"].isin(shots["possession"])].copy()

        # Map possession -> max shot xG
        possession_xg = shots.groupby("possession")["shot.statsbomb_xg"].max()
        throwins["shot_xG"] = throwins["possession"].map(possession_xg)

        throwins = throwins.dropna(subset=["location_x", "location_y", "pass.end_location_x", "pass.end_location_y"])

        pitch = VerticalPitch(pitch_type='statsbomb', half=True, pitch_color='white', line_color='black')
        fig, ax = pitch.draw(figsize=(7, 5))

        # Normalize xG for color mapping
        norm = mcolors.Normalize(vmin=0, vmax=throwins["shot_xG"].max() if not throwins["
