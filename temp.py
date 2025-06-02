import streamlit as st
import pandas as pd
import os
import ast
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Set Piece Goals Dashboard", layout="wide")

@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    events = pd.read_excel(os.path.join(base_path, "events2.xlsx"))
    goals = pd.read_excel(os.path.join(base_path, "merged_output.xlsx"))
    ti = pd.read_excel(os.path.join(base_path, "TI.xlsx"))
    return events, goals, ti

def extract_xy(loc):
    try:
        if isinstance(loc, str):
            loc = ast.literal_eval(loc)
        if isinstance(loc, (list, tuple)):
            return [loc[0] if len(loc) > 0 else None, loc[1] if len(loc) > 1 else None]
    except Exception:
        pass
    return [None, None]

# Load data
events, goals, ti = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filter Set Piece Goals")
set_piece_filter = st.sidebar.multiselect("Set Piece Type", goals['Set Piece Type'].unique())
team_filter = st.sidebar.multiselect("Team", goals['Team'].unique())
match_filter = st.sidebar.multiselect("Match", goals['Match'].unique())
position_filter = st.sidebar.multiselect("Position", goals['Position'].unique())
body_part_filter = st.sidebar.multiselect("Body Part", goals['Body Part'].unique())
nation_filter = st.sidebar.multiselect("Nation", goals['Nation'].unique())
league_filter = st.sidebar.multiselect("League", goals['League'].unique())
season_filter = st.sidebar.multiselect("Season", goals['Season'].unique())
first_time_filter = st.sidebar.selectbox("First Time Shot", ["All", "Yes", "No"])
xg_range = st.sidebar.slider("xG Range", float(goals['xG'].min()), float(goals['xG'].max()), (0.0, float(goals['xG'].max())))

# Filter data
filtered = goals.copy()
if set_piece_filter:
    filtered = filtered[filtered['Set Piece Type'].isin(set_piece_filter)]
if team_filter:
    filtered = filtered[filtered['Team'].isin(team_filter)]
if match_filter:
    filtered = filtered[filtered['Match'].isin(match_filter)]
if position_filter:
    filtered = filtered[filtered['Position'].isin(position_filter)]
if body_part_filter:
    filtered = filtered[filtered['Body Part'].isin(body_part_filter)]
if nation_filter:
    filtered = filtered[filtered['Nation'].isin(nation_filter)]
if league_filter:
    filtered = filtered[filtered['League'].isin(league_filter)]
if season_filter:
    filtered = filtered[filtered['Season'].isin(season_filter)]
if first_time_filter == "Yes":
    filtered = filtered[filtered['First Time Shot'] == True]
elif first_time_filter == "No":
    filtered = filtered[filtered['First Time Shot'] == False]
filtered = filtered[(filtered['xG'] >= xg_range[0]) & (filtered['xG'] <= xg_range[1])]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Goals", len(filtered))
col2.metric("Avg xG", round(filtered['xG'].mean(), 3))
col3.metric("First Time Goals", filtered['First Time Shot'].sum())

# Goal map
fig = px.scatter(
    filtered,
    x="X", y="Y",
    color="xG", size="xG",
    hover_data=['Player', 'Team', 'xG', 'Minute'],
    color_continuous_scale="Turbo",
    labels={"xG": "Expected Goals"},
    title="Set Piece Goals"
)
fig.update_layout(yaxis=dict(autorange='reversed'))
st.plotly_chart(fig, use_container_width=True)

# xG bar chart
fig_bar = px.histogram(filtered, x="xG", nbins=30, title="xG Distribution")
st.plotly_chart(fig_bar, use_container_width=True)

# Data Table
st.subheader("📋 Filtered Goal Events")
st.dataframe(filtered)

# Download option
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(filtered)
st.download_button("Download CSV", csv, "filtered_goals.csv", "text/csv")

# Interactive Pitch Visualization for Throw-ins Leading to Shots
st.subheader("🌀 Throw-ins Leading to Shots (Interactive)")

pitch_data = ti[(ti["type.name"] == "Pass") & (ti["play_pattern.name"] == "From Throw In")].copy()
shots = ti[ti["type.name"] == "Shot"]
pitch_data = pitch_data[pitch_data["possession"].isin(shots["possession"])]

xg_map = shots.groupby("possession")["shot.statsbomb_xg"].max()
pitch_data = pitch_data.merge(xg_map, how="left", on="possession")

pitch_data[["location_x", "location_y"]] = pitch_data["location"].apply(extract_xy).apply(pd.Series)
pitch_data[["pass.end_location_x", "pass.end_location_y"]] = pitch_data["pass.end_location"].apply(extract_xy).apply(pd.Series)
pitch_data = pitch_data.dropna(subset=["location_x", "location_y", "pass.end_location_x", "pass.end_location_y", "shot.statsbomb_xg"])

# Plotly figure
fig = go.Figure()

for _, row in pitch_data.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["location_x"], row["pass.end_location_x"]],
        y=[row["location_y"], row["pass.end_location_y"]],
        mode='lines+markers',
        line=dict(color=row["shot.statsbomb_xg"], width=2),
        marker=dict(size=4, color=row["shot.statsbomb_xg"], colorscale='Viridis', cmin=0, cmax=1),
        hovertemplate=(
            f"Player: {row['player.name']}<br>"
            f"Team: {row['team.name']}<br>"
            f"xG: {row['shot.statsbomb_xg']:.2f}<br>"
            f"Start: ({row['location_x']:.1f}, {row['location_y']:.1f})<br>"
            f"End: ({row['pass.end_location_x']:.1f}, {row['pass.end_location_y']:.1f})<extra></extra>"
        ),
        showlegend=False
    ))

fig.update_layout(
    title="Throw-ins Leading to Shots",
    xaxis=dict(range=[0, 120], visible=False),
    yaxis=dict(range=[80, 0], visible=False),
    width=500,
    height=400,
    plot_bgcolor='white'
)

st.plotly_chart(fig, use_container_width=False)

# DataFrame Display
st.dataframe(pitch_data[[
    "team.name", "player.name", "possession", "location_x", "location_y",
    "pass.end_location_x", "pass.end_location_y", "shot.statsbomb_xg"
]].dropna())
