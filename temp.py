import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import ast

st.set_page_config(page_title="Goal Map | Set Pieces", layout="wide")
st.title("⚽ Goal Map by Set Piece Type")
st.markdown("##### A detailed interactive visualization of goals from set pieces — refined in Washington Post style.")

# Load and merge Excel files
def load_and_merge_excel(files):
    dataframes = []
    for file in files:
        path = os.path.join(os.path.dirname(__file__), file)
        try:
            df = pd.read_excel(path)
            dataframes.append(df)
        except FileNotFoundError:
            st.error(f"File not found: {file}")
            st.stop()
    return pd.concat(dataframes, ignore_index=True)

df = load_and_merge_excel(["events2.xlsx", "events.xlsx"])

# Parse location column
def parse_location(loc):
    try:
        if isinstance(loc, str):
            return ast.literal_eval(loc)
        else:
            return loc
    except:
        return [None, None, None]

df[['location_x', 'location_y', 'location_z']] = df['location'].apply(parse_location).apply(pd.Series)

# Validate required columns
required_cols = {
    "shot.outcome.name", "location_x", "location_y", "play_pattern.name",
    "player.name", "team.name", "position.name", "shot.statsbomb_xg", "Match"
}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Missing columns: {', '.join(missing_cols)}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
set_piece_options = ["All", "From Corner", "From Throw In", "From Free Kick"]
selected_pattern = st.sidebar.selectbox("Set Piece Type", set_piece_options)

# Initial filter: goals from upper half
filtered_df = df[
    (df["shot.outcome.name"] == "Goal") &
    (df["location_x"] >= 60)
].copy()

# Set piece filter
if selected_pattern != "All":
    filtered_df = filtered_df[filtered_df["play_pattern.name"] == selected_pattern]

if filtered_df.empty:
    st.warning("No goals found for selected set piece and area.")
    st.stop()

# Dropdowns
players = ["All"] + sorted(filtered_df["player.name"].dropna().unique())
teams = ["All"] + sorted(filtered_df["team.name"].dropna().unique())
matches = ["All"] + sorted(filtered_df["Match"].dropna().unique())
positions = ["All"] + sorted(filtered_df["position.name"].dropna().unique())

selected_player = st.sidebar.selectbox("Player", players)
selected_team = st.sidebar.selectbox("Team", teams)
selected_match = st.sidebar.selectbox("Match", matches)
selected_position = st.sidebar.selectbox("Position", positions)

# xG range slider
min_xg = float(filtered_df["shot.statsbomb_xg"].min())
max_xg = float(filtered_df["shot.statsbomb_xg"].max())
xg_range = st.sidebar.slider("xG Range", min_value=0.0, max_value=round(max_xg + 0.05, 2),
                             value=(round(min_xg, 2), round(max_xg, 2)), step=0.01)

# Apply filters
if selected_player != "All":
    filtered_df = filtered_df[filtered_df["player.name"] == selected_player]
if selected_team != "All":
    filtered_df = filtered_df[filtered_df["team.name"] == selected_team]
if selected_match != "All":
    filtered_df = filtered_df[filtered_df["Match"] == selected_match]
if selected_position != "All":
    filtered_df = filtered_df[filtered_df["position.name"] == selected_position]

filtered_df = filtered_df[filtered_df["shot.statsbomb_xg"].between(xg_range[0], xg_range[1])]

if filtered_df.empty:
    st.warning("No goals found for this combination of filters.")
    st.stop()

# Optional data table
if st.checkbox("Show data table"):
    st.dataframe(filtered_df[[
        "player.name", "team.name", "Match", "position.name", 
        "play_pattern.name", "shot.statsbomb_xg", "location_x", "location_y"
    ]])

# Prepare plot coordinates
x = filtered_df["location_y"]
y = filtered_df["location_x"] - 60

# Custom hover text
hover_texts = [
    f"<b>Player:</b> {p}<br><b>Team:</b> {t}<br><b>Position:</b> {pos}<br><b>xG:</b> {xg:.2f}<br><b>Match:</b> {match}"
    for p, t, pos, xg, match in zip(
        filtered_df["player.name"],
        filtered_df["team.name"],
        filtered_df["position.name"],
        filtered_df["shot.statsbomb_xg"],
        filtered_df["Match"]
    )
]

# Create figure
fig = go.Figure()

# Add goal points
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        size=12,
        color='#FFD700',
        line=dict(width=1.5, color='black'),
        opacity=0.95
    ),
    text=hover_texts,
    hoverinfo='text',
    name="Goals"
))

# Half-pitch design (Washington Post style)
pitch_shapes = [
    # Pitch boundary
    dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="#B0B0B0", width=2)),
    # Penalty area
    dict(type="rect", x0=18, y0=42, x1=62, y1=60, line=dict(color="#AAAAAA", width=1)),
    # Six-yard box
    dict(type="rect", x0=30, y0=54, x1=50, y1=60, line=dict(color="#AAAAAA", width=1)),
    # Goal line
    dict(type="line", x0=36, y0=60, x1=44, y1=60, line=dict(color="black", width=4)),
    # Penalty spot
    dict(type="circle", x0=39.5, y0=48.5, x1=40.5, y1=49.5, line=dict(color="#AAAAAA", width=1)),
    # Penalty arc (partial circle)
    dict(
        type="path",
        path="M36,48 Q40,44 44,48",
        line=dict(color="#AAAAAA", width=1)
    )
]

fig.update_layout(
    title=dict(
        text=f"Goals from {selected_pattern}" if selected_pattern != "All" else "All Set Piece Goals",
        x=0.5,
        font=dict(size=24, family="Georgia", color="#333333")
    ),
    xaxis=dict(range=[0, 80], showgrid=False, visible=False),
    yaxis=dict(range=[0, 60], showgrid=False, visible=False, scaleanchor="x"),
    shapes=pitch_shapes,
    plot_bgcolor="#FAFAFA",
    paper_bgcolor="#FAFAFA",
    margin=dict(l=20, r=20, t=60, b=20),
    font=dict(family="Georgia", size=14, color="#333333"),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)
