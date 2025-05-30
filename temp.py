import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Interactive Vertical Pitch", layout="centered")
st.title("⚽ Interactive Goal Map by Set Piece Type")

# Load local Excel file
csv_path = os.path.join(os.path.dirname(__file__), "events.xlsx")
try:
    df = pd.read_excel(csv_path)
except FileNotFoundError:
    st.error("Excel file not found. Make sure 'events.xlsx' is in the same folder as this script.")
    st.stop()

# Required columns
required_cols = {
    "shot.outcome.name", "location_x", "location_y", "play_pattern.name",
    "player_name", "team_name", "player_position_name", "statsbomb_xg", "Match"
}

missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Excel must contain columns: {', '.join(missing_cols)}")
    st.stop()

# Sidebar filter for set piece type
set_piece_options = ["From Corner", "From Throw In", "From Free Kick"]
selected_pattern = st.sidebar.selectbox("Set Piece Type", set_piece_options)

# Filter for goals from selected set piece in upper pitch (location_x >= 60)
filtered_df = df[
    (df["shot.outcome.name"] == "Goal") &
    (df["play_pattern.name"] == selected_pattern) &
    (df["location_x"] >= 60)
].copy()

if filtered_df.empty:
    st.warning("No goals found for this set piece and pitch half.")
    st.stop()

# Shift pitch coordinates: make the upper half start at y=0
# location_x runs 0-120 (pitch length), location_y runs 0-80 (pitch width)
# After filtering location_x >= 60, subtract 60 to set attacking half starting at 0

# X axis: pitch width (0-80), Y axis: pitch length (60-120) shifted to 0-60
x = filtered_df["location_y"]      # width: 0 to 80
y = filtered_df["location_x"] - 60  # length shifted to 0-60

# Build hover text including Match info
hover_texts = [
    f"Player: {p}<br>Team: {t}<br>Position: {pos}<br>xG: {xg:.2f}<br>Match: {match}"
    for p, t, pos, xg, match in zip(
        filtered_df["player_name"],
        filtered_df["team_name"],
        filtered_df["player_position_name"],
        filtered_df["statsbomb_xg"],
        filtered_df["Match"]
    )
]

# Create plotly figure with pitch shapes
plot = go.Figure()

plot.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        size=10,
        color='gold',
        line=dict(color='black', width=1.5)
    ),
    text=hover_texts,
    hoverinfo='text',
    name='Goals'
))

pitch_shapes = [
    # Outer boundaries (width=80, length=60)
    dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="white", width=3)),

    # Penalty box
    dict(type="rect", x0=18, y0=42, x1=62, y1=60, line=dict(color="white", width=2)),

    # Six-yard box
    dict(type="rect", x0=30, y0=54, x1=50, y1=60, line=dict(color="white", width=2)),

    # Goal line
    dict(type="line", x0=36, y0=60, x1=44, y1=60, line=dict(color="white", width=5)),

    # Penalty spot (circle)
    dict(type="circle", x0=38.5, y0=48.5, x1=41.5, y1=51.5, line=dict(color="white", width=2)),
]

plot.update_layout(
    title=f"Goals from {selected_pattern}",
    xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[0, 60], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
    shapes=pitch_shapes,
    plot_bgcolor='#144A29',
    paper_bgcolor='#144A29',
    height=700,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(plot, use_container_width=True)
