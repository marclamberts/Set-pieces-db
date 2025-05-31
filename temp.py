import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import ast

st.set_page_config(page_title="Interactive Goal Map", layout="centered")
st.title("⚽ Goal Map by Set Piece Type")

# Load and merge up to 3 Excel files
def load_data():
    base_path = os.path.dirname(__file__)
    files = ["events2.xlsx","events.xlsx","norway.xlsx"]
    dfs = []

    st.subheader("📂 File Loading Logs")
    for f in files:
        path = os.path.join(base_path, f)
        st.write(f"🔍 Trying to load: `{f}`")
        try:
            df = pd.read_excel(path)
            st.write(f"✅ Loaded `{f}` | Rows: {len(df)}")
            st.write(f"📋 Columns: {df.columns.tolist()}")
            dfs.append(df)
        except Exception as e:
            st.write(f"❌ Failed to load `{f}`: {e}")

    if not dfs:
        st.error("No data files loaded. Ensure at least one Excel file is present.")
        st.stop()

    return pd.concat(dfs, ignore_index=True)

df = load_data()

# Parse location column
def parse_location(loc):
    try:
        return ast.literal_eval(loc) if isinstance(loc, str) else loc
    except:
        return [None, None, None]

df[['location_x', 'location_y', 'location_z']] = df['location'].apply(parse_location).apply(pd.Series)

# Check required columns
required_cols = {
    "shot.outcome.name", "location_x", "location_y", "play_pattern.name",
    "team.name", "position.name", "shot.statsbomb_xg", "Match",
    "shot.first_time", "shot.body_part.name"
}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Missing required columns: {', '.join(missing_cols)}")
    st.stop()

# Filter: goals in upper half of pitch
filtered_df = df[(df["shot.outcome.name"] == "Goal") & (df["location_x"] >= 60)].copy()

# Sidebar filters
st.sidebar.header("Filters")
set_piece_options = ["All", "From Corner", "From Throw In", "From Free Kick"]
selected_pattern = st.sidebar.selectbox("Set Piece Type", set_piece_options)

first_time_options = ["All", "True", "False"]
selected_first_time = st.sidebar.selectbox("First-Time Shot", first_time_options)

body_parts = ["All"] + sorted(filtered_df["shot.body_part.name"].dropna().unique())
selected_body_part = st.sidebar.selectbox("Body Part Used", body_parts)

matches = ["All"] + sorted(filtered_df["Match"].dropna().unique())
positions = ["All"] + sorted(filtered_df["position.name"].dropna().unique())

selected_match = st.sidebar.selectbox("Match", matches)
selected_position = st.sidebar.selectbox("Position", positions)

min_xg = float(filtered_df["shot.statsbomb_xg"].min())
max_xg = float(filtered_df["shot.statsbomb_xg"].max())
xg_range = st.sidebar.slider("xG Range", min_value=0.0, max_value=round(max_xg + 0.05, 2),
                             value=(round(min_xg, 2), round(max_xg, 2)), step=0.01)

# Apply filters
if selected_pattern != "All":
    filtered_df = filtered_df[filtered_df["play_pattern.name"] == selected_pattern]
if selected_match != "All":
    filtered_df = filtered_df[filtered_df["Match"] == selected_match]
if selected_position != "All":
    filtered_df = filtered_df[filtered_df["position.name"] == selected_position]
if selected_first_time != "All":
    filtered_df = filtered_df[filtered_df["shot.first_time"] == (selected_first_time == "True")]
if selected_body_part != "All":
    filtered_df = filtered_df[filtered_df["shot.body_part.name"] == selected_body_part]
filtered_df = filtered_df[filtered_df["shot.statsbomb_xg"].between(xg_range[0], xg_range[1])]

if filtered_df.empty:
    st.warning("No goals found for this combination of filters.")
    st.stop()

# Show table
if st.checkbox("Show data table"):
    st.dataframe(filtered_df[[
        "team.name", "Match", "position.name",
        "play_pattern.name", "shot.first_time", "shot.body_part.name",
        "shot.statsbomb_xg", "location_x", "location_y"
    ]])

# Prepare data for plot
x = filtered_df["location_y"]
y = filtered_df["location_x"] - 60

hover_texts = [
    f"<b>Team:</b> {t}<br><b>Match:</b> {m}<br><b>Pos:</b> {p}<br><b>xG:</b> {xg:.2f}"
    f"<br><b>First-Time:</b> {ft}<br><b>Body:</b> {bp}"
    for t, m, p, xg, ft, bp in zip(
        filtered_df["team.name"],
        filtered_df["Match"],
        filtered_df["position.name"],
        filtered_df["shot.statsbomb_xg"],
        filtered_df["shot.first_time"],
        filtered_df["shot.body_part.name"]
    )
]

# Create goal plot
plot = go.Figure()

plot.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=10, color='#FFD700', line=dict(color='black', width=1.2)),
    text=hover_texts,
    hoverinfo='text',
    name='Goals'
))

# Draw pitch
pitch_shapes = [
    dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="#CCCCCC", width=2)),
    dict(type="rect", x0=18, y0=42, x1=62, y1=60, line=dict(color="#AAAAAA", width=1)),
    dict(type="rect", x0=30, y0=54, x1=50, y1=60, line=dict(color="#AAAAAA", width=1)),
    dict(type="line", x0=36, y0=60, x1=44, y1=60, line=dict(color="#AAAAAA", width=4)),
    dict(type="circle", x0=38.5, y0=48.5, x1=41.5, y1=51.5, line=dict(color="#AAAAAA", width=1)),
]

plot.update_layout(
    title=dict(text=f"Goals from {selected_pattern}", font=dict(size=22, family="Georgia")),
    xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[0, 60], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
    shapes=pitch_shapes,
    plot_bgcolor='#F9F9F9',
    paper_bgcolor='#F9F9F9',
    height=700,
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(family="Georgia", size=14, color="#333333")
)

st.plotly_chart(plot, use_container_width=True)
