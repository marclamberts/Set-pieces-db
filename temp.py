import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import ast

st.set_page_config(page_title="Advanced Goal Map", layout="wide")
st.title("⚽ Advanced Set Piece Goal Analysis")

# Load data
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

# Parse location
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
    "shot.first_time", "shot.body_part.name",
    "competition.country_name", "competition.competition_name", "season.season_name"
}
if missing := (required_cols - set(df.columns)):
    st.error(f"Missing columns: {missing}")
    st.stop()

# Filter for shots
df = df[df["location_x"].notna() & df["shot.statsbomb_xg"].notna()]
df_goals = df[(df["shot.outcome.name"] == "Goal") & (df["location_x"] >= 60)].copy()

# Sidebar filters
with st.sidebar:
    st.header("🎯 Filters")
    filters = {
        "Set Piece Type": st.selectbox("Set Piece", ["All"] + df["play_pattern.name"].dropna().unique().tolist()),
        "Team": st.selectbox("Team", ["All"] + df["team.name"].dropna().unique().tolist()),
        "Match": st.selectbox("Match", ["All"] + df["Match"].dropna().unique().tolist()),
        "Position": st.selectbox("Position", ["All"] + df["position.name"].dropna().unique().tolist()),
        "Body Part": st.selectbox("Body Part", ["All"] + df["shot.body_part.name"].dropna().unique().tolist()),
        "Nation": st.selectbox("Nation", ["All"] + df["competition.country_name"].dropna().unique().tolist()),
        "League": st.selectbox("League", ["All"] + df["competition.competition_name"].dropna().unique().tolist()),
        "Season": st.selectbox("Season", ["All"] + df["season.season_name"].dropna().unique().tolist()),
        "First-Time": st.selectbox("First-Time Shot", ["All", "True", "False"]),
    }
    xg_range = st.slider("xG Range", float(df["shot.statsbomb_xg"].min()), float(df["shot.statsbomb_xg"].max()), (0.0, 1.0), 0.01)

# Apply filters
filtered = df_goals.copy()
if filters["Set Piece Type"] != "All":
    filtered = filtered[filtered["play_pattern.name"] == filters["Set Piece Type"]]
if filters["Team"] != "All":
    filtered = filtered[filtered["team.name"] == filters["Team"]]
if filters["Match"] != "All":
    filtered = filtered[filtered["Match"] == filters["Match"]]
if filters["Position"] != "All":
    filtered = filtered[filtered["position.name"] == filters["Position"]]
if filters["Body Part"] != "All":
    filtered = filtered[filtered["shot.body_part.name"] == filters["Body Part"]]
if filters["Nation"] != "All":
    filtered = filtered[filtered["competition.country_name"] == filters["Nation"]]
if filters["League"] != "All":
    filtered = filtered[filtered["competition.competition_name"] == filters["League"]]
if filters["Season"] != "All":
    filtered = filtered[filtered["season.season_name"] == filters["Season"]]
if filters["First-Time"] != "All":
    is_ft = filters["First-Time"] == "True"
    filtered = filtered[filtered["shot.first_time"] == is_ft]

filtered = filtered[filtered["shot.statsbomb_xg"].between(*xg_range)]

# Show summary stats
st.sidebar.markdown("----")
st.sidebar.metric("Total Goals", len(filtered))
st.sidebar.metric("Avg. xG", round(filtered["shot.statsbomb_xg"].mean(), 3))

# Exit early if no data
if filtered.empty:
    st.warning("No goals found for this filter.")
    st.stop()

# Pitch plot
x = filtered["location_y"]
y = filtered["location_x"] - 60

hover_texts = [
    f"{row['team.name']} vs {row['Match']}<br>xG: {row['shot.statsbomb_xg']:.2f}<br>Body: {row['shot.body_part.name']}"
    for _, row in filtered.iterrows()
]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='markers+text',
    marker=dict(size=10, color='#1E90FF', line=dict(color='white', width=1.2)),
    text=filtered["shot.statsbomb_xg"].round(2).astype(str),
    hovertext=hover_texts,
    hoverinfo="text"
))

# Pitch shapes (simplified)
fig.update_layout(
    title=f"🗺️ Goal Map: {filters['Set Piece Type']}",
    xaxis=dict(range=[0, 80], visible=False),
    yaxis=dict(range=[0, 60], visible=False, scaleanchor="x"),
    shapes=[
        dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="#000", width=2)),
        dict(type="rect", x0=18, y0=42, x1=62, y1=60, line=dict(color="gray", width=1)),
    ],
    height=600,
    plot_bgcolor="#f9f9f9",
    paper_bgcolor="#f9f9f9"
)

# Display output
tab1, tab2, tab3 = st.tabs(["📊 Goal Map", "📋 Data Table", "📈 xG Histogram"])

with tab1:
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(filtered[[
        "team.name", "Match", "play_pattern.name", "position.name",
        "shot.body_part.name", "shot.first_time", "shot.statsbomb_xg",
        "competition.country_name", "competition.competition_name", "season.season_name"
    ]])

with tab3:
    st.subheader("xG Distribution")
    st.bar_chart(filtered["shot.statsbomb_xg"].value_counts().sort_index())

# Optional export
st.download_button("📥 Download Filtered Data", filtered.to_csv(index=False), file_name="filtered_goals.csv")

