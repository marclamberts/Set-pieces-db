import streamlit as st
import pandas as pd
import os
import plotly.express as px
import ast
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# -------------------- Config --------------------
st.set_page_config(page_title="Set Piece Goals Dashboard", layout="wide")

# Password Gate
PASSWORD = "PrincessWay2526"
password_input = st.text_input("Enter password to continue:", type="password")
if password_input != PASSWORD:
    st.stop()

st.title("Goals from Set Pieces")

ECONOMIST_COLORS = {
    "background": "#f5f5f5",
    "primary": "#3d6e70",
    "secondary": "#e3120b",
    "text": "#121212"
}

# -------------------- Load Data --------------------
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
        return ast.literal_eval(loc) if isinstance(loc, str) else loc
    except:
        return [None, None, None]

# ---- FIX fragmentation warning here ----
parsed_locs = df['location'].apply(parse_location)
loc_df = pd.DataFrame(parsed_locs.tolist(), columns=['location_x', 'location_y', 'location_z'])
df = pd.concat([df, loc_df], axis=1).copy()

required_cols = {
    "shot.outcome.name", "location_x", "location_y", "play_pattern.name",
    "team.name", "position.name", "shot.statsbomb_xg", "Match",
    "shot.first_time", "shot.body_part.name",
    "competition.country_name", "competition.competition_name", "season.season_name"
}
if missing := (required_cols - set(df.columns)):
    st.error(f"Missing columns: {missing}")
    st.stop()

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

# -------------------- KPIs --------------------
st.subheader("📊 Summary Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Filtered Goals", len(filtered))
col2.metric("Average xG", round(filtered["shot.statsbomb_xg"].mean(), 3))
col3.metric("Most Frequent Set Piece", filtered["play_pattern.name"].mode()[0] if not filtered.empty else "N/A")

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 mplsoccer Pitch", "📈 Scatter", "🔥 Heatmap", "📊 Breakdown", "📋 Data"])

# ---- mplsoccer Pitch ----
with tab1:
    st.subheader("Goal Locations on Pitch")
    pitch = Pitch(pitch_type='statsbomb', line_color='black', pitch_color='white')
    fig, ax = pitch.draw(figsize=(10, 6))
    pitch.scatter(filtered["location_x"], filtered["location_y"], ax=ax, color=ECONOMIST_COLORS["secondary"], s=100, edgecolors="black")
    st.pyplot(fig)

# ---- Plotly Scatter ----
with tab2:
    fig = px.scatter(
        filtered,
        x="location_y", y=filtered["location_x"] - 60,
        color="shot.body_part.name",
        hover_name="team.name",
        size="shot.statsbomb_xg",
        title="Goal Map by Body Part"
    )
    fig.update_layout(yaxis=dict(range=[0, 60]), xaxis=dict(range=[0, 80]), height=600)
    st.plotly_chart(fig, use_container_width=True)

# ---- Heatmap ----
with tab3:
    heatmap_fig = px.density_heatmap(
        filtered, x="location_y", y="location_x",
        nbinsx=50, nbinsy=50,
        color_continuous_scale="Reds",
        title="Goal Density from Set Pieces"
    )
    heatmap_fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(heatmap_fig, use_container_width=True)

# ---- Breakdown ----
with tab4:
    st.subheader("Top Scoring Teams")
    st.bar_chart(filtered["team.name"].value_counts().head(10))

    if "player.name" in filtered.columns:
        st.subheader("Top Scoring Players")
        st.bar_chart(filtered["player.name"].value_counts().head(10))

# ---- Data Table ----
with tab5:
    st.dataframe(filtered)

# -------------------- Download --------------------
st.download_button(
    label="⬇️ Download Filtered Data",
    data=filtered.to_csv(index=False),
    file_name="filtered_goals.csv"
)

# -------------------- Styling --------------------
st.markdown(f"""
    <style>
        .main {{
            background-color: {ECONOMIST_COLORS['background']};
        }}
        .stButton>button {{
            background-color: {ECONOMIST_COLORS['primary']};
            color: white;
            border-radius: 4px;
        }}
        .stButton>button:hover {{
            background-color: {ECONOMIST_COLORS['secondary']};
        }}
        .stDataFrame {{
            font-family: Arial, sans-serif;
        }}
    </style>
""", unsafe_allow_html=True)
