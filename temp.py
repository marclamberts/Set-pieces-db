import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import ast

st.set_page_config(page_title="Database", layout="wide")
st.title("Goals from set pieces")

# Economist-style colors
ECONOMIST_COLORS = {
    "background": "#f5f5f5",
    "primary": "#3d6e70",
    "secondary": "#e3120b",
    "text": "#121212"
}

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

# Required columns check
required_cols = {
    "shot.outcome.name", "location_x", "location_y", "play_pattern.name",
    "team.name", "position.name", "shot.statsbomb_xg", "Match",
    "shot.first_time", "shot.body_part.name",
    "competition.country_name", "competition.competition_name", "season.season_name"
}
if missing := (required_cols - set(df.columns)):
    st.error(f"Missing columns: {missing}")
    st.stop()

# Filter shots
df = df[df["location_x"].notna() & df["shot.statsbomb_xg"].notna()]
df_goals = df[(df["shot.outcome.name"] == "Goal") & (df["location_x"] >= 60)].copy()

# Sidebar filters with two columns
with st.sidebar:
    st.markdown(f"""
        <style>
            .sidebar .sidebar-content {{
                background-color: {ECONOMIST_COLORS['background']};
            }}
        </style>
    """, unsafe_allow_html=True)

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

    st.markdown("----")
    st.markdown(f"""
        <div style="color: {ECONOMIST_COLORS['primary']}; font-weight: bold;">
            <p>Total Goals: {len(df_goals)}</p>
            <p>Avg. xG: {round(df_goals['shot.statsbomb_xg'].mean(), 3)}</p>
        </div>
    """, unsafe_allow_html=True)

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

if filtered.empty:
    st.warning("No goals found for this filter.")
    st.stop()

# Create goal map
x = filtered["location_y"]
y = filtered["location_x"] - 60

hover_texts = [
    f"<b>{row['team.name']}</b> vs {row['Match']}<br>"
    f"xG: {row['shot.statsbomb_xg']:.2f}<br>"
    f"Body: {row['shot.body_part.name']}<br>"
    f"Position: {row['position.name']}"
    for _, row in filtered.iterrows()
]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='markers',
    marker=dict(
        size=12,
        color=ECONOMIST_COLORS['secondary'],
        line=dict(color='white', width=1.5),
        opacity=0.8
    ),
    hovertext=hover_texts,
    hoverinfo="text"
))

fig.update_layout(
    title=f"<b>Goal Map: {filters['Set Piece Type']}</b>",
    title_font=dict(size=20, color=ECONOMIST_COLORS['text']),
    xaxis=dict(range=[0, 80], visible=False),
    yaxis=dict(range=[0, 60], visible=False, scaleanchor="x"),
    shapes=[
        dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="#000", width=2)),
        dict(type="rect", x0=18, y0=42, x1=62, y1=60, line=dict(color="gray", width=1)),
    ],
    height=600,
    plot_bgcolor=ECONOMIST_COLORS['background'],
    paper_bgcolor=ECONOMIST_COLORS['background'],
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    )
)

# Show tabs (goal map and data)
tab1, tab2 = st.tabs(["📊 Goal Map", "📋 Data Table"])

with tab1:
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(
        filtered[[
            "team.name", "Match", "play_pattern.name", "position.name",
            "shot.body_part.name", "shot.first_time", "shot.statsbomb_xg",
            "competition.country_name", "competition.competition_name", "season.season_name"
        ]].style.apply(lambda x: [
            f"background-color: {ECONOMIST_COLORS['background']}; color: {ECONOMIST_COLORS['text']}"
            for _ in x
        ], axis=1)
    )

# Export filtered data
st.download_button(
    label="📥 Download Filtered Data",
    data=filtered.to_csv(index=False),
    file_name="filtered_goals.csv",
    help="Download the filtered data as a CSV file"
)

# Global style
st.markdown(f"""
    <style>
        .main {{
            background-color: {ECONOMIST_COLORS['background']};
        }}
        .stButton>button {{
            background-color: {ECONOMIST_COLORS['primary']};
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
        }}
        .stButton>button:hover {{
            background-color: {ECONOMIST_COLORS['secondary']};
            color: white;
        }}
        .stDataFrame {{
            font-family: Arial, sans-serif;
        }}
    </style>
""", unsafe_allow_html=True)
