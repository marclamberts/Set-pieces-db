# -------------------- Imports & Config --------------------
import streamlit as st
import pandas as pd
import os
import ast
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Set Piece Dashboard",
    layout="wide",
    page_icon="⚽"
)

PASSWORD = "PrincessWay2526"

# -------------------- Style --------------------
professional_style = """
<style>
    .main { background-color: #f8f9fa; }
    .sidebar .sidebar-content { background-color: #ffffff; box-shadow: 2px 0 10px rgba(0,0,0,0.1); }
    h1, h2, h3, h4, h5, h6 { color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button { background-color: #3498db; color: white; border-radius: 6px; }
    .stButton>button:hover { background-color: #2980b9; transform: translateY(-1px); box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 8px 8px 0 0; background-color: #ecf0f1; }
    .stTabs [aria-selected="true"] { background-color: #3498db; color: white; }
    [data-testid="metric-container"] { background-color: white; border-radius: 8px; padding: 15px; border-left: 4px solid #3498db; }
</style>
"""
st.markdown(professional_style, unsafe_allow_html=True)

# -------------------- Authentication --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Set Piece Dashboard")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Please authenticate to continue")
        password = st.text_input("Enter password:", type="password", key="password_input")
        if st.button("Login", key="login_button"):
            if password == PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        st.caption("© 2023 Football Analytics Team")
    st.stop()

import os
import ast
import pandas as pd
import streamlit as st

import os
import ast
import pandas as pd
import streamlit as st

# -------------------- Load Data --------------------
@st.cache_data(ttl=3600)
def load_data():
    base_path = os.path.dirname(__file__)

    # Load Excel files with engine specified
    df_main = pd.read_excel(os.path.join(base_path, "clean_merged.xlsx"))
    df_filtered = pd.read_excel(os.path.join(base_path, "filtered_goals_all_matches.xlsx"))

    # Merge on match_id, allow .x/.y suffixes for overlapping columns
    df = pd.merge(df_main, df_filtered, on="match_id", how="outer", suffixes=('', '_drop'))

    # Drop all duplicate columns that came from df_filtered
    df = df[[col for col in df.columns if not col.endswith('_drop')]]

    # Clean common string columns if they exist
    for col in ["competition.country_name", "competition.competition_name", "season.season_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df

# Load and preprocess data
df = load_data()

# Safely parse location column
df['location'] = df['location'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [None, None, None])

# Expand location into separate columns
loc_df = df['location'].apply(pd.Series)
loc_df.columns = ['location_x', 'location_y', 'location_z']
df = pd.concat([df, loc_df], axis=1)

# Drop duplicates based on relevant columns
df = df.drop_duplicates(subset=[
    'location_x', 'location_y', 'shot.statsbomb_xg',
    'team.name', 'player.name', 'Match', 'shot.body_part.name'
])

# Filter valid shots
df = df[df['location_x'].notna() & df['shot.statsbomb_xg'].notna()]

# Filter for goals from inside the final third
df_goals = df[(df['shot.outcome.name'] == 'Goal') & (df['location_x'] >= 60)].copy()




# -------------------- Sidebar Filters --------------------
with st.sidebar:
    st.markdown("### 🔍 Filter Options")
    filters = {}
    filters["Set Piece Type"] = st.selectbox(
        "Set Piece", 
        ["All"] + sorted(df["play_pattern.name"].dropna().unique()),
        key="set_piece_filter"
    )
    filters["Team"] = st.selectbox(
        "Team", 
        ["All"] + sorted(df["team.name"].dropna().unique()),
        key="team_filter"
    )
    filters["Position"] = st.selectbox(
        "Position", 
        ["All"] + sorted(df["position.name"].dropna().unique()),
        key="position_filter"
    )
    filters["Nation"] = st.selectbox(
        "Nation", 
        ["All"] + sorted(df["competition.country_name"].dropna().unique()),
        key="nation_filter"
    )
    filters["Match"] = st.selectbox(
        "Match", 
        ["All"] + sorted(df["Match"].dropna().unique()),
        key="match_filter"
    )
    filters["Body Part"] = st.selectbox(
        "Body Part", 
        ["All"] + sorted(df["shot.body_part.name"].dropna().unique()),
        key="body_part_filter"
    )
    filters["League"] = st.selectbox(
        "League", 
        ["All"] + sorted(df["competition.competition_name"].dropna().unique()),
        key="league_filter"
    )
    filters["Season"] = st.selectbox(
        "Season", 
        ["All"] + sorted(df["season.season_name"].dropna().unique()),
        key="season_filter"
    )
    filters["First-Time"] = st.selectbox(
        "First-Time Shot", 
        ["All", "Yes", "No"],
        key="first_time_filter"
    )
    xg_range = st.slider(
        "xG Range", 
        float(df["shot.statsbomb_xg"].min()), 
        float(df["shot.statsbomb_xg"].max()), 
        (0.0, 1.0), 
        0.01,
        key="xg_slider"
    )

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
    filtered = filtered[filtered["shot.first_time"] == (filters["First-Time"] == "Yes")]
filtered = filtered[filtered["shot.statsbomb_xg"].between(*xg_range)]

if filtered.empty:
    st.warning("No goals found matching these filters.")
    st.stop()

# -------------------- Metrics --------------------
st.title("⚽ Set Piece Goals Analysis")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Goals", len(filtered))
col2.metric("Avg. xG", f"{filtered['shot.statsbomb_xg'].mean():.3f}")
col3.metric("Top Team", filtered['team.name'].mode()[0])
col4.metric("Most Common Type", filtered['play_pattern.name'].mode()[0])

# -------------------- Tabs --------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "General Dashboard", "Goal Map", "Data Explorer", "xG Analysis", "Goal Placement", "Header Data", "Summary Report"
])

with tab0:
    st.markdown("### 📊 General Overview")
    col1, col2 = st.columns(2)
    with col1:
        team_counts = filtered["team.name"].value_counts().reset_index()
        team_counts.columns = ["Team", "Goals"]
        fig_team = px.bar(team_counts, x="Team", y="Goals", color="Goals", text="Goals")
        st.plotly_chart(fig_team, use_container_width=True)
    with col2:
        type_counts = filtered["play_pattern.name"].value_counts().reset_index()
        type_counts.columns = ["Set Piece Type", "Goals"]
        fig_type = px.bar(type_counts, x="Set Piece Type", y="Goals", color="Goals", text="Goals")
        st.plotly_chart(fig_type, use_container_width=True)

with tab1:
    st.markdown("### 📍 Goal Locations")
    fig = go.Figure()
    pitch_length, pitch_width = 60, 80
    fig.update_layout(
        xaxis=dict(range=[0, pitch_width], showgrid=False, zeroline=False, visible=False, scaleanchor="y"),
        yaxis=dict(range=[0, pitch_length], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        height=700,
        shapes=[
            dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="black", width=2)),
            dict(type="rect", x0=18, y0=0, x1=62, y1=18, line=dict(color="black", width=2)),
            dict(type="rect", x0=30, y0=0, x1=50, y1=6, line=dict(color="black", width=2)),
            dict(type="line", x0=30, y0=0, x1=50, y1=0, line=dict(color="black", width=4)),
            dict(type="circle", x0=38, y0=7, x1=40, y1=9, line=dict(color="black", width=2)),
            dict(type="path", path="M 18 0 A 20 22 0 0 1 62 0", line=dict(color="black", width=2)),
            dict(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color="black", width=2)),
            dict(type="path", path="M 30 60 A 20 20 0 0 1 50 60", line=dict(color="black", width=2)),
        ]
    )
    filtered_half = filtered[filtered["location_x"] >= 60].copy()
    filtered_half["plot_x"] = filtered_half["location_y"]
    filtered_half["plot_y"] = 120 - filtered_half["location_x"]
    hover_text = (
        "Player: " + filtered_half["player.name"] +
        "<br>Team: " + filtered_half["team.name"] +
        "<br>xG: " + filtered_half["shot.statsbomb_xg"].round(2).astype(str) +
        "<br>Body Part: " + filtered_half["shot.body_part.name"] +
        "<br>Match: " + filtered_half["Match"]
    )
    fig.add_trace(go.Scatter(
        x=filtered_half["plot_x"],
        y=filtered_half["plot_y"],
        mode='markers',
        marker=dict(size=filtered_half["shot.statsbomb_xg"] * 40 + 6, color='#e74c3c', line=dict(width=1, color='#2c3e50')),
        text=hover_text,
        hoverinfo='text'
    ))
    st.plotly_chart(fig, use_container_width=True)

    selected_player = st.selectbox("Select Player", sorted(filtered_half["player.name"].unique()), key="player_selector")
    st.dataframe(filtered_half[filtered_half["player.name"] == selected_player][[
        "player.name", "team.name", "shot.statsbomb_xg", "shot.body_part.name", "Match", "competition.competition_name"
    ]])

with tab2:
    st.markdown("### 🔍 Data Table")
    st.dataframe(filtered, use_container_width=True)

with tab3:
    st.markdown("### 📊 xG & Timeline Visualizations")
    chart_type = st.selectbox("Choose Chart Type", ["xG Histogram", "Box Plot", "xG by Category", "Scatter Plot", "Goals Over Time"], key="chart_type_selector")
    if chart_type == "xG Histogram":
        st.bar_chart(filtered["shot.statsbomb_xg"])
    elif chart_type == "Box Plot":
        st.plotly_chart(px.box(filtered, y="shot.statsbomb_xg", points="all"), use_container_width=True)
    elif chart_type == "xG by Category":
        category = st.selectbox("Group xG by:", ["team.name", "shot.body_part.name", "play_pattern.name", "position.name"], key="category_selector")
        data = filtered.groupby(category)["shot.statsbomb_xg"].mean().sort_values(ascending=False).reset_index()
        fig_bar = px.bar(data, x=category, y="shot.statsbomb_xg", text="shot.statsbomb_xg")
        st.plotly_chart(fig_bar, use_container_width=True)
    elif chart_type == "Scatter Plot":
        fig = px.scatter(filtered, x="location_x", y="location_y", size="shot.statsbomb_xg", color="team.name", hover_name="player.name")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Goals Over Time":
        if "date" in filtered.columns:
            filtered["match_date"] = pd.to_datetime(filtered["date"], errors="coerce")
            filtered = filtered[filtered["match_date"].notna()]
            goals_by_date = filtered.groupby("match_date").size().reset_index(name="Goals")
            xg_by_date = filtered.groupby("match_date")["shot.statsbomb_xg"].mean().reset_index(name="Avg_xG")
            st.plotly_chart(px.line(goals_by_date, x="match_date", y="Goals", markers=True), use_container_width=True)
            st.plotly_chart(px.line(xg_by_date, x="match_date", y="Avg_xG", markers=True), use_container_width=True)

with tab4:
    GOAL_WIDTH = 7.32
    GOAL_HEIGHT = 2.44
    LEFT_POST_Y = 36.8
    RIGHT_POST_Y = 43.2

    st.markdown("### 🥅 Goal Placement from shot.end_location string (6 Zones, Player POV)")

    def split_end_location(s):
        try:
            x_str, y_str, z_str = s.split(',')
            return float(x_str), float(y_str), float(z_str)
        except Exception:
            return None, None, None

    filtered[['shot.end_location_x', 'shot.end_location_y', 'shot.end_location_z']] = filtered['shot.end_location'].apply(
        lambda s: pd.Series(split_end_location(s)))
    
    goals = filtered.dropna(subset=['shot.end_location_y']).copy()
    goals = goals[(goals['shot.end_location_y'] >= LEFT_POST_Y) & (goals['shot.end_location_y'] <= RIGHT_POST_Y)]

    if goals.empty:
        st.info("No goals with shot.end_location_y inside goalposts found.")
    else:
        goals['goal_x_m'] = (goals['shot.end_location_y'] - LEFT_POST_Y) * (GOAL_WIDTH / (RIGHT_POST_Y - LEFT_POST_Y))
        goals['goal_z_m'] = goals['shot.end_location_z'].fillna(0)

        xg = goals['shot.statsbomb_xg'].fillna(0)
        marker_size = np.interp(xg, (xg.min(), xg.max()), (6, 20))
        marker_color = xg

        fig = go.Figure()

        fig.add_shape(type="rect", x0=0, y0=0, x1=GOAL_WIDTH, y1=GOAL_HEIGHT,
                      line=dict(color="black", width=3))
        fig.add_shape(type="line", x0=0, y0=GOAL_HEIGHT/2, x1=GOAL_WIDTH, y1=GOAL_HEIGHT/2,
                      line=dict(color="gray", dash="dash"))
        fig.add_shape(type="line", x0=GOAL_WIDTH/3, y0=0, x1=GOAL_WIDTH/3, y1=GOAL_HEIGHT,
                      line=dict(color="gray", dash="dash"))
        fig.add_shape(type="line", x0=2*GOAL_WIDTH/3, y0=0, x1=2*GOAL_WIDTH/3, y1=GOAL_HEIGHT,
                      line=dict(color="gray", dash="dash"))

        fig.add_trace(go.Scatter(
            x=goals['goal_x_m'],
            y=goals['goal_z_m'],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=marker_color,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="xG"),
                line=dict(width=1, color='DarkSlateGrey'),
                opacity=0.8
            ),
            text=goals['player.name'],
            hovertemplate=(
                "Player: %{text}<br>"
                "Width: %{x:.2f} m<br>"
                "Height: %{y:.2f} m<br>"
                "xG: %{marker.color:.3f}<extra></extra>"
            ),
            name="Goals"
        ))

        fig.update_layout(
            title="Goal Placement on Goal Face from shot.end_location string (Size & Color by xG)",
            xaxis=dict(title="Goal Width (meters)", range=[0, GOAL_WIDTH], showgrid=False, zeroline=False),
            yaxis=dict(title="Goal Height (meters)", range=[0, GOAL_HEIGHT], showgrid=False, zeroline=False),
            height=600,
            width=700,
            plot_bgcolor='white',
            yaxis_scaleanchor="x"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Goals Data Sample")
        st.dataframe(goals[[
            "player.name", "team.name", "shot.end_location", "shot.end_location_x", "shot.end_location_y", "shot.end_location_z", "shot.statsbomb_xg"
        ]])

with tab5:
    st.markdown("### 🧠 Header Data Analysis")
    
    # Filter for headers only
    headers = filtered[filtered['shot.body_part.name'] == 'Head'].copy()
    
    if headers.empty:
        st.warning("No header goals found matching these filters.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Header Goals by Team")
            team_headers = headers['team.name'].value_counts().reset_index()
            team_headers.columns = ['Team', 'Header Goals']
            fig = px.bar(team_headers, x='Team', y='Header Goals', color='Header Goals')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### Header xG Distribution")
            fig = px.histogram(headers, x='shot.statsbomb_xg', nbins=15, 
                              title='Distribution of xG for Headers')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Header Goal Locations")
        
        # Create pitch for headers
        fig = go.Figure()
        pitch_length, pitch_width = 60, 80
        fig.update_layout(
            xaxis=dict(range=[0, pitch_width], showgrid=False, zeroline=False, visible=False, scaleanchor="y"),
            yaxis=dict(range=[0, pitch_length], showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='white',
            height=600,
            shapes=[
                dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="black", width=2)),
                dict(type="rect", x0=18, y0=0, x1=62, y1=18, line=dict(color="black", width=2)),
                dict(type="rect", x0=30, y0=0, x1=50, y1=6, line=dict(color="black", width=2)),
                dict(type="line", x0=30, y0=0, x1=50, y1=0, line=dict(color="black", width=4)),
                dict(type="circle", x0=38, y0=7, x1=40, y1=9, line=dict(color="black", width=2)),
                dict(type="path", path="M 18 0 A 20 22 0 0 1 62 0", line=dict(color="black", width=2)),
                dict(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color="black", width=2)),
                dict(type="path", path="M 30 60 A 20 20 0 0 1 50 60", line=dict(color="black", width=2)),
            ]
        )
        
        headers_half = headers[headers["location_x"] >= 60].copy()
        headers_half["plot_x"] = headers_half["location_y"]
        headers_half["plot_y"] = 120 - headers_half["location_x"]
        
        hover_text = (
            "Player: " + headers_half["player.name"] +
            "<br>Team: " + headers_half["team.name"] +
            "<br>xG: " + headers_half["shot.statsbomb_xg"].round(2).astype(str) +
            "<br>Match: " + headers_half["Match"]
        )
        
        fig.add_trace(go.Scatter(
            x=headers_half["plot_x"],
            y=headers_half["plot_y"],
            mode='markers',
            marker=dict(
                size=headers_half["shot.statsbomb_xg"] * 40 + 6, 
                color='#3498db', 
                line=dict(width=1, color='#2c3e50')
            ),
            text=hover_text,
            hoverinfo='text'
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Header Goals Data")
        st.dataframe(headers[[
            "player.name", "team.name", "shot.statsbomb_xg", "Match", 
            "competition.competition_name", "play_pattern.name"
        ]].sort_values("shot.statsbomb_xg", ascending=False))

with tab6:
    st.markdown("### 📑 Summary Report")
    st.write("#### 🔢 Basic Stats")
    st.write(filtered.describe(include='all'))

    st.write("#### 💡 Grouped Stats")
    cols_to_group = ["team.name", "shot.body_part.name", "play_pattern.name"]
    for col in cols_to_group:
        group_data = filtered.groupby(col).agg(
            Goals=('player.name', 'count'),
            Avg_xG=('shot.statsbomb_xg', 'mean')
        ).sort_values(by='Goals', ascending=False)
        st.write(f"**Grouped by {col}**")
        st.dataframe(group_data)

st.download_button(
    "Download CSV", 
    data=filtered.to_csv(index=False), 
    file_name="set_piece_goals.csv",
    key="download_button"
)

st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>© 2025 Outswinger FC Analytics | Powered by Marc Lamberts</p>
    </div>
""", unsafe_allow_html=True)
