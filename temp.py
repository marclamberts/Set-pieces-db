# -------------------- Imports & Config --------------------
import streamlit as st
import pandas as pd
import os
import ast
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Set Piece Dashboard",
    layout="wide",
    page_icon="⚽"
)

PASSWORD = "PrincessWay2526"

# -------------------- Style --------------------
fivethirtyeight_style = """
<style>
    :root {
        --fte-blue: #1a73e8;
        --fte-red: #ed3b3b;
        --fte-green: #4caf50;
        --fte-purple: #9c27b0;
        --fte-dark: #202020;
        --fte-light: #f8f8f8;
        --fte-gray: #757575;
    }
    
    .main { 
        background-color: white; 
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .sidebar .sidebar-content { 
        background-color: white; 
        border-right: 1px solid #e0e0e0;
    }
    
    h1, h2, h3, h4, h5, h6 { 
        color: var(--fte-dark);
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h1 {
        border-bottom: 3px solid var(--fte-blue);
        padding-bottom: 8px;
        font-size: 2.2rem;
    }
    
    .stButton>button { 
        background-color: var(--fte-blue); 
        color: white; 
        border-radius: 4px;
        font-weight: 600;
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .stButton>button:hover { 
        background-color: #0d5bba; 
        transform: none; 
        box-shadow: none;
    }
    
    .stTabs [data-baseweb="tab-list"] { 
        gap: 0px;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab"] { 
        padding: 10px 20px; 
        background-color: transparent;
        border: none;
        font-weight: 600;
        color: var(--fte-gray);
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .stTabs [aria-selected="true"] { 
        background-color: transparent;
        color: var(--fte-blue);
        border-bottom: 3px solid var(--fte-blue);
    }
    
    [data-testid="metric-container"] { 
        background-color: white; 
        border-radius: 0px; 
        padding: 15px; 
        border-left: 4px solid var(--fte-blue);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .leaderboard-table { 
        width: 100%; 
        border-collapse: collapse;
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .leaderboard-table th { 
        background-color: var(--fte-dark); 
        color: white; 
        padding: 12px; 
        text-align: left;
        font-weight: 700;
    }
    
    .leaderboard-table td { 
        padding: 10px; 
        border-bottom: 1px solid #e0e0e0;
    }
    
    .leaderboard-table tr:nth-child(even) { 
        background-color: #f8f8f8; 
    }
    
    .leaderboard-table tr:hover { 
        background-color: #f0f0f0; 
    }
    
    .plot-container {
        background-color: white;
        padding: 15px;
    }
    
    .stDataFrame {
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .stMarkdown {
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }
    
    .footer {
        font-size: 0.8em;
        color: var(--fte-gray);
        border-top: 2px solid #e0e0e0;
        padding-top: 15px;
        margin-top: 30px;
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }
    
    /* FiveThirtyEight-style annotations */
    .annotation {
        font-size: 0.85em;
        color: var(--fte-gray);
        font-style: italic;
        margin-top: 5px;
    }
</style>
"""
st.markdown(fivethirtyeight_style, unsafe_allow_html=True)

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
        st.markdown('<div class="footer">© 2023 Football Analytics Team</div>', unsafe_allow_html=True)
    st.stop()

# -------------------- Load Data --------------------
@st.cache_data(ttl=3600)
def load_data():
    base_path = os.path.dirname(__file__)

    # Load single merged Excel file
    df = pd.read_excel(os.path.join(base_path, "db.xlsx"))

    # Clean and ensure string type for filter columns
    filter_columns = ["competition.country_name", "competition.competition_name", "season.season_name"]
    for col in filter_columns:
        if col in df.columns:
            # Convert to string, strip whitespace, and replace any NaN/None with empty string
            df[col] = df[col].astype(str).str.strip().replace('nan', '').replace('None', '')
    
    return df

@st.cache_data(ttl=3600)
def load_german_data():
    base_path = os.path.dirname(__file__)
    return pd.read_excel(os.path.join(base_path, "ger.xlsx"))

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
    st.markdown("### Filter Options")
    filters = {}
    filters["Set Piece Type"] = st.selectbox(
        "Set Piece", 
        ["All"] + sorted(df["play_pattern.name"].dropna().unique().tolist()),
        key="set_piece_filter"
    )
    filters["Team"] = st.selectbox(
        "Team", 
        ["All"] + sorted(df["team.name"].dropna().unique().tolist()),
        key="team_filter"
    )
    filters["Position"] = st.selectbox(
        "Position", 
        ["All"] + sorted(df["position.name"].dropna().unique().tolist()),
        key="position_filter"
    )
    
    # Enhanced filter options with all values
    country_options = ["All"] + sorted([x for x in df["competition.country_name"].unique().tolist() if x and str(x) != 'nan'])
    filters["Nation"] = st.selectbox(
        "Nation", 
        country_options,
        key="nation_filter"
    )
    
    competition_options = ["All"] + sorted([x for x in df["competition.competition_name"].unique().tolist() if x and str(x) != 'nan'])
    filters["League"] = st.selectbox(
        "League", 
        competition_options,
        key="league_filter"
    )
    
    season_options = ["All"] + sorted([x for x in df["season.season_name"].unique().tolist() if x and str(x) != 'nan'])
    filters["Season"] = st.selectbox(
        "Season", 
        season_options,
        key="season_filter"
    )
    
    filters["Match"] = st.selectbox(
        "Match", 
        ["All"] + sorted(df["Match"].dropna().unique().tolist()),
        key="match_filter"
    )
    filters["Body Part"] = st.selectbox(
        "Body Part", 
        ["All"] + sorted(df["shot.body_part.name"].dropna().unique().tolist()),
        key="body_part_filter"
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
st.title("Set Piece Goals Analysis")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Goals", len(filtered))
col2.metric("Avg. xG", f"{filtered['shot.statsbomb_xg'].mean():.3f}")
col3.metric("Top Team", filtered['team.name'].mode()[0])
col4.metric("Most Common Type", filtered['play_pattern.name'].mode()[0])

# -------------------- Tabs --------------------
tab0, tab1, tab4, tab_leaderboard, tab_corner = st.tabs([
    "General Dashboard", "Goal Map", "Goal Placement", "Leaderboard", "Corner Analysis"
])

with tab0:
    st.markdown("### General Overview")
    col1, col2 = st.columns(2)
    with col1:
        team_counts = filtered["team.name"].value_counts().reset_index()
        team_counts.columns = ["Team", "Goals"]
        fig_team = px.bar(team_counts, x="Team", y="Goals", 
                         color_discrete_sequence=["#1a73e8"],
                         template="plotly_white")
        fig_team.update_layout(
            plot_bgcolor='white', 
            paper_bgcolor='white',
            title="Goals by Team",
            title_font=dict(size=18),
            margin=dict(t=40, b=40),
            showlegend=False
        )
        fig_team.update_traces(marker_line_width=0)
        st.plotly_chart(fig_team, use_container_width=True)
        st.markdown('<div class="annotation">Number of set piece goals by team</div>', unsafe_allow_html=True)
    with col2:
        type_counts = filtered["play_pattern.name"].value_counts().reset_index()
        type_counts.columns = ["Set Piece Type", "Goals"]
        fig_type = px.bar(type_counts, x="Set Piece Type", y="Goals",
                         color_discrete_sequence=["#ed3b3b"],
                         template="plotly_white")
        fig_type.update_layout(
            plot_bgcolor='white', 
            paper_bgcolor='white',
            title="Goals by Set Piece Type",
            title_font=dict(size=18),
            margin=dict(t=40, b=40),
            showlegend=False
        )
        fig_type.update_traces(marker_line_width=0)
        st.plotly_chart(fig_type, use_container_width=True)
        st.markdown('<div class="annotation">Distribution across different set piece types</div>', unsafe_allow_html=True)

with tab1:
    st.markdown("### Goal Locations")
    fig = go.Figure()
    pitch_length, pitch_width = 60, 80
    fig.update_layout(
        xaxis=dict(range=[0, pitch_width], showgrid=False, zeroline=False, visible=False, scaleanchor="y"),
        yaxis=dict(range=[0, pitch_length], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700,
        margin=dict(t=40, b=40),
        title="Goal Locations from Set Pieces",
        title_font=dict(size=18),
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
        marker=dict(
            size=filtered_half["shot.statsbomb_xg"] * 40 + 6, 
            color=filtered_half["shot.statsbomb_xg"],
            colorscale='Bluered',
            colorbar=dict(title="xG"),
            line=dict(width=0.5, color='black')
        ),
        text=hover_text,
        hoverinfo='text'
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="annotation">Location of set piece goals (size and color by xG)</div>', unsafe_allow_html=True)

    selected_player = st.selectbox("Select Player", sorted(filtered_half["player.name"].unique()), key="player_selector")
    st.dataframe(filtered_half[filtered_half["player.name"] == selected_player][[
        "player.name", "team.name", "shot.statsbomb_xg", "shot.body_part.name", "Match", "competition.competition_name"
    ]])

with tab4:
    GOAL_WIDTH = 7.32
    GOAL_HEIGHT = 2.44
    LEFT_POST_Y = 36.8
    RIGHT_POST_Y = 43.2

    st.markdown("### Goal Placement from shot.end_location")

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
                      line=dict(color="#757575", dash="dash"))
        fig.add_shape(type="line", x0=GOAL_WIDTH/3, y0=0, x1=GOAL_WIDTH/3, y1=GOAL_HEIGHT,
                      line=dict(color="#757575", dash="dash"))
        fig.add_shape(type="line", x0=2*GOAL_WIDTH/3, y0=0, x1=2*GOAL_WIDTH/3, y1=GOAL_HEIGHT,
                      line=dict(color="#757575", dash="dash"))

        fig.add_trace(go.Scatter(
            x=goals['goal_x_m'],
            y=goals['goal_z_m'],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=marker_color,
                colorscale='Bluered',
                showscale=True,
                colorbar=dict(title="xG"),
                line=dict(width=0.5, color='black'),
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
            title="Goal Placement (Size & Color by xG)",
            title_font=dict(size=18),
            xaxis=dict(title="Goal Width (meters)", range=[0, GOAL_WIDTH], showgrid=False, zeroline=False),
            yaxis=dict(title="Goal Height (meters)", range=[0, GOAL_HEIGHT], showgrid=False, zeroline=False),
            height=600,
            width=700,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=40, b=40),
            yaxis_scaleanchor="x"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="annotation">Where set piece goals are being placed (divided into 6 zones)</div>', unsafe_allow_html=True)

        st.markdown("### Goals Data Sample")
        st.dataframe(goals[[
            "player.name", "team.name", "shot.end_location", "shot.end_location_x", "shot.end_location_y", "shot.end_location_z", "shot.statsbomb_xg"
        ]])

with tab_leaderboard:
    st.markdown("### Performance Leaderboard")
    
    # Add a selectbox to choose the metric for ranking
    leaderboard_metric = st.selectbox(
        "Rank players by:", 
        ["Total Goals", "Total xG", "Average xG per Goal", "Goals per Match"],
        key="leaderboard_metric"
    )
    
    # Calculate the metrics for each player
    if not filtered.empty:
        leaderboard_data = filtered.groupby(['player.name', 'team.name']).agg(
            Total_Goals=('player.name', 'count'),
            Total_xG=('shot.statsbomb_xg', 'sum'),
            Matches=('Match', 'nunique')
        ).reset_index()
        
        leaderboard_data['Avg_xG_per_Goal'] = leaderboard_data['Total_xG'] / leaderboard_data['Total_Goals']
        leaderboard_data['Goals_per_Match'] = leaderboard_data['Total_Goals'] / leaderboard_data['Matches']
        
        # Sort based on selected metric
        if leaderboard_metric == "Total Goals":
            leaderboard_data = leaderboard_data.sort_values('Total_Goals', ascending=False)
            metric_col = 'Total_Goals'
        elif leaderboard_metric == "Total xG":
            leaderboard_data = leaderboard_data.sort_values('Total_xG', ascending=False)
            metric_col = 'Total_xG'
        elif leaderboard_metric == "Average xG per Goal":
            leaderboard_data = leaderboard_data.sort_values('Avg_xG_per_Goal', ascending=False)
            metric_col = 'Avg_xG_per_Goal'
        else:  # Goals per Match
            leaderboard_data = leaderboard_data.sort_values('Goals_per_Match', ascending=False)
            metric_col = 'Goals_per_Match'
        
        # Format the numbers for display
        leaderboard_data[metric_col] = leaderboard_data[metric_col].round(3)
        
        # Display the leaderboard
        st.markdown(f"#### Top 20 Players by {leaderboard_metric}")
        
        # Create a styled table
        st.write(
            f"""
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Player</th>
                        <th>Team</th>
                        <th>{leaderboard_metric}</th>
                        <th>Total Goals</th>
                        <th>Total xG</th>
                        <th>Matches</th>
                    </tr>
                </thead>
                <tbody>
            """,
            unsafe_allow_html=True
        )
        
        for i, row in leaderboard_data.head(20).iterrows():
            st.write(
                f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{row['player.name']}</td>
                    <td>{row['team.name']}</td>
                    <td><strong>{row[metric_col]}</strong></td>
                    <td>{row['Total_Goals']}</td>
                    <td>{row['Total_xG']:.2f}</td>
                    <td>{row['Matches']}</td>
                </tr>
                """,
                unsafe_allow_html=True
            )
        
        st.write("</tbody></table>", unsafe_allow_html=True)
        st.markdown('<div class="annotation">Performance metrics for set piece specialists</div>', unsafe_allow_html=True)
        
        # Show full data as an option
        if st.checkbox("Show full leaderboard data"):
            st.dataframe(leaderboard_data)
            

# Imports to add at the top
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

@st.cache_data
def load_german_data():
    # Your loading logic here
    # Example:
    # return pd.read_csv('german_data.csv')
    pass

with tab_corner:
    st.markdown("### Corner Kick Sequence Analysis")
    
    # Load German data for corner analysis
    df_german = load_german_data()

    # Sidebar filters using df_german
    st.sidebar.markdown("### Corner Filter Options")
    corner_team_filter = st.sidebar.selectbox(
        "Team (Corners)", 
        ["All"] + sorted(df_german["team.name"].dropna().unique().tolist()),
        key="corner_team_filter"
    )
    corner_player_filter = st.sidebar.selectbox(
        "Player (Corners)", 
        ["All"] + sorted(df_german["player.name"].dropna().unique().tolist()),
        key="corner_player_filter"
    )
    corner_technique_filter = st.sidebar.selectbox(
        "Corner Technique", 
        ["All"] + sorted(df_german["pass.technique.name"].dropna().unique().tolist()),
        key="corner_technique_filter"
    )
    corner_side_filter = st.sidebar.selectbox(
        "Corner Side", 
        ["All", "Left", "Right"],
        key="corner_side_filter"
    )

    # Prepare corner data
    df_corner = df_german.copy()

    if 'pass.end_location' in df_corner.columns:
        # Handle string format
        mask_str = df_corner['pass.end_location'].astype(str).str.contains(r'\[.*,.*\]', na=False)
        if mask_str.any():
            df_corner.loc[mask_str, ['pass_end_x', 'pass_end_y']] = df_corner.loc[mask_str, 'pass.end_location'].str.strip('[]').str.split(',', expand=True).astype(float)
        
        # Handle list format
        mask_list = df_corner['pass.end_location'].apply(lambda x: isinstance(x, list))
        if mask_list.any():
            df_corner.loc[mask_list, ['pass_end_x', 'pass_end_y']] = pd.DataFrame(df_corner.loc[mask_list, 'pass.end_location'].tolist())

    if 'index' in df_corner.columns:
        df_corner = df_corner.sort_values(by='index').reset_index(drop=True)
    elif 'event_id' in df_corner.columns:
        df_corner = df_corner.sort_values(by='event_id').reset_index(drop=True)

    # Identify corner passes
    corner_passes = df_corner[
        (df_corner['type.name'] == 'Pass') &
        (df_corner['play_pattern.name'] == 'From Corner') &
        (df_corner['pass.type.name'] == 'Corner')
    ].copy()

    if not corner_passes.empty:
        results = []

        for idx, row in corner_passes.iterrows():
            if 'location' in row and isinstance(row['location'], (list, tuple)) and len(row['location']) >= 1:
                side = 'Left' if row['location'][0] < 60 else 'Right'
            else:
                side = 'Unknown'

            pass_height = row.get('pass.height.name', 'Unknown')
            pass_body_part = row.get('pass.body_part.name', 'Unknown')
            pass_outcome = row.get('pass.outcome.name', 'Unknown')
            pass_technique = row.get('pass.technique.name', 'Unknown')

            start_team = row['possession_team.id'] if 'possession_team.id' in row else row.get('team.id', None)
            subsequent_events = df_corner.iloc[idx + 1:]
            
            if start_team:
                same_possession = subsequent_events[
                    subsequent_events.get('possession_team.id', subsequent_events.get('team.id')) == start_team
                ]
            else:
                same_possession = pd.DataFrame()

            if same_possession.empty:
                classification = 'No first contact - no shot'
            else:
                first_contact = same_possession.iloc[0]
                if first_contact['type.name'] == 'Shot':
                    classification = 'First contact - direct shot'
                else:
                    shots_nearby = same_possession.head(3)[same_possession.head(3)['type.name'] == 'Shot']
                    if not shots_nearby.empty:
                        classification = 'First contact - shot within 3 seconds'
                    else:
                        any_shot = same_possession[same_possession['type.name'] == 'Shot']
                        if not any_shot.empty:
                            classification = 'No first contact - shot'
                        else:
                            classification = 'First contact - no shot'

            results.append({
                'corner_index': idx,
                'classification': classification,
                'side': side,
                'pass_height': pass_height,
                'pass_body_part': pass_body_part,
                'pass_outcome': pass_outcome,
                'pass_technique': pass_technique,
                'pass_end_x': row.get('pass_end_x', None),
                'pass_end_y': row.get('pass_end_y', None),
                'team.name': row['team.name'],
                'player.name': row['player.name'],
                'Match': row.get('Match', 'Unknown'),
                'competition.competition_name': row.get('competition.competition_name', 'Unknown'),
                'season.season_name': row.get('season.season_name', 'Unknown')
            })

        corner_summary = pd.DataFrame(results)

        # Apply filters
        filtered_corners = corner_summary.copy()
        if corner_team_filter != "All":
            filtered_corners = filtered_corners[filtered_corners["team.name"] == corner_team_filter]
        if corner_player_filter != "All":
            filtered_corners = filtered_corners[filtered_corners["player.name"] == corner_player_filter]
        if corner_technique_filter != "All":
            filtered_corners = filtered_corners[filtered_corners["pass_technique"] == corner_technique_filter]
        if corner_side_filter != "All":
            filtered_corners = filtered_corners[filtered_corners["side"] == corner_side_filter]

        if not filtered_corners.empty:
            xg_total = 0.0
            xg_inswinger = 0.0
            xg_outswinger = 0.0
            xg_per_corner = []

            for idx, row in filtered_corners.iterrows():
                corner_index = row['corner_index']
                possession_team = df_corner.loc[corner_index, 'possession_team.id'] if 'possession_team.id' in df_corner.columns else df_corner.loc[corner_index].get('team.id', None)

                if possession_team:
                    subsequent_events = df_corner.iloc[corner_index + 1:]
                    same_possession = subsequent_events[
                        subsequent_events.get('possession_team.id', subsequent_events.get('team.id')) == possession_team
                    ]
                    shots = same_possession[same_possession['type.name'] == 'Shot']

                    if 'shot.statsbomb_xg' in shots.columns:
                        corner_xg = shots['shot.statsbomb_xg'].fillna(0).sum()
                    else:
                        corner_xg = 0.0
                else:
                    corner_xg = 0.0

                xg_per_corner.append(corner_xg)
                xg_total += corner_xg

                pass_technique = row['pass_technique']
                if isinstance(pass_technique, str):
                    if 'inswinger' in pass_technique.lower():
                        xg_inswinger += corner_xg
                    elif 'outswinger' in pass_technique.lower():
                        xg_outswinger += corner_xg

            filtered_corners['xg_per_corner'] = xg_per_corner

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Corners", len(filtered_corners))
            col2.metric("Total xG Generated", f"{xg_total:.2f}")
            col3.metric("Avg xG per Corner", f"{xg_total/len(filtered_corners):.3f}" if len(filtered_corners) > 0 else "0")

            st.markdown("#### Corner Pass End Locations")
            valid_locations = filtered_corners.dropna(subset=['pass_end_x', 'pass_end_y'])

            if not valid_locations.empty:
                colors = {
                    'First contact - direct shot': '#FF0000',
                    'First contact - shot within 3 seconds': '#0000FF', 
                    'No first contact - shot': '#00FF00',
                    'First contact - no shot': '#FFA500',
                    'No first contact - no shot': '#808080'
                }

                fig = go.Figure()

                fig.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="black", width=2))
                fig.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="black", width=2))

                for classification in valid_locations['classification'].unique():
                    subset = valid_locations[valid_locations['classification'] == classification]
                    fig.add_trace(go.Scatter(
                        x=subset['pass_end_x'],
                        y=subset['pass_end_y'],
                        mode='markers',
                        name=classification,
                        marker=dict(
                            color=colors.get(classification, '#000000'),
                            size=10,
                            line=dict(color='black', width=1)
                        )
                    ))

                fig.update_layout(
                    title='Corner Pass End Locations by Classification',
                    xaxis=dict(title='X Position', range=[120, 0]),
                    yaxis=dict(title='Y Position', range=[0, 80]),
                    height=600,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid location data found for corner passes.")

            st.markdown("#### Corner Outcome Distribution")
            classification_counts = filtered_corners['classification'].value_counts()
            fig_pie = px.pie(
                names=classification_counts.index,
                values=classification_counts.values,
                title="Corner Outcome Breakdown"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No corners found matching the selected filters.")
    else:
        st.info("No corner passes found in the dataset.")

st.download_button(
    "Download CSV", 
    data=filtered.to_csv(index=False), 
    file_name="set_piece_goals.csv",
    key="download_button"
)

st.markdown("""
    <div class="footer">
        <p>© 2025 Outswinger FC Analytics | Powered by Marc Lamberts</p>
    </div>
""", unsafe_allow_html=True)