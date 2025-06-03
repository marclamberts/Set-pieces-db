import streamlit as st
import pandas as pd
import os
import ast
import plotly.graph_objects as go
import plotly.express as px

# -------------------- Config --------------------
st.set_page_config(
    page_title="Set Piece Goals Dashboard",
    layout="wide",
    page_icon="⚽"
)

PASSWORD = "PrincessWay2526"

# -------------------- CSS --------------------
professional_style = """
<style>
    .main { background-color: #f8f9fa; }
    .sidebar .sidebar-content { background-color: #ffffff; box-shadow: 2px 0 10px rgba(0,0,0,0.1); }
    h1, h2, h3, h4, h5, h6 { color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stButton>button {
        background-color: #3498db; color: white; border-radius: 6px;
        border: none; padding: 8px 16px; font-weight: 500; transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9; transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px; border-radius: 8px 8px 0 0; background-color: #ecf0f1;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] { background-color: #3498db; color: white; }
    .stDataFrame {
        border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    [data-testid="metric-container"] {
        background-color: white; border-radius: 8px; padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #3498db;
    }
</style>
"""
st.markdown(professional_style, unsafe_allow_html=True)

# -------------------- Auth --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Set Piece Analysis Dashboard")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### Please authenticate to continue")
        password = st.text_input("Enter password:", type="password")
        if st.button("Login"):
            if password == PASSWORD:
                st.session_state.authenticated = True
                st.experimental_rerun()
            else:
                st.error("Incorrect password")
        st.caption("© 2023 Football Analytics Team")
    st.stop()

# -------------------- Data Load --------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    df = pd.read_excel(os.path.join(base_path, "db.xlsx"))
    df["competition.country_name"] = df["competition.country_name"].astype(str).str.strip()
    st.session_state.available_nations = sorted(df["competition.country_name"].dropna().unique().tolist())
    return df

with st.spinner("Loading data..."):
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

df = df.drop_duplicates(
    subset=['location_x', 'location_y', 'shot.statsbomb_xg', 'team.name', 'player.name', 'Match', 'shot.body_part.name'],
    keep='first'
)
df = df[df["location_x"].notna() & df["shot.statsbomb_xg"].notna()]
df_goals = df[(df["shot.outcome.name"] == "Goal") & (df["location_x"] >= 60)].copy()

# -------------------- Sidebar Filters --------------------
with st.sidebar:
    st.markdown("### 🔍 Filter Options")
    with st.expander("⚙️ Filter Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            filters = {}
            filters["Set Piece Type"] = st.selectbox("Set Piece", ["All"] + sorted(df["play_pattern.name"].dropna().unique().tolist()))
            filters["Team"] = st.selectbox("Team", ["All"] + sorted(df["team.name"].dropna().unique().tolist()))
            filters["Position"] = st.selectbox("Position", ["All"] + sorted(df["position.name"].dropna().unique().tolist()))
            filters["Nation"] = st.selectbox("Nation", ["All"] + st.session_state.available_nations)
        with col2:
            filters["Match"] = st.selectbox("Match", ["All"] + sorted(df["Match"].dropna().unique().tolist()))
            filters["Body Part"] = st.selectbox("Body Part", ["All"] + sorted(df["shot.body_part.name"].dropna().unique().tolist()))
            filters["League"] = st.selectbox("League", ["All"] + sorted(df["competition.competition_name"].dropna().unique().tolist()))
            filters["Season"] = st.selectbox("Season", ["All"] + sorted(df["season.season_name"].dropna().unique().tolist()))
        filters["First-Time"] = st.selectbox("First-Time Shot", ["All", "Yes", "No"])
        xg_range = st.slider("xG Range", float(df["shot.statsbomb_xg"].min()), float(df["shot.statsbomb_xg"].max()), (0.0, 1.0), 0.01)

# -------------------- Apply Filters --------------------
filtered = df_goals.copy()
for key, col in [
    ("Set Piece Type", "play_pattern.name"), ("Team", "team.name"), ("Match", "Match"),
    ("Position", "position.name"), ("Body Part", "shot.body_part.name"),
    ("Nation", "competition.country_name"), ("League", "competition.competition_name"),
    ("Season", "season.season_name")
]:
    if filters[key] != "All":
        filtered = filtered[filtered[col] == filters[key]]
if filters["First-Time"] != "All":
    filtered = filtered[filtered["shot.first_time"] == (filters["First-Time"] == "Yes")]
filtered = filtered[filtered["shot.statsbomb_xg"].between(*xg_range)]

if filtered.empty:
    st.warning("No goals found matching these filters. Please adjust your criteria.")
    st.stop()

# -------------------- Header --------------------
st.title("⚽ Set Piece Goals Analysis")
st.markdown("---")

# -------------------- Metrics --------------------
st.subheader("📊 Overview Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Goals", len(filtered))
with col2:
    st.metric("Avg. xG", f"{filtered['shot.statsbomb_xg'].mean():.3f}")
with col3:
    st.metric("Top Team", filtered['team.name'].mode()[0] if not filtered.empty else "N/A")
with col4:
    st.metric("Most Common Type", filtered['play_pattern.name'].mode()[0] if not filtered.empty else "N/A")

# -------------------- Tabs --------------------
tab0, tab1, tab2, tab3 = st.tabs(["📌 Dashboard", "📊 Goal Map", "📋 Data Explorer", "📈 xG Analysis"])

with tab0:
    st.markdown("### 📌 Summary Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Teams**")
        st.bar_chart(filtered["team.name"].value_counts().head(5))
    with col2:
        st.markdown("**Top Players**")
        st.bar_chart(filtered["player.name"].value_counts().head(5))
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Body Part Distribution**")
        st.dataframe(filtered["shot.body_part.name"].value_counts().reset_index().rename(columns={"index": "Body Part", "shot.body_part.name": "Goals"}))
    with col4:
        st.markdown("**Set Piece Types**")
        st.dataframe(filtered["play_pattern.name"].value_counts().reset_index().rename(columns={"index": "Type", "play_pattern.name": "Goals"}))

with tab1:
    st.markdown("### 🎯 Goal Locations")
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[60, 120], showgrid=False, zeroline=False, visible=False),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.add_trace(go.Scatter(
        x=filtered["location_y"], y=filtered["location_x"],
        mode='markers',
        marker=dict(size=filtered["shot.statsbomb_xg"] * 40 + 6, color='#3498db', line=dict(width=1, color='#2c3e50'), opacity=0.8),
        text=filtered.apply(lambda row: f"<b>Team:</b> {row['team.name']}<br><b>Player:</b> {row.get('player.name', 'N/A')}<br><b>Body Part:</b> {row['shot.body_part.name']}<br><b>Match:</b> {row['Match']}<br><b>xG:</b> {row['shot.statsbomb_xg']:.2f}", axis=1),
        hoverinfo='text'
    ))
    fig.update_layout(shapes=[
        dict(type='rect', x0=0, x1=80, y0=60, y1=120, line=dict(color='#2c3e50', width=2)),
        dict(type='rect', x0=18, x1=62, y0=96, y1=120, line=dict(color='#2c3e50')),
        dict(type='rect', x0=30, x1=50, y0=114, y1=120, line=dict(color='#2c3e50')),
        dict(type='line', x0=40, x1=40, y0=60, y1=120, line=dict(color='#7f8c8d', dash='dash'))
    ])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🔍 Detailed Data View")
    st.dataframe(filtered, use_container_width=True)

with tab3:
    st.markdown("### 📈 xG Distribution & Insights")
    viz_type = st.selectbox("Select Visualization Type", [
        "xG Histogram", "Average xG by Category", "Top xG Goals",
        "xG vs. Goal Map", "xG Density (Box Plot)"
    ])
    if viz_type == "xG Histogram":
        st.bar_chart(filtered["shot.statsbomb_xg"])

    elif viz_type == "Average xG by Category":
        category = st.selectbox("Group by:", ["Team", "Body Part", "Play Pattern", "Position", "Nation", "League"])
        group_col = {
            "Team": "team.name", "Body Part": "shot.body_part.name",
            "Play Pattern": "play_pattern.name", "Position": "position.name",
            "Nation": "competition.country_name", "League": "competition.competition_name"
        }[category]
        xg_by_cat = filtered.groupby(group_col)["shot.statsbomb_xg"].mean().sort_values(ascending=False)
        st.bar_chart(xg_by_cat)

    elif viz_type == "Top xG Goals":
        top_goals = filtered.sort_values(by="shot.statsbomb_xg", ascending=False).head(10)
        st.dataframe(top_goals[["team.name", "player.name", "shot.statsbomb_xg", "Match", "play_pattern.name"]])

    elif viz_type == "xG vs. Goal Map":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered["location_y"], y=filtered["location_x"],
            mode='markers',
            marker=dict(
                size=filtered["shot.statsbomb_xg"] * 40 + 6,
                color=filtered["shot.statsbomb_xg"],
                colorscale="Viridis", showscale=True,
                colorbar=dict(title="xG")
            ),
            text=filtered.apply(lambda row: f"{row['player.name']} ({row['team.name']})<br>xG: {row['shot.statsbomb_xg']:.2f}", axis=1),
            hoverinfo='text'
        ))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=600, plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "xG Density (Box Plot)":
        category = st.selectbox("Boxplot by:", ["Team", "Body Part", "Play Pattern"])
        group_col = {
            "Team": "team.name",
            "Body Part": "shot.body_part.name",
            "Play Pattern": "play_pattern.name"
        }[category]
        fig = px.box(filtered, x=group_col, y="shot.statsbomb_xg", color=group_col)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- Download --------------------
st.markdown("---")
st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=filtered.to_csv(index=False),
    file_name="set_piece_goals_analysis.csv",
    mime="text/csv",
    use_container_width=True
)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>© 2025 Outswinger FC - Data Consultancy | Powered by Streamlit</p>
    </div>
""", unsafe_allow_html=True)
