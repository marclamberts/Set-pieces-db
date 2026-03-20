import streamlit as st
import pandas as pd
import plotly.express as px

# ---- CONFIG ----
st.set_page_config(page_title="Corners Dashboard", layout="wide")

st.title("⚽ Allsvenskan Corners Dashboard")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_excel("Allsvenskan - Corners 2025.xlsx")
    return df

df = load_data()

# ---- CLEAN COLUMN NAMES ----
df.columns = df.columns.str.strip()

# Try to standardize expected column names
# Adjust these if your file uses different names
home_col = [c for c in df.columns if "home" in c.lower()][0]
away_col = [c for c in df.columns if "away" in c.lower()][0]
home_corners_col = [c for c in df.columns if "home" in c.lower() and "corner" in c.lower()][0]
away_corners_col = [c for c in df.columns if "away" in c.lower() and "corner" in c.lower()][0]

# ---- CREATE TOTALS ----
df["Total Corners"] = df[home_corners_col] + df[away_corners_col]

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Filters")

teams = sorted(set(df[home_col]).union(set(df[away_col])))
selected_team = st.sidebar.selectbox("Select Team", ["All"] + teams)

min_corners = st.sidebar.slider("Min Total Corners", 0, int(df["Total Corners"].max()), 0)

# ---- FILTER DATA ----
filtered_df = df.copy()

if selected_team != "All":
    filtered_df = filtered_df[
        (filtered_df[home_col] == selected_team) |
        (filtered_df[away_col] == selected_team)
    ]

filtered_df = filtered_df[filtered_df["Total Corners"] >= min_corners]

# ---- METRICS ----
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Matches", len(filtered_df))
col2.metric("Avg Corners", round(filtered_df["Total Corners"].mean(), 2))
col3.metric("Max Corners", int(filtered_df["Total Corners"].max()))

# ---- HISTOGRAM ----
st.subheader("📈 Total Corners Distribution")

fig_hist = px.histogram(
    filtered_df,
    x="Total Corners",
    nbins=20,
    title="Distribution of Total Corners"
)

st.plotly_chart(fig_hist, use_container_width=True)

# ---- TEAM ANALYSIS ----
st.subheader("🏟️ Team Analysis")

# Home + Away combined stats
home_stats = df.groupby(home_col)[home_corners_col].mean()
away_stats = df.groupby(away_col)[away_corners_col].mean()

team_avg = (home_stats + away_stats) / 2
team_avg = team_avg.sort_values(ascending=False).reset_index()
team_avg.columns = ["Team", "Avg Corners"]

fig_bar = px.bar(
    team_avg,
    x="Team",
    y="Avg Corners",
    title="Average Corners per Team"
)

st.plotly_chart(fig_bar, use_container_width=True)

# ---- MATCH TABLE ----
st.subheader("📋 Match Data")

st.dataframe(filtered_df, use_container_width=True)
