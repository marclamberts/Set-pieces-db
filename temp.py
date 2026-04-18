import os, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Allsvenskan Set Piece Studio",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Colour palette ──────────────────────────────────────────────────────────
BG     = "#07111f"
BG_2   = "#0b1730"
CARD   = "#101a2b"
CARD_2 = "#16243a"
TEXT   = "#f3f7fc"
MUTED  = "#99adc7"
ACCENT = "#5da8ff"
SUCCESS= "#34d399"
WARNING= "#fbbf24"
PURPLE = "#a78bfa"
ORANGE = "#fb923c"
RED    = "#f87171"
BORDER = "rgba(255,255,255,0.08)"

TYPE_COLORS = {"Corner":"#5da8ff","Free Kick":"#34d399","Throw-In":"#fb923c","Other":"#a78bfa"}
OUTCOME_COLORS = {"Goal":"#34d399","Saved":"#fbbf24","Off T":"#fb923c","Blocked":"#f87171","Wayward":"#99adc7","Saved Off Target":"#a78bfa","Post":"#8ad6ff"}
QUAL_PALETTE   = [ACCENT,SUCCESS,WARNING,PURPLE,ORANGE,"#8ad6ff","#6ee7b7","#f472b6"]
px.defaults.template = "plotly_dark"

# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
body,.stApp{{
  background:
    radial-gradient(ellipse 1200px 700px at 90% -10%,rgba(93,168,255,0.12) 0%,transparent 60%),
    radial-gradient(ellipse 900px 600px at -10% 20%,rgba(52,211,153,0.08) 0%,transparent 55%),
    linear-gradient(180deg,{BG} 0%,{BG_2} 100%);
  color:{TEXT};
}}
.block-container{{max-width:1620px;padding-top:1rem;padding-bottom:2rem;}}
header[data-testid="stHeader"]{{background:rgba(0,0,0,0);}}
#MainMenu,footer{{visibility:hidden;}}
.hero{{
  background:linear-gradient(135deg,rgba(93,168,255,0.18) 0%,rgba(93,168,255,0.05) 55%,rgba(52,211,153,0.08) 100%);
  border:1px solid rgba(93,168,255,0.18);border-radius:30px;
  padding:30px 34px 22px 34px;box-shadow:0 16px 48px rgba(0,0,0,0.22);margin-bottom:16px;
}}
.hero-title{{font-size:2.6rem;font-weight:900;line-height:1;letter-spacing:-.03em;margin-bottom:.5rem;}}
.hero-title span{{color:{ACCENT};}}
.hero-sub{{color:{MUTED};font-size:1rem;line-height:1.6;max-width:960px;}}
.kpi{{background:linear-gradient(160deg,{CARD} 0%,{CARD_2} 100%);border:1px solid {BORDER};border-radius:18px;padding:14px;min-height:96px;}}
.kpi-label{{color:{MUTED};text-transform:uppercase;font-size:.68rem;letter-spacing:.1em;font-weight:700;}}
.kpi-value{{margin-top:8px;font-size:1.75rem;font-weight:900;line-height:1;}}
.kpi-foot{{margin-top:6px;font-size:.82rem;color:{MUTED};}}
.segment-card{{background:linear-gradient(160deg,{CARD} 0%,{CARD_2} 100%);border:1px solid {BORDER};border-radius:26px;padding:20px;min-height:210px;box-shadow:0 10px 30px rgba(0,0,0,0.18);}}
.segment-pill{{display:inline-block;padding:.3rem .65rem;border-radius:999px;font-size:.78rem;font-weight:700;border:1px solid rgba(255,255,255,0.12);margin-bottom:.7rem;}}
.segment-title{{font-size:1.4rem;font-weight:900;margin-bottom:.35rem;}}
.segment-sub{{color:{MUTED};font-size:.93rem;line-height:1.55;min-height:64px;}}
.section-title{{font-size:1.1rem;font-weight:850;margin:.1rem 0 .2rem 0;}}
.section-sub{{color:{MUTED};font-size:.9rem;margin-bottom:.75rem;}}
.panel{{background:rgba(255,255,255,0.02);border:1px solid {BORDER};border-radius:22px;padding:16px 18px 10px 18px;margin-bottom:12px;}}
.empty-state{{text-align:center;padding:52px 24px;color:{MUTED};font-size:.94rem;border:1px dashed rgba(255,255,255,0.10);border-radius:18px;background:rgba(255,255,255,0.015);}}
.footer-note{{color:#6b87a8;font-size:.82rem;margin-top:1rem;padding-top:10px;border-top:1px solid {BORDER};}}
div[data-testid="stDataFrame"]{{border:1px solid {BORDER};border-radius:14px;overflow:hidden;}}
.stTabs [data-baseweb="tab-list"]{{gap:6px;background:rgba(255,255,255,0.03);border-radius:14px;padding:4px;border:1px solid {BORDER};}}
.stTabs [aria-selected="true"]{{background:rgba(93,168,255,0.18)!important;color:#d4e8ff!important;}}
div.stButton>button{{width:100%;border-radius:14px;border:1px solid rgba(255,255,255,0.10);background:rgba(255,255,255,0.03);color:{TEXT};font-weight:700;padding:.65rem .85rem;}}
div.stButton>button:hover{{border-color:rgba(93,168,255,0.30);background:rgba(93,168,255,0.10);}}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_num(s): return pd.to_numeric(s, errors="coerce")
def hpct(v,d=1):  return "—" if pd.isna(v) else f"{v*100:.{d}f}%"
def hval(v,d=2):  return "—" if pd.isna(v) else f"{v:.{d}f}"

def find_col(df, candidates):
    lm = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lm: return lm[c.lower()]
    for c in df.columns:
        for cand in candidates:
            if cand.lower() in str(c).lower(): return c
    return None

def parse_xy(cell, idx=0):
    if pd.isna(cell): return np.nan
    try:
        parts = [float(x.strip()) for x in str(cell).split(",")]
        return parts[idx] if len(parts) > idx else np.nan
    except: return np.nan

def parse_freeze_frame(cell):
    """Return list of (x, y) tuples from packed freeze-frame string."""
    if pd.isna(cell): return []
    try:
        nums = [float(x.strip()) for x in str(cell).split(",")]
        return [(nums[i], nums[i+1]) for i in range(0, len(nums)-1, 2)]
    except: return []

def sp_bucket(s):
    s = str(s).lower()
    if "corner" in s: return "Corner"
    if "free kick" in s: return "Free Kick"
    if "throw" in s: return "Throw-In"
    return "Other"

def side_y(y):
    if pd.isna(y): return "Unknown"
    return "Right" if y < 40 else "Left"

def delivery_zone(y):
    if pd.isna(y): return "Unknown"
    if y < 25: return "Near Post"
    if y <= 55: return "Central"
    return "Far Post"

def end_zone(x, y):
    if pd.isna(x) or pd.isna(y): return "Unknown"
    if x >= 114 and 28 <= y <= 52: return "6-yard box"
    if x >= 108 and 18 <= y <= 62: return "Penalty area"
    if x >= 100 and 18 <= y <= 62: return "Deep box"
    return "Outside danger"

def figure_base(fig, h=420, title=None):
    fig.update_layout(
        height=h, title=title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8,r=8,t=44 if title else 8,b=8),
        font=dict(color=TEXT), legend_title_text="",
        hoverlabel=dict(bgcolor="#0d1c31", font_color=TEXT),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig

def sh(title, sub=""):
    st.markdown(
        f'<div class="section-title">{title}</div>' +
        (f'<div class="section-sub">{sub}</div>' if sub else ""),
        unsafe_allow_html=True)

def kpi(label, value, foot=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>', unsafe_allow_html=True)

def empty(msg="No data for current selection."):
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)

# ── Pitch drawing ─────────────────────────────────────────────────────────────
PITCH_GREEN = "#1a4731"
PITCH_LINE  = "rgba(255,255,255,0.85)"

def add_pitch_shapes(fig, half=True, flip=False):
    """
    Statsbomb coordinate system: x=0..120 (length), y=0..80 (width)
    We display attack end (x=100..120) with x on horizontal axis,
    y on vertical axis.
    half=True → only show attacking half (x >= 60)
    flip=False → y as-is (y=0 bottom, y=80 top; keeper at y=80 right side, y=0 left side)
    """
    shapes = []
    # pitch background
    x_min = 60 if half else 0
    x_max = 120

    # outer pitch boundary
    shapes.append(dict(type="rect", x0=x_min, y0=0, x1=x_max, y1=80,
                       fillcolor=PITCH_GREEN, opacity=0.35, line=dict(color=PITCH_LINE, width=2), layer="below"))
    # half-way line
    if not half:
        shapes.append(dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=PITCH_LINE, width=1.5, dash="dash")))
    # penalty area
    shapes.append(dict(type="rect", x0=102, y0=18, x1=120, y1=62,
                       line=dict(color=PITCH_LINE, width=1.5), fillcolor="rgba(0,0,0,0)"))
    # 6-yard box
    shapes.append(dict(type="rect", x0=114, y0=30, x1=120, y1=50,
                       line=dict(color=PITCH_LINE, width=1.5), fillcolor="rgba(0,0,0,0)"))
    # goal
    shapes.append(dict(type="line", x0=120, y0=36, x1=120, y1=44,
                       line=dict(color="#00FF00", width=5)))
    # penalty spot
    shapes.append(dict(type="circle", x0=107.8, y0=39.8, x1=108.2, y1=40.2,
                       fillcolor="white", line=dict(color="white", width=1)))
    fig.update_layout(shapes=shapes)
    return fig

def pitch_layout(fig, h=580, half=True, title=None):
    x_min = 58 if half else -2
    fig.update_xaxes(range=[x_min, 122], visible=False)
    fig.update_yaxes(range=[-2, 82], visible=False)
    fig.update_layout(
        height=h, title=title,
        paper_bgcolor="#0b1421", plot_bgcolor=PITCH_GREEN,
        margin=dict(l=10,r=10,t=44 if title else 10,b=10),
        font=dict(color=TEXT), legend_title_text="",
        hoverlabel=dict(bgcolor="#0d1c31", font_color=TEXT, font_size=13),
    )
    add_pitch_shapes(fig, half=half)
    return fig

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_DIR = ""

@st.cache_data
def load_all():
    swe  = pd.read_excel(os.path.join(DATA_DIR, "SWE SP.xlsx"))
    cors = pd.read_excel(os.path.join(DATA_DIR, "Allsvenskan - Corners 2025.xlsx"))
    dlay = pd.read_excel(os.path.join(DATA_DIR, "corner_delays.xlsx"))
    duel = pd.read_excel(os.path.join(DATA_DIR, "duel_hops_rating_summary.xlsx"))
    return swe, cors, dlay, duel

@st.cache_data
def prepare_swe(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={
        "match_id":"match_id","team.name":"team","SP_Type":"sp_type_raw",
        "Taker":"Taker","Shooter":"Shooter","shot.statsbomb_xg":"shot_xg",
        "shot.outcome.name":"shot_outcome","shot.freeze_frame":"freeze_frame",
        "shot_x":"shot_x","shot_y":"shot_y",
        "Occupation_Rating":"Occ_Rating","Proximity_Rating":"Prox_Rating",
        "Duel_Win_Prob":"Duel_Win_Prob","OPS_Opponent_Rating":"OPS_Rating",
    })
    # parse timestamp → minute
    ts = df["timestamp"].astype(str).str.split(":", expand=True)
    if ts.shape[1] >= 3:
        df["Minute"] = safe_num(ts[1]).fillna(0)
        df["Second"] = safe_num(ts[2].str.replace(r"[^0-9.]","",regex=True)).fillna(0)
    else:
        df["Minute"] = 0; df["Second"] = 0

    df["pass_x"] = df["location.pass"].apply(lambda v: parse_xy(v, 0))
    df["pass_y"] = df["location.pass"].apply(lambda v: parse_xy(v, 1))
    df["shot_z"] = df["location.shot"].apply(lambda v: parse_xy(v, 2)) if "location.shot" in df.columns else np.nan
    df["shot_xg"] = safe_num(df["shot_xg"]).fillna(0)
    df["sp_type"] = df["sp_type_raw"].apply(sp_bucket)
    df["side"]    = df["pass_y"].apply(side_y)
    df["end_z"]   = df.apply(lambda r: end_zone(r.get("shot_x"), r.get("shot_y")), axis=1)
    df["dzone"]   = df["shot_y"].apply(delivery_zone)
    df["goal"]    = df["shot_outcome"].astype(str).str.lower().str.contains("goal", na=False)
    df["led_shot"]= (df["shot_xg"] > 0) | (df["shot_outcome"].notna())
    df["event_min"]= df["Minute"] + df["Second"]/60
    df["phase"]   = pd.cut(df["event_min"], bins=[-0.1,15,30,45,60,75,120],
                           labels=["0-15","16-30","31-45","46-60","61-75","76+"],right=True).astype(str)
    # team match label (from possession col if no match name)
    df["match_label"] = "Match " + df["match_id"].astype(str)
    return df

@st.cache_data
def prepare_corners(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["sp_type"] = "Corner"
    df["side"] = df["pass_location_y"].apply(side_y)
    df["dzone"] = df["pass_end_location_y"].apply(delivery_zone)
    df["end_z"] = df.apply(lambda r: end_zone(r["pass_end_location_x"], r["pass_end_location_y"]), axis=1)
    df["shot_xg"] = safe_num(df["shot.statsbomb_xg"]).fillna(0)
    df["led_shot"] = (df["shot_xg"] > 0) | df["Shooter"].notna()
    df["goal"] = df["SP_outcome"].astype(str).str.lower().str.contains("goal", na=False) | \
                 df["shot.outcome.name"].astype(str).str.lower().str.contains("goal", na=False)
    df["shot_outcome"] = df["shot.outcome.name"]
    df["technique"]    = df["pass.technique.name"]
    df["body_part"]    = df["pass.body_part.name"]
    df["height"]       = df["pass.height.name"]
    df["team"]         = df["pass_team_name"]
    df["pass_x"]       = df["pass_location_x"]
    df["pass_y"]       = df["pass_location_y"]
    df["shot_x"]       = df["shot_location_x"]
    df["shot_y"]       = df["shot_location_y"]
    df["match_label"]  = df["Match"]
    return df

@st.cache_data
def prepare_delays(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # extract match_id from filename
    df["match_id"] = df["match"].str.extract(r"match_(\d+)").astype(float)
    return df

# Load
swe_raw, cors_raw, dlay_raw, duel_raw = load_all()
swe  = prepare_swe(swe_raw)
cors = prepare_corners(cors_raw)
dlay = prepare_delays(dlay_raw)
duel = duel_raw.copy()
duel.columns = [str(c).strip() for c in duel.columns]

# ── Shotmap with freeze-frame overlay ────────────────────────────────────────
def shotmap_figure(df_shots, duel_df=None, title="Shotmap"):
    fig = go.Figure()
    pitch_layout(fig, h=600, half=True, title=title)

    if df_shots.empty:
        return fig

    shots = df_shots[df_shots["shot_xg"] > 0].copy()
    if shots.empty:
        shots = df_shots.copy()

    # colour by outcome
    for outcome, grp in shots.groupby("shot_outcome", dropna=False):
        color = OUTCOME_COLORS.get(str(outcome), MUTED)
        size  = np.clip(safe_num(grp["shot_xg"]) * 180 + 12, 12, 60).fillna(14)

        # build rich hover
        hover_texts = []
        for _, r in grp.iterrows():
            taker   = str(r.get("Taker","—"))
            shooter = str(r.get("Shooter","—")) if "Shooter" in r.index else str(r.get("shooter","—"))
            team    = str(r.get("team","—"))
            xg      = r.get("shot_xg", 0)
            occ     = r.get("Occ_Rating", np.nan)
            prox    = r.get("Prox_Rating", np.nan)
            duel_p  = r.get("Duel_Win_Prob", np.nan)
            ops     = r.get("OPS_Rating", np.nan)
            ot      = str(r.get("shot_outcome","—"))
            minute  = int(r.get("Minute",0))
            sx      = r.get("shot_x", np.nan)
            sy      = r.get("shot_y", np.nan)

            # duel rating lookup
            dr_row = None
            if duel_df is not None and shooter != "—" and not duel_df.empty:
                match_mask = duel_df["Player"].astype(str).str.strip() == shooter.strip()
                if match_mask.any():
                    dr_row = duel_df[match_mask].iloc[0]

            txt  = f"<b>⚽ {shooter}</b><br>"
            txt += f"Team: {team}  |  Min: {minute}'<br>"
            txt += f"Taker: {taker}<br>"
            txt += f"<b>xG: {xg:.3f}</b>  Outcome: {ot}<br>"
            if not pd.isna(sx): txt += f"Location: ({sx:.1f}, {sy:.1f})<br>"
            txt += "──────────────────<br>"
            if not pd.isna(occ):  txt += f"Occupation Rating: {occ:.2f}<br>"
            if not pd.isna(prox): txt += f"Proximity Rating: {prox:.2f}<br>"
            if not pd.isna(duel_p): txt += f"Duel Win Prob: {duel_p:.2f}<br>"
            if not pd.isna(ops):  txt += f"OPS Opponent Rating: {ops:.2f}<br>"
            if dr_row is not None:
                txt += f"<b>Duel Hops Rating: {dr_row['Rating']:.3f}</b>"
            hover_texts.append(txt)

        fig.add_trace(go.Scatter(
            x=shots[shots["shot_outcome"]==outcome]["shot_x"] if "shot_outcome" in shots.columns else grp["shot_x"],
            y=grp["shot_y"],
            mode="markers",
            name=str(outcome),
            marker=dict(size=size, color=color, opacity=0.82,
                        line=dict(color="white", width=1.2)),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            customdata=grp[["shot_xg"]].values,
        ))

    # Add freeze-frame for SELECTED shot on second call (expensive – skip by default)
    return fig

def shotmap_with_freeze(row, title="Shot Detail"):
    """Single-shot detail figure with freeze-frame player positions."""
    fig = go.Figure()
    pitch_layout(fig, h=560, half=True, title=title)

    sx = row.get("shot_x", np.nan)
    sy = row.get("shot_y", np.nan)
    if pd.isna(sx) or pd.isna(sy):
        return fig

    # freeze frame defenders (packed x,y pairs)
    ff = parse_freeze_frame(row.get("freeze_frame", np.nan))

    if ff:
        fx = [p[0] for p in ff]
        fy = [p[1] for p in ff]
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="markers",
            name="Defenders",
            marker=dict(size=14, color=RED, opacity=0.7,
                        symbol="circle", line=dict(color="white",width=1)),
            hovertemplate="Defender position<br>x:%{x:.1f}, y:%{y:.1f}<extra></extra>",
        ))

    # shot location
    fig.add_trace(go.Scatter(
        x=[sx], y=[sy], mode="markers",
        name="Shot",
        marker=dict(size=22, color=SUCCESS, opacity=0.9,
                    symbol="star", line=dict(color="white",width=1.5)),
        hovertemplate=f"<b>xG: {row.get('shot_xg',0):.3f}</b><br>x:{sx:.1f}, y:{sy:.1f}<extra></extra>",
    ))
    return fig

# ── Delivery map (pass start → end) ──────────────────────────────────────────
def delivery_map(df, title="Delivery Map"):
    fig = go.Figure()
    pitch_layout(fig, h=600, half=True, title=title)

    plot = df.dropna(subset=["pass_x","pass_y","shot_x","shot_y"]).copy()
    if plot.empty:
        return fig

    for zone, grp in plot.groupby("dzone", dropna=False):
        idx = list(["Near Post","Central","Far Post","Unknown"]).index(zone) if zone in ["Near Post","Central","Far Post","Unknown"] else 3
        color = QUAL_PALETTE[idx % len(QUAL_PALETTE)]

        # draw arrows pass → end
        for _, r in grp.iterrows():
            fig.add_annotation(
                ax=r["pass_x"], ay=r["pass_y"],
                x=r["shot_x"],  y=r["shot_y"],
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=color, opacity=0.35, showarrow=True,
            )

        hover = [
            f"<b>{r.get('Taker','—')}</b> → {r.get('Shooter','—')}<br>"
            f"Team: {r.get('team','—')}<br>"
            f"Zone: {zone}<br>xG: {r.get('shot_xg',0):.3f}<br>"
            f"Outcome: {r.get('shot_outcome','—')}"
            for _, r in grp.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=grp["shot_x"], y=grp["shot_y"],
            mode="markers", name=str(zone),
            marker=dict(size=10, color=color, opacity=0.85,
                        line=dict(color="white",width=0.8)),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    return fig

# ── Routines map (pass origin heat / cluster) ─────────────────────────────────
def routines_map(df, title="Delivery Origin Map"):
    """Show where deliveries originate from across the pitch."""
    fig = go.Figure()
    pitch_layout(fig, h=560, half=False, title=title)

    plot = df.dropna(subset=["pass_x","pass_y"]).copy()
    if plot.empty:
        return fig

    for sp, grp in plot.groupby("sp_type", dropna=False):
        color = TYPE_COLORS.get(sp, MUTED)
        hover = [
            f"<b>{r.get('Taker','—')}</b><br>Team: {r.get('team','—')}<br>"
            f"Type: {sp}<br>Outcome: {r.get('shot_outcome','—')}"
            for _, r in grp.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=grp["pass_x"], y=grp["pass_y"],
            mode="markers", name=sp,
            marker=dict(size=9, color=color, opacity=0.7,
                        line=dict(color="white",width=0.5)),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))
    return fig

# ── Summary builders ──────────────────────────────────────────────────────────
def team_summary(df):
    if df.empty: return pd.DataFrame()
    return (
        df.groupby("team", dropna=False)
        .agg(Events=("match_id","size"),
             Matches=("match_id",pd.Series.nunique),
             Shots=("led_shot","sum"),
             xG=("shot_xg","sum"),
             Goals=("goal","sum"))
        .assign(
            ShotRate=lambda d: d.Shots/d.Events.replace(0,np.nan),
            xG_per_event=lambda d: d.xG/d.Events.replace(0,np.nan),
        )
        .sort_values("xG_per_event", ascending=False)
        .reset_index()
    )

def taker_summary(df):
    if df.empty: return pd.DataFrame()
    return (
        df.groupby(["team","Taker"], dropna=False)
        .agg(Events=("match_id","size"),
             Shots=("led_shot","sum"),
             xG=("shot_xg","sum"),
             Goals=("goal","sum"))
        .assign(
            ShotRate=lambda d: d.Shots/d.Events.replace(0,np.nan),
            xG_per_event=lambda d: d.xG/d.Events.replace(0,np.nan),
        )
        .sort_values(["Events","xG_per_event"], ascending=False)
        .reset_index()
    )

# ── Navigation ────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "home"

def nav(page): st.session_state["page"] = page

# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────
def home():
    st.markdown("""
    <div class="hero">
      <div class="hero-title">Allsvenskan <span>Set Piece Studio</span></div>
      <div class="hero-sub">
        Advanced set-piece analytics combining delivery intelligence, shotmap hover detail,
        freeze-frame defender positioning, duel ratings, and corner delay profiles across Allsvenskan 2025.
        Choose a segment below to begin.
      </div>
    </div>""", unsafe_allow_html=True)

    # top-line KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: kpi("Matches", f"{cors['match_id'].nunique():,}", "Corners dataset")
    with k2: kpi("Corner Events", f"{len(cors):,}", "Delivery attempts")
    with k3:
        shot_r = cors["led_shot"].mean()
        kpi("Shot Rate", hpct(shot_r), "Corners → shots")
    with k4: kpi("Total xG", hval(cors["shot_xg"].sum(), 2), "Corners xG")
    with k5: kpi("Avg Corner Delay", f"{dlay['delay_sec'].mean():.1f}s", "Time until clearance")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # segment cards
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        ("Corner Studio", ACCENT, "corner",
         "Deep-dive corner analysis: delivery maps, routines, shotmaps with freeze-frame defender positions, and taker efficiency."),
        ("Set Pieces (All)", SUCCESS, "setpieces",
         "Unified analysis across all set-piece types from the full SWE SP dataset — corners, free kicks, and throw-ins."),
        ("Corner Delays", ORANGE, "delays",
         "How long does each corner routine take before the ball is cleared? Breakdown by team and outcome type."),
        ("Duel Ratings", PURPLE, "duels",
         "Player-level duel hops ratings across Allsvenskan, showing who wins aerial duels in set-piece situations."),
    ]
    for col, (label, color, page, desc) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f"""
            <div class="segment-card">
              <div class="segment-pill" style="background:{color}22;color:{TEXT};border-color:{color}55;">{label}</div>
              <div class="segment-title">{label}</div>
              <div class="segment-sub">{desc}</div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Open →", key=f"btn_{page}"):
                nav(page); st.rerun()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    sh("League Snapshot", "Quick comparison across all corner outcomes in the dataset")

    c_left, c_right = st.columns(2)
    with c_left:
        outcome_df = cors.groupby("SP_outcome", dropna=False).size().reset_index(name="n").sort_values("n",ascending=False).head(8)
        fig = px.bar(outcome_df, x="SP_outcome", y="n", color="SP_outcome",
                     color_discrete_sequence=QUAL_PALETTE, title="Corner Outcome Distribution")
        st.plotly_chart(figure_base(fig,360,"Corner Outcome Distribution"), use_container_width=True)
    with c_right:
        tech_df = cors.groupby("technique", dropna=False).agg(
            Events=("match_id","size"), xG=("shot_xg","sum")).reset_index()
        tech_df["xG_per_event"] = tech_df.xG / tech_df.Events.replace(0,np.nan)
        fig = px.bar(tech_df.dropna(subset=["technique"]).sort_values("xG_per_event",ascending=False),
                     x="technique", y="xG_per_event", color="technique",
                     color_discrete_sequence=QUAL_PALETTE, title="xG/Event by Delivery Technique")
        st.plotly_chart(figure_base(fig,360,"xG/Event by Delivery Technique"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORNER STUDIO
# ─────────────────────────────────────────────────────────────────────────────
def corner_studio():
    c0, c_hero = st.columns([1,8])
    with c0:
        if st.button("← Home"): nav("home"); st.rerun()
    with c_hero:
        st.markdown("""<div class="hero" style="padding:22px 26px 18px 26px;">
          <div class="hero-title" style="font-size:2.1rem;">Corner <span>Studio</span></div>
          <div class="hero-sub">Delivery maps, routines, shotmaps with freeze-frame hover, and taker efficiency.</div>
        </div>""", unsafe_allow_html=True)

    df = cors.copy()
    teams   = sorted(df["team"].dropna().unique().tolist())
    takers  = sorted(df["Taker"].dropna().astype(str).unique().tolist())
    matches = sorted(df["match_label"].dropna().unique().tolist())

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    f1,f2,f3,f4,f5 = st.columns(5)
    with f1: team_f = st.selectbox("Team", ["All"]+teams, key="c_team")
    with f2: side_f = st.selectbox("Side", ["Both","Left","Right"], key="c_side")
    with f3: tech_f = st.selectbox("Technique", ["All"]+sorted(df["technique"].dropna().unique().tolist()), key="c_tech")
    with f4: taker_f= st.multiselect("Takers", takers, key="c_taker")
    with f5: shot_f = st.checkbox("Shots only", key="c_shot")
    st.markdown('</div>', unsafe_allow_html=True)

    w = df.copy()
    if team_f != "All":  w = w[w["team"]==team_f]
    if side_f != "Both": w = w[w["side"]==side_f]
    if tech_f != "All":  w = w[w["technique"]==tech_f]
    if taker_f:          w = w[w["Taker"].astype(str).isin(taker_f)]
    if shot_f:           w = w[w["led_shot"]]

    if w.empty:
        empty("No events match current filters."); return

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: kpi("Corners", f"{len(w):,}")
    with k2: kpi("Shots", f"{int(w['led_shot'].sum()):,}", "From corners")
    with k3: kpi("Goals", f"{int(w['goal'].sum()):,}")
    with k4: kpi("Shot Rate", hpct(w["led_shot"].mean()))
    with k5: kpi("Total xG", hval(w["shot_xg"].sum(),2))

    tabs = st.tabs(["🗺️ Delivery Map","⚽ Shotmap","📍 Routines","📊 Analysis","👤 Takers","📋 Data"])

    # ── Delivery Map ──
    with tabs[0]:
        sh("Delivery Map", "Where corners land on the pitch, coloured by end zone")
        col1, col2 = st.columns([3,2])
        with col1:
            fig = delivery_map(w, title=f"Corner Delivery Map ({len(w)} events)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sh("Zone Breakdown")
            zone_df = w.groupby("dzone", dropna=False).agg(
                Events=("match_id","size"), xG=("shot_xg","sum"),
                Shots=("led_shot","sum")).reset_index()
            zone_df["ShotRate"] = zone_df.Shots/zone_df.Events.replace(0,np.nan)
            zone_df["xG_per_event"] = zone_df.xG/zone_df.Events.replace(0,np.nan)
            st.dataframe(zone_df.reset_index(drop=True), use_container_width=True, height=220)

            sh("End Zone")
            ez_df = w.groupby("end_z",dropna=False).size().reset_index(name="n").sort_values("n",ascending=False)
            fig2 = px.pie(ez_df, names="end_z", values="n",
                          color_discrete_sequence=QUAL_PALETTE, hole=0.45)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT),height=260,
                               margin=dict(l=8,r=8,t=8,b=8), legend_title_text="")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Shotmap ──
    with tabs[1]:
        sh("Shotmap", "Hover over each marker for xG, metrics, and duel rating info")
        shot_w = w[w["led_shot"]].copy()
        if shot_w.empty:
            empty("No shots in current selection.")
        else:
            col1, col2 = st.columns([3,2])
            with col1:
                fig = shotmap_figure(shot_w, duel_df=duel,
                                     title=f"Shotmap — {len(shot_w)} shots (hover for details)")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                sh("Shot Details Table", "Click a row to see freeze-frame")
                display_cols = ["team","Taker","Shooter","shot_xg","shot_outcome","Minute","shot_x","shot_y"]
                dc = [c for c in display_cols if c in shot_w.columns]
                shot_tbl = shot_w[dc].reset_index(drop=True).round(3)
                sel = st.dataframe(shot_tbl, use_container_width=True, height=320,
                                   on_select="rerun", selection_mode="single-row")
                selected_rows = sel.selection.rows if hasattr(sel,"selection") else []
                if selected_rows:
                    row = shot_w.iloc[selected_rows[0]]
                    st.markdown(f"**Selected:** {row.get('Shooter','—')} | xG {row.get('shot_xg',0):.3f} | {row.get('shot_outcome','—')}")
                    ff_count = len(parse_freeze_frame(row.get("freeze_frame", np.nan)))
                    st.markdown(f"Freeze-frame has **{ff_count}** player positions")
                    if ff_count > 0:
                        fig_ff = shotmap_with_freeze(row, title=f"Freeze Frame — {row.get('Shooter','Shot')}")
                        st.plotly_chart(fig_ff, use_container_width=True)

            # xG distribution
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(shot_w, x="shot_xg", nbins=20, color_discrete_sequence=[ACCENT],
                                   title="xG Distribution")
                st.plotly_chart(figure_base(fig,320,"xG Distribution"), use_container_width=True)
            with c2:
                out_df = shot_w.groupby("shot_outcome",dropna=False).agg(
                    n=("shot_xg","size"), xG=("shot_xg","sum")).reset_index()
                fig = px.bar(out_df.sort_values("n",ascending=False),
                             x="shot_outcome", y="n", color="shot_outcome",
                             color_discrete_map=OUTCOME_COLORS, title="Shot Outcomes")
                st.plotly_chart(figure_base(fig,320,"Shot Outcomes"), use_container_width=True)

    # ── Routines ──
    with tabs[2]:
        sh("Delivery Origin Map", "Where on the pitch corner kicks originate from")
        col1, col2 = st.columns([3,2])
        with col1:
            fig = routines_map(w, title="Corner Kick Origins")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sh("Technique Breakdown")
            tech_df = w.groupby("technique",dropna=False).agg(
                Events=("match_id","size"), xG=("shot_xg","sum"),
                Goals=("goal","sum"), Shots=("led_shot","sum")).reset_index().dropna(subset=["technique"])
            tech_df["xG/event"] = (tech_df.xG / tech_df.Events.replace(0,np.nan)).round(4)
            st.dataframe(tech_df.reset_index(drop=True), use_container_width=True, height=200)
            sh("Side Distribution")
            side_df = w.groupby("side",dropna=False).size().reset_index(name="n")
            fig2 = px.pie(side_df, names="side", values="n", hole=0.45,
                          color_discrete_sequence=[ACCENT, ORANGE])
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT),
                               height=240, margin=dict(l=8,r=8,t=8,b=8), legend_title_text="")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Analysis ──
    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            phase_df = w.groupby("phase",dropna=False).size().reset_index(name="n")
            fig = px.line(phase_df, x="phase", y="n", markers=True,
                          color_discrete_sequence=[ACCENT], title="Corner Volume by Match Phase")
            st.plotly_chart(figure_base(fig,360,"Corner Volume by Match Phase"), use_container_width=True)
        with c2:
            ts = team_summary(w)
            fig = px.scatter(ts, x="ShotRate", y="xG_per_event", size="Events",
                             text="team", color_discrete_sequence=[ACCENT],
                             title="Team Efficiency (Shot Rate vs xG/Event)")
            fig.update_traces(textposition="top center")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(figure_base(fig,360,"Team Efficiency"), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            ts2 = team_summary(w)
            fig = px.bar(ts2.head(14), x="team", y="Events", color="xG_per_event",
                         color_continuous_scale="Blues", title="Events by Team")
            fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-40)
            st.plotly_chart(figure_base(fig,360,"Events by Team"), use_container_width=True)
        with c4:
            fig = px.histogram(w, x="Minute", nbins=24, color_discrete_sequence=[SUCCESS],
                               title="Corner Minute Distribution")
            st.plotly_chart(figure_base(fig,360,"Minute Distribution"), use_container_width=True)

    # ── Takers ──
    with tabs[4]:
        tk = taker_summary(w)
        if tk.empty:
            empty()
        else:
            st.dataframe(tk.reset_index(drop=True), use_container_width=True, height=380)
            plot = tk[tk["Events"]>=2].head(14).copy()
            if not plot.empty:
                plot["label"] = plot["Taker"].astype(str) + " (" + plot["team"].astype(str) + ")"
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(plot.sort_values("xG_per_event",ascending=False),
                                 x="label", y="xG_per_event", color="xG_per_event",
                                 color_continuous_scale="Blues", title="Top Takers — xG/Event")
                    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-40)
                    st.plotly_chart(figure_base(fig,360), use_container_width=True)
                with c2:
                    fig = px.bar(plot.sort_values("ShotRate",ascending=False),
                                 x="label", y="ShotRate", color="ShotRate",
                                 color_continuous_scale="Blues", title="Top Takers — Shot Rate")
                    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-40)
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(figure_base(fig,360), use_container_width=True)

            # duel ratings cross-reference
            sh("Taker Duel Ratings", "Cross-referencing with duel_hops_rating_summary")
            merged = tk.merge(duel.rename(columns={"Player":"Taker","Team":"duel_team","Rating":"DuelRating"}),
                              on="Taker", how="inner")
            if not merged.empty:
                st.dataframe(merged[["Taker","team","Events","xG_per_event","ShotRate","DuelRating"]]
                             .sort_values("DuelRating",ascending=False).reset_index(drop=True),
                             use_container_width=True, height=300)

    # ── Data ──
    with tabs[5]:
        st.dataframe(w.reset_index(drop=True), use_container_width=True, height=500)
        st.download_button("⬇ Download CSV", w.to_csv(index=False).encode(),
                           "corners_filtered.csv", "text/csv", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ALL SET PIECES
# ─────────────────────────────────────────────────────────────────────────────
def setpieces_page():
    c0, c_hero = st.columns([1,8])
    with c0:
        if st.button("← Home"): nav("home"); st.rerun()
    with c_hero:
        st.markdown("""<div class="hero" style="padding:22px 26px 18px 26px;">
          <div class="hero-title" style="font-size:2.1rem;">All <span>Set Pieces</span></div>
          <div class="hero-sub">Full SWE SP dataset — corners, free kicks, throw-ins, with duel metrics overlay.</div>
        </div>""", unsafe_allow_html=True)

    df = swe.copy()
    teams  = sorted(df["team"].dropna().unique().tolist())
    types  = sorted(df["sp_type"].dropna().unique().tolist())
    takers = sorted(df["Taker"].dropna().astype(str).unique().tolist())

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns(4)
    with f1: team_f  = st.selectbox("Team", ["All"]+teams, key="sp_team")
    with f2: type_f  = st.multiselect("Type", types, default=types, key="sp_type")
    with f3: taker_f = st.multiselect("Takers", takers, key="sp_taker")
    with f4: shot_f  = st.checkbox("Shots only", key="sp_shot")
    st.markdown('</div>', unsafe_allow_html=True)

    w = df.copy()
    if team_f != "All":  w = w[w["team"]==team_f]
    if type_f:           w = w[w["sp_type"].isin(type_f)]
    if taker_f:          w = w[w["Taker"].astype(str).isin(taker_f)]
    if shot_f:           w = w[w["led_shot"]]

    if w.empty:
        empty(); return

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: kpi("Events", f"{len(w):,}")
    with k2: kpi("Matches", f"{w['match_id'].nunique():,}")
    with k3: kpi("Shots", f"{int(w['led_shot'].sum()):,}")
    with k4: kpi("Shot Rate", hpct(w["led_shot"].mean()))
    with k5: kpi("Total xG", hval(w["shot_xg"].sum(),2))

    tabs = st.tabs(["⚽ Shotmap","🗺️ Delivery Map","📍 Origins","📊 Overview","📋 Teams","👤 Takers","📋 Data"])

    with tabs[0]:
        sh("Shotmap", "Hover for full xG details, duel metrics, and player ratings")
        shot_w = w[w["led_shot"]].copy()
        if shot_w.empty:
            empty("No shots in selection.")
        else:
            col1, col2 = st.columns([3,2])
            with col1:
                fig = shotmap_figure(shot_w, duel_df=duel, title=f"Shotmap ({len(shot_w)} shots)")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                out_df = shot_w.groupby("shot_outcome",dropna=False).agg(
                    n=("shot_xg","size"), xG=("shot_xg","sum")).reset_index()
                fig2 = px.bar(out_df.sort_values("n",ascending=False),
                              x="shot_outcome", y="n", color="shot_outcome",
                              color_discrete_map=OUTCOME_COLORS, title="Outcomes")
                st.plotly_chart(figure_base(fig2,320), use_container_width=True)
                sh("Top xG Shots")
                dc = ["team","Taker","Shooter","shot_xg","shot_outcome","sp_type","Minute"]
                dc = [c for c in dc if c in shot_w.columns]
                st.dataframe(shot_w[dc].sort_values("shot_xg",ascending=False).head(20)
                             .reset_index(drop=True).round(3), use_container_width=True, height=260)

    with tabs[1]:
        sh("Delivery Map", "End locations of set-piece deliveries, coloured by delivery zone")
        fig = delivery_map(w, title="Set Piece Delivery Destinations")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        sh("Delivery Origins", "Where set pieces are taken from")
        fig = routines_map(w, title="Set Piece Origin Map")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        c1,c2 = st.columns(2)
        with c1:
            type_df = w.groupby("sp_type",dropna=False).agg(
                n=("match_id","size"), xG=("shot_xg","sum"), Shots=("led_shot","sum")).reset_index()
            type_df["xG/event"] = type_df.xG/type_df.n.replace(0,np.nan)
            fig = px.bar(type_df, x="sp_type", y="n", color="sp_type",
                         color_discrete_map=TYPE_COLORS, title="Volume by Type", text="n")
            st.plotly_chart(figure_base(fig,360), use_container_width=True)
        with c2:
            fig = px.bar(type_df, x="sp_type", y="xG/event", color="sp_type",
                         color_discrete_map=TYPE_COLORS, title="xG/Event by Type", text_auto=".3f")
            st.plotly_chart(figure_base(fig,360), use_container_width=True)

    with tabs[4]:
        ts = team_summary(w)
        if ts.empty: empty()
        else:
            st.dataframe(ts.reset_index(drop=True), use_container_width=True, height=360)
            c1,c2 = st.columns(2)
            with c1:
                fig = px.bar(ts.head(14), x="team", y="xG", color="ShotRate",
                             color_continuous_scale="Blues", title="Team xG")
                fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-40)
                st.plotly_chart(figure_base(fig,360), use_container_width=True)
            with c2:
                fig = px.scatter(ts, x="ShotRate", y="xG_per_event", size="Events",
                                 text="team", color_discrete_sequence=[ACCENT],
                                 title="Efficiency Map")
                fig.update_traces(textposition="top center")
                fig.update_xaxes(tickformat=".0%")
                st.plotly_chart(figure_base(fig,360), use_container_width=True)

    with tabs[5]:
        tk = taker_summary(w)
        st.dataframe(tk.reset_index(drop=True), use_container_width=True, height=400)

    with tabs[6]:
        st.dataframe(w.reset_index(drop=True), use_container_width=True, height=500)
        st.download_button("⬇ Download CSV", w.to_csv(index=False).encode(),
                           "set_pieces_filtered.csv","text/csv",use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# CORNER DELAYS
# ─────────────────────────────────────────────────────────────────────────────
def delays_page():
    c0, c_hero = st.columns([1,8])
    with c0:
        if st.button("← Home"): nav("home"); st.rerun()
    with c_hero:
        st.markdown("""<div class="hero" style="padding:22px 26px 18px 26px;">
          <div class="hero-title" style="font-size:2.1rem;">Corner <span>Delays</span></div>
          <div class="hero-sub">How long does each corner routine take before the ball goes out of play? Breakdown by type and period.</div>
        </div>""", unsafe_allow_html=True)

    df = dlay.copy()

    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Total Corners", f"{len(df):,}")
    with k2: kpi("Avg Delay", f"{df['delay_sec'].mean():.1f}s")
    with k3: kpi("Median Delay", f"{df['delay_sec'].median():.1f}s")
    with k4: kpi("Max Delay", f"{df['delay_sec'].max():.1f}s")

    tabs = st.tabs(["📊 Distribution","📋 By Type","🕐 By Period","📋 Data"])

    with tabs[0]:
        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="delay_sec", nbins=40, color_discrete_sequence=[ACCENT],
                               title="Corner Delay Distribution (seconds)")
            fig.add_vline(x=df["delay_sec"].mean(), line_dash="dash", line_color=WARNING,
                          annotation_text=f"Mean {df['delay_sec'].mean():.1f}s", annotation_position="top right")
            st.plotly_chart(figure_base(fig,400,"Delay Distribution"), use_container_width=True)
        with c2:
            fig = px.box(df, y="delay_sec", color_discrete_sequence=[ACCENT],
                         title="Delay Spread (box plot)")
            st.plotly_chart(figure_base(fig,400), use_container_width=True)

    with tabs[1]:
        type_df = df.groupby("out_event_type",dropna=False).agg(
            Count=("delay_sec","size"),
            Mean_delay=("delay_sec","mean"),
            Median=("delay_sec","median"),
            Max=("delay_sec","max")).reset_index().sort_values("Mean_delay",ascending=False)
        st.dataframe(type_df.reset_index(drop=True).round(2), use_container_width=True, height=300)
        fig = px.bar(type_df.head(10), x="out_event_type", y="Mean_delay",
                     color="Mean_delay", color_continuous_scale="Blues",
                     title="Mean Delay by Clearance Type")
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
        st.plotly_chart(figure_base(fig,380), use_container_width=True)

    with tabs[2]:
        period_df = df.groupby("period").agg(
            Count=("delay_sec","size"), Mean=("delay_sec","mean"),
            Median=("delay_sec","median")).reset_index()
        period_df["period"] = period_df["period"].map({1:"First Half",2:"Second Half"})
        st.dataframe(period_df.round(2), use_container_width=True, height=120)
        fig = px.histogram(df, x="delay_sec", color=df["period"].map({1:"H1",2:"H2"}),
                           nbins=35, barmode="overlay", opacity=0.75,
                           color_discrete_sequence=[ACCENT, ORANGE],
                           title="Delay Distribution by Half")
        fig.update_layout(legend_title_text="Half")
        st.plotly_chart(figure_base(fig,400), use_container_width=True)

    with tabs[3]:
        st.dataframe(df.reset_index(drop=True), use_container_width=True, height=500)
        st.download_button("⬇ Download CSV", df.to_csv(index=False).encode(),
                           "corner_delays.csv","text/csv",use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# DUEL RATINGS
# ─────────────────────────────────────────────────────────────────────────────
def duels_page():
    c0, c_hero = st.columns([1,8])
    with c0:
        if st.button("← Home"): nav("home"); st.rerun()
    with c_hero:
        st.markdown("""<div class="hero" style="padding:22px 26px 18px 26px;">
          <div class="hero-title" style="font-size:2.1rem;">Duel <span>Ratings</span></div>
          <div class="hero-sub">Player-level duel hops ratings for aerial duels in set-piece situations across Allsvenskan.</div>
        </div>""", unsafe_allow_html=True)

    df = duel.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Rating" not in df.columns:
        rating_col = [c for c in df.columns if "rating" in c.lower()]
        if rating_col: df = df.rename(columns={rating_col[0]:"Rating"})

    teams = sorted(df["Team"].dropna().unique().tolist()) if "Team" in df.columns else []

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    with f1: team_f = st.selectbox("Team", ["All"]+teams, key="d_team")
    with f2: min_r  = st.slider("Min Rating", 0.0, 1.0, 0.0, 0.01, key="d_minr")
    with f3: top_n  = st.slider("Show top N", 10, len(df), min(50,len(df)), 5, key="d_topn")
    st.markdown('</div>', unsafe_allow_html=True)

    w = df.copy()
    if team_f != "All" and "Team" in w.columns: w = w[w["Team"]==team_f]
    if "Rating" in w.columns: w = w[w["Rating"] >= min_r]

    if w.empty:
        empty(); return

    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi("Players", f"{len(w):,}")
    with k2: kpi("Avg Rating", hval(w["Rating"].mean(),3))
    with k3: kpi("Top Rating", hval(w["Rating"].max(),3))
    with k4: kpi("Teams", f"{w['Team'].nunique():,}" if "Team" in w.columns else "—")

    tabs = st.tabs(["🏆 Rankings","📊 Distribution","🆚 Team Comparison","📋 Data"])

    plot = w.sort_values("Rating",ascending=False).head(top_n).copy()

    with tabs[0]:
        sh("Top Rated Players", f"Top {top_n} by duel hops rating")
        fig = px.bar(plot, x="Player" if "Player" in plot.columns else plot.columns[0],
                     y="Rating", color="Rating",
                     color_continuous_scale="Blues",
                     hover_data=["Team"] if "Team" in plot.columns else None,
                     title=f"Top {top_n} Duel Ratings")
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(figure_base(fig, 500), use_container_width=True)

    with tabs[1]:
        c1,c2 = st.columns(2)
        with c1:
            fig = px.histogram(w, x="Rating", nbins=30, color_discrete_sequence=[PURPLE],
                               title="Rating Distribution")
            fig.add_vline(x=w["Rating"].mean(), line_dash="dash", line_color=WARNING,
                          annotation_text=f"Mean {w['Rating'].mean():.3f}")
            st.plotly_chart(figure_base(fig,380), use_container_width=True)
        with c2:
            if "Team" in w.columns:
                fig = px.box(w, x="Team", y="Rating", color="Team",
                             color_discrete_sequence=QUAL_PALETTE, title="Rating by Team")
                fig.update_layout(showlegend=False, xaxis_tickangle=-40)
                st.plotly_chart(figure_base(fig,380), use_container_width=True)

    with tabs[2]:
        if "Team" in w.columns:
            t_df = w.groupby("Team").agg(
                Players=("Player","size") if "Player" in w.columns else ("Team","size"),
                AvgRating=("Rating","mean"),
                MaxRating=("Rating","max"),
                MinRating=("Rating","min")).reset_index().sort_values("AvgRating",ascending=False)
            st.dataframe(t_df.reset_index(drop=True).round(4), use_container_width=True, height=350)
            fig = px.bar(t_df, x="Team", y="AvgRating", color="AvgRating",
                         color_continuous_scale="Blues", title="Average Duel Rating by Team")
            fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-40)
            st.plotly_chart(figure_base(fig,380), use_container_width=True)

    with tabs[3]:
        st.dataframe(w.sort_values("Rating",ascending=False).reset_index(drop=True),
                     use_container_width=True, height=500)
        st.download_button("⬇ Download CSV", w.to_csv(index=False).encode(),
                           "duel_ratings.csv","text/csv",use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state["page"]
if page == "home":
    home()
elif page == "corner":
    corner_studio()
elif page == "setpieces":
    setpieces_page()
elif page == "delays":
    delays_page()
elif page == "duels":
    duels_page()

st.markdown(
    '<div class="footer-note">⚽ Allsvenskan Set Piece Studio · Corners · Free Kicks · Throw-Ins · Delays · Duel Ratings</div>',
    unsafe_allow_html=True)
