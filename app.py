"""
app.py  —  Streamlit Web Dashboard
-------------------------------------
Full interactive dashboard for the Mental Health Mood Tracker.
Combines data logging, predictions, charts, and weekly report
into a single browser-based app.

Run:
    streamlit run app.py
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
from datetime import date, datetime

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

LOG_FILE   = os.path.join(BASE_DIR, "data", "user_logs.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.pkl")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")

RISK_NAMES  = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
RISK_EMOJI  = {0: "🟢", 1: "🟡", 2: "🔴"}
RISK_BG     = {0: "#E8F5E9", 1: "#FFF3E0", 2: "#FFEBEE"}

PALETTE = {
    "mood":    "#5C6BC0",
    "sleep":   "#26C6DA",
    "stress":  "#EF5350",
    "energy":  "#FFA726",
    "exercise":"#66BB6A",
    "grid":    "#EEEEEE",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mood Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* General */
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #F8F9FA;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 12px 16px;
  }

  /* Risk banner */
  .risk-banner {
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 1rem;
    text-align: center;
  }
  .risk-banner h2 { margin: 0; font-size: 1.6rem; }
  .risk-banner p  { margin: 4px 0 0; font-size: 0.95rem; opacity: 0.85; }

  /* Section headers */
  .section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #3F51B5;
    border-bottom: 2px solid #E8EAF6;
    padding-bottom: 4px;
    margin-bottom: 12px;
  }

  /* Log table */
  .log-row-low    { background: #E8F5E9 !important; }
  .log-row-medium { background: #FFF3E0 !important; }
  .log-row-high   { background: #FFEBEE !important; }

  /* Advice card */
  .advice-card {
    background: #F3F4FF;
    border-left: 4px solid #3F51B5;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.92rem;
  }

  /* Footer */
  .footer {
    text-align: center;
    color: #9E9E9E;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #EEEEEE;
  }
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_data():
    if not os.path.isfile(LOG_FILE):
        return pd.DataFrame()
    df = pd.read_csv(LOG_FILE, parse_dates=["date"])
    numeric = ["mood_score","sleep_hours","exercise_minutes","stress_level","energy_level"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["social_encoded"]   = df["social_interaction"].str.lower().map(
        {"low":0,"medium":1,"high":2}).fillna(1)
    df["appetite_encoded"] = df["appetite_change"].str.lower().map(
        {"reduced":0,"normal":1,"increased":2}).fillna(1)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_resource
def load_model():
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        meta  = joblib.load(META_PATH)
        return model, meta["feature_names"]
    except Exception:
        return None, None


def predict(df, model, feature_cols):
    if model is None or df.empty:
        return None, None
    X = df[feature_cols].fillna(df[feature_cols].median()).astype(float)
    return model.predict(X), model.predict_proba(X)


def save_entry(entry: dict):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    cols = ["date","mood_score","sleep_hours","exercise_minutes",
            "stress_level","social_interaction","appetite_change","energy_level","notes"]
    write_header = not os.path.isfile(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
    import csv
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()
        writer.writerow({c: entry.get(c, "") for c in cols})


def entry_exists_today(df):
    if df.empty:
        return False
    return str(date.today()) in df["date"].astype(str).values


# ── Chart helpers ─────────────────────────────────────────────────────────────

def fig_mood_trend(df, predictions=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    ax.axhspan(0, 4,  alpha=0.08, color=RISK_COLORS[2], zorder=0)
    ax.axhspan(4, 6,  alpha=0.08, color=RISK_COLORS[1], zorder=0)
    ax.axhspan(6, 10, alpha=0.08, color=RISK_COLORS[0], zorder=0)

    mood  = df["mood_score"]
    dates = df["date"]

    if len(df) >= 3:
        ax.plot(dates, mood.rolling(3, min_periods=1).mean(),
                color="#B0BEC5", linewidth=1.5, linestyle="--", label="3-day avg")

    ax.plot(dates, mood, color=PALETTE["mood"], linewidth=2.5,
            marker="o", markersize=7, label="Mood", zorder=3)

    if predictions is not None:
        for i, (d, m, r) in enumerate(zip(dates, mood, predictions)):
            ax.scatter(d, m, color=RISK_COLORS[int(r)], s=90, zorder=4)

    ax.set_ylim(0, 11)
    ax.set_ylabel("Mood Score (1–10)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=30)
    ax.grid(color="#EEEEEE", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Daily Mood Trend", fontweight="bold", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    return fig


def fig_scatter(df, x_col, y_col, x_label, predictions=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    colors = ([RISK_COLORS[int(r)] for r in predictions]
              if predictions is not None
              else [PALETTE["mood"]] * len(df))

    ax.scatter(df[x_col], df[y_col], c=colors, s=90,
               alpha=0.85, edgecolors="white", linewidth=0.8)

    if len(df) >= 3:
        z = np.polyfit(df[x_col].fillna(0), df[y_col].fillna(0), 1)
        xs = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color="#9E9E9E", linewidth=1.5, linestyle="--")

    corr = df[x_col].corr(df[y_col])
    ax.text(0.05, 0.95, f"r = {corr:+.2f}", transform=ax.transAxes,
            fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    patches = [mpatches.Patch(color=RISK_COLORS[i], label=RISK_NAMES[i]) for i in range(3)]
    ax.legend(handles=patches, fontsize=7, loc="lower right")
    ax.set_xlabel(x_label); ax.set_ylabel("Mood Score")
    ax.set_ylim(0, 11)
    ax.grid(color="#EEEEEE"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title(f"{x_label} vs Mood", fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_heatmap(df):
    metrics = ["mood_score","sleep_hours","stress_level","energy_level","exercise_minutes"]
    labels  = ["Mood","Sleep","Stress","Energy","Exercise"]
    sub = df.tail(14).copy()
    sub["day"] = sub["date"].dt.strftime("%b %d")
    heat = sub[metrics].T.values.astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(sub)*0.65), 3.5))
    fig.patch.set_facecolor("#FAFAFA")
    im = ax.imshow(heat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=10)
    ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub["day"].tolist(), rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(metrics)):
        for j in range(len(sub)):
            v = heat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 3 or v > 8 else "black", fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Score")
    ax.set_title("Wellness Heatmap", fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_risk_bar(df, predictions):
    counts = {RISK_NAMES[i]: int((predictions == i).sum()) for i in range(3)}
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    bars = ax.bar(counts.keys(), counts.values(),
                  color=[RISK_COLORS[i] for i in range(3)],
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Days"); ax.set_ylim(0, max(counts.values()) + 2)
    ax.set_title("Risk Distribution", fontweight="bold", fontsize=11)
    ax.grid(axis="y", color="#EEEEEE"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 Mood Tracker")
    st.caption("Your personal mental wellness AI")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📋 Today's Check-in", "📊 Dashboard", "📅 History", "📄 Report"],
        label_visibility="collapsed",
    )

    st.divider()

    df_side = load_data()
    if not df_side.empty:
        st.caption(f"**Days logged:** {len(df_side)}")
        st.caption(f"**Since:** {df_side['date'].min().strftime('%b %d, %Y')}")
        model_side, _ = load_model()
        if model_side:
            st.caption("**Model:** ✅ Loaded")
        else:
            st.caption("**Model:** ⚠️ Run train.py first")

    st.divider()
    st.caption("⚠️ Educational project only.\nNot a clinical tool.")
    st.caption("iCall: 9152987821")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: TODAY'S CHECK-IN
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📋 Today's Check-in":
    st.title("📋 Today's Check-in")
    st.caption(f"Date: {date.today().strftime('%A, %B %d, %Y')}")

    df = load_data()

    if entry_exists_today(df):
        st.success("✅ You've already logged today! Come back tomorrow.")
        last = df[df["date"].astype(str) == str(date.today())].iloc[-1]
        st.markdown("**Today's entry:**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mood", f"{last['mood_score']:.0f}/10")
        c2.metric("Sleep", f"{last['sleep_hours']:.1f}h")
        c3.metric("Stress", f"{last['stress_level']:.0f}/10")
        c4.metric("Energy", f"{last['energy_level']:.0f}/10")
    else:
        st.info("Answer all questions honestly — only you can see this data.")
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">Mood & Energy</div>', unsafe_allow_html=True)
            mood   = st.slider("😊 Mood Score", 1, 10, 6,
                               help="1 = very low / sad   |   10 = excellent / happy")
            energy = st.slider("⚡ Energy Level", 1, 10, 6,
                               help="1 = completely drained   |   10 = full of energy")
            st.markdown('<div class="section-header">Sleep</div>', unsafe_allow_html=True)
            sleep  = st.number_input("😴 Hours slept last night", 0.0, 24.0, 7.0, 0.5,
                                     help="You can use decimals — e.g. 7.5")

        with col2:
            st.markdown('<div class="section-header">Stress & Activity</div>', unsafe_allow_html=True)
            stress   = st.slider("😰 Stress Level", 1, 10, 5,
                                 help="1 = completely calm   |   10 = extremely stressed")
            exercise = st.number_input("🏃 Exercise today (minutes)", 0, 300, 0, 5,
                                      help="Any movement counts. Enter 0 if none.")
            st.markdown('<div class="section-header">Lifestyle</div>', unsafe_allow_html=True)
            social   = st.selectbox("👥 Social Interaction", ["low","medium","high"],
                                    index=1, help="How much did you interact with others?")
            appetite = st.selectbox("🍽️ Appetite Change", ["normal","reduced","increased"],
                                    help="Compared to your usual appetite")

        notes = st.text_area("📝 Notes (optional)",
                             placeholder="e.g. felt anxious in the evening, had a tough meeting...",
                             max_chars=300)

        st.divider()

        if st.button("💾 Save Today's Entry", type="primary", use_container_width=True):
            entry = {
                "date": str(date.today()),
                "mood_score": mood,
                "sleep_hours": sleep,
                "exercise_minutes": exercise,
                "stress_level": stress,
                "social_interaction": social,
                "appetite_change": appetite,
                "energy_level": energy,
                "notes": notes,
            }
            save_entry(entry)
            st.cache_data.clear()
            st.success("✅ Entry saved! Head to the Dashboard to see your prediction.")
            st.balloons()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.title("📊 Dashboard")

    df = load_data()
    if df.empty:
        st.warning("No data yet. Log your first entry in Today's Check-in!")
        st.stop()

    model, feature_cols = load_model()
    preds, probs = predict(df, model, feature_cols)

    # ── Today's risk banner ───────────────────────────────────────────────────
    if preds is not None:
        today_risk = int(preds[-1])
        today_conf = float(probs[-1][today_risk]) * 100
        color      = RISK_COLORS[today_risk]
        bg         = RISK_BG[today_risk]
        emoji      = RISK_EMOJI[today_risk]

        st.markdown(f"""
        <div class="risk-banner" style="background:{bg}; border: 2px solid {color}">
            <h2 style="color:{color}">{emoji} Today's Risk: {RISK_NAMES[today_risk].upper()}</h2>
            <p style="color:{color}">Model confidence: {today_conf:.0f}%
            &nbsp;|&nbsp; Based on {len(df)} logged day(s)</p>
        </div>
        """, unsafe_allow_html=True)

        # Advice
        advice = {
            0: ["Keep up your current sleep and exercise routine!",
                "Your social connections are clearly helping — nurture them."],
            1: ["Try to get at least 7 hours of sleep tonight.",
                "Even a 15-minute walk can lift your mood noticeably.",
                "Reach out to a friend or family member today."],
            2: ["Please consider talking to someone you trust today.",
                "iCall (India): 9152987821  |  Vandrevala: 1860-2662-345",
                "Small steps matter — even a glass of water or a short walk."],
        }
        for tip in advice[today_risk]:
            st.markdown(f'<div class="advice-card">→ {tip}</div>', unsafe_allow_html=True)

    st.divider()

    # ── 7-day averages ────────────────────────────────────────────────────────
    week = df.tail(7)
    st.markdown('<div class="section-header">7-Day Averages</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("😊 Mood",     f"{week['mood_score'].mean():.1f}/10",
              delta=f"{week['mood_score'].diff().mean():+.1f} trend")
    c2.metric("😴 Sleep",    f"{week['sleep_hours'].mean():.1f}h",
              delta="✓ Good" if week['sleep_hours'].mean() >= 7 else "⚠ Low")
    c3.metric("😰 Stress",   f"{week['stress_level'].mean():.1f}/10")
    c4.metric("⚡ Energy",   f"{week['energy_level'].mean():.1f}/10")
    c5.metric("🏃 Exercise", f"{week['exercise_minutes'].mean():.0f} min/day")

    st.divider()

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Mood Trend</div>', unsafe_allow_html=True)
    st.pyplot(fig_mood_trend(df, preds), use_container_width=True)

    st.divider()

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">Sleep vs Mood</div>', unsafe_allow_html=True)
        st.pyplot(fig_scatter(df, "sleep_hours", "mood_score", "Sleep Hours", preds))
    with col_r:
        st.markdown('<div class="section-header">Stress vs Mood</div>', unsafe_allow_html=True)
        st.pyplot(fig_scatter(df, "stress_level", "mood_score", "Stress Level", preds))

    st.divider()

    # ── Charts row 3 ──────────────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown('<div class="section-header">Weekly Heatmap</div>', unsafe_allow_html=True)
        st.pyplot(fig_heatmap(df), use_container_width=True)
    with col_r:
        if preds is not None:
            st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
            st.pyplot(fig_risk_bar(df, preds))


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📅 History":
    st.title("📅 Full History")

    df = load_data()
    if df.empty:
        st.warning("No entries logged yet.")
        st.stop()

    model, feature_cols = load_model()
    preds, probs = predict(df, model, feature_cols)

    # Filters
    col1, col2 = st.columns([2, 1])
    with col1:
        days = st.slider("Show last N days", 7, min(len(df), 90), min(len(df), 30))
    with col2:
        risk_filter = st.multiselect("Filter by risk", ["Low","Medium","High"],
                                     default=["Low","Medium","High"])

    df_view = df.tail(days).reset_index(drop=True)
    pred_view = preds[-days:] if preds is not None else None

    # Build display table
    rows = []
    for i, (_, row) in enumerate(df_view.iterrows()):
        r = RISK_NAMES[int(pred_view[i])] if pred_view is not None else "—"
        conf = f"{float(probs[-(days-i)][int(pred_view[i])]) * 100:.0f}%" if pred_view is not None else "—"
        rows.append({
            "Date":     row["date"].strftime("%Y-%m-%d"),
            "Mood":     f"{row['mood_score']:.0f}/10",
            "Sleep":    f"{row['sleep_hours']:.1f}h",
            "Exercise": f"{row['exercise_minutes']:.0f}m",
            "Stress":   f"{row['stress_level']:.0f}/10",
            "Energy":   f"{row['energy_level']:.0f}/10",
            "Social":   str(row.get("social_interaction","")).title(),
            "Risk":     r,
            "Confidence": conf,
            "Notes":    str(row.get("notes",""))[:40],
        })

    df_table = pd.DataFrame(rows)
    if risk_filter:
        df_table = df_table[df_table["Risk"].isin(risk_filter)]

    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Risk": st.column_config.TextColumn("Risk", width="small"),
            "Notes": st.column_config.TextColumn("Notes", width="medium"),
        },
    )

    # Summary stats
    st.divider()
    st.markdown('<div class="section-header">Period Summary</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if pred_view is not None:
        low_pct  = (pred_view == 0).mean() * 100
        med_pct  = (pred_view == 1).mean() * 100
        high_pct = (pred_view == 2).mean() * 100
        c1.metric("🟢 Low Risk Days",    f"{low_pct:.0f}%")
        c2.metric("🟡 Medium Risk Days", f"{med_pct:.0f}%")
        c3.metric("🔴 High Risk Days",   f"{high_pct:.0f}%")

    # Correlation insights
    st.divider()
    st.markdown('<div class="section-header">Correlation Insights</div>', unsafe_allow_html=True)
    numeric_df = df_view.select_dtypes(include="number")
    if "mood_score" in numeric_df.columns:
        corr = numeric_df.corr()["mood_score"].drop("mood_score").sort_values(ascending=False)
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature","Correlation with Mood"]
        corr_df = corr_df[corr_df["Feature"].isin(
            ["sleep_hours","exercise_minutes","stress_level","energy_level"])]
        st.dataframe(corr_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: REPORT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📄 Report":
    st.title("📄 Weekly Report")

    df = load_data()
    if df.empty:
        st.warning("No data yet. Log your first entry in Today's Check-in!")
        st.stop()

    model, feature_cols = load_model()
    preds, probs = predict(df, model, feature_cols)

    week = df.tail(7)
    week_pred = preds[-7:] if preds is not None else None

    # ── Header card ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #3F51B5, #5C6BC0);
                border-radius: 12px; padding: 24px; color: white; margin-bottom: 1rem;">
        <h2 style="margin:0">🧠 Mood Tracker — Weekly Report</h2>
        <p style="margin:4px 0 0; opacity:0.85">
            {week['date'].iloc[0].strftime('%B %d')} – {week['date'].iloc[-1].strftime('%B %d, %Y')}
            &nbsp;|&nbsp; Generated: {date.today().strftime('%B %d, %Y')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">7-Day Averages</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mood",     f"{week['mood_score'].mean():.1f}/10")
    c2.metric("Sleep",    f"{week['sleep_hours'].mean():.1f}h")
    c3.metric("Stress",   f"{week['stress_level'].mean():.1f}/10")
    c4.metric("Energy",   f"{week['energy_level'].mean():.1f}/10")
    c5.metric("Exercise", f"{week['exercise_minutes'].mean():.0f} min")

    # ── Risk breakdown ────────────────────────────────────────────────────────
    if week_pred is not None:
        st.divider()
        st.markdown('<div class="section-header">Risk Breakdown</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Low",    f"{(week_pred==0).sum()} days")
        c2.metric("🟡 Medium", f"{(week_pred==1).sum()} days")
        c3.metric("🔴 High",   f"{(week_pred==2).sum()} days")

    # ── Insights ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">Personalised Insights</div>', unsafe_allow_html=True)

    avg_mood   = week["mood_score"].mean()
    avg_sleep  = week["sleep_hours"].mean()
    avg_stress = week["stress_level"].mean()
    avg_ex     = week["exercise_minutes"].mean()

    insights = []
    if avg_sleep < 7:
        insights.append(f"**Sleep:** Averaging {avg_sleep:.1f}h — below the 7h target. Try going to bed 30 min earlier.")
    else:
        insights.append(f"**Sleep:** Averaging {avg_sleep:.1f}h — great job meeting the 7+ hour target!")
    if avg_stress > 6:
        insights.append(f"**Stress:** Average {avg_stress:.1f}/10 is high. Try 5 min of deep breathing daily.")
    else:
        insights.append(f"**Stress:** Average {avg_stress:.1f}/10 is manageable. Keep your stress habits up.")
    if avg_ex < 20:
        insights.append("**Exercise:** Less than 20 min/day. Even a 15-min walk has measurable mood benefits.")
    else:
        insights.append(f"**Exercise:** {avg_ex:.0f} min/day average — physical activity is boosting your mood!")
    if avg_mood >= 7:
        insights.append(f"**Mood:** Strong {avg_mood:.1f}/10 average this week. You're doing well!")
    elif avg_mood >= 5:
        insights.append(f"**Mood:** Moderate {avg_mood:.1f}/10. Focus on sleep and stress reduction.")
    else:
        insights.append(f"**Mood:** Low {avg_mood:.1f}/10 this week. Consider talking to someone you trust.")

    for tip in insights:
        st.markdown(f'<div class="advice-card">{tip}</div>', unsafe_allow_html=True)

    # ── Generate PDF button ───────────────────────────────────────────────────
    st.divider()
    if st.button("📥 Generate & Download PDF Report", type="primary", use_container_width=True):
        with st.spinner("Building your PDF..."):
            try:
                import visualize, report
                visualize.run()
                out_path = report.run()
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=f.read(),
                        file_name=os.path.basename(out_path),
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")

    st.divider()
    st.caption("⚠️ This report is for educational purposes only and is not a clinical diagnosis. "
               "iCall (India): 9152987821  |  Vandrevala Foundation: 1860-2662-345")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Mental Health Mood Tracker — Python AI/ML Project &nbsp;|&nbsp; '
    'Built with Streamlit, scikit-learn, matplotlib</div>',
    unsafe_allow_html=True,
)
