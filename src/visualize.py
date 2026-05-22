"""
visualize.py  —  Charts & Visualizations
------------------------------------------
Generates 5 charts from your personal user_logs.csv and saves them
as PNG images inside a charts/ folder.

Charts produced:
  1. mood_trend.png        — daily mood line chart with risk zones
  2. sleep_vs_mood.png     — scatter plot: sleep hours vs mood score
  3. stress_vs_mood.png    — scatter plot: stress level vs mood score
  4. weekly_heatmap.png    — heatmap of all metrics across the week
  5. risk_distribution.png — pie/bar chart of Low/Medium/High days

Run:
    python src\visualize.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works on all OS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE    = os.path.join(BASE_DIR, "data", "user_logs.csv")
CHARTS_DIR  = os.path.join(BASE_DIR, "charts")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH   = os.path.join(BASE_DIR, "models", "model_meta.pkl")

RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"
GREEN="\033[92m"; YELLOW="\033[93m"; DIM="\033[2m"

# ── Colour palette (consistent across all charts) ─────────────────────────────
PALETTE = {
    "low":    "#4CAF50",   # green
    "medium": "#FF9800",   # orange
    "high":   "#F44336",   # red
    "mood":   "#5C6BC0",   # indigo
    "sleep":  "#26C6DA",   # cyan
    "stress": "#EF5350",   # red
    "energy": "#FFA726",   # amber
    "bg":     "#FAFAFA",
    "grid":   "#EEEEEE",
}

RISK_COLORS = {0: PALETTE["low"], 1: PALETTE["medium"], 2: PALETTE["high"]}
RISK_NAMES  = {0: "Low", 1: "Medium", 2: "High"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_charts_dir():
    os.makedirs(CHARTS_DIR, exist_ok=True)


def _style():
    """Apply a clean, consistent style to every chart."""
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["bg"],
        "axes.grid":         True,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.8,
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
    })


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(LOG_FILE, parse_dates=["date"])
    numeric = ["mood_score","sleep_hours","exercise_minutes",
               "stress_level","energy_level"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["social_encoded"]  = df["social_interaction"].str.lower().map(
        {"low":0,"medium":1,"high":2}).fillna(1)
    df["appetite_encoded"] = df["appetite_change"].str.lower().map(
        {"reduced":0,"normal":1,"increased":2}).fillna(1)
    return df.sort_values("date").reset_index(drop=True)


def _get_predictions(df):
    """Returns prediction array if model exists, else None."""
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        feats = ["mood_score","sleep_hours","exercise_minutes",
                 "stress_level","energy_level","social_encoded","appetite_encoded"]
        X = df[feats].fillna(df[feats].median()).astype(float)
        return model.predict(X)
    except Exception:
        return None


def _save(name):
    path = os.path.join(CHARTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {GREEN}✓{RESET} Saved  →  charts\\{name}")
    return path


# ── Chart 1: Mood Trend ───────────────────────────────────────────────────────

def chart_mood_trend(df, predictions=None) -> str:
    fig, ax = plt.subplots(figsize=(12, 5))

    dates = df["date"]
    mood  = df["mood_score"]

    # Risk zone shading
    ax.axhspan(0,  4,  alpha=0.08, color=PALETTE["high"],   zorder=0, label="High risk zone")
    ax.axhspan(4,  6,  alpha=0.08, color=PALETTE["medium"], zorder=0, label="Medium risk zone")
    ax.axhspan(6,  10, alpha=0.08, color=PALETTE["low"],    zorder=0, label="Low risk zone")

    # 7-day rolling average
    if len(df) >= 3:
        rolling = mood.rolling(3, min_periods=1).mean()
        ax.plot(dates, rolling, color="#B0BEC5", linewidth=1.5,
                linestyle="--", label="3-day avg", zorder=2)

    # Main mood line
    ax.plot(dates, mood, color=PALETTE["mood"], linewidth=2.5,
            marker="o", markersize=7, zorder=3, label="Daily mood")

    # Colour each dot by risk if predictions available
    if predictions is not None:
        for i, (d, m, r) in enumerate(zip(dates, mood, predictions)):
            ax.scatter(d, m, color=RISK_COLORS[r], s=80, zorder=4)

    # Annotate best & worst
    best_i  = mood.idxmax()
    worst_i = mood.idxmin()
    ax.annotate(f"Best: {int(mood[best_i])}",
                xy=(dates[best_i], mood[best_i]),
                xytext=(0, 14), textcoords="offset points",
                ha="center", fontsize=8, color=PALETTE["low"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["low"], lw=1))
    ax.annotate(f"Worst: {int(mood[worst_i])}",
                xy=(dates[worst_i], mood[worst_i]),
                xytext=(0, -20), textcoords="offset points",
                ha="center", fontsize=8, color=PALETTE["high"],
                arrowprops=dict(arrowstyle="-", color=PALETTE["high"], lw=1))

    ax.set_title("Daily Mood Score — Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mood Score (1–10)")
    ax.set_ylim(0, 11)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    plt.xticks(rotation=35)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    return _save("mood_trend.png")


# ── Chart 2: Sleep vs Mood Scatter ────────────────────────────────────────────

def chart_sleep_vs_mood(df, predictions=None) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ([RISK_COLORS[r] for r in predictions]
              if predictions is not None
              else [PALETTE["mood"]] * len(df))

    ax.scatter(df["sleep_hours"], df["mood_score"],
               c=colors, s=100, alpha=0.85, edgecolors="white", linewidth=0.8)

    # Trend line
    if len(df) >= 3:
        z = np.polyfit(df["sleep_hours"].fillna(0), df["mood_score"].fillna(0), 1)
        p = np.poly1d(z)
        xs = np.linspace(df["sleep_hours"].min(), df["sleep_hours"].max(), 100)
        ax.plot(xs, p(xs), color="#9E9E9E", linewidth=1.5, linestyle="--", label="Trend")

    # Correlation annotation
    corr = df["sleep_hours"].corr(df["mood_score"])
    ax.text(0.05, 0.95, f"Correlation r = {corr:+.2f}",
            transform=ax.transAxes, fontsize=10,
            color=PALETTE["low"] if corr > 0.5 else "#607D8B",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Legend patches
    patches = [mpatches.Patch(color=RISK_COLORS[i], label=RISK_NAMES[i])
               for i in range(3)]
    ax.legend(handles=patches, title="Risk Level", fontsize=8, loc="lower right")

    ax.set_title("Sleep Hours vs Mood Score")
    ax.set_xlabel("Sleep Hours")
    ax.set_ylabel("Mood Score (1–10)")
    ax.set_ylim(0, 11)

    return _save("sleep_vs_mood.png")


# ── Chart 3: Stress vs Mood Scatter ──────────────────────────────────────────

def chart_stress_vs_mood(df, predictions=None) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ([RISK_COLORS[r] for r in predictions]
              if predictions is not None
              else [PALETTE["stress"]] * len(df))

    ax.scatter(df["stress_level"], df["mood_score"],
               c=colors, s=100, alpha=0.85, edgecolors="white", linewidth=0.8)

    if len(df) >= 3:
        z = np.polyfit(df["stress_level"].fillna(0), df["mood_score"].fillna(0), 1)
        p = np.poly1d(z)
        xs = np.linspace(df["stress_level"].min(), df["stress_level"].max(), 100)
        ax.plot(xs, p(xs), color="#9E9E9E", linewidth=1.5, linestyle="--")

    corr = df["stress_level"].corr(df["mood_score"])
    ax.text(0.05, 0.95, f"Correlation r = {corr:+.2f}",
            transform=ax.transAxes, fontsize=10,
            color=PALETTE["high"] if corr < -0.5 else "#607D8B",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    patches = [mpatches.Patch(color=RISK_COLORS[i], label=RISK_NAMES[i])
               for i in range(3)]
    ax.legend(handles=patches, title="Risk Level", fontsize=8, loc="upper right")

    ax.set_title("Stress Level vs Mood Score")
    ax.set_xlabel("Stress Level (1–10)")
    ax.set_ylabel("Mood Score (1–10)")
    ax.set_ylim(0, 11)

    return _save("stress_vs_mood.png")


# ── Chart 4: Weekly Heatmap ───────────────────────────────────────────────────

def chart_weekly_heatmap(df) -> str:
    metrics = ["mood_score", "sleep_hours", "stress_level",
               "energy_level", "exercise_minutes"]
    labels  = ["Mood", "Sleep\n(hrs)", "Stress", "Energy", "Exercise\n(min)"]

    # Use last 14 days max for readability
    sub = df.tail(14).copy()
    sub["day_label"] = sub["date"].dt.strftime("%b %d")

    heat_data = sub[metrics].T.values.astype(float)

    fig, ax = plt.subplots(figsize=(max(8, len(sub) * 0.7), 5))

    im = ax.imshow(heat_data, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=10)

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["day_label"].tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(labels, fontsize=9)

    # Value annotations
    for i in range(len(metrics)):
        for j in range(len(sub)):
            val = heat_data[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 3 or val > 8 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Score", shrink=0.8)
    ax.set_title("Weekly Wellness Heatmap  (green = good, red = poor)")

    return _save("weekly_heatmap.png")


# ── Chart 5: Risk Distribution ────────────────────────────────────────────────

def chart_risk_distribution(df, predictions) -> str:
    if predictions is None:
        print(f"  {YELLOW}⚠  No model found — skipping risk distribution chart{RESET}")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    counts = {RISK_NAMES[i]: int((predictions == i).sum()) for i in range(3)}
    colors = [PALETTE["low"], PALETTE["medium"], PALETTE["high"]]

    # Left: pie chart
    wedge_props = dict(linewidth=2, edgecolor="white")
    wedges, texts, autotexts = ax1.pie(
        counts.values(), labels=counts.keys(),
        colors=colors, autopct="%1.0f%%",
        wedgeprops=wedge_props, startangle=90,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(12)
    ax1.set_title("Overall Risk Distribution")

    # Right: bar chart over time
    risk_series = pd.Series(predictions)
    dates_labels = df["date"].dt.strftime("%b %d").tolist()
    bar_colors   = [RISK_COLORS[r] for r in predictions]

    ax2.bar(range(len(predictions)), [1] * len(predictions),
            color=bar_colors, width=0.8, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(range(len(predictions)))
    ax2.set_xticklabels(dates_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks([])
    ax2.set_title("Risk Level Per Day")
    ax2.set_ylabel("")
    ax2.grid(False)

    patches = [mpatches.Patch(color=RISK_COLORS[i], label=f"{RISK_NAMES[i]}: {counts[RISK_NAMES[i]]} days")
               for i in range(3)]
    ax2.legend(handles=patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    return _save("risk_distribution.png")


# ── Master run ────────────────────────────────────────────────────────────────

def run() -> list:
    _ensure_charts_dir()
    _style()

    print(f"\n{CYAN}{BOLD}{'─'*50}{RESET}")
    print(f"{CYAN}{BOLD}   📊  Generating Charts{RESET}")
    print(f"{CYAN}{'─'*50}{RESET}\n")

    df          = _load_data()
    predictions = _get_predictions(df)

    paths = []
    paths.append(chart_mood_trend(df, predictions))
    paths.append(chart_sleep_vs_mood(df, predictions))
    paths.append(chart_stress_vs_mood(df, predictions))
    paths.append(chart_weekly_heatmap(df))

    if predictions is not None:
        p = chart_risk_distribution(df, predictions)
        if p:
            paths.append(p)

    print(f"\n{GREEN}{BOLD}  ✅  All charts saved to charts\\ folder!{RESET}")
    print(f"  {DIM}Run python src\\report.py to generate your PDF summary.{RESET}\n")

    return [p for p in paths if p]


if __name__ == "__main__":
    run()
