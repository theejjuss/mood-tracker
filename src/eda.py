"""
eda.py  —  Exploratory Data Analysis
--------------------------------------
Loads your personal user_logs.csv and prints a full statistical
summary so you understand your data before touching the ML model.

Run directly:
    python src/eda.py
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Path helpers ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "data", "user_logs.csv")

# ── Colours ───────────────────────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD  = "\033[1m";  CYAN  = "\033[96m"
GREEN  = "\033[92m"; YELLOW= "\033[93m"; RED   = "\033[91m"
DIM    = "\033[2m";  BLUE  = "\033[94m"


def _section(title):
    print(f"\n{CYAN}{BOLD}{'─'*55}{RESET}")
    print(f"{CYAN}{BOLD}   {title}{RESET}")
    print(f"{CYAN}{'─'*55}{RESET}")


def load_data() -> pd.DataFrame:
    """Loads the CSV, parses dates, and enforces correct dtypes."""
    if not os.path.isfile(LOG_FILE):
        print(f"{RED}  ✗ No log file found at: {LOG_FILE}{RESET}")
        print(f"  Run 'python main.py' first to log some entries.\n")
        sys.exit(1)

    df = pd.read_csv(LOG_FILE, parse_dates=["date"])

    # Enforce numeric types for columns that should be numbers
    numeric_cols = [
        "mood_score", "sleep_hours", "exercise_minutes",
        "stress_level", "energy_level",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


def basic_info(df: pd.DataFrame):
    """Prints shape, date range, and column types."""
    _section("📁  Dataset Overview")
    print(f"  Rows (days logged) : {BOLD}{len(df)}{RESET}")
    print(f"  Columns            : {len(df.columns)}")
    print(f"  Date range         : {df['date'].min().date()}  →  {df['date'].max().date()}")
    print(f"  Days span          : {(df['date'].max() - df['date'].min()).days + 1} calendar days")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"\n  {GREEN}✓ No missing values found!{RESET}")
    else:
        print(f"\n  {YELLOW}⚠  Missing values detected:{RESET}")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"      {col}: {cnt} ({pct:.1f}%)")


def descriptive_stats(df: pd.DataFrame):
    """Prints mean, std, min, max for each numeric column."""
    _section("📊  Descriptive Statistics")

    numeric = df.select_dtypes(include="number")
    stats   = numeric.describe().T[["mean","std","min","max"]]

    labels = {
        "mood_score":        "Mood score      (1–10)",
        "sleep_hours":       "Sleep hours     (0–24)",
        "exercise_minutes":  "Exercise min    (0–300)",
        "stress_level":      "Stress level    (1–10)",
        "energy_level":      "Energy level    (1–10)",
    }

    print(f"\n  {'FEATURE':<28} {'MEAN':>6}  {'STD':>6}  {'MIN':>5}  {'MAX':>5}")
    print(f"  {'─'*28} {'─'*6}  {'─'*6}  {'─'*5}  {'─'*5}")

    for col, row in stats.iterrows():
        label = labels.get(col, col)
        print(f"  {label:<28} {row['mean']:>6.2f}  {row['std']:>6.2f}"
              f"  {row['min']:>5.1f}  {row['max']:>5.1f}")


def category_breakdown(df: pd.DataFrame):
    """Shows value counts for categorical columns."""
    _section("🏷️   Categorical Columns")

    cat_cols = ["social_interaction", "appetite_change"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        total  = counts.sum()
        print(f"\n  {BOLD}{col.replace('_',' ').title()}{RESET}")
        for val, cnt in counts.items():
            bar = "█" * int(cnt / total * 20)
            print(f"    {val:<12} {cnt:>3}  {bar}  ({cnt/total*100:.0f}%)")


def correlation_summary(df: pd.DataFrame):
    """Shows how each feature correlates with mood score."""
    _section("🔗  Correlation with Mood Score")

    numeric = df.select_dtypes(include="number")
    corr    = numeric.corr()["mood_score"].drop("mood_score").sort_values(ascending=False)

    print(f"\n  {'FEATURE':<22} {'CORRELATION':>12}  {'RELATIONSHIP'}")
    print(f"  {'─'*22} {'─'*12}  {'─'*20}")

    for feat, val in corr.items():
        if val > 0.5:
            tag = f"{GREEN}  Strong positive ↑{RESET}"
        elif val > 0.2:
            tag = f"{GREEN}  Moderate positive ↑{RESET}"
        elif val < -0.5:
            tag = f"{RED}  Strong negative ↓{RESET}"
        elif val < -0.2:
            tag = f"{RED}  Moderate negative ↓{RESET}"
        else:
            tag = f"{DIM}  Weak / no correlation{RESET}"

        label = feat.replace("_", " ").title()
        print(f"  {label:<22} {val:>+.3f}       {tag}")


def weekly_trends(df: pd.DataFrame):
    """Groups data by week and shows average mood trend."""
    _section("📅  Weekly Mood Trend")

    df2 = df.copy()
    df2["week"] = df2["date"].dt.to_period("W")
    weekly = df2.groupby("week")["mood_score"].mean()

    print(f"\n  {'WEEK':<25} {'AVG MOOD':>9}  {'TREND BAR'}")
    print(f"  {'─'*25} {'─'*9}  {'─'*20}")

    prev = None
    for week, avg in weekly.items():
        bar = "▓" * int(avg * 2)
        if prev is None:
            arrow = ""
        elif avg > prev + 0.3:
            arrow = f" {GREEN}↑{RESET}"
        elif avg < prev - 0.3:
            arrow = f" {RED}↓{RESET}"
        else:
            arrow = f" {DIM}→{RESET}"
        print(f"  {str(week):<25} {avg:>9.2f}  {CYAN}{bar}{RESET}{arrow}")
        prev = avg


def data_quality_check(df: pd.DataFrame):
    """Flags potential data quality issues."""
    _section("🔍  Data Quality Check")

    issues = 0

    # Duplicate dates
    dupes = df[df.duplicated("date")]
    if not dupes.empty:
        print(f"  {YELLOW}⚠  Duplicate dates found: {len(dupes)} rows{RESET}")
        issues += 1

    # Outliers (values beyond expected range)
    checks = [
        ("mood_score",       1, 10),
        ("sleep_hours",      0, 24),
        ("exercise_minutes", 0, 300),
        ("stress_level",     1, 10),
        ("energy_level",     1, 10),
    ]
    for col, lo, hi in checks:
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if not bad.empty:
            print(f"  {YELLOW}⚠  {col}: {len(bad)} value(s) outside [{lo}–{hi}]{RESET}")
            issues += 1

    # Gaps in dates (missing days)
    date_range = pd.date_range(df["date"].min(), df["date"].max())
    missing_days = set(date_range) - set(df["date"])
    if missing_days:
        print(f"  {DIM}  ℹ  {len(missing_days)} day(s) not logged (normal — you don't have to log every day){RESET}")

    if issues == 0:
        print(f"  {GREEN}✓ All values are within valid ranges. Data looks clean!{RESET}")


def quick_insights(df: pd.DataFrame):
    """Prints 3–5 human-readable insights from the data."""
    _section("💡  Key Insights")

    avg_mood    = df["mood_score"].mean()
    avg_sleep   = df["sleep_hours"].mean()
    avg_stress  = df["stress_level"].mean()
    avg_energy  = df["energy_level"].mean()

    # Best and worst days
    best_day  = df.loc[df["mood_score"].idxmax()]
    worst_day = df.loc[df["mood_score"].idxmin()]

    # Sleep vs mood correlation
    sleep_corr = df["sleep_hours"].corr(df["mood_score"])
    stress_corr = df["stress_level"].corr(df["mood_score"])
    ex_corr    = df["exercise_minutes"].corr(df["mood_score"])

    print(f"\n  1. Your average mood is {BOLD}{avg_mood:.1f}/10{RESET} "
          f"{'— doing well! 🙂' if avg_mood >= 6 else '— could improve 😔'}")

    print(f"\n  2. You average {BOLD}{avg_sleep:.1f} hrs of sleep{RESET}. "
          f"{'Good — 7+ hours is the healthy target ✓' if avg_sleep >= 7 else 'Try to aim for 7+ hours for better mood.'}")

    print(f"\n  3. Sleep {'strongly' if abs(sleep_corr) > 0.5 else 'moderately'} "
          f"{'helps' if sleep_corr > 0 else 'does not clearly affect'} your mood "
          f"(r = {sleep_corr:+.2f}).")

    print(f"\n  4. Stress {'strongly drags' if stress_corr < -0.5 else 'negatively affects'} "
          f"your mood (r = {stress_corr:+.2f}). Average stress: {avg_stress:.1f}/10.")

    print(f"\n  5. Your best day was {BOLD}{best_day['date'].date()}{RESET} "
          f"(mood {best_day['mood_score']})  |  "
          f"Worst: {BOLD}{worst_day['date'].date()}{RESET} "
          f"(mood {worst_day['mood_score']}).")

    if ex_corr > 0.3:
        print(f"\n  6. {GREEN}Exercise is clearly boosting your mood (r = {ex_corr:+.2f}). Keep it up!{RESET}")


def run():
    df = load_data()
    basic_info(df)
    descriptive_stats(df)
    category_breakdown(df)
    correlation_summary(df)
    weekly_trends(df)
    data_quality_check(df)
    quick_insights(df)

    print(f"\n{CYAN}{BOLD}{'─'*55}{RESET}")
    print(f"{GREEN}{BOLD}  ✅  EDA complete! Run preprocess.py next.{RESET}")
    print(f"{CYAN}{'─'*55}{RESET}\n")

    return df   # useful when imported in a notebook


if __name__ == "__main__":
    run()
