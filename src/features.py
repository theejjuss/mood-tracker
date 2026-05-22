"""
features.py  —  Feature Engineering
--------------------------------------
Takes the clean DataFrame from preprocess.py and creates new,
more informative columns (features) that help the ML model
detect patterns that raw daily scores miss.

Key features engineered:
  • 7-day rolling averages   (trend context)
  • Volatility scores        (how stable is mood/sleep)
  • Wellness composite score (single health summary)
  • Risk label               (Low / Medium / High)

Run directly:
    python src/features.py
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR + "/src")

from preprocess import clean, get_feature_columns

RESET = "\033[0m"; BOLD = "\033[1m"; CYAN = "\033[96m"
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; DIM = "\033[2m"


# ── Rolling averages ──────────────────────────────────────────────────────────

def add_rolling_averages(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Adds a rolling 7-day average for each numeric feature.
    Why? A single bad day looks different from a week of bad days.
    The model learns better from trends than single data points.

    min_periods=1 ensures we get values even at the start of the dataset
    (when there aren't 7 prior rows yet).
    """
    df = df.copy()
    numeric_cols = [
        "mood_score", "sleep_hours", "exercise_minutes",
        "stress_level", "energy_level",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_7d_avg"] = (
                df[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .round(2)
            )
    return df


# ── Volatility ────────────────────────────────────────────────────────────────

def add_volatility(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Adds 7-day rolling standard deviation for mood and sleep.
    Why? High variability in mood (mood swings) is a stronger warning sign
    than a consistently low mood — the model should know about this.
    """
    df = df.copy()
    for col in ["mood_score", "sleep_hours"]:
        if col in df.columns:
            df[f"{col}_volatility"] = (
                df[col]
                .rolling(window=window, min_periods=2)
                .std()
                .fillna(0)
                .round(3)
            )
    return df


# ── Day-of-week ───────────────────────────────────────────────────────────────

def add_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds day_of_week (0=Mon … 6=Sun) and is_weekend (0/1).
    Why? Many people show lower mood on Mondays or higher mood on weekends.
    This gives the model a time context it wouldn't otherwise have.
    """
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek          # 0 = Monday
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    return df


# ── Mood change ───────────────────────────────────────────────────────────────

def add_mood_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds mood_change: difference between today's mood and yesterday's.
    A sudden large drop is a stronger signal than a gradual decline.
    """
    df = df.copy()
    df["mood_change"] = df["mood_score"].diff().fillna(0).round(2)
    return df


# ── Wellness composite score ──────────────────────────────────────────────────

def add_wellness_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a single wellness_score (0–10) that combines all features
    into one number. Useful for quick summaries and visualizations.

    Formula (weighted average, designed for interpretability):
      • Mood        weight 0.30   (most direct indicator)
      • Sleep       weight 0.25   (fundamental to mental health)
      • Stress      weight 0.20   (inverted — high stress = lower score)
      • Energy      weight 0.15
      • Exercise    weight 0.10   (normalised to 0–10 from 0–300 min)
    """
    df = df.copy()

    mood_norm     = df["mood_score"] / 10
    sleep_norm    = df["sleep_hours"].clip(0, 10) / 10
    stress_norm   = 1 - (df["stress_level"] / 10)   # inverted
    energy_norm   = df["energy_level"] / 10
    exercise_norm = df["exercise_minutes"].clip(0, 60) / 60  # cap at 60 min

    df["wellness_score"] = (
        mood_norm     * 0.30 +
        sleep_norm    * 0.25 +
        stress_norm   * 0.20 +
        energy_norm   * 0.15 +
        exercise_norm * 0.10
    ).round(3) * 10   # scale back to 0–10

    return df


# ── Risk label ────────────────────────────────────────────────────────────────

def add_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the TARGET column the ML model will learn to predict.

    Label logic (using wellness_score):
      • wellness_score >= 6.0  →  0  (Low risk)
      • wellness_score >= 4.0  →  1  (Medium risk)
      • wellness_score <  4.0  →  2  (High risk)

    This is a simplified heuristic. In Week 3, we train on the Kaggle
    dataset which has clinically-informed labels.
    """
    df = df.copy()

    def _label(score):
        if score >= 6.0:
            return 0   # Low
        elif score >= 4.0:
            return 1   # Medium
        else:
            return 2   # High

    df["risk_label"]      = df["wellness_score"].apply(_label)
    df["risk_label_text"] = df["risk_label"].map({
        0: "Low",
        1: "Medium",
        2: "High",
    })
    return df


# ── Master function ───────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full feature engineering pipeline.
    Call this after preprocess.clean() in Week 3.
    """
    df = add_rolling_averages(df)
    df = add_volatility(df)
    df = add_day_features(df)
    df = add_mood_change(df)
    df = add_wellness_score(df)
    df = add_risk_label(df)
    return df


def get_all_feature_columns() -> list:
    """
    Returns the full expanded feature list (base + engineered).
    Used by train.py in Week 3 to select X columns.
    """
    base = get_feature_columns()
    engineered = [
        "mood_score_7d_avg",
        "sleep_hours_7d_avg",
        "stress_level_7d_avg",
        "energy_level_7d_avg",
        "exercise_minutes_7d_avg",
        "mood_score_volatility",
        "sleep_hours_volatility",
        "day_of_week",
        "is_weekend",
        "mood_change",
        "wellness_score",
    ]
    return base + engineered


# ── Demo run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: load and clean raw data
    df_clean = clean()

    # Step 2: engineer features
    df_feat = engineer_features(df_clean)

    print(f"\n{CYAN}{BOLD}  Engineered Features Added{RESET}")
    print(f"  {'─'*45}")

    new_cols = [c for c in df_feat.columns if c not in df_clean.columns]
    for col in new_cols:
        sample = df_feat[col].iloc[-1]
        print(f"  {GREEN}+{RESET} {col:<35} sample: {sample}")

    print(f"\n{CYAN}{BOLD}  Risk Label Distribution{RESET}")
    print(f"  {'─'*30}")
    counts = df_feat["risk_label_text"].value_counts()
    total  = len(df_feat)
    for label, cnt in counts.items():
        bar = "█" * int(cnt / total * 20)
        print(f"  {label:<10} {cnt:>3} days  {CYAN}{bar}{RESET}")

    print(f"\n{CYAN}{BOLD}  Latest Day Summary{RESET}")
    print(f"  {'─'*30}")
    last = df_feat.iloc[-1]
    print(f"  Date           : {last['date'].date()}")
    print(f"  Wellness score : {last['wellness_score']:.2f} / 10")
    print(f"  7-day mood avg : {last['mood_score_7d_avg']:.2f}")
    print(f"  Risk level     : ", end="")

    risk = last["risk_label_text"]
    color = GREEN if risk == "Low" else (YELLOW if risk == "Medium" else RED)
    print(f"{color}{BOLD}{risk}{RESET}")

    print(f"\n{GREEN}{BOLD}  ✅  Feature engineering complete!{RESET}")
    print(f"  {DIM}Total features for ML model: {len(get_all_feature_columns())}{RESET}\n")
