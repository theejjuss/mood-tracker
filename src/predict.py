"""
predict.py  —  Personal Risk Prediction
-----------------------------------------
Loads your saved model and runs it against YOUR personal user_logs.csv.
Prints a risk prediction for each logged day and a final summary.

Run:
    python src\predict.py
"""

import os, sys, joblib
import pandas as pd
import numpy as np
from datetime import date

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.pkl")
LOG_FILE   = os.path.join(BASE_DIR, "data", "user_logs.csv")

RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"
GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; DIM="\033[2m"

RISK_NAMES  = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {0: GREEN, 1: YELLOW, 2: RED}
RISK_EMOJI  = {0: "🟢", 1: "🟡", 2: "🔴"}

ADVICE = {
    0: [
        "Great work — keep up your current routine!",
        "Your sleep and activity levels are supporting your mood well.",
        "Maintain your social connections — they're clearly helping.",
    ],
    1: [
        "Try to get at least 7 hours of sleep tonight.",
        "Even a 15-minute walk can lift your mood noticeably.",
        "Reach out to a friend or family member today.",
        "Take 5 minutes to write down 3 things you're grateful for.",
    ],
    2: [
        "Please consider talking to someone you trust today.",
        "iCall (India): 9152987821  |  Vandrevala: 1860-2662-345",
        "Even small steps matter — try a short walk or a glass of water.",
        "You don't have to feel this way alone. Help is available.",
    ],
}


def _section(title):
    print(f"\n{CYAN}{BOLD}{'─'*52}{RESET}")
    print(f"{CYAN}{BOLD}   {title}{RESET}")
    print(f"{CYAN}{'─'*52}{RESET}")


def load_model():
    for path, name in [(MODEL_PATH, "model.pkl"), (META_PATH, "model_meta.pkl")]:
        if not os.path.isfile(path):
            print(f"{RED}  ✗ {name} not found. Run train.py first.{RESET}\n")
            sys.exit(1)

    model = joblib.load(MODEL_PATH)
    meta  = joblib.load(META_PATH)
    return model, meta["feature_names"]


def load_personal_logs() -> pd.DataFrame:
    if not os.path.isfile(LOG_FILE):
        print(f"{RED}  ✗ user_logs.csv not found.{RESET}")
        print(f"  Run 'python main.py' to log your first entry.\n")
        sys.exit(1)

    df = pd.read_csv(LOG_FILE, parse_dates=["date"])

    numeric_cols = ["mood_score","sleep_hours","exercise_minutes",
                    "stress_level","energy_level"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode categories
    df["social_encoded"]  = df["social_interaction"].str.lower().map(
        {"low": 0, "medium": 1, "high": 2}).fillna(1)
    df["appetite_encoded"] = df["appetite_change"].str.lower().map(
        {"reduced": 0, "normal": 1, "increased": 2}).fillna(1)

    return df.sort_values("date").reset_index(drop=True)


def predict_all(model, feature_cols, df: pd.DataFrame):
    """Runs prediction for every row in the personal log."""
    X = df[feature_cols].fillna(df[feature_cols].median()).astype(float)

    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)   # confidence scores

    return predictions, probabilities


def show_history(df, predictions, probabilities):
    _section("📅  Your Personal Risk History")

    print(f"\n  {'DATE':<12} {'MOOD':>5} {'SLEEP':>6} {'STRESS':>7}  "
          f"{'RISK':<10} {'CONFIDENCE':>10}")
    print(f"  {'─'*12} {'─'*5} {'─'*6} {'─'*7}  {'─'*10} {'─'*10}")

    for i, (_, row) in enumerate(df.iterrows()):
        risk  = predictions[i]
        conf  = probabilities[i][risk] * 100
        color = RISK_COLORS[risk]
        emoji = RISK_EMOJI[risk]

        print(
            f"  {str(row['date'].date()):<12} "
            f"{row['mood_score']:>5.0f} "
            f"{row['sleep_hours']:>6.1f} "
            f"{row['stress_level']:>7.0f}  "
            f"{color}{emoji} {RISK_NAMES[risk]:<8}{RESET} "
            f"{conf:>9.1f}%"
        )


def show_today_prediction(df, predictions, probabilities):
    _section("🎯  Today's Risk Assessment")

    last_row  = df.iloc[-1]
    last_pred = predictions[-1]
    last_prob = probabilities[-1]
    color     = RISK_COLORS[last_pred]
    emoji     = RISK_EMOJI[last_pred]

    print(f"\n  Date          : {last_row['date'].date()}")
    print(f"  Mood score    : {last_row['mood_score']:.0f}/10")
    print(f"  Sleep hours   : {last_row['sleep_hours']:.1f}")
    print(f"  Stress level  : {last_row['stress_level']:.0f}/10")
    print(f"  Energy level  : {last_row['energy_level']:.0f}/10")

    print(f"\n  {BOLD}Risk Level    :  {color}{emoji}  {RISK_NAMES[last_pred].upper()}{RESET}")

    # Confidence bar for each class
    print(f"\n  Model confidence:")
    for i, name in RISK_NAMES.items():
        pct = last_prob[i] * 100
        bar = "▓" * int(pct / 5)
        c   = RISK_COLORS[i]
        print(f"    {name:<8} {pct:>5.1f}%  {c}{bar}{RESET}")

    # Personalised advice
    import random
    random.seed(int(last_row["stress_level"]))
    advice_list = ADVICE[last_pred]
    print(f"\n  {BOLD}Advice:{RESET}")
    for tip in advice_list[:2]:
        print(f"  {color}→{RESET} {tip}")


def show_trend_summary(predictions):
    _section("📊  7-Day Risk Trend Summary")

    recent = predictions[-7:] if len(predictions) >= 7 else predictions
    counts = {0: 0, 1: 0, 2: 0}
    for p in recent:
        counts[p] += 1

    total = len(recent)
    print(f"\n  Based on your last {total} logged day(s):\n")

    for risk, cnt in counts.items():
        if cnt == 0:
            continue
        bar   = "█" * int(cnt / total * 25)
        color = RISK_COLORS[risk]
        emoji = RISK_EMOJI[risk]
        print(f"  {emoji} {RISK_NAMES[risk]:<8} {cnt:>2} day(s)  {color}{bar}{RESET}")

    # Overall trend message
    avg_risk = np.mean(recent)
    if avg_risk < 0.5:
        msg = f"{GREEN}Your mental wellness trend looks strong. Keep it up!{RESET}"
    elif avg_risk < 1.2:
        msg = f"{YELLOW}Moderate risk trend. Focus on sleep and stress management.{RESET}"
    else:
        msg = f"{RED}High risk trend detected. Please reach out for support.{RESET}"

    print(f"\n  {msg}")


def run():
    model, feature_cols = load_model()

    _section("📂  Loading Your Personal Logs")
    df = load_personal_logs()
    print(f"  Found {len(df)} logged day(s)")
    print(f"  Features used: {feature_cols}")

    _section("🤖  Running Predictions")
    predictions, probabilities = predict_all(model, feature_cols, df)
    print(f"  Predictions complete for {len(predictions)} day(s)")

    show_history(df, predictions, probabilities)
    show_today_prediction(df, predictions, probabilities)
    show_trend_summary(predictions)

    print(f"\n{CYAN}{BOLD}{'─'*52}{RESET}")
    print(f"{GREEN}{BOLD}  ✅  Prediction complete!{RESET}")
    print(f"  Run {CYAN}python main.py{RESET} daily to keep tracking.")
    print(f"  Week 4 next: charts, visualizations & PDF report.\n")


if __name__ == "__main__":
    run()
