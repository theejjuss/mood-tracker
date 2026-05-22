"""
report.py  -  Weekly PDF Report Generator
-------------------------------------------
Pulls your mood data, model predictions, and charts together into
a clean, readable PDF saved as reports/mood_report_YYYY-MM-DD.pdf

Run:
    python src\report.py
"""

import os, sys
import pandas as pd
import numpy as np
from datetime import date, timedelta

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE    = os.path.join(BASE_DIR, "data", "user_logs.csv")
CHARTS_DIR  = os.path.join(BASE_DIR, "charts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH   = os.path.join(BASE_DIR, "models", "model_meta.pkl")

RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"
GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; DIM="\033[2m"

RISK_NAMES  = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS_HEX = {0: (76,175,80), 1: (255,152,0), 2: (244,67,54)}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(LOG_FILE, parse_dates=["date"])
    numeric = ["mood_score","sleep_hours","exercise_minutes","stress_level","energy_level"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["social_encoded"]   = df["social_interaction"].str.lower().map(
        {"low":0,"medium":1,"high":2}).fillna(1)
    df["appetite_encoded"] = df["appetite_change"].str.lower().map(
        {"reduced":0,"normal":1,"increased":2}).fillna(1)
    return df.sort_values("date").reset_index(drop=True)


def get_predictions(df):
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        feats = ["mood_score","sleep_hours","exercise_minutes",
                 "stress_level","energy_level","social_encoded","appetite_encoded"]
        X = df[feats].fillna(df[feats].median()).astype(float)
        return model.predict(X), model.predict_proba(X)
    except Exception:
        return None, None


def generate_charts_if_needed():
    """Runs visualize.py if charts haven't been generated yet."""
    mood_chart = os.path.join(CHARTS_DIR, "mood_trend.png")
    if not os.path.isfile(mood_chart):
        print(f"  {YELLOW}Charts not found - generating them first...{RESET}")
        sys.path.insert(0, os.path.join(BASE_DIR, "src"))
        import visualize
        visualize.run()


# ── PDF builder ───────────────────────────────────────────────────────────────

def build_pdf(df, predictions, probabilities):
    from fpdf import FPDF

    os.makedirs(REPORTS_DIR, exist_ok=True)
    today     = date.today()
    filename  = f"mood_report_{today}.pdf"
    out_path  = os.path.join(REPORTS_DIR, filename)

    # Last 7 days
    week_df   = df.tail(7).reset_index(drop=True)
    week_pred = predictions[-7:] if predictions is not None else None
    week_prob = probabilities[-7:] if probabilities is not None else None

    # ── Stats ─────────────────────────────────────────────────────────────────
    avg_mood   = week_df["mood_score"].mean()
    avg_sleep  = week_df["sleep_hours"].mean()
    avg_stress = week_df["stress_level"].mean()
    avg_energy = week_df["energy_level"].mean()
    avg_ex     = week_df["exercise_minutes"].mean()

    today_row  = df.iloc[-1]
    today_risk = int(predictions[-1]) if predictions is not None else None

    risk_counts = {}
    if week_pred is not None:
        for i in range(3):
            risk_counts[RISK_NAMES[i]] = int((week_pred == i).sum())

    # ── PDF setup ─────────────────────────────────────────────────────────────
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(63, 81, 181)       # indigo
    pdf.rect(0, 0, 210, 38, style="F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "Mental Health Mood Tracker", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_xy(10, 20)
    pdf.cell(0, 8, f"Weekly Report  |  Generated: {today.strftime('%B %d, %Y')}", ln=True)

    pdf.set_xy(10, 30)
    date_range = f"{week_df['date'].iloc[0].strftime('%b %d')}  to  {week_df['date'].iloc[-1].strftime('%b %d, %Y')}"
    pdf.cell(0, 6, f"Period: {date_range}  |  Days tracked: {len(week_df)}", ln=True)

    pdf.set_text_color(30, 30, 30)
    pdf.set_y(46)

    # ── Today's risk box ──────────────────────────────────────────────────────
    if today_risk is not None:
        r, g, b = RISK_COLORS_HEX[today_risk]
        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_x(10)
        label = f"Today's Risk Level:  {RISK_NAMES[today_risk].upper()}"
        conf  = float(probabilities[-1][today_risk]) * 100 if probabilities is not None else 0
        pdf.cell(190, 12, f"{label}   ({conf:.0f}% confidence)", ln=True,
                 align="C", fill=True, border=0)
        pdf.set_text_color(30, 30, 30)
        pdf.ln(5)

    # ── Weekly stats grid ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_x(10)
    pdf.cell(0, 8, "7-Day Averages", ln=True)
    pdf.ln(2)

    stats = [
        ("Mood Score",      f"{avg_mood:.1f} / 10",   (92, 107, 192)),
        ("Sleep Hours",     f"{avg_sleep:.1f} hrs",    (38, 198, 218)),
        ("Stress Level",    f"{avg_stress:.1f} / 10",  (239, 83, 80)),
        ("Energy Level",    f"{avg_energy:.1f} / 10",  (255, 167, 38)),
        ("Exercise",        f"{avg_ex:.0f} min/day",   (102, 187, 106)),
    ]

    col_w = 36
    x_start = 10
    for i, (label, value, color) in enumerate(stats):
        x = x_start + i * (col_w + 2)
        r, g, b = color
        pdf.set_fill_color(r, g, b)
        pdf.set_xy(x, pdf.get_y())
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(col_w, 7, label, ln=0, align="C", fill=True)
        pdf.set_xy(x, pdf.get_y() + 7)
        pdf.set_fill_color(245, 245, 245)
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(col_w, 10, value, ln=0, align="C", fill=True, border=1)

    pdf.ln(18)

    # ── Risk distribution ─────────────────────────────────────────────────────
    if risk_counts:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_x(10)
        pdf.cell(0, 8, "Risk Level Breakdown  (last 7 days)", ln=True)
        pdf.ln(1)

        total = sum(risk_counts.values())
        for risk_name, cnt in risk_counts.items():
            risk_id = [k for k, v in RISK_NAMES.items() if v == risk_name][0]
            r, g, b = RISK_COLORS_HEX[risk_id]
            bar_w = int((cnt / max(total, 1)) * 120)

            pdf.set_x(10)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(40, 7, f"{risk_name} Risk:", ln=0)

            pdf.set_fill_color(r, g, b)
            pdf.cell(bar_w if bar_w > 0 else 2, 7, "", ln=0, fill=True)

            pdf.set_fill_color(220, 220, 220)
            pdf.cell(120 - bar_w, 7, "", ln=0, fill=True)

            pdf.set_text_color(30, 30, 30)
            pdf.cell(30, 7, f"  {cnt} day(s)  ({cnt/total*100:.0f}%)", ln=True)

        pdf.ln(4)

    # ── Daily log table ───────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_x(10)
    pdf.cell(0, 8, "Daily Log", ln=True)
    pdf.ln(1)

    headers = ["Date", "Mood", "Sleep", "Stress", "Energy", "Social", "Risk"]
    col_ws  = [32, 20, 20, 20, 20, 28, 26]

    # Header row
    pdf.set_fill_color(63, 81, 181)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(10)
    for h, w in zip(headers, col_ws):
        pdf.cell(w, 8, h, border=0, align="C", fill=True)
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 9)
    for i, (_, row) in enumerate(week_df.iterrows()):
        bg = (255, 255, 255) if i % 2 == 0 else (245, 245, 250)
        pdf.set_fill_color(*bg)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(10)

        if week_pred is not None:
            risk_id  = int(week_pred[i])
            risk_lbl = RISK_NAMES[risk_id]
        else:
            risk_id, risk_lbl = 0, "-"

        vals = [
            row["date"].strftime("%b %d, %Y"),
            f"{row['mood_score']:.0f}/10",
            f"{row['sleep_hours']:.1f}h",
            f"{row['stress_level']:.0f}/10",
            f"{row['energy_level']:.0f}/10",
            str(row.get("social_interaction", "")).title(),
            risk_lbl,
        ]

        for j, (v, w) in enumerate(zip(vals, col_ws)):
            if j == 6 and week_pred is not None:   # colour the risk cell
                r2, g2, b2 = RISK_COLORS_HEX[risk_id]
                pdf.set_fill_color(r2, g2, b2)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(w, 7, v, border=0, align="C", fill=True)
                pdf.set_fill_color(*bg)
                pdf.set_text_color(30, 30, 30)
            else:
                pdf.cell(w, 7, str(v), border=0, align="C", fill=True)
        pdf.ln()

    pdf.ln(6)

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_files = [
        ("mood_trend.png",        "Mood Trend Over Time"),
        ("sleep_vs_mood.png",     "Sleep Hours vs Mood Score"),
        ("stress_vs_mood.png",    "Stress Level vs Mood Score"),
        ("risk_distribution.png", "Risk Distribution"),
    ]

    for fname, title in chart_files:
        fpath = os.path.join(CHARTS_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        # Check if we need a new page
        if pdf.get_y() > 180:
            pdf.add_page()

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(10)
        pdf.set_text_color(63, 81, 181)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_text_color(30, 30, 30)

        pdf.image(fpath, x=10, w=185)
        pdf.ln(4)

    # ── Footer / disclaimer ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_x(10)
    pdf.set_text_color(63, 81, 181)
    pdf.cell(0, 10, "Insights & Recommendations", ln=True)
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 10)

    insights = []

    if avg_sleep < 7:
        insights.append(f">> Sleep: Averaging {avg_sleep:.1f} hrs - below the recommended 7+ hrs. "
                        "Try going to bed 30 minutes earlier each night.")
    else:
        insights.append(f">> Sleep: Averaging {avg_sleep:.1f} hrs - great! Consistent sleep supports stable mood.")

    if avg_stress > 6:
        insights.append(f">> Stress: Average stress {avg_stress:.1f}/10 is high. "
                        "Try 5 minutes of deep breathing daily or a short evening walk.")
    else:
        insights.append(f">> Stress: Average stress {avg_stress:.1f}/10 is manageable. Keep up good stress habits.")

    if avg_ex < 20:
        insights.append(">> Exercise: Less than 20 min/day on average. "
                        "Even a 15-minute walk has measurable mood benefits.")
    else:
        insights.append(f">> Exercise: Averaging {avg_ex:.0f} min/day - physical activity is clearly supporting your mood.")

    if avg_mood >= 7:
        insights.append(f">> Mood: Strong average of {avg_mood:.1f}/10 this week. You are doing well - keep your routines!")
    elif avg_mood >= 5:
        insights.append(f">> Mood: Moderate average of {avg_mood:.1f}/10. Focus on sleep and reducing stress for improvement.")
    else:
        insights.append(f">> Mood: Low average of {avg_mood:.1f}/10 this week. Consider talking to someone you trust.")

    for tip in insights:
        pdf.set_x(10)
        pdf.multi_cell(190, 7, tip)
        pdf.ln(2)

    pdf.ln(6)
    pdf.set_fill_color(255, 243, 224)
    pdf.set_x(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(190, 6,
        "DISCLAIMER: This report is generated by an educational ML project and is NOT a "
        "clinical diagnosis. Always consult a qualified mental health professional for "
        "medical advice. iCall (India): 9152987821  |  Vandrevala Foundation: 1860-2662-345",
        fill=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    pdf.output(out_path)
    return out_path, filename


# ── Master run ────────────────────────────────────────────────────────────────

def run():
    print(f"\n{CYAN}{BOLD}{'─'*52}{RESET}")
    print(f"{CYAN}{BOLD}   📄  Generating Weekly PDF Report{RESET}")
    print(f"{CYAN}{'─'*52}{RESET}\n")

    if not os.path.isfile(LOG_FILE):
        print(f"{RED}  ✗ No log file found. Run python main.py first.{RESET}\n")
        sys.exit(1)

    print(f"  Loading data …")
    df = load_data()
    print(f"  {len(df)} entries found")

    print(f"  Loading model predictions …")
    predictions, probabilities = get_predictions(df)
    if predictions is not None:
        print(f"  {GREEN}✓ Model predictions loaded{RESET}")
    else:
        print(f"  {YELLOW}⚠  No model found - report will skip risk predictions{RESET}")

    generate_charts_if_needed()

    print(f"  Building PDF …", end=" ", flush=True)
    out_path, filename = build_pdf(df, predictions, probabilities)
    print(f"{GREEN}done{RESET}")

    print(f"\n{GREEN}{BOLD}  ✅  Report saved!{RESET}")
    print(f"  {CYAN}reports\\{filename}{RESET}")
    print(f"\n  Open the PDF from your reports\\ folder.")
    print(f"  Share it, print it, or keep it as a personal record.\n")

    return out_path


if __name__ == "__main__":
    run()
