"""
train.py  —  Model Training
-----------------------------
Loads the Kaggle depression_anxiety_data.csv, maps its clinical columns
to risk labels, trains a Random Forest classifier, and saves the model
to models/model.pkl ready for prediction on your personal data.

Run:
    python src\train.py
"""

import os, sys, joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE_CSV = os.path.join(BASE_DIR, "data", "depression_anxiety_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.pkl")

RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"
GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; DIM="\033[2m"

def _section(title):
    print(f"\n{CYAN}{BOLD}{'─'*52}{RESET}")
    print(f"{CYAN}{BOLD}   {title}{RESET}")
    print(f"{CYAN}{'─'*52}{RESET}")


# ── Step 1: Load Kaggle dataset ───────────────────────────────────────────────

def load_kaggle(path=KAGGLE_CSV) -> pd.DataFrame:
    if not os.path.isfile(path):
        print(f"{RED}  ✗ File not found: {path}{RESET}")
        print(f"  Make sure depression_anxiety_data.csv is in your data\\ folder.\n")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"  Loaded Kaggle dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


# ── Step 2: Build risk label from PHQ-9 score ────────────────────────────────
#
# PHQ-9 is the gold-standard clinical depression screening tool.
# Score ranges (standard clinical thresholds):
#   0–4   → Minimal / No depression  → Risk: 0 (Low)
#   5–9   → Mild depression          → Risk: 0 (Low)
#   10–14 → Moderate depression      → Risk: 1 (Medium)
#   15–19 → Moderately severe        → Risk: 2 (High)
#   20–27 → Severe depression        → Risk: 2 (High)

PHQ_RISK_MAP = {
    range(0,  10): 0,   # Low
    range(10, 15): 1,   # Medium
    range(15, 28): 2,   # High
}

def phq_to_risk(score):
    try:
        s = int(score)
    except (ValueError, TypeError):
        return 1   # default Medium if unknown
    for r, label in PHQ_RISK_MAP.items():
        if s in r:
            return label
    return 2   # very high score → High

RISK_NAMES = {0: "Low", 1: "Medium", 2: "High"}


# ── Step 3: Select & engineer features from Kaggle columns ───────────────────
#
# The Kaggle dataset uses clinical scores. We map them to our 7 personal
# features so the trained model can also predict from your daily logs.
#
# Kaggle column      → Our feature
# phq_score (0-27)   → mood_score    (inverted & scaled to 1-10)
# epworth_score(0-24)→ sleep_hours   (inverted & scaled to 0-10)
# gad_score (0-21)   → stress_level  (scaled to 1-10)
# depressiveness     → energy_level  (inverted: True=low energy)
# anxiousness        → social_encoded(True=avoidant=0)
# sleepiness         → exercise_minutes (inverted proxy)
# suicidal           → appetite_encoded (severe signal)

def build_features(df: pd.DataFrame):
    """
    Maps Kaggle clinical columns → our 7 personal feature columns.
    Returns X (features) and y (risk labels) as numpy arrays.
    """
    feat = pd.DataFrame()

    # mood_score: PHQ inversely relates to mood (high PHQ = low mood)
    # PHQ range 0-27 → mood 1-10 (inverted)
    feat["mood_score"] = 10 - (
        pd.to_numeric(df["phq_score"], errors="coerce")
        .clip(0, 27) / 27 * 9
    ).round(1)

    # sleep_hours: Epworth sleepiness scale (high = sleepy = poor sleep)
    # Epworth 0-24 → sleep 0-10 (inverted)
    feat["sleep_hours"] = 10 - (
        pd.to_numeric(df["epworth_score"], errors="coerce")
        .clip(0, 24) / 24 * 10
    ).round(1)

    # exercise_minutes: sleepiness as proxy (0-24 → 0-60 min, inverted)
    feat["exercise_minutes"] = (
        (1 - pd.to_numeric(df["epworth_score"], errors="coerce")
         .clip(0, 24) / 24) * 60
    ).round(0)

    # stress_level: GAD-7 anxiety score (0-21 → 1-10)
    feat["stress_level"] = (
        pd.to_numeric(df["gad_score"], errors="coerce")
        .clip(0, 21) / 21 * 9 + 1
    ).round(1)

    # energy_level: depressiveness (True = low energy)
    feat["energy_level"] = df["depressiveness"].map(
        {True: 3, False: 7, 1: 3, 0: 7, "True": 3, "False": 7}
    ).fillna(5)

    # social_encoded: anxiousness (True = avoidant = low social)
    feat["social_encoded"] = df["anxiousness"].map(
        {True: 0, False: 2, 1: 0, 0: 2, "True": 0, "False": 2}
    ).fillna(1)

    # appetite_encoded: suicidal as severe appetite signal
    feat["appetite_encoded"] = df["suicidal"].map(
        {True: 0, False: 1, 1: 0, 0: 1, "True": 0, "False": 1}
    ).fillna(1)

    # Target label
    y = df["phq_score"].apply(phq_to_risk)

    # Drop rows with any NaN
    feat["_y"] = y
    feat = feat.dropna()
    y = feat.pop("_y").astype(int)
    X = feat.astype(float)

    return X, y


# ── Step 4: Train Random Forest ───────────────────────────────────────────────

def train(X, y):
    """
    Splits data 80/20, trains a Random Forest classifier, and returns
    the trained model along with the test split for evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,   # keeps class proportions equal in both splits
    )

    model = RandomForestClassifier(
        n_estimators=200,     # 200 decision trees in the forest
        max_depth=8,          # prevents overfitting
        min_samples_split=5,  # needs 5+ samples to split a node
        class_weight="balanced",  # handles unequal Low/Med/High counts
        random_state=42,
    )

    print(f"  Training on {len(X_train)} samples …", end=" ", flush=True)
    model.fit(X_train, y_train)
    print(f"{GREEN}done{RESET}")

    return model, X_train, X_test, y_train, y_test


# ── Step 5: Save model ────────────────────────────────────────────────────────

def save_model(model, feature_names):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"feature_names": feature_names, "risk_names": RISK_NAMES}, META_PATH)
    print(f"  {GREEN}✓ Model saved  →  models\\model.pkl{RESET}")
    print(f"  {GREEN}✓ Metadata saved  →  models\\model_meta.pkl{RESET}")


# ── Step 6: Feature importance ────────────────────────────────────────────────

def show_feature_importance(model, feature_names):
    _section("🌲  Feature Importance  (what the model learned)")

    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: -x[1])

    print(f"\n  {'FEATURE':<25} {'IMPORTANCE':>10}  BAR")
    print(f"  {'─'*25} {'─'*10}  {'─'*20}")

    for feat, imp in pairs:
        bar   = "█" * int(imp * 80)
        color = GREEN if imp > 0.2 else (CYAN if imp > 0.1 else DIM)
        print(f"  {feat:<25} {imp:>10.4f}  {color}{bar}{RESET}")


# ── Master run ────────────────────────────────────────────────────────────────

def run():
    _section("📂  Loading Kaggle Dataset")
    df = load_kaggle()

    _section("🏷️   Building Features & Risk Labels")
    X, y = build_features(df)

    counts = y.value_counts().sort_index()
    total  = len(y)
    print(f"\n  Risk label distribution ({total} samples):")
    for label, cnt in counts.items():
        bar = "█" * int(cnt / total * 25)
        print(f"    {RISK_NAMES[label]:<8} {cnt:>4}  {CYAN}{bar}{RESET}  ({cnt/total*100:.1f}%)")

    print(f"\n  Features used ({len(X.columns)}):")
    for col in X.columns:
        print(f"    • {col}")

    _section("🤖  Training Random Forest Classifier")
    model, X_train, X_test, y_train, y_test = train(X, y)

    _section("💾  Saving Model")
    save_model(model, list(X.columns))

    show_feature_importance(model, list(X.columns))

    _section("✅  Training Complete")
    print(f"\n  {GREEN}{BOLD}Model is ready!{RESET}")
    print(f"  Next step: run  {CYAN}python src\\evaluate.py{RESET}  to see accuracy & results.")
    print(f"  Then run  {CYAN}python src\\predict.py{RESET}  to predict YOUR personal risk.\n")

    # Pass test data to evaluate.py via a temp file
    import json
    tmp = {
        "X_test":  X_test.values.tolist(),
        "y_test":  y_test.tolist(),
        "columns": list(X.columns),
    }
    tmp_path = os.path.join(BASE_DIR, "models", "test_split.pkl")
    joblib.dump(tmp, tmp_path)

    return model, X_test, y_test


if __name__ == "__main__":
    run()
