"""
preprocess.py  —  Data Cleaning & Encoding
-------------------------------------------
Cleans the raw CSV, handles missing values, encodes categorical columns,
and returns a model-ready DataFrame.

Run directly to see a before/after comparison:
    python src/preprocess.py
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "data", "user_logs.csv")

RESET = "\033[0m"; BOLD = "\033[1m"; CYAN = "\033[96m"
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; DIM = "\033[2m"


# ── Category mappings ─────────────────────────────────────────────────────────
# We convert text categories into ordered numbers the model can understand.

SOCIAL_MAP = {"low": 0, "medium": 1, "high": 2}
APPETITE_MAP = {"reduced": 0, "normal": 1, "increased": 2}


def load_raw(filepath=LOG_FILE) -> pd.DataFrame:
    """
    Loads the CSV and enforces correct column types.
    Accepts a custom filepath so it can also load the Kaggle dataset in Week 3.
    """
    if not os.path.isfile(filepath):
        print(f"{RED}  ✗ File not found: {filepath}{RESET}")
        sys.exit(1)

    df = pd.read_csv(filepath, parse_dates=["date"])

    numeric_cols = [
        "mood_score", "sleep_hours", "exercise_minutes",
        "stress_level", "energy_level",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values using sensible strategies:
    - Numeric columns  → forward fill first, then fill with column median
    - Category columns → forward fill, then fill with mode (most common value)
    - notes column     → empty string (it's optional free text)
    """
    df = df.copy()

    numeric_cols = ["mood_score", "sleep_hours", "exercise_minutes",
                    "stress_level", "energy_level"]
    cat_cols     = ["social_interaction", "appetite_change"]

    before_missing = df.isnull().sum().sum()

    # Numeric: forward fill (yesterday's value) then median
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            df[col] = df[col].fillna(df[col].median())

    # Categorical: forward fill then mode
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "normal")

    # Notes: just use empty string
    if "notes" in df.columns:
        df["notes"] = df["notes"].fillna("")

    after_missing = df.isnull().sum().sum()

    if before_missing > 0:
        print(f"  {YELLOW}ℹ  Filled {before_missing} missing values → {after_missing} remaining{RESET}")
    else:
        print(f"  {GREEN}✓ No missing values to handle{RESET}")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only the first entry per date if duplicates exist."""
    before = len(df)
    df = df.drop_duplicates(subset="date", keep="first")
    removed = before - len(df)
    if removed:
        print(f"  {YELLOW}ℹ  Removed {removed} duplicate date(s){RESET}")
    else:
        print(f"  {GREEN}✓ No duplicate dates found{RESET}")
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clips values to their valid ranges to handle any data entry errors.
    Does NOT delete rows — just corrects values that are out of bounds.
    """
    df = df.copy()
    clips = {
        "mood_score":       (1, 10),
        "sleep_hours":      (0, 24),
        "exercise_minutes": (0, 300),
        "stress_level":     (1, 10),
        "energy_level":     (1, 10),
    }
    clipped_any = False
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            bad = ((df[col] < lo) | (df[col] > hi)).sum()
            if bad:
                df[col] = df[col].clip(lo, hi)
                print(f"  {YELLOW}ℹ  Clipped {bad} out-of-range value(s) in '{col}'{RESET}")
                clipped_any = True

    if not clipped_any:
        print(f"  {GREEN}✓ All values are within valid ranges{RESET}")

    return df


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts text categories to numbers:
      social_interaction : low=0  medium=1  high=2
      appetite_change    : reduced=0  normal=1  increased=2
    Keeps original columns alongside so you can still read them.
    """
    df = df.copy()

    if "social_interaction" in df.columns:
        df["social_encoded"] = (
            df["social_interaction"]
            .str.strip()
            .str.lower()
            .map(SOCIAL_MAP)
            .fillna(1)   # default: medium
            .astype(int)
        )

    if "appetite_change" in df.columns:
        df["appetite_encoded"] = (
            df["appetite_change"]
            .str.strip()
            .str.lower()
            .map(APPETITE_MAP)
            .fillna(1)   # default: normal
            .astype(int)
        )

    print(f"  {GREEN}✓ Categorical columns encoded{RESET}")
    return df


def normalize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalises numeric features to the 0–1 range.
    This helps many ML algorithms converge faster.
    Creates new _norm columns and leaves originals intact.
    """
    df = df.copy()
    ranges = {
        "mood_score":       (1, 10),
        "sleep_hours":      (0, 24),
        "exercise_minutes": (0, 300),
        "stress_level":     (1, 10),
        "energy_level":     (1, 10),
    }
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            df[f"{col}_norm"] = (df[col] - lo) / (hi - lo)

    print(f"  {GREEN}✓ Numeric columns normalised (0–1 range){RESET}")
    return df


def get_feature_columns() -> list:
    """
    Returns the exact list of feature column names that will be used
    for ML training in Week 3.
    Call this from train.py to stay in sync.
    """
    return [
        "mood_score",
        "sleep_hours",
        "exercise_minutes",
        "stress_level",
        "energy_level",
        "social_encoded",
        "appetite_encoded",
    ]


def clean(filepath=LOG_FILE) -> pd.DataFrame:
    """
    Master function: runs the full cleaning pipeline and returns
    a clean, model-ready DataFrame.

    Steps:
      1. Load raw CSV
      2. Remove duplicates
      3. Handle missing values
      4. Clip outliers
      5. Encode categories
      6. Normalize numeric columns
    """
    print(f"\n{CYAN}{BOLD}  Preprocessing pipeline{RESET}")
    print(f"  {DIM}{'─'*40}{RESET}")

    df = load_raw(filepath)
    print(f"  Loaded {len(df)} rows from CSV")

    df = remove_duplicates(df)
    df = handle_missing(df)
    df = clip_outliers(df)
    df = encode_categories(df)
    df = normalize_numeric(df)

    print(f"\n  {GREEN}{BOLD}✓ Clean DataFrame ready — {len(df)} rows, {len(df.columns)} columns{RESET}")
    return df


# ── Demo run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_raw   = load_raw()
    df_clean = clean()

    print(f"\n{CYAN}{BOLD}  Before vs After{RESET}")
    print(f"  {'─'*40}")
    print(f"  Raw shape   : {df_raw.shape}")
    print(f"  Clean shape : {df_clean.shape}")

    print(f"\n{CYAN}{BOLD}  Clean column list:{RESET}")
    for col in df_clean.columns:
        print(f"    • {col}")

    print(f"\n{CYAN}{BOLD}  Sample row (most recent):{RESET}")
    last = df_clean.iloc[-1]
    for col in get_feature_columns():
        print(f"    {col:<25} {last[col]}")

    print(f"\n{GREEN}{BOLD}  ✅  Preprocessing complete! features.py next.{RESET}\n")
