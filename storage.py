"""
storage.py
----------
Handles all reading and writing of mood log data to CSV.
The CSV file lives at data/user_logs.csv relative to the project root.
"""

import csv
import os
from datetime import date


# ── Path setup ───────────────────────────────────────────────────────────────

# This file lives in src/, so we go one level up to reach the project root.
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LOG_FILE   = os.path.join(DATA_DIR, "user_logs.csv")

# Exact column order used in every row and the CSV header.
COLUMNS = [
    "date",
    "mood_score",          # 1–10
    "sleep_hours",         # 0.0–24.0
    "exercise_minutes",    # 0–300
    "stress_level",        # 1–10
    "social_interaction",  # low / medium / high
    "appetite_change",     # normal / reduced / increased
    "energy_level",        # 1–10
    "notes",               # free-text (optional)
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_data_dir():
    """Creates the data/ directory if it doesn't already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def _file_exists():
    """Returns True if the CSV log file already exists and has content."""
    return os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 0


# ── Public API ────────────────────────────────────────────────────────────────

def save_entry(entry: dict) -> bool:
    """
    Appends one daily entry (a dict) to the CSV file.
    Creates the file and writes the header row on first use.

    Parameters
    ----------
    entry : dict
        Must contain all keys listed in COLUMNS.

    Returns
    -------
    bool
        True if saved successfully, False on any error.
    """
    _ensure_data_dir()

    try:
        write_header = not _file_exists()

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)

            if write_header:
                writer.writeheader()

            # Only write the known columns — ignore any extra keys.
            row = {col: entry.get(col, "") for col in COLUMNS}
            writer.writerow(row)

        return True

    except OSError as e:
        print(f"\n  [storage] Error saving entry: {e}")
        return False


def load_all_entries() -> list[dict]:
    """
    Reads every entry from the CSV and returns them as a list of dicts.
    Returns an empty list if the file doesn't exist yet.
    """
    if not _file_exists():
        return []

    try:
        with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    except OSError as e:
        print(f"\n  [storage] Error reading log: {e}")
        return []


def entry_exists_for_today() -> bool:
    """
    Returns True if there is already a log entry dated today.
    Useful to prevent duplicate entries for the same day.
    """
    today = str(date.today())
    entries = load_all_entries()
    return any(e.get("date") == today for e in entries)


def get_recent_entries(n: int = 7) -> list[dict]:
    """
    Returns the last n entries (default: 7 days) sorted oldest-first.
    Useful for computing rolling averages and trend visualizations.
    """
    entries = load_all_entries()
    return entries[-n:] if len(entries) >= n else entries


def count_entries() -> int:
    """Returns the total number of logged days."""
    return len(load_all_entries())
