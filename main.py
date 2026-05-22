"""
main.py
-------
Entry point for the Mood Tracker project.
Run this file every day to log your mental health data.

Usage:
    python main.py          ← log today's entry
    python main.py --view   ← view your recent entries
"""

import sys
import os

# Add src/ to the path so we can import our modules cleanly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from input import collect_entry
from storage import (
    save_entry,
    load_all_entries,
    entry_exists_for_today,
    get_recent_entries,
    count_entries,
)


# ── Colours ───────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


# ── View mode ─────────────────────────────────────────────────────────────────

def show_recent_entries():
    """Prints the last 7 days of entries as a readable table."""
    entries = get_recent_entries(7)

    if not entries:
        print(f"\n  {YELLOW}No entries yet. Run 'python main.py' to log your first day!{RESET}\n")
        return

    print(f"\n{CYAN}{BOLD}{'─' * 70}{RESET}")
    print(f"{CYAN}{BOLD}   📋  Last {len(entries)} entry/entries{RESET}")
    print(f"{CYAN}{BOLD}{'─' * 70}{RESET}")

    # Header row
    print(
        f"\n  {'DATE':<12} {'MOOD':>5} {'SLEEP':>6} {'EXERCISE':>9} "
        f"{'STRESS':>7} {'ENERGY':>7}  {'SOCIAL':<8}"
    )
    print(f"  {'─'*12} {'─'*5} {'─'*6} {'─'*9} {'─'*7} {'─'*7}  {'─'*8}")

    for e in entries:
        print(
            f"  {e.get('date','?'):<12} "
            f"{e.get('mood_score','?'):>5} "
            f"{e.get('sleep_hours','?'):>6} "
            f"{e.get('exercise_minutes','?'):>9} "
            f"{e.get('stress_level','?'):>7} "
            f"{e.get('energy_level','?'):>7}  "
            f"{e.get('social_interaction','?'):<8}"
        )
        if e.get("notes"):
            print(f"  {DIM}  Note: {e['notes']}{RESET}")

    print(f"\n  {DIM}Total entries logged: {count_entries()}{RESET}\n")


# ── Main flow ─────────────────────────────────────────────────────────────────

def main():
    # ── --view flag ───────────────────────────────────────────────────────────
    if "--view" in sys.argv or "-v" in sys.argv:
        show_recent_entries()
        return

    # ── Duplicate-entry guard ─────────────────────────────────────────────────
    if entry_exists_for_today():
        print(f"\n{YELLOW}{BOLD}  ⚠  You've already logged an entry for today.{RESET}")
        print(f"  Run 'python main.py --view' to see your recent entries.\n")
        return

    # ── Collect & save ────────────────────────────────────────────────────────
    try:
        entry = collect_entry()
    except KeyboardInterrupt:
        # User typed 'q' or pressed Ctrl+C — nothing was saved.
        return

    success = save_entry(entry)

    if success:
        total = count_entries()
        print(f"{GREEN}{BOLD}  ✅  Entry saved successfully!{RESET}")
        print(f"  {DIM}Total days logged: {total}{RESET}")
        print(f"  {DIM}Run 'python main.py --view' to see your history.{RESET}\n")

        # Gentle milestone messages.
        if total == 1:
            print(f"  {CYAN}🎉 Great start! Consistency is the key — come back tomorrow.{RESET}\n")
        elif total == 7:
            print(f"  {CYAN}🔥 One full week logged! You're ready for Week 2 — data analysis.{RESET}\n")
        elif total == 30:
            print(f"  {CYAN}🏆 30 days! You now have enough data to train your ML model.{RESET}\n")
    else:
        print(f"\n{RED}  ❌  Something went wrong. Your entry was not saved. Try again.{RESET}\n")


if __name__ == "__main__":
    main()
