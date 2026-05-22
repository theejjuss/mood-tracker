"""
input.py
--------
Interactive CLI that collects one day's mental health data from the user.
Each question is prompted with a description so the user understands what
they're entering. All values are validated before being accepted.
"""

from datetime import date
from validate import (
    validate_scale,
    validate_float,
    validate_choice,
    validate_minutes,
)


# ── Colour helpers (no external library needed) ───────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


def _header():
    print(f"\n{CYAN}{BOLD}{'─' * 50}{RESET}")
    print(f"{CYAN}{BOLD}   🧠  Mood Tracker — Daily Check-in{RESET}")
    print(f"{CYAN}{BOLD}{'─' * 50}{RESET}")
    print(f"{DIM}   Date: {date.today()}   |   Answer honestly — only you see this.{RESET}\n")


def _ask(prompt, hint, validator, *args):
    """
    Repeatedly shows `prompt` until the user gives a valid answer.
    `validator` is one of the functions from validate.py.
    `*args` are forwarded to the validator after the raw input string.
    Returns the validated value.
    """
    print(f"  {BOLD}{prompt}{RESET}")
    if hint:
        print(f"  {DIM}{hint}{RESET}")

    while True:
        raw = input(f"  {CYAN}→ {RESET}").strip()

        # Allow blank to quit mid-entry.
        if raw.lower() in ("q", "quit", "exit"):
            print(f"\n  {YELLOW}Entry cancelled. Nothing was saved.{RESET}\n")
            raise KeyboardInterrupt

        value, error = validator(raw, *args)
        if error:
            print(f"{RED}{error}{RESET}")
        else:
            print(f"  {GREEN}✓ Saved{RESET}\n")
            return value


def collect_entry() -> dict:
    """
    Runs the full check-in questionnaire and returns a dict with today's data.
    Raises KeyboardInterrupt if the user types 'q' to quit early.
    """
    _header()
    print(f"  {DIM}Tip: type 'q' at any question to cancel.\n{RESET}")

    # ── 1. Mood ───────────────────────────────────────────────────────────────
    mood = _ask(
        "1. How is your overall mood today?  (1–10)",
        "1 = very low / sad   •   10 = excellent / happy",
        validate_scale,
        "Mood score",
    )

    # ── 2. Sleep ──────────────────────────────────────────────────────────────
    sleep = _ask(
        "2. How many hours did you sleep last night?  (0–24)",
        "You can use decimals — e.g. 7.5 for 7 hours 30 minutes",
        validate_float,
        "Sleep hours",
        0.0,
        24.0,
    )

    # ── 3. Exercise ───────────────────────────────────────────────────────────
    exercise = _ask(
        "3. How many minutes did you exercise today?  (0–300)",
        "Any movement counts — walking, yoga, gym, etc.  Enter 0 if none.",
        validate_minutes,
    )

    # ── 4. Stress ─────────────────────────────────────────────────────────────
    stress = _ask(
        "4. What is your stress level today?  (1–10)",
        "1 = completely calm   •   10 = extremely stressed",
        validate_scale,
        "Stress level",
    )

    # ── 5. Social interaction ─────────────────────────────────────────────────
    social = _ask(
        "5. How much did you interact with others today?",
        "Options: low  /  medium  /  high",
        validate_choice,
        "Social interaction",
        ["low", "medium", "high"],
    )

    # ── 6. Appetite ───────────────────────────────────────────────────────────
    appetite = _ask(
        "6. Has your appetite changed compared to normal?",
        "Options: normal  /  reduced  /  increased",
        validate_choice,
        "Appetite change",
        ["normal", "reduced", "increased"],
    )

    # ── 7. Energy ─────────────────────────────────────────────────────────────
    energy = _ask(
        "7. How is your energy level today?  (1–10)",
        "1 = completely drained   •   10 = full of energy",
        validate_scale,
        "Energy level",
    )

    # ── 8. Notes (optional) ───────────────────────────────────────────────────
    print(f"  {BOLD}8. Any notes about today?  (optional — press Enter to skip){RESET}")
    print(f"  {DIM}e.g. 'had a tough meeting', 'felt anxious in the evening'{RESET}")
    notes = input(f"  {CYAN}→ {RESET}").strip()
    if notes:
        print(f"  {GREEN}✓ Saved{RESET}\n")
    else:
        print(f"  {DIM}  (skipped){RESET}\n")

    return {
        "date":               str(date.today()),
        "mood_score":         mood,
        "sleep_hours":        sleep,
        "exercise_minutes":   exercise,
        "stress_level":       stress,
        "social_interaction": social,
        "appetite_change":    appetite,
        "energy_level":       energy,
        "notes":              notes,
    }
