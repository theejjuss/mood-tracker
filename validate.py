"""
validate.py
-----------
All input validation for the mood tracker.
Keeps input.py clean by separating validation logic.
"""


def validate_scale(value_str, field_name):
    """
    Validates that the input is an integer between 1 and 10.
    Returns (int, None) on success or (None, error_message) on failure.
    """
    try:
        value = int(value_str.strip())
    except ValueError:
        return None, f"  ✗ '{value_str}' is not a valid number. Please enter a whole number."

    if not 1 <= value <= 10:
        return None, f"  ✗ {field_name} must be between 1 and 10. You entered {value}."

    return value, None


def validate_float(value_str, field_name, min_val, max_val):
    """
    Validates a decimal number within a min/max range.
    Used for sleep hours (0.0 – 24.0).
    """
    try:
        value = float(value_str.strip())
    except ValueError:
        return None, f"  ✗ '{value_str}' is not a valid number. Try something like 7.5"

    if not min_val <= value <= max_val:
        return None, f"  ✗ {field_name} must be between {min_val} and {max_val}. You entered {value}."

    return value, None


def validate_choice(value_str, field_name, choices):
    """
    Validates that the input matches one of the allowed choices.
    Case-insensitive comparison.
    Returns the matched choice in lowercase or an error message.
    """
    value = value_str.strip().lower()
    choices_lower = [c.lower() for c in choices]

    if value not in choices_lower:
        options = " / ".join(choices)
        return None, f"  ✗ Invalid option. Please choose from: {options}"

    return value, None


def validate_minutes(value_str):
    """
    Validates exercise minutes (0 – 300).
    Returns (int, None) or (None, error_message).
    """
    try:
        value = int(value_str.strip())
    except ValueError:
        return None, f"  ✗ '{value_str}' is not valid. Enter a whole number like 30."

    if not 0 <= value <= 300:
        return None, "  ✗ Minutes must be between 0 and 300."

    return value, None
