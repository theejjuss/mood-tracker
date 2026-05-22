"""
evaluate.py  —  Model Evaluation
----------------------------------
Loads the saved model and test split, then prints:
  • Overall accuracy
  • Per-class precision, recall, F1-score
  • Confusion matrix (visual)
  • Cross-validation score (more reliable than single split)

Run AFTER train.py:
    python src\evaluate.py
"""

import os, sys, joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.pkl")
TEST_PATH  = os.path.join(BASE_DIR, "models", "test_split.pkl")
KAGGLE_CSV = os.path.join(BASE_DIR, "data", "depression_anxiety_data.csv")

RESET="\033[0m"; BOLD="\033[1m"; CYAN="\033[96m"
GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; DIM="\033[2m"

RISK_NAMES  = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {0: GREEN, 1: YELLOW, 2: RED}


def _section(title):
    print(f"\n{CYAN}{BOLD}{'─'*52}{RESET}")
    print(f"{CYAN}{BOLD}   {title}{RESET}")
    print(f"{CYAN}{'─'*52}{RESET}")


def load_artifacts():
    """Loads model, metadata and test split saved by train.py."""
    for path, name in [(MODEL_PATH, "model.pkl"), (META_PATH, "model_meta.pkl"), (TEST_PATH, "test_split.pkl")]:
        if not os.path.isfile(path):
            print(f"{RED}  ✗ {name} not found. Run train.py first.{RESET}\n")
            sys.exit(1)

    model = joblib.load(MODEL_PATH)
    meta  = joblib.load(META_PATH)
    split = joblib.load(TEST_PATH)

    X_test = np.array(split["X_test"])
    y_test = np.array(split["y_test"])

    return model, meta, X_test, y_test


def show_accuracy(y_test, y_pred):
    _section("🎯  Overall Accuracy")

    acc = accuracy_score(y_test, y_pred)
    bar = "█" * int(acc * 40)
    color = GREEN if acc >= 0.75 else (YELLOW if acc >= 0.60 else RED)

    print(f"\n  Accuracy  :  {color}{BOLD}{acc*100:.1f}%{RESET}  {color}{bar}{RESET}")

    if acc >= 0.80:
        grade = f"{GREEN}Excellent{RESET}"
    elif acc >= 0.70:
        grade = f"{GREEN}Good{RESET}"
    elif acc >= 0.60:
        grade = f"{YELLOW}Acceptable{RESET}"
    else:
        grade = f"{RED}Needs improvement — try logging more data{RESET}"

    print(f"  Grade     :  {grade}")
    print(f"\n  {DIM}(Accuracy = how often the model predicts the correct risk level){RESET}")


def show_classification_report(y_test, y_pred):
    _section("📋  Per-Class Report  (Precision · Recall · F1)")

    print(f"\n  {DIM}Precision = when model says 'High', how often is it right?")
    print(f"  Recall    = of all actual 'High' cases, how many did it catch?")
    print(f"  F1-score  = balance of precision and recall (higher = better){RESET}\n")

    report = classification_report(
        y_test, y_pred,
        target_names=["Low", "Medium", "High"],
        output_dict=True,
    )

    print(f"  {'CLASS':<10} {'PRECISION':>10} {'RECALL':>8} {'F1-SCORE':>10} {'SUPPORT':>9}")
    print(f"  {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*9}")

    for cls in ["Low", "Medium", "High"]:
        r = report[cls]
        color = GREEN if r["f1-score"] >= 0.70 else (YELLOW if r["f1-score"] >= 0.50 else RED)
        print(
            f"  {cls:<10} {r['precision']:>10.2f} {r['recall']:>8.2f} "
            f"  {color}{r['f1-score']:>8.2f}{RESET} {int(r['support']):>9}"
        )

    print(f"\n  {'─'*49}")
    wa = report["weighted avg"]
    print(
        f"  {'Weighted avg':<10} {wa['precision']:>10.2f} {wa['recall']:>8.2f} "
        f"  {BOLD}{wa['f1-score']:>8.2f}{RESET} {int(wa['support']):>9}"
    )


def show_confusion_matrix(y_test, y_pred):
    _section("🗺️   Confusion Matrix")

    cm    = confusion_matrix(y_test, y_pred)
    names = ["Low", "Medium", "High"]

    print(f"\n  {DIM}Rows = Actual label  |  Columns = Predicted label")
    print(f"  Diagonal (top-left to bottom-right) = correct predictions{RESET}\n")

    # Header row
    print(f"  {'':12}", end="")
    for n in names:
        print(f"  {'Pred '+n:^12}", end="")
    print()
    print(f"  {'─'*52}")

    for i, row_name in enumerate(names):
        print(f"  {'Act '+row_name:<12}", end="")
        for j, val in enumerate(cm[i]):
            if i == j:   # correct prediction — green
                cell = f"{GREEN}{BOLD}{val:^12}{RESET}"
            elif val > 0:
                cell = f"{RED}{val:^12}{RESET}"
            else:
                cell = f"{'0':^12}"
            print(f"  {cell}", end="")
        print()

    # Interpretation
    total   = cm.sum()
    correct = cm.diagonal().sum()
    print(f"\n  {DIM}Correctly classified: {correct}/{total} samples{RESET}")


def show_cross_validation(model, X_test, y_test):
    _section("🔄  Cross-Validation  (5-fold)")

    print(f"\n  {DIM}Cross-validation splits data 5 ways and tests each split.")
    print(f"  More reliable than a single train/test split.{RESET}\n")

    scores = cross_val_score(model, X_test, y_test, cv=min(5, len(y_test)), scoring="accuracy")

    for i, s in enumerate(scores, 1):
        bar   = "▓" * int(s * 30)
        color = GREEN if s >= 0.70 else (YELLOW if s >= 0.55 else RED)
        print(f"  Fold {i}:  {color}{s*100:.1f}%  {bar}{RESET}")

    print(f"\n  Mean accuracy : {BOLD}{scores.mean()*100:.1f}%{RESET}  (± {scores.std()*100:.1f}%)")


def show_what_it_means():
    _section("💡  What These Numbers Mean For You")
    print(f"""
  The model was trained on {BOLD}clinical depression screening data{RESET}.
  It learned patterns like:

    {GREEN}•{RESET} Low PHQ-9 score + good sleep + low anxiety  →  Low risk
    {YELLOW}•{RESET} Moderate PHQ + poor sleep + high anxiety    →  Medium risk
    {RED}•{RESET} High PHQ-9 + depressiveness + anxiousness    →  High risk

  In Week 4, this trained model will take YOUR daily mood logs
  and predict your personal risk level in real time.

  {DIM}Remember: this is an educational ML project, not a clinical tool.
  Always speak to a professional for mental health support.{RESET}
""")


def run():
    model, meta, X_test, y_test = load_artifacts()
    y_pred = model.predict(X_test)

    show_accuracy(y_test, y_pred)
    show_classification_report(y_test, y_pred)
    show_confusion_matrix(y_test, y_pred)
    show_cross_validation(model, X_test, y_test)
    show_what_it_means()

    print(f"{CYAN}{BOLD}{'─'*52}{RESET}")
    print(f"{GREEN}{BOLD}  ✅  Evaluation complete!{RESET}")
    print(f"  Run {CYAN}python src\\predict.py{RESET} to predict YOUR personal risk.\n")


if __name__ == "__main__":
    run()
