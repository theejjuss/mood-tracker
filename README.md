# 🧠 Mental Health Mood Tracker

A personal AI/ML project that tracks daily mental wellness data and predicts depression risk.
Built in Python — beginner friendly, privacy-first (all data stays on your device).

---

## Project Structure

```
mood-tracker/
├── main.py               ← Run this every day
├── requirements.txt
├── data/
│   └── user_logs.csv     ← Your daily entries (auto-created)
├── models/
│   └── model.pkl         ← Trained ML model (Week 3)
└── src/
    ├── input.py          ← CLI questionnaire
    ├── validate.py       ← Input validation
    ├── storage.py        ← CSV read/write
    ├── preprocess.py     ← Data cleaning (Week 2)
    ├── train.py          ← Model training (Week 3)
    ├── predict.py        ← Risk prediction (Week 4)
    └── visualize.py      ← Charts & reports (Week 4)
```

---

## Setup

### Step 1 — Make sure Python is installed
```bash
python --version   # Should be 3.10 or higher
```

### Step 2 — Create a virtual environment (recommended)
```bash
# Create
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install libraries (Week 1 needs none)
```bash
pip install -r requirements.txt
```

---

## Usage

### Log today's mood (run this every day!)
```bash
python main.py
```

### View your recent entries
```bash
python main.py --view
```

---

## What each question measures

| Question | Why it matters |
|---|---|
| Mood score (1–10) | Core indicator of emotional state |
| Sleep hours | Poor sleep is strongly linked to depression |
| Exercise minutes | Physical activity is a natural mood booster |
| Stress level (1–10) | Chronic stress is a major risk factor |
| Social interaction | Isolation is a key depression warning sign |
| Appetite change | Eating disruption signals emotional distress |
| Energy level (1–10) | Low energy / fatigue is a primary symptom |
| Notes | Free text for context the numbers miss |

---

## Build roadmap

| Week | Goal |
|---|---|
| ✅ Week 1 | Data collection CLI + CSV storage |
| Week 2 | Data exploration + preprocessing |
| Week 3 | Train Random Forest ML model |
| Week 4 | Prediction engine + visualizations |
| Bonus | Streamlit web dashboard |

---

## Privacy note

All your data is stored **locally** in `data/user_logs.csv`.
Nothing is sent to the internet. Only you can see it.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
It is NOT a clinical tool and cannot diagnose any mental health condition.
If you are struggling, please reach out to a qualified healthcare professional.
iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345
```
