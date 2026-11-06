"""Print the model score distribution on test.jsonl.

Run from the AstraThought checkout, inside its virtual environment.
This script does NOT train, and it writes no files. It only prints,
so Thea can set the classifier service HIGH / LOW band from data
instead of guessing.

    .\\.venv\\Scripts\\Activate.ps1
    python score_distribution.py

For every row in test.jsonl it prints the predict_proba score,
sorted lowest to highest, with the true label and whether the
0.5 threshold would have found it, missed it, or false alarmed
on it. At the end it prints summary counts at the 0.5 threshold
and at the placeholder 0.25 / 0.75 band, plus a band table showing
how many task and non-task rows fall below 0.25, between 0.25
and 0.75, and above 0.75.

Use the band table to pick HIGH and LOW once, then leave them
alone. Do not tune them against test.jsonl after that, or the
test set stops being held out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "tasks-AstraThought-v2"
TEST_FILE = HERE / "test.jsonl"

HIGH = 0.75
LOW = 0.25


def load_rows():
    with open(TEST_FILE, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    from setfit import SetFitModel

    if not MODEL_DIR.is_dir():
        print(f"Model folder not found: {MODEL_DIR}")
        sys.exit(1)
    if not TEST_FILE.is_file():
        print(f"Test file not found: {TEST_FILE}")
        sys.exit(1)
    rows = load_rows()
    print(f"Scoring {len(rows)} test rows with {MODEL_DIR}")
    model = SetFitModel.from_pretrained(str(MODEL_DIR))
    texts = [row["text"] for row in rows]
    probs = model.predict_proba(texts)
    scored = []
    for row, prob in zip(rows, probs):
        score = float(prob[1])
        label = int(row["label"])
        predicted = 1 if score >= 0.5 else 0
        if label == 1 and predicted == 1:
            outcome = "found"
        elif label == 1:
            outcome = "MISSED"
        elif predicted == 1:
            outcome = "FALSE ALARM"
        else:
            outcome = "correct drop"
        scored.append((score, label, outcome, row["text"],
                       row.get("reason", "")))
    scored.sort(key=lambda item: item[0])

    print()
    print("Score  Label  Outcome       Text")
    for score, label, outcome, text, reason in scored:
        shown = text[:90]
        print(f"{score:.3f}  {label}      {outcome:<12} {shown}"
              + (f" ({reason})" if reason else ""))

    found = sum(1 for _, label, outcome, _, _ in scored
                if outcome == "found")
    missed = sum(1 for _, label, outcome, _, _ in scored
                 if outcome == "MISSED")
    false_alarms = sum(1 for _, label, outcome, _, _ in scored
                       if outcome == "FALSE ALARM")
    tasks = sum(1 for _, label, _, _, _ in scored if label == 1)
    print()
    print(f"At 0.5 threshold: found {found} | missed {missed} | "
          f"false alarms {false_alarms} | test tasks {tasks}")
    if found + false_alarms:
        print(f"Precision {found / (found + false_alarms):.1%}")
    if found + missed:
        print(f"Recall {found / (found + missed):.1%}")

    print()
    print("Band table (placeholder HIGH 0.75, LOW 0.25):")
    print("Band            Tasks  Non-tasks  Total")
    for name, low, high in (("below 0.25", None, LOW),
                            ("0.25 to 0.75", LOW, HIGH),
                            ("above 0.75", HIGH, None)):
        in_band = [s for s in scored
                   if (low is None or s[0] > low)
                   and (high is None or s[0] < high)]
        # Rows exactly on a boundary fall in the middle band in the
        # service (review is LOW < score < HIGH), so count them there.
        if name == "0.25 to 0.75":
            in_band = [s for s in scored
                       if LOW <= s[0] <= HIGH]
        n_tasks = sum(1 for s in in_band if s[1] == 1)
        n_not = len(in_band) - n_tasks
        print(f"{name:<15} {n_tasks:<6} {n_not:<10} {len(in_band)}")
    print()
    print("Pick HIGH and LOW from this table, then set them in "
          "classifier_service.py and cogs/tasks.py, and leave them alone.")


if __name__ == "__main__":
    main()
