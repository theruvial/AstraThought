"""Score the model on eval_feedback.jsonl, the real #astra-feedback test set.

Run from the AstraThought checkout, inside its virtual environment.
This script does NOT train, and it writes no files. It only prints.

    .\\.venv\\Scripts\\Activate.ps1
    python eval_feedback.py

eval_feedback.jsonl holds every piece posted in #astra-feedback up to
the day it was built, each labeled by Thea. NEVER add these rows to
train.jsonl, and never tune thresholds against them more than once,
or the set stops measuring anything. Grow it instead: after each batch
of new human rulings, append the new pieces with their labels.

What it prints:
    1. precision and recall at the 0.5 line
    2. every piece the model gets wrong, with its score
    3. what the whole pipeline (classifier_service.decide_piece, with the
       rules in the checkout you are running) would do with each piece:
       how many are auto-posted, sent to review, or auto-dropped, and
       how many of the auto-posted are wrong and of the auto-dropped
       are real tasks. The silent losses matter most.
    4. how many eval pieces also appear in train.jsonl or test.jsonl
       (those are leaks: they make the numbers look better than they are)

With only a few dozen rows, one piece moves the numbers a lot. Read the
counts, not just the percentages.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import classifier_service as service  # noqa: E402

MODEL_DIR = HERE / "tasks-AstraThought-v2"
EVAL_FILE = HERE / "eval_feedback.jsonl"


def load(path: Path) -> list:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def main() -> None:
    from setfit import SetFitModel

    rows = load(EVAL_FILE)
    model = SetFitModel.from_pretrained(str(MODEL_DIR))
    scores = [float(p[1]) for p in model.predict_proba([r["text"] for r in rows])]
    tasks = sum(r["label"] for r in rows)
    print(f"{len(rows)} pieces: {tasks} tasks, {len(rows) - tasks} not tasks")

    tp = sum(1 for r, s in zip(rows, scores) if r["label"] == 1 and s >= 0.5)
    fp = sum(1 for r, s in zip(rows, scores) if r["label"] == 0 and s >= 0.5)
    fn = sum(1 for r, s in zip(rows, scores) if r["label"] == 1 and s < 0.5)
    print(f"\nAt the 0.5 line: precision {tp}/{tp + fp} = {100 * tp / max(1, tp + fp):.0f}%, "
          f"recall {tp}/{tp + fn} = {100 * tp / max(1, tp + fn):.0f}%")
    for r, s in zip(rows, scores):
        if (r["label"] == 1 and s < 0.5) or (r["label"] == 0 and s >= 0.5):
            kind = "MISSED TASK " if r["label"] == 1 else "FALSE ALARM "
            print(f"   {kind} score {s:.2f}  {r['text'][:90]}")

    table: dict = {}
    for r, s in zip(rows, scores):
        decision = service.decide_piece(r["text"], s, r["text"])["decision"]
        table.setdefault((decision, r["label"]), 0)
        table[(decision, r["label"])] += 1
    print("\nWhat the pipeline would do:")
    for decision in ("auto_task", "review", "auto_drop"):
        t, n = table.get((decision, 1), 0), table.get((decision, 0), 0)
        print(f"   {decision:<10} {t + n:3d} pieces: {t} real tasks, {n} not tasks")
    posted = table.get(("auto_task", 1), 0) + table.get(("auto_task", 0), 0)
    reviewed = table.get(("review", 1), 0) + table.get(("review", 0), 0)
    print(f"   auto-posted wrong: {table.get(('auto_task', 0), 0)} of {posted}")
    print(f"   real tasks lost silently: {table.get(('auto_drop', 1), 0)} of {tasks}")
    print(f"   sent to review: {reviewed} of {len(rows)} ({100 * reviewed / len(rows):.0f}%)")

    seen = set()
    for name in ("train.jsonl", "test.jsonl"):
        seen.update(norm(r["text"]) for r in load(HERE / name))
    leaks = [r for r in rows if norm(r["text"]) in seen]
    print(f"\nEval pieces that also appear in train or test data: {len(leaks)}")
    for r in leaks:
        print(f"   {r['text'][:90]}")


if __name__ == "__main__":
    main()
