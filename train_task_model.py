"""Trains the Astra task classifier, then scores it on test.jsonl.

Run from this folder, inside the virtual environment:
    python train_task_model.py
See README.md for setup.
"""

import json
from pathlib import Path

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

HERE = Path(__file__).resolve().parent
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = HERE / "tasks-AstraThought-v2"


def load(name):
    with open(HERE / name, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main():
    train_rows, test_rows = load("train.jsonl"), load("test.jsonl")
    tasks = sum(row["label"] for row in train_rows)
    print(f"Training on {len(train_rows)} items: {tasks} tasks, {len(train_rows) - tasks} not tasks")

    train_set = Dataset.from_list([{"text": r["text"], "label": r["label"]} for r in train_rows])
    model = SetFitModel.from_pretrained(BASE_MODEL)
    print(f"Running on: {model.device}")
    arguments = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=20, seed=42)
    Trainer(model=model, args=arguments, train_dataset=train_set).train()
    model.save_pretrained(str(MODEL_DIR))
    print(f"Saved to {MODEL_DIR}")

    predictions = [int(p) for p in model.predict([r["text"] for r in test_rows])]
    found = missed = false_alarms = 0
    print("\nMistakes on the test set:")
    for row, predicted in zip(test_rows, predictions):
        if row["label"] == 1 and predicted == 1:
            found += 1
        elif row["label"] == 1:
            missed += 1
            print(f"  MISSED      {row['text'][:90]}")
        elif predicted == 1:
            false_alarms += 1
            print(f"  FALSE ALARM {row['text'][:90]} ({row['reason']})")
    precision = found / max(1, found + false_alarms)
    recall = found / max(1, found + missed)
    print(f"\nFound {found} | missed {missed} | false alarms {false_alarms}")
    print(f"Precision {precision:.0%} | recall {recall:.0%}")


if __name__ == "__main__":
    main()
