"""Asks the trained model about one message.

    python check_message.py "add npcs to the crew UI"
"""

import sys
from pathlib import Path

from setfit import SetFitModel

MODEL_DIR = Path(__file__).resolve().parent / "tasks-AstraThought-v2"


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python check_message.py "message text"')
    text = " ".join(sys.argv[1:])
    model = SetFitModel.from_pretrained(str(MODEL_DIR))
    chance = float(model.predict_proba([text])[0][1])
    print(f"{'TASK' if chance >= 0.5 else 'NOT A TASK'} ({chance:.0%} sure it is a task)")


if __name__ == "__main__":
    main()
