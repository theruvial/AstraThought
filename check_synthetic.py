"""Clean up synthetic training data written by ChatGPT, and spot-check it.

Run from the AstraThought checkout. It needs no model and no GPU. It
trains nothing, and it never touches train.jsonl, test.jsonl or
eval_feedback.jsonl. It only reads them and writes the files named below.

Step 1, clean the batches (see synthetic_data_prompt.md):

    python check_synthetic.py batch01.md batch02.md batch03.md

Reads each file, ignores code fences, headings and anything that is not
a JSON line, then:
    - rejects lines with a bad label, a reason that does not fit the
      label, fewer than 2 or more than 30 words, or an @mention
    - drops repeats and near-repeats of earlier lines (a sentence with one
      word swapped, added or dropped counts as a repeat)
    - drops lines that match train.jsonl, test.jsonl or eval_feedback.jsonl
      exactly or nearly, so nothing leaks into the test sets
    - prints counts per kind next to the quotas the prompt asked for, and
      how concentrated the first words are (a sign of lazy generation)
Writes synthetic_clean.jsonl (rows marked "source": "synthetic",
"reviewed": false) and a numbered spot-check sample of 100 lines with the
labels hidden, synthetic_review_sample.txt, plus synthetic_review_key.json.

Step 2, check ChatGPT's labels. Read the sample, and list the numbers
that are NOT tasks:

    python check_synthetic.py --score synthetic_review_key.json --not-tasks 3,7,12

It prints how often you agree with ChatGPT, overall and per label, and
lists every disagreement. If agreement is low, fix the prompt and
regenerate instead of training on the data. ChatGPT's labels are
guesses, and a model trained on guesses learns the guesser's mistakes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

TASK_KINDS = {"task_requirement", "task_feature", "task_imperative", "task_bug", "task_asset"}
NOT_TASK_KINDS = {"chat", "question", "suggestion", "own_plan", "request",
                  "changelog", "too_vague", "meta", "observation"}
# What synthetic_data_prompt.md asks for across all ten batches.
QUOTAS = {
    "task_requirement": 200, "task_feature": 90, "task_imperative": 80,
    "task_bug": 90, "task_asset": 60,
    "suggestion": 90, "question": 70, "observation": 70, "own_plan": 40,
    "changelog": 40, "chat": 70, "request": 40, "too_vague": 30, "meta": 30,
}
MIN_WORDS, MAX_WORDS = 2, 30
SIMILAR = 0.7   # shared words / all words in either line
OVERLAP = 0.8   # shared words / words in the shorter line (one line inside the other)
REFERENCE_FILES = ("train.jsonl", "test.jsonl", "eval_feedback.jsonl")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def words(text: str) -> frozenset:
    return frozenset(normalize(text).split())


def similar(a: frozenset, b: frozenset) -> bool:
    """True for near repeats: one word swapped, added or dropped.

    Short lines (under 4 words) only count as repeats when identical, so
    "not sure" and "not sure about that" are not treated as one line.
    """
    if not a or not b:
        return False
    shared = len(a & b)
    if shared / len(a | b) >= SIMILAR:
        return True
    smaller = min(len(a), len(b))
    return smaller >= 4 and shared / smaller >= OVERLAP


def parse_lines(text: str):
    """(rows, rejects, ignored): JSON lines from a batch file."""
    rows, rejects, ignored = [], [], 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            ignored += 1
            continue
        try:
            data = json.loads(line)
        except ValueError:
            rejects.append((line, "not valid JSON"))
            continue
        problem = check_row(data)
        if problem:
            rejects.append((line, problem))
            continue
        label = int(data["label"])
        rows.append({
            "text": re.sub(r"\s+", " ", data["text"]).strip(),
            "label": label,
            "reason": "task" if label == 1 else data["reason"],
            "kind": data["reason"],
            "source": "synthetic",
            "reviewed": False,
        })
    return rows, rejects, ignored


def check_row(data) -> str | None:
    if not isinstance(data, dict):
        return "not an object"
    text, label, reason = data.get("text"), data.get("label"), data.get("reason")
    if not isinstance(text, str) or not text.strip():
        return "no text"
    if label not in (0, 1) or isinstance(label, bool):
        return "label is not 0 or 1"
    if reason not in TASK_KINDS | NOT_TASK_KINDS:
        return f"unknown reason {reason!r}"
    if (label == 1) != (reason in TASK_KINDS):
        return f"reason {reason!r} does not fit label {label}"
    count = len(text.split())
    if count < MIN_WORDS or count > MAX_WORDS:
        return f"{count} words (allowed {MIN_WORDS} to {MAX_WORDS})"
    if "@" in text:
        return "contains an @"
    return None


def drop_repeats(rows: list):
    """Drop exact and near repeats of an earlier row. Returns (kept, dropped)."""
    kept, dropped, seen = [], 0, []
    for row in rows:
        w = words(row["text"])
        if any(w == s or similar(w, s) for s in seen):
            dropped += 1
            continue
        seen.append(w)
        kept.append(row)
    return kept, dropped


def load_reference(root: Path = HERE) -> dict:
    """{file name: [word sets]} for the data synthetic rows must not echo."""
    out = {}
    for name in REFERENCE_FILES:
        path = root / name
        out[name] = []
        if path.is_file():
            with open(path, encoding="utf-8") as handle:
                out[name] = [words(json.loads(l)["text"]) for l in handle if l.strip()]
    return out


def drop_leaks(rows: list, reference: dict):
    """Drop rows that match reference data. Returns (kept, {file: count})."""
    kept, hits = [], Counter()
    for row in rows:
        w = words(row["text"])
        match = next((name for name, sets in reference.items()
                      if any(w == s or similar(w, s) for s in sets)), None)
        if match:
            hits[match] += 1
        else:
            kept.append(row)
    return kept, hits


def build_sample(rows: list, size: int = 100, seed: int = 11):
    chosen = random.Random(seed).sample(rows, min(size, len(rows)))
    key = {str(i): {"text": r["text"], "label": r["label"], "kind": r["kind"]}
           for i, r in enumerate(chosen, 1)}
    return key


def agreement(key: dict, not_tasks: set):
    """Compare Thea's calls with ChatGPT's labels. Returns a stats dict."""
    agree, per, disagreements = 0, Counter(), []
    total = Counter()
    for n, item in key.items():
        mine = 0 if int(n) in not_tasks else 1
        total[item["label"]] += 1
        if mine == item["label"]:
            agree += 1
            per[item["label"]] += 1
        else:
            disagreements.append((int(n), item, mine))
    return {"agree": agree, "total": len(key), "per": per, "per_total": total,
            "disagreements": sorted(disagreements, key=lambda d: d[0])}


def clean(paths: list, out_dir: Path, root: Path = HERE) -> int:
    rows, rejects, ignored = [], [], 0
    for path in paths:
        r, rej, ign = parse_lines(Path(path).read_text(encoding="utf-8"))
        rows += r
        rejects += rej
        ignored += ign
        print(f"{Path(path).name}: {len(r)} usable lines, {len(rej)} rejected")
    rows, repeats = drop_repeats(rows)
    rows, leaks = drop_leaks(rows, load_reference(root))
    print(f"\nignored non-JSON lines: {ignored} | rejected: {len(rejects)} | "
          f"repeats dropped: {repeats} | matched existing data and dropped: {sum(leaks.values())}")
    for name, count in leaks.items():
        note = "  <- check the prompt did not leak real messages" if name != "train.jsonl" else ""
        print(f"   matched {name}: {count}{note}")
    for line, why in rejects[:8]:
        print(f"   rejected ({why}): {line[:90]}")
    if not rows:
        print("Nothing usable.")
        return 1
    tasks = sum(r["label"] for r in rows)
    print(f"\nkept {len(rows)} rows: {tasks} tasks, {len(rows) - tasks} not tasks")
    print("kind                  got   asked for (all 10 batches)")
    counts = Counter(r["kind"] for r in rows)
    for kind in list(TASK_KINDS) + list(NOT_TASK_KINDS):
        print(f"   {kind:<18} {counts.get(kind, 0):4d}   {QUOTAS.get(kind, 0)}")
    firsts = Counter(r["text"].split()[0].lower() for r in rows)
    top = ", ".join(f"{w} {100 * n / len(rows):.0f}%" for w, n in firsts.most_common(6))
    print(f"most common first words: {top}")
    lengths = sorted(len(r["text"].split()) for r in rows)
    print(f"words per line: median {lengths[len(lengths) // 2]}, range {lengths[0]} to {lengths[-1]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "synthetic_clean.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    key = build_sample(rows)
    with open(out_dir / "synthetic_review_key.json", "w", encoding="utf-8") as handle:
        json.dump(key, handle, ensure_ascii=False, indent=1)
    with open(out_dir / "synthetic_review_sample.txt", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(f"{n}. {item['text']}" for n, item in key.items()) + "\n")
    print(f"\nwrote synthetic_clean.jsonl, synthetic_review_sample.txt ({len(key)} lines) "
          "and synthetic_review_key.json")
    print("Next: read the sample, then run --score with the numbers that are NOT tasks.")
    return 0


def score(key_path: Path, not_tasks_arg: str) -> int:
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    not_tasks = {int(x) for x in re.split(r"[,\s]+", not_tasks_arg.strip()) if x}
    unknown = [n for n in not_tasks if str(n) not in key]
    if unknown:
        print(f"Numbers not in the sample: {unknown}")
        return 1
    stats = agreement(key, not_tasks)
    pct = 100 * stats["agree"] / stats["total"]
    print(f"You agree with ChatGPT on {stats['agree']} of {stats['total']} ({pct:.0f}%)")
    for label, name in ((1, "task"), (0, "not task")):
        total = stats["per_total"][label]
        if total:
            print(f"   ChatGPT said {name}: {stats['per'][label]} of {total} agreed")
    for n, item, mine in stats["disagreements"]:
        print(f"   {n}. ChatGPT {'task' if item['label'] else 'not task'} ({item['kind']}), "
              f"you {'task' if mine else 'not task'}: {item['text']}")
    verdict = ("Good enough to use after you fix or drop these lines."
               if pct >= 90 else
               "Too low. Tighten the prompt and regenerate instead of training on it.")
    print(verdict)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="*", help="batch files from ChatGPT")
    parser.add_argument("--score", metavar="KEY", help="synthetic_review_key.json")
    parser.add_argument("--not-tasks", default="", help="comma-separated sample numbers that are not tasks")
    args = parser.parse_args(argv)
    if args.score:
        return score(Path(args.score), args.not_tasks)
    if not args.files:
        parser.error("give at least one batch file, or --score")
    return clean(args.files, Path.cwd())


if __name__ == "__main__":
    sys.exit(main())
