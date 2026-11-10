# Synthetic training data

Invented Discord-style messages for the AstraThought task classifier.
Every line was written from imagination. No real team messages, no real
names, no @mentions. No model was trained to make these files.

## Files

- `batch01.jsonl` … `batch10.jsonl` — 100 rows each, 1000 rows total.
  Each row: `{"text": "...", "label": 1 or 0, "reason": "...",
  "source": "synthetic"}`.
- `synthetic_all.jsonl` — all 1000 synthetic rows, batches in order.

Per batch of 100: label 1 = 52 (task_requirement 20, task_feature 9,
task_imperative 8, task_bug 9, task_asset 6), label 0 = 48 (suggestion 9,
question 7, observation 7, own_plan 4, changelog 4, chat 7, request 4,
too_vague 3, meta 3). Totals across all batches: 520 tasks, 480 not-tasks.
Median length 7 words. Each batch has 12 matched pairs: an impersonal
requirement statement (label 1) next to the same idea as a hedged
suggestion or question (label 0).

Topics: 01 ship modules and combat, 02 star map and travel, 03 menus and
lobby, 04 art, icons and animation, 05 sound and music, 06 crew and NPCs,
07 encounters, events and hazards, 08 balance and numbers, 09 bugs,
performance and networking, 10 tools, debug UI and dev workflow.

## Combined file (repo root)

`train_plus_synthetic.jsonl` — 2911 rows: the current `train.jsonl` from
`main` (1911 rows, after the one-word rows were removed) followed by all
1000 synthetic rows. `train.jsonl` itself is untouched.

## Comparing accuracy

Train three models the same way and score each on `test.jsonl` and on
`eval_feedback.jsonl` (never train on those):

1. `train.jsonl` only — 1911 rows
2. `synthetic/synthetic_all.jsonl` only — 1000 rows
3. `train_plus_synthetic.jsonl` — 2911 rows

Note: `check_synthetic.py` drops near-duplicates, and the matched pairs
above are near-duplicates by design (a requirement statement and its
question version). Its kept/dropped counts will differ from the counts
here; that is expected, not an error in the batches.
