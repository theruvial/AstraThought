# AstraThought classifier service

Serves the local SetFit task classifier over HTTP on loopback only,
so the Astrabot tasks cog (which runs in a Python without PyTorch)
can ask it about Discord messages without loading the model itself.

Model: SetFit, base `sentence-transformers/all-MiniLM-L6-v2`,
current version **tasks-AstraThought-v2** (the v2 retrain, 2026-11-06:
precision 77%, recall 93% on test.jsonl after the asteroid-field
correction). Only the most current model is kept. There is no
model_v1 backup any more.

## Files in this folder (AstraThought repo)

- `classifier_service.py` - the service. Stdlib only, no new deps.
- `split_tasks.py` - the splitter, stdlib only. Same file the
  training prep uses. One copy, in this repo, not in Astrabot.
- `score_distribution.py` - print the model score distribution
  on test.jsonl, to set the HIGH / LOW band from data. No training,
  no file writes.
- `.gitignore` - updated copy for the repo: `tasks-AstraThought-v*/`
  instead of `model/` and `model_v*/`. The model folder itself is
  never committed.

## Model folder rename (do once, on the host)

The current model folder, the v2 retrain currently named `model`,
is renamed to `tasks-AstraThought-v2`, and the old `model_v1`
backup folder is deleted. Run in the AstraThought checkout:

```powershell
cd C:\codingShit\AstraThought
Rename-Item model tasks-AstraThought-v2
Remove-Item model_v1 -Recurse -Force
```

After the rename, the service loads `./tasks-AstraThought-v2`.
The `.gitignore` change keeps any future `tasks-AstraThought-v*`
folder out of git, the same way `model/` and `model_v*/` were
ignored before.

`train_task_model.py` and `check_message.py` in the repo already
point at `./tasks-AstraThought-v2` (main, commit `b8b3221c`), so a
fresh retrain creates the folder the service loads. No rename
needed after retraining.

## How to run it

Run from the AstraThought checkout, inside its virtual environment.
Python 3.14 venv, torch 2.14.0+cu126. Never upgrade torch past
2.14: 2.15+ dropped GTX 1070 support.

```powershell
cd C:\codingShit\AstraThought
.\.venv\Scripts\Activate.ps1
python classifier_service.py
```

That binds `127.0.0.1:13383` and loads
`./tasks-AstraThought-v2` once at startup. Run it as another
pane in Astrabot Console, next to the bot, so it stays up while
the bot is up. If the model folder is missing, the service still
starts, `/health` reports `model_loaded` false, and `/classify`
returns 503.

Environment variables (all optional):

- `CLASSIFIER_HOST` - default `127.0.0.1`. Do not change this.
  Loopback only, no auth, by design.
- `CLASSIFIER_PORT` - default `13383`.
- `ASTRATHOUGHT_MODEL_DIR` - default `./tasks-AstraThought-v2`.
- `TASK_OVERRIDES_PATH` - default `data/task_overrides.json`
  next to the service. The cog writes this file when a human
  rules on a review item or a confirmed task.

## Endpoints

```powershell
curl http://127.0.0.1:13383/health
```

```powershell
curl -X POST http://127.0.0.1:13383/classify -H "Content-Type: application/json" -d '{"text": "salvage is broken"}'
```

```powershell
curl -X POST http://127.0.0.1:13383/classify_batch -H "Content-Type: application/json" -d '{"texts": ["salvage is broken", "asteroid field is not an asteroid field"]}'
```

Response per split piece:

```json
{"text": "salvage is broken", "score": 0.12, "decision": "auto_drop", "trigger": "none"}
```

Decisions: `auto_task` (score at or above HIGH), `auto_drop`
(score at or below LOW), `review` (in between, or a review
trigger fires). HIGH 0.75 and LOW 0.25 are placeholders. Set
them from `score_distribution.py` output before going live.

## What you see in the console

Every message the bot sends for classification is logged in the
console pane, so you can watch the classifier think. One block
per message: what came in, how it split, and for each piece its
score, decision, trigger, and a plain-words reason. The same
reason is also in the `reason` field of each piece in the
`/classify` response. Nothing is logged for `/health` checks.

```text
classify: 'Fix the lighting layers. Fix the flocking in the north sector' -> 2 piece(s)
  [auto_task 0.90 none] Fix the lighting layers -- score 0.90 is at or above 0.75
  [review 0.50 split_disagreement] Fix the flocking in the north sector -- other pieces in this message were auto-decided, this one is in the band (0.25-0.75)
classify: 'salvage is broken' -> 1 piece(s)
  [review 0.40 short_observation] salvage is broken -- 3 words, no imperative verb, same shape as ruled both ways
classify: 'Fix the guns, we should make them stronger' -> 1 piece(s)
  [review 0.80 suggestion_marker] Fix the guns, we should make them stronger -- model says task but the text says 'we should'
```

Review triggers: `score_band`, `model_regex_disagreement`,
`suggestion_marker`, `short_observation`, `split_disagreement`,
and `human_override` (a hit in the overrides file, which skips
the model and returns score 1.0 for label 1 or 0.0 for label 0).

## Setting the band

Run once, in the AstraThought venv, before going live:

```powershell
cd C:\codingShit\AstraThought
.\.venv\Scripts\Activate.ps1
python score_distribution.py
```

It prints every test row sorted by score, with found / missed /
false alarm at the 0.5 threshold, plus a band table showing how
many task and non-task rows fall below 0.25, between 0.25 and
0.75, and above 0.75. Pick HIGH and LOW from that table once,
set them in `classifier_service.py`, and leave them alone. Do
not tune them against test.jsonl after that.

## How it fits with the bot

Astrabot `cogs/tasks.py` watches mcdungeons #general, sends each
message to this service, and routes the pieces by the decision
the service returns: `auto_task` pieces become wiki tasks and
embeds in #astra-feedback, `review` pieces become "Needs review"
embeds in #astra-feedback with a thread, `auto_drop` pieces are
dropped silently. If this service is down, the cog logs it and
does nothing. It never crashes the bot and never retries in a loop.

## Retraining

This service does not train. Retraining is a separate batch job:

```powershell
cd C:\codingShit\AstraThought
git pull
.\.venv\Scripts\Activate.ps1
python train_task_model.py
```

It retrains from the base model, about 5 minutes, and is judged
on the unchanged `test.jsonl`. Retrain only when 50 new labels
have been collected, no matter how long that takes. No
per-ruling micro-training: one label is too small to train on,
a full retrain takes about 5 minutes, and per-ruling training
has no test gate. Labels are collected by the cog in
`data/task_labels.jsonl` and merged into `train.jsonl` by Thea
in a batch, the same way her rulings were applied before. See
the note above about updating `train_task_model.py` to
`tasks-AstraThought-v2` before the next retrain.
