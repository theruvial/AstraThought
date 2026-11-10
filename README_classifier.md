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

Astrabot Console starts it: the console runs
`C:\codingShit\AstraThought\.venv\Scripts\python.exe -u classifier_service.py`
from the AstraThought checkout, shows its output in a Classifier pane
at the bottom of the window, restarts it on crashes like the other
services, and stops it when the console closes. "Restart Classifier"
in the top bar restarts it on its own; "Pull + Restart" does not pull
AstraThought, so it does not restart the classifier either.

To run it by hand instead, from the AstraThought checkout, inside its
virtual environment. Python 3.14 venv, torch 2.14.0+cu126. Never upgrade
torch past 2.14: 2.15+ dropped GTX 1070 support.

```powershell
cd C:\codingShit\AstraThought
.\.venv\Scripts\Activate.ps1
python classifier_service.py
```

That binds `127.0.0.1:13383` and loads
`./tasks-AstraThought-v2` once at startup. If the model folder is missing, the service still
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
  #1 [auto_task 0.90 none] Fix the lighting layers -- says task
    reason: score 0.90 is at or above 0.75
  #2 [review 0.50 split_disagreement] Fix the flocking in the north sector -- leans task
    reason: other pieces in this message were auto-decided, this one is in the band (0.25-0.75)
classify: 'salvage is broken' -> 1 piece(s)
  #3 [review 0.40 short_observation] salvage is broken -- leans not a task
    reason: 3 words, no imperative verb, same shape as ruled both ways
classify: 'Fix the guns, we should make them stronger' -> 1 piece(s)
  #4 [review 0.80 suggestion_marker] Fix the guns, we should make them stronger -- leans task
    reason: model says task but the text says 'we should'
```

## Ruling on decisions from the Astrabot Console

Each logged piece has an id number. The console's decisions panel
shows the last 50, most recent first, with a search box to filter
them by text, and an approve button (check mark) and a disagree
button (X) on each row. Clicking one records a training label for
that piece and updates the overrides file, so the same text is
decided by your ruling from then on, without restarting the service
or retraining. Ruled rows stay in the list, marked with what was
recorded, and their buttons go quiet.

- Approve means the bot was right about that decision. Disagree
  means it was wrong.
- Approve/disagree answers "is this a task?". For a review decision
  it answers the lean shown after the `--`: score at or above 0.5
  leans task, below leans not a task.
- The panel fetches `GET /decisions` when the service logs a new
  decision, and when the service starts; it never polls. It posts
  rulings to `POST /decisions/<id>/rule`. If the service is not
  running, the panel says so instead of showing decisions.

Every ruling is appended to the labels file as a training label
(same JSONL shape as the cog's labels, reviewer `console`).

The labels and overrides files are shared with the cog: set
`TASKS_LABELS_PATH` and `TASK_OVERRIDES_PATH` once on the host
(`setx`, machine-wide) and both the service and the bot write the
same files, which is what `/task_stats` counts and what the
50-label retrain trigger reads. With neither set, the service
writes `data/task_labels.jsonl` and `data/task_overrides.json`
next to itself, in the AstraThought checkout, which the cog does
not read.

Review triggers: `score_band`, `model_regex_disagreement`,
`suggestion_marker`, `short_observation`, `split_disagreement`,
and `human_override` (a hit in the overrides file, which skips
the model and returns score 1.0 for label 1 or 0.0 for label 0).

### Confident model outvotes the rule

`model_regex_disagreement` normally sends a piece to review when the
model and the cheap rule disagree. The rule only counts `TODO:`,
`.open`, `.assign` and an imperative first word, so it misses
feedback phrased as an observation ("the jump gate is visible after
the warp flash"). When the model says task at 95% or higher
(`DISAGREEMENT_SKIP_SCORE`) and the rule says not, the rule is
outvoted and the piece goes on to the remaining checks. The reverse
case (the rule says task, the model says not) still goes to review.

Evidence: Thea's 37 rulings on these disagreements in
#astra-feedback were 36 tasks and 1 not a task. Replaying them
through the new rule, 26 become `auto_task` (all real tasks) and 11
still go to review through `short_observation` (10 of those 11 were
real tasks). A short observation in #astra-feedback is therefore
worth a second look as a separate change.

### Current model (2026-11-09)

Trained by `train_task_model.py` on `train.jsonl` (1,911 real rows) plus
`synthetic/synthetic_all.jsonl` (1,000 invented rows written by Muse),
same settings as before (1 epoch, 20 iterations, seed 42). Nothing from
`test.jsonl` or `eval_feedback.jsonl` is in the training data.

| Training data | #astra-feedback (60 real) | other-channel test (86) |
|---|---|---|
| real only | 44 of 50 found, 4 false alarms | 26 of 29, 8 false alarms |
| invented only | 43 of 50, 1 false alarm | 22 of 29, 18 false alarms |
| **real + invented (current)** | **46 of 50, 1 false alarm** | 24 of 29, 7 false alarms |

Through the full pipeline on #astra-feedback: 35 auto-posted (1 wrong),
16 to review, 1 real task lost silently. Five-fold cross-validation
that also trained on the real #astra-feedback rows gave the same 46/50,
so those rows were left out and the set stays clean. The four missed
tasks are all short requirement statements the model scores near zero
("Not every node needs a planet"); the statement net routes them to
review. With 60 rows, one piece moves the percentages by about 2 points.

### The statement net, and long pieces under the floor

A piece under the review floor (0.10) is normally dropped. Three
exceptions go to review instead: an explicit `TODO:` or `.open` /
`.assign` marker, a problem word such as "broken", and a requirement
statement (`is_requirement_statement`): 4 or more words with should,
needs, must, every, each or similar, not a question, not starting in the
first person. Triggers: `statement_net` and `floor_exception`.

Why: the model scores short impersonal statements like "Every room should
have a 1x1 main module" at about 0.007, the same as chat, so no cutoff can
tell them apart. On the 60 real #astra-feedback pieces, 7 of 50 tasks
were silently dropped this way.

This also fixes a bug in the earlier problem-word exception: a long piece
(8 words or more) passed the floor and was then dropped by the LOW cutoff,
so "the door sound is missing in the cargo bay when you jump through the
gate" was still lost. Anything that passes a floor exception now ends in
review.

Measured with the real model:

| | Real tasks dropped silently | Sent to review | Auto-posted wrong |
|---|---|---|---|
| `eval_feedback.jsonl` before | 7 of 50 | 11 of 60 | 2 |
| `eval_feedback.jsonl` after | 2 of 50 | 17 of 60 | 2 |
| `test.jsonl` before (other channels) | 0 of 29 | 12 of 86 | 4 |
| `test.jsonl` after (other channels) | 0 of 29 | 18 of 86 | 4 |

In #astra-feedback the 6 extra reviews were 5 real tasks and 1 not a task.
On the older test set, which is mostly other channels, all 6 extra
reviews were not tasks, so the net costs about 7 extra reviews per 100
pieces of ordinary chat. The bot only reads #astra-feedback. The two
tasks still lost have no cue word ("some get a star only", "in game chat
for connected players"). The net was designed after seeing the seven
misses, so treat the first row pair as optimistic and confirm with new
rulings.

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

## Review floor

A piece scoring under 10% (`REVIEW_FLOOR` in `classifier_service.py`)
is dropped instead of going to review, because ordinary chat like
"sounds good to me" scores about 1%. Two exceptions still go through
the normal rules: explicit `TODO:` or `.open` / `.assign` markers, and
problem words such as "broken", because "salvage is broken" is a real
task that scores the same as chat. The Decisions panel shows these
with the trigger `below_review_floor`.

## How it fits with the bot

Astrabot `cogs/tasks.py` watches mcdungeons #astra-feedback, sends
each message to this service, and routes the pieces by the decision
the service returns: `auto_task` pieces become wiki tasks and
embeds in #todo, `review` pieces become "Needs review" embeds in
#human-intervention with a thread, `auto_drop` pieces are dropped
silently. A human "Is a task" ruling moves the piece to #todo. If this service is down, the cog logs it and
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
