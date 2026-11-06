# AstraThought

A small classifier that decides whether a Discord message is a task, trained on Thea's judgment of mcdungeons #general.

## Files

| File | What it is |
|---|---|
| `train.jsonl` | Items the model learns from: older mcdungeons #general, plus Aboard The Astra #task-assignment and #work (`source` says which) |
| `train_recent_only.jsonl` | Backup of the first training set, #general only |
| `test.jsonl` | Newer items kept out of training, used to score it |
| `train_task_model.py` | Trains the model, saves it to `model\`, prints its score |
| `check_message.py` | Tries the trained model on one message |
| `requirements.txt` | Pinned packages for the virtual environment |

Each line of the data holds `text`, `label` (1 is a task, 0 is not), `reason`, `position` and `reviewed`. Only `text` and `label` are used for training. `reason` records why something was ruled out.

These files contain real messages from the server. Keep them out of any public repo.

## Set up (once)

Open PowerShell. This downloads a few GB, because it includes the GPU build of PyTorch.

```powershell
cd C:\codingShit\AstraThought
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, run this once in that window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then check the GPU works. This should print `True` and the card's name:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements.txt` pins PyTorch 2.14 with CUDA 12.6, because 2.15 and later no longer support the GTX 1070. Do not run `pip install --upgrade torch`. On a newer GPU, or with no GPU, remove the `--extra-index-url` line and the `torch` line first.

## Train

```powershell
cd C:\codingShit\AstraThought
.\.venv\Scripts\Activate.ps1
python train_task_model.py
```

The first run also downloads the base model, `all-MiniLM-L6-v2`, about 90 MB. The script prints `Running on: cuda:0` for the GPU or `Running on: cpu` for the CPU. On the CPU it should take a few minutes, though it may be slower while the bot is busy.

It finishes by listing its mistakes and printing precision and recall on `test.jsonl`. Compare those with the regex on recent #general: 96% precision and 89% recall, measured before the final label review.

## Try it

```powershell
python check_message.py "add npcs to the crew UI"
python check_message.py "anyway im off to bed now"
```

## Improving it

1. Read the mistakes printed after training.
2. If a label was wrong, fix that line. Change `label` to 1 or 0 in `train.jsonl` or `test.jsonl`.
3. If it keeps missing a kind of task, add new examples to `train.jsonl` only. Adding them to `test.jsonl` would make the score meaningless.
4. Train again. Delete `model\` first if you want a clean start.

Do not tune against `test.jsonl` for long. Every round of fixes based on its mistakes makes its score less honest, so check new messages from Discord now and then.

## What was left out on purpose

- **Reposts, duplicates and items already marked DONE.** They are ruled out by context, not wording: `Customer sprite` reads like a task and only fails because it repeats an earlier one. Training on those would teach the wrong lesson. Handle them with a separate duplicate check.
- **Bare links, code blocks, email addresses and exact repeats.** Filter these out before calling the model.

## Using it

- Split messages into lines or sentences first. One message can hold many tasks, and in #general 12 messages held 44 of the 113 tasks.
- Strip `TODO:`, `-` and mentions before classifying, because the training text has them removed.
- Treat `.open`, `.assign`, `TODO:` and dash lines as strong signals on their own.
- Most `chat` and `design_note` labels were not individually reviewed by Thea; `reviewed` marks the ones that were.
