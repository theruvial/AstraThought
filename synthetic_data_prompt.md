# Prompt for generating synthetic training data with ChatGPT

Purpose: teach the task classifier the kinds of messages it currently
gets wrong. In #astra-feedback it silently drops short, impersonal
requirement statements ("X should have Y", "Every room needs Z", a bare
feature name) because they look like chat to a model trained mostly on
other channels.

How to use it:

1. Open a new ChatGPT conversation. Paste **Part 1** once.
2. Paste **Part 2** once per batch, changing the batch number and topic.
   Ten batches of 100 lines make 1,000. ChatGPT cannot write 1,000 lines
   in one reply, so do not ask for that.
3. Each batch comes back as a downloadable markdown file
   (`batch01.md`, `batch02.md`, ...). If your ChatGPT cannot create
   files, it will paste the lines instead. Save that as a text file
   with the same name. Either works: the checker ignores code fences,
   headings and anything that is not a JSON line.
4. Run `check_synthetic.py` on the files. It strips code fences, drops
   bad lines and duplicates, drops anything that matches `train.jsonl`,
   `test.jsonl` or `eval_feedback.jsonl`, and writes a clean file plus a
   sample for you to spot-check.
5. Thea trains. Nothing here trains a model.

Rules that protect the test set:

- The prompt below contains no real team messages and none from
  `eval_feedback.jsonl`. Do not paste any into ChatGPT. They would leak
  into the training data, and the test set would stop measuring anything.
- ChatGPT's labels are guesses. Spot-check a sample before training. A
  model trained on someone else's guesses learns their mistakes.
- Synthetic rows get `"source": "synthetic"`. Keep them under about a
  third of `train.jsonl`, and never add them to `test.jsonl` or
  `eval_feedback.jsonl`.

---

## Part 1: paste this once

```text
You are helping me build training data for a small text classifier. The
classifier reads messages from a Discord channel where a small indie game
team posts feedback and requests for their game, and decides whether each
message is a TASK (something a teammate should go and do) or NOT A TASK.

THE GAME: a co-op space ship game. Vocabulary you can use: ship modules
(engines, shields, guns, medbay, reactor, cargo bay), rooms and doors,
crew characters and NPCs, a star map made of nodes and sectors, jump
gates and warps, asteroids and hazards, encounters and events, a lobby
and menus, the navigation screen, a debug UI, icons and animations,
sound effects and music, cameras, hit points, costs and speeds.

WRITE LIKE REAL DISCORD MESSAGES. Casual, short, varied. Median length 7
words, range 2 to 25 words. Never a single word. About 15% should have a
typo, lowercase only, a missing end punctuation, or run-on phrasing. No
numbering, no markdown, no emojis, no real people's names, no @mentions.
Do not start many lines with the same word. Do not write polished,
textbook sentences.

LABEL 1 (TASK): a teammate could read it and know a specific change to make
to the game, its art or audio, its UI, or its tools. It does NOT need an
action verb. These are all tasks:
  - impersonal requirement statements: "Shields should recharge slower
    after a hit", "Every ship starts with one spare repair kit", "Not
    every sector needs a shop"
  - bare feature names or short noun phrases asking for a feature: "mute
    button in the lobby", "a second door for the engine room"
  - imperative requests: "Add a ping to the star map", "Swap the red
    alert icon"
  - specific bug reports: "crew get stuck in the cargo bay door after a
    warp", "the nav screen freezes when you open it twice"
  - art, audio and UI replacement requests: "replace the reactor icon with
    something that glows", "quieter alarm sound"

LABEL 0 (NOT A TASK). Every one of these is clearly not a task:
  - chat: reactions, thanks, agreement, jokes, greetings ("ha that was
    close", "thanks for the quick fix")
  - questions to the group: "do we want 3 or 4 hp on guns?", "who is
    doing the lobby screen?"
  - suggestions that are hedged or first person: "we should maybe look at
    cheaper engines at some point", "I think the map could be bigger",
    "might be cool to have weather", "we could also try fog"
  - someone's own plans: "I'm gonna fix the door tonight", "going to test
    the new build after lunch"
  - status updates: "pushed the shield changes", "build is up on the
    server"
  - quick requests between two people, not about the game: "can you
    update my branch?", "ping me when the build finishes"
  - vague or context-less: "do it like last time", "same as before",
    "the other one"
  - announcements and test messages about the channel or bot: "this
    channel is for feedback only", "testing the new bot"
  - observations or feelings with no specific change asked for: "the
    engine room feels empty today", "that boss was brutal", "the new
    music is kinda loud lol", "engines eat way too much power" (a
    complaint with no specific change)

THE KEY CONTRAST to get right, with many pairs: an impersonal definite
statement about how the game should be is a TASK ("Doors should close
slower"). The same idea phrased as a first-person suggestion, hedge or
question is NOT ("I think doors could close slower", "maybe doors should
close slower?", "should doors close slower?"). Generate plenty of matched
pairs like this.

If a line could reasonably be labeled either way, do not write it. Every
line must be clearly one or the other.

OUTPUT FORMAT. Output ONLY lines of JSON, one per line, nothing else: no
intro, no commentary, no code fences, no blank lines. Each line:
{"text": "...", "label": 1, "reason": "..."}
label is 1 or 0. reason is one of:
  label 1: task_requirement, task_feature, task_imperative, task_bug,
           task_asset
  label 0: chat, question, suggestion, own_plan, request, changelog,
           too_vague, meta, observation

Never repeat a line from an earlier batch in this conversation, and never
write near-duplicates (same sentence with one word swapped).

DELIVERY. For each batch, create a downloadable markdown file named
batchNN.md (NN is the batch number, two digits). The file contains a
single code block holding the JSON lines and nothing else, and you give me
the download link. Also state the line count. If you cannot create files,
output the lines directly in one code block instead. Do not summarise the
lines, and do not change them after writing them.

Reply "ready" and wait for my first batch request.
```

---

## Part 2: paste once per batch

Change the batch number and the topic each time. Topics, in order:
1 ship modules and combat, 2 star map and travel, 3 menus and lobby,
4 art, icons and animation, 5 sound and music, 6 crew and NPCs,
7 encounters, events and hazards, 8 balance and numbers (hp, costs,
speeds), 9 bugs, performance and networking, 10 tools, debug UI and
dev workflow.

```text
Batch 1 of 10. Topic: ship modules and combat. Write exactly 100 lines in
the JSON line format. Use this mix, with the reason values (deliver it as batch01.md, as agreed):
  label 1 (52 lines): task_requirement 20, task_feature 9,
    task_imperative 8, task_bug 9, task_asset 6
  label 0 (48 lines): suggestion 9, question 7, observation 7,
    own_plan 4, changelog 4, chat 7, request 4, too_vague 3, meta 3
Mix the order. Include many matched pairs (a definite statement next to a
hedged or first-person version of the same idea). Mostly stay on the
topic, but a few off-topic chat lines are fine. Output only the 100 lines.
```

If a reply comes back short, send: `You wrote N lines. Write the missing
(100 - N) with the same mix, no repeats.` If it drifts (polished
sentences, wrong labels, repeated openings), send: `Re-read the rules.
Lines are too clean and too similar. Rewrite this batch: shorter, messier,
more varied openings, and every label 1 must name a specific change.`

Totals across ten batches: 520 tasks (200 requirement statements, 90
feature phrases, 80 imperatives, 90 bug reports, 60 art or audio
requests) and 480 not-tasks. The surplus of requirement statements is on
purpose: that is the weak spot.
