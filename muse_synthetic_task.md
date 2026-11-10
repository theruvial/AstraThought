You are helping me build training data for a small text classifier. The
classifier reads messages from a Discord channel where a small indie game
team posts feedback and requests for their game, and decides whether each
message is a TASK (something a teammate should go and do) or NOT A TASK.

WHAT YOU MAY AND MAY NOT DO
- Work in C:\codingShit\AstraThought. Write only the files named below.
- Do NOT open train.jsonl, test.jsonl, eval_feedback.jsonl,
  train_before_rules.jsonl, train_recent_only.jsonl, or any other file of
  real team messages. Do NOT read Discord history. Write every line from
  your own imagination. Real messages would leak into training data and
  ruin the test set.
- Do NOT train, retrain or modify any model, and do not change any file
  other than the batch files and the checker's output.
- Do NOT merge, commit or push anything. Leave that to me.

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

ROW FORMAT. Each row is one JSON object per line:
{"text": "...", "label": 1, "reason": "...", "source": "synthetic"}
label is 1 or 0. reason is one of:
  label 1: task_requirement, task_feature, task_imperative, task_bug,
           task_asset
  label 0: chat, question, suggestion, own_plan, request, changelog,
           too_vague, meta, observation

Never repeat a line, and never write near-duplicates (same sentence with
one word swapped), within a batch or across batches.

THE WORK. Write ten batches of exactly 100 rows each, to
C:\codingShit\AstraThought\synthetic\batch01.jsonl through batch10.jsonl
(create the folder). The files contain only the JSON lines: no fences, no
headings, no blank lines. Write one batch at a time, count its lines, and
rewrite it if the count is not 100. Do not edit lines after writing them
except to fix the count or remove a repeat. Topics, in order:
  1 ship modules and combat      6 crew and NPCs
  2 star map and travel          7 encounters, events and hazards
  3 menus and lobby              8 balance and numbers (hp, costs, speeds)
  4 art, icons and animation     9 bugs, performance and networking
  5 sound and music              10 tools, debug UI and dev workflow
Mostly stay on each batch's topic, but a few off-topic chat lines are fine.

Mix for EVERY batch of 100 (mix the order within the batch):
  label 1 (52 lines): task_requirement 20, task_feature 9,
    task_imperative 8, task_bug 9, task_asset 6
  label 0 (48 lines): suggestion 9, question 7, observation 7,
    own_plan 4, changelog 4, chat 7, request 4, too_vague 3, meta 3
Include many matched pairs (a definite statement next to a hedged or
first-person version of the same idea).

Across ten batches that is 520 tasks (200 requirement statements, 90
feature phrases, 80 imperatives, 90 bug reports, 60 art or audio
requests) and 480 not-tasks. The surplus of requirement statements is on
purpose: that is the weak spot.

QUALITY CHECK BEFORE YOU FINISH. For each batch, reread it once and fix:
polished or textbook sentences, repeated openings (no word may start more
than 6 lines in a batch), labels that fail the rules above, and any label
1 that does not name a specific change.

THEN RUN THE CHECKER. From C:\codingShit\AstraThought, with the project's
Python, run:
  py check_synthetic.py synthetic\batch*.jsonl
Report its output to me verbatim (counts kept, dropped, and why), then
stop. Do not read the review sample's answer key, do not run it with
--score, and do not train anything.
