"""AstraThought classifier service.

Serves the local SetFit task classifier over HTTP on loopback only,
so the Astrabot tasks cog (which runs in a Python without PyTorch)
can ask it about Discord messages without loading the model itself.

Run from the AstraThought checkout, inside its virtual environment:

    .\\.venv\\Scripts\\Activate.ps1
    python classifier_service.py

Endpoints (all JSON, loopback only, no auth):

    GET  /health
        {"ok": true, "model_loaded": true, "model_dir": "..."}
    POST /classify
        {"text": "salvage is broken"}
        -> {"ok": true, "pieces": [{"text": ..., "score": 0.12,
        "decision": "auto_drop", "trigger": "none",
        "reason": "score 0.12 is at or below 0.25"}]}

    Every classification is also written to the console (stdout via
    logging): the message as received, how it split, and for each
    piece its score, decision, trigger, and a plain-words reason.
    That is the classifier's thought process, visible in the console
    pane while the bot runs. Nothing is logged for /health checks.

    Each logged piece also gets an id number (#1, #2, ...). Type in
    the console window to rule on one:

        <id> agree      the bot was right about decision #id
        <id> disagree   the bot was wrong about decision #id
        list             show recent decisions with their ids

    Agree/disagree answers "is this a task?". For review decisions
    it answers the lean (score >= 0.5 leans task). The ruling is
    appended to the labels file (env TASKS_LABELS_PATH, same file
    the cog writes, so /task_stats counts it and the 50-label
    retrain trigger sees it) and updates the overrides file, so it
    takes effect immediately without restarting the service.
    POST /classify_batch
        {"texts": ["...", "..."]}
        -> {"ok": true, "results": [{"text": ..., "pieces": [...]}, ...]}

Decisions, per piece:
    auto_task  score >= HIGH (0.75)
    auto_drop  score <= LOW (0.25)
    review     anything in between, or any review trigger fires

HIGH and LOW are placeholders. They must be set by measuring the
score distribution on test.jsonl with score_distribution.py first.
Do not pick final thresholds by guessing, and do not tune them
against test.jsonl after that measurement, or the test set stops
being held out. One measurement pass to pick the band, then leave
it alone.

Review triggers, in addition to the score band:
    model_regex_disagreement  a cheap rule (TODO:, imperative start,
        ".open" / ".assign") says task and the model says not, or the
        reverse. The handoff regex scored 92% precision / 68% recall
        with different errors from the model, so disagreement is a
        good reason to ask a human.
    suggestion_marker  the model flags the piece as a task, but it
        contains suggestion wording ("we should", "I think",
        "maybe we need", "we need to"). These were 3 of the 8 false
        alarms in the v2 test run.
    short_observation  under 8 words, no imperative verb, e.g.
        "salvage is broken" (a task) vs "asteroid field is not an
        asteroid field" (not a task, per Thea). Same shape, opposite
        rulings, so a human makes the call.
    split_disagreement  the message split into several pieces, some
        auto and some in the review band. The band piece goes to
        review with this trigger instead of "score_band".

Human overrides:
    Before the model runs, each piece is checked against an overrides
    file (env TASK_OVERRIDES_PATH, default data/task_overrides.json
    next to this file) mapping normalized text to label 0 or 1.
    A hit skips the model and returns decision auto_task / auto_drop
    with trigger "human_override" and score 1.0 / 0.0. The file is
    reloaded on every request, so a ruling takes effect immediately
    without restarting the service or retraining. The cog writes
    this file when a human rules on a review item or a confirmed
    task. Overrides never go into train.jsonl on their own.

Model loading:
    The SetFit model is loaded once at startup from
    ./tasks-AstraThought-v2 (env ASTRATHOUGHT_MODEL_DIR). SetFit and
    torch are imported lazily inside load_model(), so importing this
    module without torch installed does not crash. If the model cannot be loaded, the
    service still starts and /health reports model_loaded false,
    but /classify and /classify_batch return 503. Never upgrade
    torch past 2.14.0+cu126: 2.15+ dropped GTX 1070 support.

Retraining:
    This service does not train. Retraining is a separate batch job
    (train_task_model.py, from the base model, about 5 minutes) and
    only happens when 50 new labels have been collected, no matter
    how long that takes. No per-ruling micro-training.

Stdlib only. No new dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from split_tasks import IMPERATIVES, split_message

log = logging.getLogger("classifier_service")

HERE = Path(__file__).resolve().parent

# Placeholder thresholds. Set from score_distribution.py output.
HIGH = 0.75
LOW = 0.25

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 13383
DEFAULT_MODEL_DIR = HERE / "tasks-AstraThought-v2"

SUGGESTION_MARKERS = (
    "we should",
    "i think",
    "maybe we need",
    "we need to",
    "we could",
    "i feel like",
    "should we",
)

_RULE_TODO_RE = re.compile(r"^\s*(TODO|FIXME|XXX)\s*:", re.IGNORECASE)
_RULE_COMMAND_RE = re.compile(r"^\s*\.(open|assign)\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase, for override lookup."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _first_word(text: str) -> str:
    match = re.match(r"([A-Za-z]+)", text.strip())
    return match.group(1).lower() if match else ""


def _has_imperative(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text.lower())
    return any(word in IMPERATIVES for word in words)


def rule_says_task(piece: str, raw_text: str = "") -> bool:
    """Cheap rule: TODO:, .open/.assign, or an imperative start.

    This is a disagreement detector, not a task detector on its own.
    The splitter strips TODO: from pieces, so check the raw text too.
    """
    if _RULE_TODO_RE.search(raw_text or piece):
        return True
    if _RULE_COMMAND_RE.search(piece):
        return True
    return _first_word(piece) in IMPERATIVES


def is_short_observation(piece: str) -> bool:
    """Under 8 words, no imperative verb anywhere in the piece."""
    words = piece.split()
    if len(words) >= 8 or len(words) == 0:
        return False
    return not _has_imperative(piece)


def _one_line(text: str, limit: int = 160) -> str:
    """Collapse a message to one log line, truncated."""
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def decide_piece(piece: str, score: float, raw_text: str = "",
                 split_disagrees: bool = False) -> dict:
    """Decide one split piece from its model score and the triggers.

    The returned dict also carries "reason": one plain-words line
    saying why this decision was made. The reason is what the
    console log shows, so a human watching the service can follow
    the classifier's thought process piece by piece.
    """
    model_says_task = score >= 0.5
    rule_task = rule_says_task(piece, raw_text)
    lowered = piece.lower()

    if rule_task != model_says_task:
        return {"text": piece, "score": round(score, 4),
                "decision": "review", "trigger": "model_regex_disagreement",
                "reason": (f"model says {'task' if model_says_task else 'not a task'} "
                           f"(score {score:.2f}) but the rule says "
                           f"{'task' if rule_task else 'not a task'}")}
    marker = next((m for m in SUGGESTION_MARKERS if m in lowered), None)
    if model_says_task and marker is not None:
        return {"text": piece, "score": round(score, 4),
                "decision": "review", "trigger": "suggestion_marker",
                "reason": f"model says task but the text says '{marker}'"}
    if is_short_observation(piece):
        words = len(piece.split())
        return {"text": piece, "score": round(score, 4),
                "decision": "review", "trigger": "short_observation",
                "reason": (f"{words} word{'s' if words != 1 else ''}, "
                           "no imperative verb, same shape as ruled both ways")}
    if split_disagrees and LOW < score < HIGH:
        return {"text": piece, "score": round(score, 4),
                "decision": "review", "trigger": "split_disagreement",
                "reason": ("other pieces in this message were auto-decided, "
                           f"this one is in the band ({LOW}-{HIGH})")}
    if score >= HIGH:
        return {"text": piece, "score": round(score, 4),
                "decision": "auto_task", "trigger": "none",
                "reason": f"score {score:.2f} is at or above {HIGH}"}
    if score <= LOW:
        return {"text": piece, "score": round(score, 4),
                "decision": "auto_drop", "trigger": "none",
                "reason": f"score {score:.2f} is at or below {LOW}"}
    return {"text": piece, "score": round(score, 4),
            "decision": "review", "trigger": "score_band",
            "reason": f"score {score:.2f} is between {LOW} and {HIGH}"}


def overrides_path() -> Path:
    raw = os.environ.get("TASK_OVERRIDES_PATH")
    if raw:
        return Path(raw)
    return HERE / "data" / "task_overrides.json"


def labels_path() -> Path:
    """Where console rulings (and the cog's rulings) are saved.

    Same env var as the cog (TASKS_LABELS_PATH), so setting it once
    on the host makes the service and the cog write one shared
    labels file, which is what /task_stats counts and what the
    50-label retrain trigger reads.
    """
    raw = os.environ.get("TASKS_LABELS_PATH")
    if raw:
        return Path(raw)
    return HERE / "data" / "task_labels.jsonl"


def load_overrides(path: Path | None = None) -> dict:
    """Read the overrides file fresh. Missing or broken file means none."""
    path = path or overrides_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for key, value in data.items():
        try:
            out[normalize_text(str(key))] = int(value)
        except (TypeError, ValueError):
            continue
    return out


_CONSOLE_HELP = [
    "Classifier console commands:",
    "  <id> agree      the bot was right about decision #id",
    "  <id> disagree   the bot was wrong about decision #id",
    "  <id>            show decision #id and what agree/disagree would record",
    "  list            show recent decisions with their ids",
    "  help            show this",
    "Agree/disagree answers 'is this a task?'. For review decisions it",
    "answers the lean (score >= 0.5 leans task). Every ruling is saved",
    "as a training label and updates the overrides file, so it takes",
    "effect immediately without restarting the service.",
]


class DecisionRegistry:
    """Numbered console decisions, so a human can rule on them by id.

    Every classified piece gets a sequential id when it is logged.
    Typing "<id> agree" or "<id> disagree" in the console records a
    training label for that piece's text and updates the overrides
    file, so the ruling takes effect immediately. Agree means the bot
    was right; disagree means the opposite. For review pieces there
    is no verdict, so agree/disagree answers the lean: score >= 0.5
    leans task, below leans not a task.

    Only the most recent KEEP decisions are kept; ids keep counting
    up, and an id that has expired answers "unknown". Labels use the
    same JSONL shape as the cog's labels file, with reviewer
    "console", so Thea's batch merge picks them up with the rest.
    """

    KEEP = 200

    def __init__(self, labels_path, overrides_path):
        self.labels_path = Path(labels_path)
        self.overrides_path = Path(overrides_path)
        self._lock = threading.Lock()
        self._next_id = 1
        self._entries = {}
        self._order = []

    def record(self, pieces):
        """Number each piece and store it. Returns {id: entry} in order."""
        numbered = {}
        with self._lock:
            for piece in pieces:
                entry_id = self._next_id
                self._next_id += 1
                entry = dict(piece)
                entry["id"] = entry_id
                if piece["decision"] == "auto_task":
                    entry["label"] = 1
                elif piece["decision"] == "auto_drop":
                    entry["label"] = 0
                else:
                    entry["label"] = 1 if piece["score"] >= 0.5 else 0
                self._entries[entry_id] = entry
                self._order.append(entry_id)
                numbered[entry_id] = entry
            while len(self._order) > self.KEEP:
                self._entries.pop(self._order.pop(0), None)
        return numbered

    @staticmethod
    def entry_line(entry):
        """One console line for a numbered decision."""
        if entry["decision"] == "review":
            verdict = ("leans task" if entry["label"] == 1
                       else "leans not a task")
        else:
            verdict = ("says task" if entry["label"] == 1
                       else "says not a task")
        return (f"#{entry['id']} [{entry['decision']} {entry['score']:.2f} "
                f"{entry['trigger']}] {_one_line(entry['text'], 80)}"
                f" -- {verdict}")

    def _list_lines(self):
        with self._lock:
            entries = [self._entries[i] for i in self._order[-10:]]
        if not entries:
            return ["No decisions yet. They appear here as messages "
                    "are classified."]
        return ["Recent decisions:"] + [self.entry_line(e) for e in entries]

    def _describe_lines(self, entry):
        label = entry["label"]
        return [
            self.entry_line(entry),
            f"  reason: {entry.get('reason', '')}",
            f"  'agree' would record it as "
            f"{'a task' if label else 'not a task'}; 'disagree' as "
            f"{'not a task' if label else 'a task'}.",
        ]

    def _save_label(self, entry, label):
        record = {
            "text": entry["text"],
            "label": label,
            "source_message_id": None,
            "model_score": entry["score"],
            "trigger": entry["trigger"],
            "reviewer_id": "console",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self._lock:
            self.labels_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.labels_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            overrides = load_overrides(self.overrides_path)
            overrides[normalize_text(entry["text"])] = label
            tmp = self.overrides_path.with_suffix(".json.tmp")
            self.overrides_path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(overrides, ensure_ascii=False, indent=2),
                encoding="utf-8")
            tmp.replace(self.overrides_path)

    def handle_line(self, line):
        """Handle one console line. Returns output lines to print."""
        tokens = line.strip().lower().split()
        if not tokens:
            return []
        if tokens in (["list"], ["l"]):
            return self._list_lines()
        if tokens in (["help"], ["h"], ["?"]):
            return list(_CONSOLE_HELP)
        entry_id = None
        verdict = None
        for token in tokens:
            if token in ("agree", "a"):
                verdict = "agree"
            elif token in ("disagree", "d"):
                verdict = "disagree"
            elif token.lstrip("#").isdigit():
                entry_id = int(token.lstrip("#"))
        if entry_id is None and verdict is None:
            return ["I did not understand that. Type 'help' for commands."]
        if entry_id is None:
            return ["Agree or disagree with which decision? Give its id, "
                    "e.g. '12 agree'."]
        with self._lock:
            entry = self._entries.get(entry_id)
        if entry is None:
            return [f"Decision #{entry_id} is unknown or has expired. "
                    "Type 'list' to see recent ones."]
        if verdict is None:
            return self._describe_lines(entry)
        label = entry["label"] if verdict == "agree" else 1 - entry["label"]
        self._save_label(entry, label)
        return [f"#{entry_id} recorded as "
                f"{'a task' if label else 'not a task'} "
                f"({verdict} with the bot). Label saved, override updated."]


DECISION_LOG = DecisionRegistry(labels_path(), overrides_path())


def _console_loop():
    """Read console input in the background. Never blocks serving."""
    print("Classifier console: type '<id> agree' or '<id> disagree' to rule "
          "on a decision, 'list' to see recent ones, 'help' for commands.",
          flush=True)
    try:
        for line in sys.stdin:
            for out in DECISION_LOG.handle_line(line):
                print(out, flush=True)
    except Exception as exc:  # noqa: BLE001 - console input must not kill the server
        log.warning("Console input ended: %s", exc)


class Classifier:
    """Holds the loaded SetFit model and classifies split pieces."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Load the model once. Lazy import so this module imports
        without torch installed. Returns True if loaded."""
        try:
            from setfit import SetFitModel
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            log.error("Could not import setfit: %s", exc)
            return False
        try:
            self.model = SetFitModel.from_pretrained(str(self.model_dir))
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            log.error("Could not load model from %s: %s", self.model_dir, exc)
            return False
        return True

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def score_pieces(self, pieces: list) -> list:
        """predict_proba for each piece. Empty list in, empty list out."""
        if not pieces:
            return []
        with self._lock:
            probs = self.model.predict_proba(pieces)
        return [float(row[1]) for row in probs]

    def classify_text(self, text: str) -> dict:
        """Split one message, check overrides, classify each piece."""
        pieces = split_message(text or "")
        if not pieces:
            log.info("classify: %r -> 0 pieces, dropped",
                     _one_line(text, 200))
            return {"ok": True, "pieces": []}
        overrides = load_overrides()
        results = []
        # Score only the pieces that are not overridden, in one batch.
        to_score = []
        to_score_index = []
        for index, piece in enumerate(pieces):
            label = overrides.get(normalize_text(piece))
            if label is not None:
                results.append({
                    "text": piece,
                    "score": 1.0 if label == 1 else 0.0,
                    "decision": "auto_task" if label == 1 else "auto_drop",
                    "trigger": "human_override",
                    "reason": (f"a human ruled this a {'task' if label == 1 else 'not a task'} "
                               "in task_overrides.json, model skipped"),
                })
            else:
                to_score.append(piece)
                to_score_index.append(index)
        scores = self.score_pieces(to_score) if to_score else []
        scored = {}
        for piece, score in zip(to_score, scores):
            scored[piece] = score
        # A message split into several pieces where some are auto and
        # some are in the band: the band pieces get split_disagreement.
        decisions = [scored.get(p) for p in pieces]
        has_auto = any(s is not None and (s >= HIGH or s <= LOW)
                       for s in decisions)
        has_band = any(s is not None and LOW < s < HIGH for s in decisions)
        split_disagrees = has_auto and has_band
        final = {}
        for index, piece in enumerate(pieces):
            if results and any(r["text"] == piece for r in results):
                continue
            score = scored[piece]
            final[piece] = decide_piece(
                piece, score, raw_text=text or "",
                split_disagrees=split_disagrees)
        ordered = []
        for piece in pieces:
            hit = next((r for r in results if r["text"] == piece), None)
            ordered.append(hit if hit is not None else final[piece])
        self._log_thought_process(text, pieces, ordered)
        return {"ok": True, "pieces": ordered}

    @staticmethod
    def _log_thought_process(text: str, pieces: list, ordered: list) -> None:
        """Write the classifier's thought process to the console.

        One block per message: what came in, how it split, and for
        each piece the score, the decision, the trigger, and the
        plain-words reason. This is what the console pane shows while
        the bot is running, so a human can see why each message was
        posted, dropped, or sent to review.
        """
        log.info("classify: %r -> %d piece(s)",
                 _one_line(text, 200), len(pieces))
        for entry in DECISION_LOG.record(ordered).values():
            log.info("  %s", DecisionRegistry.entry_line(entry))
            if entry.get("reason"):
                log.info("    reason: %s", entry["reason"])


CLASSIFIER = Classifier(DEFAULT_MODEL_DIR)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep stdout quiet, use logging
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > 1_000_000:
            return None
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler naming
        if self.path == "/health":
            self._send_json(200, {
                "ok": True,
                "model_loaded": CLASSIFIER.loaded,
                "model_dir": str(CLASSIFIER.model_dir),
            })
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler naming
        if self.path not in ("/classify", "/classify_batch"):
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        if not CLASSIFIER.loaded:
            self._send_json(503, {"ok": False, "error": "model not loaded"})
            return
        payload = self._read_json()
        if payload is None:
            self._send_json(400, {"ok": False, "error": "invalid JSON body"})
            return
        try:
            if self.path == "/classify":
                text = payload.get("text")
                if not isinstance(text, str):
                    self._send_json(400, {"ok": False,
                                          "error": "text is required"})
                    return
                self._send_json(200, CLASSIFIER.classify_text(text))
                return
            texts = payload.get("texts")
            if not isinstance(texts, list) or not all(
                    isinstance(t, str) for t in texts):
                self._send_json(400, {"ok": False,
                                      "error": "texts must be a list of strings"})
                return
            results = []
            for text in texts:
                result = CLASSIFIER.classify_text(text)
                results.append({"text": text, "pieces": result["pieces"]})
            self._send_json(200, {"ok": True, "results": results})
        except Exception as exc:  # noqa: BLE001 - never crash on one request
            log.exception("Classify failed: %s", exc)
            self._send_json(500, {"ok": False, "error": "classify failed"})


def main() -> None:
    parser = argparse.ArgumentParser(description="AstraThought classifier service")
    parser.add_argument("--host", default=os.environ.get(
        "CLASSIFIER_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get(
        "CLASSIFIER_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--model-dir", default=os.environ.get(
        "ASTRATHOUGHT_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    global CLASSIFIER  # noqa: PLW0603 - set once at startup
    CLASSIFIER = Classifier(args.model_dir)
    if not CLASSIFIER.load():
        log.warning("Model did not load from %s. /health will report "
                    "model_loaded false and /classify will return 503.",
                    args.model_dir)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    log.info("Classifier service listening on %s:%s, model %s, loaded=%s, "
             "auto_task >= %s, auto_drop <= %s, between them goes to review",
             args.host, args.port, args.model_dir, CLASSIFIER.loaded,
             HIGH, LOW)
    threading.Thread(target=_console_loop, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
