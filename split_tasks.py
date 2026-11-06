"""Split a raw Discord message into individual candidate task units.

Stdlib only. Shared module: use it both for training-data preparation
and at runtime in the tasks cog / check_message.py, before classifying.
The splitter does NOT decide whether a piece is a task. It only cuts
a message into pieces. Classification (model) or a human decides labels.

Rules, in order:
  1. Drop code blocks (```...``` and inline `...`). If nothing is left, return [].
  2. Split on newlines first. A newline is the strongest boundary in Discord.
  3. Split each line on list markers: leading dash / bullet / star, or
     numbered "1." / "1)" at the start of a line.
  4. Split on sentence boundaries (. ! ? followed by whitespace and a
     letter, digit, quote, or opening bracket). Lowercase sentence starts
     are common in Discord, so capitals are not required. Abbreviations
     (i.e., e.g., etc., alt., def.), decimals (0.5), and ellipses are
     explicitly not split.
  5. Split on semicolons.
  6. Split on ", " followed by an imperative verb, but ONLY when the
     piece before the comma also starts with (or contains) an imperative
     verb. This is the conservative comma rule: it catches
     "Fix X, fix Y" but leaves "Design a flamethrower that spreads fire,
     has high damage, and makes the player move slower" alone, because
     "has" / "makes" are not imperatives.
  7. Clean each piece: strip TODO:, bullets, numbering, mentions
     (<@123>, @name), leading "new task for you:" style prefixes are
     left in place (they are a signal, not noise, for the classifier),
     collapse whitespace.
  8. Drop empty pieces, bare links, and pieces that are only punctuation
     or a single short word that is not a verb phrase (kept simple: drop
     pieces under 3 characters).

Known limits, documented not hidden:
  - "and" joining two tasks ("Build X and move to Y") is NOT split,
    except when it follows a comma (step 6). Splitting on bare "and"
    breaks single tasks like "add npcs to the crew UI and fix the map".
  - Gerund joins ("getting the bridge functional, getting the squire
    functional") are NOT split. They need a human or a better parser.
  - Abbreviations ("e.g.", "IE", version numbers) can cause a false
    sentence split. The regex requires a capital/digit after the
    boundary, which avoids most of these.
  - A split piece is not automatically a task. See the report for the
    labeling rule.
"""

import re

# Imperative verbs seen in the training data, plus common ones.
# Used ONLY for the comma rule, not as a task detector on its own.
IMPERATIVES = {
    "add", "fix", "remove", "make", "change", "update", "implement",
    "create", "move", "adjust", "increase", "decrease", "replace",
    "enable", "disable", "build", "set", "use", "allow", "show", "hide",
    "load", "save", "continue", "upgrade", "standardize", "design",
    "complete", "test", "apply", "note", "check", "merge", "revert",
    "rename", "convert", "clean", "split", "combine", "limit", "open",
    "close", "start", "stop", "write", "draw", "get", "give", "put",
    "take", "drop", "post", "send", "buy", "sell",
}

_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
_MENTION_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#[0-9]+>")
_AT_MENTION_RE = re.compile(r"@\w+")
_TODO_RE = re.compile(r"^\s*(TODO|FIXME|XXX)\s*:\s*", re.IGNORECASE)
_LIST_RE = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z0-9\"'(\[])")
_ABBREV = {"i.e", "e.g", "etc", "alt", "def", "vs", "approx", "fig",
           "mr", "mrs", "ms", "dr", "st", "gen", "b", "w", "ie"}
def _split_sentences(text):
    """Split on . ! ? boundaries, skipping abbreviations, decimals and ellipses."""
    parts, start = [], 0
    for m in _SENTENCE_RE.finditer(text):
        boundary = m.start()  # whitespace start; period is just before it
        # find the sentence-ending punctuation
        j = boundary - 1
        while j >= 0 and text[j] in ".!?":
            j -= 1
        punct_run = text[j + 1:boundary]
        if len(punct_run) > 1:  # ellipsis "...." - not a sentence boundary
            continue
        # inside parentheses: "(Happy emoji, angry emoji, neutral emoji.
        # STAR WITH TONGUE OUT EMOJI)" is one list, not two sentences
        if text[:boundary].count("(") > text[:boundary].count(")"):
            continue
        # dotted abbreviations: "i.e." and "e.g." end in a single letter,
        # so the word-before check alone misses them
        if text[max(0, boundary - 4):boundary].lower() in ("i.e.", "e.g."):
            continue
        # word before the period
        k = j
        while k >= 0 and (text[k].isalnum() or text[k] in "'-"):
            k -= 1
        word_before = text[k + 1:j + 1].lower()
        if word_before in _ABBREV:
            continue
        if j > 0 and text[j - 1].isdigit() and text[j + 1].isdigit() if False else False:
            continue
        # decimal like 0.5 - period between digits
        if j > 0 and j + 1 < len(text) and text[j - 1].isdigit() and text[j + 1].isdigit():
            continue
        parts.append(text[start:m.end()].strip() if False else text[start:boundary].strip())
        # start of next sentence is after the whitespace
        start = m.end()
    parts.append(text[start:].strip())
    return [s for s in parts if s]
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _strip_code(text):
    text = _CODE_BLOCK_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    return text


def _clean(piece):
    piece = _MENTION_RE.sub("", piece)
    piece = _AT_MENTION_RE.sub("", piece)
    piece = _TODO_RE.sub("", piece)
    piece = _LIST_RE.sub("", piece)
    piece = re.sub(r"^and\s+", "", piece, flags=re.IGNORECASE)
    piece = re.sub(r"\s+", " ", piece).strip(" \t\r\n.,;:")
    # strip again after punctuation trim, in case TODO was after a bullet
    piece = _TODO_RE.sub("", piece).strip()
    return piece


def _is_junk(piece):
    if not piece or len(piece) < 3:
        return True
    if _URL_RE.match(piece):
        return True
    if re.fullmatch(r"[^\w]+", piece):
        return True
    return False


def _split_on_comma_imperative(piece):
    """Split 'Verb ..., verb ...' on commas before an imperative verb.

    Only splits when the head of the piece starts with an imperative
    verb (or 'continue to', 'make sure to', etc.). Returns a list.
    """
    head = piece.strip()
    first_word = re.match(r"([A-Za-z]+)", head)
    if not first_word or first_word.group(1).lower() not in IMPERATIVES:
        return [piece]
    # split on ", " followed by an imperative verb, or ", and <verb>"
    pattern = re.compile(
        r",\s+(?:and\s+)?(?=(?:" + "|".join(sorted(IMPERATIVES)) + r")\b)",
        re.IGNORECASE,
    )
    parts = pattern.split(piece)
    # re.split with a lookahead group returns only the pieces (group is
    # empty because the verb is in a lookahead, so no extra elements)
    return [p.strip() for p in parts if p.strip()]


def split_message(text):
    """Return a list of candidate task units from one raw message."""
    if not text or not text.strip():
        return []
    text = _strip_code(text)
    units = []
    # 2. newlines, then 3. list markers are handled per line in _clean,
    # but a list item can itself contain several sentences, so split
    # lines first, then sentences, then semicolons, then commas.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # A line may contain several list items if pasted without newlines;
        # that case is rare in the data and is left alone.
        for sent in _split_sentences(line):
            for semi in sent.split(";"):
                units.extend(_split_on_comma_imperative(semi))
    cleaned = []
    seen = set()
    for unit in units:
        piece = _clean(unit)
        if _is_junk(piece):
            continue
        # keep the first occurrence, drop exact repeats in one message
        key = piece.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(piece)
    return cleaned


if __name__ == "__main__":
    import sys
    candidates = [
        "Fix lighting layers, continue to tweak flocks, make sure to note any irregularities in the flocking so steve can help fix them, fix pilot ship guns",
        'Today we merge branches. Upgrade Input Manager to handle all the stuff Thea is doing. Build a left click interact "Load timer", and move to pixel perfect camera.',
        "the ui we made looks good but def too small. we can upscale it tomorrow. Want to focus on getting the larger bridge size functional, getting the squire functional.",
    ]
    for text in candidates:
        print("IN :", text)
        for i, unit in enumerate(split_message(text), 1):
            print(f"  {i}. {unit}")
        print()
    if len(sys.argv) > 1:
        print("ARG:", " ".join(sys.argv[1:]))
        for i, unit in enumerate(split_message(" ".join(sys.argv[1:])), 1):
            print(f"  {i}. {unit}")
