"""Tests for check_synthetic.py. No model or network needed.

Run with:  python -m unittest test_check_synthetic
"""

import json
import tempfile
import unittest
from pathlib import Path

import check_synthetic as cs


def line(text, label=1, reason="task_requirement"):
    return json.dumps({"text": text, "label": label, "reason": reason})


class ParseTests(unittest.TestCase):
    def test_fences_headings_and_chatter_are_ignored(self):
        body = "\n".join([
            "# batch 1", "```json", line("shields should recharge slower"), "```",
            "Here are your lines!", line("thanks for the quick fix", 0, "chat"),
        ])
        rows, rejects, ignored = cs.parse_lines(body)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rejects, [])
        self.assertEqual(ignored, 4)

    def test_task_rows_get_reason_task_and_keep_their_kind(self):
        rows, _, _ = cs.parse_lines(line("add a ping to the star map", 1, "task_imperative"))
        self.assertEqual(rows[0]["reason"], "task")
        self.assertEqual(rows[0]["kind"], "task_imperative")
        self.assertEqual(rows[0]["source"], "synthetic")
        self.assertFalse(rows[0]["reviewed"])

    def test_bad_lines_are_rejected_with_a_reason(self):
        cases = {
            "not json {": "not valid JSON",
            line("ok", 1, "task_bug"): "words",
            line("this has a label that is wrong", 2, "chat"): "label",
            line("a task with a chat reason here", 1, "chat"): "does not fit",
            line("an unknown reason for this line", 0, "banter"): "unknown reason",
            line("ping @someone about the build", 0, "request"): "@",
            line(" ".join(["word"] * 31), 0, "chat"): "words",
        }
        for text, expected in cases.items():
            rows, rejects, _ = cs.parse_lines(text if text.startswith("{") else "{" + text)
            self.assertEqual(rows, [], text)
            self.assertIn(expected, rejects[0][1], text)

    def test_boolean_labels_are_rejected(self):
        body = json.dumps({"text": "doors should close slower", "label": True,
                           "reason": "task_requirement"})
        rows, rejects, _ = cs.parse_lines(body)
        self.assertEqual(rows, [])


class DedupeTests(unittest.TestCase):
    def _rows(self, *texts):
        return [{"text": t, "label": 1, "reason": "task", "kind": "task_requirement"} for t in texts]

    def test_exact_and_near_repeats_are_dropped(self):
        rows = self._rows("Doors should close slower", "doors should close slower!",
                          "doors should close a bit slower", "engines need a second door")
        kept, dropped = cs.drop_repeats(rows)
        self.assertEqual([r["text"] for r in kept],
                         ["Doors should close slower", "engines need a second door"])
        self.assertEqual(dropped, 2)

    def test_a_single_swapped_word_counts_as_a_repeat(self):
        rows = self._rows("shields should recharge slower after a hit",
                          "shields should recharge faster after a hit")
        kept, dropped = cs.drop_repeats(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 1)

    def test_short_lines_only_repeat_when_identical(self):
        kept, _ = cs.drop_repeats(self._rows("not sure", "not sure about that one"))
        self.assertEqual(len(kept), 2)

    def test_different_sentences_survive(self):
        kept, dropped = cs.drop_repeats(self._rows(
            "shields should recharge slower", "add a mute button to the lobby"))
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)


class LeakTests(unittest.TestCase):
    def test_rows_matching_reference_data_are_dropped_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "eval_feedback.jsonl").write_text(
                json.dumps({"text": "Every room should have a main module", "label": 1}) + "\n",
                encoding="utf-8")
            (root / "train.jsonl").write_text(
                json.dumps({"text": "donut progress bar for cooking", "label": 1}) + "\n",
                encoding="utf-8")
            reference = cs.load_reference(root)
            rows = [{"text": t} for t in (
                "every room should have a main module",
                "Donut progress bar for cooking",
                "shields should recharge slower")]
            kept, hits = cs.drop_leaks(rows, reference)
            self.assertEqual([r["text"] for r in kept], ["shields should recharge slower"])
            self.assertEqual(hits["eval_feedback.jsonl"], 1)
            self.assertEqual(hits["train.jsonl"], 1)

    def test_missing_reference_files_are_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(cs.load_reference(Path(tmp)),
                             {"train.jsonl": [], "test.jsonl": [], "eval_feedback.jsonl": []})


class AgreementTests(unittest.TestCase):
    KEY = {
        "1": {"text": "a", "label": 1, "kind": "task_bug"},
        "2": {"text": "b", "label": 0, "kind": "chat"},
        "3": {"text": "c", "label": 1, "kind": "task_feature"},
        "4": {"text": "d", "label": 0, "kind": "suggestion"},
    }

    def test_full_agreement(self):
        stats = cs.agreement(self.KEY, {2, 4})
        self.assertEqual(stats["agree"], 4)
        self.assertEqual(stats["disagreements"], [])

    def test_disagreements_are_listed(self):
        stats = cs.agreement(self.KEY, {1, 2, 4})
        self.assertEqual(stats["agree"], 3)
        self.assertEqual([d[0] for d in stats["disagreements"]], [1])
        self.assertEqual(stats["per_total"][1], 2)


class EndToEndTests(unittest.TestCase):
    def test_clean_writes_the_three_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp) / "batch01.md"
            batch.write_text("```\n" + "\n".join([
                line("shields should recharge slower"),
                line("thanks for the quick fix", 0, "chat"),
                line("add a mute button to the lobby", 1, "task_imperative"),
            ]) + "\n```\n", encoding="utf-8")
            out = Path(tmp) / "out"
            self.assertEqual(cs.clean([batch], out, root=Path(tmp)), 0)
            rows = [json.loads(l) for l in (out / "synthetic_clean.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue((out / "synthetic_review_sample.txt").is_file())
            key = json.loads((out / "synthetic_review_key.json").read_text(encoding="utf-8"))
            self.assertEqual(len(key), 3)


if __name__ == "__main__":
    unittest.main()
