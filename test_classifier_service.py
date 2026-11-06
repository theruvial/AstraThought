"""Tests for the decision rules. No model needed: scores are passed in.

Run with the venv active:  python -m unittest test_classifier_service
"""

import unittest

import classifier_service as service


def decide(text, score, raw=None):
    return service.decide_piece(text, score, raw if raw is not None else text)


class ReviewFloorTests(unittest.TestCase):
    def test_chat_under_the_floor_is_dropped(self):
        for text in ("sounds good to me", "lol nice one", "ok", "thanks"):
            result = decide(text, 0.007)
            self.assertEqual(result["decision"], "auto_drop", text)
            self.assertEqual(result["trigger"], "below_review_floor")

    def test_the_floor_also_covers_a_rule_disagreement(self):
        self.assertEqual(decide("check this out", 0.05)["decision"], "auto_drop")

    def test_at_the_floor_still_reviews(self):
        self.assertEqual(decide("sounds good to me", 0.10)["decision"], "review")

    def test_problem_words_still_reach_review(self):
        for text in ("salvage is broken", "the map crashes", "door sound missing", "doors not working"):
            self.assertEqual(decide(text, 0.007)["decision"], "review", text)

    def test_explicit_markers_still_reach_review(self):
        self.assertEqual(decide("fix the map", 0.01, "TODO: fix the map")["decision"], "review")
        self.assertEqual(decide(".open doorFix fix it", 0.01)["decision"], "review")

    def test_problem_words_match_whole_words_only(self):
        self.assertTrue(service.below_review_floor("nice debugging session", 0.01))


class UnchangedRulesTests(unittest.TestCase):
    def test_confident_tasks_still_auto_post(self):
        self.assertEqual(decide("add npcs to the crew UI", 0.95)["decision"], "auto_task")

    def test_model_and_rule_disagreement_still_reviews(self):
        result = decide("the cargo bay door seems odd today", 0.99)
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["trigger"], "model_regex_disagreement")

    def test_short_observations_over_the_floor_still_review(self):
        self.assertEqual(decide("asteroid field looks odd", 0.3)["trigger"], "short_observation")


if __name__ == "__main__":
    unittest.main()
