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


class StatementNetTests(unittest.TestCase):
    def test_short_requirement_statements_go_to_review_not_the_bin(self):
        for text in ("Doors should close slower after a warp",
                     "every ship needs a spare engine",
                     "Not every sector needs a shop",
                     "shields must not recharge during a jump"):
            self.assertEqual(decide(text, 0.007)["decision"], "review", text)

    def test_a_long_statement_gets_its_own_trigger(self):
        text = "cameras that get destroyed should make it so you only see your own side of the ship"
        result = decide(text, 0.007)
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["trigger"], "statement_net")

    def test_a_long_problem_word_piece_is_reviewed_not_dropped(self):
        text = "the door sound is missing in the cargo bay when you jump through the gate"
        result = decide(text, 0.007)
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["trigger"], "floor_exception")

    def test_first_person_questions_and_short_lines_are_still_dropped(self):
        for text in ("we should maybe look at fog sometime",
                     "I think every room needs a window",
                     "should doors close slower?",
                     "every day lol",
                     "thanks for the quick fix everyone",
                     "that boss was brutal"):
            result = decide(text, 0.007)
            self.assertEqual(result["decision"], "auto_drop", text)
            self.assertEqual(result["trigger"], "below_review_floor")

    def test_above_the_floor_the_net_does_nothing(self):
        result = decide("Doors should close slower after a warp today please", 0.30)
        self.assertNotIn(result["trigger"], ("statement_net", "floor_exception"))

    def test_statement_check_itself(self):
        self.assertTrue(service.is_requirement_statement("Modules should have 2 to 3 hp"))
        self.assertFalse(service.is_requirement_statement("should have hp"))
        self.assertFalse(service.is_requirement_statement("Should doors close slower?"))
        self.assertFalse(service.is_requirement_statement("we need every door fixed"))
        self.assertFalse(service.is_requirement_statement("the door closed fast"))


class RequirementStatementAgreementTests(unittest.TestCase):
    def test_a_confident_requirement_statement_is_auto_posted(self):
        result = decide("Modules should have 2 to 3 hp", 0.76)
        self.assertEqual(result["decision"], "auto_task")

    def test_the_rule_counts_a_requirement_statement_as_a_task(self):
        self.assertTrue(service.rule_says_task("Doors should close slower after a warp"))
        self.assertFalse(service.rule_says_task("we should maybe look at fog sometime"))

    def test_a_first_person_suggestion_still_goes_to_review(self):
        result = decide("we should maybe add more hp to modules", 0.9)
        self.assertEqual(result["decision"], "review")


class BugReportTests(unittest.TestCase):
    def test_short_bug_reports_are_auto_posted_when_the_model_is_sure(self):
        for text in ("Bug in ship sway", "found bug in x feature",
                     "found a bug in the lobby menu", "Noticed the shield bar is missing",
                     "there's a glitch in the warp screen"):
            self.assertEqual(decide(text, 0.97)["decision"], "auto_task", text)

    def test_the_bug_check_itself(self):
        self.assertTrue(service.is_bug_report("Bug in ship sway"))
        self.assertTrue(service.is_bug_report("found bug in x feature"))
        self.assertTrue(service.is_bug_report("I found a bug in the map"))
        self.assertFalse(service.is_bug_report("is this a bug?"))
        self.assertFalse(service.is_bug_report("I'm stuck on level three"))
        self.assertFalse(service.is_bug_report("we should fix the bug someday"))
        self.assertFalse(service.is_bug_report("that boss was brutal"))
        self.assertFalse(service.is_bug_report("bug"))

    def test_a_bug_report_the_model_doubts_still_goes_to_review(self):
        result = decide("Bug in ship sway", 0.4)
        self.assertEqual(result["decision"], "review")


class UnchangedRulesTests(unittest.TestCase):
    def test_confident_tasks_still_auto_post(self):
        self.assertEqual(decide("add npcs to the crew UI", 0.95)["decision"], "auto_task")

    def test_model_and_rule_disagreement_still_reviews_below_95(self):
        text = "the cargo bay door looks too narrow next to the other doors on deck"
        result = decide(text, 0.90)
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["trigger"], "model_regex_disagreement")

    def test_rule_saying_task_against_a_low_model_score_still_reviews(self):
        result = decide("add a thing", 0.30)
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["trigger"], "model_regex_disagreement")


class HighScoreOutvotesRuleTests(unittest.TestCase):
    OBSERVATION = "the jump gate is still visible on the second node after the warp flash"

    def test_confident_model_outvotes_a_rule_that_says_not_a_task(self):
        result = decide(self.OBSERVATION, 0.97)
        self.assertEqual(result["decision"], "auto_task")
        self.assertEqual(result["trigger"], "none")

    def test_the_cutoff_is_inclusive(self):
        self.assertEqual(decide(self.OBSERVATION, service.DISAGREEMENT_SKIP_SCORE)["decision"], "auto_task")
        just_under = service.DISAGREEMENT_SKIP_SCORE - 0.001
        self.assertEqual(decide(self.OBSERVATION, just_under)["trigger"], "model_regex_disagreement")

    def test_other_triggers_still_apply_to_an_outvoted_piece(self):
        # Short and observational: still goes to review, for its own reason.
        self.assertEqual(decide("the cargo bay door seems off today", 0.99)["trigger"], "short_observation")
        # A suggestion phrase: still goes to review.
        text = "we should make the jump gate a little more visible in the ship view"
        self.assertEqual(decide(text, 0.99)["trigger"], "suggestion_marker")

    def test_a_human_marker_is_unaffected(self):
        # The rule says task, so there is nothing to outvote.
        self.assertEqual(decide("fix the map", 0.99, "TODO: fix the map")["decision"], "auto_task")

    def test_short_observations_over_the_floor_still_review(self):
        self.assertEqual(decide("asteroid field looks odd", 0.3)["trigger"], "short_observation")


if __name__ == "__main__":
    unittest.main()
