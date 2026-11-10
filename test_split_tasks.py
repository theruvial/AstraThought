import unittest

from split_tasks import split_message


class MultiTaskSplitTests(unittest.TestCase):
    def test_and_before_an_imperative_with_a_determiner_splits(self):
        self.assertEqual(
            split_message("add a mute button to the lobby and fix the nav freeze"),
            ["add a mute button to the lobby", "fix the nav freeze"])

    def test_two_commands_with_determiners_split(self):
        self.assertEqual(len(split_message("set the volume and use the default mixer")), 2)

    def test_also_and_then_split(self):
        self.assertEqual(
            split_message("add a mute button to the lobby also fix the nav freeze"),
            ["add a mute button to the lobby", "fix the nav freeze"])
        self.assertEqual(
            split_message("fix the nav freeze then add a ping to the map"),
            ["fix the nav freeze", "add a ping to the map"])

    def test_and_also_splits(self):
        self.assertEqual(
            split_message("fix the nav freeze and also add a ping to the map"),
            ["fix the nav freeze", "add a ping to the map"])

    def test_two_statements_split_on_and_or_comma(self):
        expected = ["doors should close slower", "shields should recharge slower"]
        self.assertEqual(
            split_message("doors should close slower and shields should recharge slower"), expected)
        self.assertEqual(
            split_message("doors should close slower, shields should recharge slower"), expected)

    def test_one_line_numbered_list_loses_its_digits(self):
        self.assertEqual(
            split_message("1. add a ping 2. fix the freeze 3. quieter alarm"),
            ["add a ping", "fix the freeze", "quieter alarm"])

    def test_single_tasks_stay_whole(self):
        for text in (
            "add npcs and items to the crew UI",
            "design a flamethrower that spreads fire, has high damage, and slows the player",
            "doors close slower and they should be louder",
            "hearts should not be centered, they should be left aligned",
            "asteroids should purely from right to left and move slower",
            "the nav freezes and it needs to be fixed",
            "we should add a ping and we need a map",
            "version 2.5 adds a ping",
        ):
            self.assertEqual(len(split_message(text)), 1, text)

    def test_a_statement_pair_needs_both_halves_to_be_statements(self):
        text = "doors close slower and shields should recharge slower"
        self.assertEqual(len(split_message(text)), 1)

    def test_existing_rules_still_work(self):
        self.assertEqual(
            split_message("Fix lighting layers, make sure to note irregularities, fix pilot ship guns"),
            ["Fix lighting layers", "make sure to note irregularities", "fix pilot ship guns"])
        self.assertEqual(split_message("a\nb c d\n- add a ping"), ["b c d", "add a ping"])


if __name__ == "__main__":
    unittest.main()
