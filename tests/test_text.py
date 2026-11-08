import unittest

from reading_companion.text import join_text, keep_word, split_glued


class TextTest(unittest.TestCase):
    def test_hyphen_mended_across_lines(self):
        self.assertEqual(join_text(["our ability to tran-", "scend this robot"]), "our ability to transcend this robot")
        self.assertEqual(join_text(["long-term", "memories"]), "long-term memories")
        self.assertEqual(join_text(["ends with a dash -", "Next"]), "ends with a dash - Next")

    def test_glued_footnote_and_next_sentence_split(self):
        parts = [w[0] for w in split_glued([("truth’.He", 0.9, 0, 0, 90, 20)])]
        self.assertEqual(parts, ["truth’.", "He"])
        parts = split_glued([("deterioration’.%Many", 0.5, 100, 0, 300, 20)])
        self.assertEqual([w[0] for w in parts], ["deterioration’.", "%Many"])
        self.assertEqual(parts[0][4], parts[1][2])  # boxes share out the width

    def test_keep_word(self):
        self.assertTrue(keep_word("behaviour", 0.12))      # mostly letters: kept even when unsure
        self.assertFalse(keep_word("~", 0.9))               # highlighter end mark
        self.assertFalse(keep_word("|", 0.9))
        self.assertFalse(keep_word("%2", 0.2))


if __name__ == "__main__":
    unittest.main()
