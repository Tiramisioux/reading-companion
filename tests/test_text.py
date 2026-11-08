"""The text of the marked words: which lines are read, which words of a line count, how lines join into a passage."""
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2

from reading_companion.bands import Band
from reading_companion.marks import Mark
from reading_companion.ocr import get_engine
from reading_companion.pipeline import Options, join_lines, marked_words, middle_words, process_folder, text_lines

from .synthetic import make_page
from .test_bands import P, hl, page


def mark(kind, line, x0=150, x1=850):
    yc = 100 + (line - 1) * P
    m = Mark(kind, x0, int(yc + 14), x1, int(yc + 18), "black", 40.0, 0.0, 0.0, (30, 30, 30), 400)
    m.lines = [line]
    return m


class TextLinesTest(unittest.TestCase):
    def test_margin_notes_and_context_are_not_read(self):
        h = hl(5)
        h.lines = [5]
        note = mark("margin_note", 7)
        line = mark("margin_line", 9)
        b = Band(5, 9, 4, 10, 0.0, 0.0, 0, 0, [h, note, line])
        self.assertEqual(text_lines(b, 20), [5, 9])

    def test_top_and_bottom_margin_lines_are_not_printed_lines(self):
        h = hl(1)
        h.lines = [0, 1]
        b = Band(0, 1, 1, 2, 0.0, 0.0, 0, 0, [h])
        self.assertEqual(text_lines(b, 20), [1])


class MarkedWordsTest(unittest.TestCase):
    # four words on line 5, in flat x (the image x is the flat x here: ox 0, kx 1); the line is 20 px high in the image
    WORDS = [("ex", 1.0, 200, 0, 260, 20), ("dolore", 1.0, 300, 0, 400, 20), ("dolor", 1.0, 410, 0, 500, 20),
             ("nulla", 1.0, 510, 0, 600, 20)]

    def line_page(self, *marks):
        an, geo = page(marks=marks)
        return an, geo, an.lines[4]

    def test_a_highlight_that_spills_onto_the_next_word_does_not_take_it(self):
        m = hl(5)
        m.x0, m.x1, m.lines = 300, 520, [5]          # covers "dolore dolor" and 10 of the 90 px of "nulla"
        an, geo, ln = self.line_page(m)
        an.tint[int(ln.center - 12):int(ln.center + 12), 300:520] = True
        self.assertEqual(marked_words(an, geo.pitch, ln, self.WORDS, 0.0, 1.0), "dolore dolor")

    def test_a_word_mostly_highlighted_counts(self):
        m = hl(5)
        m.x0, m.x1, m.lines = 227, 500, [5]          # 34 of the 60 px of "ex" (57 %) and all of the next two
        an, geo, ln = self.line_page(m)
        an.tint[int(ln.center - 12):int(ln.center + 12), 227:501] = True
        self.assertEqual(marked_words(an, geo.pitch, ln, self.WORDS, 0.0, 1.0), "dolore dolor")
        an.tint[int(ln.center - 12):int(ln.center + 12), 220:227] = True   # now 41 of 60 px (68 %)
        m.x0 = 220
        self.assertEqual(marked_words(an, geo.pitch, ln, self.WORDS, 0.0, 1.0), "ex dolore dolor")

    def test_the_words_above_an_underline(self):
        an, geo, ln = self.line_page(mark("underline", 5, 290, 505))
        self.assertEqual(marked_words(an, geo.pitch, ln, self.WORDS, 0.0, 1.0), "dolore dolor")

    def test_a_margin_line_marks_the_whole_line(self):
        an, geo, ln = self.line_page(mark("margin_line", 5, 900, 905))
        self.assertEqual(marked_words(an, geo.pitch, ln, self.WORDS, 0.0, 1.0), "ex dolore dolor nulla")

    def test_the_neighbour_lines_caught_in_the_crop_are_left_out(self):
        words = [("above", 1.0, 0, 0, 50, 4), ("middle", 1.0, 60, 8, 120, 32), ("line", 1.0, 130, 8, 170, 32),
                 ("below", 1.0, 0, 37, 50, 40)]
        self.assertEqual([w[0] for w in middle_words(words, 40)], ["middle", "line"])


class JoinTest(unittest.TestCase):
    def test_consecutive_lines_join_and_a_broken_word_is_mended(self):
        lines = [{"line": 4, "text": "the way-"}, {"line": 5, "text": "finding of"}, {"line": 6, "text": "Polynesia"}]
        self.assertEqual(join_lines(lines), "the wayfinding of Polynesia")

    def test_a_gap_between_marked_lines(self):
        self.assertEqual(join_lines([{"line": 4, "text": "one"}, {"line": 7, "text": "two"}]), "one … two")

    def test_a_dash_and_a_capital_stay(self):
        lines = [{"line": 1, "text": "north --"}, {"line": 2, "text": "and"}, {"line": 3, "text": "A-"}, {"line": 4, "text": "Bomb"}]
        self.assertEqual(join_lines(lines), "north -- and A- Bomb")


class EndToEndTextTest(unittest.TestCase):
    """The real OCR engine (Apple Vision on macOS, tesseract elsewhere) on a page with a highlight over two words that
    spills onto a third, a full-line highlight, an underline and a margin line."""

    @classmethod
    def setUpClass(cls):
        engine, _ = get_engine("auto")
        if engine.name == "none":
            raise unittest.SkipTest("no OCR engine")
        cls.tmp = Path(tempfile.mkdtemp(prefix="rc-text-"))
        _, truth = make_page(seed=1)
        w5 = [w for w in truth["words"] if w[0] == 5]
        cls.expected = [w5[1][1], w5[2][1]]
        cls.spill = w5[3][1]
        img, cls.truth = make_page(highlight={5: ("yellow", w5[1][2], w5[2][3] + 22), 9: "teal"}, underline=(14,),
                                   margin_line=(18, 19), seed=1)
        cv2.imwrite(str(cls.tmp / "IMG_1.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cls.man = process_folder(cls.tmp, opts=Options(ocr="auto", context=1, review=False))
        cls.by_line = {s["lines_marked"][0]: s for s in cls.man["stripes"]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_only_the_highlighted_words(self):
        s = self.by_line[5]
        self.assertEqual([t["line"] for t in s["text"]], [5])
        self.assertEqual(len(s["passage"].split()), 2, s["passage"])
        self.assertNotIn(self.spill, s["passage"].split())
        if self.man["ocr"]["engine"] == "vision":
            self.assertEqual(s["passage"].split(), self.expected)

    def test_margin_line_reads_both_lines_and_no_context(self):
        s = self.by_line[18]
        self.assertEqual([t["line"] for t in s["text"]], [18, 19])
        self.assertEqual(tuple(s["lines_shown"]), (17, 20))

    def test_the_passage_is_inside_the_stripe(self):
        from reading_companion.imageio import read_stripe_meta
        meta = read_stripe_meta(Path(self.man["output"]) / self.by_line[9]["file"])
        self.assertEqual(meta["passage"], self.by_line[9]["passage"])
        self.assertTrue(meta["passage"])


class NoOcrTest(unittest.TestCase):
    def test_without_ocr_there_is_no_text(self):
        tmp = Path(tempfile.mkdtemp(prefix="rc-notext-"))
        try:
            img, _ = make_page(highlight={5: "yellow"}, seed=2)
            cv2.imwrite(str(tmp / "IMG_1.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            man = process_folder(tmp, opts=Options(ocr="none", review=False))
            self.assertIsNone(man["stripes"][0]["text"])
            self.assertIsNone(man["stripes"][0]["passage"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
