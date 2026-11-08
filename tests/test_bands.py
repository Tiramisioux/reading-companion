import unittest

import numpy as np

from reading_companion.bands import assign_lines, make_bands
from reading_companion.geometry import DewarpField, Geometry, TextBlock
from reading_companion.marks import Layout, Mark, PageAnalysis, TextLine

P = 50.0


def page(n=20, marks=()):
    h, w = 1400, 1000
    lines = [TextLine(i + 1, 100 + i * P, 100 + i * P - P / 2, 100 + i * P + P / 2, 0.2) for i in range(n)]
    an = PageAnalysis(np.zeros((h, w, 3), np.uint8), np.full((h, w), 95.0), np.zeros((h, w)), np.zeros((h, w)),
                      np.full((h, w), 90.0), np.ones((h, w), bool), np.ones((h, w), bool), np.zeros((h, w), bool),
                      np.zeros((h, w), bool))
    an.lines = lines
    an.layout = Layout(np.full(h, 50.0), np.full(h, 950.0), np.full(h, 100.0), np.full(h, 900.0), 0, h - 1)
    an.marks = list(marks)
    geo = Geometry((h, w), 1.0, 0.0, np.eye(2, 3), np.eye(2, 3), (w, h), DewarpField(500.0, np.array([]), np.zeros((0, 3))),
                   TextBlock(100, 900, 75, 1100), P)
    return an, geo


def hl(line, color="yellow"):
    yc = 100 + (line - 1) * P
    m = Mark("highlight", 150, int(yc - 12), 850, int(yc + 12), color, 100.0, 40.0, 90.0, (250, 230, 100), 1000)
    return m


def tint_for(an, marks):
    for m in marks:
        an.tint[m.y0:m.y1, m.x0:m.x1] = True


class BandsTest(unittest.TestCase):
    def run_bands(self, marks, context=0):
        an, geo = page(marks=marks)
        tint_for(an, [m for m in marks if m.kind == "highlight"])
        assign_lines(an, P)
        return make_bands(an, geo, context=context)

    def test_two_colours_on_consecutive_lines_are_one_stripe(self):
        bands = self.run_bands([hl(4, "yellow"), hl(5, "teal"), hl(6, "teal")])
        self.assertEqual([(b.first_line, b.last_line) for b in bands], [(4, 6)])
        self.assertEqual(bands[0].colors, ["yellow", "teal"])

    def test_one_unmarked_line_separates(self):
        bands = self.run_bands([hl(4), hl(6)])
        self.assertEqual([(b.first_line, b.last_line) for b in bands], [(4, 4), (6, 6)])

    def test_context_lines(self):
        bands = self.run_bands([hl(4), hl(5)], context=1)
        self.assertEqual([(b.ctx_first, b.ctx_last) for b in bands], [(3, 6)])

    def test_context_at_page_edges(self):
        bands = self.run_bands([hl(1), hl(20)], context=1)
        self.assertEqual([(b.ctx_first, b.ctx_last) for b in bands], [(1, 2), (19, 20)])

    def test_underline_belongs_to_line_above_and_joins_next_highlight(self):
        yc = 100 + 6 * P  # line 7 centre
        ul = Mark("underline", 200, int(yc + 0.45 * P), 600, int(yc + 0.5 * P), "black", 0, 2.0, 25.0, (30, 30, 30), 400)
        bands = self.run_bands([ul, hl(8, "teal")])
        self.assertEqual([(b.first_line, b.last_line) for b in bands], [(7, 8)])

    def test_margin_line_spans_its_lines(self):
        y0, y1 = 100 + 9 * P - 20, 100 + 12 * P + 20
        ml = Mark("margin_line", 920, int(y0), 926, int(y1), "red", 25.0, 45.0, 40.0, (200, 30, 40), 600)
        bands = self.run_bands([ml])
        self.assertEqual([(b.first_line, b.last_line) for b in bands], [(10, 13)])

    def test_note_in_bottom_margin_joins_last_line_with_context(self):
        note = Mark("margin_note", 300, int(100 + 19 * P + 40), 600, int(100 + 19 * P + 90), "black", 0, 2.0, 25.0, (30, 30, 30), 900)
        note.where = "bottom"
        bands = self.run_bands([hl(20), note], context=1)
        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].ctx_first, bands[0].ctx_last), (19, 20))
        self.assertGreaterEqual(bands[0].y1, note.y1)

    def test_full_page_width(self):
        bands = self.run_bands([hl(4)])
        self.assertEqual((bands[0].x0, bands[0].x1), (50, 950))


if __name__ == "__main__":
    unittest.main()
