"""Printed things that are no mark of the reader give no stripe: a table's rules, a figure's line art, its caption.
Found as false positives on real photos (tests/fixtures/wayfinding_labels.json); the pages here are synthetic."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

from reading_companion.ocr import OcrEngine
from reading_companion.pipeline import Options, process_page

from .synthetic import make_page

ROOT = Path(__file__).resolve().parents[1]


class PrintedIsNotAMarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rc-printed-"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_page(self, name, **kw):
        img, truth = make_page(**kw)
        path = self.tmp / f"{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        r = process_page(path, 1, OcrEngine(), Options(context=1, review=False, text=False))
        self.assertIsNone(r.error)
        return r

    def test_table_rules_are_not_underlines(self):
        r = self.run_page("table", rules=(8, 11, 14, 17), seed=4)
        self.assertEqual([(b.first_line, b.kinds) for b in r.bands], [])
        self.assertTrue(any(m.why == "a printed rule" for m in r.analysis.rejected))

    def test_figure_and_caption_give_nothing(self):
        r = self.run_page("figure", figure=(6, 14), caption=True, seed=5)
        self.assertEqual([(b.first_line, b.kinds) for b in r.bands], [])
        self.assertEqual(len(r.analysis.figures), 1)

    def test_highlight_beside_a_figure_is_kept(self):
        r = self.run_page("figure_hl", figure=(6, 14), caption=True, highlight={18: "yellow"}, seed=6)
        self.assertEqual([b.kinds for b in r.bands], [["highlight"]])
        self.assertEqual(r.bands[0].colors, ["yellow"])

    def test_text_is_not_a_figure(self):
        # a page of text with a pen margin line and underlines: its lines' gaps are empty, so no figure
        r = self.run_page("text", underline=(5, 6, 7), margin_line=(10, 14), seed=7)
        self.assertEqual(r.analysis.figures, [])
        kinds = sorted({k for b in r.bands for k in b.kinds})
        self.assertEqual(kinds, ["margin_line", "underline"])


class LabelScorerTest(unittest.TestCase):
    """tools/score_labels.py: matching stripes by their outlines (no photos needed)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tools"))
        import score_labels
        cls.sl = score_labels

    @staticmethod
    def stripe(y0, y1, x0=0.0, x1=1000.0, n=4):
        xs = [x0 + (x1 - x0) * k / (n - 1) for k in range(n)]
        return [[x, y0] for x in xs] + [[x, y1] for x in reversed(xs)]

    def test_vertical_overlap(self):
        a = self.stripe(100, 300)
        self.assertAlmostEqual(self.sl.vertical_overlap(a, a)[0], 1.0, places=6)
        iou, held_a, held_b = self.sl.vertical_overlap(a, self.stripe(200, 400))
        self.assertAlmostEqual(iou, 100 / 300, places=6)
        self.assertAlmostEqual(held_a, 0.5, places=6)
        self.assertEqual(self.sl.vertical_overlap(a, self.stripe(100, 300, 2000, 3000))[0], 0.0)

    def test_match_one_to_one_and_merged(self):
        labels = [{"source_polygon": self.stripe(100, 300)}, {"source_polygon": self.stripe(320, 420)}]
        produced = [{"source_polygon": self.stripe(90, 440)}, {"source_polygon": self.stripe(900, 1000)}]
        matched, merged, unmatched = self.sl.match_photo(labels, produced, 0.5, 0.8)
        self.assertEqual([(i, j) for i, j, _ in matched], [(0, 0)])
        self.assertEqual([(i, j) for i, j, _ in merged], [(1, 0)])
        self.assertEqual(unmatched, [1])

    def test_categories_cover_the_labels(self):
        labels = json.loads((ROOT / "tests/fixtures/wayfinding_labels.json").read_text(encoding="utf-8"))["stripes"]
        cats = json.loads((ROOT / "tests/fixtures/wayfinding_fp_categories.json").read_text(encoding="utf-8"))
        fps = {s["id"] for s in labels if s["label"] == "false_positive"}
        tps = {s["id"] for s in labels if s["label"] == "true_positive"}
        self.assertEqual(set(cats["categories"]), fps)
        self.assertTrue(set(cats["tp_without_mark"]) <= tps)


if __name__ == "__main__":
    unittest.main()
