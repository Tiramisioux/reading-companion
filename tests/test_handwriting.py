"""The reader's handwriting is found where it was missed on real photos (tests/fixtures/wayfinding_labels.json: a sweep
of the photos found notes on 19 of them that 0.4.0 did not see). Synthetic pages, no photos needed."""
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from reading_companion.ocr import OcrEngine
from reading_companion.pipeline import Options, process_page

from .synthetic import make_page


class HandwritingFoundTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rc-hand-"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_page(self, name, **kw):
        img, truth = make_page(**kw)
        path = self.tmp / f"{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        r = process_page(path, 1, OcrEngine(), Options(context=1, review=False, text=False))
        self.assertIsNone(r.error)
        return r, truth

    def test_a_line_written_under_the_text_is_a_note_not_a_printed_line(self):
        # handwriting has letter texture too: it used to be taken for a 25th printed line, and then it was text
        r, _ = self.run_page("below", note_below=True, seed=8)
        self.assertEqual(len(r.analysis.lines), 24)
        self.assertEqual([(b.kinds, [m.where for m in b.marks]) for b in r.bands], [(["margin_note"], ["bottom"])])

    def test_a_note_close_to_the_page_edge_is_found_and_the_stripe_holds_it(self):
        r, _ = self.run_page("right", note_right=10, seed=9)
        notes = [m for m in r.analysis.marks if m.kind == "margin_note"]
        self.assertEqual(len(notes), 1)
        self.assertIn(10, notes[0].lines)
        b = r.bands[0]
        self.assertGreaterEqual(b.x1, notes[0].x1)          # the stripe reaches past the note, to the paper's edge

    def test_a_note_written_up_the_margin_is_one_note_beside_its_lines(self):
        r, _ = self.run_page("up", note_up=(6, 16), seed=10)
        self.assertEqual(len(r.bands), 1)
        self.assertEqual(r.bands[0].kinds, ["margin_note"])
        self.assertLessEqual(r.bands[0].first_line, 7)
        self.assertGreaterEqual(r.bands[0].last_line, 15)


class PenBlackTest(unittest.TestCase):
    def test_pure_black_pen_pixels_are_ink_of_a_note(self):
        """A phone renders a pen's deepest black as 0; such pixels used to fall outside the photo (holes in ``valid``)
        and a note could vanish from its own ink."""
        from reading_companion.marks import fill_holes
        m = np.ones((40, 40), bool)
        m[10:20, 10:12] = False                             # pure black pen stroke inside the photo
        m[:, :3] = False                                    # the border left by the rotation
        f = fill_holes(m)
        self.assertTrue(f[10:20, 10:12].all())
        self.assertFalse(f[:, :3].any())


if __name__ == "__main__":
    unittest.main()
