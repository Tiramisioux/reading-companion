"""Fingers holding the book: never a mark (not even when a thumb touches a highlighted line), painted out of the stripes; and a
coloured pen underlining three lines in a row keeps the middle line."""
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from reading_companion.pipeline import Options, process_folder

from .synthetic import make_page


def run(**kw):
    tmp = Path(tempfile.mkdtemp(prefix="rc-finger-"))
    img, truth = make_page(**kw)
    cv2.imwrite(str(tmp / "IMG_1.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    man = process_folder(tmp, opts=Options(ocr="none", review=False))
    return tmp, man, truth


class FingerTest(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_finger_alone_is_no_mark(self):
        self.tmp, man, _ = run(highlight={5: "yellow"}, finger=(12, 15), seed=5)
        page = man["pages"][0]
        self.assertEqual(len(page["fingers"]), 1)
        self.assertEqual([m["lines"] for m in page["marks"]], [[5]])
        self.assertEqual([s["lines_marked"] for s in man["stripes"]], [[5, 5]])

    def test_a_thumb_on_a_highlighted_line_leaves_the_highlight_and_is_painted_out(self):
        self.tmp, man, truth = run(highlight={5: "yellow"}, finger=(4, 7), seed=5)
        page = man["pages"][0]
        self.assertEqual(len(page["fingers"]), 1)
        self.assertEqual([(m["kind"], m["lines"]) for m in page["marks"]], [("highlight", [5])])
        stripe = cv2.cvtColor(cv2.imread(str(Path(man["output"]) / man["stripes"][0]["file"])), cv2.COLOR_BGR2LAB).astype(np.float32)
        right = stripe[:, int(stripe.shape[1] * 0.9):]                 # where the finger was: paper again, not skin
        chroma = np.hypot(right[..., 1] - 128, right[..., 2] - 128)
        self.assertLess(float(np.median(chroma)), 12.0)

    def test_three_lines_underlined_in_red_are_one_passage(self):
        self.tmp, man, _ = run(underline=(9, 10, 11), underline_color=(200, 30, 40), seed=5)
        self.assertEqual([s["lines_marked"] for s in man["stripes"]], [[9, 11]])
        self.assertEqual(man["pages"][0]["body_lines"], 24)


if __name__ == "__main__":
    unittest.main()
