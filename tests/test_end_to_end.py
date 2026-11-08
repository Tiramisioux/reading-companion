"""The whole pipeline on synthetic pages (rendered here, no photos needed)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2

from reading_companion.imageio import read_stripe_meta
from reading_companion.pipeline import Options, process_folder

from .synthetic import make_page


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rc-test-"))
        pages = [
            make_page(highlight={5: "yellow", 6: "teal", 7: "teal"}, underline=(12,), margin_line=(16, 18), seed=1),
            make_page(highlight={3: "pink", 20: "orange"}, note_beside=10, angle=2.5, seed=2),
            make_page(seed=3),  # nothing marked
        ]
        for i, (img, _) in enumerate(pages, 1):
            cv2.imwrite(str(cls.tmp / f"IMG_{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cls.man = process_folder(cls.tmp, opts=Options(ocr="none", context=1))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def stripes_of(self, src):
        return [s for s in self.man["stripes"] if s["source"] == src]

    def test_page_one(self):
        st = self.stripes_of("IMG_1.jpg")
        self.assertEqual([tuple(s["lines_marked"]) for s in st], [(5, 7), (12, 12), (16, 18)])
        self.assertEqual([tuple(s["lines_shown"]) for s in st], [(4, 8), (11, 13), (15, 19)])
        self.assertEqual(st[0]["colors"][:2], ["yellow", "teal"])
        self.assertIn("underline", st[1]["kinds"])
        self.assertIn("margin_line", st[2]["kinds"])

    def test_page_two_rotated_with_note(self):
        st = self.stripes_of("IMG_2.jpg")
        marked = [tuple(s["lines_marked"]) for s in st]
        self.assertIn((3, 3), marked)
        self.assertIn((20, 20), marked)
        self.assertTrue(any("margin_note" in s["kinds"] for s in st), marked)

    def test_unmarked_page_gives_nothing(self):
        self.assertEqual(self.stripes_of("IMG_3.jpg"), [])

    def test_files_names_order_and_metadata(self):
        out = Path(self.man["output"])
        names = [s["file"] for s in self.man["stripes"]]
        self.assertEqual(names[0], "img_1_note1.jpg")
        for n in names:
            self.assertTrue((out / n).exists(), n)
        meta = read_stripe_meta(out / names[0])
        self.assertEqual((meta["source"], meta["note"]), ("IMG_1.jpg", 1))
        self.assertTrue((out / "manifest.json").exists() and (out / "index.html").exists())
        json.loads((out / "manifest.json").read_text())

    def test_rerun_replaces_previous_outputs(self):
        out = Path(self.man["output"])
        before = sorted(p.name for p in out.glob("*.jpg"))
        process_folder(self.tmp, opts=Options(ocr="none", context=1))
        after = sorted(p.name for p in out.glob("*.jpg"))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
