import tempfile
import unittest
from pathlib import Path

from reading_companion.naming import list_images, natural_key, slug, stripe_name, unique, DEFAULT_TEMPLATE


class NamingTest(unittest.TestCase):
    def test_natural_order(self):
        names = ["IMG_10.jpg", "IMG_2.jpg", "IMG_1.jpg", "img_3.JPG"]
        self.assertEqual(sorted(names, key=natural_key), ["IMG_1.jpg", "IMG_2.jpg", "img_3.JPG", "IMG_10.jpg"])

    def test_default_name_with_everything(self):
        self.assertEqual(stripe_name(DEFAULT_TEMPLATE, "IMG_2001", "10", "202", 1, 3, 7), "img_2001_chapter_10_p_202_note1.jpg")

    def test_missing_parts_are_left_out(self):
        self.assertEqual(stripe_name(DEFAULT_TEMPLATE, "IMG_2001", "", "224", 2, 2, 1), "img_2001_p_224_note2.jpg")
        self.assertEqual(stripe_name(DEFAULT_TEMPLATE, "IMG_2001", "", "", 1, 1, 1), "img_2001_note1.jpg")
        self.assertEqual(stripe_name(DEFAULT_TEMPLATE, "Scan 3", "epilogue", "", 1, 1, 1), "scan_3_chapter_epilogue_note1.jpg")

    def test_custom_template(self):
        self.assertEqual(stripe_name("{seq}_{chapter}_{page}_{note}", "x", "", "12", 1, 1, 5), "0005_x_12_1.jpg")

    def test_slug(self):
        self.assertEqual(slug("Am I Here?"), "am_i_here")
        self.assertEqual(slug("Från A till Ö"), "från_a_till_ö")

    def test_unique(self):
        taken = set()
        self.assertEqual(unique("a.jpg", taken), "a.jpg")
        self.assertEqual(unique("a.jpg", taken), "a_2.jpg")

    def test_list_images_skips_output_and_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for n in ["2.jpg", "10.jpg", "1.jpg", ".hidden.jpg", "notes.txt"]:
                (d / n).write_bytes(b"x")
            (d / "reading-companion").mkdir()
            (d / "reading-companion" / "1_note1.jpg").write_bytes(b"x")
            got = [p.name for p in list_images(d, exclude=[d / "reading-companion"])]
            self.assertEqual(got, ["1.jpg", "2.jpg", "10.jpg"])


if __name__ == "__main__":
    unittest.main()
