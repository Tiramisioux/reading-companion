"""Writing stripes, cleaning up after a run, odd inputs and the command line contract."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, IptcImagePlugin

from reading_companion.imageio import load_rgb, read_stripe_meta, save_stripe
from reading_companion.pipeline import _clean_previous

PKG_ROOT = Path(__file__).resolve().parents[1]


class CaptionTest(unittest.TestCase):
    def test_caption_in_iptc_xmp_and_exif(self):
        cap = "\u2018The future of our species depends on our ability\u2019 \u2013 \u00d6ver gr\u00e4nsen"
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x_note1.jpg"
            save_stripe(f, np.full((40, 120, 3), 230, np.uint8), "Book - chapter 3 - p. 42 - note 1/1",
                        {"source": "IMG_1.jpg", "note": 1, "chapter": {"number": "3", "title": "\u00d6ver gr\u00e4nsen"}},
                        "reading-companion test", caption=cap, keywords=["Book", "chapter 3", "p. 42"])
            with Image.open(f) as im:
                iptc = IptcImagePlugin.getiptcinfo(im)
                desc = im.getexif().get(0x010E)
                xmp = im.info["xmp"].decode("utf-8")
            self.assertEqual(iptc[(2, 120)].decode("utf-8"), cap)
            self.assertEqual(iptc[(2, 105)].decode("utf-8"), "Book - chapter 3 - p. 42 - note 1/1")
            # stored as UTF-8 (as exiftool writes and reads it); Pillow hands ASCII tags back as Latin-1
            desc = desc if isinstance(desc, bytes) else desc.encode("latin-1")
            self.assertEqual(desc.decode("utf-8"), cap)
            self.assertIn(cap, xmp)
            self.assertEqual(read_stripe_meta(f)["chapter"]["title"], "\u00d6ver gr\u00e4nsen")

    def test_foreign_user_comment_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "other.jpg"
            ex = Image.Exif()
            ex.get_ifd(0x8769)[0x9286] = b"ASCII\x00\x00\x00not json"
            Image.new("RGB", (8, 8)).save(f, exif=ex.tobytes())
            self.assertEqual(read_stripe_meta(f), {})


class CleanupTest(unittest.TestCase):
    def test_tampered_manifest_cannot_delete_outside(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            out = d / "out"
            (out / "review").mkdir(parents=True)
            (d / "victim.jpg").write_bytes(b"keep")
            (out / "a_note1.jpg").write_bytes(b"x")
            (out / "manifest.json").write_text(json.dumps({
                "stripes": [{"file": "a_note1.jpg", "source": "a.jpg"}, {"file": "../victim.jpg", "source": "b.jpg"}],
                "pages": [{"file": "a.jpg", "review": "review/../../victim.jpg"}]}))
            _clean_previous(out)
            self.assertFalse((out / "a_note1.jpg").exists())
            self.assertTrue((d / "victim.jpg").exists())

    def test_malformed_manifest_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "manifest.json").write_text("[1, 2, 3]")
            _clean_previous(Path(d))  # no exception


class InputTest(unittest.TestCase):
    def test_16_bit_scan_is_not_white(self):
        with tempfile.TemporaryDirectory() as d:
            a = np.full((50, 60), 30000, np.uint16)
            a[10:20, 10:50] = 3000
            f = Path(d) / "scan.png"
            Image.fromarray(a).save(f)
            rgb, _ = load_rgb(f)
            self.assertLess(int(rgb[15, 30, 0]), 40)
            self.assertGreater(int(rgb[40, 30, 0]), 100)


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        r = subprocess.run([sys.executable, "-m", "reading_companion", "extract", *map(str, args), "--quiet", "--ocr", "none"],
                           capture_output=True, text=True, cwd=PKG_ROOT, timeout=600)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1, r.stdout + r.stderr)
        return r.returncode, json.loads(lines[0])

    def test_bad_template_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as d:
            code, out = self.run_cli(d, "--name-template", "{stem}_{foo}")
            self.assertEqual(code, 2)
            self.assertIn("foo", out["error"])

    def test_out_must_not_be_the_photo_folder(self):
        with tempfile.TemporaryDirectory() as d:
            code, out = self.run_cli(d, "--out", d)
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
