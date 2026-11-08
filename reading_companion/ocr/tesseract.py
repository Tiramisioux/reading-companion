"""Tesseract backend: one ``tesseract`` call per row image (rows are tiny, calls are fast)."""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import List, Optional, Sequence

import cv2
import numpy as np

from . import OcrEngine, Word


class TesseractEngine(OcrEngine):
    name = "tesseract"

    def __init__(self, languages: Optional[str] = None):
        out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, timeout=30)
        first = (out.stdout or out.stderr).strip().splitlines()
        self.version = first[0] if first else "tesseract"
        langs = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=30).stdout.split()
        wanted = [l for l in (languages or "eng+swe").replace(",", "+").split("+") if l]
        have = [l for l in wanted if l in langs] or (["eng"] if "eng" in langs else langs[1:2])
        self.lang = "+".join(have) if have else "eng"

    def read(self, images: Sequence[np.ndarray]) -> List[List[Word]]:
        out: List[List[Word]] = []
        with tempfile.TemporaryDirectory(prefix="rc-ocr-") as tmp:
            for i, img in enumerate(images):
                g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                pad = max(10, g.shape[0] // 4)
                _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                best: List[Word] = []
                best_score = -1.0
                # grey and binarised crops, line then word mode: keep the most confident reading
                for v, variant in enumerate((g, bw)):
                    path = os.path.join(tmp, f"row{i}_{v}.png")
                    cv2.imwrite(path, cv2.copyMakeBorder(variant, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255))
                    for psm in ("7", "8"):
                        words = self._run(path, psm)
                        if words:
                            score = float(np.mean([w[1] for w in words])) * min(1.0, sum(len(w[0]) for w in words) / 3.0)
                            if score > best_score:
                                best, best_score = words, score
                            break
                out.append(best)
        return out

    def _run(self, path: str, psm: str) -> List[Word]:
        try:
            res = subprocess.run(["tesseract", path, "-", "--psm", psm, "-l", self.lang, "tsv"],
                                 capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return []
        return _parse_tsv(res.stdout)


def _parse_tsv(tsv: str) -> List[Word]:
    words: List[Word] = []
    for line in tsv.splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 12 or f[0] != "5":
            continue
        text = f[11].strip()
        if not text:
            continue
        try:
            conf = float(f[10]) / 100.0
            x, y, w, h = int(f[6]), int(f[7]), int(f[8]), int(f[9])
        except ValueError:
            continue
        words.append((text, max(0.0, conf), x, y, x + w, y + h))
    return words

