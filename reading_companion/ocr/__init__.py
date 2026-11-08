"""OCR backends, used only to read printed page numbers, running heads and chapter headings.

Engines
-------
vision     Apple Vision (macOS) through a small Swift helper, compiled on first use with
           ``swiftc`` from the Command Line Tools. Best quality; no extra install on a Mac.
tesseract  The ``tesseract`` command line program (``brew install tesseract``,
           ``apt install tesseract-ocr``).
none       No OCR: page numbers and chapters are left empty.
auto       vision on macOS when it can be built, else tesseract when installed, else none.

Every engine reads a batch of small grayscale row images and returns, per image, a list of
words ``(text, confidence 0..1, x0, y0, x1, y1)`` in that image's pixel coordinates.
"""
from __future__ import annotations

import platform
import shutil
from typing import List, Optional, Sequence, Tuple

import numpy as np

Word = Tuple[str, float, int, int, int, int]


class OcrEngine:
    name = "none"
    version = ""

    def read(self, images: Sequence[np.ndarray], mode: str = "line") -> List[List[Word]]:
        """mode "line": each image is one printed line; "block": several lines (a stripe)."""
        return [[] for _ in images]


def get_engine(choice: str = "auto", languages: Optional[str] = None, binary: Optional[str] = None) -> Tuple[OcrEngine, str]:
    """Return (engine, note). ``note`` explains a fallback, e.g. when Vision cannot be built."""
    choice = (choice or "auto").lower()
    notes = []
    if choice in ("auto", "vision"):
        if platform.system() == "Darwin" or binary:
            from .vision import VisionEngine
            try:
                return VisionEngine(languages=languages, binary=binary), ""
            except Exception as exc:  # no swiftc, build failed, ...
                notes.append(f"vision unavailable: {exc}")
                if choice == "vision":
                    return OcrEngine(), "; ".join(notes)
        elif choice == "vision":
            return OcrEngine(), "vision needs macOS"
    if choice in ("auto", "tesseract"):
        if shutil.which("tesseract"):
            from .tesseract import TesseractEngine
            try:
                return TesseractEngine(languages=languages), "; ".join(notes)
            except Exception as exc:
                notes.append(f"tesseract unavailable: {exc}")
        else:
            notes.append("tesseract not installed")
    if choice not in ("auto", "vision", "tesseract", "none"):
        notes.append(f"unknown OCR engine {choice!r}")
    return OcrEngine(), "; ".join(notes)
