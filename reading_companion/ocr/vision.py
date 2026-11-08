"""Apple Vision backend (macOS): builds ``rc-vision`` from ``rc_vision.swift`` with ``swiftc`` on
first use (cached), then reads all row images of a page in one call.

The helper prints the same line shape as the note-ingest ``ni-vision`` tool (top-left,
normalised coordinates), so ``--ocr-bin /path/to/rc-vision`` may point at a prebuilt copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from . import OcrEngine, Word

SOURCE = Path(__file__).with_name("rc_vision.swift")


def cache_dir() -> Path:
    base = Path.home() / "Library" / "Caches" if Path("/System/Library").exists() else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "reading-companion"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_helper() -> Path:
    src = SOURCE.read_bytes()
    tag = hashlib.sha1(src).hexdigest()[:10]
    out = cache_dir() / f"rc-vision-{tag}"
    if out.exists():
        return out
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise RuntimeError("swiftc not found (install the Xcode Command Line Tools: xcode-select --install)")
    env = dict(os.environ)
    if not env.get("DEVELOPER_DIR") and Path("/Library/Developer/CommandLineTools").exists():
        env["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"
    res = subprocess.run([swiftc, "-O", "-o", str(out), str(SOURCE)], capture_output=True, text=True, env=env, timeout=600)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError("swiftc failed: " + (res.stderr or res.stdout).strip()[:500])
    return out


class VisionEngine(OcrEngine):
    name = "vision"

    def __init__(self, languages: Optional[str] = None, binary: Optional[str] = None):
        self.binary = Path(binary) if binary else build_helper()
        if not self.binary.exists():
            raise RuntimeError(f"{self.binary} not found")
        self.languages = languages or "en-US,sv-SE"
        self.version = f"Apple Vision ({self.binary.name})"

    def read(self, images: Sequence[np.ndarray]) -> List[List[Word]]:
        if not images:
            return []
        with tempfile.TemporaryDirectory(prefix="rc-vision-") as tmp:
            paths = []
            for i, img in enumerate(images):
                p = os.path.join(tmp, f"row{i}.png")
                cv2.imwrite(p, img)
                paths.append(p)
            res = subprocess.run([str(self.binary), "--languages", self.languages, *paths],
                                 capture_output=True, text=True, timeout=300)
            by_file = {}
            for line in res.stdout.splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_file[d.get("file")] = d
            out: List[List[Word]] = []
            for p, img in zip(paths, images):
                d = by_file.get(p, {})
                h, w = img.shape[:2]
                words: List[Word] = []
                for ln in d.get("lines", []):
                    x0, y0 = ln["x"] * w, ln["y"] * h
                    x1, y1 = x0 + ln["w"] * w, y0 + ln["h"] * h
                    toks = ln["text"].split()
                    total = sum(len(t) for t in toks) + max(0, len(toks) - 1)
                    pos = 0
                    for t in toks:  # spread a line's box over its words (for folio positions)
                        a = x0 + (x1 - x0) * pos / max(1, total)
                        b = x0 + (x1 - x0) * (pos + len(t)) / max(1, total)
                        words.append((t, float(ln.get("conf", 0.0)), int(a), int(y0), int(b), int(y1)))
                        pos += len(t) + 1
                out.append(words)
            return out
