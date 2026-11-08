"""A synthetic book page (no copyrighted text): printed lines of pseudo-words, a running head
with a page number, highlights, an underline, a margin line and a handwritten note."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

WORDS = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore "
         "et dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi aliquip "
         "ex ea commodo consequat duis aute irure in reprehenderit voluptate velit esse cillum fugiat nulla").split()

HIGHLIGHTS = {"yellow": (255, 236, 90), "teal": (90, 200, 215), "pink": (250, 150, 190), "orange": (250, 175, 110), "green": (150, 225, 140)}


def make_page(n_lines: int = 24, highlight: Optional[Dict[int, str]] = None, underline: Tuple[int, ...] = (),
              margin_line: Optional[Tuple[int, int]] = None, note_beside: Optional[int] = None,
              head: str = "224          WAYFINDING", angle: float = 0.0, seed: int = 3,
              width: int = 1500, height: int = 2000) -> Tuple[np.ndarray, Dict]:
    """RGB page photo-like image and the truth: {"lines": [(y_center, x0, x1)], ...}.
    ``highlight`` maps 1-based line -> colour name."""
    rng = np.random.default_rng(seed)
    paper = np.array([238, 232, 220], np.float32)
    img = np.ones((height, width, 3), np.float32) * paper
    # gentle illumination gradient + noise
    yy, xx = np.mgrid[0:height, 0:width]
    img *= (0.92 + 0.08 * (xx / width))[..., None]
    img += rng.normal(0, 2.0, img.shape)
    ink = np.zeros((height, width), np.uint8)
    x0, x1 = 230, 1290
    pitch = 62
    top = 260
    font, scale, thick = cv2.FONT_HERSHEY_COMPLEX, 0.95, 2
    lines = []
    for i in range(n_lines):
        y = top + i * pitch
        x = x0
        last = (i % 7 == 6)  # paragraph ends are short
        limit = x0 + (x1 - x0) * (0.45 if last else 1.0)
        while True:
            w = WORDS[int(rng.integers(len(WORDS)))]
            (tw, th), _ = cv2.getTextSize(w, font, scale, thick)
            if x + tw > limit:
                break
            cv2.putText(ink, w, (x, y), font, scale, 255, thick, cv2.LINE_AA)
            x += tw + 16
        lines.append((y - 9, x0, x))
    # running head
    cv2.putText(ink, head, (x0, top - 2 * pitch), font, 0.8, 255, 2, cv2.LINE_AA)
    # highlights (translucent, multiply)
    hl = np.ones((height, width, 3), np.float32)
    for ln, colour in (highlight or {}).items():
        yc, a, b = lines[ln - 1]
        c = np.array(HIGHLIGHTS[colour], np.float32) / 255.0
        hl[yc - 20:yc + 16, a - 6:b + 6] = c
    img = img * hl
    # underlines (pen)
    for ln in underline:
        yc, a, b = lines[ln - 1]
        cv2.line(ink, (a + 40, yc + 20), (min(b, a + 600), yc + 23), 255, 3, cv2.LINE_AA)
    # red margin line beside lines a..b
    if margin_line:
        ya = lines[margin_line[0] - 1][0] - 22
        yb = lines[margin_line[1] - 1][0] + 22
        red = np.zeros((height, width), np.uint8)
        cv2.line(red, (x1 + 50, ya), (x1 + 52, yb), 255, 6, cv2.LINE_AA)
        r = (red.astype(np.float32) / 255.0)[..., None]
        img = img * (1 - r) + np.array([200, 30, 40], np.float32) * r
    # a handwritten note in the left margin: joined loops
    if note_beside:
        yc = lines[note_beside - 1][0]
        pts, x = [], 60.0
        for k in range(9):
            t = np.linspace(0, 1, 12)
            h = 26 if k % 3 else 50
            pts += list(zip(x + 14 * t - 4 * np.sin(2 * np.pi * t), yc + 10 - h * np.sin(np.pi * t)))
            x += 14
        cv2.polylines(ink, [np.array(pts, np.int32)], False, 255, 3, cv2.LINE_AA)
    a = (ink.astype(np.float32) / 255.0)[..., None]
    img = img * (1 - a) + np.array([35, 33, 30], np.float32) * a
    img = np.clip(img, 0, 255).astype(np.uint8)
    if angle:
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(60, 60, 70))
    return img, {"lines": lines, "pitch": pitch}
