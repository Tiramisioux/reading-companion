"""A synthetic book page (no copyrighted text): printed lines of pseudo-words, a running head
with a page number, highlights, an underline, a margin line and a handwritten note."""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

WORDS = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore "
         "et dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi aliquip "
         "ex ea commodo consequat duis aute irure in reprehenderit voluptate velit esse cillum fugiat nulla").split()

HIGHLIGHTS = {"yellow": (255, 236, 90), "teal": (90, 200, 215), "pink": (250, 150, 190), "orange": (250, 175, 110), "green": (150, 225, 140)}


def _scribble(ink: np.ndarray, x: float, y: float, letters: int, vertical: bool = False) -> None:
    """Joined loops like handwriting, ``letters`` of them, from (x, y) rightwards (or upwards when ``vertical``), with a
    tall loop every third letter and a small wobble of the baseline."""
    pts, s = [], 0.0
    for k in range(letters):
        t = np.linspace(0, 1, 12)
        h = 26 if k % 3 else 50
        along = s + 14 * t - 4 * np.sin(2 * np.pi * t)
        across = 10 - h * np.sin(np.pi * t) + 3 * np.sin(k * 0.9)
        pts += list(zip(x + across, y - along) if vertical else zip(x + along, y + across))
        s += 14
    cv2.polylines(ink, [np.array(pts, np.int32)], False, 255, 3, cv2.LINE_AA)


def make_page(n_lines: int = 24, highlight: Optional[Dict[int, str]] = None, underline: Tuple[int, ...] = (),
              margin_line: Optional[Tuple[int, int]] = None, note_beside: Optional[int] = None,
              head: str = "224          WAYFINDING", angle: float = 0.0, seed: int = 3,
              width: int = 1500, height: int = 2000, underline_color: Optional[Tuple[int, int, int]] = None,
              finger: Union[None, bool, Tuple[int, int]] = None, rules: Tuple[int, ...] = (),
              figure: Optional[Tuple[int, int]] = None, caption: bool = False, note_below: bool = False,
              note_right: Optional[int] = None, note_up: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict]:
    """RGB page photo-like image and the truth: {"lines": [(y_center, x0, x1)], ...}.
    ``highlight`` maps 1-based line -> colour name. Printed things that are no mark of the reader: ``rules`` draws a
    table's rule across the whole column under each of these lines; ``figure`` = (a, b) prints line art (a map's coast,
    a plan's rooms, hatching) instead of the text of lines a..b, and ``caption`` a short centred line under it.
    Handwriting (joined loops): ``note_beside`` = line, in the left margin; ``note_below``, a line of it just under the
    last printed line; ``note_right`` = line, in the right margin close to the page's edge; ``note_up`` = (a, b),
    written sideways up the right margin beside lines a..b."""
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
    words = []          # (line, word, x0, x1) of every printed word
    for i in range(n_lines):
        y = top + i * pitch
        x = x0
        if figure and figure[0] <= i + 1 <= figure[1]:
            lines.append((y - 9, x0, x0))
            continue
        last = (i % 7 == 6)  # paragraph ends are short
        limit = x0 + (x1 - x0) * (0.45 if last else 1.0)
        while True:
            w = WORDS[int(rng.integers(len(WORDS)))]
            (tw, th), _ = cv2.getTextSize(w, font, scale, thick)
            if x + tw > limit:
                break
            cv2.putText(ink, w, (x, y), font, scale, 255, thick, cv2.LINE_AA)
            words.append((i + 1, w, x, x + tw))
            x += tw + 16
        lines.append((y - 9, x0, x))
    # running head
    cv2.putText(ink, head, (x0, top - 2 * pitch), font, 0.8, 255, 2, cv2.LINE_AA)
    # a table's printed rules: ruler-straight, the column's full width
    for ln in rules:
        yr = lines[ln - 1][0] + 30
        cv2.line(ink, (x0, yr), (x1, yr), 255, 2, cv2.LINE_8)
    # printed line art in place of some lines: a frame, a wandering coast, rooms, hatching, a few labels
    if figure:
        fy0, fy1 = lines[figure[0] - 1][0] - 25, lines[figure[1] - 1][0] + 15
        cv2.rectangle(ink, (x0 + 20, fy0), (x1 - 20, fy1), 255, 2)
        t = np.linspace(0, 1, 200)
        coast = np.stack([x0 + 60 + (x1 - x0 - 160) * t, (fy0 + fy1) / 2 + (fy1 - fy0) * 0.3 * np.sin(7 * t) * np.cos(3 * t)], 1)
        cv2.polylines(ink, [coast.astype(np.int32)], False, 255, 2, cv2.LINE_AA)
        for k in range(4):
            rx = x0 + 80 + k * 230
            cv2.rectangle(ink, (rx, fy0 + 40), (rx + 180, fy0 + 40 + int(0.35 * (fy1 - fy0))), 255, 2)
        for k in range(0, int(fy1 - fy0) - 80, 14):
            cv2.line(ink, (x1 - 260, fy1 - 20 - k), (x1 - 200, fy1 - 60 - k), 255, 1, cv2.LINE_AA)
        for k, w in enumerate(("north", "river", "hill")):
            cv2.putText(ink, w, (x0 + 120 + k * 300, fy1 - 40 - 30 * k), font, 0.6, 255, 1, cv2.LINE_AA)
        if caption:
            cap = "7. lorem ipsum dolor sit amet"
            (tw, _), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, 1)
            cv2.putText(ink, cap, ((x0 + x1 - tw) // 2, fy1 + 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, 255, 1, cv2.LINE_AA)
    # highlights (translucent, multiply)
    hl = np.ones((height, width, 3), np.float32)
    for ln, colour in (highlight or {}).items():
        yc, a, b = lines[ln - 1]
        if isinstance(colour, tuple):          # (colour, x0, x1): a highlight over part of the line, in page pixels
            colour, a, b = colour[0], colour[1] + 6, colour[2] - 6
        c = np.array(HIGHLIGHTS[colour], np.float32) / 255.0
        hl[yc - 20:yc + 16, a - 6:b + 6] = c
    img = img * hl
    # underlines (pen; black, or a coloured pen drawn over the page)
    pen = np.zeros((height, width), np.uint8)
    for ln in underline:
        yc, a, b = lines[ln - 1]
        cv2.line(ink if underline_color is None else pen, (a + 40, yc + 20), (min(b, a + 600), yc + 23), 255, 3, cv2.LINE_AA)
    if underline_color is not None:
        r = (pen.astype(np.float32) / 255.0)[..., None]
        img = img * (1 - r) + np.array(underline_color, np.float32) * r
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
    if note_below:          # a line of handwriting under the last printed line (taken for a printed line before 0.4.0)
        _scribble(ink, x0 + 120, lines[-1][0] + int(1.45 * pitch), 26)
    if note_right:          # in the right margin, close to the page's edge
        _scribble(ink, x1 + 70, lines[note_right - 1][0], 8)
    if note_up:             # written sideways up the right margin
        ya, yb = lines[note_up[0] - 1][0], lines[note_up[1] - 1][0]
        _scribble(ink, x1 + 110, yb, max(4, (yb - ya) // 14), vertical=True)
    a = (ink.astype(np.float32) / 255.0)[..., None]
    img = img * (1 - a) + np.array([35, 33, 30], np.float32) * a
    # a finger holding the page beside lines a..b: it comes in over the page's right edge from the table beyond it
    if isinstance(finger, tuple):
        edge = x1 + 120
        img[:, edge:] = np.array([58, 56, 62], np.float32)
        ya = lines[finger[0] - 1][0] - 30
        yb = lines[finger[1] - 1][0] + 30
        skin = np.zeros((height, width), np.uint8)
        cv2.ellipse(skin, ((width + x1 - 40) // 2, (ya + yb) // 2), ((width - x1 + 120) // 2, (yb - ya) // 2), 0, 0, 360, 255, -1)
        shade = np.linspace(0.75, 1.0, width, dtype=np.float32)[None, :, None]
        s_ = (cv2.GaussianBlur(skin, (0, 0), 3).astype(np.float32) / 255.0)[..., None]
        img = img * (1 - s_) + np.array([196, 132, 98], np.float32) * shade * s_
    elif finger:  # True: a thumb holding the page, over the right end of lines 9-13
        fm = np.zeros((height, width), np.uint8)
        yc = lines[10][0]
        cv2.ellipse(fm, (x1 - 20, yc), (120, 170), 0, 0, 360, 255, -1)
        f_ = cv2.GaussianBlur(fm, (0, 0), 6).astype(np.float32)[..., None] / 255.0
        shade = (0.85 + 0.15 * (xx[..., None] - (x1 - 140)) / 260.0).clip(0.7, 1.0)
        img = img * (1 - f_) + np.array([196, 132, 104], np.float32) * shade * f_
    img = np.clip(img, 0, 255).astype(np.uint8)
    if angle:
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(60, 60, 70))
    return img, {"lines": lines, "pitch": pitch, "words": words}
