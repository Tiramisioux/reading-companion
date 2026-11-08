"""Group marked text lines into bands (stripes) and cut them from the full-resolution photo.

Rules (from the user):
- marks on consecutive text lines are ONE stripe, whatever their colour or kind;
- a completely unmarked text line between two marks separates them;
- every stripe also shows the line before and the line after the marked lines (``context``);
- a stripe is a plain horizontal cut across the full width of the page (margin to margin), so
  notes in the margin are seen next to the lines they were written beside.

Notes in the top margin count as a virtual line 0 just above the first printed line, notes in the
bottom margin as a virtual line N+1 just below the last one; a note there and a highlight on the
first (last) line are therefore consecutive and share one stripe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .geometry import Geometry
from .marks import Mark, PageAnalysis, TextLine

GAP_BREAK = 1.7      # marked lines further apart than this many pitches (blank space) are not consecutive
CONTEXT_GAP = 4.0    # a context line further than this many pitches away (a figure, a big gap) is left out
NEAR_EDGE = 12.0     # a top/bottom note further than this many pitches from the text stands alone
JOIN_GAP = 2.5       # a top/bottom note this close to the first/last line is consecutive with it


@dataclass
class Band:
    first_line: int               # first marked line (1-based; 0 = top margin, N+1 = bottom margin)
    last_line: int
    ctx_first: int                # first and last printed line shown (with context)
    ctx_last: int
    y0: float                     # flat-frame cut above
    y1: float                     # flat-frame cut below
    x0: int
    x1: int
    marks: List[Mark] = field(default_factory=list)
    loose: bool = False           # a mark that touches no printed line (shown on its own)

    @property
    def kinds(self) -> List[str]:
        return sorted({m.kind for m in self.marks})

    @property
    def colors(self) -> List[str]:
        seen: List[str] = []
        for m in sorted(self.marks, key=lambda m: m.y0):
            if m.color not in seen:
                seen.append(m.color)
        return seen


def _line_by_center(lines: List[TextLine], y: float) -> Optional[TextLine]:
    best, bd = None, 1e9
    for ln in lines:
        d = abs(ln.center - y)
        if d < bd:
            best, bd = ln, d
    return best


def _covered_columns(an: PageAnalysis, m: Mark, r0: float, r1: float) -> int:
    y0, y1 = int(max(m.y0, r0)), int(min(m.y1, r1)) + 1
    if y1 <= y0:
        return 0
    sub = an.tint[y0:y1, m.x0:m.x1]
    return int((sub.sum(axis=0) > 0).sum())


def assign_lines(an: PageAnalysis, pitch: float) -> None:
    """Fill m.lines for every mark (1-based printed lines; 0 / N+1 for the top / bottom margin)."""
    lines = an.lines
    p = pitch
    n = len(lines)
    for m in an.marks:
        m.lines = []
        if not lines:
            continue
        if m.kind == "highlight":
            for ln in lines:
                c0, c1 = ln.center - 0.22 * p, ln.center + 0.22 * p
                if min(m.y1, c1) - max(m.y0, c0) >= 0.5 * (c1 - c0) and _covered_columns(an, m, c0, c1) >= 0.8 * p:
                    m.lines.append(ln.index)
            if not m.lines:
                ln = _line_by_center(lines, m.yc)
                if ln is not None and abs(ln.center - m.yc) < 0.6 * p:
                    m.lines.append(ln.index)
        elif m.kind == "underline":
            cands = [ln for ln in lines if 0.05 * p <= m.yc - ln.center <= 1.0 * p]
            if cands:
                m.lines.append(max(cands, key=lambda ln: ln.center).index)
            else:
                ln = _line_by_center(lines, m.yc)
                if ln is not None and abs(ln.center - m.yc) < 0.8 * p:
                    m.lines.append(ln.index)
        elif m.kind == "margin_line":
            m.lines = [ln.index for ln in lines if m.y0 - 0.05 * p <= ln.center <= m.y1 + 0.05 * p]
            if not m.lines:
                ln = _line_by_center(lines, m.yc)
                if ln is not None and abs(ln.center - m.yc) < 0.8 * p:
                    m.lines.append(ln.index)
        elif m.where == "top":
            gap = lines[0].top - m.y1
            if gap <= JOIN_GAP * p:
                m.lines = [0]          # just above line 1: one stripe with marks on line 1
            elif gap <= NEAR_EDGE * p:
                m.lines = [-1]         # higher up (a chapter opening): its own stripe down to line 1
        elif m.where == "bottom":
            gap = m.y0 - lines[-1].bottom
            if gap <= JOIN_GAP * p:
                m.lines = [n + 1]
            elif gap <= NEAR_EDGE * p:
                m.lines = [n + 2]
        else:  # a note beside the printed lines
            m.lines = [ln.index for ln in lines if m.y0 - 0.25 * p <= ln.center <= m.y1 + 0.25 * p]
            if not m.lines:
                ln = _line_by_center(lines, m.yc)
                if ln is not None:
                    m.lines.append(ln.index)


def make_bands(an: PageAnalysis, geo: Geometry, context: int = 1) -> List[Band]:
    p = geo.pitch
    lines = {ln.index: ln for ln in an.lines}
    n = len(an.lines)
    marked: Dict[int, List[Mark]] = {}
    loose: List[Mark] = []
    for m in an.marks:
        if m.lines:
            for i in m.lines:
                marked.setdefault(i, []).append(m)
        else:
            loose.append(m)

    def consecutive(a: int, b: int) -> bool:
        if b != a + 1 or a == -1 or b == n + 2:
            return False
        if a == 0 or b == n + 1:
            return True
        return (lines[b].center - lines[a].center) < GAP_BREAK * p

    groups: List[List[int]] = []
    for i in sorted(marked):
        if groups and consecutive(groups[-1][-1], i):
            groups[-1].append(i)
        else:
            groups.append([i])

    bands: List[Band] = []
    for g in groups:
        ms: List[Mark] = []
        for i in g:
            for m in marked[i]:
                if m not in ms:
                    ms.append(m)
        real = [i for i in g if 1 <= i <= n]
        # context: the printed line(s) just before and after the marked ones
        if real:
            lo, hi = real[0], real[-1]
            for _ in range(context):
                if lo - 1 >= 1 and lines[lo].center - lines[lo - 1].center < CONTEXT_GAP * p:
                    lo -= 1
                if hi + 1 <= n and lines[hi + 1].center - lines[hi].center < CONTEXT_GAP * p:
                    hi += 1
        elif g[0] <= 0:                       # a note in the top margin: the line(s) after it
            lo, hi = 1, min(n, max(1, context))
        else:                                 # a note in the bottom margin: the line(s) before it
            lo, hi = max(1, n - max(1, context) + 1), n
        y0 = lines[lo].top if 1 <= lo <= n else min(m.y0 for m in ms) - 0.3 * p
        y1 = lines[hi].bottom if 1 <= hi <= n else max(m.y1 for m in ms) + 0.3 * p
        prev_c = lines[lo - 1].center if (lo - 1) in lines else None
        next_c = lines[hi + 1].center if (hi + 1) in lines else None
        lim_top = (prev_c + 0.32 * p) if prev_c is not None else -1e9
        lim_bot = (next_c - 0.32 * p) if next_c is not None else 1e9
        for m in ms:
            if m.kind == "margin_note":
                y0 = min(y0, m.y0 - 0.3 * p)
                y1 = max(y1, m.y1 + 0.3 * p)
            elif m.kind in ("underline", "highlight"):
                y0 = min(y0, max(m.y0 - 0.06 * p, lim_top))
                y1 = max(y1, min(m.y1 + 0.06 * p, lim_bot))
        bands.append(Band(g[0], g[-1], lo, hi, float(y0), float(y1), 0, 0, ms))
    for m in loose:
        bands.append(Band(0, 0, 0, 0, m.y0 - 0.4 * p, m.y1 + 0.4 * p, 0, 0, [m], loose=True))
    bands.sort(key=lambda b: b.y0)
    # a loose mark that overlaps another stripe joins it
    merged: List[Band] = []
    for b in bands:
        if merged and b.y0 < merged[-1].y1 - 0.2 * p and (b.loose or merged[-1].loose):
            o = merged[-1]
            o.y0, o.y1 = min(o.y0, b.y0), max(o.y1, b.y1)
            o.marks.extend(m for m in b.marks if m not in o.marks)
            if o.loose and not b.loose:
                o.first_line, o.last_line, o.ctx_first, o.ctx_last, o.loose = b.first_line, b.last_line, b.ctx_first, b.ctx_last, False
        else:
            merged.append(b)
    h, w = an.Lw.shape
    lay = an.layout
    for b in merged:
        b.y0 = float(max(0.0, b.y0))
        b.y1 = float(min(h - 1.0, b.y1))
        r0, r1 = int(b.y0), int(b.y1) + 1
        if lay is not None and r1 > r0:
            x0 = float(np.max(lay.page_l[r0:r1]))
            x1 = float(np.min(lay.page_r[r0:r1]))
            for m in b.marks:
                if m.kind in ("margin_line", "margin_note"):
                    x0 = min(x0, max(float(np.min(lay.page_l[r0:r1])), m.x0 - 0.2 * p))
                    x1 = max(x1, min(float(np.max(lay.page_r[r0:r1])), m.x1 + 0.2 * p))
        else:
            x0, x1 = float(geo.block.x0), float(geo.block.x1)
        b.x0, b.x1 = int(round(x0)), int(round(max(x1, x0 + 1)))
    return merged


def cut_stripe(src: np.ndarray, geo: Geometry, band: Band) -> np.ndarray:
    """Cut the band from the full-resolution source, straightened (RGB uint8)."""
    step = geo.scale  # flat px per output px
    w_out = int(round((band.x1 - band.x0) / step))
    h_out = int(round((band.y1 - band.y0) / step))
    mx, my = geo.flat_maps(band.x0, band.y0, max(1, w_out), max(1, h_out), step)
    return cv2.remap(src, mx, my, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
