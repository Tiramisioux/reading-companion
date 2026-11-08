"""Find the reader's marks in the flat (deskewed, dewarped) analysis frame.

Colour is judged after a *local white balance*: every pixel is divided, channel by channel,
by the colour of the bare paper around it (estimated from bright, near-neutral pixels by
normalised convolution). Paper becomes neutral white whether it lies in sunlight, in shadow or
under a warm lamp; a translucent highlighter keeps its tint; bleed-through from the other side
of the leaf and the colour fringes around printed letters stay close to neutral.

Kinds of marks
--------------
highlight    translucent colour over text (any hue)
underline    a long, thin, horizontal pen stroke just below a printed line (dark or coloured)
margin_line  a long, thin, vertical pen stroke in the margin beside the text
margin_note  any other cluster of ink in the margins (handwriting, ticks, asterisks)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import Geometry, odd, runs, smooth1d, local_peaks

# chroma (CIE a*b* distance from neutral after the local white balance)
HL_CHROMA = 10.0          # a pixel is tinted above this
HL_COMPONENT_CHROMA = 15.0  # a stroke's median tint must reach this
INK_L = 58.0              # white-balanced L* below this is ink (paper is ~100)

COLOR_BUCKETS = [  # (name, hue_from, hue_to) in degrees of atan2(b*, a*) after white balance
    ("pink", 330.0, 20.0),
    ("red", 20.0, 45.0),
    ("orange", 45.0, 75.0),
    ("yellow", 75.0, 125.0),
    ("green", 125.0, 175.0),
    ("teal", 175.0, 235.0),
    ("blue", 235.0, 285.0),
    ("purple", 285.0, 330.0),
]


def color_name(hue: float, chroma: float, lightness: float) -> str:
    if chroma < 8.0:
        return "black" if lightness < 50 else "grey"
    h = hue % 360.0
    for name, lo, hi in COLOR_BUCKETS:
        if (lo <= h < hi) if lo < hi else (h >= lo or h < hi):
            if name == "pink" and lightness < 55:
                return "red"
            return name
    return "unknown"


@dataclass
class Mark:
    kind: str
    x0: int
    y0: int
    x1: int
    y1: int
    color: str
    hue: float
    chroma: float
    lightness: float
    rgb: Tuple[int, int, int]
    area: int
    lines: List[int] = field(default_factory=list)
    yc: float = 0.0
    where: str = ""        # margin notes: "side", "top" or "bottom"
    why: str = ""          # margin notes in the top/bottom margins: why it is handwriting

    def __post_init__(self) -> None:
        if not self.yc:
            self.yc = (self.y0 + self.y1) / 2.0

    def as_dict(self) -> Dict:
        return {
            "kind": self.kind, "color": self.color,
            "bbox_flat": [self.x0, self.y0, self.x1, self.y1],
            "hue": round(self.hue, 1), "chroma": round(self.chroma, 1), "lightness": round(self.lightness, 1),
            "rgb": list(self.rgb), "lines": self.lines,
        }


@dataclass
class TextLine:
    index: int     # 1-based, top to bottom
    center: float  # flat-frame y of the ink-profile peak (about the x-height middle)
    top: float     # cut position above the line (gap middle)
    bottom: float  # cut position below the line (gap middle)
    ink: float     # relative ink amount (short last lines of paragraphs have little)


@dataclass
class PageAnalysis:
    flat: np.ndarray            # flat-frame analysis RGB
    Lw: np.ndarray              # white-balanced L*
    aw: np.ndarray
    bw: np.ndarray
    paper_L: np.ndarray         # paper lightness (unbalanced L*) - shadows are dark here
    valid: np.ndarray           # inside the photo after rotation/dewarp
    paper: np.ndarray           # the main page's paper region (filled)
    ink: np.ndarray             # dark ink (white-balanced)
    tint: np.ndarray            # tinted pixels (white-balanced chroma)
    crop_x0: int = 0            # horizontal extent of the main page used for stripes
    crop_x1: int = 0
    limit_x0: int = 0           # nothing beyond these belongs to the main page
    limit_x1: int = 0
    lines: List[TextLine] = field(default_factory=list)
    marks: List[Mark] = field(default_factory=list)
    rejected: List[Mark] = field(default_factory=list)
    layout: Optional["Layout"] = None
    L: Optional[np.ndarray] = None           # unbalanced L* (page edges are lightness steps)
    head_lines: List[TextLine] = field(default_factory=list)  # running head / footer lines
    margin_groups: List["InkGroup"] = field(default_factory=list)  # every ink group in the top/bottom margins
    zones: dict = field(default_factory=dict)
    letter_h: float = 0.0       # median height of the body text's letters (flat px)
    skin: Optional[np.ndarray] = None        # fingers / a hand holding the book (flat px), painted out of the stripes
    skin_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# paper and white balance
# ---------------------------------------------------------------------------

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def smooth_large(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with a large sigma, computed on a downsampled copy (the result is smooth
    anyway; a direct blur with sigma in the hundreds of pixels takes seconds)."""
    f = int(sigma / 4.0)
    if f < 2:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def white_balance(flat: np.ndarray, valid: np.ndarray, block, pitch: float):
    """Return white-balanced (L, a, b), the paper lightness map and the paper-pixel mask."""
    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    k = odd(max(5, pitch * 0.35))
    env = cv2.morphologyEx(L, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    env = cv2.GaussianBlur(env, (0, 0), k / 2.0)
    cand = valid & (L > env - 4.0) & (L > 8.0)
    # global paper chroma from the text block area (paper dominates there)
    by0, by1 = max(0, block.y0), min(L.shape[0], block.y1)
    bx0, bx1 = max(0, block.x0), min(L.shape[1], block.x1)
    sub = cand[by0:by1, bx0:bx1]
    if sub.sum() > 500:
        a0 = float(np.median(a[by0:by1, bx0:bx1][sub]))
        b0 = float(np.median(b[by0:by1, bx0:bx1][sub]))
    else:
        a0 = float(np.median(a[cand])) if cand.any() else 0.0
        b0 = float(np.median(b[cand])) if cand.any() else 0.0
    paper_px = cand & (np.hypot(a - a0, b - b0) < 12.0)
    lin = _srgb_to_linear(flat)
    P = paper_px.astype(np.float32)
    sig = max(8.0, pitch * 2.0)
    den = smooth_large(P, sig)
    ref = np.empty_like(lin)
    for c in range(3):
        num = smooth_large(lin[..., c] * P, sig)
        ref[..., c] = num / np.maximum(den, 1e-4)
    # where no paper is near (big dark background), fall back to a wide estimate
    sparse = den < 0.02
    if sparse.any():
        sig2 = sig * 4
        den2 = smooth_large(P, sig2)
        for c in range(3):
            num2 = smooth_large(lin[..., c] * P, sig2)
            alt = num2 / np.maximum(den2, 1e-4)
            ref[..., c] = np.where(sparse, alt, ref[..., c])
    bal = lin / np.maximum(ref, 1e-3) * 0.92  # paper -> ~0.92 linear (L* ~ 97)
    srgb = (_linear_to_srgb(bal) * 255.0 + 0.5).astype(np.uint8)
    labw = cv2.cvtColor(srgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    Lw = labw[..., 0] * (100.0 / 255.0)
    aw = labw[..., 1] - 128.0
    bw = labw[..., 2] - 128.0
    paper_lin = ref.mean(axis=2)
    paper_L = (116.0 * np.cbrt(np.clip(paper_lin, 1e-6, 1.0)) - 16.0).astype(np.float32)
    return Lw, aw, bw, paper_L, paper_px


def paper_region(paper_px: np.ndarray, valid: np.ndarray, block, pitch: float) -> np.ndarray:
    """The main page's paper as one filled region (holes = text, highlights, margin marks)."""
    k = odd(max(3, pitch * 0.6))
    m = cv2.morphologyEx(paper_px.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return valid.copy()
    cy, cx = (block.y0 + block.y1) // 2, (block.x0 + block.x1) // 2
    # the component holding most of the text block's paper
    sub = lab[block.y0:block.y1, block.x0:block.x1]
    counts = np.bincount(sub.ravel(), minlength=n)
    counts[0] = 0
    best = int(np.argmax(counts)) if counts.max() > 0 else int(lab[cy, cx])
    reg = (lab == best).astype(np.uint8)
    return fill_holes(reg > 0) & valid


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every hole of a binary mask (flood the background from a padded border)."""
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask.astype(np.uint8)
    flood = pad.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 2)
    outside = flood[1:-1, 1:-1] == 2
    return ~outside


# ---------------------------------------------------------------------------
# text lines
# ---------------------------------------------------------------------------

def text_lines(ink: np.ndarray, x0: int, x1: int, y0: int, y1: int, pitch: float,
               tint: Optional[np.ndarray] = None) -> List[TextLine]:
    """Printed lines between rows y0..y1 inside columns x0..x1 (lines are horizontal here). ``tint``: the coloured pixels, so
    that a coloured pen's underline is known as one too."""
    h = ink.shape[0]
    lo, hi = max(0, int(y0 - pitch)), min(h, int(y1 + pitch))
    sub = ink[lo:hi, x0:x1].astype(np.uint8)
    # a long, thin horizontal run is a rule (a pen's underline, black or coloured), not letters: left in, a page underlined on
    # three lines in a row has rules stronger than the text between them, and a line's own peak is lost. A highlighter's band is
    # long too but thick, and the print under it stays
    pen = sub | tint[lo:hi, x0:x1].astype(np.uint8) if tint is not None else sub
    k = max(5, int(round(pitch * 1.5)))
    rules = cv2.morphologyEx(pen, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    thick = cv2.morphologyEx(rules, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, odd(max(3, pitch * 0.6)))))
    thin = cv2.dilate(rules & (1 - thick), np.ones((3, 1), np.uint8))
    prof = (sub & (1 - thin)).sum(axis=1).astype(np.float64)
    prof = smooth1d(prof, max(1.0, pitch * 0.1))
    if prof.max() <= 0:
        return []
    width = max(1, x1 - x0)
    pk = local_peaks(prof, max(2, int(pitch * 0.55)), prof.max() * 0.06)
    # drop weak peaks that are too close to a strong one (accents, descender rows)
    centers = [p + lo for p in pk if prof[p] > 0.004 * width]
    out: List[TextLine] = []
    for i, c in enumerate(centers):
        prev_c = centers[i - 1] if i > 0 else None
        next_c = centers[i + 1] if i + 1 < len(centers) else None
        top = (c + prev_c) / 2.0 if prev_c is not None and c - prev_c < 1.8 * pitch else c - 0.5 * pitch
        bot = (c + next_c) / 2.0 if next_c is not None and next_c - c < 1.8 * pitch else c + 0.5 * pitch
        out.append(TextLine(i + 1, float(c), float(top), float(bot), float(prof[c - lo] / width)))
    return out


# ---------------------------------------------------------------------------
# page layout: text column, page limits, margins
# ---------------------------------------------------------------------------

def line_texture(ink: np.ndarray, ln: TextLine, x0: int, x1: int, p: float) -> Tuple[float, float, float, int, int]:
    """(extent in pitches, ink transitions per 100 px per row, ink fill, x_start, x_end) of a line's
    x-height core inside columns x0..x1. Printed text: 4-9 transitions, fill 0.1-0.4; keyboard
    rows, shadow edges and page edges: almost solid, few transitions."""
    r0, r1 = int(max(0, ln.center - 0.2 * p)), int(ln.center + 0.2 * p) + 1
    sub = ink[r0:r1, x0:x1]
    rs = runs(sub.any(axis=0))
    merged: List[List[int]] = []
    for a_, b_ in rs:
        if merged and a_ - merged[-1][1] < 1.2 * p:
            merged[-1][1] = b_
        else:
            merged.append([a_, b_])
    if not merged:
        return 0.0, 0.0, 0.0, x0, x0
    a_, b_ = max(merged, key=lambda r: r[1] - r[0])
    seg = sub[:, a_:b_].astype(np.int8)
    trans = float((np.diff(seg, axis=1) == 1).sum()) / max(1, b_ - a_) / max(1, seg.shape[0]) * 100.0
    return (b_ - a_) / p, trans, float(seg.mean()), x0 + a_, x0 + b_


def validate_lines(lines: List[TextLine], ink: np.ndarray, blk, p: float) -> Tuple[List[TextLine], List[TextLine]]:
    """Keep printed body lines; return (body lines renumbered 1..N, head/foot lines).

    A running head (or a footer) is a short line set apart from the body by extra space."""
    kept: List[Tuple[TextLine, float, int, int]] = []
    for ln in lines:
        ext, trans, fill, a_, b_ = line_texture(ink, ln, blk.x0, blk.x1, p)
        if ext >= 1.5 and trans >= 2.5 and fill <= 0.6:
            kept.append((ln, ext, a_, b_))
    width_p = max(1.0, (blk.x1 - blk.x0) / p)
    head: List[TextLine] = []
    while len(kept) >= 3:
        ln, ext, _, _ = kept[0]
        gap = kept[1][0].center - ln.center
        if gap > 1.25 * p and ext < 0.6 * width_p:
            head.append(ln)
            kept.pop(0)
        else:
            break
    while len(kept) >= 3:
        ln, ext, _, _ = kept[-1]
        gap = ln.center - kept[-2][0].center
        if gap > 1.6 * p and ext < 0.35 * width_p:
            head.append(ln)
            kept.pop()
        else:
            break
    body: List[TextLine] = []
    for i, (ln, _, _, _) in enumerate(kept):
        prev_c = kept[i - 1][0].center if i > 0 else None
        next_c = kept[i + 1][0].center if i + 1 < len(kept) else None
        top = (ln.center + prev_c) / 2.0 if prev_c is not None and ln.center - prev_c < 1.8 * p else ln.center - 0.5 * p
        bot = (ln.center + next_c) / 2.0 if next_c is not None and next_c - ln.center < 1.8 * p else ln.center + 0.5 * p
        body.append(TextLine(i + 1, ln.center, top, bot, ln.ink))
    return body, head


def _step_scan(prof: np.ndarray, start: int, direction: int, max_dist: int, k_in: int, k_out: int, thr_frac: float) -> int:
    """First index from ``start`` moving in ``direction`` where the profile steps down for good:
    the mean of the k_in samples behind exceeds the mean of the k_out samples ahead by
    thr_frac of the inside level, and the region beyond those stays dark too (a pen line or a
    narrow gutter gets bright again and is not an edge). -1 when there is none."""
    n = len(prof)
    bad = ~np.isfinite(prof)
    cs = np.concatenate([[0.0], np.cumsum(np.where(bad, 0.0, prof))])
    cb = np.concatenate([[0], np.cumsum(bad)])

    def mean(a: int, b: int) -> float:
        a, b = max(0, a), min(n, b)
        if b <= a or cb[b] - cb[a] > 0:
            return np.nan   # touches the photo border: no evidence
        return (cs[b] - cs[a]) / (b - a)

    best_i, best_d = -1, 0.0
    for step in range(k_in, max_dist):
        i = start + direction * step
        if i - k_in < 0 or i + k_in >= n:
            break
        if direction > 0:
            inside, out1, out2 = mean(i - k_in, i), mean(i, i + k_out), mean(i + k_out, i + 3 * k_out)
        else:
            inside, out1, out2 = mean(i + 1, i + 1 + k_in), mean(i + 1 - k_out, i + 1), mean(i + 1 - 3 * k_out, i + 1 - k_out)
        if np.isnan(inside) or np.isnan(out1):
            break   # reached the photo border
        thr = max(5.0, thr_frac * inside)
        d = inside - out1
        if d >= thr and (np.isnan(out2) or inside - out2 >= 0.7 * thr):
            if d > best_d:
                best_i, best_d = i, d
            elif best_i >= 0 and d < 0.6 * best_d:
                break
        elif best_i >= 0:
            break
    return best_i


def _fit_edge(cands: List[Tuple[float, float]], n_bands: int, p: float) -> Optional[Tuple[float, float]]:
    """Robust straight line x = a + b*v through (v, x) candidates; None unless most bands agree."""
    if len(cands) < max(3, 0.35 * n_bands):
        return None
    vs = np.array([c[0] for c in cands], dtype=np.float64)
    xs = np.array([c[1] for c in cands], dtype=np.float64)
    keep = np.ones(len(vs), dtype=bool)
    a, b = float(np.median(xs)), 0.0
    for _ in range(5):
        if keep.sum() < 3:
            return None
        if np.ptp(vs[keep]) > 0:
            b, a = np.polyfit(vs[keep], xs[keep], 1)
        else:
            a, b = float(np.median(xs[keep])), 0.0
        r = np.abs(xs - (a + b * vs))
        new = r < 0.5 * p
        if np.array_equal(new, keep):
            break
        keep = new
    if keep.sum() < max(3, 0.35 * n_bands) or abs(b) > 0.2:
        return None
    return float(a), float(b)


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    ok = np.isfinite(a)
    cnt = ok.sum(axis=axis)
    tot = np.where(ok, a, 0.0).sum(axis=axis)
    out = np.full(cnt.shape, np.nan, dtype=np.float64)
    np.divide(tot, cnt, out=out, where=cnt > 0)
    return out


def find_page_edges(L: np.ndarray, valid: np.ndarray, lines: List[TextLine], col_l, col_r, p: float, W: float):
    """Outer edges of the main page's paper: left/right as x = a + b*v (None when not seen),
    top/bottom as y = a + b*x."""
    h, w = L.shape
    Lm = np.where(valid, L, np.nan).astype(np.float32)
    k_in, k_out = max(3, int(0.3 * p)), max(4, int(1.0 * p))
    res = {"left": None, "right": None, "top": None, "bottom": None}
    if not lines:
        return res
    y_lo, y_hi = int(lines[0].top), int(lines[-1].bottom)
    for side, direction in (("left", -1), ("right", 1)):
        cands, nb = [], 0
        for r0 in range(y_lo, max(y_lo + 1, y_hi - int(p)), max(1, int(p))):
            r1 = min(h, r0 + int(2 * p))
            v = (r0 + r1) / 2.0
            with np.errstate(all="ignore"):
                prof = _nanmean(Lm[r0:r1], 0)
            start = int((col_l[int(v)] if side == "left" else col_r[int(v)]) + direction * 0.1 * p)
            nb += 1
            x = _step_scan(prof, start, direction, int(0.45 * W), k_in, k_out, 0.10)
            if x >= 0:
                cands.append((v, float(x)))
        res[side] = _fit_edge(cands, nb, p)
    xa = int(np.median(col_l) + 0.1 * W)
    xb = int(np.median(col_r) - 0.1 * W)
    for side, direction, start, reach in (("top", -1, int(lines[0].top - 0.2 * p), int(20 * p)),
                                          ("bottom", 1, int(lines[-1].bottom + 0.2 * p), int(12 * p))):
        cands, nb = [], 0
        for c0 in range(xa, max(xa + 1, xb - int(p)), max(1, int(p))):
            c1 = min(w, c0 + int(2 * p))
            with np.errstate(all="ignore"):
                prof = _nanmean(Lm[:, c0:c1], 1)
            nb += 1
            y = _step_scan(prof, start, direction, reach, k_in, k_out, 0.10)
            if y >= 0:
                cands.append(((c0 + c1) / 2.0, float(y)))
        res[side] = _fit_edge(cands, nb, p)
    return res



@dataclass
class Layout:
    """Where the main page is, per flat-frame row (arrays of length = image height)."""
    page_l: np.ndarray            # leftmost column of the main page (paper edge, gutter, crease)
    page_r: np.ndarray
    text_l: np.ndarray            # left edge of the printed text (from the lines' own ends)
    text_r: np.ndarray
    page_top: int                 # first/last row of the main page's paper
    page_bottom: int
    gutter_l: float = -1.0        # facing page boundary (constant), -1 = none found
    gutter_r: float = -1.0
    xl: Tuple[float, float] = (0.0, 0.0)   # robust straight column edges x = a + b*v
    xr: Tuple[float, float] = (0.0, 0.0)
    edges: Optional[dict] = None           # step-scan page edges (for the review sheet)


def _robust_edge(vs: np.ndarray, xs: np.ndarray, side: str, p: float) -> Tuple[float, float]:
    """Fit x = a + b*v to line ends; indents (left) and short lines (right) are one-sided outliers."""
    if len(vs) == 0:
        return 0.0, 0.0
    if len(vs) < 3:
        return (float(np.min(xs)) if side == "left" else float(np.max(xs))), 0.0
    keep = np.ones(len(vs), dtype=bool)
    a, b = float(np.median(xs)), 0.0
    for _ in range(4):
        A = np.stack([np.ones(keep.sum()), vs[keep]], axis=1)
        sol, *_ = np.linalg.lstsq(A, xs[keep], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = xs - (a + b * vs)
        new = (r < 0.35 * p) if side == "left" else (r > -0.35 * p)
        if new.sum() < max(3, 0.3 * len(vs)) or np.array_equal(new, keep):
            break
        keep = new
    return a, float(np.clip(b, -0.12, 0.12))


def line_extents(ink: np.ndarray, lines: List[TextLine], x_lo: int, x_hi: int, p: float):
    """[x_start, x_end] of each line's printed ink (word gaps bridged, never past x_lo/x_hi)."""
    out = []
    for ln in lines:
        r0, r1 = int(max(0, ln.center - 0.3 * p)), int(ln.center + 0.3 * p) + 1
        cols = ink[r0:r1, x_lo:x_hi].any(axis=0)
        rs = runs(cols)
        if not rs:
            out.append(None)
            continue
        merged: List[List[int]] = []
        for a_, b_ in rs:
            if merged and a_ - merged[-1][1] < 1.2 * p:
                merged[-1][1] = b_
            else:
                merged.append([a_, b_])
        a_, b_ = max(merged, key=lambda r: r[1] - r[0])
        out.append((x_lo + a_, x_lo + b_))
    return out


@dataclass
class Segment:
    x0: int
    y0: int
    x1: int
    y1: int
    lightness: float
    chroma: float
    hue: float
    rgb: Tuple[int, int, int]
    area: int
    edge: bool = False

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2.0


def vertical_segments(mask: np.ndarray, an: "PageAnalysis", p: float, min_len: float) -> List[Segment]:
    vo = odd(max(5, p * 0.9))
    ver = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vo)))
    ver = cv2.morphologyEx(ver, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (odd(max(3, p * 0.1)), odd(max(3, p * 0.3)))))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ver, connectivity=8)
    segs: List[Segment] = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if hh < min_len or ww > 0.6 * p:
            continue
        sl = (slice(y, y + hh), slice(x, x + ww))
        sel = lab[sl] == i
        L_, C_, H_, rgb = _stats(an, sl, sel)
        segs.append(Segment(int(x), int(y), int(x + ww), int(y + hh), L_, C_, H_, rgb, int(area)))
    return segs


def chain_segments(segs: List[Segment], p: float) -> List[List[Segment]]:
    """Group vertical segments lying on one straight, steep line (a page edge or gutter crease
    broken up by shadows). Segments side by side (a double pen line) are not chained."""
    chains: List[List[Segment]] = []
    for s_ in sorted(segs, key=lambda s: s.y0):
        best, bd = None, 1e9
        for ch in chains:
            last = ch[-1]
            gap = s_.y0 - last.y1
            if gap < -0.5 * p or gap > 4.0 * p:
                continue
            if len(ch) >= 2:
                ys = np.array([(c.y0 + c.y1) / 2.0 for c in ch])
                xs = np.array([c.xc for c in ch])
                pred = np.polyval(np.polyfit(ys, xs, 1), (s_.y0 + s_.y1) / 2.0)
            else:
                pred = last.xc
            d = abs(s_.xc - pred)
            if d < 0.45 * p + 0.12 * max(0, gap) and d < bd:
                best, bd = ch, d
        if best is None:
            chains.append([s_])
        else:
            best.append(s_)
    return chains


def _chain_line(ch: List[Segment]) -> Tuple[float, float]:
    ys = np.array([(c.y0 + c.y1) / 2.0 for c in ch], dtype=np.float64)
    xs = np.array([c.xc for c in ch], dtype=np.float64)
    if len(ch) == 1 or np.ptp(ys) < 1:
        return float(xs.mean()), 0.0
    b, a = np.polyfit(ys, xs, 1)
    return float(a), float(b)


def _facing_runs(ink: np.ndarray, blk, h: int, w: int) -> List[Tuple[int, int]]:
    """Dense text columns other than the main block, seen in the white-balanced ink (finds a
    facing page even when it lies in shadow)."""
    from .geometry import _transitions, _bridge
    rows = slice(max(0, blk.y0), min(h, blk.y1))
    col = smooth1d(_transitions(ink[rows], 0), w * 0.003)
    main = col[blk.x0:blk.x1]
    if main.size == 0 or main.max() <= 0:
        return []
    ref = float(np.percentile(main, 75))
    out = []
    for s_, e_ in _bridge(runs(col > 0.3 * ref), w * 0.006):
        if (e_ <= blk.x0 + 2 or s_ >= blk.x1 - 2) and e_ - s_ >= 0.015 * w:
            out.append((int(s_), int(e_)))
    return out


def build_layout(an: "PageAnalysis", geo: Geometry) -> Tuple[Layout, List[Segment]]:
    blk = geo.block
    p = geo.pitch
    h, w = an.Lw.shape
    W = blk.width
    ink = an.ink & an.valid
    rows = np.arange(h, dtype=np.float64)
    # 1. facing page / gutter (constant per side)
    gl, gr = -1.0, -1.0
    for s_, e_ in sorted(set(blk.other_runs) | set(_facing_runs(ink, blk, h, w))):
        if e_ <= blk.x0 + 2:
            gl = max(gl, (e_ + blk.x0) / 2.0)
        elif s_ >= blk.x1 - 2:
            gr = (s_ + blk.x1) / 2.0 if gr < 0 else min(gr, (s_ + blk.x1) / 2.0)
    lo = int(max(0, blk.x0 - 0.3 * W, gl if gl >= 0 else 0))
    hi = int(min(w, blk.x1 + 0.3 * W, gr if gr >= 0 else w))
    # 2. long vertical strokes (pen margin lines, page edges, creases) - kept out of the line extents
    pen = (an.ink | (an.tint & (an.Lw < 80.0))) & an.valid
    segs = vertical_segments(pen & _row_mask(h, w, np.full(h, lo), np.full(h, hi)), an, p, 1.0 * p)
    ink_text = ink.copy()
    for sg in segs:
        ink_text[sg.y0:sg.y1, max(0, sg.x0 - 2):sg.x1 + 2] = False
    # 3. each line's own extent, the straight column edges, the text zone per row
    ext = line_extents(ink_text, an.lines, lo, hi, p)
    full = [(ln, e) for ln, e in zip(an.lines, ext) if e is not None and (e[1] - e[0]) > 0.5 * W]
    if full:
        vs = np.array([ln.center for ln, _ in full])
        xl = _robust_edge(vs, np.array([e[0] for _, e in full], dtype=np.float64), "left", p)
        xr = _robust_edge(vs, np.array([e[1] for _, e in full], dtype=np.float64), "right", p)
    else:
        xl, xr = (float(blk.x0), 0.0), (float(blk.x1), 0.0)
    col_l = xl[0] + xl[1] * rows
    col_r = xr[0] + xr[1] * rows
    text_l, text_r = col_l.copy(), col_r.copy()
    for ln, e in zip(an.lines, ext):
        if e is None:
            continue
        r0, r1 = int(max(0, ln.top)), int(min(h, ln.bottom + 1))
        # a line may poke out a little (hanging punctuation, an outdent); more than that is a
        # note or a mark written next to the line, not text
        text_l[r0:r1] = np.minimum(text_l[r0:r1], np.maximum(e[0], col_l[r0:r1] - 0.6 * p))
        text_r[r0:r1] = np.maximum(text_r[r0:r1], np.minimum(e[1], col_r[r0:r1] + 0.6 * p))
    # 4. page edges / gutter creases: long straight chains of vertical strokes
    block_h = max(1.0, blk.y1 - blk.y0)
    edge_l = np.full(h, -1e9)
    edge_r = np.full(h, 1e9)
    vmid = (blk.y0 + blk.y1) / 2.0
    for ch in chain_segments(segs, p):
        span = max(c.y1 for c in ch) - min(c.y0 for c in ch)
        coloured = float(np.median([c.chroma for c in ch])) >= 25.0
        if not coloured and (span >= 0.45 * block_h or (len(ch) >= 3 and span >= 0.3 * block_h)):
            a_, b_ = _chain_line(ch)
            xs_ = a_ + b_ * rows
            mid = a_ + b_ * vmid
            for c in ch:
                c.edge = True
            if mid < xl[0] + xl[1] * vmid:
                edge_l = np.maximum(edge_l, xs_ + 0.15 * p)
            elif mid > xr[0] + xr[1] * vmid:
                edge_r = np.minimum(edge_r, xs_ - 0.15 * p)
    # 5. the paper's outer edges (a lasting step from bright paper to something darker)
    edges = find_page_edges(an.L, an.valid, an.lines, col_l, col_r, p, W)
    if edges["left"] is not None:
        a_, b_ = edges["left"]
        edge_l = np.maximum(edge_l, a_ + b_ * rows + 0.1 * p)
    if edges["right"] is not None:
        a_, b_ = edges["right"]
        edge_r = np.minimum(edge_r, a_ + b_ * rows - 0.1 * p)
    vl = np.where(an.valid.any(axis=1), np.argmax(an.valid, axis=1), 0).astype(np.float64)
    vr = np.where(an.valid.any(axis=1), w - np.argmax(an.valid[:, ::-1], axis=1), w).astype(np.float64)
    page_l = np.maximum.reduce([col_l - 0.25 * W, edge_l, np.full(h, gl if gl >= 0 else -1e9), vl])
    page_r = np.minimum.reduce([col_r + 0.25 * W, edge_r, np.full(h, gr if gr >= 0 else 1e9), vr])
    page_l = np.clip(page_l, 0, w - 1)
    page_r = np.clip(page_r, page_l + 1, w)
    # 6. top and bottom of the page
    first, last = (an.lines[0], an.lines[-1]) if an.lines else (None, None)
    xmid = (np.median(col_l) + np.median(col_r)) / 2.0
    page_top = int(first.top - 14.0 * p) if first else 0
    page_bottom = int(last.bottom + 10.0 * p) if last else h - 1
    if edges["top"] is not None:
        page_top = max(page_top, int(edges["top"][0] + edges["top"][1] * xmid + 0.1 * p))
    if edges["bottom"] is not None:
        page_bottom = min(page_bottom, int(edges["bottom"][0] + edges["bottom"][1] * xmid - 0.1 * p))
    # the photo's own border is applied per pixel (an.valid) - a slanted top edge must not cut
    # off a page number in the corner
    page_top, page_bottom = max(0, page_top), min(h - 1, page_bottom)
    lay = Layout(page_l, page_r, text_l, text_r, int(page_top), int(page_bottom), gl, gr, xl, xr)
    lay.edges = edges
    return lay, segs


# ---------------------------------------------------------------------------
# print or handwriting? (top and bottom margins hold printed running heads, page numbers,
# chapter headings and footnotes next to the reader's own notes)
# ---------------------------------------------------------------------------

@dataclass
class InkGroup:
    x0: int
    y0: int
    x1: int
    y1: int
    comps: List[Tuple[int, int, int, int, int]]   # x0, y0, x1, y1, area
    handwriting: bool = False
    why: str = ""


def _rows_of(comps, tol_frac: float = 0.5) -> List[List[Tuple[int, int, int, int, int]]]:
    rows: List[List] = []
    for c in sorted(comps, key=lambda c: (c[1] + c[3]) / 2.0):
        yc = (c[1] + c[3]) / 2.0
        hh = c[3] - c[1]
        placed = False
        for r in rows:
            ry0 = np.median([q[1] for q in r]); ry1 = np.median([q[3] for q in r])
            rh = max(1.0, ry1 - ry0)
            if abs(yc - (ry0 + ry1) / 2.0) <= tol_frac * max(rh, hh):
                r.append(c)
                placed = True
                break
        if not placed:
            rows.append([c])
    return rows


def _segments_of(row, gap_px: float):
    row = sorted(row, key=lambda c: c[0])
    out = [[row[0]]]
    for c in row[1:]:
        if c[0] - max(q[2] for q in out[-1]) > gap_px:
            out.append([c])
        else:
            out[-1].append(c)
    return out


def midline_crossings(mask: np.ndarray) -> int:
    """How many separate ink runs cross the middle third of a component (max over three rows):
    a printed letter has at most four (m, w, M, W); a joined handwritten word has many."""
    h = mask.shape[0]
    best = 0
    for f in (0.4, 0.5, 0.6):
        row = mask[min(h - 1, int(h * f))].astype(np.int8)
        best = max(best, int((np.diff(row, prepend=0) == 1).sum()))
    return best


def classify_group(comps, xh: float) -> Tuple[bool, str]:
    """Handwriting (True) or print (False) for one horizontal group of ink components.

    Print sits on a common baseline (most letter bottoms within a few px) and consists of
    separate letters; handwriting has wandering bottoms, slanted rows and joined (cursive)
    letters several times wider than tall."""
    comps = [c for c in comps if len(c) < 7 or c[6]]
    if not comps:
        return False, "fragments only"
    hs = np.array([c[3] - c[1] for c in comps], dtype=np.float64)
    ws = np.array([c[2] - c[0] for c in comps], dtype=np.float64)
    areas = np.array([c[4] for c in comps], dtype=np.float64)
    body = hs >= 0.3 * xh                       # ignore dots, commas, thin rules
    if body.sum() == 0:
        return False, "no letters"
    cross = np.array([c[5] if len(c) > 5 else 0 for c in comps])
    multi = (cross >= 5) & (ws >= 1.5 * xh)
    if multi.any():
        return True, f"joined letters ({int(cross.max())} crossings)"
    tall_wide = (ws >= 2.2 * hs) & (hs >= 0.45 * xh)
    wide_frac = areas[tall_wide].sum() / max(1.0, areas[body].sum())
    if wide_frac >= 0.35:
        return True, f"joined strokes {wide_frac:.2f}"
    n = int(body.sum())
    if n >= 4:
        bottoms = np.array([c[3] for c in comps], dtype=np.float64)[body]
        xs = np.array([(c[0] + c[2]) / 2.0 for c in comps], dtype=np.float64)[body]
        H = float(np.median(hs[body]))
        # baseline through the bottoms (descenders are a minority)
        if np.ptp(xs) > 0:
            bb, aa = np.polyfit(xs, bottoms, 1)
        else:
            bb, aa = 0.0, float(np.median(bottoms))
        res = bottoms - (aa + bb * xs)
        aligned = float(np.mean(np.abs(res - np.median(res)) <= max(2.0, 0.09 * H)))
        if aligned < 0.65:
            return True, f"baseline {aligned:.2f}"
        return False, f"print baseline {aligned:.2f}"
    # one to three letter-sized pieces: digits of a page or chapter number, an ornament
    if np.any(tall_wide):
        return True, "joined strokes (short)"
    return False, "short print-like"


def _merge_print_groups(groups: List[InkGroup]) -> List[InkGroup]:
    """Print groups that touch or overlap side by side are one word or number (a page number
    whose digits fell into two groups would otherwise be read as two numbers)."""
    groups = list(groups)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            a = groups[i]
            if a.handwriting:
                continue
            for j in range(i + 1, len(groups)):
                b = groups[j]
                if b.handwriting:
                    continue
                hh = max(a.y1 - a.y0, b.y1 - b.y0)
                gap = max(a.x0, b.x0) - min(a.x1, b.x1)
                yov = min(a.y1, b.y1) - max(a.y0, b.y0)
                if gap < 0.6 * hh and yov > 0.3 * min(a.y1 - a.y0, b.y1 - b.y0):
                    a.x0, a.y0, a.x1, a.y1 = min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1)
                    a.comps = a.comps + b.comps
                    a.why = a.why + " + merged"
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return groups


def head_foot_notes(an: "PageAnalysis", geo: Geometry, lay: Layout, zone: Tuple[int, int]) -> List[InkGroup]:
    """Handwritten groups in a top or bottom margin zone (rows zone[0]..zone[1])."""
    p = geo.pitch
    h, w = an.Lw.shape
    y0, y1 = max(0, zone[0]), min(h, zone[1])
    if y1 - y0 < 0.3 * p:
        return []
    mask = note_ink(an)
    sub = np.zeros_like(mask)
    sub[y0:y1] = mask[y0:y1]
    sub &= _row_mask(h, w, lay.page_l + 0.2 * p, lay.page_r - 0.2 * p)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 8 or max(ww, hh) < 0.08 * p:
            continue
        cm = lab[y:y + hh, x:x + ww] == i
        if not stroke_like(cm, p):
            continue
        # small fragments (thin print broken by the ink threshold) join their word but do not
        # vote on print/handwriting
        major = area >= (0.05 * p) ** 2 * 4 and max(ww, hh) >= 0.12 * p
        comps.append((int(x), int(y), int(x + ww), int(y + hh), int(area), midline_crossings(cm), major))
    if not comps:
        return []
    xh = 0.42 * p  # body x-height is roughly 0.4 of the line pitch
    out: List[InkGroup] = []
    for row in _rows_of(comps):
        H = float(np.median([c[3] - c[1] for c in row]))
        for seg in _segments_of(row, max(1.6 * H, 0.9 * p)):
            hw, why = classify_group(seg, xh)
            g = InkGroup(min(c[0] for c in seg), min(c[1] for c in seg), max(c[2] for c in seg), max(c[3] for c in seg), seg, hw, why)
            out.append(g)
    # a handwritten group often spans two rows (ascenders, a second line): merge nearby notes
    out = _merge_print_groups(out)
    for g in out:
        if g.handwriting:
            if not on_paper(an, g.x0, g.y0, g.x1, g.y1, sub, p):
                g.handwriting, g.why = False, g.why + " | not on paper"
            elif (g.x1 - g.x0) < 0.8 * p or sum(c[4] for c in g.comps if len(c) < 7 or c[6]) < 0.1 * p * 1.2 * p:
                g.handwriting, g.why = False, g.why + " | too small"
    an.margin_groups.extend(out)
    notes = [g for g in out if g.handwriting]
    # a note absorbs the loose bits around it (loops, dots, strokes that fell into tiny groups);
    # real print next to it (a running head: a regular baseline) stays separate
    weak = [g for g in out if not g.handwriting and not g.why.startswith("print baseline")]
    for nt in notes:
        grown = True
        while grown:
            grown = False
            for g in weak:
                if g is nt:
                    continue
                if (min(nt.x1 + 0.3 * p, g.x1) > max(nt.x0 - 0.3 * p, g.x0)
                        and min(nt.y1 + 0.3 * p, g.y1) > max(nt.y0 - 0.3 * p, g.y0)
                        and not (g.x0 >= nt.x0 and g.x1 <= nt.x1 and g.y0 >= nt.y0 and g.y1 <= nt.y1)):
                    nt.x0, nt.y0, nt.x1, nt.y1 = min(nt.x0, g.x0), min(nt.y0, g.y0), max(nt.x1, g.x1), max(nt.y1, g.y1)
                    grown = True
    merged: List[InkGroup] = []
    for g in sorted(notes, key=lambda g: g.y0):
        for m in merged:
            if g.y0 - m.y1 < 0.8 * p and min(g.x1, m.x1) - max(g.x0, m.x0) > -1.5 * p:
                m.x0, m.y0, m.x1, m.y1 = min(m.x0, g.x0), min(m.y0, g.y0), max(m.x1, g.x1), max(m.y1, g.y1)
                m.comps += g.comps
                m.why += "; " + g.why
                break
        else:
            merged.append(InkGroup(g.x0, g.y0, g.x1, g.y1, list(g.comps), True, g.why))
    return merged


def note_ink(an: "PageAnalysis") -> np.ndarray:
    """Ink a reader writes with: dark pen, or vivid coloured pen. Skin (fingers holding the book)
    is a warm, moderately saturated colour and is left out."""
    C = np.hypot(an.aw, an.bw)
    hue = (np.degrees(np.arctan2(an.bw, an.aw)) + 360.0) % 360.0
    warm = (hue < 70.0) | (hue > 345.0)
    vivid = np.where(warm, C >= 32.0, C >= 22.0)
    return (an.ink | (an.tint & vivid & (an.Lw < 72.0))) & an.valid


def stroke_like(mask: np.ndarray, p: float) -> bool:
    """Pen strokes are thin: most of the ink lies within ~0.08 pitch of its edge (a finger,
    a shadow or a dark object is a thick blob)."""
    if mask.sum() == 0:
        return False
    dt = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    return float(np.percentile(dt[mask], 90)) <= max(2.5, 0.09 * p)


def on_paper(an: "PageAnalysis", x0: int, y0: int, x1: int, y1: int, inkmask: np.ndarray, p: float) -> bool:
    """The ground around a note is bare paper: bright and neutral after the white balance."""
    h, w = an.Lw.shape
    pad = int(0.3 * p)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    sub_ink = cv2.dilate(inkmask[Y0:Y1, X0:X1].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ring = ~sub_ink & an.valid[Y0:Y1, X0:X1]
    if ring.sum() < 50 or not an.valid[Y0:Y1, X0:X1].all():
        return False
    Lr = an.Lw[Y0:Y1, X0:X1][ring]
    Cr = np.hypot(an.aw[Y0:Y1, X0:X1][ring], an.bw[Y0:Y1, X0:X1][ring])
    return float(np.median(Lr)) >= 82.0 and float(np.median(Cr)) <= 10.0


# ---------------------------------------------------------------------------
# masks and components
# ---------------------------------------------------------------------------

def _stats(an: "PageAnalysis", sl, sel):
    aw, bw, Lw = an.aw[sl][sel], an.bw[sl][sel], an.Lw[sl][sel]
    if len(aw) == 0:
        return 0.0, 0.0, 0.0, (0, 0, 0)
    a_m, b_m, L_m = float(np.median(aw)), float(np.median(bw)), float(np.median(Lw))
    rgb = an.flat[sl][sel].reshape(-1, 3)
    return L_m, float(np.hypot(a_m, b_m)), float(np.degrees(np.arctan2(b_m, a_m)) % 360.0), tuple(int(v) for v in np.median(rgb, axis=0))


def _mark(kind, x0, y0, x1, y1, L_, C_, H_, rgb, area) -> Mark:
    return Mark(kind, int(x0), int(y0), int(x1), int(y1), color_name(H_, C_, L_), H_, C_, L_, rgb, int(area), yc=(y0 + y1) / 2.0)


def skin_like(hue: float, L_: float) -> bool:
    """A finger over the page: a warm hue and too dark for highlighter. Highlighter is
    translucent, so even orange, pink and red ink on paper stays light (white-balanced
    L* about 70-97); skin, in light or shade, is darker (L* about 30-62)."""
    return 5.0 <= hue <= 60.0 and L_ < 66.0


def pen_like(L_: float, C_: float) -> bool:
    return L_ <= 45.0 or (C_ >= 22.0 and L_ <= 78.0)


def _row_mask(h: int, w: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    xs = np.arange(w)[None, :]
    return (xs >= np.asarray(lo)[:, None]) & (xs < np.asarray(hi)[:, None])


def _on_lines(an: "PageAnalysis", y0: float, y1: float, p: float) -> bool:
    return any(min(y1, ln.center + 0.25 * p) - max(y0, ln.center - 0.25 * p) > 0.2 * p for ln in an.lines)


def _off_page(an: "PageAnalysis", lay: "Layout", x: int, y: int, comp: np.ndarray, p: float) -> bool:
    """A tinted blob that is cut off by the page's edge and runs on beyond it: a finger or a hand holding the book. A highlight,
    however tall, ends on the paper; this one reaches the page edge and the tint goes on past it (over several rows or columns)."""
    h, w = an.tint.shape
    reach = max(3, int(0.3 * p))
    hh, ww = comp.shape
    on = 0
    for k in range(hh):
        cols = np.nonzero(comp[k])[0]
        if cols.size == 0:
            continue
        r = y + k
        left, right = int(lay.page_l[r]), int(lay.page_r[r])
        if x + cols.max() >= right - 2 and an.tint[r, right:min(w, right + reach)].mean() > 0.5:
            on += 1
        elif x + cols.min() <= left + 2 and left > 0 and an.tint[r, max(0, left - reach):left].mean() > 0.5:
            on += 1
    if on >= max(2, 0.25 * p):
        return True
    for k in range(ww):                                   # a thumb on the top or bottom edge
        rows = np.nonzero(comp[:, k])[0]
        if rows.size == 0:
            continue
        c = x + k
        if y + rows.max() >= lay.page_bottom - 2 and an.tint[lay.page_bottom:min(h, lay.page_bottom + reach), c].mean() > 0.5:
            on += 1
        elif y + rows.min() <= lay.page_top + 2 and lay.page_top > 0 and an.tint[max(0, lay.page_top - reach):lay.page_top, c].mean() > 0.5:
            on += 1
    return on >= max(2, 0.25 * p)


def skin_coloured(hue, chroma):
    """Warm and moderately saturated, as skin is in any light; a highlighter is far more saturated (works on arrays too)."""
    return ((hue < 75.0) | (hue > 340.0)) & (chroma > 8.0) & (chroma < 48.0)


def find_skin(an: "PageAnalysis", p: float) -> None:
    """Fill an.skin: the fingers (or the hand) holding the book, found before the marks so that a highlight a thumb touches keeps
    its own pixels, and so the stripes can be cleaned. Skin is warm, moderately saturated and thick: an opening as wide as a line
    leaves it and removes highlight bands, pen strokes and print; what remains must be cut off by the page's edge and go on past
    it (``_off_page``)."""
    lay = an.layout
    h, w = an.Lw.shape
    an.skin = np.zeros((h, w), bool)
    an.skin_boxes = []
    if lay is None:
        return
    C = np.hypot(an.aw, an.bw)
    hue = (np.degrees(np.arctan2(an.bw, an.aw)) + 360.0) % 360.0
    warm = (skin_coloured(hue, C) & (an.Lw < 88.0) & an.valid).astype(np.uint8)
    kc = odd(max(3, p * 0.2))
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    ko = odd(max(5, p * 0.95))
    core = cv2.morphologyEx(warm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < (0.8 * p) ** 2:
            continue
        sl = (slice(y, y + hh), slice(x, x + ww))
        comp = lab[sl] == i
        if not _off_page(an, lay, x, y, comp & an.tint[sl], p):
            continue
        an.skin[sl] |= fill_holes(comp)
        an.skin_boxes.append((int(x), int(y), int(x + ww), int(y + hh)))
    if an.skin.any():                                     # the opening shaved the finger's rim: grow it back over warm pixels only
        d = odd(max(3, p * 0.3))
        grown = cv2.dilate(an.skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))) > 0
        an.skin = cv2.dilate(((grown & (warm > 0)) | an.skin).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def detect_marks(an: "PageAnalysis", geo: Geometry, segs: List[Segment]) -> None:
    """Fill an.marks (and an.rejected, for the review sheet)."""
    p = geo.pitch
    blk = geo.block
    h, w = an.Lw.shape
    lay = an.layout
    first = an.lines[0] if an.lines else None
    last = an.lines[-1] if an.lines else None
    body_top = int(first.top) if first else blk.y0
    body_bot = int(last.bottom) if last else blk.y1
    search = _row_mask(h, w, lay.page_l, lay.page_r) & an.valid
    search[:lay.page_top] = False
    search[lay.page_bottom + 1:] = False
    textzone = _row_mask(h, w, lay.text_l - 0.15 * p, lay.text_r + 0.15 * p)
    textzone[:max(0, int(body_top - 0.2 * p))] = False
    textzone[int(body_bot + 0.2 * p):] = False

    marks: List[Mark] = []
    rejected: List[Mark] = []

    find_skin(an, p)

    # ---- tinted strokes on the printed lines: highlights (and coloured underlines) ---------------
    tint = an.tint & search & ~an.skin
    ks = odd(max(3, p * 0.3))
    near_skin = cv2.dilate(an.skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))) > 0
    ko = odd(max(3, p * 0.07))
    tint = cv2.morphologyEx(tint.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))) > 0
    kc = (odd(max(3, p * 0.45)), odd(max(3, p * 0.18)))
    tint_c = cv2.morphologyEx(tint.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, kc)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tint_c.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < (p * 0.25) ** 2:
            continue
        sl = (slice(y, y + hh), slice(x, x + ww))
        comp = lab[sl] == i
        sel = comp & tint[sl]
        L_, C_, H_, rgb = _stats(an, sl, sel)
        m = _mark("highlight", x, y, x + ww, y + hh, L_, C_, H_, rgb, sel.sum())
        if hh >= 1.0 * p and ww <= 0.6 * p and hh >= 2.5 * ww:
            continue  # a vertical coloured stroke: a margin line (handled below)
        if skin_coloured(H_, C_) and _off_page(an, lay, x, y, sel, p):
            m.why = "cut off by the page edge (a finger?)"
            rejected.append(m)
            continue
        if hh > ww and near_skin[sl][comp].any():
            m.why = "a tall tint against a finger (its shadow)"
            rejected.append(m)
            continue
        inside = float(textzone[sl][comp].mean())
        if ww >= 2.0 * p and hh <= 0.35 * p and L_ < 70:
            m.kind = "underline"
            if inside < 0.6 or not _on_lines(an, y - 0.9 * p, y + hh, p):
                rejected.append(m)
                continue
        elif (ww < 1.0 * p or hh < 0.15 * p or inside < 0.5 or C_ < HL_COMPONENT_CHROMA
              or not _on_lines(an, y, y + hh, p)
              or (hh > 1.6 * p and (area / float(ww * hh)) > 0.85 and ww < 4 * p)
              or skin_like(H_, L_)):
            rejected.append(m)
            continue
        marks.append(m)

    # ---- dark underlines ----------------------------------------------------------------------
    ink = an.ink & search
    lo = odd(max(5, p * 0.6))
    hor = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (lo, 1)))
    hor = cv2.morphologyEx(hor, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (odd(max(3, p * 0.2)), odd(max(3, p * 0.1)))))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hor, connectivity=8)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if ww < 2.0 * p or hh > 0.5 * p:
            continue
        sl = (slice(y, y + hh), slice(x, x + ww))
        sel = lab[sl] == i
        L_, C_, H_, rgb = _stats(an, sl, sel)
        m = _mark("underline", x, y, x + ww, y + hh, L_, C_, H_, rgb, sel.sum())
        above = an.ink[max(0, int(y - 0.9 * p)):max(0, int(y - 0.12 * p)), x:x + ww]
        density = above.mean() if above.size else 0.0
        inside = float(textzone[sl][sel].mean())
        if density < 0.04 or inside < 0.6:
            rejected.append(m)
            continue
        marks.append(m)

    # ---- margin lines -------------------------------------------------------------------------
    block_h = max(1.0, blk.y1 - blk.y0)
    for sg in segs:
        m = _mark("margin_line", sg.x0, sg.y0, sg.x1, sg.y1, sg.lightness, sg.chroma, sg.hue, sg.rgb, sg.area)
        yc = int(np.clip((sg.y0 + sg.y1) // 2, 0, h - 1))
        if sg.edge:
            rejected.append(m)
            continue
        if not (lay.page_l[yc] + 0.15 * p <= sg.xc <= lay.page_r[yc] - 0.15 * p):
            continue
        if sg.y1 < body_top - 0.5 * p or sg.y0 > body_bot + 0.5 * p:
            continue
        if lay.text_l[yc] - 0.1 * p < sg.xc < lay.text_r[yc] + 0.1 * p:
            continue  # inside the text: printed glyphs (drop caps, brackets), not a margin line
        if not pen_like(sg.lightness, sg.chroma) or (sg.y1 - sg.y0 > 0.5 * block_h and sg.chroma < 25.0):
            rejected.append(m)
            continue
        marks.append(m)

    marks = _merge_same(marks, p)

    # ---- notes in the side margins (beside the printed lines) ----------------------------------
    marg = note_ink(an) & search & ~textzone
    marg[:max(0, int(body_top - 0.3 * p))] = False
    marg[int(body_bot + 0.3 * p):] = False
    for m in marks:
        if m.kind in ("margin_line", "highlight", "underline"):
            pad = int(0.15 * p)
            marg[max(0, m.y0 - pad):m.y1 + pad, max(0, m.x0 - pad):m.x1 + pad] = False
    for sg in segs:
        if sg.edge:
            marg[max(0, sg.y0 - 6):sg.y1 + 6, max(0, sg.x0 - 6):sg.x1 + 6] = False
    clear = _row_mask(h, w, lay.page_l + 0.35 * p, lay.page_r - 0.35 * p)
    kd = odd(max(3, p * 0.35))
    grp = cv2.dilate(marg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grp, connectivity=8)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        sl = (slice(y, y + hh), slice(x, x + ww))
        comp = lab[sl] == i
        sel = comp & marg[sl]
        px = int(sel.sum())
        if px < (p * 0.12) * (p * 0.8) or max(ww, hh) < 0.6 * p:
            continue
        L_, C_, H_, rgb = _stats(an, sl, sel)
        m = _mark("margin_note", x, y, x + ww, y + hh, L_, C_, H_, rgb, px)
        m.where = "side"
        lone_tall = ww < 1.0 * p and hh > 2.5 * ww
        reasons = [r for r, bad in (("near page edge", not clear[sl][comp].all()), ("not pen", not pen_like(L_, C_)),
                                    ("too tall", hh > 12 * p), ("lone tall stroke", lone_tall),
                                    ("thick", not stroke_like(sel, p)),
                                    ("not on paper", not on_paper(an, x, y, x + ww, y + hh, marg, p))) if bad]
        if reasons:
            m.why = ", ".join(reasons)
            rejected.append(m)
            continue
        marks.append(m)

    # ---- notes in the top and bottom margins ---------------------------------------------------
    an.zones = {"top": (lay.page_top + int(0.15 * p), int(body_top - 0.15 * p)),
                "bottom": (int(body_bot + 0.15 * p), lay.page_bottom - int(0.15 * p))}
    for where, zone in an.zones.items():
        for g in head_foot_notes(an, geo, lay, zone):
            sl = (slice(g.y0, g.y1), slice(g.x0, g.x1))
            sel = an.ink[sl] | an.tint[sl]
            L_, C_, H_, rgb = _stats(an, sl, sel)
            m = _mark("margin_note", g.x0, g.y0, g.x1, g.y1, L_, C_, H_, rgb, int(sel.sum()))
            m.where = where
            m.why = g.why
            marks.append(m)

    an.marks = sorted(marks, key=lambda m: (m.y0, m.x0))
    an.rejected = rejected


def _merge_same(marks: List[Mark], p: float) -> List[Mark]:
    out: List[Mark] = []
    for m in sorted(marks, key=lambda m: -m.area):
        dup = False
        for o in out:
            if o.kind != m.kind:
                continue
            ix = min(o.x1, m.x1) - max(o.x0, m.x0)
            iy = min(o.y1, m.y1) - max(o.y0, m.y0)
            if ix > -0.05 * p and iy > -0.05 * p:
                o.x0, o.y0, o.x1, o.y1 = min(o.x0, m.x0), min(o.y0, m.y0), max(o.x1, m.x1), max(o.y1, m.y1)
                o.yc = (o.y0 + o.y1) / 2.0
                if o.chroma < m.chroma:
                    o.color, o.hue, o.chroma, o.rgb = m.color, m.hue, m.chroma, m.rgb
                o.area += m.area
                dup = True
                break
        if not dup:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# page analysis entry point
# ---------------------------------------------------------------------------

def body_letter_height(ink: np.ndarray, lines: List[TextLine], blk, p: float) -> float:
    """Median height of the letters on the body lines (about the x-height)."""
    hs: List[int] = []
    for ln in lines[: 12]:
        r0, r1 = int(max(0, ln.center - 0.6 * p)), int(ln.center + 0.6 * p)
        sub = ink[r0:r1, blk.x0:blk.x1].astype(np.uint8)
        n, _lab, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        for i in range(1, n):
            w_, h_ = stats[i][2], stats[i][3]
            if 0.15 * p <= h_ <= 0.9 * p and w_ <= 1.5 * p:
                hs.append(int(h_))
    return float(np.median(hs)) if hs else 0.38 * p


def analyse(flat: np.ndarray, geo: Geometry) -> PageAnalysis:
    h, w = flat.shape[:2]
    valid = flat.sum(axis=2) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    Lw, aw, bw, paper_L, paper_px = white_balance(flat, valid, geo.block, geo.pitch)
    paper = paper_region(paper_px, valid, geo.block, geo.pitch)
    usable = paper_L > 12.0
    ink = (Lw < INK_L) & valid & usable
    tint = (np.hypot(aw, bw) > HL_CHROMA) & valid & usable & (Lw > 25.0)
    an = PageAnalysis(flat, Lw, aw, bw, paper_L, valid, paper, ink, tint)
    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB)
    an.L = lab[..., 0].astype(np.float32) * (100.0 / 255.0)
    blk = geo.block
    cand = text_lines(ink & valid, blk.x0, blk.x1, blk.y0, blk.y1, geo.pitch, tint)
    # the texture test on the print alone: under a dark highlighter the whole band is "ink"; its letters are what is clearly
    # darker, in the brightest channel, than the band beside them on the same rows
    V = flat.max(axis=2)
    band = cv2.dilate(V, cv2.getStructuringElement(cv2.MORPH_RECT, (odd(geo.pitch), 3))).astype(np.float32)
    letters = (ink & ~tint) | (tint & (V < 0.75 * band))
    an.lines, an.head_lines = validate_lines(cand, letters & valid, blk, geo.pitch)
    lay, segs = build_layout(an, geo)
    an.layout = lay
    an.letter_h = body_letter_height(ink, an.lines, blk, geo.pitch)
    detect_marks(an, geo, segs)
    an.limit_x0 = int(np.median(lay.page_l[blk.y0:blk.y1])) if blk.y1 > blk.y0 else int(lay.page_l.min())
    an.limit_x1 = int(np.median(lay.page_r[blk.y0:blk.y1])) if blk.y1 > blk.y0 else int(lay.page_r.max())
    return an
