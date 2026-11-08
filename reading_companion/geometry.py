"""Page geometry: analysis scale, ink mask, deskew, main text block, text-line tracks, dewarp.

Frames
------
source    the full-resolution photo after EXIF orientation (what the user sees)
analysis  the source scaled by ``scale`` so that its long side is about ``ANALYSIS_LONG_SIDE``
rotated   the analysis frame rotated by ``-angle`` degrees around its centre on an expanded canvas
flat      the rotated frame with every column shifted vertically so that text lines are straight
          (the "dewarp"); all mark detection and band geometry happen in this frame

``Geometry.flat_to_source`` maps flat-frame points back to source pixels; stripes are cut from the
source with ``cv2.remap`` so they keep full resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

ANALYSIS_LONG_SIDE = 2000
MAX_SKEW_DEG = 12.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def to_lab(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CIE L*a*b* as float32 arrays: L in 0..100, a and b centred on 0."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab[..., 0] * (100.0 / 255.0), lab[..., 1] - 128.0, lab[..., 2] - 128.0


def odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def smooth1d(v: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.3:
        return v.astype(np.float64)
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(v.astype(np.float64), k, mode="same")


def runs(mask1d: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) index pairs of the True runs of a 1-D boolean array."""
    m = np.concatenate([[False], mask1d.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def local_peaks(v: np.ndarray, min_dist: int, min_height: float) -> List[int]:
    """Indices of local maxima at least ``min_dist`` apart (greedy by height)."""
    if len(v) < 3:
        return []
    cand = np.flatnonzero((v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]) & (v[1:-1] >= min_height)) + 1
    order = cand[np.argsort(-v[cand])]
    taken: List[int] = []
    blocked = np.zeros(len(v), dtype=bool)
    for i in order:
        if not blocked[i]:
            taken.append(int(i))
            blocked[max(0, i - min_dist + 1):i + min_dist] = True
    return sorted(taken)


# ---------------------------------------------------------------------------
# ink and paper
# ---------------------------------------------------------------------------

def background_lightness(L: np.ndarray, stroke_px: int) -> np.ndarray:
    """Upper envelope of L*: dark strokes thinner than ``stroke_px`` are closed away."""
    k = odd(max(5, stroke_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bg = cv2.morphologyEx(L, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(bg, (0, 0), k / 2.0)


def ink_mask(L: np.ndarray, bg: np.ndarray, rel: float = 0.25, min_drop: float = 8.0) -> np.ndarray:
    """Pixels clearly darker than their local background (printed text, pen ink)."""
    drop = bg - L
    return (drop > np.maximum(min_drop, rel * bg)) & (bg > 15.0)


def paper_like(L: np.ndarray, a: np.ndarray, b: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Bright, near-neutral pixels that are plausibly bare paper."""
    bright = L > bg - 6.0
    sat = np.hypot(a, b)
    return bright & (sat < 25.0) & (L > 25.0)


# ---------------------------------------------------------------------------
# deskew
# ---------------------------------------------------------------------------

def estimate_skew(ink: np.ndarray, weight: Optional[np.ndarray] = None,
                  max_deg: float = MAX_SKEW_DEG) -> Tuple[float, float]:
    """Text-line angle in degrees (positive: lines rise to the right) by projection-profile
    variance maximisation. Returns (angle, sharpness) where sharpness compares the best profile
    with the profile at 90 degrees off (a portrait page photographed sideways scores low)."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 200:
        return 0.0, 0.0
    if len(xs) > 250_000:
        sel = np.random.default_rng(0).choice(len(xs), 250_000, replace=False)
        xs, ys = xs[sel], ys[sel]
    w = None if weight is None else weight[ys, xs].astype(np.float64)
    xs = xs.astype(np.float64) - ink.shape[1] / 2.0
    ys = ys.astype(np.float64) - ink.shape[0] / 2.0
    span = int(np.hypot(*ink.shape)) + 4

    def score(deg: float) -> float:
        t = np.deg2rad(deg)
        # a line rising to the right by +deg has image-y decreasing with x: y' = y + x*tan
        yp = ys * np.cos(t) + xs * np.sin(t)
        h = np.bincount((yp + span / 2).astype(np.int64), weights=w, minlength=span)
        h = smooth1d(h, 1.0)
        return float(np.mean(h * h))

    coarse = np.arange(-max_deg, max_deg + 1e-6, 0.5)
    sc = [score(d) for d in coarse]
    best = float(coarse[int(np.argmax(sc))])
    fine = np.arange(best - 0.5, best + 0.5 + 1e-6, 0.05)
    sf = [score(d) for d in fine]
    best = float(fine[int(np.argmax(sf))])
    side = score(best + 90.0)
    return best, (max(sf) / side if side > 0 else 99.0)


def rotation_matrices(shape: Tuple[int, int], angle: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Affine matrix analysis->rotated (undoing a skew of ``angle`` degrees) on an expanded
    canvas, its inverse, and the rotated canvas size (w, h)."""
    h, w = shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    Minv = cv2.invertAffineTransform(M)
    return M, Minv, (nw, nh)


# ---------------------------------------------------------------------------
# text block
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    x0: int
    x1: int
    y0: int
    y1: int
    other_runs: List[Tuple[int, int]] = field(default_factory=list)  # facing-page text columns

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def _transitions(mask: np.ndarray, axis: int) -> np.ndarray:
    """Number of ink-run starts along ``axis`` (0: going down each column, 1: along each row)."""
    m = mask.astype(np.int8)
    if axis == 0:
        d = np.diff(m, axis=0, prepend=0)
        return (d == 1).sum(axis=0).astype(np.float64)
    d = np.diff(m, axis=1, prepend=0)
    return (d == 1).sum(axis=1).astype(np.float64)


def _bridge(rs: List[Tuple[int, int]], gap: float) -> List[List[int]]:
    out: List[List[int]] = []
    for s, e in rs:
        if out and s - out[-1][1] < gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def _split_gutter(col: np.ndarray, x0: int, x1: int) -> Tuple[int, int, List[Tuple[int, int]]]:
    width = x1 - x0
    if width < 50:
        return x0, x1, []
    med = float(np.median(col[x0:x1]))
    split: List[Tuple[int, int]] = []
    outer = int(width * 0.35)
    # right side
    seg = col[x1 - outer:x1]
    i = int(np.argmin(seg[: max(1, len(seg) - 3)]))
    vx = x1 - outer + i
    if seg[i] < 0.45 * med and col[vx:x1].max() > 0.6 * med and (x1 - vx) < 0.35 * width:
        split.append((vx, x1))
        x1 = vx
    # left side
    seg = col[x0:x0 + outer]
    i = int(np.argmin(seg[3:])) + 3 if len(seg) > 4 else 0
    vx = x0 + i
    if seg[i] < 0.45 * med and col[x0:vx + 1].max() > 0.6 * med and (vx - x0) < 0.35 * width:
        split.append((x0, vx))
        x0 = vx
    # trim the valley slope so the block ends where its own text ends
    thr = 0.25 * med
    while x1 - 1 > x0 and col[x1 - 1] < thr:
        x1 -= 1
    while x0 + 1 < x1 and col[x0] < thr:
        x0 += 1
    return x0, x1, split


def _text_rows(use: np.ndarray, x0: int, x1: int) -> Tuple[int, int]:
    """Vertical extent of the body text inside columns [x0, x1): rows whose ink, gathered over
    about one line pitch, covers a large share of the block width (printed lines do; keyboards,
    fingers, shadows and page edges do not). Gaps up to ~3.5 pitches (section breaks, headings)
    are bridged."""
    h = use.shape[0]
    sub = use[:, x0:x1]
    prof = smooth1d(sub.sum(axis=1).astype(np.float64), 1.0)
    pitch = estimate_pitch(prof, lo=max(8, int(h * 0.008)), hi=int(h * 0.12)) or h * 0.028
    k = max(3, int(round(pitch)))
    # a column "has ink near row y" if any ink lies within +-pitch/2: dilate vertically
    dil = cv2.dilate(sub.astype(np.uint8), np.ones((k, 1), np.uint8)) > 0
    cover = smooth1d(dil.mean(axis=1), pitch * 0.25)
    good = cover > 0.40
    rs = _bridge(runs(good), pitch * 3.5)
    if not rs:
        return 0, 0
    mass = [cover[s:e].sum() for s, e in rs]
    b = int(np.argmax(mass))
    y0, y1 = rs[b]
    # the dilation widened each run by about half a pitch on both sides
    return int(y0 + pitch * 0.3), int(y1 - pitch * 0.3)


def find_text_block(ink: np.ndarray, valid: np.ndarray) -> Optional[TextBlock]:
    """The main page's body text.

    Columns: count of ink-run starts down each column (text gives one or more per printed line;
    margin pen lines, the gutter shadow and page edges give few), smoothed; the dense run of
    columns with the largest mass is the main block, other runs are the facing page or clutter.
    Rows: count of ink-run starts along each row inside those columns.
    """
    h, w = ink.shape
    use = ink & valid
    col = smooth1d(_transitions(use, 0), w * 0.003)
    if col.max() <= 0:
        return None
    ref = np.percentile(col[col > 0], 90)
    merged = _bridge(runs(col > 0.25 * ref), w * 0.012)
    if not merged:
        return None
    masses = [col[s:e].sum() for s, e in merged]
    best = int(np.argmax(masses))
    x0, x1 = merged[best]
    others = [(s, e) for i, (s, e) in enumerate(merged) if i != best]
    # a facing page fused to the main block through a narrow gutter shows as a deep valley in
    # the outer part of the run followed by a rise into the other page's text
    x0, x1, split = _split_gutter(col, x0, x1)
    others.extend(split)
    others.sort()
    y0, y1 = _text_rows(use, x0, x1)
    if y1 <= y0:
        return None
    return TextBlock(int(x0), int(x1), int(y0), int(y1), others)


# ---------------------------------------------------------------------------
# text-line tracks and the dewarp field
# ---------------------------------------------------------------------------

def estimate_pitch(profile: np.ndarray, lo: int = 12, hi: int = 200) -> float:
    """Text line pitch (px) from the autocorrelation of a row profile."""
    p = profile - profile.mean()
    if not np.any(p):
        return 0.0
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = min(hi, len(ac) - 2)
    if hi <= lo:
        return 0.0
    seg = ac[lo:hi]
    peaks = local_peaks(seg, 3, 0.05)
    if not peaks:
        return 0.0
    first = peaks[0] + lo
    # prefer the fundamental: the first strong autocorrelation peak
    strong = [pk + lo for pk in peaks if seg[pk] > 0.5 * seg[peaks].max()]
    cand = strong[0] if strong else first
    # sub-pixel refine
    i = cand
    if 1 <= i < len(ac) - 1:
        d = ac[i - 1] - 2 * ac[i] + ac[i + 1]
        if d != 0:
            return float(i + 0.5 * (ac[i - 1] - ac[i + 1]) / d)
    return float(i)


@dataclass
class LineTrack:
    xs: np.ndarray
    ys: np.ndarray


def track_lines(ink: np.ndarray, block: TextBlock, pitch: float, n_slices: int = 8) -> List[LineTrack]:
    """Follow every text line across vertical slices of the block (ink-profile peaks linked
    slice to slice)."""
    x0, x1 = block.x0, block.x1
    y0 = max(0, int(block.y0 - pitch))
    y1 = min(ink.shape[0], int(block.y1 + pitch))
    n = max(3, n_slices)
    edges = np.linspace(x0, x1, n + 1).astype(int)
    slice_peaks: List[Tuple[float, np.ndarray]] = []
    for i in range(n):
        sx0, sx1 = edges[i], edges[i + 1]
        prof = ink[y0:y1, sx0:sx1].sum(axis=1).astype(np.float64)
        prof = smooth1d(prof, max(1.0, pitch * 0.12))
        if prof.max() <= 0:
            slice_peaks.append(((sx0 + sx1) / 2.0, np.array([])))
            continue
        pk = local_peaks(prof, max(2, int(pitch * 0.6)), prof.max() * 0.12)
        slice_peaks.append(((sx0 + sx1) / 2.0, np.array(pk, dtype=np.float64) + y0))
    # link: start from the slice with most peaks, extend left and right
    tracks: List[dict] = []
    order = np.argsort([-len(p) for _, p in slice_peaks])
    used = [np.zeros(len(p), dtype=bool) for _, p in slice_peaks]
    tol = pitch * 0.35
    for si in order:
        cx, pk = slice_peaks[si]
        for j, y in enumerate(pk):
            if used[si][j]:
                continue
            used[si][j] = True
            pts = {int(si): float(y)}
            for direction in (-1, 1):
                prev_y, prev_i, slope = float(y), int(si), 0.0
                k = int(si) + direction
                while 0 <= k < n:
                    cxk, pkk = slice_peaks[k]
                    if len(pkk):
                        pred = prev_y + slope * (cxk - slice_peaks[prev_i][0])
                        d = np.abs(pkk - pred)
                        d[used[k]] = 1e9
                        m = int(np.argmin(d))
                        if d[m] < tol:
                            used[k][m] = True
                            new_y = float(pkk[m])
                            if len(pts) >= 1:
                                slope = 0.5 * slope + 0.5 * (new_y - prev_y) / (cxk - slice_peaks[prev_i][0])
                            pts[k] = new_y
                            prev_y, prev_i = new_y, k
                        elif abs(k - prev_i) > 2:
                            break
                    k += direction
            tracks.append(pts)
    out = []
    for pts in tracks:
        ks = sorted(pts)
        out.append(LineTrack(np.array([slice_peaks[k][0] for k in ks]), np.array([pts[k] for k in ks])))
    out.sort(key=lambda t: float(np.median(t.ys)))
    return out


@dataclass
class DewarpField:
    """Vertical displacement d(x, Y) = c1(Y)*(x-xc) + c2(Y)*(x-xc)^2 (+ c3 cubic) in the rotated
    frame, where Y is the straightened ("flat") row. Coefficients vary smoothly with Y."""
    xc: float
    coef_y: np.ndarray       # sample rows Y_i (sorted)
    coefs: np.ndarray        # (len(Y), 3) coefficients for (x-xc), (x-xc)^2, (x-xc)^3
    max_abs: float = 0.0
    x_lo: float = -1e9       # outside [x_lo, x_hi] the displacement is held constant (no wild
    x_hi: float = 1e9        # cubic extrapolation into margins or the facing page)

    def disp(self, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if len(self.coef_y) == 0:
            return np.zeros(np.broadcast(x, Y).shape, dtype=np.float32)
        c = [np.interp(Y, self.coef_y, self.coefs[:, i]) for i in range(self.coefs.shape[1])]
        dx = np.clip(x, self.x_lo, self.x_hi) - self.xc
        return (c[0] * dx + c[1] * dx * dx + c[2] * dx * dx * dx).astype(np.float32)


def fit_dewarp(tracks: List[LineTrack], block: TextBlock, pitch: float) -> DewarpField:
    xc = (block.x0 + block.x1) / 2.0
    half = max(1.0, (block.x1 - block.x0) / 2.0)
    rows, coefs, weights = [], [], []
    for t in tracks:
        if len(t.xs) < 3 or (t.xs.max() - t.xs.min()) < 0.35 * (block.x1 - block.x0):
            continue
        u = (t.xs - xc) / half
        deg = 3 if len(t.xs) >= 6 else (2 if len(t.xs) >= 4 else 1)
        A = np.stack([np.ones_like(u)] + [u ** p for p in range(1, deg + 1)], axis=1)
        sol, *_ = np.linalg.lstsq(A, t.ys, rcond=None)
        resid = t.ys - A @ sol
        if np.abs(resid).max() > pitch * 0.3:
            # one outlier slice: drop the worst point and refit
            keep = np.abs(resid) < np.abs(resid).max()
            if keep.sum() >= 3:
                A2 = A[keep]
                sol, *_ = np.linalg.lstsq(A2, t.ys[keep], rcond=None)
        c = np.zeros(3)
        for p in range(1, deg + 1):
            c[p - 1] = sol[p] / half ** p
        rows.append(sol[0])
        coefs.append(c)
        weights.append(t.xs.max() - t.xs.min())
    if len(rows) < 2:
        return DewarpField(xc, np.array([]), np.zeros((0, 3)))
    rows = np.array(rows)
    coefs = np.array(coefs)
    order = np.argsort(rows)
    rows, coefs = rows[order], coefs[order]
    # robust smoothing of each coefficient along Y (running median over 5 tracks, then mean over 3)
    sm = coefs.copy()
    for i in range(len(rows)):
        lo, hi = max(0, i - 2), min(len(rows), i + 3)
        sm[i] = np.median(coefs[lo:hi], axis=0)
    sm2 = sm.copy()
    for i in range(len(rows)):
        lo, hi = max(0, i - 1), min(len(rows), i + 2)
        sm2[i] = sm[lo:hi].mean(axis=0)
    field_ = DewarpField(xc, rows, sm2, x_lo=float(block.x0), x_hi=float(block.x1))
    xs = np.array([block.x0, block.x1], dtype=np.float64)
    field_.max_abs = float(max(np.abs(field_.disp(xs, np.full(2, y))).max() for y in rows))
    return field_


# ---------------------------------------------------------------------------
# the page geometry object
# ---------------------------------------------------------------------------

@dataclass
class Geometry:
    src_shape: Tuple[int, int]          # (h, w) of the source
    scale: float                        # analysis = source * scale
    angle: float                        # skew undone (degrees)
    M: np.ndarray                       # analysis -> rotated
    Minv: np.ndarray                    # rotated -> analysis
    rot_size: Tuple[int, int]           # rotated/flat canvas (w, h)
    dewarp: DewarpField
    block: TextBlock                    # in the flat frame
    pitch: float                        # line pitch in flat-frame (analysis) px
    sideways: bool = False

    def flat_to_analysis(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        yr = v + self.dewarp.disp(u, v)
        xr = u
        Mi = self.Minv
        xa = Mi[0, 0] * xr + Mi[0, 1] * yr + Mi[0, 2]
        ya = Mi[1, 0] * xr + Mi[1, 1] * yr + Mi[1, 2]
        return xa.astype(np.float32), ya.astype(np.float32)

    def flat_to_source(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xa, ya = self.flat_to_analysis(u, v)
        return xa / self.scale, ya / self.scale

    def source_to_flat(self, X: np.ndarray, Y: np.ndarray, iters: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse of flat_to_source (fixed-point on the vertical displacement)."""
        X = np.asarray(X, dtype=np.float64) * self.scale
        Y = np.asarray(Y, dtype=np.float64) * self.scale
        M = self.M
        xr = M[0, 0] * X + M[0, 1] * Y + M[0, 2]
        yr = M[1, 0] * X + M[1, 1] * Y + M[1, 2]
        v = yr.copy()
        for _ in range(iters):
            v = yr - self.dewarp.disp(xr, v)
        return xr, v

    def flat_maps(self, x0: float, y0: float, w: int, h: int, step: float) -> Tuple[np.ndarray, np.ndarray]:
        """cv2.remap maps (source pixels) for a flat-frame window starting at (x0, y0), sampled
        every ``step`` flat px, ``w`` x ``h`` output pixels."""
        us = x0 + np.arange(w, dtype=np.float32) * step
        vs = y0 + np.arange(h, dtype=np.float32) * step
        U, V = np.meshgrid(us, vs)
        X, Y = self.flat_to_source(U, V)
        return X.astype(np.float32), Y.astype(np.float32)


def analysis_image(src: np.ndarray, long_side: int = ANALYSIS_LONG_SIDE) -> Tuple[np.ndarray, float]:
    h, w = src.shape[:2]
    s = min(1.0, long_side / float(max(h, w)))
    if s >= 0.999:
        return src.copy(), 1.0
    out = cv2.resize(src, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return out, s


def _remap_mask(mask: np.ndarray, maps: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(mask.astype(np.uint8), maps[0], maps[1], interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0


def build_geometry(src: np.ndarray, n_slices: int = 12) -> Tuple[Geometry, np.ndarray]:
    """Analyse the page and return its Geometry plus the flat-frame analysis RGB image.

    Two passes: the block found in the rotated frame can still include a facing page whose lines
    curl into the gutter; after one dewarp the gutter is clean, the block is found again in the
    flat frame (x is shared by both frames) and the lines are re-tracked inside it.
    """
    ana, scale = analysis_image(src)
    h, w = ana.shape[:2]
    L, a, b = to_lab(ana)
    stroke = max(9, int(round(max(h, w) * 0.009)))
    bg = background_lightness(L, stroke)
    ink = ink_mask(L, bg)
    bright_bg = bg > 0.55 * np.percentile(bg, 95)
    angle, _sharp = estimate_skew(ink & bright_bg)
    M, Minv, (nw, nh) = rotation_matrices((h, w), angle)
    rot = cv2.warpAffine(ana, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpAffine(np.ones((h, w), np.uint8), M, (nw, nh), flags=cv2.INTER_NEAREST) > 0
    Lr, _ar, _br = to_lab(rot)
    bgr_ = background_lightness(Lr, stroke)
    inkr = ink_mask(Lr, bgr_) & valid
    brightr = (bgr_ > 0.55 * np.percentile(bgr_[valid], 95)) & valid
    use = inkr & brightr
    block = find_text_block(inkr, brightr)
    if block is None:
        block = TextBlock(int(nw * 0.1), int(nw * 0.9), int(nh * 0.1), int(nh * 0.9))
    prof = use[block.y0:block.y1, block.x0:block.x1].sum(axis=1).astype(np.float64)
    pitch = estimate_pitch(smooth1d(prof, 1.0), lo=max(8, int(nh * 0.008)), hi=int(nh * 0.12)) or nh * 0.028
    geo = Geometry((src.shape[0], src.shape[1]), scale, angle, M, Minv, (nw, nh),
                   DewarpField((block.x0 + block.x1) / 2.0, np.array([]), np.zeros((0, 3))), block, pitch)
    U, V = np.meshgrid(np.arange(nw, dtype=np.float32), np.arange(nh, dtype=np.float32))
    for _ in range(2):
        tracks = track_lines(use, block, pitch, n_slices=n_slices)
        geo.dewarp = fit_dewarp(tracks, block, pitch)
        # flat-frame masks: the dewarp only moves pixels vertically inside the rotated frame
        yr = V + geo.dewarp.disp(U, V)
        maps = (U, yr.astype(np.float32))
        ink_f = _remap_mask(inkr, maps)
        bright_f = _remap_mask(brightr, maps)
        nb = find_text_block(ink_f, bright_f)
        if nb is not None:
            # a later pass may refine the block but never widen it past an earlier gutter split
            tol = pitch * 0.5
            nb.x0 = int(max(nb.x0, block.x0 - tol))
            nb.x1 = int(min(nb.x1, block.x1 + tol))
            nb.other_runs = sorted(set(nb.other_runs) | set(block.other_runs))
            block = nb
        prof = (ink_f & bright_f)[block.y0:block.y1, block.x0:block.x1].sum(axis=1).astype(np.float64)
        p2 = estimate_pitch(smooth1d(prof, 1.0), lo=max(8, int(nh * 0.008)), hi=int(nh * 0.12))
        if p2 > 0:
            pitch = p2
        geo.block, geo.pitch = block, pitch
    xa, ya = geo.flat_to_analysis(U, V)
    flat = cv2.remap(ana, xa, ya, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return geo, flat
