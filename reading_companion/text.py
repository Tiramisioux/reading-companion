"""The text of a stripe (OCR) and the words the reader marked in it, for the caption.

One OCR call per stripe image (already straightened, so its lines are straight); each word's box
is mapped back onto the page and assigned to a printed line. A word counts as marked when
highlighter covers most of its letters, when it sits above an underline, or when its line is
beside a margin line or a margin note. Handwritten notes are read separately (best effort: Apple
Vision reads handwriting, Tesseract mostly does not, so low-confidence readings are dropped).
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .bands import Band
from .geometry import Geometry
from .marks import Mark, PageAnalysis
from .ocr import OcrEngine, Word

LETTER_PX = 40.0      # OCR reads best with letters about this tall (tested 32/40/48 px)
MIN_WORD_CONF = 0.3
NOTE_MIN_CONF = 0.55


def scaled_gray(rgb: np.ndarray, letter_src_px: float) -> Tuple[np.ndarray, float]:
    """Dark ink on even white, scaled for OCR. The brightest colour channel turns any
    highlighter (yellow, teal, pink, orange) near-white while ink stays dark; dividing by the
    local background then evens out shadows. Returns (image, scale factor)."""
    v = rgb.max(axis=2).astype(np.float32) if rgb.ndim == 3 else rgb.astype(np.float32)
    k = int(max(15, letter_src_px * 2.5)) | 1
    bg = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3.0)
    g = np.clip(v / np.maximum(bg, 1.0) * 255.0, 0, 255).astype(np.uint8)
    f = float(np.clip(LETTER_PX / max(1.0, letter_src_px), 1.0, 3.0))
    if f > 1.01:
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return g, f


ARTIFACTS = set("~|»«=_^•·°©®™\\/")


def keep_word(t: str, conf: float) -> bool:
    """OCR scores words under highlighter low even when it reads them nearly right: keep
    low-confidence words that are mostly letters; drop stray marks from highlighter ends."""
    t = t.strip()
    if not t or all(ch in ARTIFACTS or ch in "-" for ch in t):
        return False
    letters = sum(ch.isalpha() for ch in t)
    if letters >= max(1, 0.6 * len(t)):
        return conf >= 0.1
    return conf >= MIN_WORD_CONF


def join_text(pieces: Sequence[str]) -> str:
    """Join line texts; a word broken at a line end ('tran-' + 'scend') is mended."""
    out = ""
    for t in pieces:
        t = t.strip()
        if not t:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t
        elif out:
            out += " " + t
        else:
            out = t
    return out


GLUE = re.compile(r"(?<=[.!?’'\"”])(?=[A-Z0-9%*\\])")


def split_glued(words: Sequence[Word]) -> List[Word]:
    """'truth’.He' -> 'truth’.' + 'He': a footnote number or the next sentence glued to a word
    by the OCR; the box is shared out by character count."""
    out: List[Word] = []
    for (t, c, x0, y0, x1, y1) in words:
        parts = [q for q in GLUE.split(t) if q]
        if len(parts) <= 1:
            out.append((t, c, x0, y0, x1, y1))
            continue
        total = sum(len(q) for q in parts)
        pos = 0
        for q in parts:
            a = x0 + (x1 - x0) * pos / total
            b = x0 + (x1 - x0) * (pos + len(q)) / total
            out.append((q, c, int(a), y0, int(b), y1))
            pos += len(q)
    return out


def _coverage(an: PageAnalysis, x0: float, y0: float, x1: float, y1: float) -> float:
    """Share of a word's columns that carry highlighter in the middle of its letters."""
    h, w = an.tint.shape
    yc0 = int(np.clip(y0 + 0.3 * (y1 - y0), 0, h - 1))
    yc1 = int(np.clip(y1 - 0.3 * (y1 - y0), yc0 + 1, h))
    xa, xb = int(np.clip(x0, 0, w - 1)), int(np.clip(x1, 1, w))
    if xb <= xa:
        return 0.0
    return float(an.tint[yc0:yc1, xa:xb].any(axis=0).mean())


def band_text(an: PageAnalysis, geo: Geometry, b: Band, words: List[Word], f: float) -> Dict:
    p = geo.pitch
    lay = an.layout
    lines = {ln.index: ln for ln in an.lines if 1 <= b.ctx_first <= ln.index <= b.ctx_last}
    per_line: Dict[int, List[Tuple[float, float, float, float, str, float]]] = {i: [] for i in lines}
    confs = []
    for (t, c, x0, y0, x1, y1) in split_glued(words):
        if not keep_word(t, c):
            continue
        t = t.strip().strip("".join(ARTIFACTS)) or t.strip()
        fx0, fx1 = b.x0 + x0 / f * geo.scale, b.x0 + x1 / f * geo.scale
        fy0, fy1 = b.y0 + y0 / f * geo.scale, b.y0 + y1 / f * geo.scale
        yc, xc = (fy0 + fy1) / 2.0, (fx0 + fx1) / 2.0
        best = min(lines.values(), key=lambda ln: abs(ln.center - yc), default=None)
        if best is None or abs(best.center - yc) > 0.55 * p:
            continue
        if lay is not None:
            r = int(np.clip(best.center, 0, len(lay.text_l) - 1))
            if xc < lay.text_l[r] - 0.4 * p or xc > lay.text_r[r] + 0.4 * p:
                continue  # in the margin: handwriting or the facing page, not the printed line
        per_line[best.index].append((fx0, fy0, fx1, fy1, t, c))
        confs.append(c)
    for i in per_line:
        per_line[i].sort(key=lambda w: w[0])

    marked = {i: [False] * len(per_line[i]) for i in per_line}
    for m in b.marks:
        for i in m.lines:
            if i not in per_line:
                continue
            ws = per_line[i]
            if m.kind == "highlight":
                for k, (x0, y0, x1, y1, t, c) in enumerate(ws):
                    if min(x1, m.x1) - max(x0, m.x0) > 0.3 * (x1 - x0) and _coverage(an, x0, y0, x1, y1) >= 0.5:
                        marked[i][k] = True
            elif m.kind == "underline":
                for k, (x0, y0, x1, y1, t, c) in enumerate(ws):
                    if m.x0 - 0.15 * p <= (x0 + x1) / 2.0 <= m.x1 + 0.15 * p:
                        marked[i][k] = True
            elif m.kind == "margin_line" or (m.kind == "margin_note" and m.where == "side"):
                marked[i] = [True] * len(ws)

    # the quote: marked words in reading order; a run of unmarked words between them is "…"
    seq: List[Tuple[str, bool, int]] = []
    for i in sorted(per_line):
        for k, w in enumerate(per_line[i]):
            seq.append((w[4], marked[i][k], i))
    first = next((n for n, s in enumerate(seq) if s[1]), None)
    last = max((n for n, s in enumerate(seq) if s[1]), default=None)
    quote_parts: List[str] = []
    if first is not None:
        line_of = seq[first][2]
        cur: List[str] = []
        gap = False
        for t, mk, i in seq[first:last + 1]:
            if not mk:
                gap = True
                continue
            if gap and cur:
                cur.append("…")
            gap = False
            if i != line_of:
                quote_parts.append(" ".join(cur))
                cur, line_of = [], i
            cur.append(t)
        quote_parts.append(" ".join(cur))
    quote = join_text(quote_parts).replace(" … ", " … ")
    text = join_text([" ".join(w[4] for w in per_line[i]) for i in sorted(per_line)])
    return {
        "quote": quote,
        "text": text,
        "lines": [{"line": i, "text": " ".join(w[4] for w in per_line[i]),
                   "marked": [w[4] for k, w in enumerate(per_line[i]) if marked[i][k]]} for i in sorted(per_line)],
        "ocr_confidence": round(float(np.mean(confs)), 2) if confs else None,
    }


def note_crop(src: np.ndarray, geo: Geometry, m: Mark, letter_src_px: float) -> Tuple[np.ndarray, float]:
    pad = 0.25 * geo.pitch
    x0, y0 = m.x0 - pad, m.y0 - pad
    w = int((m.x1 - m.x0 + 2 * pad) / geo.scale)
    h = int((m.y1 - m.y0 + 2 * pad) / geo.scale)
    mx, my = geo.flat_maps(x0, y0, max(1, w), max(1, h), geo.scale)
    crop = cv2.remap(src, mx, my, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return scaled_gray(crop, letter_src_px * 1.2)


def read_texts(src: np.ndarray, geo: Geometry, an: PageAnalysis, bands: Sequence[Band],
               stripes: Sequence[np.ndarray], engine: OcrEngine) -> List[Dict]:
    """Per band: {"quote", "text", "lines", "notes", "ocr_confidence"} (empty without OCR)."""
    if engine.name == "none" or not bands:
        return [{"quote": "", "text": "", "lines": [], "notes": [], "ocr_confidence": None} for _ in bands]
    letter_src = max(1.0, an.letter_h / geo.scale)
    imgs, facs = [], []
    for st in stripes:
        g, f = scaled_gray(st, letter_src)
        imgs.append(g)
        facs.append(f)
    words = engine.read(imgs, mode="block")
    out = [band_text(an, geo, b, w, f) for b, w, f in zip(bands, words, facs)]
    # handwriting: each note on its own (best effort)
    notes = [(bi, m) for bi, b in enumerate(bands) for m in b.marks if m.kind == "margin_note"]
    crops = [note_crop(src, geo, m, letter_src) for _, m in notes]
    nw = engine.read([c[0] for c in crops], mode="block") if crops else []
    for o in out:
        o["notes"] = []
    for (bi, m), ws in zip(notes, nw):
        good = [w for w in ws if w[0].strip()]
        if not good:
            continue
        conf = float(np.mean([w[1] for w in good]))
        txt = " ".join(w[0] for w in good)
        letters = sum(ch.isalpha() for ch in txt)
        if conf >= NOTE_MIN_CONF and letters >= max(2, 0.6 * len(txt.replace(" ", ""))):
            out[bi]["notes"].append(txt)
    return out


def caption_of(t: Dict) -> str:
    cap = t.get("quote") or ""
    for n in t.get("notes") or []:
        cap = (cap + "\n\n" if cap else "") + f"Note: {n}"
    return cap
