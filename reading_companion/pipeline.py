"""Run the extractor over one photo or a folder of photos."""
from __future__ import annotations

import datetime as _dt
import html
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import __version__
from .bands import Band, assign_lines, cut_stripe, make_bands
from .geometry import Geometry, build_geometry
from .imageio import load_rgb, save_stripe
from .marks import PageAnalysis, analyse
from .metadata import PageMeta, PrintRow, consolidate, read_page
from .naming import DEFAULT_TEMPLATE, chapter_label, list_images, stripe_name, unique
from .ocr import OcrEngine, get_engine

OUT_DIRNAME = "reading-companion"
KIND_COLORS = {"highlight": (0, 170, 255), "underline": (255, 0, 200), "margin_line": (0, 0, 255), "margin_note": (255, 120, 0)}
TEXT_KINDS = ("highlight", "underline", "margin_line")   # the marks whose printed lines are read; a margin note marks no text


@dataclass
class Options:
    context: int = 1
    ocr: str = "auto"
    ocr_bin: Optional[str] = None
    languages: Optional[str] = None
    book_title: Optional[str] = None
    name_template: str = DEFAULT_TEMPLATE
    quality: int = 90
    review: bool = True
    text: bool = True          # read the printed text of the marked lines (needs an OCR engine)


@dataclass
class PageResult:
    index: int
    path: Path
    meta: PageMeta = field(default_factory=PageMeta)
    bands: List[Band] = field(default_factory=list)
    stripes: List[np.ndarray] = field(default_factory=list)
    polygons: List[List[List[float]]] = field(default_factory=list)
    info: Dict = field(default_factory=dict)
    review: Optional[np.ndarray] = None
    line_text: Optional[Dict[int, str]] = None   # printed text of the marked lines (1-based line -> text); None = not read
    error: Optional[str] = None
    geo: Optional[Geometry] = None          # runtime only (evaluation, debugging)
    analysis: Optional[PageAnalysis] = None


# ---------------------------------------------------------------------------
# one page
# ---------------------------------------------------------------------------

def print_rows(an: PageAnalysis, geo: Geometry) -> List[PrintRow]:
    """Printed groups above and below the body (running head, page number, chapter heading,
    footer), one per OCR crop. Slivers (page edges, rules) are left out."""
    p = geo.pitch
    out: List[PrintRow] = []
    last_bottom = an.lines[-1].bottom if an.lines else None
    for g in an.margin_groups:
        if g.handwriting:
            continue
        w_, h_ = g.x1 - g.x0, g.y1 - g.y0
        if (w_ < 0.25 * p and h_ < 0.25 * p) or h_ < 0.12 * p or h_ > 3.0 * w_ + 0.5 * p:
            continue
        where = "top" if an.lines and g.y1 <= an.lines[0].top + 0.2 * p else "bottom"
        if where == "bottom" and last_bottom is not None and g.y0 - last_bottom > 4.0 * p:
            continue
        hs = sorted(c[3] - c[1] for c in g.comps if len(c) < 7 or c[6]) or sorted(c[3] - c[1] for c in g.comps)
        hgt = float(hs[len(hs) // 2]) if hs else float(h_)
        out.append(PrintRow(where, g.x0, g.y0, g.x1, g.y1, hgt))
    return out


def text_lines(b: Band, n: int) -> List[int]:
    """The printed lines of a stripe whose text was marked: those carrying a highlight, an underline or a margin line
    (1-based, top to bottom). The context lines and the lines beside a margin note only are not among them."""
    out = set()
    for m in b.marks:
        if m.kind in TEXT_KINDS:
            out.update(i for i in m.lines if 1 <= i <= n)
    return sorted(out)


def line_row(an: PageAnalysis, geo: Geometry, ln) -> PrintRow:
    """One body line as a row to read: the text column at the line's height, between the gap middles above and below."""
    lay, p = an.layout, geo.pitch
    if lay is not None:
        x0, x1 = lay.xl[0] + lay.xl[1] * ln.center, lay.xr[0] + lay.xr[1] * ln.center
    else:
        x0, x1 = geo.block.x0, geo.block.x1
    return PrintRow("body", float(x0) - 0.3 * p, float(ln.top), float(x1) + 0.3 * p, float(ln.bottom), float(an.letter_h or 0.45 * p))


WORD_COVER = 0.6   # a word counts as marked when a highlight (or the pen stroke under it) covers this share of its width,
                   # so a highlighter that spills onto a letter or two of the next word does not take that word along


def middle_words(words, height: int) -> list:
    """The words of one line image that belong to the line in its middle (a crop between gap middles may also catch the
    ascenders or descenders of its neighbours), left to right."""
    if not words:
        return []
    mid = height / 2.0

    def centre(w) -> float:
        return (w[3] + w[5]) / 2.0

    keep = [w for w in words if abs(centre(w) - mid) <= 0.3 * height]
    if not keep:
        best = min(words, key=lambda w: abs(centre(w) - mid))
        keep = [w for w in words if (w[3], w[5]) == (best[3], best[5])]
    return sorted(keep, key=lambda w: w[2])


def word_cover(an: PageAnalysis, pitch: float, ln, m, a: float, b: float) -> float:
    """The share of a word on line ``ln`` (flat x from ``a`` to ``b``) that mark ``m`` covers: for a highlight the columns
    with tint in the line's letter band inside the mark, for an underline the overlap with the pen stroke, for a margin
    line all of it (it marks whole lines)."""
    if b <= a:
        return 0.0
    if m.kind == "highlight":
        h, w = an.tint.shape[:2]
        y0, y1 = int(max(m.y0, ln.center - 0.22 * pitch, 0)), int(min(m.y1, ln.center + 0.22 * pitch, h - 1)) + 1
        x0, x1 = int(max(a, m.x0, 0)), int(min(b, m.x1, w - 1)) + 1
        if y1 <= y0 or x1 <= x0:
            return 0.0
        return float((an.tint[y0:y1, x0:x1].sum(axis=0) > 0).sum()) / (b - a)
    if m.kind == "underline":
        return max(0.0, min(b, m.x1) - max(a, m.x0)) / (b - a)
    return 1.0


def marked_words(an: PageAnalysis, pitch: float, ln, words, ox: float, kx: float) -> str:
    """The text of the marked words of a body line: ``words`` are the line image's OCR words (from ``middle_words``),
    ``flat x = ox + kx * image x``. A word is kept when some highlight, underline or margin line on that line covers
    ``WORD_COVER`` of it."""
    marks = [m for m in an.marks if m.kind in TEXT_KINDS and ln.index in m.lines]
    keep = []
    for wd in words:
        a, b = ox + kx * wd[2], ox + kx * wd[4]
        if any(word_cover(an, pitch, ln, m, a, b) >= WORD_COVER for m in marks):
            keep.append(wd[0])
    return " ".join(keep).strip()


def join_lines(lines: List[Dict]) -> str:
    """One passage from the marked lines of a stripe: consecutive lines joined with a space (a word broken with a hyphen
    at the end of a line is joined again), a gap in the line numbers shown as ' … '."""
    out, prev = "", None
    for rec in lines:
        t = rec["text"].strip()
        if not t:
            continue
        if not out:
            out = t
        elif prev is not None and rec["line"] == prev + 1:
            if out.endswith("-") and not out.endswith("--") and t[:1].islower():
                out = out[:-1] + t
            else:
                out += " " + t
        else:
            out += " … " + t
        prev = rec["line"]
    return out


def row_map(geo: Geometry, r: PrintRow, tight: bool = False) -> Tuple[float, float, float, float]:
    """``(x0, y0, pad_x, scale)`` of a row crop: its flat-frame origin, and ``scale`` = image pixels per flat pixel."""
    pad = 2.0 if tight else 0.35 * (r.y1 - r.y0) + 4
    padx = 0.8 * (r.y1 - r.y0) + 8
    letter = max(1.0, r.height / geo.scale)
    f = float(np.clip(48.0 / letter, 0.5, 4.0))
    return r.x0 - padx, r.y0 - pad, padx, f / geo.scale


def row_image(src: np.ndarray, geo: Geometry, r: PrintRow, tight: bool = False) -> np.ndarray:
    """Grayscale crop of a printed row at full resolution, scaled to ~48 px letter height. ``tight``: no extra height
    (a body line, whose band already reaches the gap middles). ``row_map`` gives the crop's place in the flat frame."""
    x0, y0, padx, _ = row_map(geo, r, tight)
    pad = r.y0 - y0
    w = int((r.x1 - r.x0 + 2 * padx) / geo.scale)
    h = int((r.y1 - r.y0 + 2 * pad) / geo.scale)
    mx, my = geo.flat_maps(x0, y0, max(1, w), max(1, h), geo.scale)
    crop = cv2.remap(src, mx, my, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    letter = max(1.0, r.height / geo.scale)
    f = float(np.clip(48.0 / letter, 0.5, 4.0))
    g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    if hi > lo + 10:
        g = np.clip((g.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return g


def band_polygon(geo: Geometry, b: Band, n: int = 16) -> List[List[float]]:
    """The stripe's outline in source pixels (top edge left->right, bottom edge right->left)."""
    xs = np.linspace(b.x0, b.x1, n)
    tx, ty = geo.flat_to_source(xs, np.full(n, b.y0))
    bx, by = geo.flat_to_source(xs[::-1], np.full(n, b.y1))
    return [[round(float(x), 1), round(float(y), 1)] for x, y in zip(np.concatenate([tx, bx]), np.concatenate([ty, by]))]


def review_image(src: np.ndarray, geo: Geometry, an: PageAnalysis, bands: List[Band], polys) -> np.ndarray:
    s = min(1.0, 1400.0 / max(src.shape[:2]))
    img = cv2.cvtColor(cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    for m in an.marks:
        xs = np.array([m.x0, m.x1, m.x1, m.x0], dtype=np.float64)
        ys = np.array([m.y0, m.y0, m.y1, m.y1], dtype=np.float64)
        X, Y = geo.flat_to_source(xs, ys)
        pts = (np.stack([X, Y], 1) * s).astype(np.int32)
        cv2.polylines(img, [pts], True, KIND_COLORS.get(m.kind, (0, 0, 0)), 2)
    for k, (b, poly) in enumerate(zip(bands, polys), 1):
        pts = (np.array(poly) * s).astype(np.int32)
        cv2.polylines(img, [pts], True, (200, 0, 200), 3)
        x, y = pts[:, 0].max() - 70, pts[:, 1].min() + 26
        cv2.putText(img, f"note{k}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, f"note{k}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2, cv2.LINE_AA)
    return img


def process_page(path: Path, index: int, engine: OcrEngine, opts: Options) -> PageResult:
    res = PageResult(index, Path(path))
    try:
        src, src_info = load_rgb(path)
        geo, flat = build_geometry(src)
        an = analyse(flat, geo)
        assign_lines(an, geo.pitch)
        bands = make_bands(an, geo, context=opts.context)
        res.bands, res.geo, res.analysis = bands, geo, an
        res.stripes = [cut_stripe(src, geo, b) for b in bands]
        res.polygons = [band_polygon(geo, b) for b in bands]
        rows = print_rows(an, geo)
        by_index = {ln.index: ln for ln in an.lines}
        wanted = sorted({i for b in bands for i in text_lines(b, len(an.lines))}) if opts.text else []
        if engine.name != "none" and (rows or wanted):
            # one call per page: the rows above and below the text, then the marked body lines
            imgs = [row_image(src, geo, r) for r in rows]
            body = [line_row(an, geo, by_index[i]) for i in wanted]
            lines = [row_image(src, geo, r, tight=True) for r in body]
            words = engine.read(imgs + lines)
            for r, wds, im in zip(rows, words[:len(rows)], imgs):
                r.words, r.img_w = wds, int(im.shape[1])
            if opts.text:
                res.line_text = {}
                for i, r, wds, im in zip(wanted, body, words[len(rows):], lines):
                    ox, _, _, k = row_map(geo, r, tight=True)
                    res.line_text[i] = marked_words(an, geo.pitch, by_index[i], middle_words(wds, int(im.shape[0])), ox, 1.0 / k)
        lay = an.layout
        mid = (geo.block.y0 + geo.block.y1) // 2
        col_x0 = float(lay.xl[0] + lay.xl[1] * mid) if lay else float(geo.block.x0)
        col_x1 = float(lay.xr[0] + lay.xr[1] * mid) if lay else float(geo.block.x1)
        res.meta = read_page(rows, geo.pitch, col_x0, col_x1,
                             an.lines[0].top if an.lines else None, an.lines[-1].bottom if an.lines else None,
                             letter_h=an.letter_h)
        res.info = {
            "size": [int(src.shape[1]), int(src.shape[0])],
            "skew_deg": round(geo.angle, 2),
            "line_pitch_px": round(geo.pitch / geo.scale, 1),
            "body_lines": len(an.lines),
            "dewarp_px": round(geo.dewarp.max_abs / geo.scale, 1),
            "marks": [dict(m.as_dict(), where=m.where or None) for m in an.marks],
            **src_info,
        }
        if opts.review:
            res.review = review_image(src, geo, an, bands, res.polygons)
    except Exception as exc:  # one bad photo must not stop the folder
        res.error = f"{type(exc).__name__}: {exc}"
        res.info["traceback"] = traceback.format_exc(limit=4)
    return res


# ---------------------------------------------------------------------------
# a folder
# ---------------------------------------------------------------------------

def _describe(book: Optional[str], meta: PageMeta, note: int, notes: int, b: Band, source: str) -> str:
    parts = []
    if book:
        parts.append(book)
    if meta.chapter_number or meta.chapter_title:
        ch = "chapter " + (meta.chapter_number or "")
        if meta.chapter_title:
            ch = (ch + f" ({meta.chapter_title})") if meta.chapter_number else f"chapter: {meta.chapter_title}"
        parts.append(ch.strip())
    if meta.page_number:
        parts.append(f"p. {meta.page_number}")
    parts.append(f"note {note}/{notes}")
    what = ", ".join(f"{c} {k.replace('_', ' ')}" for k, c in _kind_colors(b))
    if what:
        parts.append(what)
    parts.append(f"from {source}")
    return " - ".join(parts)


def _kind_colors(b: Band) -> List[Tuple[str, str]]:
    seen = []
    for m in sorted(b.marks, key=lambda m: m.y0):
        kc = (m.kind, m.color)
        if kc not in seen:
            seen.append(kc)
    return seen


def _clean_previous(out_dir: Path) -> None:
    """Remove the files a previous run listed in its manifest (only those)."""
    man = out_dir / "manifest.json"
    if not man.exists():
        return
    try:
        old = json.loads(man.read_text(encoding="utf-8"))
    except Exception:
        return
    for rec in old.get("stripes", []):
        f = out_dir / rec.get("file", "")
        if f.is_file() and f.parent == out_dir:
            f.unlink()
    for rec in old.get("pages", []):
        r = rec.get("review")
        if r:
            f = out_dir / r
            if f.is_file() and out_dir in f.parents:
                f.unlink()


def process_folder(folder: Path, out_dir: Optional[Path] = None, opts: Optional[Options] = None,
                   progress: Optional[Callable[[float, str], None]] = None, files: Optional[Sequence[Path]] = None,
                   engine: Optional[OcrEngine] = None) -> Dict:
    """``engine`` overrides ``opts.ocr`` (tests)."""
    opts = opts or Options()
    folder = Path(folder)
    out_dir = Path(out_dir) if out_dir else folder / OUT_DIRNAME
    images = [Path(f) for f in files] if files else list_images(folder, exclude=[out_dir])
    engine, ocr_note = (engine, "") if engine is not None else get_engine(opts.ocr, opts.languages, opts.ocr_bin)
    t0 = time.time()
    results: List[PageResult] = []
    for i, path in enumerate(images, 1):
        if progress:
            progress((i - 1) / max(1, len(images)), path.name)
        results.append(process_page(path, i, engine, opts))
    book = consolidate([r.meta for r in results], folder.name, opts.book_title)

    out_dir.mkdir(parents=True, exist_ok=True)
    _clean_previous(out_dir)
    review_dir = out_dir / "review"
    if opts.review:
        review_dir.mkdir(exist_ok=True)
    taken: set = set()
    stripes_out: List[Dict] = []
    pages_out: List[Dict] = []
    seq = 0
    software = f"reading-companion {__version__}"
    for r in results:
        m = r.meta
        names = []
        chapter = chapter_label(m.chapter_number, m.chapter_title)
        for k, (b, img, poly) in enumerate(zip(r.bands, r.stripes, r.polygons), 1):
            seq += 1
            name = unique(stripe_name(opts.name_template, r.path.stem, chapter, m.page_number or "", k, len(r.bands), seq,
                                      book.get("title") or ""), taken)
            rec = {
                "file": name, "seq": seq, "source": r.path.name, "source_index": r.index,
                "note": k, "notes_on_page": len(r.bands),
                "book": book.get("title"),
                "page_number": m.page_number, "page_number_source": m.page_number_source,
                "chapter": ({"number": m.chapter_number, "title": m.chapter_title, "source": m.chapter_source}
                            if (m.chapter_number or m.chapter_title) else None),
                "running_head": m.running_head,
                "lines_marked": [b.first_line, b.last_line], "lines_shown": [b.ctx_first, b.ctx_last],
                "kinds": b.kinds, "colors": b.colors,
                "marks": [{"kind": mk.kind, "color": mk.color, "lines": mk.lines, "where": mk.where or None,
                           "rgb": list(mk.rgb)} for mk in b.marks],
                "size": [int(img.shape[1]), int(img.shape[0])],
            }
            if r.line_text is not None:      # the marked words, line by line (None: not read, no OCR or --no-text)
                text = [{"line": i, "text": r.line_text[i]} for i in text_lines(b, r.info.get("body_lines", 0))
                        if r.line_text.get(i)]
                rec["text"], rec["passage"] = text, join_lines(text)
            else:
                rec["text"], rec["passage"] = None, None
            rec["source_polygon"] = poly
            desc = _describe(book.get("title"), m, k, len(r.bands), b, r.path.name)
            emb = {k2: v for k2, v in rec.items() if k2 not in ("source_polygon", "marks", "text")}
            save_stripe(out_dir / name, img, desc, emb, software, r.info.get("captured"), opts.quality)
            stripes_out.append(rec)
            names.append(name)
        review_name = None
        if opts.review and r.review is not None:
            review_name = f"review/{r.index:03d}_{r.path.stem}.jpg"
            cv2.imwrite(str(out_dir / review_name), r.review, [cv2.IMWRITE_JPEG_QUALITY, 82])
        pages_out.append({
            "index": r.index, "file": r.path.name, "stripes": names, "review": review_name,
            **m.as_dict(), **{k2: v for k2, v in r.info.items() if k2 != "traceback"},
            "error": r.error,
        })
    manifest = {
        "tool": "reading-companion", "version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "input": str(folder.resolve()), "output": str(out_dir.resolve()),
        "options": {"context_lines": opts.context, "ocr": opts.ocr, "name_template": opts.name_template,
                    "book_title": opts.book_title, "text": opts.text},
        "ocr": {"engine": engine.name, "version": engine.version, "note": ocr_note or None},
        "book": book,
        "pages": pages_out,
        "stripes": stripes_out,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    if opts.review:
        write_review_html(out_dir, manifest)
    if progress:
        progress(1.0, "done")
    return manifest


def write_review_html(out_dir: Path, manifest: Dict) -> None:
    """index.html next to the stripes: each photo with its stripes outlined, then the stripes."""
    e = html.escape
    passage = {s["file"]: s.get("passage") for s in manifest["stripes"]}
    rows = []
    for pg in manifest["pages"]:
        ch = pg.get("chapter") or {}
        meta = " · ".join(x for x in [
            f"p. {pg['page_number']}" if pg.get("page_number") else "page number: not printed",
            (f"chapter {ch.get('number') or ''} {ch.get('title') or ''} ({ch.get('source')})".strip() if ch else "chapter: unknown"),
            f"running head: {pg['running_head']}" if pg.get("running_head") else "",
        ] if x)
        err = f"<p class=err>{e(pg['error'])}</p>" if pg.get("error") else ""
        strips = "".join(
            f"<figure><img src='{e(s)}' loading=lazy><figcaption>{e(s)}</figcaption>"
            + (f"<blockquote>{e(passage[s])}</blockquote>" if passage.get(s) else "") + "</figure>" for s in pg["stripes"]
        ) or "<p class=none>no marks found</p>"
        rev = f"<img class=page src='{e(pg['review'])}' loading=lazy>" if pg.get("review") else ""
        rows.append(f"<section><h2>{pg['index']}. {e(pg['file'])}</h2><p class=meta>{e(meta)}</p>{err}"
                    f"<div class=grid>{rev}<div class=strips>{strips}</div></div></section>")
    book = manifest.get("book") or {}
    doc = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>Reading companion review</title>
<style>
:root{{--bg:#f6f4ef;--fg:#222;--muted:#666;--card:#fff;--line:#ddd}}
@media (prefers-color-scheme: dark){{:root{{--bg:#1d1c1a;--fg:#eee;--muted:#aaa;--card:#2a2926;--line:#444}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:0 0 18px}}
h1{{font-size:20px}} h2{{font-size:16px;margin:0 0 4px}} .meta{{color:var(--muted);margin:0 0 10px}}
.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr);gap:14px;align-items:start}}
@media (max-width:760px){{.grid{{grid-template-columns:1fr}}}}
img{{max-width:100%;display:block;border-radius:6px}} img.page{{border:1px solid var(--line)}}
figure{{margin:0 0 12px}} figcaption{{color:var(--muted);font-size:12px;word-break:break-all}}
blockquote{{margin:4px 0 0;padding:4px 10px;border-left:3px solid var(--line)}}
.err{{color:#c33}} .none{{color:var(--muted)}}
</style>
<h1>{e(book.get('title') or 'Reading companion')} — {len(manifest['stripes'])} stripes from {len(manifest['pages'])} photos</h1>
{''.join(rows)}"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
