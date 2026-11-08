"""Score the extractor against hand-labelled stripes of real photos (true / false positives).

    python3 tools/score_labels.py FIXTURE_DIR [--context 1] [--jobs 3] [--report out.json] [--list]

FIXTURE_DIR holds ``labels.json`` and ``pages/`` (the photos the labelled stripes came from), as
in the private fixtures repository (``wayfinding/``). Every photo is read on its own (no OCR:
detection does not depend on it) and each stripe produced is matched to the labelled stripes of
the same photo by the overlap of their ``source_polygon``: the *vertical IoU*, i.e. per column of
the photo where both stripes lie, the overlap of their [top, bottom] intervals over their union,
averaged over those columns. Pairs at or above ``--iou`` (0.5) are matched one to one, best
first. A labelled stripe no produced stripe matches, but which one produced stripe holds for
``--hold`` (80 %) of its height, counts as found too ("merged": it now shares a stripe).

Reported:

- **recall**: labelled true positives still found, of all of them;
- **false positives left**: labelled false positives still produced (per category when
  ``FIXTURE_DIR/fp_categories.json`` or ``--categories`` gives one per stripe id);
- **unlabelled**: stripes that match no label at all. They come from photos where the labelled
  run produced something else: count them as suspect (most likely false positives) and look.

The photos are copyrighted book pages: they stay outside this repository, and the report
holds only file names, kinds, colours and geometry (never the book's text).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reading_companion.ocr import OcrEngine  # noqa: E402
from reading_companion.pipeline import Options, process_page  # noqa: E402


# ---------------------------------------------------------------------------
# geometry of a stripe outline
# ---------------------------------------------------------------------------

def _edges(poly: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(xs_top, ys_top, xs_bot, ys_bot), x increasing: a stripe outline is its top edge left to
    right, then its bottom edge right to left."""
    a = np.asarray(poly, dtype=np.float64)
    n = len(a) // 2
    top, bot = a[:n], a[n:][::-1]
    ot, ob = np.argsort(top[:, 0]), np.argsort(bot[:, 0])
    return top[ot, 0], top[ot, 1], bot[ob, 0], bot[ob, 1]


def _span(poly) -> Tuple[float, float]:
    a = np.asarray(poly, dtype=np.float64)
    return float(a[:, 0].min()), float(a[:, 0].max())


def vertical_overlap(pa, pb, samples: int = 24) -> Tuple[float, float, float]:
    """(IoU, share of a's height inside b, share of b's height inside a), each averaged over the
    columns both stripes cover. All zero when they share no column."""
    ax0, ax1 = _span(pa)
    bx0, bx1 = _span(pb)
    x0, x1 = max(ax0, bx0), min(ax1, bx1)
    if x1 - x0 < 1.0:
        return 0.0, 0.0, 0.0
    xs = np.linspace(x0, x1, samples)
    ea, eb = _edges(pa), _edges(pb)
    at, ab = np.interp(xs, ea[0], ea[1]), np.interp(xs, ea[2], ea[3])
    bt, bb = np.interp(xs, eb[0], eb[1]), np.interp(xs, eb[2], eb[3])
    inter = np.clip(np.minimum(ab, bb) - np.maximum(at, bt), 0.0, None)
    union = np.maximum(ab, bb) - np.minimum(at, bt)
    ha, hb = np.maximum(ab - at, 1e-6), np.maximum(bb - bt, 1e-6)
    return float(np.mean(inter / np.maximum(union, 1e-6))), float(np.mean(inter / ha)), float(np.mean(inter / hb))


# ---------------------------------------------------------------------------
# one photo
# ---------------------------------------------------------------------------

def run_photo(args) -> Dict:
    path, context = args
    res = process_page(Path(path), 1, OcrEngine(), Options(context=context, review=False, text=False))
    out = {"source": Path(path).name, "error": res.error, "stripes": []}
    for k, (b, poly) in enumerate(zip(res.bands, res.polygons), 1):
        out["stripes"].append({
            "note": k, "kinds": b.kinds, "colors": b.colors,
            "lines_marked": [b.first_line, b.last_line],
            "marks": [{"kind": m.kind, "color": m.color, "where": m.where or None, "why": m.why or None} for m in b.marks],
            "source_polygon": poly,
        })
    return out


def match_photo(labels: List[Dict], produced: List[Dict], iou_min: float, hold_min: float):
    """Pairs (label index, produced index, iou), the labels found only as part of a larger stripe
    (label index, produced index, held share), and the produced stripes that match nothing."""
    pairs = []
    for i, lb in enumerate(labels):
        for j, pr in enumerate(produced):
            iou, held_l, held_p = vertical_overlap(lb["source_polygon"], pr["source_polygon"])
            pairs.append((iou, held_l, held_p, i, j))
    used_l, used_p, matched = set(), set(), []
    for iou, _hl, _hp, i, j in sorted(pairs, reverse=True):
        if iou < iou_min:
            break
        if i in used_l or j in used_p:
            continue
        used_l.add(i)
        used_p.add(j)
        matched.append((i, j, iou))
    merged = []
    for i in range(len(labels)):
        if i in used_l:
            continue
        best = max(((hl, j) for iou, hl, hp, ii, j in pairs if ii == i), default=(0.0, -1))
        if best[0] >= hold_min:
            merged.append((i, best[1], best[0]))
    held_p = {j for _, j, _ in merged}
    unmatched = [j for j in range(len(produced)) if j not in used_p and j not in held_p]
    return matched, merged, unmatched


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def score(data: Dict, results: Dict[str, Dict], categories: Dict[str, str], iou_min: float, hold_min: float) -> Dict:
    """``data``: the labels (labels.json); ``results``: photo name -> ``run_photo`` output."""
    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for s in data["stripes"]:
        by_source[s["source"]].append(s)
    tp_total = sum(1 for s in data["stripes"] if s["label"] == "true_positive")
    fp_total = sum(1 for s in data["stripes"] if s["label"] == "false_positive")
    found: List[Dict] = []
    lost: List[Dict] = []
    fp_left: List[Dict] = []
    fp_gone: List[Dict] = []
    unlabelled: List[Dict] = []
    errors = []
    for src in sorted(by_source):
        labels = by_source[src]
        r = results.get(src)
        if r is None or r.get("error"):
            errors.append({"source": src, "error": (r or {}).get("error", "not run")})
            produced: List[Dict] = []
        else:
            produced = r["stripes"]
        matched, merged, unmatched = match_photo(labels, produced, iou_min, hold_min)
        hit = {i: (j, v, "match") for i, j, v in matched}
        hit.update({i: (j, v, "merged") for i, j, v in merged})
        for i, lb in enumerate(labels):
            rec = {"id": lb["id"], "source": src, "kinds": lb["kinds"], "colors": lb["colors"]}
            if lb["label"] == "false_positive":
                rec["category"] = categories.get(lb["id"], "uncategorised")
            if i in hit:
                j, v, how = hit[i]
                rec.update(how=how, overlap=round(v, 2), now_kinds=produced[j]["kinds"], now_colors=produced[j]["colors"])
                (found if lb["label"] == "true_positive" else fp_left).append(rec)
            else:
                (lost if lb["label"] == "true_positive" else fp_gone).append(rec)
        for j in unmatched:
            pr = produced[j]
            unlabelled.append({"source": src, "note": pr["note"], "kinds": pr["kinds"], "colors": pr["colors"],
                               "marks": pr["marks"], "lines_marked": pr["lines_marked"]})
    cats = Counter(categories.get(s["id"], "uncategorised") for s in data["stripes"] if s["label"] == "false_positive")
    left = Counter(r["category"] for r in fp_left)
    return {
        "true_positives": tp_total, "found": len(found), "recall": round(len(found) / max(1, tp_total), 4),
        "found_merged": sum(1 for r in found if r["how"] == "merged"),
        "false_positives": fp_total, "false_positives_left": len(fp_left),
        "unlabelled": len(unlabelled),
        "per_category": {c: {"labelled": n, "left": left.get(c, 0)} for c, n in sorted(cats.items())},
        "lost": lost, "fp_left": fp_left, "fp_gone": fp_gone, "unlabelled_stripes": unlabelled, "errors": errors,
    }


def print_report(rep: Dict, show: bool) -> None:
    print(f"true positives found : {rep['found']}/{rep['true_positives']}  recall {100 * rep['recall']:.1f} %"
          f"  ({rep['found_merged']} only as part of a larger stripe)")
    print(f"false positives left : {rep['false_positives_left']}/{rep['false_positives']}")
    print(f"unlabelled (suspect) : {rep['unlabelled']}")
    if rep["errors"]:
        print(f"errors               : {len(rep['errors'])}")
    print("false positives per category (left / labelled):")
    for c, v in rep["per_category"].items():
        print(f"  {c:24s} {v['left']:3d} / {v['labelled']:3d}")
    if show:
        for title, key in (("lost true positives", "lost"), ("false positives left", "fp_left"),
                           ("unlabelled stripes", "unlabelled_stripes"), ("errors", "errors")):
            if rep[key]:
                print(f"\n{title}:")
                for r in rep[key]:
                    print("  " + json.dumps(r, ensure_ascii=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fixture", type=Path, help="folder with labels.json and pages/")
    ap.add_argument("--context", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=1, help="photos read in parallel")
    ap.add_argument("--iou", type=float, default=0.5, help="vertical IoU for a match (default 0.5)")
    ap.add_argument("--hold", type=float, default=0.8, help="share of a label's height a larger stripe must hold")
    ap.add_argument("--categories", type=Path, help="JSON {stripe id: category} for the false positives")
    ap.add_argument("--only", nargs="*", help="score only these photos (file names)")
    ap.add_argument("--cache", type=Path, help="read/write the produced stripes here (skip the run when present)")
    ap.add_argument("--report", type=Path, help="write the full report as JSON")
    ap.add_argument("--list", action="store_true", help="list lost, left and unlabelled stripes")
    args = ap.parse_args(argv)

    fixture = args.fixture
    data = json.loads((fixture / "labels.json").read_text(encoding="utf-8"))
    sources = sorted({s["source"] for s in data["stripes"]})
    if args.only:
        keep = set(args.only)
        sources = [s for s in sources if s in keep]
        data["stripes"] = [s for s in data["stripes"] if s["source"] in keep]
    cat_path = args.categories or (fixture / "fp_categories.json")
    if not cat_path.exists():
        alt = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "wayfinding_fp_categories.json"
        cat_path = alt if (args.categories is None and alt.exists()) else cat_path
    categories = json.loads(cat_path.read_text(encoding="utf-8")) if cat_path.exists() else {}
    categories = categories.get("categories", categories)

    results: Dict[str, Dict] = {}
    if args.cache and args.cache.exists():
        results = {r["source"]: r for r in json.loads(args.cache.read_text(encoding="utf-8"))}
    todo = [s for s in sources if s not in results]
    jobs = [(str(fixture / "pages" / s), args.context) for s in todo]
    if jobs:
        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as ex:
                for k, r in enumerate(ex.map(run_photo, jobs), 1):
                    results[r["source"]] = r
                    print(f"\r{k}/{len(jobs)}", end="", file=sys.stderr, flush=True)
        else:
            for k, j in enumerate(jobs, 1):
                r = run_photo(j)
                results[r["source"]] = r
                print(f"\r{k}/{len(jobs)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        if args.cache:
            args.cache.write_text(json.dumps([results[s] for s in sorted(results)], indent=0), encoding="utf-8")

    rep = score(data, {s: results[s] for s in sources if s in results}, categories, args.iou, args.hold)
    print_report(rep, args.list)
    if args.report:
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
