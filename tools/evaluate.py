"""Score the extractor against hand-made ground truth (probe points), on local photos.

    python3 tools/evaluate.py GROUND_TRUTH.json PHOTO_DIR [--context 0|1]

Ground truth (JSON, keyed by photo number or file stem): per photo ``expected_stripes`` with
``include_points`` (on the marked lines: must be inside the stripe) and ``exclude_points`` (on
the unmarked line just above/below: must be outside with --context 0, inside with --context 1).
A stripe matches when it holds at least 80 % of a ground-truth stripe's include points. Extra
stripes (matching nothing) are false positives.

The photos and the ground truth stay outside the repository (book pages are copyrighted).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reading_companion.ocr import OcrEngine  # noqa: E402
from reading_companion.pipeline import Options, process_page  # noqa: E402


def inside(geo, band, pts):
    if not pts:
        return np.zeros(0, dtype=bool)
    a = np.array(pts, dtype=np.float64)
    u, v = geo.source_to_flat(a[:, 0], a[:, 1])
    return (u >= band.x0) & (u <= band.x1) & (v >= band.y0) & (v <= band.y1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path)
    ap.add_argument("photos", type=Path)
    ap.add_argument("--context", type=int, default=0)
    args = ap.parse_args()
    gt = json.loads(args.gt.read_text())
    opts = Options(context=args.context, review=False)
    tot = {"gt": 0, "found": 0, "extra": 0, "excl_bad": 0, "excl_total": 0, "ctx_bad": 0}
    for key in sorted(gt, key=lambda k: int(k) if k.isdigit() else k):
        g = gt[key]
        photo = next((p for p in args.photos.iterdir() if p.stem == key), None)
        if photo is None:
            print(f"{key}: photo not found")
            continue
        res = process_page(photo, int(key) if key.isdigit() else 0, OcrEngine(), opts)
        if res.error:
            print(f"{key}: ERROR {res.error}")
            continue
        used = set()
        lines_out = []
        for s in g["expected_stripes"]:
            tot["gt"] += 1
            best, frac = None, 0.0
            for i, b in enumerate(res.bands):
                f = float(inside(res.geo, b, s["include_points"]).mean())
                if f > frac:
                    best, frac = i, f
            if best is None or frac < 0.8:
                lines_out.append(f"  MISSING lines {s['first_line']}-{s['last_line']} ({s['reason']}) best cover {frac:.0%}")
                continue
            used.add(best)
            tot["found"] += 1
            b = res.bands[best]
            ex = inside(res.geo, b, s["exclude_points"])
            tot["excl_total"] += len(ex)
            if args.context == 0:
                bad = int(ex.sum())
                tot["excl_bad"] += bad
                note = f"adjacent-line points inside: {bad}/{len(ex)}"
            else:
                bad = int((~ex).sum())
                tot["ctx_bad"] += bad
                note = f"context points outside: {bad}/{len(ex)}"
            lines_out.append(f"  ok lines {s['first_line']}-{s['last_line']} -> stripe {best + 1} "
                             f"(marked {b.first_line}-{b.last_line}, shown {b.ctx_first}-{b.ctx_last}), cover {frac:.0%}, {note}")
        extra = [i for i in range(len(res.bands)) if i not in used]
        tot["extra"] += len(extra)
        for i in extra:
            b = res.bands[i]
            lines_out.append(f"  EXTRA stripe {i + 1}: marked {b.first_line}-{b.last_line} kinds {b.kinds}")
        print(f"{key}: {len(g['expected_stripes'])} expected, {len(res.bands)} produced")
        print("\n".join(lines_out))
    print(json.dumps(tot))
    ok = tot["found"] == tot["gt"] and tot["extra"] == 0 and tot["excl_bad"] == 0 and tot["ctx_bad"] == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
