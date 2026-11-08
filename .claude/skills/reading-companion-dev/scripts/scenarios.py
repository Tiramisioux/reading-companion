"""Render synthetic book pages from a scenario file, run the real extractor on them and score the
stripes against what the scenario says should come out. No photos or book text needed.

    .venv/bin/python .claude/skills/reading-companion-dev/scripts/scenarios.py [SCENARIOS.json]
        [--only NAME ...] [--ocr none|tesseract|auto] [--keep DIR] [--json]

Run from the repository root (it imports reading_companion and tests.synthetic from there).
Without SCENARIOS.json the bundled default set (scenarios.default.json next to this script) runs.

Scenario format (a JSON list):
  {"name": "two-colours-one-passage",
   "context": 1,                                   # optional, default 1
   "pages": [
     {"page": {"highlight": {"5": "yellow", "6": "teal"}, "underline": [12],
               "margin_line": [16, 18], "note_beside": 10, "angle": 2.5, "seed": 1,
               "head": "224          WAYFINDING", "n_lines": 24},
      "expect": {"marked": [[5, 6], [12, 12], [16, 18]],      # exact list of stripes, in order
                 "shown":  [[4, 7], [11, 13], [15, 19]],       # optional
                 "kinds_include": {"0": ["highlight"]},         # optional, by stripe index
                 "colors_start": {"0": ["yellow", "teal"]},     # optional
                 "page_number": "224"}}]}                       # optional, needs --ocr tesseract
"page" holds keyword arguments of tests/synthetic.py make_page (highlight keys are 1-based lines).
A scenario may carry "known_issue": "why" for an intended behaviour the extractor does not meet
yet: it then reports XFAIL (or XPASS once fixed) and does not fail the run; "needs_ocr": true
skips it under --ocr none. Exit 0 when every scenario passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
try:
    import cv2  # noqa: E402
    from reading_companion.pipeline import Options, process_folder  # noqa: E402
    from tests.synthetic import make_page  # noqa: E402
except ImportError as exc:  # pragma: no cover
    sys.exit(f"run from the reading-companion repository root with its venv ({exc})")

DEFAULT = Path(__file__).with_name("scenarios.default.json")


def page_kwargs(spec: dict) -> dict:
    kw = dict(spec)
    if "highlight" in kw:
        kw["highlight"] = {int(k): v for k, v in kw["highlight"].items()}
    for key in ("underline",):
        if key in kw:
            kw[key] = tuple(kw[key])
    if kw.get("margin_line"):
        kw["margin_line"] = tuple(kw["margin_line"])
    return kw


def check_page(stripes: list, page: dict, expect: dict) -> list:
    problems = []
    got = [list(s["lines_marked"]) for s in stripes]
    if "marked" in expect and got != expect["marked"]:
        problems.append(f"marked {got} != expected {expect['marked']}")
    if "shown" in expect:
        shown = [list(s["lines_shown"]) for s in stripes]
        if shown != expect["shown"]:
            problems.append(f"shown {shown} != expected {expect['shown']}")
    for idx, kinds in expect.get("kinds_include", {}).items():
        i = int(idx)
        have = stripes[i]["kinds"] if i < len(stripes) else []
        missing = [k for k in kinds if k not in have]
        if missing:
            problems.append(f"stripe {i}: kinds {have} lack {missing}")
    for idx, cols in expect.get("colors_start", {}).items():
        i = int(idx)
        have = stripes[i]["colors"][:len(cols)] if i < len(stripes) else []
        if have != cols:
            problems.append(f"stripe {i}: colours {have} != {cols}")
    if "page_number" in expect and page.get("page_number") != expect["page_number"]:
        problems.append(f"page number {page.get('page_number')!r} != {expect['page_number']!r}")
    return problems


def run(scn: dict, ocr: str, keep: Path | None) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"rc-scn-{scn['name']}-"))
    try:
        for i, p in enumerate(scn["pages"], 1):
            img, _ = make_page(**page_kwargs(p.get("page", {})))
            cv2.imwrite(str(tmp / f"IMG_{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        man = process_folder(tmp, opts=Options(ocr=ocr, context=int(scn.get("context", 1)), review=keep is not None))
        problems = []
        for i, p in enumerate(scn["pages"], 1):
            src = f"IMG_{i}.jpg"
            page = next((pg for pg in man["pages"] if pg.get("file") == src), {})
            if page.get("error"):
                problems.append(f"{src}: ERROR {page['error']}")
                continue
            stripes = [s for s in man["stripes"] if s["source"] == src]
            problems += [f"{src}: {msg}" for msg in check_page(stripes, page, p.get("expect", {}))]
        if keep is not None:
            dest = keep / scn["name"]
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(tmp, dest)
        r = {"name": scn["name"], "ok": not problems, "problems": problems, "seconds": man.get("seconds")}
        if scn.get("known_issue"):
            # an expectation the extractor does not meet yet: reported, but not a failure
            r["status"] = "XPASS" if r["ok"] else "XFAIL"
            r["known_issue"] = scn["known_issue"]
            r["ok"] = True
        return r
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="?", type=Path, default=DEFAULT)
    ap.add_argument("--only", nargs="*", default=None, help="run just these scenario names")
    ap.add_argument("--ocr", default="none", choices=["none", "tesseract", "auto", "vision"])
    ap.add_argument("--keep", type=Path, help="copy each rendered folder and its output here, for review by eye")
    ap.add_argument("--json", action="store_true", help="one JSON line per scenario instead of text")
    args = ap.parse_args()
    scns = json.loads(args.scenarios.read_text())
    if args.only:
        scns = [s for s in scns if s["name"] in args.only]
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)
    results = []
    for scn in scns:
        if scn.get("needs_ocr") and args.ocr == "none":
            r = {"name": scn["name"], "ok": True, "skipped": "needs --ocr tesseract", "problems": []}
        else:
            r = run(scn, args.ocr, args.keep)
        results.append(r)
        if args.json:
            print(json.dumps(r, ensure_ascii=False))
        else:
            tag = "SKIP" if r.get("skipped") else r.get("status") or ("PASS" if r["ok"] else "FAIL")
            print(f"{tag}  {r['name']}" + (f"  ({r['skipped']})" if r.get("skipped") else ""))
            for msg in r["problems"]:
                print(f"      {msg}")
            if r.get("status") == "XFAIL":
                print(f"      known issue: {r['known_issue']}")
            elif r.get("status") == "XPASS":
                print("      known issue now passes: drop its known_issue field")
    failed = [r["name"] for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
