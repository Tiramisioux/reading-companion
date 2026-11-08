"""Check the command-line contract end to end on rendered pages: stdout is one JSON line,
progress goes to stderr, files and manifest agree, metadata round-trips through the JPEGs,
a rerun replaces only its own files, and errors exit 1 (missing input) or 2 (usage).

    .venv/bin/python .claude/skills/reading-companion-dev/scripts/smoke_cli.py [--ocr none|tesseract] [--keep DIR]

Run from the repository root. Exit 0 when every check passes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
from reading_companion.imageio import read_stripe_meta  # noqa: E402
from tests.synthetic import make_page  # noqa: E402

FAILS: list = []


def check(cond: bool, what: str) -> None:
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        FAILS.append(what)


def cli(*args: str):
    return subprocess.run([sys.executable, "-m", "reading_companion", *args], cwd=ROOT, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", default="none")
    ap.add_argument("--keep", type=Path, help="leave the rendered folder and output here")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="rc-smoke-"))
    photos = tmp / "Synthetic Book"
    photos.mkdir()
    pages = [dict(highlight={5: "yellow", 6: "teal"}, underline=(12,), seed=1, head="224          WAYFINDING"),
             dict(seed=3),
             dict(margin_line=(4, 6), seed=4, head="WAYFINDING          225")]
    for i, kw in enumerate(pages, 1):
        img, _ = make_page(**kw)
        # IMG_2 / IMG_10 naming checks natural sort
        name = {1: "IMG_2.jpg", 2: "IMG_9.jpg", 3: "IMG_10.jpg"}[i]
        cv2.imwrite(str(photos / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    try:
        r = cli("extract", str(photos), "--ocr", args.ocr)
        check(r.returncode == 0, f"extract exits 0 (got {r.returncode}; stderr tail: {r.stderr[-300:]!r})")
        out_lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        check(len(out_lines) == 1, f"stdout is exactly one line (got {len(out_lines)})")
        summary = json.loads(out_lines[-1]) if out_lines else {}
        check(summary.get("ok") is True, "summary has ok: true")
        for key in ("output", "manifest", "pages", "stripes", "book", "ocr", "errors", "seconds"):
            check(key in summary, f"summary has {key!r}")
        prog = [json.loads(ln) for ln in r.stderr.splitlines() if ln.startswith("{")]
        check(bool(prog) and all("progress" in p for p in prog), "stderr carries JSON progress lines")
        man_path = Path(summary.get("manifest", tmp / "none"))
        check(man_path.exists(), "manifest.json exists")
        man = json.loads(man_path.read_text()) if man_path.exists() else {"pages": [], "stripes": []}
        out = man_path.parent
        check([p["file"] for p in man["pages"]] == ["IMG_2.jpg", "IMG_9.jpg", "IMG_10.jpg"], "pages in natural sort order")
        check(summary.get("stripes") == len(man["stripes"]) == 3, f"3 stripes (got {len(man['stripes'])})")
        check([s["seq"] for s in man["stripes"]] == sorted(s["seq"] for s in man["stripes"]), "stripes in seq order")
        for s in man["stripes"]:
            f = out / s["file"]
            check(f.exists(), f"stripe file {s['file']} exists")
            if f.exists():
                meta = read_stripe_meta(f)
                check(meta.get("source") == s["source"] and meta.get("note") == s["note"], f"{s['file']} EXIF record matches manifest")
        check((out / "index.html").exists() and (out / "review").is_dir(), "review page written")
        if args.ocr != "none":
            p1 = man["pages"][0]
            check(p1.get("page_number") == "224", f"OCR read page number 224 (got {p1.get('page_number')!r})")
        stray = out / "keep-me.txt"
        stray.write_text("not ours")
        before = sorted(p.name for p in out.glob("*.jpg"))
        r2 = cli("extract", str(photos), "--ocr", args.ocr, "--quiet")
        check(r2.returncode == 0 and r2.stderr.strip() == "", "rerun with --quiet: exit 0, empty stderr")
        check(sorted(p.name for p in out.glob("*.jpg")) == before, "rerun gives the same stripe files")
        check(stray.exists(), "rerun leaves files it did not write")
        r3 = cli("extract", str(tmp / "does-not-exist"))
        check(r3.returncode == 1 and "error" in json.loads(r3.stdout or "{}"), "missing input: exit 1 with JSON error")
        r4 = cli("extract", str(photos), "--ocr", "bogus")
        check(r4.returncode == 2 and json.loads(r4.stdout or "{}").get("error", "").startswith("usage"), "bad option: exit 2 with JSON usage error")
        if args.keep:
            shutil.rmtree(args.keep, ignore_errors=True)
            shutil.copytree(tmp, args.keep)
            print(f"kept in {args.keep}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failed check(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
