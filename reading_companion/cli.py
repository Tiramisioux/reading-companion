"""Command line.

    python3 -m reading_companion extract FOLDER [options]

stdout: one JSON line (the summary); progress lines {"progress": 0..1, "file": ...} on stderr;
a failure prints {"error": ...} and exits 1; a usage error exits 2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .naming import DEFAULT_TEMPLATE
from .pipeline import OUT_DIRNAME, Options, process_folder


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        print(json.dumps({"error": f"usage: {message}"}, ensure_ascii=False))
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    ap = JsonArgumentParser(prog="reading_companion", description="Cut the passages you marked in photographed book pages into stripes.")
    ap.add_argument("--version", action="version", version=f"reading-companion {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="extract stripes from a folder of page photos (or single files)")
    ex.add_argument("inputs", nargs="+", type=Path, help="a folder of photos, or photo files")
    ex.add_argument("--out", type=Path, help=f"output folder (default: <folder>/{OUT_DIRNAME})")
    ex.add_argument("--context", type=int, default=1, help="printed lines shown before and after each marked passage (default 1)")
    ex.add_argument("--ocr", default="auto", choices=["auto", "vision", "tesseract", "none"],
                    help="how to read page numbers and chapters (default auto: Vision on macOS, else tesseract, else none)")
    ex.add_argument("--ocr-bin", help="a prebuilt rc-vision helper")
    ex.add_argument("--languages", help="OCR languages (Vision: en-US,sv-SE; tesseract: eng+swe)")
    ex.add_argument("--book-title", help="the book's title, when the running heads cannot tell it")
    ex.add_argument("--name-template", default=DEFAULT_TEMPLATE, help=f"stripe file names (default {DEFAULT_TEMPLATE!r})")
    ex.add_argument("--quality", type=int, default=90, help="JPEG quality (default 90)")
    ex.add_argument("--no-review", action="store_true", help="skip the review sheets and index.html")
    ex.add_argument("--no-text", action="store_true", help="do not read the text of the marked words (OCR)")
    ex.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = [p.expanduser() for p in args.inputs]
        missing = [str(p) for p in inputs if not p.exists()]
        if missing:
            print(json.dumps({"error": f"not found: {', '.join(missing)}"}, ensure_ascii=False))
            return 1
        if len(inputs) == 1 and inputs[0].is_dir():
            folder, files = inputs[0], None
        else:
            files = [p for p in inputs if p.is_file()]
            folder = files[0].parent if files else inputs[0]
        opts = Options(context=max(0, args.context), ocr=args.ocr, ocr_bin=args.ocr_bin, languages=args.languages,
                       book_title=args.book_title, name_template=args.name_template,
                       quality=int(min(100, max(30, args.quality))), review=not args.no_review,
                       text=not args.no_text)

        def progress(frac: float, name: str) -> None:
            if not args.quiet:
                print(json.dumps({"progress": round(frac, 3), "file": name}, ensure_ascii=False), file=sys.stderr, flush=True)

        man = process_folder(folder, args.out, opts, progress, files)
        errors = [{"file": p["file"], "error": p["error"]} for p in man["pages"] if p.get("error")]
        print(json.dumps({"ok": True, "output": man["output"], "manifest": str(Path(man["output"]) / "manifest.json"),
                          "pages": len(man["pages"]), "stripes": len(man["stripes"]), "book": man["book"],
                          "ocr": man["ocr"]["engine"], "errors": errors, "seconds": man["seconds"]}, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover - last resort
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
