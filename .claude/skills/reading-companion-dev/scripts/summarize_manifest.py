"""Summarise a reading-companion manifest.json for a person: per photo the page number and
chapter (with where each came from), the marks found and the stripes cut, plus anything that
needs a look (errors, pages with marks but no stripes, pages without a page number).

    python3 .claude/skills/reading-companion-dev/scripts/summarize_manifest.py OUTPUT_DIR_OR_MANIFEST

Standard library only; it never changes the output folder.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    p = Path(sys.argv[1]).expanduser()
    man_path = p / "manifest.json" if p.is_dir() else p
    man = json.loads(man_path.read_text())
    stripes_by_src: dict = {}
    for s in man["stripes"]:
        stripes_by_src.setdefault(s["source"], []).append(s)
    book = man.get("book") or {}
    title = book.get("title") if isinstance(book, dict) else book
    print(f"book: {title or '-'}   ocr: {man['ocr']['engine']}   context: {man['options']['context_lines']}")
    print(f"{len(man['pages'])} photos, {len(man['stripes'])} stripes  ->  {man['output']}\n")
    attention = []
    for pg in man["pages"]:
        src = pg["file"]
        if pg.get("error"):
            print(f"{src}: ERROR {pg['error']}")
            attention.append(f"{src}: could not be read ({pg['error']})")
            continue
        ch = pg.get("chapter") if isinstance(pg.get("chapter"), dict) else {}
        chap = ch.get("number") or ch.get("title") or "-"
        chs = ch.get("source")
        pn = pg.get("page_number") or "-"
        kinds = Counter(m["kind"] for m in pg.get("marks", []))
        print(f"{src}: p. {pn} ({pg.get('page_number_source') or 'none'}), head {pg.get('running_head') or '-'!r}, chapter {chap} ({chs or 'none'}), "
              f"skew {pg.get('skew_deg')}°, {pg.get('body_lines')} lines, marks {dict(kinds) or '{}'}")
        for s in stripes_by_src.get(src, []):
            print(f"    {s['file']}: marked {s['lines_marked']} shown {s['lines_shown']} {'/'.join(s['kinds'])} {'/'.join(s['colors'])}")
        if kinds and not stripes_by_src.get(src):
            attention.append(f"{src}: marks found but no stripe cut")
        if pg.get("page_number") is None and man["ocr"]["engine"] != "none":
            attention.append(f"{src}: no page number read")
    if attention:
        print("\nneeds a look:")
        for a in attention:
            print(f"  - {a}")
    review = Path(man["output"]) / "index.html"
    if review.exists():
        print(f"\nreview page: {review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
