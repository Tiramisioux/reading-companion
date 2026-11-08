"""File names for stripes and the order of input files.

Default name: ``{stem}{chapter_part}{page_part}_note{note}``, e.g.
``img_2001_chapter_10_p_202_note1.jpg`` - the parts that could not be read are left out
(``img_2001_p_224_note2.jpg``, ``img_2001_note1.jpg``).

Template fields: stem, chapter, chapter_part, page, page_part, note, notes, seq (global order,
zero-padded), book. A missing chapter/page renders as ``x`` in {chapter}/{page} and as an empty
string in {chapter_part}/{page_part}.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif", ".webp", ".bmp"}
DEFAULT_TEMPLATE = "{stem}{chapter_part}{page_part}_note{note}"


def natural_key(name: str):
    """'IMG_2' < 'IMG_10' (Finder order)."""
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", name)]


def list_images(folder: Path, exclude: Iterable[Path] = ()) -> List[Path]:
    ex = {Path(e).resolve() for e in exclude}
    out = []
    for p in Path(folder).iterdir():
        if p.name.startswith(".") or not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if any(par in ex for par in [p.resolve(), *p.resolve().parents]):
            continue
        out.append(p)
    return sorted(out, key=lambda p: natural_key(p.name))


def slug(text: Optional[str], max_len: int = 40) -> str:
    """Lower-case ASCII-ish file-name part: letters, digits and underscores (å/ä/ö kept)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).casefold()
    t = re.sub(r"[^0-9a-zåäöæøüéèàáíóúñç]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t[:max_len].rstrip("_")


def chapter_label(number: Optional[str], title: Optional[str]) -> str:
    if number:
        return slug(number)
    return slug(title, 32)


TEMPLATE_FIELDS = ("stem", "chapter", "chapter_part", "page", "page_part", "note", "notes", "seq", "book")


def check_template(template: str) -> str:
    """'' when the template formats, else what is wrong (unknown field, stray brace)."""
    try:
        template.format(**{k: "x" for k in TEMPLATE_FIELDS})
    except (KeyError, IndexError, ValueError) as exc:
        return f"name template {template!r}: {type(exc).__name__} {exc} (fields: {', '.join(TEMPLATE_FIELDS)})"
    return ""


def stripe_name(template: str, stem: str, chapter: str, page: str, note: int, notes: int, seq: int, book: str = "") -> str:
    fields: Dict[str, str] = {
        "stem": slug(stem, 60) or "page",
        "chapter": chapter or "x",
        "chapter_part": f"_chapter_{chapter}" if chapter else "",
        "page": page or "x",
        "page_part": f"_p_{page}" if page else "",
        "note": str(note),
        "notes": str(notes),
        "seq": f"{seq:04d}",
        "book": slug(book, 40),
    }
    name = template.format(**fields)
    name = re.sub(r"[/\\:\0]", "_", name).strip(" ._") or f"note{note}"
    return name + ".jpg"


def unique(name: str, taken: set) -> str:
    if name not in taken:
        taken.add(name)
        return name
    stem, ext = name.rsplit(".", 1)
    k = 2
    while f"{stem}_{k}.{ext}" in taken:
        k += 1
    out = f"{stem}_{k}.{ext}"
    taken.add(out)
    return out
