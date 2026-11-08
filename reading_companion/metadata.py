"""Page number and chapter, read from the printed running head, footer and chapter headings.

Only what is printed on a photo is claimed; anything derived is labelled with its source:

page_number          as printed in the running head (``header``) or footer (``footer``); never inferred
running_head         the running-head words (book title on some pages, chapter on others)
chapter              {number, title, source} with source
                     ``chapter_opening``  the page starts the chapter (big number/title)
                     ``running_head``     the running head names the chapter (it is not the book title)
                     ``adjacent_page``    the page next to it by page number is in that chapter and
                                          no chapter can start in between (see ``infer_chapters``)
                     ``same_page``        another photo of the same page number says so
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .ocr import Word

FOLIO_RE = re.compile(r"^[\(\[]?([0-9OoIl|]{1,4})[\)\]\.,]?$")
ROMAN_LOWER = re.compile(r"^[ivxlc]{1,7}$")
ROMAN_UPPER = re.compile(r"^[IVXLC]{1,7}$")
CHAPTER_WORD = re.compile(r"^(chapter|kapitel|part|del|book|bok)\b\.?\s*(.*)$", re.I)
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS_SV = {w: i for i, w in enumerate(
    "noll ett två tre fyra fem sex sju åtta nio tio elva tolv tretton fjorton femton sexton sjutton arton nitton tjugo".split())}


@dataclass
class PrintRow:
    """A printed row outside the body text (running head, page number, chapter heading, footer)."""
    where: str                    # "top" or "bottom"
    x0: float
    y0: float
    x1: float
    y1: float
    height: float                 # median letter height (flat px)
    words: List[Word] = field(default_factory=list)   # OCR words in the row image's pixels
    img_w: int = 1                # width of the OCR'd row image

    @property
    def text(self) -> str:
        return " ".join(w[0] for w in self.words)


@dataclass
class PageMeta:
    page_number: Optional[str] = None
    page_number_value: Optional[int] = None
    page_number_source: Optional[str] = None
    side: Optional[str] = None                 # "left" (verso) / "right" (recto), from the folio's position
    running_head: Optional[str] = None
    chapter_opening: bool = False
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None
    chapter_source: Optional[str] = None
    ocr_rows: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["chapter"] = ({"number": self.chapter_number, "title": self.chapter_title, "source": self.chapter_source}
                        if (self.chapter_number or self.chapter_title) else None)
        for k in ("chapter_number", "chapter_title", "chapter_source"):
            d.pop(k)
        return d


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------

def norm_key(text: Optional[str]) -> str:
    """Casefolded letters and digits only (for comparing running heads and titles)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text).casefold()
    return "".join(ch for ch in t if ch.isalnum())


def roman_value(s: str) -> Optional[int]:
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    s = s.lower()
    if not s or any(ch not in vals for ch in s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = vals[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total if total > 0 else None


def folio_token(text: str) -> Optional[Tuple[str, Optional[int]]]:
    """A printed page number: 1-4 digits (one look-alike letter allowed: O->0, I/l/|->1) or a
    lower-case roman numeral (front matter)."""
    t = text.strip().strip("·•*")
    m = FOLIO_RE.match(t)
    if m:
        raw = m.group(1)
        digits = sum(ch.isdigit() for ch in raw)
        if digits >= 1 and digits >= len(raw) - 1:
            val = int(raw.translate(str.maketrans("OoIl|", "00111")))
            if 0 < val < 5000:
                return str(val), val
    if ROMAN_LOWER.match(t):
        v = roman_value(t)
        if v:
            return t, v
    return None


def chapter_number_token(text: str) -> Optional[str]:
    """A chapter number in a heading: "10", old-style "IO" (= 10), "X", "Chapter 10",
    "CHAPTER TEN", "Kapitel 3", "Part II"."""
    t = text.strip().strip(".:")
    m = CHAPTER_WORD.match(t)
    if m:
        t = m.group(2).strip().strip(".:")
        if not t:
            return None
    low = t.lower()
    if low in NUMBER_WORDS:
        return str(NUMBER_WORDS[low])
    if low in NUMBER_WORDS_SV:
        return str(NUMBER_WORDS_SV[low])
    if re.fullmatch(r"[0-9]{1,3}", t):
        return str(int(t))
    # old-style figures read as letters: "IO" -> 10, "I2" -> 12 (only digits and look-alikes)
    if re.fullmatch(r"[0-9OoIlS]{1,3}", t) and any(ch in "Oo0" for ch in t) or re.fullmatch(r"[0-9]*[OoIl][0-9OoIl]*", t) and any(ch.isdigit() for ch in t):
        v = t.translate(str.maketrans("OoIlS", "00115"))
        if v.isdigit():
            return str(int(v))
    if ROMAN_UPPER.match(t) and len(t) >= 2:
        v = roman_value(t)
        if v:
            return t
    return None


def pretty_title(text: str) -> str:
    """'AM I HERE?' -> 'Am I Here?'; 'W A Y F I N D I N G' -> 'Wayfinding'; keeps mixed case."""
    toks = text.split()
    if len(toks) >= 4 and all(len(t) == 1 for t in toks):
        text = "".join(toks)
    text = re.sub(r"\s+", " ", text).strip(" .,:;-–—|\"'“”‘’()[]")
    if text.isupper():
        small = {"a", "an", "and", "the", "of", "in", "on", "to", "for", "at", "by", "or", "och", "i", "på", "av"}
        out = []
        for i, w in enumerate(text.split()):
            lw = w.lower()
            if lw == "i" and i > 0 and len(text.split()) > 1 and text.split()[i - 1].lower() in ("am", "was", "are"):
                out.append("I")
            elif i > 0 and lw in small and len(w) > 1:
                out.append(lw)
            else:
                out.append(w[:1].upper() + w[1:].lower())
        text = " ".join(out)
    return text


# ---------------------------------------------------------------------------
# one page
# ---------------------------------------------------------------------------

def _lines_of(rows: Sequence[PrintRow]) -> List[List[PrintRow]]:
    """Groups that share a printed line (vertical overlap), left to right."""
    lines: List[List[PrintRow]] = []
    for r in sorted(rows, key=lambda r: (r.y0 + r.y1) / 2.0):
        for ln in lines:
            y0 = min(q.y0 for q in ln); y1 = max(q.y1 for q in ln)
            if min(y1, r.y1) - max(y0, r.y0) > 0.4 * min(y1 - y0, r.y1 - r.y0):
                ln.append(r)
                break
        else:
            lines.append([r])
    return [sorted(ln, key=lambda r: r.x0) for ln in lines]


def read_page(rows: Sequence[PrintRow], pitch: float, col_x0: float, col_x1: float, first_line_top: Optional[float],
              last_line_bottom: Optional[float], min_conf: float = 0.3, letter_h: float = 0.0) -> PageMeta:
    meta = PageMeta()
    col_mid = (col_x0 + col_x1) / 2.0
    letter_h = letter_h or 0.38 * pitch

    def good(r: PrintRow) -> List[Word]:
        return [w for w in r.words if w[1] >= min_conf and w[0].strip()]

    def text_of(groups: Sequence[PrintRow]) -> str:
        return " ".join(w[0] for g in groups for w in good(g)).strip()

    def loose(r: PrintRow) -> List[Word]:
        # Tesseract scores letter-spaced small capitals low even when it reads them right:
        # accept low-confidence words that are mostly letters (or a page number)
        return [w for w in r.words if w[0].strip() and (w[1] >= min_conf or folio_token(w[0]) or
                (w[1] >= 0.1 and sum(ch.isalpha() for ch in w[0]) >= max(3, 0.7 * len(w[0]))))]

    def loose_text(groups: Sequence[PrintRow]) -> str:
        return " ".join(w[0] for g in groups for w in loose(g)).strip()

    tops = [ln for ln in _lines_of([r for r in rows if r.where == "top"])]
    bots = [ln for ln in _lines_of([r for r in rows if r.where == "bottom"])]
    for where, lines in (("top", tops), ("bottom", bots)):
        for ln in lines:
            t = " ".join(w[0] for g in ln for w in g.words)
            if t.strip():
                meta.ocr_rows.append(f"{where}: {t}")

    def height(ln: Sequence[PrintRow]) -> float:
        return max(g.height for g in ln)

    def gap(ln: Sequence[PrintRow]) -> float:
        return (first_line_top - max(g.y1 for g in ln)) if first_line_top is not None else 0.0

    def is_heading(ln: Sequence[PrintRow]) -> bool:
        # chapter headings are large and stand well clear of the text; a running head in small
        # capitals can be almost as tall but sits just above the first line
        hgt = height(ln)
        return hgt >= 2.0 * letter_h or (hgt >= 1.45 * letter_h and gap(ln) >= 2.2 * pitch)

    headings = [ln for ln in tops if is_heading(ln) and text_of(ln)]
    if headings:
        number, title_parts = None, []
        for ln in headings:
            txt = text_of(ln)
            n = chapter_number_token(txt)
            if n is not None and number is None:
                number = n
            elif sum(ch.isalpha() for ch in txt) >= 3:
                title_parts.append(txt)
        if number is not None or title_parts:
            meta.chapter_opening = True
            meta.chapter_number = number
            meta.chapter_title = pretty_title(" ".join(title_parts)) if title_parts else None
            meta.chapter_source = "chapter_opening"

    # running head: the body-sized printed line closest above the body
    head = None
    if first_line_top is not None:
        def meaningful(ln: Sequence[PrintRow]) -> bool:
            toks = loose_text(ln).split()
            letters = sum(ch.isalpha() for t in toks for ch in t)
            return letters >= 4 or any(folio_token(t) for t in toks)

        cands = [ln for ln in tops if not is_heading(ln) and gap(ln) <= 3.5 * pitch and meaningful(ln)]
        if cands:
            head = max(cands, key=lambda ln: max(g.y1 for g in ln))
    if head is not None:
        folio_g = None
        ends = [head[0], head[-1]] if len(head) > 1 else [head[0]]
        for g in ends:  # the page number sits at an end of the line (or alone)
            toks = [w[0] for w in loose(g)]
            if len(toks) == 1 and folio_token(toks[0]):
                folio_g = g
                break
        if folio_g is None:  # one OCR group holding "224 WAYFINDING"
            for g in ends:
                toks = [w[0] for w in loose(g)]
                for tok in (toks[:1] + toks[-1:]) if toks else []:
                    if folio_token(tok):
                        folio_g, ftok = g, tok
                        break
                if folio_g is not None:
                    break
        else:
            ftok = loose(folio_g)[0][0]
        if folio_g is not None:
            txt, val = folio_token(ftok)
            meta.page_number, meta.page_number_value, meta.page_number_source = txt, val, "header"
            meta.side = "left" if (folio_g.x0 + folio_g.x1) / 2.0 < col_mid else "right"
        rest = [w[0] for g in head for w in loose(g) if not (g is folio_g and folio_token(w[0]))]
        words = [t for t in rest if not folio_token(t)]
        head_txt = " ".join(words).strip()
        if sum(ch.isalpha() for ch in head_txt) >= 3:
            meta.running_head = pretty_title(head_txt)
    # footer page number: a lone number below the body
    if meta.page_number is None and last_line_bottom is not None:
        for ln in bots:
            toks = [w[0] for g in ln for w in good(g)]
            if min(g.y0 for g in ln) - last_line_bottom <= 4.0 * pitch and len(toks) == 1 and folio_token(toks[0]):
                txt, val = folio_token(toks[0])
                meta.page_number, meta.page_number_value, meta.page_number_source = txt, val, "footer"
                break
    return meta


# ---------------------------------------------------------------------------
# the whole folder (one book)
# ---------------------------------------------------------------------------

def _lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similar(a: str, b: str) -> bool:
    """Same running head despite OCR slips ('wayfinding' ~ 'wayfindta')."""
    if not a or not b:
        return False
    if a == b:
        return True
    return _lev(a, b) <= max(1, int(0.3 * max(len(a), len(b))))


def unify_running_heads(metas: Sequence[PageMeta]) -> None:
    """Give near-identical running heads (OCR slips) one spelling: the most common one."""
    heads = [m.running_head for m in metas if m.running_head]
    if not heads:
        return
    keys = Counter(norm_key(h) for h in heads)
    canon: Dict[str, str] = {}
    for k, _ in keys.most_common():
        target = next((c for c in canon.values() if similar(k, norm_key(c))), None)
        spelled = next(h for h in heads if norm_key(h) == k)
        canon[k] = target or spelled
    for m in metas:
        if m.running_head:
            m.running_head = canon.get(norm_key(m.running_head), m.running_head)


def detect_book_title(metas: Sequence[PageMeta], folder_name: str = "", book_title: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """(book title, source). A running head that equals the folder name or ``book_title`` is the
    book title; otherwise one that repeats on pages of a single side (all verso, say) while the
    other side shows different heads (chapter names) is."""
    if book_title:
        return book_title, "option"
    heads = [(m, norm_key(m.running_head)) for m in metas if m.running_head]
    fk = norm_key(folder_name)
    if fk:
        for m, k in heads:
            if k and (similar(k, fk) or (len(k) >= 5 and k in fk)):
                return m.running_head, "folder_name"
    counts = Counter(k for _, k in heads if k)
    for k, c in counts.most_common():
        if c < 2:
            break
        mine = [m for m, kk in heads if kk == k]
        others = {kk for _, kk in heads if kk and kk != k}
        sides = {m.side for m in mine if m.side}
        parities = {m.page_number_value % 2 for m in mine if m.page_number_value}
        if others and (len(sides) == 1 or len(parities) == 1):
            return mine[0].running_head, "running_head"
    return None, None


def infer_chapters(metas: List[PageMeta], book_key: str) -> None:
    """Fill chapter_* where the folder supports it (see the module docstring)."""
    # chapter names from running heads that are not the book title
    number_by_title: Dict[str, str] = {}
    for m in metas:
        if m.chapter_opening and m.chapter_title and m.chapter_number:
            number_by_title[norm_key(m.chapter_title)] = m.chapter_number
    for m in metas:
        if m.chapter_source is None and m.running_head and book_key and not similar(norm_key(m.running_head), book_key):
            m.chapter_title = m.running_head
            m.chapter_source = "running_head"
        if m.chapter_title and not m.chapter_number:
            k = norm_key(m.chapter_title)
            m.chapter_number = next((n for t, n in number_by_title.items() if similar(t, k)), None)
        if m.chapter_number and not m.chapter_title:
            for t, n in number_by_title.items():
                if n == m.chapter_number:
                    m.chapter_title = next((x.chapter_title for x in metas if norm_key(x.chapter_title) == t), None)
    # photos of the same page share what one of them shows
    by_num: Dict[int, List[PageMeta]] = {}
    for m in metas:
        if m.page_number_value is not None:
            by_num.setdefault(m.page_number_value, []).append(m)
    for group in by_num.values():
        known = next((g for g in group if g.chapter_source), None)
        head = next((g.running_head for g in group if g.running_head), None)
        for g in group:
            if known is not None and g.chapter_source is None:
                g.chapter_number, g.chapter_title, g.chapter_source = known.chapter_number, known.chapter_title, "same_page"
            if g.running_head is None and head:
                g.running_head = head
    # neighbours: page P+1 is in chapter C and does not start it -> P is in C (a chapter cannot
    # start between them); page P-1 is in C and P does not start a chapter -> P is in C
    changed = True
    while changed:
        changed = False
        for m in metas:
            if m.chapter_source is not None or m.page_number_value is None:
                continue
            nxt = [x for x in by_num.get(m.page_number_value + 1, []) if x.chapter_source and not x.chapter_opening]
            prv = [x for x in by_num.get(m.page_number_value - 1, []) if x.chapter_source]
            src = nxt[0] if nxt else (prv[0] if prv and not m.chapter_opening else None)
            if src is not None:
                m.chapter_number, m.chapter_title, m.chapter_source = src.chapter_number, src.chapter_title, "adjacent_page"
                changed = True


def consolidate(metas: List[PageMeta], folder_name: str = "", book_title: Optional[str] = None) -> Dict:
    unify_running_heads(metas)
    title, source = detect_book_title(metas, folder_name, book_title)
    infer_chapters(metas, norm_key(title) if title else "")
    return {"title": pretty_title(title) if title else None, "title_source": source}
