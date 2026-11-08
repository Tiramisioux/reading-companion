"""Loading photos (EXIF orientation applied; HEIC when pillow-heif is installed) and saving
stripes with their metadata inside the JPEG (EXIF ImageDescription + UserComment, XMP)."""
from __future__ import annotations

import datetime as _dt
import io
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageOps

try:  # optional HEIC/HEIF support (iPhone photos)
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
    HEIF = True
except Exception:  # pragma: no cover - optional
    HEIF = False

EXIF_IFD = 0x8769
TAG_DESCRIPTION = 0x010E
TAG_SOFTWARE = 0x0131
TAG_DATETIME = 0x0132
TAG_ARTIST = 0x013B
TAG_DATETIME_ORIGINAL = 0x9003
TAG_USER_COMMENT = 0x9286


def load_rgb(path: Path) -> Tuple[np.ndarray, Dict]:
    """RGB uint8 array (EXIF orientation applied) and a few source EXIF facts."""
    if Path(path).suffix.lower() in (".heic", ".heif") and not HEIF:
        raise RuntimeError("HEIC/HEIF photos need pillow-heif: pip install pillow-heif")
    with Image.open(path) as im:
        info = {}
        try:
            ex = im.getexif()
            dto = ex.get_ifd(EXIF_IFD).get(TAG_DATETIME_ORIGINAL) or ex.get(TAG_DATETIME)
            if dto:
                info["captured"] = str(dto)
        except Exception:
            pass
        im = ImageOps.exif_transpose(im)
        if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
            a = np.asarray(im, dtype=np.float64)
            hi = 65535.0 if a.max() > 255 else 255.0
            g = np.clip(a / hi * 255.0, 0, 255).astype(np.uint8)
            rgb = np.stack([g, g, g], axis=2)
        else:
            rgb = np.asarray(im.convert("RGB"))
    return rgb, info


def _xmp(fields: Dict[str, str], description: str, title: str, source: str, keywords: List[str]) -> bytes:
    rc = "".join(f"<rc:{k}>{escape(str(v))}</rc:{k}>" for k, v in fields.items() if v not in (None, ""))
    subj = ("<dc:subject><rdf:Bag>" + "".join(f"<rdf:li>{escape(k)}</rdf:li>" for k in keywords)
            + "</rdf:Bag></dc:subject>") if keywords else ""
    x = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" '
        'xmlns:rc="https://github.com/tiramisioux/reading-companion/ns/1.0/">'
        f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{escape(title)}</rdf:li></rdf:Alt></dc:title>'
        f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{escape(description)}</rdf:li></rdf:Alt></dc:description>'
        f'<photoshop:Headline>{escape(title)}</photoshop:Headline>'
        f'<dc:source>{escape(source)}</dc:source>{subj}{rc}'
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
    return x.encode("utf-8")


def _iptc_segment(caption: str, headline: str, name: str, keywords: List[str]) -> bytes:
    """An APP13 (Photoshop 3.0 / IPTC-IIM) segment: Caption-Abstract, Headline, Object Name and
    Keywords in UTF-8 - the "caption" most photo tools show (Photos, Preview, Lightroom)."""
    def ds(rec: int, num: int, val: bytes) -> bytes:
        val = val[:32000]
        return b"\x1c" + bytes([rec, num]) + struct.pack(">H", len(val)) + val
    data = ds(1, 90, b"\x1b%G") + ds(2, 0, b"\x00\x04")
    data += ds(2, 5, name.encode("utf-8")[:64])
    if headline:
        data += ds(2, 105, headline.encode("utf-8")[:256])
    for k in keywords:
        data += ds(2, 25, k.encode("utf-8")[:64])
    if caption:
        data += ds(2, 120, caption.encode("utf-8")[:2000])
    res = b"8BIM" + struct.pack(">H", 0x0404) + b"\x00\x00" + struct.pack(">I", len(data)) + data
    if len(data) % 2:
        res += b"\x00"
    payload = b"Photoshop 3.0\x00" + res
    return b"\xff\xed" + struct.pack(">H", len(payload) + 2) + payload


def _insert_after_app_segments(jpeg: bytes, seg: bytes) -> bytes:
    """Put a segment after the JFIF/EXIF/XMP (APP0/APP1) segments at the start of a JPEG."""
    if jpeg[:2] != b"\xff\xd8":
        return jpeg
    i = 2
    while i + 4 <= len(jpeg) and jpeg[i] == 0xFF and jpeg[i + 1] in (0xE0, 0xE1):
        i += 2 + struct.unpack(">H", jpeg[i + 2:i + 4])[0]
    return jpeg[:i] + seg + jpeg[i:]


def save_stripe(path: Path, rgb: np.ndarray, title: str, meta: Dict, software: str,
                captured: Optional[str] = None, quality: int = 90, caption: str = "",
                keywords: Optional[List[str]] = None) -> None:
    """Write the stripe as JPEG carrying its metadata.

    caption (the OCR'd words the reader marked; the summary when there is no OCR) goes to
    IPTC Caption-Abstract, XMP dc:description and EXIF ImageDescription (UTF-8, as exiftool
    writes it); the summary (book, chapter, page, note) to IPTC Headline and XMP dc:title /
    photoshop:Headline; the full JSON record to EXIF UserComment."""
    keywords = [k for k in (keywords or []) if k]
    text = caption or title
    im = Image.fromarray(rgb)
    exif = Image.Exif()
    exif[TAG_DESCRIPTION] = text.encode("utf-8")
    exif[TAG_SOFTWARE] = software
    exif[TAG_DATETIME] = _dt.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    exif[0x011A] = 72.0      # XResolution
    exif[0x011B] = 72.0      # YResolution
    exif[0x0128] = 2         # ResolutionUnit: inch
    exif[0x0213] = 1         # YCbCrPositioning: centred
    sub = exif.get_ifd(EXIF_IFD)
    payload = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    sub[TAG_USER_COMMENT] = b"UNICODE\x00" + payload.encode("utf-16-be")
    if captured:
        sub[TAG_DATETIME_ORIGINAL] = captured
    flat = {k: (", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v) for k, v in meta.items()
            if not isinstance(v, (dict, list)) or (isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v))}
    ch = meta.get("chapter") or {}
    if isinstance(ch, dict):
        flat["chapter_number"], flat["chapter_title"] = ch.get("number"), ch.get("title")
    xmp = _xmp(flat, text, title, str(meta.get("source", "")), keywords)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True, exif=exif.tobytes(), xmp=xmp)
    data = _insert_after_app_segments(buf.getvalue(), _iptc_segment(text, title, Path(path).stem, keywords))
    Path(path).write_bytes(data)


def read_stripe_meta(path: Path) -> Dict:
    """The JSON record stored in a stripe's EXIF UserComment (for the ingest step)."""
    with Image.open(path) as im:
        raw = im.getexif().get_ifd(EXIF_IFD).get(TAG_USER_COMMENT)
    if not raw:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    head, body = raw[:8], raw[8:]
    text = body.decode("utf-16-be", "replace") if head.startswith(b"UNICODE") else body.decode("utf-8", "replace")
    try:
        d = json.loads(text)
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {}
