"""Loading photos (EXIF orientation applied; HEIC when pillow-heif is installed) and saving
stripes with their metadata inside the JPEG (EXIF ImageDescription + UserComment, XMP)."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
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
        rgb = np.asarray(im.convert("RGB"))
    return rgb, info


def _xmp(fields: Dict[str, str], description: str, title: str, source: str) -> bytes:
    rc = "".join(f"<rc:{k}>{escape(str(v))}</rc:{k}>" for k, v in fields.items() if v not in (None, ""))
    x = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:rc="https://github.com/tiramisioux/reading-companion/ns/1.0/">'
        f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{escape(title)}</rdf:li></rdf:Alt></dc:title>'
        f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{escape(description)}</rdf:li></rdf:Alt></dc:description>'
        f'<dc:source>{escape(source)}</dc:source>{rc}'
        "</rdf:Description></rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
    return x.encode("utf-8")


def save_stripe(path: Path, rgb: np.ndarray, description: str, meta: Dict, software: str,
                captured: Optional[str] = None, quality: int = 90) -> None:
    """Write the stripe as JPEG carrying its metadata:
    EXIF ImageDescription = a readable summary, EXIF UserComment = the full JSON record,
    XMP dc:title/dc:description/dc:source + rc:* fields (Finder/Spotlight read these)."""
    im = Image.fromarray(rgb)
    exif = Image.Exif()
    exif[TAG_DESCRIPTION] = description
    exif[TAG_SOFTWARE] = software
    exif[TAG_DATETIME] = _dt.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    sub = exif.get_ifd(EXIF_IFD)
    payload = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    sub[TAG_USER_COMMENT] = b"UNICODE\x00" + payload.encode("utf-16-be")
    if captured:
        sub[TAG_DATETIME_ORIGINAL] = captured
    flat = {k: (", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v) for k, v in meta.items()
            if not isinstance(v, dict)}
    xmp = _xmp(flat, description, Path(path).stem, str(meta.get("source", "")))
    im.save(path, "JPEG", quality=quality, optimize=True, exif=exif.tobytes(), xmp=xmp)


def read_stripe_meta(path: Path) -> Dict:
    """The JSON record stored in a stripe's EXIF UserComment (for the ingest step)."""
    with Image.open(path) as im:
        raw = im.getexif().get_ifd(EXIF_IFD).get(TAG_USER_COMMENT)
    if not raw:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    head, body = raw[:8], raw[8:]
    text = body.decode("utf-16-be") if head.startswith(b"UNICODE") else body.decode("utf-8", "replace")
    return json.loads(text)
