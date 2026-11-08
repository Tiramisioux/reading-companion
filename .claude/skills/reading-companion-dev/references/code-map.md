# reading-companion code map

Where each behaviour lives, and which file to open for each kind of wrong result.
All paths are relative to the repository root. Line numbers drift; search for the names.

## Pipeline, one photo (`reading_companion/pipeline.py: process_page`)

| Step | Function | Module | What it decides |
|---|---|---|---|
| load | `load_rgb` | `imageio.py` | EXIF orientation, HEIC via pillow-heif |
| geometry | `build_geometry` | `geometry.py` | analysis scale (`ANALYSIS_LONG_SIDE = 2000`), skew (`estimate_skew`, max `MAX_SKEW_DEG = 12`), text block and gutter split (`find_text_block`, `_split_gutter`), line pitch (`estimate_pitch`), curl (`track_lines` + `fit_dewarp`). Produces the flat (straightened) image and the maps `source_to_flat` / `flat_to_source`. |
| analysis | `analyse` | `marks.py` | white balance (`white_balance`), printed lines (`text_lines`, thin pen rules taken out first; `validate_lines` by letter texture, under a dark highlighter on the letters darker than it in the brightest channel), layout (`build_layout`: column edges, page edges, gutter), then `detect_marks` |
| marks | `detect_marks` | `marks.py` | fingers first (`find_skin`: `skin_coloured` and `_off_page`, cut off by the page edge; `an.skin`, `an.skin_boxes`), then five sections, in order: tinted strokes = highlights (outside `an.skin`); dark underlines; margin lines; side-margin notes; top/bottom-margin notes (`head_foot_notes`, `classify_group` tells handwriting from print) |
| lines | `assign_lines` | `bands.py` | which printed lines each mark belongs to; 0 / N+1 = just above / below the text, -1 / N+2 = further away |
| stripes | `make_bands`, `cut_stripe`, `paint_out` | `bands.py` | consecutive marked lines grouped (`GAP_BREAK = 1.7` pitches), context lines added (`CONTEXT_GAP = 4.0`), full-width cut through the straightening map, fingers painted out with the rows' paper colour |
| print rows | `print_rows`, `row_image` | `pipeline.py` | which ink groups outside the body are OCR'd (running head, folio, chapter heading) |
| marked text | `line_row`, `row_image(tight=True, dark=…)`, `dark_highlight`, `marked_words` | `pipeline.py` | the marked body lines, one OCR call per page with the print rows; brightest channel against the local light (a dark highlighter also stretched against the darkest print near it); whole words ≥ 60 % covered |
| caption | `save_stripe(…, caption=, keywords=)` | `imageio.py` | the passage (the marked printed words) → IPTC Caption-Abstract, XMP dc:description, EXIF ImageDescription (UTF-8); summary → Headline / dc:title; book, chapter, page → keywords. **Handwriting is never OCR'd** (user rule 2026-11-08): a margin note's stripe gets the summary as caption, and `print_rows` skips handwritten groups |
| page meta | `read_page` | `metadata.py` | page number from header/footer (`FOLIO_RE`, roman numerals), running head, chapter opening |

## Pipeline, the folder (`process_folder`)

`consolidate` in `metadata.py` reads the folder as one book: `unify_running_heads`,
`detect_book_title` (repeating head on one side, folder name or `--book-title`),
`infer_chapters` (chapter number linked to title; `adjacent_page`, `same_page`). Then names
(`naming.py: stripe_name`, `DEFAULT_TEMPLATE`, natural sort in `list_images`) and writes JPEGs
with EXIF/XMP (`imageio.py: save_stripe`), `manifest.json`, `index.html` and `review/`.
A rerun deletes only what the previous manifest listed (`_clean_previous`).

## Tuning constants

| Constant | File | Effect when raised |
|---|---|---|
| `HL_CHROMA = 10.0` | `marks.py` | fewer pixels count as tinted (pale highlighters missed) |
| `HL_COMPONENT_CHROMA = 15.0` | `marks.py` | a highlight stroke needs a stronger median tint |
| `INK_L = 58.0` | `marks.py` | lighter strokes count as ink (pencil found, but more bleed-through) |
| `COLOR_BUCKETS` | `marks.py` | hue ranges behind the colour names; edit to rename/re-split colours |
| `GAP_BREAK = 1.7` | `bands.py` | marked lines further apart still join one stripe |
| `CONTEXT_GAP = 4.0` | `bands.py` | context lines across a bigger gap (figure, section break) are kept |
| `NEAR_EDGE = 12.0` | `bands.py` | top/bottom notes further from the text still get a stripe |
| `JOIN_GAP = 2.5` | `bands.py` | top/bottom notes further from the text still join line 1 / line N |

Thresholds inside functions are written in units of the line pitch `p` (for example
`0.25 * p`). Keep new thresholds pitch-relative so they hold at any photo resolution.

## Symptom → where to look

| Symptom | First place to look |
|---|---|
| a highlight not found, or its colour wrong | `white_balance`, `HL_CHROMA`, tinted-strokes section of `detect_marks`, `color_name` / `COLOR_BUCKETS` |
| bleed-through or the facing page taken as marks | `white_balance`, `_facing_runs`, `build_layout` step 1, `on_paper` |
| underline missed or put on the wrong line | dark-underlines section of `detect_marks`; underline branch of `assign_lines` (`0.05 * p <= m.yc - ln.center <= 1.0 * p`) |
| margin line missed or too long | `vertical_segments`, `chain_segments`, margin-lines section |
| side note missed | side-margin notes section, `note_ink`, `stroke_like`, `INK_L` |
| side note takes one line too many | note branch of `assign_lines` (`m.y0 - 0.25 * p <= ln.center <= m.y1 + 0.25 * p`) |
| running head or page number taken for a note | `head_foot_notes`, `classify_group`, `_merge_print_groups` |
| two passages merged or one split | `consecutive` inside `make_bands`, `GAP_BREAK`, `validate_lines` (a lost line shifts every index) |
| stripe cut crooked or through a line | `estimate_skew`, `track_lines`, `fit_dewarp` (check `skew_deg`, `dewarp_px` in the manifest) |
| line numbers off by some amount | `text_lines` / `validate_lines` (a missing or extra printed line) |
| no page number, or a wrong one | `print_rows` (row filtered out?), `row_image` (crop), `read_page`, `FOLIO_RE` |
| chapter wrong or missing | `read_page` chapter opening, `infer_chapters`, `detect_book_title` |
| stripe extends into the desk | `find_page_edges` (README known limit: fixed 25 % fallback) |
| a finger taken for a highlight, or a highlight lost to a finger | `find_skin`, `skin_coloured` (hue < 75 or > 340, chroma 8–48), `_off_page`; the finger rejections in the highlight loop (see `r.analysis.rejected`) |
| marked words missing or garbled in `passage` | `dark_highlight` (share > 0.15), `row_image` tight branches, `marked_words` |

## Debugging one photo

```python
import sys; sys.path.insert(0, ".")
from pathlib import Path
from reading_companion.ocr import OcrEngine
from reading_companion.pipeline import Options, process_page
r = process_page(Path("PHOTO.jpg"), 1, OcrEngine(), Options(context=1))
print(r.error, r.info["skew_deg"], r.info["line_pitch_px"], r.info["body_lines"])
for m in r.analysis.marks: print(m.kind, m.color, round(m.y0), round(m.y1), m.lines, m.where)
for ln in r.analysis.lines[:5]: print(ln.index, round(ln.top), round(ln.bottom))
print([(b.first_line, b.last_line, b.ctx_first, b.ctx_last, b.kinds) for b in r.bands])
```

`r.analysis.rejected` holds candidate marks that were turned down, which is usually the
fastest way to see why something real was missed. `r.review` is the review sheet (BGR).
Coordinates of marks and lines are in the flat analysis image; map back with
`r.geo.flat_to_source(xs, ys)`.
