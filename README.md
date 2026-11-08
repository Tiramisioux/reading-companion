# reading-companion

Cut the passages you marked in photographed book pages into **stripes**: horizontal cuts across
the full width of the page, one per marked passage, saved as JPEGs in file order with the
**chapter and page number** in the file name and inside the JPEG.

Deterministic image processing (OpenCV); no language model. OCR is used only to read the
printed page number, running head and chapter heading.

```
img_2001_chapter_10_p_202_note1.jpg
img_2001_chapter_10_p_202_note2.jpg
img_2002_chapter_epilogue_p_221_note1.jpg
img_2003_p_224_note1.jpg          (chapter not known for this page)
img_2004_note1.jpg                (no page number printed on the photo)
```

## What counts as a mark

| Kind | What | Found by |
|---|---|---|
| `highlight` | highlighter of any colour over the text | colour after a local white balance against the paper |
| `underline` | a pen stroke under a printed line | long thin horizontal ink with printed text right above it |
| `margin_line` | a vertical pen line beside lines (often doubled, red or black) | long thin vertical ink outside the text column |
| `margin_note` | handwriting in the side, top or bottom margin | pen ink on bare paper; in the top/bottom margins it must look handwritten (joined letters, wandering baseline) so printed running heads, page numbers and chapter titles are not taken for notes |

Not marks: bleed-through from the other side of the leaf, the facing page, fingers, shadows,
printed rules, the running head and page number.

## Stripe rules

- Marks on **consecutive printed lines are one stripe**, whatever their colour or kind (two
  highlighter colours in one passage, an underline followed by a highlight, a margin line beside
  six lines).
- A completely **unmarked line between marks separates** them.
- Every stripe shows the **line before and the line after** the marked lines (`--context 1`,
  the default; `--context 0` cuts just the marked lines). Two passages one line apart therefore
  share that line, and stay two notes.
- A stripe is a plain horizontal cut across the **full width of the page**, margin to margin, so
  a note in the margin is seen next to the lines it was written beside. The page is straightened
  first (skew and the curl towards the gutter), so the cut follows the printed lines.
- A note in the top margin is shown with the first printed line(s) below it (on a chapter opening
  that includes the chapter title); a note in the bottom margin with the last line(s) above it.
  A note close to the text (within about 2.5 lines) joins a marked first/last line into one stripe.

## Page number and chapter (if present)

| Field | Comes from |
|---|---|
| page number | the number at the end of the running head, or alone in the footer. **Never guessed**: no printed number, no page number. |
| running head | the words of the running head (`WAYFINDING`, `EPILOGUE`, `AM I HERE?`) |
| book title | a running head that repeats on pages of one side (all left pages, say) while the other side changes, or one that matches the folder name or `--book-title` |
| chapter | `chapter_opening`: the page shows the chapter number/title in large print; `running_head`: the running head names the chapter (and is not the book title); `adjacent_page`: the page next to it by page number is in that chapter and no chapter can start between them; `same_page`: another photo of the same page says so |

A chapter number is linked to its title across the folder (a chapter-opening photo of
"10 / Am I Here?" gives the pages headed AM I HERE? the number 10). The file name uses the
chapter number when known, otherwise the title (`chapter_epilogue`).

## Install (macOS)

```bash
cd reading-companion
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

OCR, pick one:

- **Apple Vision** (best): nothing to install beyond the Command Line Tools
  (`xcode-select --install`). The helper `ocr/rc_vision.swift` is compiled once with `swiftc`
  into `~/Library/Caches/reading-companion/`.
- **Tesseract**: `brew install tesseract` (add `tesseract-lang` for Swedish running heads).
- **None**: `--ocr none` (no page numbers or chapters in the names).

iPhone HEIC photos: `.venv/bin/pip install pillow-heif`.

## Use

```bash
.venv/bin/python -m reading_companion extract "/path/to/Wayfinding photos"
```

Output goes to `<folder>/reading-companion/`:

| File | What |
|---|---|
| `*.jpg` | the stripes, in input file order (natural sort: `IMG_2` before `IMG_10`), top to bottom per photo |
| `manifest.json` | every page (page number, running head, chapter + source, skew, line pitch, marks) and every stripe (file, source photo, note number, marked and shown lines, kinds, colours, the stripe's outline in the source photo) |
| `index.html` + `review/` | each photo with its stripes outlined, next to the stripes, for checking by eye |

stdout is one JSON line (`{"ok": true, "stripes": 16, ...}`); progress lines go to stderr.
Running again replaces the files the previous run listed in its manifest (only those).

Options: `--out DIR`, `--context N`, `--ocr auto|vision|tesseract|none`, `--languages`,
`--book-title "Wayfinding"`, `--name-template "{stem}{chapter_part}{page_part}_note{note}"`
(fields: stem, chapter, chapter_part, page, page_part, note, notes, seq, book),
`--quality 90`, `--no-review`, `--quiet`.

### Metadata inside each JPEG

- EXIF `ImageDescription`: `Wayfinding - chapter 10 (Am I Here?) - p. 202 - note 2/3 - yellow highlight, black underline - from IMG_2001.jpg`
- EXIF `UserComment`: the stripe's JSON record (as in the manifest);
  `reading_companion.imageio.read_stripe_meta(path)` reads it back
- XMP: `dc:title`, `dc:description`, `dc:source` and `rc:*` fields

## Right-click a folder in Finder (macOS)

`macos/quick-action.zsh` runs the extractor on the folders (or photos) selected in Finder,
shows a notification with the number of stripes and opens the review page. Set it up once in
Automator: *New → Quick Action*, "Workflow receives current **files or folders** in **Finder**",
add *Run Shell Script* (shell `/bin/zsh`, pass input **as arguments**) with the body

```bash
"$HOME/path/to/reading-companion/macos/quick-action.zsh" "$@"
```

Settings by environment: `RC_HOME`, `RC_PYTHON` (default `.venv/bin/python`), `RC_ARGS`
(e.g. `--context 2`), `RC_LOG` (default `~/Library/Logs/reading-companion.log`).

## For an Obsidian ingest step

What a later ingest can rely on (this tool writes nothing into a vault):

- `manifest.json` → `stripes[]`, in reading order (`seq`): `file`, `source` (the photo),
  `note` / `notes_on_page`, `page_number` (+ `page_number_source`), `chapter`
  (`number`, `title`, `source`), `running_head`, `book`, `lines_marked`, `lines_shown`,
  `kinds`, `colors`, `marks[]`, `size`, `source_polygon` (where the stripe sits on the photo).
- `manifest.json` → `pages[]`: per photo what was read (page number, running head, chapter,
  OCR rows), its stripes, skew and line pitch, and `error` when a photo could not be read.
- Each stripe JPEG carries the same record (EXIF `UserComment`, read it with
  `reading_companion.imageio.read_stripe_meta`) and a readable summary (EXIF
  `ImageDescription`, XMP `dc:description`), so a stripe moved out of its folder still knows
  where it came from.
- Highlight colours are reported as names (`yellow`, `orange`, `pink`, `red`, `green`, `teal`,
  `blue`, `purple`; pen ink `black`/`grey`), so a colour code (yellow = quote, pink = disagree, …)
  can be mapped in the ingest.

## How it works

1. **Geometry**: ink relative to the local paper lightness; skew by projection-profile
   variance; the text column from the density of ink strokes (a facing page across the gutter
   is split off); text lines tracked across slices and a smooth vertical displacement field fitted
   to them, so curled lines come out straight (two passes; the second only inside the column).
2. **White balance**: each pixel divided by the colour of the paper around it (sunlight, shadow
   and lamp light all become neutral paper; highlighter keeps its tint; bleed-through stays
   near neutral).
3. **Layout**: printed lines (checked by their letter texture, so keyboards and shadow edges are
   not lines), the running head, the column edges (they may slant with perspective), the page
   edges and the gutter.
4. **Marks**: highlight, underline, margin line and note detectors (above), each mark assigned
   to the printed lines it touches.
5. **Stripes**: consecutive marked lines grouped, context lines added, cut from the
   full-resolution photo through the straightening map.
6. **Metadata**: OCR of the printed rows above and below the text; then the folder is read as
   one book (book title, chapter numbers and titles, neighbouring pages).

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

The tests render synthetic pages (no book text needed). To score the extractor on your own
photos against hand-made ground truth (probe points on the marked lines and their neighbours):

```bash
.venv/bin/python tools/evaluate.py my-ground-truth.json "/path/to/photos" --context 0
```

Photos of book pages are copyrighted: keep them and their ground truth out of the repository
(`local/` is git-ignored).

## Known limits

- The page's outer edge is not always visible; then the stripe extends a fixed 25 % of the text
  width into the margin and may show a little of the desk or the stacked page edges.
- Handwriting in the top/bottom margins is told from print by shape; a very short note there
  (one or two letters) can be missed. Notes in the side margins are found whatever their size.
- Pencil (light grey) notes are fainter than the ink threshold and may be missed.
- One page per photo: on a two-page spread only the larger/denser page is processed.
- Apple Vision OCR is built and run on macOS only; the tests use Tesseract or none.
