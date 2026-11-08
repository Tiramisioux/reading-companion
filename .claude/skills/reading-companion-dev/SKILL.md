---
name: reading-companion-dev
description: Operate, test and improve the reading-companion extractor, which cuts the passages marked in photographed book pages (highlights, underlines, margin lines, handwritten notes) into stripe JPEGs named by chapter and page. Use when running it on a folder of page photos, reviewing or summarising its manifest, setting it up (venv, OCR, macOS Quick Action), running its tests, scoring it against synthetic scenarios or ground truth, or changing its detection, banding, naming or OCR code.
---

# reading-companion

Deterministic OpenCV pipeline; OCR only reads the page number, running head and chapter
heading. Repository root holds `reading_companion/` (code), `tests/`, `tools/evaluate.py`,
`macos/quick-action.zsh`. Read the repository README for the user-facing rules; read
`references/code-map.md` (next to this file) before changing code.

All commands run from the repository root. The skill's scripts live in
`.claude/skills/reading-companion-dev/scripts/` (abbreviated `$S` below).

```bash
S=.claude/skills/reading-companion-dev/scripts
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

OCR: on macOS `--ocr auto` builds the Apple Vision helper with `swiftc`; elsewhere install
Tesseract (`apt-get install -y tesseract-ocr`, or `brew install tesseract`, plus
`tesseract-lang` for Swedish). Without either, `auto` silently falls back to `none`, and pages
get no page numbers or chapters. HEIC photos need `.venv/bin/pip install pillow-heif`.
Check which engine actually ran in the summary's `ocr` field, not the flag you passed.

## Operate

```bash
.venv/bin/python -m reading_companion extract "/path/to/photos" [--book-title "Title"] [--context 1] [--ocr auto]
```

- Output goes to `<folder>/reading-companion/` (or `--out`): stripe JPEGs, `manifest.json`,
  `index.html` + `review/` for checking by eye.
- stdout is exactly one JSON line: `ok`, `output`, `manifest`, `pages`, `stripes`, `book`,
  `ocr`, `errors`, `seconds`. Progress is JSON on stderr. Exit 1 = runtime error or missing
  input (JSON `error`); exit 2 = usage error.
- A rerun deletes only files the previous manifest listed, so it is safe on a folder the
  user has added files to.
- Photos of book pages are copyrighted. Never commit them, their stripes or their ground
  truth; `local/` is git-ignored for this.

After a run, summarise for the user instead of pasting the manifest:

```bash
python3 $S/summarize_manifest.py "/path/to/photos/reading-companion"
```

It lists per photo the page number and its source, running head, chapter and its source,
marks, stripes, and a "needs a look" list (errors, marks with no stripe, unread page numbers).
Point the user at `index.html` for visual checking. Report pages with `error` explicitly.

macOS Finder integration: `macos/quick-action.zsh` (README section "Right-click a folder in
Finder"). It reads `RC_HOME`, `RC_PYTHON`, `RC_ARGS`, `RC_LOG`; failures go to
`~/Library/Logs/reading-companion.log`.

## Test

Run all three layers after any code change; the first two are fast enough to run every time.

1. **Unit + synthetic end-to-end** (what CI runs on Python 3.9 and 3.12):
   ```bash
   .venv/bin/python -m unittest discover -s tests -t .
   ```
2. **CLI contract** on rendered pages: one JSON line, progress, natural sort, EXIF round trip,
   rerun safety, exit codes. Add `--ocr tesseract` when Tesseract is installed to also check
   that the page number is read.
   ```bash
   .venv/bin/python $S/smoke_cli.py --ocr tesseract
   ```
3. **Scenarios**: synthetic pages that encode the README's stripe rules (consecutive lines
   join, one unmarked line separates and is shared as context, context 0 and 2, skew, first
   and last lines, margin lines and notes, OCR page number).
   ```bash
   .venv/bin/python $S/scenarios.py --ocr tesseract          # all
   .venv/bin/python $S/scenarios.py --only context-2 --keep /tmp/rc-look   # one, kept for viewing
   ```
   PASS/FAIL per scenario; `XFAIL` is a known issue recorded in the scenario file (not a
   failure); `XPASS` means a known issue got fixed, so remove its `known_issue` field.
   The format is documented at the top of `scenarios.py`; scenarios take any keyword argument
   of `tests/synthetic.py: make_page`.

On real photos with hand-made ground truth (kept outside the repository), use the
repository's own scorer, ideally with both contexts:

```bash
.venv/bin/python tools/evaluate.py local/gt.json "/path/to/photos" --context 0
.venv/bin/python tools/evaluate.py local/gt.json "/path/to/photos" --context 1
```

Report results as counts (found / expected, extra stripes, context errors), and quote the
failing lines verbatim.

On hand-labelled stripes (true/false positives, as in the private
`Tiramisioux/reading-companion-fixtures`, `wayfinding/`), use the label scorer. Run it before and after a
detection change and report recall (also "with a mark") and false positives left per category; never accept a
lost true positive that carries a reader's mark:

```bash
.venv/bin/python tools/score_labels.py ../reading-companion-fixtures/wayfinding --jobs 3 --list
```

The categories and the audited label noise are in `tests/fixtures/wayfinding_fp_categories.json` (ids only).
Look at a lost stripe on its photo before calling it noise, and keep everything derived from the photos (review
sheets, crops, notes quoting the book) outside this repository.

## Improve

Work test-first and keep every change deterministic.

1. **Reproduce.** Turn the reported failure into a synthetic scenario (add it to a scenario
   file, or a test in `tests/` if `make_page` can express it). If `make_page` cannot render the
   case (pencil, perspective, a facing page, a chapter opening), extend `tests/synthetic.py`
   with a new keyword argument that defaults to the old behaviour, so existing tests render
   identical pixels.
2. **Locate.** Use the symptom table in `references/code-map.md`, then the debugging snippet
   there to print the marks, their assigned lines and the rejected candidates for one page.
3. **Change the smallest thing.** Keep thresholds in units of the line pitch `p`. Prefer
   tightening the rule that fired wrongly over adding a new special case.
4. **Verify.** Run the three test layers. A heuristic change can fix one photo and break
   others, so if the user has ground truth, run `tools/evaluate.py` before and after and
   report both totals. Any regression in found/extra counts blocks the change unless the user
   accepts it.
5. **Lock it in.** Move the new scenario from `known_issue` to a plain expectation, or add a
   unit test, so CI guards it. Update the README when user-visible behaviour, options or the
   manifest/ingest contract change: the "For an Obsidian ingest step" section is a contract
   other tools rely on, so treat field renames there as breaking. Its consumer is the note-ingest
   plugin's book import (`~/Documents/note-ingest`, `src/bok-import.ts`, `note_ingest/bok.py`): it
   also reads the stderr progress lines, the stdout summary (`stripes`, `errors`), `--book-title`
   and `naming.list_images`, so a change there needs a change there too. In the vault a marked book
   is a notebook (an analog collection of stripes in `_analoga anteckningar/`, untagged passages,
   one subproject note in the project); `_källor/scans/böcker/` is for scans of whole books.

Known open issue (recorded as XFAIL scenario `margin-note-owns-only-its-line`): a one-line
side note with tall loops also takes the line above, because `assign_lines` widens the note's
extent by a quarter pitch before testing line centres. Fixing it needs ground-truth runs on
real photos, since the tolerance probably exists for notes written between lines.

Before committing: run the tests, keep photos and outputs out of the diff (`git status`
should show no `.jpg` or `reading-companion/` output folders), and follow the repository's
existing commit style (`reading-companion: <what changed>`).
