#!/bin/zsh
# Finder Quick Action "Reading companion: cut stripes" (macOS).
#
#   quick-action.zsh <folders or photos selected in Finder...>
#
# Runs the extractor on each selected folder (or on the selected photos, grouped by folder),
# writes the stripes to <folder>/reading-companion/, shows a notification and opens the review
# page. Automator: New Quick Action -> "Workflow receives current: files or folders in Finder"
# -> add "Run Shell Script" (shell /bin/zsh, pass input: as arguments) -> body:
#     "$HOME/path/to/reading-companion/macos/quick-action.zsh" "$@"
#
# Settings (environment; all optional):
#   RC_HOME     the reading-companion folder (default: the folder above this script)
#   RC_PYTHON   python with numpy/opencv/Pillow (default: $RC_HOME/.venv/bin/python)
#   RC_ARGS     extra arguments, e.g. "--context 2 --book-title 'Wayfinding'"
#   RC_LOG      log file (default ~/Library/Logs/reading-companion.log)

RC_HOME="${RC_HOME:-${0:A:h:h}}"
RC_PYTHON="${RC_PYTHON:-$RC_HOME/.venv/bin/python}"
RC_LOG="${RC_LOG:-$HOME/Library/Logs/reading-companion.log}"

notify() {
  osascript -e 'on run argv' -e 'display notification (item 2 of argv) with title (item 1 of argv)' -e 'end run' "Reading companion" "$1" >/dev/null 2>&1 || true
}

if [[ ! -x "$RC_PYTHON" ]]; then
  notify "Python not found: $RC_PYTHON (see README: python3 -m venv .venv)"
  exit 1
fi

typeset -A groups   # folder -> photos (empty = the whole folder)
for item in "$@"; do
  if [[ -d "$item" ]]; then
    groups[${item:A}]=""
  elif [[ -f "$item" ]]; then
    groups[${item:A:h}]+="${item:A}"$'\n'
  fi
done

for folder files in "${(@kv)groups}"; do
  print -r -- "$(date '+%F %T') extract $folder" >>"$RC_LOG"
  if [[ -z "$files" ]]; then
    out=$(cd "$RC_HOME" && eval "\"$RC_PYTHON\" -m reading_companion extract \"$folder\" --quiet $RC_ARGS" 2>>"$RC_LOG")
  else
    out=$(cd "$RC_HOME" && eval "\"$RC_PYTHON\" -m reading_companion extract ${(@q)${(f)files}} --quiet $RC_ARGS" 2>>"$RC_LOG")
  fi
  print -r -- "$out" >>"$RC_LOG"
  stripes=$(print -r -- "$out" | "$RC_PYTHON" -c 'import json,sys; d=json.loads(sys.stdin.read() or "{}"); print(d.get("stripes", d.get("error", "?")))' 2>/dev/null)
  notify "${folder:t}: $stripes stripe(s)"
  [[ -f "$folder/reading-companion/index.html" ]] && open "$folder/reading-companion/index.html"
done
exit 0
