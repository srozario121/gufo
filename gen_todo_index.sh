#!/usr/bin/env bash
# Generate the TODO.md index from todo/<id>-<slug>.md front-matter (pure bash).
# Usage:
#   gen_todo_index.sh          rewrite the index block in TODO.md
#   gen_todo_index.sh --check  verify no drift; exit non-zero if stale
#
# The block between <!-- todo:index:start --> and <!-- todo:index:end --> is
# generated: one table of active items (open · refined · in-progress · partial),
# a collapsed table of paused items when any exist,
# a collapsed table of moved stubs (draft ADR 170) when any exist, each row
# naming its destination,
# a collapsed table of done items, and a collapsed table of archived items
# (todo/archive/, ADR 0039) when any exist. Everything outside the markers is
# hand-written. TODO_ROOT overrides the repository root (fixture selftest).
set -uo pipefail
# Pin the locale before any glob or sort: bash collates pathname expansion by
# LC_COLLATE, so an en_US shell reorders rows that CI (C.UTF-8) does not.
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=todo_lib.sh
. "$HERE/todo_lib.sh"

ROOT="${TODO_ROOT:-.}"
INDEX="$ROOT/TODO.md"
START='<!-- todo:index:start -->'
END='<!-- todo:index:end -->'

[ -f "$INDEX" ] || { echo "gen_todo_index: $INDEX not found" >&2; exit 2; }
[ -d "$ROOT/todo" ] || { echo "gen_todo_index: $ROOT/todo/ not found" >&2; exit 2; }
if [ "$(grep -cxF "$START" "$INDEX")" != 1 ] || [ "$(grep -cxF "$END" "$INDEX")" != 1 ]; then
  echo "gen_todo_index: $INDEX needs exactly one '$START' and one '$END' line" >&2
  exit 4
fi

row() {
  local f="$1" p id status_key title status size deps parent link cell moved_to
  p="$(todo_parse "$f")"
  if grep -q '^E' <<< "$p"; then
    echo "gen_todo_index: $f has invalid front-matter (run make check-todo)" >&2
    return 3
  fi
  id="$(todo_field "$p" id)"
  title="$(todo_field "$p" title)"; title="${title//|/\\|}"
  status="$(todo_field "$p" status)"
  size="$(todo_field "$p" size)"; [ -n "$size" ] || size="—"
  parent="$(todo_field "$p" parent)"
  deps="$(todo_list_ids "$(todo_field "$p" depends_on)" | paste -sd, - | sed 's/,/, /g')"
  [ -n "$deps" ] || deps="—"
  link="${f#"$ROOT"/}"
  # An archived item files under Archived whatever its status says.
  todo_is_archived "$f" && status_key="archived" || status_key="$status"
  cell="[${title}](${link})"
  [ -n "$parent" ] && cell="↳ ${cell}"
  # a moved stub names where its item lives now; the Status cell still ends in the status word
  moved_to="$(todo_field "$p" moved_to)"
  [ "$status" = moved ] && [ -n "$moved_to" ] && cell="${cell} → \`${moved_to//|/\\|}\`"
  printf '%s\t| %s | %s | `%s` %s | %s | %s |\n' \
    "$status_key" "$id" "$cell" "$(todo_glyph "$status")" "$status" "$size" "$deps"
}

render_block() {
  local f rows="" r active done_rows paused_rows moved_rows archived_rows n_active=0 n_done=0 n_paused=0 n_moved=0 n_archived=0
  while IFS= read -r f; do
    r="$(row "$f")" || exit 3
    rows="${rows}${r}"$'\n'
  done < <(todo_files "$ROOT")
  # PAUSED IS NEITHER ACTIVE NOR DONE. Leaving a paused item in Active is the
  # whole problem it exists to solve — it would keep being read as the next
  # thing to pick up — and filing it under Done would be a lie about work that
  # has not happened. A MOVED stub is neither either: its work lives in another
  # repository (draft ADR 170), so it is listed apart, with its destination.
  active="$(awk -F'\t' '$1 != "done" && $1 != "paused" && $1 != "moved" && $1 != "archived" && $1 != "" { print $2 }' <<< "$rows")"
  done_rows="$(awk -F'\t' '$1 == "done" { print $2 }' <<< "$rows")"
  paused_rows="$(awk -F'\t' '$1 == "paused" { print $2 }' <<< "$rows")"
  moved_rows="$(awk -F'\t' '$1 == "moved" { print $2 }' <<< "$rows")"
  archived_rows="$(awk -F'\t' '$1 == "archived" { print $2 }' <<< "$rows")"
  [ -n "$active" ] && n_active="$(wc -l <<< "$active" | tr -d ' ')"
  [ -n "$done_rows" ] && n_done="$(wc -l <<< "$done_rows" | tr -d ' ')"
  [ -n "$paused_rows" ] && n_paused="$(wc -l <<< "$paused_rows" | tr -d ' ')"
  [ -n "$moved_rows" ] && n_moved="$(wc -l <<< "$moved_rows" | tr -d ' ')"
  [ -n "$archived_rows" ] && n_archived="$(wc -l <<< "$archived_rows" | tr -d ' ')"
  echo ""
  echo "### Active ($n_active)"
  echo ""
  echo "| # | Item | Status | Size | Depends on |"
  echo "|---|------|--------|------|------------|"
  if [ -n "$active" ]; then printf '%s\n' "$active"; else echo "| — | _None_ |  |  |  |"; fi
  echo ""
  if [ -n "$paused_rows" ]; then
    echo "<details>"
    echo "<summary>Paused ($n_paused)</summary>"
    echo ""
    echo "| # | Item | Status | Size | Depends on |"
    echo "|---|------|--------|------|------------|"
    printf '%s\n' "$paused_rows"
    echo ""
    echo "</details>"
    echo ""
  fi
  if [ -n "$moved_rows" ]; then
    echo "<details>"
    echo "<summary>Moved ($n_moved)</summary>"
    echo ""
    echo "| # | Item | Status | Size | Depends on |"
    echo "|---|------|--------|------|------------|"
    printf '%s\n' "$moved_rows"
    echo ""
    echo "</details>"
    echo ""
  fi
  echo "<details>"
  echo "<summary>Done ($n_done)</summary>"
  echo ""
  echo "| # | Item | Status | Size | Depends on |"
  echo "|---|------|--------|------|------------|"
  if [ -n "$done_rows" ]; then printf '%s\n' "$done_rows"; else echo "| — | _None_ |  |  |  |"; fi
  echo ""
  echo "</details>"
  echo ""
  if [ -n "$archived_rows" ]; then
    echo "<details>"
    echo "<summary>Archived ($n_archived)</summary>"
    echo ""
    echo "| # | Item | Status | Size | Depends on |"
    echo "|---|------|--------|------|------------|"
    printf '%s\n' "$archived_rows"
    echo ""
    echo "</details>"
    echo ""
  fi
}

render() {
  local line block
  block="$(render_block)" || exit 3
  while IFS= read -r line || [ -n "$line" ]; do
    if [ "$line" = "$START" ]; then
      printf '%s\n%s\n' "$line" "$block"
      skip=1; continue
    fi
    if [ "$line" = "$END" ]; then skip=""; printf '%s\n' "$line"; continue; fi
    [ -n "${skip:-}" ] && continue
    printf '%s\n' "$line"
  done < "$INDEX"
}

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
render > "$tmp" || exit 3

if [ "${1:-}" = "--check" ]; then
  if cmp -s "$tmp" "$INDEX"; then
    echo "==> TODO.md index OK (no drift)."
    exit 0
  fi
  echo "==> TODO.md index is STALE. Diff (current vs generated):"
  diff -u "$INDEX" "$tmp" || true
  echo "==> Run 'make todo-index' to regenerate it."
  exit 1
fi

if cmp -s "$tmp" "$INDEX"; then
  echo "==> TODO.md index already current."
else
  cat "$tmp" > "$INDEX"
  echo "==> Regenerated the TODO.md index."
fi
