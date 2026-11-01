#!/usr/bin/env bash
# Shared parser for todo/<id>-<slug>.md front-matter (pure bash + awk, zero deps).
# Sourced by gen_todo_index.sh and check_todo.sh. Copied from the ai-sdlc
# deployment repository (item 170.2), which carries more readers of the same format.
#
# The front-matter is a deliberately small YAML subset (documented in the
# deployment repository's ADR 0002, the planning artefact contract):
#
#   key: scalar                 top-level scalar ("quotes" optional)
#   key: [a, b]                 top-level flow list
#   key:                        top-level block, followed by either
#     sub: scalar               two-space-indented map entries, or
#     - id: T1                  two-space-indented list items whose further
#       sub: scalar             keys are indented four spaces
#
# todo_parse <file> prints one normalised record per line, tab-separated:
#   F <key> <value>                    top-level scalar or flow list (raw)
#   M <block> <sub> <value>            map entry inside a block
#   L <block> <index> <sub> <value>    key of the <index>-th list item
#   E <line> <message>                 a syntax error (the file is invalid)
#   B <body line number>               first body line (after the closing ---)
set -uo pipefail

todo_parse() {
  awk -v sq="'" '
    function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
    function unq(s,   a, z) {
      s = trim(s)
      a = substr(s, 1, 1); z = substr(s, length(s), 1)
      if (length(s) >= 2 && a == z && (a == "\"" || a == sq)) s = substr(s, 2, length(s) - 2)
      return s
    }
    function err(msg) { printf "E\t%d\t%s\n", NR, msg; bad = 1 }
    NR == 1 { if ($0 != "---") { err("front-matter must open with ---"); exit } ; next }
    done { next }
    $0 == "---" { printf "B\t%d\n", NR + 1; done = 1; next }
    /^[ \t]*#/ || /^[ \t]*$/ { next }
    /\t/ { err("tab in front-matter"); next }
    /^[a-z_]+:/ {
      key = $0; sub(/:.*/, "", key)
      val = $0; sub(/^[a-z_]+:/, "", val); val = trim(val)
      sub(/[ \t]+#.*$/, "", val)
      if (val == "") { block = key; idx = 0; printf "F\t%s\t\n", key }
      else { block = ""; printf "F\t%s\t%s\n", key, unq(val) }
      next
    }
    /^  - [a-z_]+:/ {
      if (block == "") { err("list item outside a block"); next }
      idx++
      line = $0; sub(/^  - /, "", line)
      k = line; sub(/:.*/, "", k); v = line; sub(/^[a-z_]+:/, "", v)
      printf "L\t%s\t%d\t%s\t%s\n", block, idx, k, unq(v)
      next
    }
    /^    [a-z_]+:/ {
      if (block == "" || idx == 0) { err("indented key outside a list item"); next }
      line = trim($0)
      k = line; sub(/:.*/, "", k); v = line; sub(/^[a-z_]+:/, "", v)
      printf "L\t%s\t%d\t%s\t%s\n", block, idx, k, unq(v)
      next
    }
    /^  [a-z_]+:/ {
      if (block == "") { err("indented key outside a block"); next }
      if (idx > 0) { err("map entry after list items in block " block); next }
      line = trim($0)
      k = line; sub(/:.*/, "", k); v = line; sub(/^[a-z_]+:/, "", v)
      printf "M\t%s\t%s\t%s\n", block, k, unq(v)
      next
    }
    { err("unrecognised front-matter line: " $0) }
    END { if (!done && !bad) printf "E\t%d\t%s\n", NR, "unterminated front-matter" }
  ' "$1"
}

# The closed resources vocabulary (ADR 0021): the optional `resources:` flow
# list may name only these. check_todo.sh validates against it and
# delivery_plan.sh schedules by it, so the list lives here, once.
#   gpu           needs the box's GPU or its memory pool
#   serve-outage  needs the managed server stopped (ADR 0008)
#   serve-slot    drives the managed server's single slot (ADR 0015)
#   serve-pin     changes the pinned serving configuration (ADR 0007)
#   langfuse      changes the Langfuse stack's lifecycle (items 27, 28)
# gpu, serve-outage and serve-slot form the box group: one per wave.
TODO_RESOURCES=" gpu serve-outage serve-slot serve-pin langfuse "
TODO_BOX_RESOURCES=" gpu serve-outage serve-slot "

# The closed theme list (item 142.2): the optional `theme:` key takes exactly one
# of these slugs. check_todo.sh validates against it and prints it in its error;
# the deployment's roadmap page keeps its own label table, one home per consumer.
TODO_THEMES=" pi-harness gufo-engine gateway bench-eval langfuse-tracing session-memory delivery-process monitoring-grafana skill-bank side-models "

# The closed repository list for a moved stub (draft ADR 170, item 170.1): `status: moved`
# requires `moved_to: <repo>#<id>`, <repo> one of these and <id> the stub's own id (ids
# are kept across repositories). In this fork the only entry is ai-sdlc, the deployment
# repository the items moved from: its reference stubs stand for the ids the moved items
# name that stayed there. check_todo.sh validates against it; gen_todo_index.sh skips a
# moved row. No reader opens another repository.
TODO_MOVED_REPOS=" ai-sdlc "
# This repository's own name in that list: a stub may not point at the repository it
# sits in. Each copy of this file sets its own name.
TODO_SELF_REPO="srozario121/gufo"

# todo_field <parsed> <key> — value of a top-level scalar (empty if absent)
todo_field() {
  awk -F'\t' -v k="$2" '$1 == "F" && $2 == k { print $3; exit }' <<< "$1"
}

# todo_has <parsed> <key> — exit 0 if the top-level key is present
todo_has() {
  awk -F'\t' -v k="$2" '$1 == "F" && $2 == k { f = 1 } END { exit !f }' <<< "$1"
}

# todo_map <parsed> <block> <sub>
todo_map() {
  awk -F'\t' -v b="$2" -v k="$3" '$1 == "M" && $2 == b && $3 == k { print $4; exit }' <<< "$1"
}

# todo_list_ids <flow-list> — "[10, 11.2]" -> one id per line
todo_list_ids() {
  local v="$1"
  v="${v#[}"; v="${v%]}"
  tr ',' '\n' <<< "$v" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s/^"//; s/"$//' | sed '/^$/d'
}

# todo_files <root> — the item files of todo/ and todo/archive/ (ADR 0039),
# sorted by numeric id (1, 2, …, 10, 10.1). Ids are unique across both.
todo_files() {
  local root="$1" f
  for f in "$root"/todo/*.md "$root"/todo/archive/*.md; do
    [ -e "$f" ] || continue
    [ "$(basename "$f")" = README.md ] && continue
    printf '%s\n' "$f"
  done | awk -F/ '{ n = $NF; split(n, a, "-"); split(a[1], v, ".");
                    printf "%d\t%d\t%s\n", v[1], (v[2] == "" ? 0 : v[2]), $0 }' \
       | sort -t "$(printf '\t')" -k1,1n -k2,2n | cut -f3-
}

# todo_is_archived <path> — exit 0 if the item file sits under todo/archive/
todo_is_archived() { case "$1" in */todo/archive/*|todo/archive/*) return 0 ;; *) return 1 ;; esac; }

# todo_item_file <root> <id> — the item file for an id, in todo/ or
# todo/archive/ (exit 1 if none). The glob is "<id>-*", so 3 is not 3.1.
todo_item_file() {
  local root="$1" id="$2" f
  for f in "$root/todo/$id"-*.md "$root/todo/archive/$id"-*.md; do
    [ -f "$f" ] && { printf '%s\n' "$f"; return 0; }
  done
  return 1
}

# todo_glyph <status>
todo_glyph() {
  case "$1" in
    open)   printf '[ ]' ;;
    done)   printf '[x]' ;;
    paused) printf '[-]' ;;
    moved)  printf '[>]' ;;
    *)      printf '[~]' ;;
  esac
}

# todo_size_for <complexity> — S / M / L / XL band
todo_size_for() {
  local c="$1"
  if   [ "$c" -le 2 ]; then printf S
  elif [ "$c" -le 4 ]; then printf M
  elif [ "$c" -le 7 ]; then printf L
  else printf XL
  fi
}

# --- the ledger's numbers, shared by todo_ledger.sh and delivery_status.sh ----------

# todo_pert <o/m/p> — PERT expected (o + 4m + p) / 6, 2 dp; empty if not a triple
todo_pert() {
  awk -F/ 'NF == 3 { printf "%.2f", ($1 + 4 * $2 + $3) / 6 }' <<< "$1"
}

# todo_num <s> — exit 0 if s is a plain non-negative number (an actual, a triple part)
todo_num() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

# todo_ledger_rows <root> — one line per valid item that carries an `estimate:`
# block, tab-separated: <id> <o/m/p> <E> <actual wall_clock_hours or empty>.
# The rows `make todo-ledger` prints, as data.
todo_ledger_rows() {
  local f p
  while IFS= read -r f; do
    p="$(todo_parse "$f")"
    grep -q '^E' <<< "$p" && continue
    grep -q $'^F\testimate\t$' <<< "$p" || continue
    local wc; wc="$(todo_map "$p" estimate wall_clock_hours)"
    printf '%s\t%s\t%s\t%s\n' "$(todo_field "$p" id)" "$wc" "$(todo_pert "$wc")" "$(todo_map "$p" actual wall_clock_hours)"
  done < <(todo_files "$1")
}

# todo_calibration <root> — "<c> <n>": the median actual/E over the ledger rows
# with a numeric actual and E > 0 (ADR 0021's ETA calibration), and n, their
# count. c is printed with 4 dp; with n = 0 it is 1.
todo_calibration() {
  todo_ledger_rows "$1" | awk -F'\t' '
    $4 ~ /^[0-9]+(\.[0-9]+)?$/ && $3 + 0 > 0 { r[++n] = $4 / $3 }
    END {
      if (n == 0) { print "1.0000 0"; exit }
      for (i = 2; i <= n; i++) { v = r[i]; for (j = i - 1; j >= 1 && r[j] > v; j--) r[j + 1] = r[j]; r[j + 1] = v }
      m = (n % 2) ? r[(n + 1) / 2] : (r[n / 2] + r[n / 2 + 1]) / 2
      printf "%.4f %d\n", m, n
    }'
}

# --- the plan record's run table (ADR 0021; item 53.2 writes it, 53.3 reads it) -------
# States a run-table row may hold, and the live ones. A live row holds its
# item's resources, so other batches read it as occupied; its implementer
# slots: running and paused k, bounced 1, verifying none (freed at the
# builder's gate). A ready row (built, verified, finalised, waiting for the
# user's merge) has nothing running: it holds no slots and no resources
# (pilot P10), and is not dispatched again.
TODO_RUN_STATES=" running verifying bounced paused ready merged "
TODO_RUN_LIVE=" running verifying bounced paused "

# todo_run_table <plan record> — the run table's rows, one per line,
# |-separated (a tab would let `read` collapse an empty cell): item worktree
# branch dispatched_at estimate state bounces notes.
# Columns are found by their header names, so a record written before the
# Bounces and Notes columns existed still reads (bounces 0, notes empty).
# Prints "E|<message>" and returns 1 when the section or a required
# column is missing, or a row's state is not in TODO_RUN_STATES.
todo_run_table() {
  awk -F'|' -v states="$TODO_RUN_STATES" '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); gsub(/`/, "", s); return s }
    /^## Run table/ { sec = 1; next }
    /^## / { if (sec) sec = 2 }
    sec != 1 || !/^\|/ { next }
    !hdr {
      for (i = 2; i < NF; i++) col[tolower(trim($i))] = i
      split("item worktree branch dispatched_at estimate state", need, " ")
      for (j = 1; j <= 6; j++) if (!(need[j] in col)) { printf "E|the run table has no \"%s\" column\n", need[j]; bad = 1; exit }
      hdr = 1; next
    }
    $0 ~ /^\|[- |:]+\|$/ { next }
    {
      st = trim($(col["state"]))
      if (index(states, " " st " ") == 0) { printf "E|run-table row \"%s\": state \"%s\" is not one of%s\n", trim($(col["item"])), st, states; bad = 1; exit }
      b = ("bounces" in col) ? trim($(col["bounces"])) : "0"; if (b == "") b = "0"
      nt = ("notes" in col) ? trim($(col["notes"])) : ""
      printf "%s|%s|%s|%s|%s|%s|%s|%s\n", trim($(col["item"])), trim($(col["worktree"])), trim($(col["branch"])), trim($(col["dispatched_at"])), trim($(col["estimate"])), st, b, nt
    }
    END { if (!bad && !hdr) { print "E|no \"## Run table\" section with a header row"; bad = 1 } exit bad }
  ' "$1"
}

# todo_ref_scan [root] — the ONLY place that reads other git refs (item 82). Prints
# one row per item file claimed on a ref, tab-separated: <id> <basename> <ref>.
#   * the main ref (TODO_XREF_MAIN, default origin/main): every item file in its
#     tree under todo/ and todo/archive/;
#   * every other <remote>/* branch (TODO_XREF_REMOTE, default origin): only the
#     files it ADDS since its merge-base with the main ref, so a branch that
#     merely predates an item claims nothing, and an archive move (same basename)
#     collides with nothing.
# Plain git (for-each-ref, ls-tree, diff --name-status), no gh, no GNU-only flag.
# Exit 3 if the main ref does not exist (nothing fetched). Read-only.
_TODO_REF_ROWS='
  { p = $0; if (st != "") { if ($1 != "A") next; p = $2 }
    if (p !~ /^todo\/(archive\/)?[0-9]+(\.[0-9]+)?-[^\/]+\.md$/) next
    b = p; sub(/^.*\//, "", b); i = b; sub(/-.*$/, "", i)
    printf "%s\t%s\t%s\n", i, b, ref }'
todo_ref_scan() {
  local root="${1:-.}" remote="${TODO_XREF_REMOTE:-origin}" mainref="${TODO_XREF_MAIN:-origin/main}"
  local ref base
  git -C "$root" rev-parse --verify -q "$mainref^{commit}" > /dev/null || return 3
  git -C "$root" ls-tree -r --name-only "$mainref" -- todo | awk -v ref="$mainref" -v st= "$_TODO_REF_ROWS"
  while IFS= read -r ref; do
    [ -n "$ref" ] && [ "$ref" != "$remote" ] && [ "$ref" != "$remote/HEAD" ] && [ "$ref" != "$mainref" ] || continue
    base="$(git -C "$root" merge-base "$mainref" "$ref" 2> /dev/null)" || continue
    # A path that main's history already touched was squash-merged from this branch
    # (the branch lives on, stale): not a claim on an id, so it is dropped here.
    git -C "$root" diff --name-status --no-renames "$base" "$ref" -- todo \
      | while IFS=$'\t' read -r st path; do
          [ -z "$(git -C "$root" log -1 --format=%H "$mainref" -- "$path" 2> /dev/null)" ] && printf '%s\t%s\n' "$st" "$path" || :
        done | awk -F'\t' -v ref="$ref" -v st=1 "$_TODO_REF_ROWS"
  done < <(git -C "$root" for-each-ref --format='%(refname:short)' "refs/remotes/$remote")
}

# todo_xref_fetch [root] — refresh the remote's refs before a scan. Exit 0 on
# success, 1 when it could not fetch (offline, no remote): the callers decide
# whether that fails or skips.
todo_xref_fetch() {
  git -C "${1:-.}" fetch --quiet --prune "${TODO_XREF_REMOTE:-origin}" > /dev/null 2>&1
}
