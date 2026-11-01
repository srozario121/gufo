#!/usr/bin/env bash
# check-todo gate — validate the hierarchical TODO (pure bash + awk, zero deps).
# Usage: check_todo.sh          (TODO_ROOT overrides the repo root)
#        check_todo.sh --fast   the hook's mode (ADR 0039): ids and todo
#                                       links only, one pass, a few seconds
#        check_todo.sh --xref   item 82: fetch, then fail when one id has two
#                                       different file names across the working tree
#                                       and the origin/* heads (a lost-race duplicate).
#                                       Needs the network, so it is `make todo-xref`,
#                                       not part of `make check`. Offline it prints
#                                       SKIPPED and exits 1, or 0 with TODO_XREF_OFFLINE_OK=1.
#
# Item files live in todo/ and todo/archive/ (ADR 0039); ids are unique across
# both and depends_on / parent / reference_class resolve across both.
#
# FAILS on:
#   * index drift — TODO.md's generated block differs from the item files
#   * file naming — todo/<id>-<slug>.md: id matches front-matter, slug is
#     kebab-case with ≤6 words
#   * duplicate ids
#   * invalid status (see STATUSES below) or contract
#     (legacy · v2)
#   * front-matter schema violations (syntax, unknown keys, missing required
#     keys; the v2 contract adds refined/size/complexity/subscores/estimate/
#     decisions/tasks — schema in the deployment repository's ADR 0002)
#   * v2 complexity that does not re-derive from its six subscores, or a size
#     outside the complexity's band
#   * a resources value outside the closed vocabulary (TODO_RESOURCES in
#     todo_lib.sh; ADR 0021), or a resources value that is not a flow list
#   * a risk outside low|medium|high, or a pre_approved_by that is not
#     '<who> <YYYY-MM-DD>' or has no `risk: low` beside it (ADR 0002)
#   * a moved stub (draft ADR 170) without `moved_to: <repo>#<id>`, with a repo
#     outside TODO_MOVED_REPOS or equal to TODO_SELF_REPO (todo_lib.sh), with an
#     id other than its own, not `contract: legacy`, or with a `tasks:` block or a
#     body checkbox; and a moved_to on an item whose status is not moved
#   * a Markdown link whose target is under todo/ and does not exist (--fast only)
#   * dangling ids — depends_on, parent or estimate.reference_class naming an
#     item that does not exist
#   * depends_on cycles (item level, and task level inside a v2 item)
#   * v2 task-id mismatch — front-matter task ids ≠ the body's **T<k>** ids
#   * a `parallel: true` task depending on a task in its own phase
# WARNS (never fails) on:
#   * an archived item (todo/archive/) whose status is not done
#   * status ↔ checkbox mismatch — `done` with unticked boxes, or a non-done
#     item whose boxes are all ticked
#   * a live (not done, not moved, not archived) item with no `theme:` of its own and no themed
#     parent (item 142.2 T8; slugs in TODO_THEMES, todo_lib.sh)
set -uo pipefail
# Pin the locale before any glob or sort: bash collates pathname expansion by
# LC_COLLATE, so an en_US shell reorders rows that CI (C.UTF-8) does not.
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=todo_lib.sh
. "$HERE/todo_lib.sh"

ROOT="${TODO_ROOT:-.}"
FAST=0 XREF=0
case "${1:-}" in
  --fast) FAST=1 ;;
  --xref) XREF=1 ;;
  '') ;;
  *) echo "usage: check_todo.sh [--fast|--xref]" >&2; exit 2 ;;
esac
fail=0 warns=0
failmsg() { echo "FAIL: $*"; fail=1; }
warnmsg() { echo "WARN: $*"; warns=$((warns + 1)); }

STATUSES=" open refined in-progress partial paused done moved "
LEGACY_KEYS=" id title status contract depends_on parent moved_to resources risk pre_approved_by theme refined size complexity subscores estimate actual decisions tasks "
V2_REQUIRED="id title status contract depends_on refined size complexity subscores estimate decisions tasks"
ESTIMATE_KEYS=" wall_clock_hours tokens usd priced review_hours prs reference_class "
# optional estimate keys: accepted, never required (item 142: box_hours, o/m/p hours of lease time)
ESTIMATE_OPTIONAL=" box_hours "
ACTUAL_KEYS=" merged prs diff calendar_days wall_clock_hours rounds questions defaults overrides tokens partial review_bounces "
SUBSCORE_KEYS=" scope_breadth coupling decision_uncertainty novelty external_dependency verification "
TASK_KEYS=" id phase depends_on parallel acceptance "

[ -d "$ROOT/todo" ] || { echo "check_todo: $ROOT/todo/ not found" >&2; exit 2; }

# --- --xref: cross-ref duplicate ids (item 82) ----------------------------------
# One id with two different basenames across the working tree, origin/main and every
# origin/* branch is a lost-race collision. The same basename on several refs is one
# item (a branch editing it, or an archive move). todo_ref_scan reads the refs.
if [ "$XREF" = 1 ]; then
  if ! todo_xref_fetch "$ROOT"; then
    echo "SKIPPED: could not fetch '${TODO_XREF_REMOTE:-origin}' (offline or no such remote); cross-ref duplicate ids were NOT checked."
    [ "${TODO_XREF_OFFLINE_OK:-0}" = 1 ] && exit 0
    echo "==> todo-xref could not verify (set TODO_XREF_OFFLINE_OK=1 to accept a skip)."
    exit 1
  fi
  rows="$(mktemp)"; trap 'rm -f "$rows"' EXIT
  todo_ref_scan "$ROOT" > "$rows"; rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "SKIPPED: ${TODO_XREF_MAIN:-origin/main} not found after fetch; cross-ref duplicate ids were NOT checked."
    [ "${TODO_XREF_OFFLINE_OK:-0}" = 1 ] && exit 0
    exit 1
  fi
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    b="$(basename "$f")"; printf '%s\t%s\t%s\n' "${b%%-*}" "$b" "working tree" >> "$rows"
  done <<< "$(todo_files "$ROOT")"
  # a gh second source is optional: fork heads are not origin/* refs, so only warn about them
  if [ "${TODO_XREF_GH:-1}" = 1 ] && command -v gh > /dev/null 2>&1; then
    forks="$(cd "$ROOT" && gh pr list --state open --json isCrossRepository --jq '[.[] | select(.isCrossRepository)] | length' 2> /dev/null)" \
      && [ "${forks:-0}" -gt 0 ] && warnmsg "$forks open fork PR(s) are not scanned (their heads are not origin/* refs)" \
      || true
  fi
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    msg=""
    while IFS= read -r b; do
      refs="$(awk -F'\t' -v d="$d" -v b="$b" '$1 == d && $2 == b { print $3 }' "$rows" | sort -u | paste -sd, - | sed 's/,/, /g')"
      msg="$msg${msg:+ vs }$b [$refs]"
    done < <(awk -F'\t' -v d="$d" '$1 == d { print $2 }' "$rows" | sort -u)
    failmsg "item id '$d' is claimed by different files: $msg - the later-opened PR renumbers (make todo-new picks a free id)"
  done < <(sort -u "$rows" | cut -f1,2 | sort -u | cut -f1 | uniq -d)
  n="$(cut -f1 "$rows" | sort -u | wc -l | tr -d ' ')"
  [ "$warns" -eq 0 ] || echo "==> $warns warning(s)."
  if [ "$fail" -ne 0 ]; then echo "==> todo-xref FAILED."; exit 1; fi
  echo "==> todo-xref OK ($n distinct item id(s) across the working tree and ${TODO_XREF_REMOTE:-origin}/*)."
  exit 0
fi

# --- 1. index drift ----------------------------------------------------------
if [ "$FAST" = 0 ] && ! TODO_ROOT="$ROOT" bash "$HERE/gen_todo_index.sh" --check > /dev/null 2>&1; then
  failmsg "TODO.md index is stale or unrenderable — run 'make todo-index' (see gen_todo_index.sh --check)"
fi

# --- per-file checks; collect ids + edges for the cross-file checks ------------
ids_file="$(mktemp)"; edges_file="$(mktemp)"; parents_file="$(mktemp)"; refs_file="$(mktemp)"
trap 'rm -f "$ids_file" "$edges_file" "$parents_file" "$refs_file"' EXIT

# detect_cycle <edges-file> — edges "from<TAB>to"; prints one node on a cycle
detect_cycle() {
  awk -F'\t' '
    { adj[$1] = adj[$1] " " $2; nodes[$1] = 1; nodes[$2] = 1 }
    function visit(n,   i, m, parts, k) {
      if (state[n] == 1) { found = n; return 1 }
      if (state[n] == 2) return 0
      state[n] = 1
      k = split(adj[n], parts, " ")
      for (i = 1; i <= k; i++) if (visit(parts[i])) return 1
      state[n] = 2
      return 0
    }
    END { for (n in nodes) if (visit(n)) { print found; exit } }
  ' "$1"
}

files="$(todo_files "$ROOT")"
[ -n "$files" ] || failmsg "no item files under $ROOT/todo/"

# --fast: one awk pass reads only what the cross-file id checks below need
# (ids, depends_on, parent, reference_class) from every file's front-matter;
# the per-file schema checks stay in the full run.
if [ "$FAST" = 1 ]; then
  printf '%s\n' "$files" | xargs -d '\n' awk -v ids="$ids_file" -v edges="$edges_file" -v parents="$parents_file" -v refs="$refs_file" '
    function flush(   i, n, a) {
      if (file == "") return
      printf "%s\t%s\n", id, file >> ids
      n = split(deps, a, /[][, ]+/); for (i = 1; i <= n; i++) if (a[i] != "") printf "%s\t%s\t%s\n", id, a[i], file >> edges
      if (parent != "") printf "%s\t%s\t%s\n", id, parent, file >> parents
      n = split(rc, a, /[][, ]+/); for (i = 1; i <= n; i++) if (a[i] != "") printf "%s\t%s\t%s\n", id, a[i], file >> refs
    }
    function val(s) { sub(/^[^:]*:[ \t]*/, "", s); sub(/[ \t]+#.*$/, "", s); gsub(/["\047]/, "", s); return s }
    FNR == 1 { flush(); file = FILENAME; id = deps = parent = rc = ""; inm = 1 }
    inm && $0 == "---" && FNR > 1 { inm = 0 }
    !inm { next }
    /^id:/ { id = val($0) }
    /^depends_on:/ { deps = val($0) }
    /^parent:/ { parent = val($0) }
    /^  reference_class:/ { rc = val($0) }
    END { flush() }'
  files=""
fi

while IFS= read -r f; do
  [ -n "$f" ] || continue
  base="$(basename "$f" .md)"
  p="$(todo_parse "$f")"
  if grep -q '^E' <<< "$p"; then
    while IFS=$'\t' read -r _ ln msg; do failmsg "$f:$ln — $msg"; done < <(grep '^E' <<< "$p")
    continue
  fi

  id="$(todo_field "$p" id)"; status="$(todo_field "$p" status)"
  contract="$(todo_field "$p" contract)"

  # an archived item is finished work (ADR 0039); reopening it means moving it back
  if todo_is_archived "$f" && [ "$status" != done ]; then
    warnmsg "$f — archived but status is '$status'; move it back to todo/ (archived items are done)"
  fi

  # naming
  fid="${base%%-*}"; slug="${base#*-}"
  case "$id" in
    ''|*[!0-9.]*|.*|*.|*..*) failmsg "$f — id '$id' must be N or N.K" ;;
  esac
  [ "$fid" = "$id" ] || failmsg "$f — file id '$fid' does not match front-matter id '$id'"
  if [ "$slug" = "$base" ] || ! [[ "$slug" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
    failmsg "$f — slug must be kebab-case: todo/<id>-<slug>.md"
  elif [ "$(tr '-' '\n' <<< "$slug" | wc -l | tr -d ' ')" -gt 6 ]; then
    failmsg "$f — slug '$slug' has more than 6 words"
  fi

  # enums
  # The valid list is PRINTED FROM $STATUSES, not typed out again. A message
  # that enumerates a list by hand drifts from the list it describes the moment
  # one is extended — which is the whole complaint of TODO item 17.
  case "$STATUSES" in *" $status "*) ;;
    *) failmsg "$f — invalid status '$status' (valid:$(printf '%s' "$STATUSES" | sed 's/ / · /g;s/^ · //;s/ · $//'))" ;;
  esac
  case "$contract" in legacy|v2) ;; *) failmsg "$f — invalid contract '$contract' (legacy · v2)" ;; esac

  # keys: unknown + required
  while IFS= read -r k; do
    case "$LEGACY_KEYS" in *" $k "*) ;; *) failmsg "$f — unknown front-matter key '$k'" ;; esac
  done < <(awk -F'\t' '$1 == "F" { print $2 }' <<< "$p")
  for k in id title status contract depends_on; do
    todo_has "$p" "$k" || failmsg "$f — missing required key '$k'"
  done
  dup="$(awk -F'\t' '$1 == "F" { c[$2]++ } END { for (k in c) if (c[k] > 1) print k }' <<< "$p")"
  [ -z "$dup" ] || failmsg "$f — duplicate front-matter key(s): $dup"

  dl="$(todo_field "$p" depends_on)"
  case "$dl" in \[*\]) ;; *) todo_has "$p" depends_on && failmsg "$f — depends_on must be a flow list, e.g. [] or [10, 11]" ;; esac

  # resources: optional; a flow list over the closed vocabulary (ADR 0021). The
  # valid list is printed from TODO_RESOURCES, for the reason given above.
  if todo_has "$p" resources; then
    rl="$(todo_field "$p" resources)"
    case "$rl" in
      \[*\])
        while IFS= read -r r; do
          case "$TODO_RESOURCES" in *" $r "*) ;;
            *) failmsg "$f — unknown resource '$r' (valid:$(printf '%s' "$TODO_RESOURCES" | sed 's/ / · /g;s/^ · //;s/ · $//'))" ;;
          esac
        done < <(todo_list_ids "$rl")
        rdup="$(todo_list_ids "$rl" | sort | uniq -d | paste -sd' ' -)"
        [ -z "$rdup" ] || failmsg "$f — duplicate resource(s): $rdup" ;;
      *) failmsg "$f — resources must be a flow list, e.g. [gpu, serve-outage], got '$rl'" ;;
    esac
  fi

  # risk / pre_approved_by: optional (ADR 0002). pre_approved_by is a human's
  # recorded pre-approval of a low-risk item (item 127), so it needs risk: low.
  rk=""
  if todo_has "$p" risk; then
    rk="$(todo_field "$p" risk)"
    case "$rk" in low|medium|high) ;; *) failmsg "$f — invalid risk '$rk' (valid: low · medium · high)" ;; esac
  fi
  if todo_has "$p" pre_approved_by; then
    pa="$(todo_field "$p" pre_approved_by)"
    [[ "$pa" =~ ^[^[:space:]]+([[:space:]][^[:space:]]+)*[[:space:]][0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] \
      || failmsg "$f — pre_approved_by must be '<who> <YYYY-MM-DD>', got '$pa'"
    [ "$rk" = low ] || failmsg "$f — pre_approved_by is valid only with 'risk: low'"
  fi

  # moved / moved_to (draft ADR 170, item 170.1): a moved stub points at the one
  # repository its item now lives in, under the same id; moved_to means nothing on
  # any other status. The repository list is printed from TODO_MOVED_REPOS.
  if [ "$status" = moved ]; then
    mt="$(todo_field "$p" moved_to)"
    if ! todo_has "$p" moved_to || [ -z "$mt" ]; then
      failmsg "$f — status 'moved' requires 'moved_to: <repo>#<id>'"
    elif ! [[ "$mt" =~ ^([^#[:space:]]+)#([^#[:space:]]+)$ ]]; then
      failmsg "$f — moved_to must be '<repo>#<id>', got '$mt'"
    else
      mrepo="${BASH_REMATCH[1]}" mid="${BASH_REMATCH[2]}"
      case "$TODO_MOVED_REPOS" in *" $mrepo "*) ;;
        *) failmsg "$f — unknown moved_to repository '$mrepo' (valid:$(printf '%s' "$TODO_MOVED_REPOS" | sed 's/ / · /g;s/^ · //;s/ · $//'))" ;;
      esac
      [ "$mrepo" != "$TODO_SELF_REPO" ] || failmsg "$f — moved_to names this repository ('$TODO_SELF_REPO'); a stub points at another one"
      [ "$mid" = "$id" ] || failmsg "$f — moved_to id '$mid' must equal the item's id '$id' (ids are kept)"
    fi
    [ "$contract" = legacy ] || failmsg "$f — a moved stub must be 'contract: legacy', got '$contract'"
    todo_has "$p" tasks && failmsg "$f — a moved stub has no tasks (its work lives in the destination)"
    mbody="$(tail -n "+$(awk -F'\t' '$1 == "B" { print $2 }' <<< "$p")" "$f")"
    grep -qE '^[[:space:]]*([-*]|[0-9]+\.) \[[ xX]\]' <<< "$mbody" \
      && failmsg "$f — a moved stub has no checkboxes (its body is one heading and one line)"
  elif todo_has "$p" moved_to; then
    failmsg "$f — moved_to is valid only with 'status: moved' (status is '$status')"
  fi

  # theme: optional; exactly one scalar slug from TODO_THEMES (item 142.2). The valid
  # list is printed from TODO_THEMES. A missing theme on a live item only warns (below).
  if todo_has "$p" theme; then
    tv="$(todo_field "$p" theme)"
    case "$tv" in
      ""|\[*|\{*|*[[:space:]]*) failmsg "$f — theme must be one scalar slug, e.g. 'theme: gateway', got '${tv:-<empty or block>}'" ;;
      *)
        case "$TODO_THEMES" in *" $tv "*) ;;
          *) failmsg "$f — unknown theme '$tv' (valid:$(printf '%s' "$TODO_THEMES" | sed 's/ / · /g;s/^ · //;s/ · $//'))" ;;
        esac ;;
    esac
  elif [ "$status" != done ] && [ "$status" != moved ] && ! todo_is_archived "$f"; then
    # report-only (item 142.2 T8): a live item should carry a theme of its own or sit under a
    # themed parent (one level, as the roadmap reader inherits it)
    pth=""
    pid="$(todo_field "$p" parent)"
    if [ -n "$pid" ] && pf="$(todo_item_file "$ROOT" "$pid")"; then
      pth="$(todo_field "$(todo_parse "$pf")" theme)"
    fi
    [ -n "$pth" ] || warnmsg "$f — live item has no 'theme:' and no themed parent (add 'theme: <slug>' to the front matter; slugs in TODO_THEMES, todo_lib.sh)"
  fi

  printf '%s\t%s\n' "$id" "$f" >> "$ids_file"
  while IFS= read -r d; do printf '%s\t%s\t%s\n' "$id" "$d" "$f" >> "$edges_file"; done < <(todo_list_ids "$dl")
  parent="$(todo_field "$p" parent)"
  if [ -n "$parent" ]; then
    printf '%s\t%s\t%s\n' "$id" "$parent" "$f" >> "$parents_file"
    case "$id" in "$parent".*) ;; *) failmsg "$f — sub-item id '$id' must be '$parent.<k>' for parent '$parent'" ;; esac
  fi

  # body: task ids and checkboxes
  bstart="$(awk -F'\t' '$1 == "B" { print $2 }' <<< "$p")"
  body="$(tail -n "+$bstart" "$f")"
  open_boxes="$(grep -cE '^[[:space:]]*([-*]|[0-9]+\.) \[ \]' <<< "$body")"
  done_boxes="$(grep -cE '^[[:space:]]*([-*]|[0-9]+\.) \[[xX]\]' <<< "$body")"
  if [ "$status" = done ] && [ "$open_boxes" -gt 0 ]; then
    warnmsg "$f — status 'done' but $open_boxes of $((open_boxes + done_boxes)) checkboxes are unticked"
  elif [ "$status" != done ] && [ "$open_boxes" -eq 0 ] && [ "$done_boxes" -gt 0 ]; then
    warnmsg "$f — status '$status' but all $done_boxes checkboxes are ticked"
  fi

  # --- v2 contract ---
  [ "$contract" = v2 ] || continue
  for k in $V2_REQUIRED; do
    todo_has "$p" "$k" || failmsg "$f — v2 item missing required key '$k'"
  done
  refined="$(todo_field "$p" refined)"
  [[ "$refined" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || failmsg "$f — refined must be YYYY-MM-DD, got '$refined'"
  size="$(todo_field "$p" size)"; cx="$(todo_field "$p" complexity)"
  if ! [[ "$cx" =~ ^([1-9]|10)$ ]]; then
    failmsg "$f — complexity must be an integer 1–10, got '$cx'"
  elif [ "$(todo_size_for "$cx")" != "$size" ]; then
    failmsg "$f — size '$size' does not match complexity $cx (expected $(todo_size_for "$cx"))"
  fi
  # complexity is derived from the six plan-level rubric sub-scores (0–3 each):
  # max(1, ceil(S × 10 / 18)); floors — any 3 → ≥5, three or more 3s → ≥8.
  sum=0 threes=0 subs_ok=1
  for k in $SUBSCORE_KEYS; do
    v="$(todo_map "$p" subscores "$k")"
    if ! [[ "$v" =~ ^[0-3]$ ]]; then
      failmsg "$f — subscores.$k must be 0–3, got '${v:-missing}'"; subs_ok=0; continue
    fi
    sum=$((sum + v)); [ "$v" -eq 3 ] && threes=$((threes + 1))
  done
  while IFS= read -r k; do
    case "$SUBSCORE_KEYS" in *" $k "*) ;; *) failmsg "$f — unknown subscores key '$k'" ;; esac
  done < <(awk -F'\t' '$1 == "M" && $2 == "subscores" { print $3 }' <<< "$p")
  if [ "$subs_ok" -eq 1 ] && [[ "$cx" =~ ^([1-9]|10)$ ]]; then
    derived=$(( (sum * 10 + 17) / 18 )); [ "$derived" -lt 1 ] && derived=1
    [ "$threes" -ge 1 ] && [ "$derived" -lt 5 ] && derived=5
    [ "$threes" -ge 3 ] && [ "$derived" -lt 8 ] && derived=8
    [ "$derived" -eq "$cx" ] || failmsg "$f — complexity $cx does not match its subscores (sum $sum, $threes at 3 → $derived)"
  fi
  for k in $ESTIMATE_KEYS; do
    [ -n "$(todo_map "$p" estimate "$k")" ] || failmsg "$f — estimate.$k is required"
  done
  while IFS= read -r k; do
    case "$ESTIMATE_KEYS$ESTIMATE_OPTIONAL" in *" $k "*) ;; *) failmsg "$f — unknown estimate key '$k'" ;; esac
  done < <(awk -F'\t' '$1 == "M" && $2 == "estimate" { print $3 }' <<< "$p")
  for k in wall_clock_hours review_hours; do
    v="$(todo_map "$p" estimate "$k")"
    [ -z "$v" ] || [[ "$v" =~ ^[0-9.]+/[0-9.]+/[0-9.]+$ ]] || failmsg "$f — estimate.$k must be o/m/p, got '$v'"
  done
  # box_hours: o/m/p hours of exclusive box time (the lease); o<=m<=p (item 142)
  v="$(todo_map "$p" estimate box_hours)"
  if [ -n "$v" ]; then
    if [[ "$v" =~ ^[0-9.]+/[0-9.]+/[0-9.]+$ ]]; then
      IFS=/ read -r bo bm bp <<< "$v"
      awk -v o="$bo" -v m="$bm" -v p="$bp" 'BEGIN { exit !(o + 0 <= m + 0 && m + 0 <= p + 0) }' \
        || failmsg "$f — estimate.box_hours must satisfy o<=m<=p, got '$v'"
    else
      failmsg "$f — estimate.box_hours must be o/m/p, got '$v'"
    fi
  elif [ "$status" = refined ] || [ "$status" = in-progress ]; then
    # report-only: a box-group item should carry box_hours (ADR 0002; filled by plan-review)
    if todo_has "$p" resources; then
      while IFS= read -r r; do
        case "$TODO_BOX_RESOURCES" in *" $r "*)
          warnmsg "$f — declares box-group resource '$r' but estimate.box_hours is unset (plan-review fills it)"; break ;;
        esac
      done < <(todo_list_ids "$(todo_field "$p" resources)")
    fi
  fi
  v="$(todo_map "$p" estimate priced)"
  [ -z "$v" ] || [[ "$v" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || failmsg "$f — estimate.priced must be YYYY-MM-DD, got '$v'"
  v="$(todo_map "$p" estimate reference_class)"
  case "$v" in ''|\[*\]) ;; *) failmsg "$f — estimate.reference_class must be a flow list of item ids, got '$v'" ;; esac
  while IFS= read -r d; do
    [ -n "$d" ] && printf '%s\t%s\t%s\n' "$id" "$d" "$f" >> "$refs_file"
  done < <(todo_list_ids "$v")
  while IFS= read -r k; do
    case "$ACTUAL_KEYS" in *" $k "*) ;; *) failmsg "$f — unknown actual key '$k'" ;; esac
  done < <(awk -F'\t' '$1 == "M" && $2 == "actual" { print $3 }' <<< "$p")

  # tasks: keys, unique ids, in-item deps, cycles, body 1:1
  ntasks="$(awk -F'\t' '$1 == "L" && $2 == "tasks" { if ($3 > n) n = $3 } END { print n + 0 }' <<< "$p")"
  [ "$ntasks" -gt 0 ] || failmsg "$f — v2 item has no tasks"
  fm_ids="$(awk -F'\t' '$1 == "L" && $2 == "tasks" && $4 == "id" { print $5 }' <<< "$p" | sort)"
  task_edges="$(mktemp)"
  i=1
  while [ "$i" -le "$ntasks" ]; do
    tid="$(awk -F'\t' -v i="$i" '$1 == "L" && $2 == "tasks" && $3 == i && $4 == "id" { print $5 }' <<< "$p")"
    [[ "$tid" =~ ^T[0-9]+$ ]] || failmsg "$f — task #$i id must be T<k>, got '$tid'"
    for k in $TASK_KEYS; do
      awk -F'\t' -v i="$i" -v k="$k" '$1 == "L" && $2 == "tasks" && $3 == i && $4 == k { f = 1 } END { exit !f }' <<< "$p" \
        || failmsg "$f — task ${tid:-#$i} missing '$k'"
    done
    while IFS= read -r k; do
      case "$TASK_KEYS" in *" $k "*) ;; *) failmsg "$f — task ${tid:-#$i} has unknown key '$k'" ;; esac
    done < <(awk -F'\t' -v i="$i" '$1 == "L" && $2 == "tasks" && $3 == i { print $4 }' <<< "$p")
    par="$(awk -F'\t' -v i="$i" '$1 == "L" && $2 == "tasks" && $3 == i && $4 == "parallel" { print $5 }' <<< "$p")"
    case "$par" in true|false|'') ;; *) failmsg "$f — task $tid parallel must be true|false" ;; esac
    phase="$(awk -F'\t' -v i="$i" '$1 == "L" && $2 == "tasks" && $3 == i && $4 == "phase" { print $5 }' <<< "$p")"
    tdeps="$(awk -F'\t' -v i="$i" '$1 == "L" && $2 == "tasks" && $3 == i && $4 == "depends_on" { print $5 }' <<< "$p")"
    while IFS= read -r d; do
      [ -n "$d" ] || continue
      grep -qx "$d" <<< "$fm_ids" || failmsg "$f — task $tid depends on unknown task '$d'"
      printf '%s\t%s\n' "$tid" "$d" >> "$task_edges"
      # [P] semantics (Spec Kit): a parallel task may not wait on a same-phase task.
      if [ "$par" = true ]; then
        dphase="$(awk -F'\t' -v d="$d" '$1 == "L" && $2 == "tasks" && $4 == "id" && $5 == d { i = $3 }
                  $1 == "L" && $2 == "tasks" && $4 == "phase" { ph[$3] = $5 } END { print ph[i] }' <<< "$p")"
        [ "$dphase" != "$phase" ] || failmsg "$f — task $tid is parallel but depends on same-phase task $d ($phase)"
      fi
    done < <(todo_list_ids "$tdeps")
    i=$((i + 1))
  done
  dupt="$(uniq -d <<< "$fm_ids")"
  [ -z "$dupt" ] || failmsg "$f — duplicate task id(s): $(paste -sd' ' - <<< "$dupt")"
  c="$(detect_cycle "$task_edges")"; rm -f "$task_edges"
  [ -z "$c" ] || failmsg "$f — task depends_on cycle through $c"
  body_ids="$(grep -oE '\*\*T[0-9]+\*\*' <<< "$body" | tr -d '*' | sort -u)"
  if [ "$(sort -u <<< "$fm_ids")" != "$body_ids" ]; then
    only_fm="$(comm -23 <(sort -u <<< "$fm_ids") <(printf '%s\n' "$body_ids") | paste -sd' ' -)"
    only_body="$(comm -13 <(sort -u <<< "$fm_ids") <(printf '%s\n' "$body_ids") | paste -sd' ' -)"
    failmsg "$f — task ids differ between front-matter and body (front-matter only: ${only_fm:-none}; body only: ${only_body:-none})"
  fi
done <<< "$files"

# --- cross-file: duplicates, dangling, cycles ----------------------------------
while IFS= read -r d; do
  [ -n "$d" ] && failmsg "duplicate item id '$d': $(awk -F'\t' -v d="$d" '$1 == d { print $2 }' "$ids_file" | paste -sd' ' -)"
done < <(cut -f1 "$ids_file" | sort | uniq -d)
# dangling ids: one awk per edge kind (a grep per edge forks hundreds of times)
dangling() { # dangling <edges-file> <what> — print "<file> — <what> names unknown item '<id>'"
  awk -F'\t' -v what="$2" 'NR == FNR { known[$1] = 1; next } !($2 in known) { printf "%s — %s names unknown item \047%s\047\n", $3, what, $2 }' "$ids_file" "$1"
}
while IFS= read -r m; do failmsg "$m"; done < <(dangling "$edges_file" depends_on; dangling "$parents_file" parent; dangling "$refs_file" estimate.reference_class)
c="$(cut -f1,2 "$edges_file" | detect_cycle /dev/stdin)"
[ -z "$c" ] || failmsg "item depends_on cycle through item $c"

# --- --fast: Markdown links into todo/ must resolve -----------------------------
# One awk pass over every tracked .md file (the same link rules as check_links.sh:
# fences, external links, anchors and <placeholders> skipped), reporting only a
# target that resolves under todo/ and does not exist.
if [ "$FAST" = 1 ]; then
  if git -C "$ROOT" rev-parse --git-dir > /dev/null 2>&1; then
    mdlist="$(git -C "$ROOT" ls-files -- '*.md' | grep -v '/vendor/' | grep -v '/node_modules/')"
  else
    mdlist="$(cd "$ROOT" && find . -name '*.md' -not -path './.git/*' | sed 's|^\./||')"
  fi
  have="$(mktemp)"
  (cd "$ROOT" && find todo | sort) > "$have"
  while IFS= read -r bad; do
    [ -n "$bad" ] && failmsg "$bad"
  done < <(cd "$ROOT" && printf '%s\n' "$mdlist" | grep -v '^$' | xargs -d '\n' awk -v have="$have" '
    function norm(p,   n, i, out, k, parts) {
      n = split(p, parts, "/"); k = 0
      for (i = 1; i <= n; i++) {
        if (parts[i] == "" || parts[i] == ".") continue
        if (parts[i] == "..") { if (k > 0) k--; continue }
        out[++k] = parts[i]
      }
      p = ""; for (i = 1; i <= k; i++) p = p (i > 1 ? "/" : "") out[i]
      return p
    }
    BEGIN { while ((getline l < have) > 0) ok[l] = 1 }
    FNR == 1 { fence = 0; dir = FILENAME; sub(/[^\/]*$/, "", dir) }
    /^(```|~~~)/ { fence = 1 - fence; next }
    fence { next }
    {
      rest = $0
      while ((i = index(rest, "](")) > 0) {
        rest = substr(rest, i + 2)
        j = index(rest, ")"); if (j == 0) break
        t = substr(rest, 1, j - 1); rest = substr(rest, j + 1)
        sub(/ .*/, "", t)
        if (t == "" || t ~ /^(https?:|mailto:|#)/ || t ~ /<.*>/) continue
        sub(/#.*/, "", t); if (t == "") continue
        r = norm(dir t)
        if (r !~ /^todo(\/|$)/) continue
        if (!(r in ok)) printf "%s:%d — dangling todo link -> %s\n", FILENAME, FNR, t
      }
    }' 2>&1)
  rm -f "$have"
fi

n="$(wc -l < "$ids_file" | tr -d ' ')"
[ "$warns" -eq 0 ] || echo "==> $warns status/checkbox warning(s) (report-only, not failing)."
if [ "$fail" -ne 0 ]; then
  echo "==> TODO check FAILED."
  exit 1
fi
echo "==> TODO OK ($n item file(s))."
