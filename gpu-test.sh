#!/usr/bin/env bash
# gpu-test.sh — the GPU half of a port's checks: a fork branch (the candidate)
# against the upstream release it sits on (the base), built and run inside the
# same container image and Nix store volume as build.sh and test.sh, with
# nothing installed on the host. Run from the fork's main clone or any worktree:
#
#   ai-sdlc/gpu-test.sh prepare   no lease needed: build both trees, no GPU
#   ai-sdlc/gpu-test.sh run       under a lease with the engine stopped: the GPU
#                                 operator tests, then the serve identity checks
#   ai-sdlc/gpu-test.sh status    what is built, and whether `run` may start
#
# prepare (the engine may be serving):
#   1. Refusals: a ref that does not resolve; a candidate not descended from the
#      base; NIX_IMAGE not by digest or not local; MemAvailable below
#      MEM_FLOOR_MIB + GF_BUILD_MEM_MIB.
#   2. `git archive` of each commit into the output folder (never the working
#      tree, so the build is exactly the commit).
#   3. One container per tree, capped at GF_BUILD_MEM_MIB with no extra swap and
#      GF_BUILD_CORES CPUs, no devices: inside `nix develop <tree>#default` (the
#      production package's own toolchain), the gpu-test configuration
#      (RelWithDebInfo, assertions on, HIP on) with GUFO_ROCR_LIBRARY_DIR set to
#      the package's patched ROCr, as the Nix shells set it. The base builds
#      `gufo`; the candidate builds `gufo` and `qwen38_flash_next_tests`. A tree
#      already built at the same commit is skipped. build.sh's watcher stops a
#      container at the memory floor or the store budget.
#
# run (the box is leased to the caller):
#   1. Refusals: prepare not finished for both trees; no lease lock at
#      GF_LEASE_LOCK; another running container with /dev/kfd mapped (the
#      engine, or another test); MemAvailable below GF_RUN_MIN_AVAIL_MIB; no /dev/kfd;
#      a model file missing. The lease is never taken or given back here.
#   2. GPU operator tests: `ctest -L gpu -R ^qwen38_flash_next\.` on the
#      candidate's build, with /dev/kfd and /dev/dri.
#   3. Serve identity checks, one server per arm on 127.0.0.1:GF_PORT, greedy
#      (temperature 0, seed 1), thinking off, MTP on, GF_MAX_TOKENS tokens, the
#      prompts below:
#        base-off   the base, no fork flags                 (the reference)
#        cand-off   the candidate, no fork flags            must equal base-off
#        cand-low   + --mtp-draft-vocab-set GF_LOW_SET      must equal cand-off
#        cand-g     + configuration G on GF_TRIM_SET        must answer every
#                   request, score head_trim_rows > 0, and return finite
#                   logprobs on a sampled request (the deferred-row path);
#                   its divergence from cand-off is reported, not gated
#      cand-low and cand-g are skipped (and say so) when their set is unset.
#   4. Report on stdout, key=value: one check_<name>=PASS|FAIL|SKIP line each,
#      out_dir. Responses, server logs and ctest output stay in the out_dir.
#
# Exit: 0 every check passed; 1 refused; 2 usage; 3 a check failed; 5 stopped
# by the memory floor or the store budget.
#
# Settings (environment): GF_CAND (ai-sdlc/v0.28.1), GF_BASE (v0.28.1),
# GF_OUT_ROOT ($HOME/.cache/gufo-fork-gpu), GF_BUILD_MEM_MIB (6144),
# GF_BUILD_CORES (4), GF_RUN_MIN_AVAIL_MIB (65536), GF_LEASE_LOCK
# (${XDG_STATE_HOME:-$HOME/.local/state}/ai-sdlc/server-measure.lock),
# GF_MODEL_DIR, GF_MODEL, GF_MTP,
# GF_TRIM_SET, GF_LOW_SET, GF_PORT (18099), GF_MAX_TOKENS (256),
# GF_SHADOW (1 adds the cand-shadow arm: the candidate with GUFO_TRIE_SHADOW set must
# reply as cand-off and write valid trie v3 shadow records; MTP item 1.32),
# GF_RUN_DOCKER_EXTRA (extra `docker run` words for the GPU containers, empty by
# default; nothing here adds --privileged). GF_DRY_RUN=1 prints the checks and
# the container commands and exits.
set -uo pipefail

GF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GF_CONF="${GF_CONF:-$GF_DIR/build.conf}"
docker_() { "${GF_DOCKER:-docker}" "$@"; }
gf_say()  { printf '%s\n' "$*" >&2; }
gf_die()  { printf 'gpu-test: %s\n' "$*" >&2; exit "${2:-1}"; }

# shellcheck disable=SC1090
[ -f "$GF_CONF" ] && . "$GF_CONF" || gf_die "no $GF_CONF" 2
for k in NIX_IMAGE NIX_VOLUME MEM_FLOOR_MIB SWAP_FLOOR_MIB BUDGET_GIB RESERVE_GIB; do
  [ -n "${!k:-}" ] || gf_die "$GF_CONF: $k is empty" 2
done

GF_CAND="${GF_CAND:-ai-sdlc/v0.28.1}"
GF_BASE="${GF_BASE:-v0.28.1}"
GF_OUT_ROOT="${GF_OUT_ROOT:-$HOME/.cache/gufo-fork-gpu}"
GF_BUILD_MEM_MIB="${GF_BUILD_MEM_MIB:-6144}"
GF_BUILD_CORES="${GF_BUILD_CORES:-4}"
GF_RUN_MIN_AVAIL_MIB="${GF_RUN_MIN_AVAIL_MIB:-65536}"
GF_LEASE_LOCK="${GF_LEASE_LOCK:-${XDG_STATE_HOME:-$HOME/.local/state}/ai-sdlc/server-measure.lock}"
GF_MODEL_DIR="${GF_MODEL_DIR:-$HOME/.cache/ai-sdlc-bench/llamacpp/unsloth__Qwen3.8-Flash-Next-GGUF}"
GF_MODEL="${GF_MODEL:-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
GF_MTP="${GF_MTP:-MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
GF_TRIM_SET="${GF_TRIM_SET:-}"
GF_LOW_SET="${GF_LOW_SET:-}"
GF_PORT="${GF_PORT:-18099}"
GF_MAX_TOKENS="${GF_MAX_TOKENS:-256}"
GF_RUN_DOCKER_EXTRA="${GF_RUN_DOCKER_EXTRA:-}"
for k in GF_BUILD_MEM_MIB GF_BUILD_CORES GF_RUN_MIN_AVAIL_MIB GF_PORT GF_MAX_TOKENS; do
  case "${!k}" in ''|*[!0-9]*) gf_die "$k must be a whole number, got '${!k}'" 2 ;; esac
done

GF_REPO="$(git -C "$GF_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" \
  || gf_die "$GF_DIR is not inside the fork's clone"
git_() { git --git-dir="$GF_REPO" "$@"; }

# The prompts: short English coding requests, fixed. One per line.
GF_PROMPTS='Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.
Explain in three sentences what a race condition is, then show a minimal C++ example that has one.
Write a bash one-liner that counts the lines of every .cpp file under src and prints the five largest.
Refactor this into idiomatic Rust: fn f(v: Vec<i32>) -> i32 { let mut s = 0; for i in 0..v.len() { s = s + v[i]; } s }
List the steps to bisect a regression with git, as a numbered list.
Write a SQL query that returns each customer with their total order value, highest first, from tables customers(id, name) and orders(id, customer_id, amount).
What does the HTTP status 409 mean, and when should an API return it instead of 400?
Write a JSON object describing a book with title, author, year and a list of two tags.'

# --- memory and the watcher (build.sh's floors and budget) -------------------
gf_mem_mib() {
  awk -v k="$1" '$1==k":"{printf "%d", $2/1024; f=1} END{if(!f) exit 1}' "${GF_MEMINFO:-/proc/meminfo}" 2>/dev/null
}
GF_PROBE='printf "%s %s\n" "$(du -sxB1 /nix | cut -f1)" "$(df -B1 --output=avail /nix | tail -1 | tr -d " ")"'
gf_watch() {  # <container> <reason file>: stops it at a floor or over the budget
  local name="$1" rf="$2" tick=0 st ma sw sf used avail why gib=$(( 1024 * 1024 * 1024 ))
  while :; do
    sleep "${GF_WATCH_S:-2}"
    st="$(docker_ inspect -f '{{.State.Running}}' "$name" 2>/dev/null)"
    case "$st" in true) ;; *) return 0 ;; esac
    why=""
    ma="$(gf_mem_mib MemAvailable)"; sw="$(gf_mem_mib SwapTotal)"; sf="$(gf_mem_mib SwapFree)"
    if [ -n "$ma" ] && [ "$ma" -lt "$MEM_FLOOR_MIB" ]; then
      why="stopped by the memory floor: MemAvailable ${ma} MiB < ${MEM_FLOOR_MIB} MiB"
    elif [ "${sw:-0}" -gt 0 ] && [ -n "$sf" ] && [ "$sf" -lt "$SWAP_FLOOR_MIB" ]; then
      why="stopped by the memory floor: SwapFree ${sf} MiB < ${SWAP_FLOOR_MIB} MiB"
    fi
    tick=$(( tick + 1 ))
    if [ -z "$why" ] && [ $(( tick % 30 )) -eq 0 ]; then
      read -r used avail <<<"$(docker_ exec "$name" sh -c "$GF_PROBE" 2>/dev/null)"
      case "${used:-x}${avail:-x}" in *[!0-9]*) ;;
        *) if [ "$used" -gt $(( BUDGET_GIB * gib )) ]; then
             why="stopped by the disk budget: the store uses $(( used / gib )) GiB > ${BUDGET_GIB} GiB"
           elif [ "$avail" -lt $(( RESERVE_GIB * gib )) ]; then
             why="stopped by the disk budget: $(( avail / gib )) GiB free < the ${RESERVE_GIB} GiB reserve"
           fi ;;
      esac
    fi
    if [ -n "$why" ]; then
      printf '%s\n' "$why" > "$rf"; gf_say "gpu-test: $why"
      docker_ stop -t 10 "$name" >/dev/null 2>&1; return 0
    fi
  done
}

# --- containers ---------------------------------------------------------------
GF_STARTED=(); GF_WPIDS=()
gf_cleanup() {
  local p c
  for p in "${GF_WPIDS[@]}"; do kill "$p" 2>/dev/null; done
  for c in "${GF_STARTED[@]}"; do docker_ rm -f "$c" >/dev/null 2>&1; done
  return 0
}
trap gf_cleanup EXIT
trap 'exit 130' INT TERM

GF_NIX_CONFIG="experimental-features = nix-command flakes
accept-flake-config = true
max-jobs = 1
cores = $GF_BUILD_CORES
sandbox = false"

gf_tree_dir() { printf '%s/%s\n' "$GF_OUT" "$1"; }   # <base|cand>

# One container, foreground, watched. <name> <memory MiB> <gpu 0|1> <tree> <script> [docker words...]
gf_container() {
  local name="$1" mem="$2" gpu="$3" tree="$4" script="$5"; shift 5
  local td; td="$(gf_tree_dir "$tree")"
  local argv=(run --name "$name" --pull never --memory "${mem}m" --memory-swap "${mem}m"
    -v "$NIX_VOLUME:/nix" -v "$td/src:/src:ro" -v "$td/build:/build" -v "$GF_OUT/logs:/logs"
    -e "NIX_CONFIG=$GF_NIX_CONFIG" -e "GF_OWNER=$(id -u):$(id -g)")
  if [ "$gpu" = 1 ]; then
    argv+=(--device /dev/kfd --device /dev/dri)
    local g; for g in $(stat -c '%g' /dev/kfd /dev/dri/card* /dev/dri/renderD* 2>/dev/null | sort -u); do argv+=(--group-add "$g"); done
    # shellcheck disable=SC2206
    [ -n "$GF_RUN_DOCKER_EXTRA" ] && argv+=($GF_RUN_DOCKER_EXTRA)
  else
    argv+=(--cpus "$GF_BUILD_CORES" --network bridge)
  fi
  argv+=("$@" "$NIX_IMAGE" bash -euo pipefail -c
    "trap 'chown -R \"\$GF_OWNER\" /build /logs 2>/dev/null || true' EXIT; $script")
  if [ "${GF_DRY_RUN:-0}" = 1 ]; then printf '  docker'; printf ' %q' "${argv[@]}"; printf '\n'; return 0; fi
  GF_STARTED+=("$name")
  docker_ rm -f "$name" >/dev/null 2>&1
  local rf="$GF_OUT/logs/$name.stopped"; rm -f "$rf"
  case " $* " in *" -d "*) docker_ "${argv[@]}" >/dev/null || return 3
                            gf_watch "$name" "$rf" & GF_WPIDS+=("$!"); return 0 ;; esac
  docker_ "${argv[@]}" > "$GF_OUT/logs/$name.log" 2>&1 & local cpid=$!
  gf_watch "$name" "$rf" & GF_WPIDS+=("$!")
  wait "$cpid"; local rc=$?
  [ -s "$rf" ] && return 5
  return "$rc"
}

# The toolchain line every container starts with: the package's patched ROCr.
GF_ENV_SCRIPT='rocr="$(nix build --no-link --print-out-paths --no-write-lock-file path:/src#default.rocmRuntime)"
export GUFO_ROCR_LIBRARY_DIR="$rocr/lib"'

# --- checks -------------------------------------------------------------------
GF_REFUSED=0
gf_check() {  # <name> <ok|REFUSED> <detail>
  printf 'check %-10s %s  %s\n' "$1" "$2" "$3" >&2
  [ "$2" = ok ] || GF_REFUSED=1
}

gf_resolve() {
  GF_CAND_SHA="$(git_ rev-parse --verify -q "$GF_CAND^{commit}")" || gf_die "$GF_CAND does not resolve" 1
  GF_BASE_SHA="$(git_ rev-parse --verify -q "$GF_BASE^{commit}")" || gf_die "$GF_BASE does not resolve" 1
  GF_OUT="$GF_OUT_ROOT/${GF_CAND_SHA:0:12}-on-${GF_BASE_SHA:0:12}"
}

gf_common_checks() {
  if git_ merge-base --is-ancestor "$GF_BASE_SHA" "$GF_CAND_SHA"; then
    gf_check ancestry ok "$GF_CAND (${GF_CAND_SHA:0:12}) descends from $GF_BASE (${GF_BASE_SHA:0:12})"
  else gf_check ancestry REFUSED "$GF_CAND does not descend from $GF_BASE"; fi
  case "$NIX_IMAGE" in *@sha256:*) ;; *) gf_check niximage REFUSED "$NIX_IMAGE is not pinned by digest"; return ;; esac
  if docker_ image inspect "$NIX_IMAGE" >/dev/null 2>&1; then gf_check niximage ok "$NIX_IMAGE"
  else gf_check niximage REFUSED "$NIX_IMAGE is not in the local store (never pulled here)"; fi
}

gf_built() {  # <tree> <sha>: 0 when the tree's build finished at that commit
  [ "$(cat "$(gf_tree_dir "$1")/built" 2>/dev/null)" = "$2" ]
}

# --- prepare ------------------------------------------------------------------
gf_prepare() {
  local ma tree sha targets td rc
  gf_common_checks
  ma="$(gf_mem_mib MemAvailable)"
  if [ -n "$ma" ] && [ "$ma" -ge $(( MEM_FLOOR_MIB + GF_BUILD_MEM_MIB )) ]; then
    gf_check memory ok "MemAvailable $ma MiB >= floor $MEM_FLOOR_MIB + cap $GF_BUILD_MEM_MIB"
  else gf_check memory REFUSED "MemAvailable ${ma:-unknown} MiB < $(( MEM_FLOOR_MIB + GF_BUILD_MEM_MIB )) MiB (floor + cap)"; fi
  [ "$GF_REFUSED" = 0 ] || gf_die "refused (the check lines above); nothing was built"
  mkdir -p "$GF_OUT/logs" || gf_die "cannot create $GF_OUT"
  for tree in base cand; do
    if [ "$tree" = base ]; then sha="$GF_BASE_SHA"; targets="gufo"
    else sha="$GF_CAND_SHA"; targets="gufo qwen38_flash_next_tests"; fi
    td="$(gf_tree_dir "$tree")"
    if gf_built "$tree" "$sha"; then gf_say "gpu-test: $tree already built at ${sha:0:12}"; echo "build_$tree=CACHED"; continue; fi
    # Extracted once per commit, so an interrupted build resumes where Ninja
    # stopped instead of seeing every source as new.
    if [ "${GF_DRY_RUN:-0}" != 1 ] && [ "$(cat "$td/src.sha" 2>/dev/null)" != "$sha" ]; then
      rm -rf "$td/src" "$td/build" "$td/built" "$td/src.sha"; mkdir -p "$td/src" "$td/build"
      git_ archive "$sha" | tar -x -C "$td/src" || gf_die "git archive of $sha failed" 3
      printf '%s\n' "$sha" > "$td/src.sha"
    fi
    gf_say "gpu-test: building $tree at ${sha:0:12} ($targets), log $GF_OUT/logs/gufo-gpu-build-$tree.log"
    gf_container "gufo-gpu-build-$tree" "$GF_BUILD_MEM_MIB" 0 "$tree" "$GF_ENV_SCRIPT
nix develop --no-write-lock-file path:/src#default -c bash -euo pipefail -c '
  cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON \
    -DENGINE_ENABLE_HIP=ON -DGUFO_BUILD_TOOLS=OFF -DGUFO_ROCR_LIBRARY_DIR=\"\$GUFO_ROCR_LIBRARY_DIR\"
  cmake --build /build --parallel $GF_BUILD_CORES --target $targets'
test -x /build/gufo"
    rc=$?
    [ "${GF_DRY_RUN:-0}" = 1 ] && continue
    case "$rc" in
      0) printf '%s\n' "$sha" > "$td/built"; echo "build_$tree=PASS" ;;
      5) echo "build_$tree=STOPPED"; gf_die "$(cat "$GF_OUT/logs/gufo-gpu-build-$tree.stopped")" 5 ;;
      *) echo "build_$tree=FAIL"; gf_die "the $tree build failed (exit $rc); see $GF_OUT/logs/gufo-gpu-build-$tree.log" 3 ;;
    esac
  done
  echo "out_dir=$GF_OUT"
}

# --- run ----------------------------------------------------------------------
gf_run_checks() {
  local ma running f
  gf_common_checks
  if gf_built base "$GF_BASE_SHA" && gf_built cand "$GF_CAND_SHA"; then gf_check prepared ok "both trees built"
  else gf_check prepared REFUSED "run \`gpu-test.sh prepare\` first"; fi
  if [ -s "$GF_LEASE_LOCK" ]; then gf_check lease ok "$(head -1 "$GF_LEASE_LOCK")"
  else gf_check lease REFUSED "no lease lock at $GF_LEASE_LOCK: take the box first"; fi
  # Any other running container with the GPU mapped is an engine (or another
  # test): docker lists images by ID, so the device is the reliable sign.
  running="$(docker_ ps -q 2>/dev/null | xargs -r "${GF_DOCKER:-docker}" inspect \
    -f '{{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' 2>/dev/null \
    | awk '/\/dev\/kfd/ && $1 !~ /^\/gufo-gpu-/ {sub(/^\//, "", $1); print $1}' | paste -sd, -)"
  if [ -z "$running" ]; then gf_check engine ok "no other container has the GPU"
  else gf_check engine REFUSED "the GPU is in use by: $running (stop the engine: take the lease)"; fi
  ma="$(gf_mem_mib MemAvailable)"
  if [ -n "$ma" ] && [ "$ma" -ge "$GF_RUN_MIN_AVAIL_MIB" ]; then gf_check memory ok "MemAvailable $ma MiB"
  else gf_check memory REFUSED "MemAvailable ${ma:-unknown} MiB < $GF_RUN_MIN_AVAIL_MIB MiB"; fi
  if [ -c /dev/kfd ]; then gf_check gpu ok "/dev/kfd"; else gf_check gpu REFUSED "no /dev/kfd"; fi
  for f in "$GF_MODEL" "$GF_MTP"; do
    if [ -f "$GF_MODEL_DIR/$f" ]; then gf_check model ok "$f"
    else gf_check model REFUSED "missing $GF_MODEL_DIR/$f"; fi
  done
  for f in "$GF_TRIM_SET" "$GF_LOW_SET"; do
    [ -z "$f" ] || [ -f "$f" ] || gf_check sets REFUSED "missing $f"
  done
}

GF_RESULTS=()
gf_result() { GF_RESULTS+=("check_$1=$2"); printf 'check_%s=%s\n' "$1" "$2"; }

gf_serve_arm() {  # <arm> <tree> [fork flags...]: serve, ask every prompt, stop
  local arm="$1" tree="$2"; shift 2
  local name="gufo-gpu-serve-$arm" mem=$(( $(gf_mem_mib MemAvailable) - MEM_FLOOR_MIB - 2048 ))
  local mounts=(-d -p "127.0.0.1:$GF_PORT:$GF_PORT" -v "$GF_MODEL_DIR:/models:ro")
  local flags=() a
  for a in "$@"; do
    case "$a" in /*) mounts+=(-v "$a:/sets/$(basename "$a"):ro"); flags+=("/sets/$(basename "$a")") ;;
                 env:*) mounts+=(-e "${a#env:}") ;;   # a container environment variable, not a flag
                 *) flags+=("$a") ;; esac
  done
  gf_container "$name" "$mem" 1 "$tree" "exec /build/gufo serve --host 0.0.0.0 --port $GF_PORT llm \
--model /models/$GF_MODEL --mtp-model /models/$GF_MTP --speculative mtp --sessions 1 --context 32768 \
--think off ${flags[*]} > /logs/$name.server.log 2>&1" "${mounts[@]}" || return 3
  [ "${GF_DRY_RUN:-0}" = 1 ] && return 0
  GF_PROMPTS="$GF_PROMPTS" python3 -I - "$GF_PORT" "$GF_OUT/results/$arm.json" "$GF_MAX_TOKENS" "$arm" <<'PY'
import json, os, sys, time, urllib.error, urllib.request
port, out, max_tokens, arm = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
base = f"http://127.0.0.1:{port}"
prompts = os.environ["GF_PROMPTS"].splitlines()
deadline = time.time() + 1800
while True:
    try:
        if urllib.request.urlopen(base + "/ready", timeout=5).status == 200: break
    except Exception: pass
    if time.time() > deadline: sys.exit("server not ready within 30 minutes")
    time.sleep(5)
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
def ask(body):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900)); return 200, r
    except urllib.error.HTTPError as e: return e.code, {"error": e.read().decode(errors="replace")}
rows = []
for p in prompts:
    status, r = ask({"model": model, "messages": [{"role": "user", "content": p}],
                     "temperature": 0, "seed": 1, "max_tokens": max_tokens})
    rows.append({"prompt": p, "status": status,
                 "content": (r.get("choices") or [{}])[0].get("message", {}).get("content"),
                 "usage": r.get("usage"), "error": r.get("error")})
sampled = None
if arm == "cand-g":
    status, r = ask({"model": model, "messages": [{"role": "user", "content": prompts[0]}],
                     "temperature": 0.7, "seed": 7, "max_tokens": 64, "logprobs": True, "top_logprobs": 2})
    sampled = {"status": status, "logprobs": (r.get("choices") or [{}])[0].get("logprobs"),
               "usage": r.get("usage"), "error": r.get("error")}
json.dump({"arm": arm, "rows": rows, "sampled": sampled}, open(out, "w"), indent=1)
PY
  local rc=$?
  docker_ logs "$name" >> "$GF_OUT/logs/$name.log" 2>&1
  docker_ rm -f "$name" >/dev/null 2>&1
  [ -s "$GF_OUT/logs/$name.stopped" ] && return 5
  return "$rc"
}

gf_compare() {  # <check> <arm a> <arm b>: PASS when every reply is 200 and byte-identical
  python3 -I - "$GF_OUT/results/$2.json" "$GF_OUT/results/$3.json" <<'PY'
import json, sys
a, b = (json.load(open(p))["rows"] for p in sys.argv[1:3])
bad = [i for i, (x, y) in enumerate(zip(a, b)) if x["status"] != 200 or y["status"] != 200 or x["content"] != y["content"]]
print(f"  {len(a) - len(bad)}/{len(a)} identical" + (f"; differ: {bad}" if bad else ""), file=sys.stderr)
sys.exit(1 if bad or len(a) != len(b) else 0)
PY
}

gf_check_g() {  # the trimmed head answers, scores rows, and keeps logprobs finite
  python3 -I - "$GF_OUT/results/cand-g.json" "$GF_OUT/results/cand-off.json" <<'PY'
import json, math, sys
g, off = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok = all(r["status"] == 200 for r in g["rows"])
rows = sum((r.get("usage") or {}).get("gufo", {}).get("head_trim_rows", 0) for r in g["rows"])
s = g["sampled"] or {}
lp = [t.get("logprob") for t in ((s.get("logprobs") or {}).get("content") or [])]
finite = s.get("status") == 200 and bool(lp) and all(isinstance(v, (int, float)) and math.isfinite(v) for v in lp)
diverged = [i for i, (x, y) in enumerate(zip(g["rows"], off["rows"])) if x["content"] != y["content"]]
print(f"  replies ok={ok} head_trim_rows={rows} sampled_logprobs_finite={finite} "
      f"diverged_from_cand_off={len(diverged)}/{len(g['rows'])} {diverged}", file=sys.stderr)
sys.exit(0 if ok and rows > 0 and finite else 1)
PY
}

gf_check_shadow() {  # trie v3's shadow log: valid records, counted
  python3 -I - "$GF_OUT/logs/trie-shadow.jsonl" <<'PY'
import json, sys
keys = {"pos", "reason", "sampled", "chain", "keep", "reached", "trie", "order", "p", "emitted", "match",
        "conf", "g_tok", "g_order", "g_p", "g_agree", "g_acc", "lk_len", "lk_tok", "lk_match"}
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except Exception as e:
    print(f"  shadow log unreadable: {e}", file=sys.stderr); sys.exit(1)
bad = [r for r in rows if set(r) != keys or r["reason"] not in ("gate", "width")]
reached = [r for r in rows if r["reached"] and r["emitted"] >= 0]
gates = [r for r in rows if r["g_tok"] >= 0]
print(f"  shadow records {len(rows)} (bad {len(bad)}): reached {len(reached)}, matched "
      f"{sum(r['match'] for r in reached)}, gate {sum(r['reason'] == 'gate' for r in rows)}; "
      f"B gate predictions {len(gates)} (agree {sum(r['g_agree'] for r in gates)}); "
      f"D lookup proposals {sum(r['lk_tok'] >= 0 for r in rows)} (matched {sum(r['lk_match'] for r in reached)})",
      file=sys.stderr)
sys.exit(0 if rows and not bad else 1)
PY
}

gf_run() {
  local rc=0 r
  gf_run_checks
  [ "$GF_REFUSED" = 0 ] || [ "${GF_DRY_RUN:-0}" = 1 ] || gf_die "refused (the check lines above); nothing was run"
  mkdir -p "$GF_OUT/logs" "$GF_OUT/results"
  local runmem=$(( $(gf_mem_mib MemAvailable) - MEM_FLOOR_MIB - 2048 ))

  gf_say "gpu-test: GPU operator tests on the candidate"
  gf_container gufo-gpu-ctest "$runmem" 1 cand "$GF_ENV_SCRIPT
nix develop --no-write-lock-file path:/src#default -c \
  ctest --test-dir /build -L gpu -R '^qwen38_flash_next\\.' --output-on-failure --timeout 900"
  r=$?; [ "$r" = 5 ] && gf_die "$(cat "$GF_OUT/logs/gufo-gpu-ctest.stopped")" 5
  [ "${GF_DRY_RUN:-0}" = 1 ] || { [ "$r" = 0 ] && gf_result gpu_ops PASS || { gf_result gpu_ops FAIL; rc=3; }; }

  local g_flags=(--vocab-trim-set "$GF_TRIM_SET" --vocab-trim-fallback-threshold 0.25
    --vocab-trim-discard-aware --vocab-trim-fallback-skip-scored)
  local arm
  for arm in base-off cand-off cand-low cand-g cand-shadow; do
    case "$arm" in
      cand-shadow) [ "${GF_SHADOW:-0}" = 1 ] || { gf_result shadow_identity SKIP; continue; } ;;
      cand-low) [ -n "$GF_LOW_SET" ] || { gf_result low_lossless SKIP; continue; } ;;
      cand-g)   [ -n "$GF_TRIM_SET" ] || { gf_result g_smoke SKIP; continue; } ;;
    esac
    gf_say "gpu-test: serve arm $arm"
    case "$arm" in
      base-off) gf_serve_arm "$arm" base ;;
      cand-off) gf_serve_arm "$arm" cand ;;
      cand-low) gf_serve_arm "$arm" cand --mtp-draft-vocab-set "$GF_LOW_SET" ;;
      cand-g)   gf_serve_arm "$arm" cand "${g_flags[@]}" ;;
      cand-shadow) rm -f "$GF_OUT/logs/trie-shadow.jsonl"
                   gf_serve_arm "$arm" cand env:GUFO_TRIE_SHADOW=/logs/trie-shadow.jsonl ;;
    esac
    r=$?; [ "$r" = 5 ] && gf_die "stopped by the memory floor during $arm" 5
    [ "$r" = 0 ] || { gf_say "gpu-test: arm $arm failed (exit $r); see $GF_OUT/logs"; rc=3; }
  done
  if [ "${GF_DRY_RUN:-0}" != 1 ]; then
    if [ -f "$GF_OUT/results/base-off.json" ] && [ -f "$GF_OUT/results/cand-off.json" ]; then
      gf_compare flags_off base-off cand-off && gf_result flags_off_identity PASS || { gf_result flags_off_identity FAIL; rc=3; }
    else gf_result flags_off_identity FAIL; rc=3; fi
    if [ -n "$GF_LOW_SET" ]; then
      [ -f "$GF_OUT/results/cand-low.json" ] && gf_compare low cand-off cand-low \
        && gf_result low_lossless PASS || { gf_result low_lossless FAIL; rc=3; }
    fi
    if [ -n "$GF_TRIM_SET" ]; then
      [ -f "$GF_OUT/results/cand-g.json" ] && gf_check_g \
        && gf_result g_smoke PASS || { gf_result g_smoke FAIL; rc=3; }
    fi
    if [ "${GF_SHADOW:-0}" = 1 ]; then
      [ -f "$GF_OUT/results/cand-shadow.json" ] && gf_compare shadow cand-off cand-shadow && gf_check_shadow \
        && gf_result shadow_identity PASS || { gf_result shadow_identity FAIL; rc=3; }
    fi
    printf '%s\n' "${GF_RESULTS[@]}" > "$GF_OUT/results/summary.txt"
  fi
  echo "out_dir=$GF_OUT"
  return "$rc"
}

gf_status() {
  local t sha
  for t in base cand; do
    [ "$t" = base ] && sha="$GF_BASE_SHA" || sha="$GF_CAND_SHA"
    if gf_built "$t" "$sha"; then echo "build_$t=done ${sha:0:12}"; else echo "build_$t=missing ${sha:0:12}"; fi
  done
  GF_DRY_RUN=1 gf_run_checks
  [ -f "$GF_OUT/results/summary.txt" ] && cat "$GF_OUT/results/summary.txt"
  echo "out_dir=$GF_OUT"
}

case "${1:-}" in
  -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//' ;;
  prepare) gf_resolve; gf_prepare ;;
  run)     gf_resolve; gf_run ;;
  status)  gf_resolve; gf_status ;;
  *) gf_die "usage: gpu-test.sh prepare|run|status (see --help)" 2 ;;
esac
