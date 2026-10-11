#!/usr/bin/env bash
# build.sh — build the gufo-runtime image from this fork's patch branch, the
# way the toolboxes release CI built the stock image, with nothing installed
# on the host. Run from the fork's MAIN clone:
#
#   cd <fork main clone> && ai-sdlc/build.sh [--compare <stock image ref>]
#   ai-sdlc/build.sh --watch <container> <reason file>   (the watcher alone)
#
# What it does (FORK.md has the why):
#   1. Host refusals, before any build container: not the fork's main clone (a
#      linked worktree, such as the tooling one, is refused); a dirty or
#      untracked tree; HEAD detached or not on ai-sdlc/$BASE_TAG; the tag not
#      at BASE_COMMIT (gufo re-points tags); HEAD not descended from
#      BASE_COMMIT; a merge in BASE_COMMIT..HEAD; NIX_IMAGE not by digest or
#      not in the local store (never pulled implicitly); MAX_JOBS x CORES over
#      16; the Nix store volume over its budget, or its filesystem short of
#      the room the budget still needs plus the reserve (one-shot du/df in the
#      Nix image); the out dir's filesystem short of the reserve; MemAvailable
#      below MEM_FLOOR_MIB + 16 GiB (the engine must be off).
#   2. One container (nixos/nix by digest, Nix 2.35.2 as the release CI): the
#      source mounted read-only, the store in the named volume NIX_VOLUME, a
#      cgroup memory cap (MemAvailable - 8 GiB, at most 96 GiB, no extra
#      swap), --cpus 16, max-jobs/cores from build.conf, and the Nix sandbox
#      OFF: an unprivileged container cannot sandbox, and the container is
#      never given --privileged or extra capabilities. The store paths are the
#      same (input-addressed), but the bytes are not: an unsandboxed build
#      runs in a random /builds/nix-<n>-<rand>/ directory instead of the
#      sandbox's /build, and gufo embeds that path (assert strings, in the
#      host code and the HIP fatbin). So a build NEVER bit-reproduces the
#      stock image, and two of these builds differ from each other. Inside: a
#      bare clone of the patch branch, toolboxes at TOOLBOXES_COMMIT exported
#      with `git archive` as a path: flake, the drv evaluated first (EXPECTED_DRV
#      set and different: stop before building), then
#      `nix build --override-input gufo-engine git+file:///work/gufo.git?ref=…&rev=<HEAD>`.
#   3. A host-side watcher stops the container at MemAvailable < MEM_FLOOR_MIB,
#      SwapFree < SWAP_FLOOR_MIB (when there is swap), or the store over
#      BUDGET_GIB / its filesystem under RESERVE_GIB (checked every 60 s).
#   4. docker load, retag to $IMAGE_REPO-unlabelled:<HEAD12> (the loaded name
#      is the stock tag's name: what it held before the load is recorded, and
#      afterwards it is removed if the load created it or pointed back if the
#      load moved it), then a label-only docker build (no new layer) to
#      $IMAGE_REPO:<HEAD12> with the org.ai-sdlc.gufo-fork.* labels.
#   5. Report on stdout, key=value: drv sandbox engine_path image image_id
#      diff_ids_sha256 build_s peak_mem_bytes store_used_bytes out_dir; with
#      --compare also compare_* lines and compare=REPRODUCES|DIFFERS: <which>.
#
# Exit: 0 built (and reproduces, with --compare); 1 refused by a host check;
# 2 usage; 3 the build failed; 4 the drv differs from EXPECTED_DRV; 5 stopped
# by the memory floor or the disk budget; 6 built, but --compare DIFFERS.
#
# Seams (for a selftest; unset in real use): GF_DRY_RUN=1 prints every host
# check's verdict and the exact build `docker run` argv, then exits (0 when all
# checks pass, 1 otherwise); GF_DOCKER the docker CLI; GF_MEMINFO a meminfo
# file; GF_DF the df used on the out dir; GF_CONF the config file;
# GF_WATCH_S the watcher's tick (2 s).
set -uo pipefail

GF_TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GF_CONF="${GF_CONF:-$GF_TOOL_DIR/build.conf}"

gf_say()  { printf '%s\n' "$*" >&2; }
gf_die()  { printf 'build: %s\n' "$*" >&2; exit "${2:-1}"; }
docker_() { "${GF_DOCKER:-docker}" "$@"; }

# --- build.conf, as text --------------------------------------------------
GF_KEYS="BASE_TAG BASE_COMMIT TOOLBOXES_REPO TOOLBOXES_COMMIT TOOLBOXES_ATTR NIX_IMAGE NIX_VOLUME BUDGET_GIB RESERVE_GIB MAX_JOBS CORES IMAGE_REPO MEM_FLOOR_MIB SWAP_FLOOR_MIB EXPECTED_DRV"
gf_load_conf() {
  local env_drv="${EXPECTED_DRV:-}" line k v
  [ -f "$GF_CONF" ] || gf_die "no $GF_CONF" 2
  for k in $GF_KEYS; do unset "$k"; done
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; *=*) ;; *) continue ;; esac
    k="${line%%=*}"; v="${line#*=}"
    case " $GF_KEYS " in *" $k "*) ;; *) continue ;; esac
    v="${v%%[[:space:]]#*}"; v="${v%"${v##*[![:space:]]}"}"
    printf -v "$k" '%s' "$v"
  done < "$GF_CONF"
  [ -n "$env_drv" ] && EXPECTED_DRV="$env_drv"
  for k in $GF_KEYS; do
    [ "$k" = EXPECTED_DRV ] && continue
    [ -n "${!k:-}" ] || gf_die "$GF_CONF: $k is empty" 2
  done
  for k in BUDGET_GIB RESERVE_GIB MAX_JOBS CORES MEM_FLOOR_MIB SWAP_FLOOR_MIB; do
    case "${!k}" in *[!0-9]*) gf_die "$GF_CONF: $k must be a whole number, got '${!k}'" 2 ;; esac
  done
  : "${EXPECTED_DRV:=}"
}

gf_mem_mib() {  # <Key>: from GF_MEMINFO (default /proc/meminfo), in MiB
  awk -v k="$1" '$1==k":"{printf "%d", $2/1024; f=1} END{if(!f) exit 1}' "${GF_MEMINFO:-/proc/meminfo}" 2>/dev/null
}

# The store probe: "<used bytes> <avail bytes>" of the volume's /nix. With a
# container name, it runs inside that container (the watcher); otherwise a
# one-shot container mounts the volume (the pre-check).
GF_PROBE='printf "%s %s\n" "$(du -sxB1 /nix | cut -f1)" "$(df -B1 --output=avail /nix | tail -1 | tr -d " ")"'
gf_store_probe() {
  if [ -n "${1:-}" ]; then docker_ exec "$1" sh -c "$GF_PROBE" 2>/dev/null
  else docker_ run --rm --pull never --network none -v "$NIX_VOLUME:/nix" "$NIX_IMAGE" sh -c "$GF_PROBE" 2>/dev/null; fi
}

# --- the watcher ------------------------------------------------------------
# Runs until the container has stopped. Writes the reason to <reason file>
# and stops the container at a floor or over the budget.
gf_watch() {  # <container> <reason file>
  local name="$1" rf="$2" tick=0 st ma sw sf probe used avail why
  local gib=$(( 1024 * 1024 * 1024 ))
  while :; do
    sleep "${GF_WATCH_S:-2}"
    st="$(docker_ inspect -f '{{.State.Running}}' "$name" 2>/dev/null)"
    case "$st" in true) ;; false) return 0 ;; *) continue ;; esac
    why=""
    ma="$(gf_mem_mib MemAvailable)"; sw="$(gf_mem_mib SwapTotal)"; sf="$(gf_mem_mib SwapFree)"
    if [ -n "$ma" ] && [ "$ma" -lt "$MEM_FLOOR_MIB" ]; then
      why="stopped by the memory floor: MemAvailable ${ma} MiB < ${MEM_FLOOR_MIB} MiB"
    elif [ "${sw:-0}" -gt 0 ] && [ -n "$sf" ] && [ "$sf" -lt "$SWAP_FLOOR_MIB" ]; then
      why="stopped by the memory floor: SwapFree ${sf} MiB < ${SWAP_FLOOR_MIB} MiB"
    fi
    tick=$(( tick + 1 ))
    if [ -z "$why" ] && [ $(( tick % 30 )) -eq 0 ]; then
      probe="$(gf_store_probe "$name")"; read -r used avail <<<"$probe"
      case "$used$avail" in ''|*[!0-9]*) ;;
        *) if [ "$used" -gt $(( BUDGET_GIB * gib )) ]; then
             why="stopped by the disk budget: the store uses $(( used / gib )) GiB > ${BUDGET_GIB} GiB"
           elif [ "$avail" -lt $(( RESERVE_GIB * gib )) ]; then
             why="stopped by the disk budget: $(( avail / gib )) GiB free < the ${RESERVE_GIB} GiB reserve"
           fi ;;
      esac
    fi
    if [ -n "$why" ]; then
      printf '%s\n' "$why" > "$rf"
      gf_say "build: $why"
      docker_ stop -t 10 "$name" >/dev/null 2>&1
      return 0
    fi
  done
}

# --- host checks ------------------------------------------------------------
GF_REFUSED=0
gf_check() {  # <name> <ok|REFUSED> <detail>
  [ "$2" = ok ] || GF_REFUSED=1
  if [ "${GF_DRY_RUN:-0}" = 1 ] || [ "$2" != ok ]; then printf 'check %-10s %s  %s\n' "$1" "$2" "$3"; fi
}

gf_host_checks() {
  local top gd cdir br tagc merges avail_mib probe used avail need dfa dir
  local gib=$(( 1024 * 1024 * 1024 ))
  top="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || { gf_check repo REFUSED "$PWD is not a git repository (run from the fork's main clone)"; return; }
  GF_SRC="$top"
  gd="$(git -C "$top" rev-parse --absolute-git-dir 2>/dev/null)"
  cdir="$(git -C "$top" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
  if [ "$gd" != "$cdir" ] || [ "$gd" != "$top/.git" ]; then
    gf_check repo REFUSED "$top is a linked worktree (its git dir is outside it): run from the fork's main clone"
  else gf_check repo ok "$top"; fi
  if [ -n "$(git -C "$top" status --porcelain 2>/dev/null)" ]; then
    gf_check clean REFUSED "the tree is dirty or has untracked files (git status --porcelain)"
  else gf_check clean ok "no changes, no untracked files"; fi
  br="$(git -C "$top" symbolic-ref -q --short HEAD 2>/dev/null)"
  GF_HEAD="$(git -C "$top" rev-parse HEAD 2>/dev/null)"
  if [ -z "$br" ]; then gf_check branch REFUSED "HEAD is detached (check out ai-sdlc/$BASE_TAG)"
  elif [ "$br" != "ai-sdlc/$BASE_TAG" ]; then gf_check branch REFUSED "on '$br', not ai-sdlc/$BASE_TAG"
  else gf_check branch ok "$br at ${GF_HEAD:0:12}"; fi
  tagc="$(git -C "$top" rev-parse -q --verify "refs/tags/$BASE_TAG^{commit}" 2>/dev/null)"
  if [ "$tagc" != "$BASE_COMMIT" ]; then gf_check tag REFUSED "$BASE_TAG is at '${tagc:-absent}', build.conf pins $BASE_COMMIT (a re-pointed tag)"
  else gf_check tag ok "$BASE_TAG = ${BASE_COMMIT:0:12}"; fi
  if ! git -C "$top" merge-base --is-ancestor "$BASE_COMMIT" HEAD 2>/dev/null; then
    gf_check descends REFUSED "HEAD is not descended from $BASE_COMMIT"
  else
    merges="$(git -C "$top" rev-list --merges "$BASE_COMMIT..HEAD" 2>/dev/null | grep -c .)"
    if [ "$merges" -gt 0 ]; then gf_check descends REFUSED "$merges merge commit(s) in $BASE_TAG..HEAD (patches are a linear series)"
    else gf_check descends ok "$(git -C "$top" rev-list --count "$BASE_COMMIT..HEAD") patch(es) on $BASE_TAG"; fi
  fi
  case "$NIX_IMAGE" in
    *@sha256:*) if docker_ image inspect "$NIX_IMAGE" >/dev/null 2>&1; then gf_check niximage ok "$NIX_IMAGE"
                else gf_check niximage REFUSED "$NIX_IMAGE is not in the local store (docker pull it by digest once; the build never pulls)"; fi ;;
    *) gf_check niximage REFUSED "NIX_IMAGE '$NIX_IMAGE' is not pinned by digest (@sha256:)" ;;
  esac
  if [ $(( MAX_JOBS * CORES )) -gt 16 ]; then gf_check jobs REFUSED "MAX_JOBS x CORES = $(( MAX_JOBS * CORES )) > 16"
  else gf_check jobs ok "max-jobs $MAX_JOBS x cores $CORES = $(( MAX_JOBS * CORES ))"; fi
  if [ "$GF_REFUSED" = 0 ]; then
    probe="$(gf_store_probe)"; read -r used avail <<<"$probe"
    case "$used$avail" in
      ''|*[!0-9]*) gf_check disk REFUSED "the store probe in $NIX_IMAGE returned '${probe:-nothing}'" ;;
      *) need=$(( BUDGET_GIB * gib - used + RESERVE_GIB * gib ))
         if [ "$used" -ge $(( BUDGET_GIB * gib )) ]; then
           gf_check disk REFUSED "the store volume $NIX_VOLUME uses $(( used / gib )) GiB, at or over the ${BUDGET_GIB} GiB budget"
         elif [ "$avail" -lt "$need" ]; then
           gf_check disk REFUSED "$(( avail / gib )) GiB free; the budget still needs $(( (BUDGET_GIB * gib - used) / gib )) GiB plus the ${RESERVE_GIB} GiB reserve"
         else gf_check disk ok "store $(( used / gib )) GiB of ${BUDGET_GIB}; $(( avail / gib )) GiB free"; fi ;;
    esac
  fi
  dir="$GF_OUT"; while [ ! -d "$dir" ]; do dir="$(dirname "$dir")"; done
  dfa="$("${GF_DF:-df}" -B1 --output=avail "$dir" 2>/dev/null | tail -1 | tr -d ' ')"
  case "$dfa" in
    ''|*[!0-9]*) gf_check outdir REFUSED "cannot read the free space under $dir" ;;
    *) if [ "$dfa" -lt $(( RESERVE_GIB * gib )) ]; then gf_check outdir REFUSED "$(( dfa / gib )) GiB free under $dir, below the ${RESERVE_GIB} GiB reserve (the image tarball lands there)"
       else gf_check outdir ok "$GF_OUT ($(( dfa / gib )) GiB free)"; fi ;;
  esac
  avail_mib="$(gf_mem_mib MemAvailable)"
  if [ -z "$avail_mib" ] || [ "$avail_mib" -lt $(( MEM_FLOOR_MIB + 16384 )) ]; then
    gf_check memory REFUSED "MemAvailable ${avail_mib:-unknown} MiB < $(( MEM_FLOOR_MIB + 16384 )) MiB: the engine must be off for a build (take the serving lease with the engine off first)"
  else gf_check memory ok "MemAvailable $avail_mib MiB"; fi
  GF_MEM_AVAIL_MIB="${avail_mib:-0}"
}

# --- the build container ------------------------------------------------------
# Expanded inside the container, never here (a quoted heredoc).
GF_INNER="$(cat <<'INNER'
trap 'cat /sys/fs/cgroup/memory.peak > /out/memory.peak 2>/dev/null || true; chown -R "$GF_OWNER" /out 2>/dev/null || true' EXIT
git config --global safe.directory "*"
git clone -q --bare --no-local --single-branch --branch "$GF_BRANCH" /src/gufo /work/gufo.git
got="$(git -C /work/gufo.git rev-parse "refs/heads/$GF_BRANCH")"
[ "$got" = "$GF_HEAD" ] || { echo "inner: the clone has $GF_BRANCH at $got, not $GF_HEAD" >&2; exit 3; }
git init -q --bare /work/tb.git
git -C /work/tb.git fetch -q --depth 1 "$TOOLBOXES_REPO" "$TOOLBOXES_COMMIT"
got="$(git -C /work/tb.git rev-parse FETCH_HEAD)"
[ "$got" = "$TOOLBOXES_COMMIT" ] || { echo "inner: toolboxes FETCH_HEAD is $got, not $TOOLBOXES_COMMIT" >&2; exit 3; }
mkdir -p /work/toolboxes
git -C /work/tb.git archive "$TOOLBOXES_COMMIT" | tar -x -C /work/toolboxes
drv="$(nix eval --raw --no-write-lock-file --override-input gufo-engine "$GF_REF" "path:/work/toolboxes#$TOOLBOXES_ATTR.drvPath")"
printf "%s\n" "$drv" > /out/drv
echo "inner: drv $drv" >&2
if [ -n "$EXPECTED_DRV" ] && [ "$drv" != "$EXPECTED_DRV" ]; then
  echo "inner: drv differs from stock — patched or drifted (expected $EXPECTED_DRV)" >&2; exit 4
fi
start="$(date +%s)"
nix build -L --no-write-lock-file --override-input gufo-engine "$GF_REF" "path:/work/toolboxes#$TOOLBOXES_ATTR" --out-link /work/result
echo $(( $(date +%s) - start )) > /out/build_s
cp -L /work/result /out/image.tar.gz
nix-store -q --requisites "$drv" | grep -E "^/nix/store/[a-z0-9]{32}-gufo-[0-9][0-9.]*[.]drv$" | xargs -r nix-store -q --outputs | sort -u > /out/engine-path
{ printf "result_hash=%s\n" "$(nix-store -q --hash "$(readlink -f /work/result)")"
  printf "store_used_bytes=%s\n" "$(du -sxB1 /nix | cut -f1)"; } > /out/resources
INNER
)"

gf_argv() {  # fills GF_ARGV: the build container's docker argv
  local mem_mib=$(( GF_MEM_AVAIL_MIB - 8192 ))
  [ "$mem_mib" -gt 98304 ] && mem_mib=98304
  GF_REF="git+file:///work/gufo.git?ref=refs/heads/ai-sdlc/$BASE_TAG&rev=$GF_HEAD"
  GF_ARGV=(run --name "$GF_NAME" --pull never --network bridge --cpus 16
    --memory "${mem_mib}m" --memory-swap "${mem_mib}m"
    -v "$GF_SRC:/src/gufo:ro" -v "$NIX_VOLUME:/nix" -v "$GF_OUT:/out"
    -e "NIX_CONFIG=experimental-features = nix-command flakes
accept-flake-config = true
max-jobs = $MAX_JOBS
cores = $CORES
sandbox = false"
    -e "GF_BRANCH=ai-sdlc/$BASE_TAG" -e "GF_HEAD=$GF_HEAD" -e "GF_REF=$GF_REF"
    -e "TOOLBOXES_REPO=$TOOLBOXES_REPO" -e "TOOLBOXES_COMMIT=$TOOLBOXES_COMMIT"
    -e "TOOLBOXES_ATTR=$TOOLBOXES_ATTR" -e "EXPECTED_DRV=$EXPECTED_DRV"
    -e "GF_OWNER=$(id -u):$(id -g)"
    "$NIX_IMAGE" bash -euo pipefail -c "$GF_INNER")
}

# --- compare against stock ------------------------------------------------------
gf_cfg() { docker_ image inspect -f "$2" "$1" 2>/dev/null; }
gf_labels() {  # <image>: sorted k=v, the fork's own labels left out
  gf_cfg "$1" '{{range $k, $v := .Config.Labels}}{{$k}}={{$v}}{{"\n"}}{{end}}' | grep -v '^org\.ai-sdlc\.gufo-fork\.' | grep . | sort
}
gf_compare() {  # <stock ref> <fork ref>
  local s="$1" f="$2" diff="" a b fld
  a="$(docker_ run --rm --pull never --network none --entrypoint /bin/readlink "$s" -f /bin/gufo 2>/dev/null)"
  b="$(docker_ run --rm --pull never --network none --entrypoint /bin/readlink "$f" -f /bin/gufo 2>/dev/null)"
  printf 'compare_engine_path_stock=%s\ncompare_engine_path_fork=%s\n' "$a" "$b"
  { [ -n "$a" ] && [ "$a" = "$b" ]; } || diff="$diff engine-path"
  a="$(gf_cfg "$s" '{{json .RootFS.Layers}}')"; b="$(gf_cfg "$f" '{{json .RootFS.Layers}}')"
  printf 'compare_layers_stock=%s\ncompare_layers_fork=%s\n' "$(printf '%s' "$a" | grep -o 'sha256:' | grep -c .)" "$(printf '%s' "$b" | grep -o 'sha256:' | grep -c .)"
  { [ -n "$a" ] && [ "$a" = "$b" ]; } || diff="$diff diff_ids"
  for fld in Env Cmd User WorkingDir Entrypoint; do
    a="$(gf_cfg "$s" "{{json .Config.$fld}}")"; b="$(gf_cfg "$f" "{{json .Config.$fld}}")"
    [ "$a" = "$b" ] || diff="$diff config.$fld"
  done
  a="$(gf_labels "$s")"; b="$(gf_labels "$f")"
  { [ -n "$a" ] && [ "$a" = "$b" ]; } || diff="$diff labels"
  a="$(docker_ run --rm --pull never --network none --entrypoint /bin/gufo "$s" --version 2>&1)"
  b="$(docker_ run --rm --pull never --network none --entrypoint /bin/gufo "$f" --version 2>&1)"
  printf 'compare_version_stock=%s\ncompare_version_fork=%s\n' "$(printf '%s' "$a" | head -1)" "$(printf '%s' "$b" | head -1)"
  { [ -n "$a" ] && [ "$a" = "$b" ]; } || diff="$diff version"
  if [ -z "$diff" ]; then printf 'compare=REPRODUCES\n'; return 0; fi
  printf 'compare=DIFFERS:%s\n' "$diff"; return 6
}

# --- main -----------------------------------------------------------------------
GF_WPID=""; GF_CNAME=""
gf_cleanup() {
  [ -n "$GF_WPID" ] && kill "$GF_WPID" 2>/dev/null
  [ -n "$GF_CNAME" ] && docker_ rm -f "$GF_CNAME" >/dev/null 2>&1
  return 0
}

main() {
  local compare="" rc loaded lid id h12 why
  case "${1:-}" in
    --watch) [ $# -eq 3 ] || gf_die "usage: build.sh --watch <container> <reason file>" 2
             gf_load_conf; gf_watch "$2" "$3"; return 0 ;;
    --compare) [ -n "${2:-}" ] || gf_die "usage: build.sh [--compare <stock image ref>]" 2; compare="$2" ;;
    -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; return 0 ;;
    '') ;;
    *) gf_die "usage: build.sh [--compare <stock image ref>] (unknown '$1')" 2 ;;
  esac
  gf_load_conf
  GF_HEAD="$(git rev-parse HEAD 2>/dev/null)"; h12="${GF_HEAD:0:12}"
  GF_OUT="${XDG_CACHE_HOME:-$HOME/.cache}/gufo-fork-build/${h12:-nohead}"
  GF_NAME="gufo-fork-build-${h12:-nohead}"
  GF_SRC="$PWD"; GF_MEM_AVAIL_MIB=0
  gf_host_checks
  if [ "${GF_DRY_RUN:-0}" = 1 ]; then
    gf_argv
    printf 'expected_drv=%s\nsandbox=off\n' "${EXPECTED_DRV:-<unset>}"
    printf 'docker_run:\n'; printf '  %s\n' "$GF_DOCKER_BIN" "${GF_ARGV[@]}"
    return "$GF_REFUSED"
  fi
  [ "$GF_REFUSED" = 0 ] || gf_die "refused (the check lines above); nothing was run"
  gf_argv
  mkdir -p "$GF_OUT" || gf_die "cannot create $GF_OUT"
  rm -f "$GF_OUT/stopped" "$GF_OUT/image.tar.gz"
  docker_ rm -f "$GF_NAME" >/dev/null 2>&1
  trap gf_cleanup EXIT; trap 'exit 130' INT TERM
  GF_CNAME="$GF_NAME"
  gf_watch "$GF_NAME" "$GF_OUT/stopped" & GF_WPID=$!
  gf_say "build: $GF_NAME building ${h12} (log: $GF_OUT/build.log)"
  docker_ "${GF_ARGV[@]}" 2>&1 | tee "$GF_OUT/build.log" >&2; rc="${PIPESTATUS[0]}"
  kill "$GF_WPID" 2>/dev/null; wait "$GF_WPID" 2>/dev/null; GF_WPID=""
  docker_ rm -f "$GF_NAME" >/dev/null 2>&1; GF_CNAME=""
  printf 'drv=%s\nsandbox=off\n' "$(cat "$GF_OUT/drv" 2>/dev/null)"
  if [ -s "$GF_OUT/stopped" ]; then why="$(cat "$GF_OUT/stopped")"; gf_die "$why" 5; fi
  [ "$rc" = 4 ] && gf_die "drv differs from stock — patched or drifted (expected ${EXPECTED_DRV}); nothing was built" 4
  [ "$rc" = 0 ] && [ -s "$GF_OUT/image.tar.gz" ] || gf_die "the build container exited $rc (see $GF_OUT/build.log)" 3
  # The archive names the image as stock does (the Nix expression's tag), so
  # loading it moves that name. Record what the name held BEFORE the load, then
  # afterwards remove the name if the load created it, or point it back at its
  # old image if the load moved it. Stock is pinned by digest either way.
  name="$(tar -xzOf "$GF_OUT/image.tar.gz" manifest.json 2>/dev/null | sed -n 's/.*"RepoTags":\["\([^"]*\)".*/\1/p' | head -1)"
  pre=""; [ -n "$name" ] && pre="$(gf_cfg "$name" '{{.Id}}')"
  loaded="$(docker_ load -i "$GF_OUT/image.tar.gz" | sed -n 's/^Loaded image: //p' | tail -1)"
  lid="$(gf_cfg "$loaded" '{{.Id}}')"
  [ -n "$lid" ] || gf_die "docker load did not report an image (loaded '$loaded')" 3
  docker_ tag "$loaded" "$IMAGE_REPO-unlabelled:$h12" || gf_die "cannot tag $loaded" 3
  if [ -n "$name" ] && [ "$loaded" = "$name" ]; then
    if [ -z "$pre" ]; then docker_ rmi "$loaded" >/dev/null 2>&1
    elif [ "$pre" != "$lid" ]; then docker_ tag "$pre" "$loaded" >/dev/null 2>&1 || gf_say "build: could not point $loaded back at $pre"
    fi
  else
    gf_say "build: left $loaded as loaded (its name was not read from the archive beforehand)"
  fi
  printf 'FROM %s-unlabelled:%s\nLABEL org.ai-sdlc.gufo-fork.commit=%s org.ai-sdlc.gufo-fork.branch=%s org.ai-sdlc.gufo-fork.base-tag=%s org.ai-sdlc.gufo-fork.base-commit=%s org.ai-sdlc.gufo-fork.toolboxes-commit=%s\n' \
    "$IMAGE_REPO" "$h12" "$GF_HEAD" "ai-sdlc/$BASE_TAG" "$BASE_TAG" "$BASE_COMMIT" "$TOOLBOXES_COMMIT" \
    | docker_ build -q --pull=false -t "$IMAGE_REPO:$h12" - >/dev/null || gf_die "the label step failed" 3
  id="$(gf_cfg "$IMAGE_REPO:$h12" '{{.Id}}')"
  [[ "$id" =~ ^sha256:[0-9a-f]{64}$ ]] || gf_die "the labelled image's id is not sha256:<64 hex>: '$id'" 3
  printf 'engine_path=%s\n' "$(tr '\n' ' ' < "$GF_OUT/engine-path" 2>/dev/null | sed 's/ $//')"
  printf 'image=%s@%s\nimage_id=%s\n' "$IMAGE_REPO" "$id" "$id"
  printf 'diff_ids_sha256=%s\n' "$(gf_cfg "$IMAGE_REPO:$h12" '{{json .RootFS.Layers}}' | sha256sum | cut -c1-64)"
  printf 'build_s=%s\npeak_mem_bytes=%s\n' "$(cat "$GF_OUT/build_s" 2>/dev/null)" "$(cat "$GF_OUT/memory.peak" 2>/dev/null)"
  sed -n 's/^store_used_bytes=/store_used_bytes=/p' "$GF_OUT/resources" 2>/dev/null
  printf 'out_dir=%s\n' "$GF_OUT"
  if [ -n "$compare" ]; then gf_compare "$compare" "$IMAGE_REPO:$h12"; return $?; fi
  return 0
}

GF_DOCKER_BIN="${GF_DOCKER:-docker}"
main "$@"
