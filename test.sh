#!/usr/bin/env bash
# test.sh — run the fork's hosted checks on the checked-out patch branch inside
# the same container image build.sh builds in, with nothing installed on the
# host. Run from the fork's main clone:
#
#   cd <fork main clone> && ai-sdlc/test.sh [check ...]
#
# The checks are the upstream flake's own checks, so they are exactly what its
# hosted CI runs: `format` (clang-format), `pr-tests` (the CPU contract suite,
# `check-pr`), `docs`, `dependency-inventory`. `cache-tests` is not a flake
# check: it builds the flake's CPU configuration the same way and runs the
# continuation-cache test, which `check-pr` leaves to the full suite. With no
# argument all five run.
#
# What it does (FORK.md has the why):
#   1. Host refusals, before any container: not the fork's main clone; a dirty
#      or untracked tree; HEAD detached or not on ai-sdlc/$BASE_TAG; NIX_IMAGE
#      not by digest or not in the local store (never pulled); MemAvailable
#      below MEM_FLOOR_MIB + GF_TEST_MEM_MIB (the engine may be running: the test
#      container is capped, but it must never push the host under the floor).
#   2. One container (nixos/nix by digest, the same store volume as the build):
#      the source mounted read-only, a cgroup memory cap of GF_TEST_MEM_MIB with
#      no extra swap, --cpus GF_TEST_CORES, one Nix job of GF_TEST_CORES cores, the
#      Nix sandbox OFF as in build.sh (an unprivileged container cannot sandbox
#      and is never given --privileged). Inside: a bare clone of the patch
#      branch, then `nix build -L --no-link <clone>#checks.x86_64-linux.<check>`
#      for each check, in order, continuing past a failure.
#   3. The build.sh watcher stops the container at MemAvailable < MEM_FLOOR_MIB,
#      SwapFree < SWAP_FLOOR_MIB, or the store over BUDGET_GIB / its filesystem
#      under RESERVE_GIB.
#   4. Report on stdout, key=value: head, one `check_<name>=PASS|FAIL` line per
#      check, peak_mem_bytes, test_s, out_dir.
#
# Exit: 0 every check passed; 1 refused by a host check; 2 usage; 3 a check
# failed; 5 stopped by the memory floor or the disk budget.
#
# Seams (for a selftest; unset in real use): GF_DRY_RUN=1 prints every host
# check's verdict and the exact `docker run` argv, then exits (0 when all
# checks pass, 1 otherwise); GF_DOCKER the docker CLI; GF_MEMINFO a meminfo
# file; GF_CONF the config file. GF_TEST_MEM_MIB (8192) and GF_TEST_CORES (4) size
# the container.
set -uo pipefail

GF_TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GF_CONF="${GF_CONF:-$GF_TOOL_DIR/build.conf}"
GF_ALL_CHECKS="format pr-tests cache-tests docs dependency-inventory"

gf_say()  { printf '%s\n' "$*" >&2; }
gf_die()  { printf 'test: %s\n' "$*" >&2; exit "${2:-1}"; }
docker_() { "${GF_DOCKER:-docker}" "$@"; }

# --- build.conf, as text (the keys this script needs; others are ignored) ----
GF_KEYS="BASE_TAG NIX_IMAGE NIX_VOLUME BUDGET_GIB RESERVE_GIB MEM_FLOOR_MIB SWAP_FLOOR_MIB"
gf_load_conf() {
  local line k v
  [ -f "$GF_CONF" ] || gf_die "no $GF_CONF" 2
  for k in $GF_KEYS; do unset "$k"; done
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; *=*) ;; *) continue ;; esac
    k="${line%%=*}"; v="${line#*=}"
    case " $GF_KEYS " in *" $k "*) ;; *) continue ;; esac
    v="${v%%[[:space:]]#*}"; v="${v%"${v##*[![:space:]]}"}"
    printf -v "$k" '%s' "$v"
  done < "$GF_CONF"
  for k in $GF_KEYS; do [ -n "${!k:-}" ] || gf_die "$GF_CONF: $k is empty" 2; done
  for k in BUDGET_GIB RESERVE_GIB MEM_FLOOR_MIB SWAP_FLOOR_MIB; do
    case "${!k}" in *[!0-9]*) gf_die "$GF_CONF: $k must be a whole number, got '${!k}'" 2 ;; esac
  done
  # The test container's own caps: small, because the engine may be running.
  # Overridable from the environment, never from build.conf (which build.sh and
  # the lock hash).
  GF_TEST_MEM_MIB="${GF_TEST_MEM_MIB:-8192}"; GF_TEST_CORES="${GF_TEST_CORES:-4}"
  for k in GF_TEST_MEM_MIB GF_TEST_CORES; do
    case "${!k}" in ''|*[!0-9]*) gf_die "$k must be a whole number, got '${!k}'" 2 ;; esac
  done
}

gf_mem_mib() {  # <Key>: from GF_MEMINFO (default /proc/meminfo), in MiB
  awk -v k="$1" '$1==k":"{printf "%d", $2/1024; f=1} END{if(!f) exit 1}' "${GF_MEMINFO:-/proc/meminfo}" 2>/dev/null
}

# --- host checks ------------------------------------------------------------
GF_REFUSED=0
gf_check() {  # <name> <ok|REFUSED> <detail>
  [ "$2" = ok ] || GF_REFUSED=1
  if [ "${GF_DRY_RUN:-0}" = 1 ] || [ "$2" != ok ]; then printf 'check %-10s %s  %s\n' "$1" "$2" "$3"; fi
}

gf_host_checks() {
  local top gd cdir br avail_mib
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
  case "$NIX_IMAGE" in
    *@sha256:*) if docker_ image inspect "$NIX_IMAGE" >/dev/null 2>&1; then gf_check niximage ok "$NIX_IMAGE"
                else gf_check niximage REFUSED "$NIX_IMAGE is not in the local store (docker pull it by digest once; the test never pulls)"; fi ;;
    *) gf_check niximage REFUSED "NIX_IMAGE '$NIX_IMAGE' is not pinned by digest (@sha256:)" ;;
  esac
  avail_mib="$(gf_mem_mib MemAvailable)"
  if [ -z "$avail_mib" ] || [ "$avail_mib" -lt $(( MEM_FLOOR_MIB + GF_TEST_MEM_MIB )) ]; then
    gf_check memory REFUSED "MemAvailable ${avail_mib:-unknown} MiB < $(( MEM_FLOOR_MIB + GF_TEST_MEM_MIB )) MiB (floor $MEM_FLOOR_MIB + cap $GF_TEST_MEM_MIB)"
  else gf_check memory ok "MemAvailable $avail_mib MiB"; fi
}

# --- the test container -------------------------------------------------------
# Expanded inside the container, never here (a quoted heredoc).
GF_INNER="$(cat <<'INNER'
trap 'cat /sys/fs/cgroup/memory.peak > /out/memory.peak 2>/dev/null || true; chown -R "$GF_OWNER" /out 2>/dev/null || true' EXIT
git config --global safe.directory "*"
git clone -q --bare --no-local --single-branch --branch "$GF_BRANCH" /src/gufo /work/gufo.git
got="$(git -C /work/gufo.git rev-parse "refs/heads/$GF_BRANCH")"
[ "$got" = "$GF_HEAD" ] || { echo "inner: the clone has $GF_BRANCH at $got, not $GF_HEAD" >&2; exit 3; }
start="$(date +%s)"
: > /out/results
rc=0
cache_tests_expr() {
  cat <<NIX
let
  f = builtins.getFlake "$GF_REF";
  pkgs = f.inputs.nixpkgs.legacyPackages.x86_64-linux;
in
pkgs.runCommand "check-cache-tests" {
  nativeBuildInputs = [ pkgs.stdenv.cc pkgs.cmake pkgs.ninja pkgs.pkg-config
    pkgs.python3 pkgs.icu pkgs.curl pkgs.libpng pkgs.libjpeg pkgs.libwebp
    pkgs.openssl ];
  src = f.outPath;
} ''
  export HOME=\$TMPDIR
  set -euo pipefail
  mkdir -p build && cd build
  cmake "\$src" -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DENGINE_ENABLE_HIP=OFF
  cmake --build . --parallel "\$NIX_BUILD_CORES" --target continuation_cache_test
  ctest --output-on-failure --timeout 120 -R '^continuation_cache_test\$'
  mkdir -p \$out
  echo "PASS: continuation cache test" > \$out/result.txt
''
NIX
}
build_check() {
  if [ "$1" = cache-tests ]; then
    nix build -L --no-link --impure --expr "$(cache_tests_expr)"
  else
    nix build -L --no-link --no-write-lock-file "$GF_REF#checks.x86_64-linux.$1"
  fi
}
for c in $GF_CHECKS; do
  echo "inner: check $c" >&2
  if build_check "$c"; then
    echo "check_$c=PASS" >> /out/results
  else
    echo "check_$c=FAIL" >> /out/results
    rc=3
  fi
done
echo $(( $(date +%s) - start )) > /out/test_s
exit $rc
INNER
)"

gf_argv() {  # fills GF_ARGV: the test container's docker argv
  GF_REF="git+file:///work/gufo.git?ref=refs/heads/ai-sdlc/$BASE_TAG&rev=$GF_HEAD"
  GF_ARGV=(run --name "$GF_NAME" --pull never --network bridge --cpus "$GF_TEST_CORES"
    --memory "${GF_TEST_MEM_MIB}m" --memory-swap "${GF_TEST_MEM_MIB}m"
    -v "$GF_SRC:/src/gufo:ro" -v "$NIX_VOLUME:/nix" -v "$GF_OUT:/out"
    -e "NIX_CONFIG=experimental-features = nix-command flakes
accept-flake-config = true
max-jobs = 1
cores = $GF_TEST_CORES
sandbox = false"
    -e "GF_BRANCH=ai-sdlc/$BASE_TAG" -e "GF_HEAD=$GF_HEAD" -e "GF_REF=$GF_REF"
    -e "GF_CHECKS=$GF_CHECKS" -e "GF_OWNER=$(id -u):$(id -g)"
    "$NIX_IMAGE" bash -euo pipefail -c "$GF_INNER")
}

# --- main -----------------------------------------------------------------------
GF_WPID=""; GF_CNAME=""
gf_cleanup() {
  [ -n "$GF_WPID" ] && kill "$GF_WPID" 2>/dev/null
  [ -n "$GF_CNAME" ] && docker_ rm -f "$GF_CNAME" >/dev/null 2>&1
  return 0
}

main() {
  local rc c h12 why
  case "${1:-}" in -h|--help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; return 0 ;; esac
  GF_CHECKS="${*:-$GF_ALL_CHECKS}"
  for c in $GF_CHECKS; do
    case " $GF_ALL_CHECKS " in *" $c "*) ;; *) gf_die "usage: test.sh [check ...] (unknown '$c'; one of: $GF_ALL_CHECKS)" 2 ;; esac
  done
  gf_load_conf
  GF_HEAD="$(git rev-parse HEAD 2>/dev/null)"; h12="${GF_HEAD:0:12}"
  GF_OUT="${XDG_CACHE_HOME:-$HOME/.cache}/gufo-fork-test/${h12:-nohead}"
  GF_NAME="gufo-fork-test-${h12:-nohead}"
  GF_SRC="$PWD"
  gf_host_checks
  if [ "${GF_DRY_RUN:-0}" = 1 ]; then
    gf_argv
    printf 'checks=%s\nsandbox=off\ndocker_run:\n' "$GF_CHECKS"; printf '  %s\n' "${GF_DOCKER:-docker}" "${GF_ARGV[@]}"
    return "$GF_REFUSED"
  fi
  [ "$GF_REFUSED" = 0 ] || gf_die "refused (the check lines above); nothing was run"
  gf_argv
  mkdir -p "$GF_OUT" || gf_die "cannot create $GF_OUT"
  rm -f "$GF_OUT/stopped" "$GF_OUT/results"
  docker_ rm -f "$GF_NAME" >/dev/null 2>&1
  trap gf_cleanup EXIT; trap 'exit 130' INT TERM
  GF_CNAME="$GF_NAME"
  # The build's watcher, from the same folder, stops the container at a floor.
  bash "$GF_TOOL_DIR/build.sh" --watch "$GF_NAME" "$GF_OUT/stopped" & GF_WPID=$!
  gf_say "test: $GF_NAME checking ${h12} ($GF_CHECKS; log: $GF_OUT/test.log)"
  docker_ "${GF_ARGV[@]}" 2>&1 | tee "$GF_OUT/test.log" >&2; rc="${PIPESTATUS[0]}"
  kill "$GF_WPID" 2>/dev/null; wait "$GF_WPID" 2>/dev/null; GF_WPID=""
  docker_ rm -f "$GF_NAME" >/dev/null 2>&1; GF_CNAME=""
  if [ -s "$GF_OUT/stopped" ]; then why="$(cat "$GF_OUT/stopped")"; gf_die "$why" 5; fi
  printf 'head=%s\n' "$GF_HEAD"
  cat "$GF_OUT/results" 2>/dev/null
  printf 'test_s=%s\npeak_mem_bytes=%s\nout_dir=%s\n' "$(cat "$GF_OUT/test_s" 2>/dev/null)" "$(cat "$GF_OUT/memory.peak" 2>/dev/null)" "$GF_OUT"
  [ "$rc" = 0 ] || gf_die "a check failed or the container exited $rc (see $GF_OUT/test.log)" 3
  return 0
}

main "$@"
