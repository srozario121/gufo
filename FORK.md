# FORK.md — srozario121/gufo

This is a fork of [gufo-org/gufo](https://github.com/gufo-org/gufo). It
carries a short series of patches on top of a pinned gufo release, and the
tooling that builds the `gufo-runtime` image from them. It exists so that one
deployment can run a patched gufo. No feature request is filed upstream; a
patch that proves itself can be offered upstream later.

gufo is MIT-licensed. Its `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES` are
left untouched on every code branch. The tooling on this branch is MIT as
well (`LICENSE-NOTE`).

## Branches

| Branch | What it holds |
|---|---|
| `ai-sdlc/v0.10.0` | The patch branch: upstream's `v0.10.0` commit plus our patches, as ordinary commits. Today it has **no patches**, so it is exactly upstream's `3020d2460211b921a8a5be1989b4d4640f432d4c` |
| `ai-sdlc/tooling` | This file and the build and leak-check tooling, on an **orphan** branch with no shared history |
| `main` | Upstream's default branch, as forked. Not used |

The tooling is not committed on the patch branch on purpose. gufo stamps its
own commit into `gufo --version` and into the image label
`org.gufo.engine.revision`, so the patch branch can only reproduce the stock
image if its HEAD is upstream's commit. One patch branch exists per base
release, `ai-sdlc/<tag>`. A move to a new release creates a new branch and
keeps the old one for rollback.

### Folder layout

The main clone checks out the patch branch. The tooling branch is a linked
worktree inside it, at `ai-sdlc/`, which the main clone hides through
`.git/info/exclude`:

```
gufo_fork/               main clone, branch ai-sdlc/v0.10.0
gufo_fork/ai-sdlc/       linked worktree, branch ai-sdlc/tooling (this file)
```

Set it up from a fresh clone:

```
git clone https://github.com/srozario121/gufo gufo_fork && cd gufo_fork
git remote add upstream https://github.com/gufo-org/gufo && git fetch upstream --tags
git checkout ai-sdlc/v0.10.0
git worktree add ai-sdlc ai-sdlc/tooling
echo /ai-sdlc/ >> .git/info/exclude
git config user.name srozario121
git config user.email 37711146+srozario121@users.noreply.github.com
ai-sdlc/install-hooks.sh
```

## The base

| What | Value |
|---|---|
| Base tag | `v0.10.0` (released 2026-10-09 11:56 UTC) |
| Base commit | `3020d2460211b921a8a5be1989b4d4640f432d4c` |
| Stock image | `ghcr.io/gufo-org/toolboxes/gufo-runtime@sha256:089221120dee1e61e04acb9bc398295fecd4dd68aac3d15dfd9ec44a6853af6a` (0.10.0) |
| Stock engine store path | `/nix/store/1fkxnrmwvg2r167mxjimzxirsb9zbwz2-gufo-0.10.0` |
| Stock image layers | 100 |
| Toolboxes commit | `f3017441bee4e3b4d4889307b0df5b71991bb0ba` (2026-10-09 11:24 UTC) — **VERIFIED** |
| Nix | 2.35.2, in `nixos/nix@sha256:7a007c766426c1877758ddc5cb87a965ac131fc78c582ce0083d922d51ae945c` |
| Expected derivation | `/nix/store/izfwqpzfim1bz2n1nj3i6nsbpdww2q6w-gufo-runtime.tar.gz.drv` |

**How the toolboxes commit was found.** The stock image's
`org.opencontainers.image.revision` label reads `unknown`, so the label cannot
name it. The toolboxes release workflow run
[37926894479](https://github.com/gufo-org/toolboxes/actions/runs/37926894479)
has `head_sha` `f3017441bee4…` and started 13 seconds after gufo's `v0.10.0`
release (its logs have expired). That commit's own `flake.lock` pins an
earlier gufo commit (`94a4fa8d…`); the release build overrides the
`gufo-engine` input with the release, and the build here does the same with
`--override-input`. Evaluated that way, both `github:gufo-org/gufo/v0.10.0` and
this fork's `3020d24` give the expected derivation above.

## Build

`ai-sdlc/build.sh` builds the image from the checked-out patch branch, with
nothing installed on the host. Run it from the main clone, with the engine
stopped (the build wants about 20 GiB of MemAvailable to start):

```
cd gufo_fork
GF_DRY_RUN=1 ai-sdlc/build.sh                 # every check, and the exact docker run
EXPECTED_DRV=/nix/store/izfwqpzfim1bz2n1nj3i6nsbpdww2q6w-gufo-runtime.tar.gz.drv \
  ai-sdlc/build.sh --compare ghcr.io/gufo-org/toolboxes/gufo-runtime@sha256:089221120dee1e61e04acb9bc398295fecd4dd68aac3d15dfd9ec44a6853af6a
```

- **Container.** Nix runs in `nixos/nix` pinned by digest (`build.conf`), never
  pulled implicitly. The store lives in the named volume `gufo-fork-nix`, under
  a 150 GiB budget with a 20 GiB reserve. Build jobs are capped at
  `max-jobs 2 x cores 8`. The container gets a cgroup memory cap and no extra
  swap. A host-side watcher stops it at MemAvailable below 3 GiB, SwapFree
  below 1 GiB, or over the disk budget.
- **Inputs.** The patch branch is cloned bare inside the container from a
  read-only mount, and given to Nix as
  `git+file:///work/gufo.git?ref=refs/heads/ai-sdlc/v0.10.0&rev=<HEAD>`. A
  `path:` input would stamp gufo `dirty`. Toolboxes is fetched at its commit
  and exported with `git archive` into a `path:` flake. A `github:` or `git+`
  flake would stamp toolboxes' revision into the labels, and the stock image
  says `unknown`.
- **Refusals.** A dirty or untracked tree, a detached HEAD or another branch,
  a re-pointed base tag, a HEAD not descended from the base commit, a merge in
  the patch range, a linked worktree, a Nix image not by digest or not local,
  more than 16 build cores, a disk budget that does not fit, or too little
  memory. Each is refused before any build container starts.
- **Result.** `docker load`, then a retag to
  `ai-sdlc/gufo-runtime-fork-unlabelled:<HEAD12>`. The loaded name is the
  stock tag's name: what it held before the load is recorded, and afterwards
  it is removed if the load created it, or pointed back if the load moved it
  (stock is pinned by digest either way). A
  label-only `docker build` (no new layer) then makes
  `ai-sdlc/gufo-runtime-fork:<HEAD12>` with the labels
  `org.ai-sdlc.gufo-fork.{commit,branch,base-tag,base-commit,toolboxes-commit}`.
  It is pinned as `ai-sdlc/gufo-runtime-fork@sha256:<image id>`.
- **`--compare`.** Compares the image with stock: the `/bin/gufo` store path,
  the layer `diff_ids`, the config (`Env`, `Cmd`, `User`, `WorkingDir`,
  `Entrypoint`, labels except the fork's own) and `gufo --version`. It prints
  `compare=REPRODUCES` or `compare=DIFFERS: <which>`.
- **The sandbox is off, so the bytes never match stock.** A Nix sandbox inside
  Docker needs `--privileged`, and the build never weakens its container, so
  `sandbox = false` (the report prints `sandbox=off`). The store paths and the
  derivation are still stock's, but the unsandboxed build runs in a random
  `/builds/nix-<n>-<rand>/` directory instead of the sandbox's `/build`, and
  gufo embeds that path in assert strings, in the host code and in the HIP
  fatbin. The empty `ai-sdlc/v0.10.0` build (2026-10-11; its evidence is
  recorded with the deployment's evaluation notes) matched stock in its
  derivation, its engine store path, `gufo --version` and 99 of its 100
  layers; only the gufo engine layer differs (its `bin/gufo` is
  25,093,760 B, stock's 25,080,832 B). So `--compare` reports `DIFFERS: diff_ids` by design:
  equivalence with stock rests on the greedy fixed-prompt check, not on the
  bytes. Two builds of the same commit differ from each other too, so a
  patch's baseline is an exact image ID of the unpatched build, never a
  rebuild of it.

Exit codes: 0 built; 1 refused; 2 usage; 3 build failed; 4 the derivation
differs from `EXPECTED_DRV` (nothing built); 5 stopped by the memory floor or
the disk budget; 6 built, but `--compare` differs.

## Rebase onto a new upstream release

1. In the main clone: `git fetch upstream --tags`.
2. Create the new patch branch at the new tag and replay the patches:
   `git checkout -b ai-sdlc/<newtag> <newtag>`, then
   `git cherry-pick <base>..ai-sdlc/<oldtag>`, or
   `git rebase --onto <newtag> <oldbase> ai-sdlc/<newtag>`. A conflict stops
   the move.
3. Update `build.conf` on the tooling branch: `BASE_TAG`, `BASE_COMMIT`, and
   the toolboxes commit that built that release, found the same way as above.
4. Run the fork's tests.
5. Build the new branch, and build the new tag unpatched. Check that the
   patched build's outputs are bit-exact with the unpatched build's where the
   patches promise it.
6. Run the deployment's pin-move checklist.
7. Record the new pin from the deployment side (its `fork-pin` verb), under
   its serving lease, in a reviewed change.

The old branch is never deleted: it is the rollback.

## Backup

Until a branch is pushed, the local clone is its only copy. Bundle every
branch and tag after each change:

```
git -C gufo_fork bundle create ~/gufo_fork-$(date +%F).bundle --branches --tags
git bundle verify ~/gufo_fork-$(date +%F).bundle
```

Restore with `git clone -b ai-sdlc/v0.10.0 <bundle> gufo_fork`, then add the
remotes and the tooling worktree as in "Folder layout".

Push state (2026-10-11): `ai-sdlc/v0.10.0` is pushed to `origin` (it equals
upstream's `3020d24`). `ai-sdlc/tooling` stays local, backed up by the
bundle, until the maintainer's dated yes to publish it is recorded.

## Push checklist

Do every item, every push:

- [ ] `ai-sdlc/install-hooks.sh --check` passes, so the pre-push hook and the
      patterns file are in place.
- [ ] **Leak check:** `ai-sdlc/leakcheck.sh --range <each branch pushed>` is
      clean. No content from the deployment's private repository goes out:
      no held-out instance ids or text, no keys or tokens, no absolute home
      paths, no private repository paths or file names, no `.env` files.
      The pre-push hook runs the same check and blocks the push.
- [ ] Every commit's author and committer is the GitHub noreply address.
- [ ] The patch branch is linear on its base tag (no merges), and its
      tests pass.
- [ ] `git push origin <branch>`; never `--force` on a pinned commit, and
      never `--no-verify`.

## Removal

Each outward change, with its removal command:

| What | Remove with |
|---|---|
| The Nix store volume | `docker volume rm gufo-fork-nix` |
| A built image | `docker rmi ai-sdlc/gufo-runtime-fork:<HEAD12>` (and `…-unlabelled:<HEAD12>` if left) |
| The Nix image | `docker rmi nixos/nix@sha256:7a007c766426c1877758ddc5cb87a965ac131fc78c582ce0083d922d51ae945c` |
| Build output | `rm -rf ~/.cache/gufo-fork-build` |
| This clone | `rm -rf gufo_fork` (after a bundle, if anything is unpushed) |
| The GitHub fork | GitHub, repository Settings, "Delete this repository" |
