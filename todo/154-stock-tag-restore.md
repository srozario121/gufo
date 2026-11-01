---
id: 154
title: "Fork tooling build.sh leaves the stock gufo-runtime tag on its last build"
status: in-progress
theme: gufo-engine
contract: v2
depends_on: []
refined: 2026-10-31
complexity: 2
size: S
subscores:
  scope_breadth: 1
  coupling: 0
  decision_uncertainty: 0
  novelty: 0
  external_dependency: 0
  verification: 1
estimate:
  wall_clock_hours: 1/1.5/3
  tokens: 300000-900000
  usd: 3-9
  priced: 2026-10-31
  review_hours: 0.25/0.5/1
  prs: 1
  reference_class: [152.5, 85.4]
decisions: [42]
tasks:
  - id: T1
    phase: P1
    depends_on: []
    parallel: false
    acceptance: "build.sh reads the archive's name by parsing manifest.json as JSON (python3 -I, first entry's RepoTags[0]; tolerant of pretty-printing, './manifest.json' and several RepoTags) instead of the one-line sed at build.sh:342; with the name read, a load that created the name removes it and a load that moved it points it back, and the 'left ... as loaded' branch is reached only when the archive has no RepoTags"
  - id: T2
    phase: P1
    depends_on: [T1]
    parallel: false
    acceptance: "gufo_fork_selftest.sh gains cases with a tiny fixture archive (a gzip tar holding a pretty-printed manifest.json, built in the test's temp dir) and a stub docker: name previously absent -> removed after load; name on another image -> pointed back; name already on the loaded image -> untouched; no RepoTags -> the 'left as loaded' line; no docker or GPU needed; the selftest's pass count is quoted"
  - id: T3
    phase: P2
    depends_on: [T1]
    parallel: true
    acceptance: "a short record in the deployment's fork-tooling README: which local ghcr.io/gufo-org/toolboxes/gufo-runtime tags (0.19.0, 0.20.0, 0.21.0, any other) name local builds rather than pulled images (read-only docker image inspect against the fork labels and the pin's digest), and whether any repo tool resolves the stock image by tag (grep of scripts/); nothing retagged or pulled"
---

## 154. Fork tooling build.sh leaves the stock gufo-runtime tag on its last build

> **Moved here on 2026-11-01** from the ai-sdlc deployment repository (its item 170.2), with the same id, title and
> status. ADR numbers, `make` targets, scripts, files and other items named below are the deployment's unless they
> name this fork. Its evidence is private and is cited in plain text ("private evidence, ai-sdlc item N").

Refined via plan-review on 2026-10-31, in a cross-item relay run over items 97, 102, 106, 154 and 155. The scan
settled this item, and no question was asked for it. Tags: `[user]` · `[override]` · `[default]` (resolved from the
evidence named). **No box time:** no lease, no GPU, no docker run. T3's tag list is read-only.

### Status (2026-10-31, built offline, no box time)

T1 to T3 built in one PR. **Measured** (hermetic): `gufo_fork_selftest.sh` case B17 runs `build.sh` non-dry
against a tag-table docker stub and a fixture archive with a pretty-printed `manifest.json`; 291 checks pass on
the fix, and the old `build.sh` fails 4 of B17's 6 checks (287 passed). T3's tag census is read-only and lives in
the deployment's fork-tooling README. **Asserted, not measured:** a real build's load step on the box
(no build was run), so the item stays `in-progress` until the next real `build.sh` run shows no "left … as
loaded" line and the stock tag unmoved. The fork's tooling copy is not synced (`gufo.sh fork-tooling --to`;
`fork-pin` refuses until it is). Nothing retagged; the open question for the user is in the README note.

### Implementation workflow (mandatory — complete in order)

1. [x] Create an isolated git worktree before writing any artifact:
       `git worktree add ../ai-sdlc-item-154 -b feat/item-154`
2. [x] Produce every deliverable below inside that worktree — never directly on `main`.
3. [ ] All quality gates must pass before opening a PR:
       `make check` exits 0 (see `CONTRIBUTING.md` → Pull Request Checklist).
4. [x] Raise a Pull Request: `gh pr create`
       **No direct commits to the default branch (`main`) are permitted.**

### Context (as filed 2026-10-24, kept short)

In item 152.5, `build.sh`'s load step printed `left ghcr.io/gufo-org/toolboxes/gufo-runtime:0.19.0 as loaded`,
so the stock tag ended up on a revert build. It was put back on `84ca2aa1d528` by hand, and that image is itself an
earlier local build. The managed image is pinned by ID, so serving was not affected.

### The cause (found at refinement, measured)

`build.sh:342` reads the name with
`sed -n 's/.*"RepoTags":\["\([^"]*\)".*/\1/p'`, which only matches `"RepoTags":["…"]` on a single line. The
archive's `manifest.json` is **pretty-printed**. The cached archive
(build.sh's cached archive of build `0c03748a271e`), read on 2026-10-31, shows `"RepoTags": [` with the tag on
the next line. So `name` is always empty, `pre` is never recorded, and every build takes the "left … as loaded"
branch. Filed hypotheses ruled out: the gzip stream reads fine with `tar -xzOf`, and the archive has a single
`RepoTags` entry.

### Design decisions (resolved)

- **Parse the manifest as JSON** `[default]`, with `python3 -I`. python3 is already a dependency of the fork
  tooling's host side. Evidence: the cause above. A sturdier regex would still depend on formatting.
- **Retag nothing** `[default]`, as filed ("Do not retag anything without the user's say-so"). T3 only records the
  state. Any repair is the user's choice afterwards.
- **No change to the pin or the managed image** `[default]`: out of scope as filed (ADR 0042's pin is by ID).

### Estimate

| Sub-score | Value | Reason |
|---|---|---|
| scope_breadth | 1 | one script block, one selftest, one README note |
| coupling | 0 | the managed image is pinned by ID |
| decision_uncertainty | 0 | the cause is known |
| novelty | 0 | a parse fix |
| external_dependency | 0 | fixture archive and stub docker |
| verification | 1 | four selftest cases |

S = 2, so complexity = max(1, ceil(20 / 18)) = 2. **2, S.** Reference class: 152.5 (where it was found), 85.4
(`build.sh`'s author). PERT 1/1.5/3 h, 0.3M–0.9M tokens, $3–9 priced 2026-10-31, review 0.25/0.5/1 h, **1 PR**.

### Risks (pre-mortem)

1. **The selftest needs docker.** *Plan change:* T2 uses a stub `docker` on `GF_DOCKER` (the script's existing
   seam, `GF_DOCKER_BIN`) and a fixture archive built in the temp dir.
2. **T3 tempts a retag.** *Accepted:* it records only. The README note ends with the open question for the user.

### Tasks

**P1**

- [x] **T1** Read the archive's name from `manifest.json` as JSON, and restore or remove the name after the load.
- [x] **T2** Selftest cases with a fixture archive and a stub docker.

**P2**

- [x] **T3** Record which stock tags name local builds and whether anything resolves them by tag.

### Acceptance criteria (definition of done)

- A load always restores the name's previous image, or removes the name if it did not exist (T1).
- Covered by fork selftest cases with no docker or GPU (T2).
- The stock tags' current state and their consumers are recorded, with nothing retagged (T3).

### Documentation

- the deployment's fork-tooling README: the load step's behaviour and the tag record (T1, T3).
- `CHANGELOG.md`: one entry.
