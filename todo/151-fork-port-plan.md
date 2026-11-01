---
id: 151
title: Port, one at a time in isolated feature branches, the changes from the mikealanni/gufo fork that could speed up our Qwen3.8-Flash-Next serving — ranked by expected percentage effect (Q4_1 decode copy first)
status: refined
theme: gufo-engine
contract: v2
depends_on: [149]
resources: [serve-outage]
refined: 2026-10-23
complexity: 8
size: XL
subscores:
  scope_breadth: 2
  coupling: 2
  decision_uncertainty: 2
  novelty: 2
  external_dependency: 3
  verification: 3
estimate:
  wall_clock_hours: 24/48/96
  tokens: 10M-30M
  usd: 100-300
  priced: 2026-10-23
  review_hours: 4/8/12
  box_hours: 8/16/32
  prs: 5
  reference_class: [149, 85.4, 85.5, 51]
decisions: [12, 19, 42]
tasks:
  - id: T1
    phase: P1
    depends_on: []
    parallel: true
    acceptance: "read-only: for upstream 6c8b2ff4 (#714), every upstream change the item lists (#703 #699 #664, #613 #683, #603 #655, #622, #690 #571, #714) is checked as an ancestor with git merge-base --is-ancestor and tabled (present or absent, commit id), and the fork's ai-sdlc/upstream-main-vocab (10 ahead of upstream-main) is recorded as based on the old commit; nothing is pushed"
  - id: T2
    phase: P1
    depends_on: [T1]
    parallel: false
    acceptance: "ai-sdlc/upstream-main is re-cut locally to 6c8b2ff4 (linear, noreply author, no merge commits), ai-sdlc/leakcheck.sh passes on the range, and the old tip 1167a944 is kept under a named backup branch; the push (a force-with-lease update of a public branch) happens only after the user's yes, which is recorded with the date"
  - id: T3
    phase: P1
    depends_on: []
    parallel: true
    acceptance: "the deployment's mikealanni-gufo preregistration note is committed before any patched build generates anything: the incumbent (unpatched re-cut image by exact image ID, never the v0.19.0 pin), per-port target legs (Q4_1 and Q5_1: decode legs S and C; WMMA: a prefill leg with cached prefixes), the adopt bar (at least +3% with the clustered interval's lower bound above 0, the A/A null passing, on every targeted leg), the sign convention and harm sign (ADR 0012), the paired design reusing item 149's A5/A7/A10 (36 held-out Pi turns, per-visit restart, per-turn seeds, 18 visits, clustered bootstrap, halves null), the quality gate that sends Q4_1 to Q5_1, the Pi task slice margin, and the WMMA microbench gate"
  - id: T4
    phase: P2
    depends_on: [T3]
    parallel: true
    acceptance: "the leased A/B and divergence harness from item 149 (the deployment's vocab-trim harness: pi_ab6.py, divergence.py) is generalised to take two arm images by ID and a leg (S, C, prefill) as arguments, with a make selftest target in make check that runs on synthetic data with no session text and no model, a specific row in the deployment's check-changed map above any catch-all and make check-changed-selftest passing; item 149's own results are not re-run"
  - id: T5
    phase: P2
    depends_on: [T2, T4]
    parallel: false
    acceptance: "under a lease (mode.sh take): the unpatched re-cut image is built with ai-sdlc/build.sh (image ID and config_hash recorded) and measured against the v0.19.0 pin image on legs S, C and prefill with the A/A null, which gives the combined effect of V1 to V5 (verify, do not port) with intervals; the box is restored to the managed pin and the restore verified; the re-cut image is the incumbent for every later task"
  - id: T6
    phase: P3
    depends_on: [T2]
    parallel: true
    acceptance: "the deployment's mikealanni-gufo design note maps with file and line how the fork's decode copy (GUFO_DECODE_COPY=balanced) selects classes, quantises at load, and is read by the target pass, the MTP draft layer and batched rows; how upstream-main loads the same tensors; the fork's licence and the attribution to carry; and answers the research's open questions (MTP layer coverage, behaviour above 8 rows, device memory), each confirmed or corrected"
  - id: T7
    phase: P3
    depends_on: [T3, T6]
    parallel: false
    acceptance: "branch ai-sdlc/port-q4-1-decode-copy cut from the re-cut ai-sdlc/upstream-main holds the Q4_1 decode copy behind a flag that defaults off, in linear commits with noreply author; CPU/unit tests pass (the quantiser against a reference, class selection, flag off is a no-op); leakcheck passes; a candidate image is built and its image ID and config_hash recorded"
  - id: T8
    phase: P3
    depends_on: [T5, T7]
    parallel: false
    acceptance: "under a lease: Q4_1 divergence report on the 100 held-out turns against the incumbent image (d, first-divergence position, the worst case not excluded; English-only wording; ADR 0012 tier 3 until T9), the 4-stream behaviour above 8 rows, whether the MTP layer is copied, device memory with and without the copy, and the weights actually served (UD-Q4_K_XL, Q8_0 trunk) confirmed from the load log; the box is restored and verified"
  - id: T9
    phase: P3
    depends_on: [T8]
    parallel: false
    acceptance: "paired speed A/B of the Q4_1 image against the incumbent under a lease on legs S and C (T3's design, A/A null reported beside it, image ID on every row), and the Pi task slice pass rate with the discordant table and exact McNemar p; a leg whose null fails is reported 'null failed' and not judged"
  - id: T10
    phase: P3
    depends_on: [T9]
    parallel: false
    acceptance: "the Q4_1 verdict (adopt as flag, keep, drop) against T3's bar with its numbers, ADR 0012 wording (never 'lossless'), and the recorded decision whether T3's quality gate failed, which is what opens T11"
  - id: T11
    phase: P4
    depends_on: [T10]
    parallel: false
    acceptance: "only if T10 records a quality-gate failure: branch ai-sdlc/port-q5-1-decode-copy cut from the re-cut upstream-main (not from the Q4_1 branch) with the Q5_1 copy behind a default-off flag, unit tests, leakcheck and a candidate image ID; otherwise closed 'not built' with T10's numbers, and T12 with it"
  - id: T12
    phase: P4
    depends_on: [T11]
    parallel: false
    acceptance: "only if T11 was built: T8 and T9 repeated for Q5_1 under a lease and a verdict against the same bar; if both Q4_1 and Q5_1 fail quality the decode-copy family is dropped and says so"
  - id: T13
    phase: P5
    depends_on: [T10, T12]
    parallel: false
    acceptance: "under a lease, a microbench measures the indexer-score kernel's share of prefill step time on the Pi-like cached-prefix turns and at 131k, and the WMMA gate is recorded against T3's threshold: build only if that share times a plausible speedup reaches the bar; otherwise 'WMMA not built' with the numbers, and T14 and T15 are closed with it"
  - id: T14
    phase: P5
    depends_on: [T13]
    parallel: false
    acceptance: "only if T13's gate passes: branch ai-sdlc/port-wmma-indexer-score cut from the re-cut upstream-main with matrix-core indexer scoring behind a default-off flag, the F32 guard, unit tests comparing selected indices with the F32 path, leakcheck and a candidate image ID"
  - id: T15
    phase: P5
    depends_on: [T14]
    parallel: false
    acceptance: "only if T14 was built: paired prefill A/B on the targeted leg under a lease with the A/A null, the divergence of selected indices and of greedy output on the held-out turns, and a verdict against the bar"
  - id: T16
    phase: P6
    depends_on: [T12, T15]
    parallel: false
    acceptance: "a verdict per port (adopt as flag, keep, drop, not built) with numbers, the V1 to V5 result from T5, the deployment's mikealanni-gufo report and summary.md updated from estimates to measurements with each estimate beside its measured value, an ADR if any port is adopted, the index, CHANGELOG and TODO index updated, and every fork branch pushed only after the user's yes and a passing leak check"
---

## 151. Isolated ports from the mikealanni/gufo fork

> **Moved here on 2026-11-01** from the ai-sdlc deployment repository (its item 170.2), with the same id, title and
> status. ADR numbers, `make` targets, scripts, files and other items named below are the deployment's unless they
> name this fork. Its evidence is private and is cited in plain text ("private evidence, ai-sdlc item N").

Refined via plan-review on 2026-10-23 (single-item run by a subagent with no question
tool: the decisions below that need the user are `[default]` marked "to confirm" and go
back to the coordinator, as in item 149's refinement). 0 rounds, 0 questions asked.
Decisions are tagged `[user]`, `[default]` or `[override]`. Filed 2026-10-22 at the
user's request after a read-only research pass
(the deployment's mikealanni-gufo report). Nothing was compiled or run in
that research, so every percentage is an estimate until a task here measures it.

### Why (as filed)

`github.com/mikealanni/gufo` (default branch `fork-enhancements`, 137 commits ahead of and 102 behind upstream, base v0.11.0 plus 9) claims large gains: prefill +16.9%
(UD-IQ4_XS, XDNA2 NPU on) and MTP decode 47 to 100 tok/s (Q8_0 trunk re-quantised to Q4_1). Almost none of it transfers as stated: our weights are UD-Q4_K_XL, our text is
sampled Pi traffic with cached prefixes, and the box has no XRT. The analysis separates what could help us from what cannot, and what upstream already has.

### Rules for every port

- **One item, one branch, one at a time**, cut from `ai-sdlc/upstream-main` (recut it first to include upstream `6c8b2ff4`, #714, newer than the `1167a944` it was cut from).
  Branch name `ai-sdlc/port-<slug>`; linear commits, no merges, noreply author; FORK.md push rules and `ai-sdlc/leakcheck.sh` apply; nothing is pushed without the user's yes.
- **Tests, in order:** CPU/unit tests; greedy identity, or for approximate items the divergence report of item 149 on the 100 held-out turns and the pass-through-agnostic
  English-only wording; then the paired speed A/B of item 149's addenda A5/A7/A10 on the Pi-like turns (single stream, then four concurrent streams) with the A/A null,
  per-visit restarts, clustered bootstrap. ADR 0012 wording: only a bit-identical item may be called lossless.
- **Acceptance bar:** at least +3% on the leg the item targets with the interval's lower bound above 0, the null passing, and no quality regression; for
  approximate items also the Pi task-slice non-inferiority check. A below-bar result is a valid "drop" verdict, recorded with its numbers.

### Ranked sub-items (highest expected percentage first)

1. **Q4_1 decode copy of the Q8_0 dense trunk** (branch `ai-sdlc/port-q4-1-decode-copy`). Re-quantise the "balanced" classes at load into a decode-only copy
   (the fork's `GUFO_DECODE_COPY=balanced`, its default): 2,757 MiB of Q8_0 in our file become 1,622 MiB of Q4_1, 1.19 GB less read per target pass.
   - **Expected:** prefill 0%; decode **+5% to +10% (central +8%)**, ceiling 8.4% (5.1 ms of a 60.6 ms round at 233 GB/s); the fork's own IQ4_XS numbers (about 5.3 ms per token) agree. Confidence medium-low.
   - **Exactness:** approximate (tier 3 until a quality A/B). The fork quotes +1.3% perplexity on IQ4_XS, one prompt.
   - **Risks / unknowns:** quality on Q4_K_XL; the fork's 8-row limit at four concurrent streams; whether the MTP draft layer is copied; +1.6 GiB device memory.
2. **Q5_1 decode copy** (`ai-sdlc/port-q5-1-decode-copy`), **only if item 1 fails its quality gate**: milder, decode +3% to +6%, confidence medium-low, approximate;
   declined upstream on quality grounds.
3. **Matrix-core (WMMA) indexer scoring** (`ai-sdlc/port-wmma-indexer-score`): prefill about +1% to +1.5% on average, up to +3% at 131k, decode 0%, confidence low;
   an approximate selector with an F32 guard; upstream rejected the idea once. Last of the three because the expected gain is the smallest.

### Verify, do not port (already in upstream main; confirm they are in `ai-sdlc/upstream-main` and measure the effect on the pin's replacement)

- V1: #703 / #699 / #664 (draft pricing, sampled decoding overhead) - decode 0% to +5%, distribution-preserving.
- V2: #613 / #683 (history drafting, n-gram prefetch) - decode 0% to +5%, exact.
- V3: #603 / #655 (draft vocabulary) - small gain, harms non-English (irrelevant to English-only use); ties to item 149's finding that the draft already restricts rows.
- V4: #622 (attention compaction) - about 0%.
- V5: #690 / #571 wide prefill, plus #714 (BF16 indexer position-invariant) - prefill +10% to +19% on 470 to 768-token prefills (upstream's figure), exact.

### Do not port

XDNA2 NPU prefill (4,096-token chunks only so about 0% on cached prefixes; lossy; needs XRT; binary `xclbin` programs; host freezes documented in the fork's `docs/NPU_FREEZE.md`);
`hc_*` and `attn_output` copies (+1.5% to +2.5% at most, the most quality-damaging classes); F16 residual, 8192/16384 chunks, expert-tile buckets, routed-GEMM scheduling,
row-tile grouping, inject-in-combine (upstream measured flat or negative); chunked WY gated delta rule (about 500x worse numerics); greedy-only draft policy (our traffic is sampled);
DFlash2 for Flash-Next (needs a draft model we lack); the 27B/35B work and the IQ-quant kernels (other models, other quants); the SDMA blit-copy re-exec (use `HSA_ENABLE_SDMA=0`
if interleaved-session faults ever appear); snapshot and cache items (they target code upstream has replaced).

### Not determined by the research

Q4_1 coverage of the MTP layer and behaviour above 8 rows; Q4_1 quality on Q4_K_XL; the fork's sampling, JSON-constraint and cache-file changes were skimmed, not read line by line.

### Implementation workflow (mandatory — complete in order)

1. [ ] Create an isolated git worktree before writing any artifact:
       `git worktree add ../ai-sdlc-item-151 -b feat/item-151`
2. [ ] Produce every deliverable below inside that worktree — never directly on `main`.
3. [ ] All quality gates must pass before opening a PR:
       `make check` exits 0 (see `CONTRIBUTING.md` → Pull Request Checklist).
4. [ ] Raise a Pull Request: `gh pr create`
       **No direct commits to the default branch (`main`) are permitted.**

### Design decisions (resolved)

Checked on 2026-10-23 against the public fork (`gh api`, read-only): branches
`ai-sdlc/upstream-main` (`1167a944`, 19 ahead of `ai-sdlc/v0.19.0`, 0 behind),
`ai-sdlc/upstream-main-vocab` (10 ahead of upstream-main, item 149's work),
`ai-sdlc/v0.19.0-vocab`, `ai-sdlc/exp-*`, `ai-sdlc/tooling`. Upstream `6c8b2ff4` is 1
commit past `1167a944` and 4 past tag v0.21.0; upstream main has since moved to
`79c2dcc5`. The managed pin is `ai-sdlc/v0.19.0` (the deployment's fork lock file), so the
pin and upstream-main differ by 19 upstream commits.

- **Re-cut base** `[default]`, to confirm: `6c8b2ff4` (#714), as the filing says. It is the
  smallest move from `1167a944` and adds the BF16 indexer fix; alternatives were the
  current tip `79c2dcc5` (moving target, unreviewed range) and tag v0.21.0 (lacks #714).
- **The incumbent is the unpatched re-cut image, not the pin** `[default]`: the ports sit
  on a base 19+ upstream commits past v0.19.0, so a pin comparison would credit V1 to V5
  to the port. T5 builds the re-cut image by ID and measures it once against the pin,
  which is also the V1 to V5 "measure the effect" step. Pin moves stay out of scope.
- **Split** `[default]`, accepted 2026-10-23 (user: use the defaults); filed as 151.1 (T1-T5), 151.2 (T6-T10), 151.3 (WMMA, T13-T15; ids were issued in order, so Q5_1 will take the next free id when T10 opens it). Complexity 8 (XL) forces the
  proposal. Recommended shape: 151.1 base and baseline (T1 to T5), 151.2 Q4_1 (T6 to
  T10), 151.3 Q5_1 (T11, T12, opened only if T10 fails the quality gate), 151.4 WMMA (T13
  to T15), with 151 as parent carrying T16. Ids come from
  `make todo-new SLUG=... PARENT=151` only on acceptance, after `git fetch && make todo-xref`.
  Alternatives: two children, or no split (`[override]`). Not obtainable here (no
  question tool).
- **WMMA is gated** `[default]`, to confirm: the research puts it at +1% to +1.5% prefill,
  under the item's own +3% bar, so T13 measures the indexer kernel's share first and the
  port is built only if the bar is reachable. A "not built" verdict is valid. Alternatives:
  build unconditionally, or drop it from the item.
- **Quality gate and the Pi task slice** `[default]`, to confirm: item 149 never ran the
  slice (A5.5, A7.7). Q4_1 is gated on the 100-turn divergence report plus slice
  non-inferiority, both pre-registered in T3; failing sends the family to Q5_1. The slice
  harness is built or reused in T4 if 149 has not produced one. Alternatives: divergence
  only (cheaper, weaker), or wait on item 149.
- **Both legs** `[default]`: Q4_1 and Q5_1 must meet the bar on legs S and C, as item 149's
  A7.5 required, because the fork's 8-row limit puts leg C at risk. WMMA is judged on the
  prefill leg. Evidence: the research lists the 8-row limit as undetermined.
- **Measurement design** `[default]`: item 149's A5/A7/A10 as they stand, 18 visits (six per
  arm), restart per visit, per-turn seeds, clustered bootstrap over turns and visits, the
  incumbent's halves as the null. Evidence: 149's A5.3 null failed twice at 8 and 9 visits
  and its variance decomposition (vocab-trim.md section 14) sized the 18-visit run.
- **A new prefill leg** `[default]`: 149's harness has no prefill leg. T3 defines it on the
  same 36 turns with cached prefixes (time to first token and prefill tokens per second),
  and T4 adds it.
- **Stacking with item 149** `[default]`: ports are measured on the plain re-cut base, never
  on top of the vocab trim. A combined run is out of scope; gains may overlap in the decode
  round (Q4_1 saves about 5.1 ms of 60.6 ms, the head about 2 ms).
- **Licence and attribution** `[default]`: T6 reads the fork's licence before any code is
  copied and records the attribution; a clean reimplementation from the described change is
  the fallback. Evidence: 109 of the fork's 131 non-merge commits carry a Claude trailer.
- **Branches** `[default]`: one port per branch from the re-cut `ai-sdlc/upstream-main`
  (Q5_1 is not stacked on Q4_1), linear commits, no merges, noreply author, leakcheck,
  and no push without the user's yes (ADR 0042). The old `upstream-main-vocab` stays
  untouched.
- **Acceptance wording** `[default]`: only a bit-identical item may be called lossless
  (ADR 0012); all three ports are approximate.
- **Theme** `[default]`: `gufo-engine` (the item's own key, kept).

### Estimate

- Scope breadth 2: three ports, a baseline, a harness change; no new subsystem.
- Coupling 2: loader, target pass, MTP draft and indexer in one engine.
- Decision uncertainty 2: the gates (Q4_1 quality, WMMA share) decide what is built.
- Novelty 2: a decode-time re-quantiser and a WMMA kernel are new here but specified by the fork.
- External dependency 3: serve-outage leases, public fork pushes, user consent.
- Verification 3: paired A/B with a null, tier 3 quality, a slice never run before.
- S = 14, complexity = max(1, ceil(140/18)) = 8; two 3s give floor 5; size XL.
- Reference class: 149 (vocab trim, 20/40/80 h planned, same harness), 85.4 (fork build),
  85.5 (fork patch, closed partial), 51 (paired A/B on the pin). Ranges are the plans'
  estimates; actuals are not re-derived here.
- PERT agent wall-clock 24/48/96 h; tokens 10M-30M; $100-300 (priced 2026-10-23, the rates
  used for item 149); human review 4/8/12 h; box hours 8/16/32 (baseline, Q4_1 divergence
  and two legs, a possible Q5_1 repeat, microbench and a possible prefill leg, restores);
  5 PRs (one per child plus the parent verdict).
- Split: proposed (see Design decisions); acceptance pending.

### Risks (pre-mortem)

"Assume this shipped `partial`. Why?"

- **A fork change does not port** (item 85.5's cache-owner patches did not): T6 maps the
  code before T7; the patch is kept small on its own branch; a port that cannot be built
  is closed with the reason.
- **No lease window or an unverified restore**: T5, T8, T9, T12, T13, T15 need
  `serve-outage`. Accepted named risk: the offline half (T1 to T4, T6, T7) still
  delivers; no speed claim is made without a leased, null-checked A/B, and the item stays
  `in-progress`.
- **The gain is below the bar**: Q4_1's ceiling is 8.4% and its central estimate +8%, but
  leg C is exposed to the 8-row limit; a below-bar result is a valid drop verdict.
  WMMA is gated by T13 for this reason.
- **Noise swamps a 3% effect**: 149's null failed twice; the design uses 18 visits and the
  clustered interval, and a leg whose null fails is "null failed", not judged.
- **Quality fails on Q4_K_XL** (the fork's +1.3% perplexity is IQ4_XS, one prompt): T8 and
  T9 measure it; Q5_1 is the fallback, and if both fail the family is dropped.
- **Public fork push or leak**: `leakcheck.sh` and an explicit yes before the re-cut push
  (a force update of a public branch, kept recoverable by the backup branch) and every port.
- **The base moves**: pinned to `6c8b2ff4` for the whole item; a later base is a later item.
- **Unverified external claims** (+16.9%, 47 to 100 tok/s, +1.3% perplexity): hypotheses
  until a task here measures them.

### Tasks

**P1 — base (offline)**

- [ ] **T1** Table which listed upstream changes are in 6c8b2ff4.
- [ ] **T2** Re-cut `ai-sdlc/upstream-main` locally; push after the user's yes.
- [ ] **T3** Pre-register the bar, legs, quality gate and WMMA gate.

**P2 — harness and baseline**

- [ ] **T4** Generalise the 149 harness to two images and three legs, with a selftest.
- [ ] **T5** Build the incumbent image; measure it against the pin (V1 to V5).

**P3 — Q4_1 decode copy**

- [ ] **T6** Design map of the fork's decode copy and the open questions.
- [ ] **T7** Q4_1 patch on `ai-sdlc/port-q4-1-decode-copy`, tests, candidate image.
- [ ] **T8** Divergence, 4-stream, MTP-layer and memory checks under a lease.
- [ ] **T9** Paired speed A/B and Pi slice for Q4_1.
- [ ] **T10** Q4_1 verdict and the quality-gate decision.

**P4 — Q5_1 (conditional)**

- [ ] **T11** Q5_1 patch, only if Q4_1 fails its quality gate.
- [ ] **T12** Q5_1 measurement and verdict, only if built.

**P5 — WMMA indexer (gated)**

- [x] **T13** Indexer share microbench and the WMMA gate. Done 2026-10-23: share 0.317% on leg P units (131k: 3.24%) against 4.5%; WMMA not built (wmma-gate.md (private evidence, ai-sdlc item 151)).
- [x] **T14** WMMA patch, only if the gate passes. Closed, not built (T13).
- [x] **T15** WMMA prefill A/B and divergence, only if built. Closed, not built (T13).

**P6 — verdict**

- [ ] **T16** Verdict per port, docs, ADR if any adopted, index; pushes on the user's yes.

### Acceptance criteria (definition of done)

- A1 (T1): the in-upstream status of V1 to V5 at the re-cut base is tabled by commit.
- A2 (T2): the re-cut branch exists locally, leak-checked, with a backup of the old tip,
  and is pushed only after a recorded yes.
- A3 (T3): the bar, legs, harm sign, design, quality gate and WMMA gate are
  pre-registered before any patched generation.
- A4 (T4): the generalised harness has a hermetic selftest in `make check` and a
  specific `check_changed.map` row.
- A5 (T5): the incumbent image ID is recorded and V1 to V5 are measured against the pin
  with a null, the box restored and verified.
- A6 (T6): the design map confirms or corrects the research's claims and open questions.
- A7 (T7): the Q4_1 branch exists, default off, tested, leak-checked, with an image ID.
- A8 (T8): divergence, 4-stream, MTP-layer and memory results are reported.
- A9 (T9): Q4_1 speed and slice results meet ADR 0012's reporting, image ID on every row.
- A10 (T10): the Q4_1 verdict and gate decision are recorded with numbers.
- A11 (T11, T12): Q5_1 is built and measured, or closed "not built" with T10's numbers.
- A12 (T13): the WMMA gate decision is recorded with numbers.
- A13 (T14, T15): WMMA is built and measured, or closed "not built" with T13's numbers.
- A14 (T16): every port has a verdict, the research docs show measured beside estimated,
  an ADR exists if any is adopted, and the index and CHANGELOG are updated.

### Documentation

- the deployment's mikealanni-gufo report (full form, updated in T16),
  `summary.md` (companion), `preregistration.md` (T3) and `design.md` (T6); sources
  bar: the fork's code and README, upstream PR threads, the GGUF headers and the leased
  runs; indexed from the deployment's evaluation index.
- the deployment's serving-scripts README and a Make selftest target for the generalised harness (T4),
  a specific row in the deployment's check-changed map above any catch-all, then
  `make check-changed-selftest`; the deployment's selftest environment list if a new `X_` variable is read.
- the deployment's ADR index row and ADR file only if a port is adopted (T16);
  `CHANGELOG.md` entry; `TODO.md` regenerated by `make todo-index`.
- In the fork: `ai-sdlc/FORK.md` gains a note per port branch and the re-cut.

### Out of scope

Pushing to the public fork without the user's yes, moving the managed pin (separate
items), stacking a port on item 149's vocab trim, and any change not listed above.
