# Fork merge plan: our patches onto upstream v0.28.1

Written 2026-11-01. Nothing below has been built or measured yet, so the conflict counts are
what `git merge-tree` reports, not finished resolutions. It is a plan, and every speed claim in it
is a hypothesis until step 5 measures it. The patches are described in the README section
"What this fork changes".

## 1. Where things stand

| | Commit | Notes |
| --- | --- | --- |
| Deployed build | `ai-sdlc/v0.19.0` = upstream `7b278451` | No patches |
| Vocab-trim series | `ai-sdlc/upstream-main-vocab` (10 commits on `1167a944`); the same series on v0.19.0 is `ai-sdlc/v0.19.0-vocab` | Configuration G meets the speed bar (+5.6% / +5.7%) |
| Q4_1 decode copy | `ai-sdlc/port-q4-1-decode-copy` (5 commits on `6c8b2ff4`) | Screen only, quality alarm |
| Trie drafting | `ai-sdlc/c2b-v2-stats` (6 commits on v0.19.0; `c2b-v2-debug` adds a debug commit) | Slower live; dropped |
| MTP training-data tools | `feat/dump-hidden` (4 commits on v0.19.0) | Tooling |
| Upstream latest release | **v0.28.1** `12effe36` (2026-10-31), 117 commits past v0.19.0 | Proposed base |
| Upstream main | `2ea2ccd3` (2026-11-01), v0.28.1 plus 5 commits | 4 of them are the in-progress `extensions` refactor (#911, #922, #924, #927) |

**The base is v0.28.1, not the tip of main.** FORK.md pins a released tag per patch branch
(`ai-sdlc/<tag>`). main's extra commits rewire model includes and the serve layer for extensions,
and they raise the vocab series' conflicted files from 6 to 10. The one Flash-Next fix in that
range, #876 ("sweep pre-budget rows densely in budget-crossing chains"), can be cherry-picked on
its own if we need it before v0.28.2 or v0.29.0.

## 2. What upstream brings between v0.19.0 and v0.28.1 that touches our path

| Upstream change | Effect upstream reports | Interaction with our patches |
| --- | --- | --- |
| #882 end MTP chains at the first unconfident draft | Greedy C1 decode **+11%**; 4-session +11%; short replies about −10% | **The largest risk to our speed-up.** Chains get shorter, so fewer verify rows sit past the first rejection. Discard-aware flagging saved most of its time on exactly those rows, so its saving shrinks. The compact head's per-row saving stays the same, but the head's share of a round changes. Re-measure; do not carry the +5.6% forward |
| #733, #739, #655, #742 small draft head used by more request kinds (lone greedy, penalties/constraints, sampled on multi-session) | Faster drafting | Our `--vocab-trim-set` replaces the draft-vocabulary prefix and table, and the two exclude each other. The resolution must route **every** request kind that upstream now sends to the small head through the compact rows when a trim set is loaded. Otherwise trimmed requests lose upstream's gain or fall back to the full head |
| #849 bound draft-ID remapping and block selection by their buffers | Fixes out-of-range reads | Our compact draft argmax maps winners back through an ID table. Apply the same bound to that mapping and to the expand kernel |
| #758 chunked DeltaNet prefill and stream overlap; #690 wide prefill; #714 BF16 indexer | Faster prefill. We measured +5.87% on leg P for v0.19.0 → `6c8b2ff4` | Our `HeadProject` call sites in `executor.cpp` and `batch.cpp` sit next to the rewritten prefill code. The prefill head call must still use the compact rows when trim is on |
| #699, #703, #664 sampled-decoding overhead and draft pricing | Up to +5% decode on sampled traffic | The pricing assumes full-head verify cost. With the trim on, verify rows are cheaper, so check that pricing is not now too conservative (follow-up, not a blocker) |
| #683 n-gram prefetch across checkpoints; #680 cache lookups | Cache speed | None expected |
| #638 structured-output vocabulary | Grammar vocabulary prepared before serving | A grammar-constrained row must not be served from the compact head alone if an allowed token is outside the set. Confirm the trimmed path either falls back or is disabled under grammar, as on v0.19.0 |
| `logit_bias` | Upstream **still rejects** non-empty `logit_bias` (`sampling_request.hpp`) | Our parser must be re-applied on upstream's reshaped validation loop. This is one of the 6 conflicted files |

## 3. Conflict map (dry run with `git merge-tree` against v0.28.1)

| Series | Conflicted files | Notes |
| --- | --- | --- |
| Vocab trim (`upstream-main-vocab`, 10 commits) | `src/cli/serve/sampling_request.hpp`, `src/models/qwen38_flash_next/engine.cpp`, `engine.hpp`, `kernels/rocm/batch.cpp`, `kernels/rocm/executor.cpp`, `tests/cli/openai_chat_test.cpp` | The first commit (draft-vocab options) applies on its own. Most of the conflicts come from the `logit_bias` commit and the trimmed-head commit |
| Q4_1 decode copy (5 commits) | none against v0.28.1 (1 against main: `device_model.cpp`) | Applies cleanly |
| Trie drafting (6 commits) | `engine.cpp`, `engine.hpp`, `generation_metrics.hpp`, `text_generation_backend.hpp`, `cmake/Checks.cmake` | Not ported (see step 4) |
| `feat/dump-hidden` (4 commits) | `CMakeLists.txt` | Trivial |

## 4. What to port, and in what order

Each series is ported as **linear cherry-picks with `-x`**, with no merge commits (`build.sh`
refuses a merge in the patch range). Every commit uses the noreply author. Each commit gets the
`check-format.py` and CPU tests before the next one is picked.

1. **`ai-sdlc/v0.28.1`** (this branch, already created locally): v0.28.1 plus the README section
   and this plan.
2. **Vocab-trim series → `ai-sdlc/v0.28.1`**, picked from `ai-sdlc/upstream-main-vocab` in
   order:
   `1ca80bb9` draft-vocab options → `e552de2c` logit_bias/suppression → `fb39f4d6` trimmed head →
   `92b91d50` compact draft + faster fallback → `ae32c06f` timing → `5f777003` counters →
   `256e320b` discard-aware → `b3a7e177` tier 2 → `dda92916` skip-scored → `d837509f` rowlog and
   count-once.
   - Keep the failed tier-2 commit. `dda92916` composes with it, and the code is off by default.
     If it is to be removed, that is a separate revert later, not a skip now.
   - When resolving each conflict, keep **upstream's** new logic (#882 chain gate, #849 bounds,
     #758 prefill, #733/#739/#742 head routing) and re-thread our trim hooks through it. Never
     pick "ours" for a whole hunk.
3. **Q4_1 decode copy**: rebase onto v0.28.1 as its own branch
   (`ai-sdlc/port-q4-1-decode-copy-v0.28.1`). Do **not** stack it on the vocab trim: item 151 says
   each port is measured alone, and its quality gate is still unresolved (d = 0.80).
4. **Trie drafting: do not port.** It was measured slower live (−13.5% median per replay). The
   `c2b-*` branches stay on v0.19.0 as the record. Drop the debug commit `a3236ad4` in any case.
5. **`feat/dump-hidden`**: rebase onto v0.28.1 as `feat/dump-hidden-v0.28.1`. Only `CMakeLists.txt`
   conflicts. It stays out of the serving branch because it is tooling with its own derivation.
6. Leave the old branches (`ai-sdlc/v0.19.0`, `ai-sdlc/v0.19.0-vocab`, `ai-sdlc/upstream-main-vocab`)
   untouched. They are the rollback, and the A/B arms in step 5 need them.

## 5. Checks that nothing is lost: features, quality, then speed

### 5a. Feature retention (offline, on the editing host)

- [ ] Every fork flag is still parsed and listed by `gufo serve --help`:
      `--vocab-trim-set`, `--vocab-trim-fallback-threshold`, `--vocab-trim-discard-aware`,
      `--vocab-trim-fallback-skip-scored`, `--vocab-trim-tier2`, `--vocab-trim-tier2-threshold`,
      `--vocab-trim-timing`, `--vocab-trim-counters`, `--vocab-trim-rowlog`,
      `--vocab-trim-count-once`, `--mtp-draft-vocab-prefix`, `--mtp-draft-vocab-set`,
      `--logit-bias`, `--suppress-nonascii-keep`, `--suppress-nonascii-value`. Script this as a
      diff of the flag list between `ai-sdlc/upstream-main-vocab` and the new branch.
- [ ] The usage `gufo` object still reports `head_trim_*`, `head_trim_t_*`, `head_trim_ctr_*` and
      `head_trim_rowhist_*`, and `/metrics` still has `gufo_head_trim_*`.
- [ ] CPU tests: `head_trim_test`, `draft_vocab_test`, `logit_sampler_test`
      (`TestTrimmedHeadRows`), `openai_chat_test` (usage fields, `logit_bias`), `HeadTrimDiscardScan`.
      Also upstream's own `mtp_sampling_test`, `session_test` and `draft_vocab_test` after the
      #733/#739 routing change.
- [ ] `nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py`, then
      `ai-sdlc/test.sh` (format, pr-tests, cache-tests, docs, dependency-inventory) with
      `GF_PATCH_BRANCH=ai-sdlc/v0.28.1` and the tooling's `build.conf` moved to v0.28.1.

### 5b. GPU and quality (on the box, under a lease)

- [ ] `head_trim_ops_test` (gpu-test preset): the fallback stays bit-equal to its reference
      kernel; the compact draft kernels equal the full-vocabulary ones; graph replay passes. Also
      upstream's `mtp_ops_test`, `select_ops_test` and `attention_ops_test` (#849).
- [ ] **Flags off are a no-op:** with fixed prompts and greedy decoding, the patched v0.28.1
      build's output is byte-identical to the unpatched v0.28.1 build.
- [ ] **Low stays lossless:** `--mtp-draft-vocab-*` greedy output is byte-identical to unpatched
      v0.28.1 (was 72/72).
- [ ] **G quality at the new base:** divergence report on the 100 held-out turns; d ≤ 0.01 (was
      1/100), no pass-through failure, seeded-sampling identity reported (was 33/36).
- [ ] Functional suites from `tests/functional/` for the Flash-Next MTP mode, including
      `long-context` and `cache`, because the verify and prefill paths change. Correctness runs on
      the candidate; per-request timings use the CLAUDE.md margins (5% and 3 ms).

### 5c. Speed retained (paired design: 18 visits, legs S and C, A/A null)

Use the item 149 harness unchanged: 36 held-out turns, a restart per visit, per-turn seeds,
clustered bootstrap. Use exact image IDs; never compare against a rebuild.

| Comparison | What it shows | Pass rule |
| --- | --- | --- |
| **A1**: v0.28.1 unpatched vs `ai-sdlc/v0.19.0` | What upstream gained on its own (#882, #742, …) | Reported only |
| **A2**: v0.28.1 + G vs v0.28.1 unpatched | **Whether the trim still pays on the new base** | ≥ +3% with the interval's lower bound above 0 on **both** S and C, null passing |
| **A3**: v0.28.1 + G vs v0.19.0 + G (`ai-sdlc/v0.19.0-vocab`) | That the merge does not lose absolute speed compared with what we would run today | Point estimate ≥ 0 on both legs, with the lower bound no worse than −2% |
| Phase timers (`--vocab-trim-timing`, one visit) | Where the head's time goes after #882 | Fallback phase d ≲ 0.35 ms per round (was 0.33); head share of a round reported |

If **A2 misses the bar**, the most likely cause is #882 absorbing the discard-aware saving. In
that case:

1. Read the phase timers and `head_trim_rowhist_*`: rows per verify call and flagged rows past the
   kept prefix.
2. Re-run item 150's threshold sweep (0.2, 0.25, 0.3) with the divergence report. Any new
   threshold is exploratory until it is pre-registered.
3. Check that #882's gate (`kMtpGateProbability`) and our fallback threshold do not both act on the
   same rows. A row the gate will discard should never trigger a full-head read.
4. Only then record a "below bar" verdict with its numbers. Keep G off and keep the branch; do not
   relax the bar.

## 6. Moving the deployment (out of scope here)

The pin moves only through the deployment's `fork-pin` step and its own review. This plan ends
with the A2/A3 verdicts. Push nothing until the user has given a dated yes and
`ai-sdlc/leakcheck.sh --range` passes on every pushed branch (FORK.md push checklist).

## 7. Rough effort

| Step | Work |
| --- | --- |
| Port the vocab series (6 conflicted files, 14 hunks against main or fewer against v0.28.1) | 0.5 to 1 day, mostly `engine.cpp` and the head routing after #733/#739 |
| Q4_1 and dump-hidden rebases | Under an hour each |
| 5a offline checks | A few hours |
| 5b and 5c on the box | About 6 to 10 box hours over 2 to 3 leases (A1, A2, A3 at 18 visits each, plus divergence) |
