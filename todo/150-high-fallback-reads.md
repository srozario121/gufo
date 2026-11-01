---
id: 150
title: Fewer and cheaper full-head fallback reads for the High level — measure what the fallback wastes, tune the threshold, then try discard-aware flagging, a two-tier fallback and skipping already-scored rows
status: partial
theme: gufo-engine
contract: legacy
depends_on: [149]
---

## 150. Fewer and cheaper fallback reads

> **Moved here on 2026-11-01** from the ai-sdlc deployment repository (its item 170.2), with the same id, title and
> status. ADR numbers, `make` targets, scripts, files and other items named below are the deployment's unless they
> name this fork. Its evidence is private and is cited in plain text ("private evidence, ai-sdlc item N").

Filed 2026-10-21 at the user's request as "fallback that skips already-scored rows"; **widened the same day** after
item 149's one-hour Pi-like baseline landed and was analysed (the gate this item was filed behind has been met).
It was executed on 2026-10-21 and 2026-10-22 inside item 149's branch (pre-registration addenda A6 to A10) without a separate
plan-review; the outcome is recorded at the end of this file.

### Why

Item 149's High level (fork branch `ai-sdlc/v0.19.0-vocab`, commits `e3234ffb`, `90c251da`, timing `16e07f05`)
scores 48,696 of 248,320 output rows and re-runs a flagged position over the whole matrix when the top probability
over the scored rows is under 0.5.

### Measured (item 149, `vocab-trim.md` sections 8 to 9.1, 2026-10-21; Pi-like held-out turns)

- **Baseline (36 turns, 8 visits):** High is +5.0% decode tok/s; gross trim saving 3.5 ms a round, fallback
  about 1.2 ms (1.9%).
- **Phase timers on the same turns (one visit, 1,853 rounds, 2,205 head calls):** the fallback kernels are
  **1.20 ms of the 2.02 ms head time per round (60%)**; the head is 3.3% of a 60.6 ms round. 68% of head calls flag
  nothing (33 us); 32% pay a full read (3.06 ms for 1 row, 3.16 for 2, 3.22 for 3 or more). 14.4% of scored rows are
  flagged, 0.38 full reads per round.
- Quality at threshold 0.5: greedy divergence 1 of 100 held-out turns; threshold 0.9 and 0.99: 0 of 100 (36.7% and
  50.6% of rows flagged). Seeded sampling: 2 of 36 turns differ, both with all differing tokens inside the set.
- Nothing below threshold 0.5 has been tried.

### Candidates, in the order to try them

1. **Counters first (cheap, decides the rest).** In the fork, count (a) flagged rows where the full-head recompute
   changes the chosen token (greedy argmax; for sampled, where the sampled token would differ), and (b) flagged
   rows that sit in the part of a verification chain that is discarded after the first rejected draft token. Run on
   the 100 held-out turns. If (a) is a few percent of flags the rule fires mostly for nothing; if (b) is large,
   candidate 3 pays.
2. **Threshold sweep below 0.5** (0.2, 0.3, 0.4) with item 149's divergence report (about 10 minutes each) and
   the timing build for reads per round; choose by the quality-versus-reads curve, not by speed alone. Any threshold
   change after seeing the 0.5 data is exploratory until pre-registered (item 149's A4.2 rule).
3. **Discard-aware flagging.** Do not flag (or do not pay for) rows past the first position that cannot be
   accepted: for greedy decoding the first mismatch between the draft token and the compact argmax is known before the
   fallback; for sampled decoding the accept test must run first. Largest candidate saving if (b) is large; also
   the most intricate change to the verification path.
4. **Two-tier fallback.** Re-score flagged rows over the first 98,304 ids (about 1.15 ms a read, held-out miss
   0.16 to 0.19%), escalate to the full matrix only when the top probability is still under the threshold. If most
   flagged rows resolve at the middle tier the average read falls from about 3.06 ms to about 1.5 ms (roughly
   half of the 1.2 ms a round). Needs a quality check of the second decision rule.
5. **Skip the rows already scored** (this item's first form). Each read covers 199,624 rows instead of 248,320
   (-19.6%): about 0.23 ms a round (0.4%) at today's 1.2 ms. The smallest lever, and it breaks bit-identity with the
   current High build (the compact projection and the fallback kernel use different quantisation, `sudot4`, Q8_1).
6. **Not proposed:** a lower-precision copy of the output matrix for the fallback (halves the read but a flagged
   position would no longer be the exact full-head answer).

### Gate and success

- Gate met: the fallback is 1.9% of a round on Pi-like turns (above the 1% line set when this was filed).
- Success for the item: fewer or cheaper reads by a measured amount, the greedy divergence report no worse than
  d = 0.01 at the pre-registered point, the net High speed gain reported with item 149's paired design, an interval
  and the A/A null beside it. Quality wording follows ADR 0012; a candidate that changes a flagged position's logits
  is not called equivalent without the report.

### Out of scope

Changing the essential set, the draft head, or the Low and Medium levels (item 149); the managed pin (a separate item).

### Outcome (2026-10-22; evidence in item 149's `vocab-trim.md` sections 10 to 17)

- **Candidate 1, counters:** at threshold 0.5 only 2 of 2,909 flagged greedy rows (0.07%) change the argmax (0 of 986 sampled); mean unscored mass
  0.003; 43.5% (greedy) and 30.6% (sampled) of flagged chain rows lie beyond the accepted prefix. The fallback mostly fires for nothing.
- **Candidate 2, sweep:** A6.2's rule picked **0.3** (1 of 100 greedy turns diverged at 0.5, 0.4, 0.3; 2 at 0.2). Later: 0.25 diverges on 1 (eligible).
- **Candidate 3, discard-aware flagging:** qualifies (phase d -21% sampled, -39% greedy, no extra diverged turn). Greedy-exact on the device; sampled and
  biased chains defer flagged rows and read only rows the verification loop reaches.
- **Candidate 4, tier-2 fallback:** **fails** (every flagged row escalated, 333 of 333 sampled; d + d2 +7%; pass-through changed). Closed.
- **Candidate 5, skip scored rows:** qualifies (phase d -19%), no extra diverged turn.
- **Final configuration** F = threshold 0.3 + discard-aware + skip-scored; **G** = the same at 0.25. Phase d per round on Pi turns: H (0.5) 1.20 ms
  against F 0.33 ms (-73%); head 3.4% of a round against 1.9%.
- **Speed, the paired design:** single-stream sampled, G **+5.6%** (+3.6 to +7.5, null +0.5%, passes); four concurrent streams, G **+5.7%**
  (+1.6 to +11.0, null -0.7%, passes). F against H (the 0.5 configuration) is not resolved on its own (+0.5% to +1.0% single-stream).
- **Metrics bug found on the way:** the decode-batch path reported the whole step's head counters to every request and the metrics summed them, so
  concurrent per-token counters were inflated by the batch size (2.45x); fork flag `--vocab-trim-count-once` (default off) fixes the accounting.

### What remains

- The fork commits (`ai-sdlc/v0.19.0-vocab`, tip `7d7608a1`) are **not pushed**; a push needs the user's yes and the leak check (ADR 0042).
- Adoption of G (a separate pinning item, since the managed pin never moved here) still needs the Pi task slice, a sampled-quality check at scale
  and the multi-session lazy-fallback test; an ADR if adopted.
- Closed without change: the lower-precision fallback copy (not proposed), and tier-2 (failed).
