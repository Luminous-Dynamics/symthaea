# Phase 6 Session 1b — kNN diagnostic: NULL result

**Status:** complete. 2026-04-18.
**Go/no-go decision:** **stop.** The cheap-encoder learned-cascade direction is empirically dead.

## Setup

Session 1 showed a +1.4% cluster separation (within-accept mean cosine
> between mean cosine). That's directional evidence but doesn't say
the signatures actually *predict* Lake outcome. Session 1b tests
that directly: leave-one-out kNN classifier over signatures, k=3,
majority vote of nearest neighbors.

Pass criterion (pre-declared): kNN accuracy must beat majority-class
baseline (always predict "accepted" = 22/31 = 71.0%).

## Result

| Metric | Value |
|--------|-------|
| kNN accuracy (leave-one-out, k=3) | 22/31 = **71.0%** |
| Majority-class baseline | 22/31 = **71.0%** |
| Lift over baseline | **+0.0 pp** |
| Accepted recall | 18/22 = 81.8% |
| Rejected recall | 4/9 = 44.4% |

**Lift: zero.** kNN matches the constant-prediction baseline exactly.

## Interpretation

The +1.4% cluster signal from Session 1 is real but too weak to
drive a learned classifier. The classifier can identify "this goal
looks like a normal accepted one" (82% recall on accepts) but fails
on rejected goals (44% recall — barely above coin flip).

This matches what we'd expect from the encoder: token-bag with
positional binding captures surface structure (presence of `∀`,
`ℝ`, `ℤ`, arithmetic operators, ...) but NOT the semantic shape
that determines whether a goal is a Pattern-B field problem, a
Pattern-D cubic system, or a straightforward linear problem. Those
distinctions require understanding the *structure* of the goal, not
its tokens.

## Decision

**Session 2 (cascade tournament) is NOT run.** It would build
infrastructure around a non-informative signal. The honest close-out
for Phase 6's learned-cascade direction is:

- The bridge works (96.9% curated, 44% auto-ingested).
- The cognitive loop works (6/13 on invariant discovery).
- **Connecting them via surface-token HDC does not work.**

## Paths forward

Ranked by marginal value per LOC:

1. **Explicit mechanisms for named failure patterns.** Pattern B's
   nonzero-witness extractor (`¬n = 3` → `n - 3 ≠ 0` via
   `sub_ne_zero.mpr`) closes `_181`/`_251`/`_267` (3 fixtures).
   Pattern D's RREF solver closes `_338` (1 fixture). Each is
   ~30-100 LOC. No learned selection needed; the mechanism either
   applies or it doesn't.
2. **Richer encoder.** Swap the token-bag for something semantically
   structured — the AST itself (tree-structured HDC) or the cognitive
   loop's `wisdom_hv` (HDC→CfC→Φ-derived signature). If a richer
   encoder shows kNN accuracy >80%, revisit the learned-cascade
   direction. Heavy dependency — would require bringing the main
   `symthaea` crate into the bridge.
3. **Publish the null.** Phase 6 produced a cleanly-documented
   negative result. That's a paper contribution in its own right:
   "surface-token HDC signatures do not discriminate Mathlib-tactic
   accept/reject on miniF2F." Decision for the user: put it in a
   paper, or let it sit in the repo as an artifact.

## Reproduction

```bash
cargo run -p symthaea-lean-bridge --example phase6_knn_diagnostic
```

Output in `docs/phase6-results/session1b_knn_diagnostic.csv` (per-goal
prediction with top-3 neighbor labels).
