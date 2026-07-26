# Butlin Indicator Evidence-Tier Design

**Status**: proposal, not yet implemented — written for review before touching `IndicatorStatus`
or the static/runtime blend.
**Scope**: `src/benchmarks/butlin/` (`indicators.rs`, `report.rs`, `ablation.rs`) and the two
Butlin test files.

## Problem this replaces

Two related defects in the current suite (`indicators.rs`), found while responding to issue #7:

1. **`IndicatorStatus::Present` is hardcoded** on every one of the 14 `indicators.push(...)` call
   sites, regardless of the computed `score`. The enum exists but carries no information — nothing
   in the report can currently read as `Partial` or `Absent`.
2. **The `0.6 × static + 0.4 × runtime` blend lets a fully-dead live signal still read as
   present.** A live signal that reads `0.0` still floors out at `0.6 × 0.85 ≈ 0.51` for most
   indicators — high enough to pass under nearly any plausible threshold. A thresholded status
   derived from this blended score would just be hardcoding wearing a thin coat of arithmetic; it
   would not fix the underlying problem.

Underneath both: the suite conflates three genuinely different claims under one score —
*architectural plausibility* (a static, hand-assigned constant reflecting "this mechanism exists
in code and is theoretically the right kind of thing"), *causal-implementation validity* (an
ablation shows disabling the named mechanism moves the named signal), and *functional validity*
(disabling the mechanism also degrades a real downstream competency, not just an internal proxy
metric). Today's single blended `score` cannot represent "we have strong evidence for the first
claim and none for the other two" — it just averages them together.

## Proposed model

Replace `IndicatorStatus`'s current three meaningless variants with an **evidence tier** — what
kind of evidence exists for this indicator, not how good its number looks:

| Tier | Meaning | How it's earned |
|---|---|---|
| `ArchitecturalOnly` | The mechanism exists and is wired into the loop; no live signal has been measured. | Default — `runtime_consciousness: None`, or the field wasn't populated. |
| `Observed` | A live signal was measured (finite, in the field's documented range) but no ablation has been run against it. | A `RuntimeConsciousnessData`/`BehavioralIndicatorSignals` value exists and is finite. |
| `CausallySupported` | An ablation targeting this indicator's mechanism dropped the indicator's own signal (`indicator_dropped == true` in `ablation.rs` terms). | An `AblationResult` row exists for this indicator and its indicator check passed. |
| `FunctionallySupported` | As above, *and* the paired downstream benchmark also degraded (`benchmark_degraded == true`) — the mechanism's removal harmed a real behavioral competency, not just an internal number. | Both `AblationResult` checks passed for this indicator. |
| `NotDemonstrated` | An ablation was attempted and did **not** show the expected effect. | Today's `KNOWN_LIMITATIONS` rows (RPT-2, HOT-1) — moved from a test-file comment into structured report data. |
| `Contradicted` | Evidence points the *wrong* direction — e.g. the diagnostic finding that structural Φ *rose* ~59% under severe network collapse. | Reserved for a signal that moves opposite to its predicted direction under ablation; distinct from `NotDemonstrated` (no effect) because the failure mode and its implications differ. |

`score` stays on `IndicatorEvidence`, but **stops being a blend**. Report two numbers instead of
one:
- `static_score: f64` — the existing hand-assigned architectural constant, unchanged, still useful
  as "how well-motivated is this indicator's design."
- `live_score: Option<f64>` — the raw, unblended probe value when one exists, `None` at
  `ArchitecturalOnly`.

No single number is asked to represent both "this is a well-designed indicator" and "this
indicator's live behavior looks good" at once. A caller who wants one number for a dashboard can
choose which to use and why; the suite itself stops deciding that for them by blending.

`GWT-1` (specialization_fraction, an aggregate derived from the *other* 13 signals rather than an
independent probe) gets a separate boolean-ish annotation — `derived_from: Option<Vec<&'static
str>>` — orthogonal to the tier. Folding "this is derived" into the tier ladder would lose
information; it's a caveat on top of whatever tier GWT-1 otherwise earns, not a rung of its own.

## What has to change to make this real, not cosmetic

The two pipelines that would need to feed the same report:

- `indicators.rs::evaluate()` — cheap, static-config-only today (`runtime_consciousness: None`),
  the thing `butlin_regression.rs` calls on every CI run. This alone can only ever produce
  `ArchitecturalOnly` or `Observed`.
- `ablation.rs::run_ablation_matrix()` — expensive, drives a real `CognitiveLoopService`, produces
  `AblationResult` per indicator. This is the *only* thing that can produce `CausallySupported` /
  `FunctionallySupported` / `NotDemonstrated` / `Contradicted`.

These are disconnected today — `evaluate()` never calls ablation code, and ablation results only
ever reach a test's `eprintln!` output, not the `ButlinReport` itself. Making tier real means one
of:

1. `evaluate()` takes an optional `&[AblationResult]` and upgrades tiers where a matching row
   exists, or
2. a separate `annotate_with_ablation_results(report, results)` pass that merges post-hoc.

(2) seems like the better fit: it preserves the property `butlin_regression.rs` currently depends
on — that the cheap gate never needs `symthaea-backend` — and matches the existing cost structure
(`butlin_regression.rs` cheap/always-on vs. `butlin_ablation_integration.rs` expensive/gated). Under
this split, `butlin_regression.rs`'s job stays almost exactly what it is today (assert
`static_score` floors, nothing about tier). `butlin_ablation_integration.rs`'s job changes from
"pass/fail with named carve-outs" to "produce the tier data" — the `KNOWN_LIMITATIONS` list becomes
real report output (`NotDemonstrated`) instead of a comment justifying why a test didn't fail.

The composite `mean_quality_score` metric (`butlin_composite_mean_meets_floor` in
`butlin_regression.rs`) would need rethinking too, once there's no single blended per-indicator
score to average — a distribution across tiers ("N functionally-supported, M causally-supported,
...") is more honest than one scalar mean, but changes what that regression test asserts. Flagging
this as a real consequence, not deciding it here.

## Open questions for review

1. Does `Observed` need a stronger bar than "finite and in range"? A signal could be finite but
   still not moving in response to anything (a frozen constant, matching the still-open
   `prediction_error` investigation) — `Observed` as defined wouldn't catch that. Possible fix:
   `Observed` requires the signal to vary across at least two distinct measurement calls, not just
   exist once.
2. Should `NotDemonstrated` and `Contradicted` block the regression gate, or only annotate it? A
   named, disclosed limitation isn't necessarily a CI failure — issue #7's ask was for a gate that
   catches an undetected *regression*, not one that requires every indicator to reach the top tier.
3. Where does this leave the existing `IndicatorStatus::Partial`/`Absent` variants — deleted
   outright, or kept as a deprecated alias during a transition window? Given this is pre-1.0
   internal benchmark code with a small number of call sites, deleting outright seems cleaner, but
   confirming before doing it.

Not implementing any of this until the tier model and the `evaluate()`/ablation plumbing split
above are agreed.
