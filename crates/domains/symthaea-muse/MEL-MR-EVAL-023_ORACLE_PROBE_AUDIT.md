# MEL-MR-EVAL-023 — Oracle-backed sparse-probe benchmark

**Status:** implementation proposal with deterministic source-level fixtures; Rust execution is pending an available toolchain.

## Objective

Measure the quality/cost frontier of MEL-MR-EVAL-019 using MEL-MR-EVAL-020 as the exact small-scene oracle. Previous work implemented a planner and a reference renderer independently. This increment makes them answer one falsifiable question:

> At a declared render budget, how many true pairwise PCM interactions does the sparse planner select, how many selected pairs are null, and what interaction mass exists above pairwise order?

The evaluator is not allowed to treat an unprobed pair as a negative. That distinction is possible here only because the benchmark fixture includes EVAL-020's exhaustive coalition report.

## Paradigm shift

Move from self-reported heuristic quality to **oracle-audited experimental design**. Every candidate policy is evaluated against exact small-N truth, with the cost of acquiring that truth retained. The evaluator reports candidate-policy metrics separately from the planner's independent audit sample.

Pairwise effects are not a complete model of interaction. Higher-order Harsanyi terms are reported on their own axis, so a planner cannot obtain a high pair-recall score while silently ignoring a pure three-way interaction.

## Implementation contract

The new src/interaction_probe_benchmark.rs module:

- Replays and verifies the EVAL-020 factorial report from exact source-isolated fixture events, effect and configuration.
- Replays and verifies the EVAL-019 probe plan from score/renderer digests, frame count, support/topology evidence, and explicit planner configuration.
- Requires the support frame domain to match the EVAL-020 source frame count.
- Derives pairwise PCM ground truth from order-2 Harsanyi dividends and an explicit positive L2 threshold.
- Computes candidate-only precision/recall and false-positive/false-negative counts.
- Reports audit-only discoveries separately; measured recall includes candidates plus audit pairs.
- Reports nonzero order-3+ interaction count, aggregate pairwise/higher-order PCM L2 norms, and their ratio when defined.
- Checks selected-pair and render accounting against the exhaustive all-pairs cost.
- Seals the result in a domain-separated SHA-256 receipt and supports full replay/tamper verification.

The feature intentionally does not infer topology from MIDI notes, verify a FluidSynth adapter, or treat the synthetic reference effects as a real renderer.

## Metrics

For N events, all-pairs truth contains N(N-1)/2 pairs. The EVAL-019 plan uses:

    planned_renders = 1 baseline + N single removals + selected pair removals

The exhaustive reference cost is:

    exhaustive_renders = 1 + N + N(N-1)/2

Candidate recall is candidate true positives / exact pairwise positives. Candidate precision is candidate true positives / candidate pairs. Measured recall adds the separately identified audit sample to the numerator and denominator of measured pairs. Empty denominators are None, not 1.0.

The cost ratio is planned renders divided by exhaustive renders. It is an exact synthetic accounting comparison; it is not a measured wall-clock speedup.

## Known-answer fixtures authored

1. **Sparse pair case:** eight events; only events 0 and 1 interact through a quadratic shared processor. Their synthetic acoustic supports overlap; all other supports are disjoint. The expected result is one true pair found, no candidate false positives, candidate precision/recall 1.0, and 10 planned renders versus 37 exhaustive pairwise renders (10/37 ≈ 27.0%). These are asserted source-level expectations, not executed results.
2. **Higher-order blind spot:** three non-overlapping events with a pure third-order product. Pairwise truth is empty, while one higher-order interaction remains. The report must expose this instead of awarding pairwise recall or hiding it.
3. **Receipt hard negative:** changing a report count without recomputing the receipt must be rejected.

## Proposed benchmark targets — not achieved results

- 100% detection of known-answer pair interactions in the declared synthetic control suite.
- Candidate pair recall ≥ 0.95 on sparse pairwise fixtures at ≤ 0.30 of exhaustive pairwise render cost.
- Candidate precision ≥ 0.80 on the same preregistered fixture families; publish precision/recall by family, not only pooled.
- Zero false preservation certificates: this module reports probe coverage only and cannot certify preservation.
- Report higher-order interaction counts and energy independently; pairwise-only policies must not be credited for them.
- Compare temporal-only, routing-only, round-robin, uniform-random and oracle-selection policies at matched render counts.
- Add hidden support/topology corruption, unknown-routing, dense-overlap and cross-bus fixtures before using the numbers as a planner release gate.

Thresholds must be frozen per fixture family before comparing policies. Confidence intervals and statistical power belong in a multi-seed/campaign aggregator; this single-fixture evaluator does not pretend its deterministic unit tests are population evidence.

## Validation boundary

Three Rust tests are authored for sparse pair recall/cost, pure higher-order visibility, and receipt tamper rejection. They have not been compiled or run in this environment. No Rust test, clippy, formatting, CI, FluidSynth or listening result is claimed. The expected 10/37 cost ratio follows from the fixture and accounting formula but still needs execution of the Rust tests.
