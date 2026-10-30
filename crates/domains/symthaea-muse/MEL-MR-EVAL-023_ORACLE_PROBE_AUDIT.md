# MEL-MR-EVAL-023 — Oracle-backed sparse-probe benchmark

**Status:** implementation proposal with deterministic source-level fixtures; Rust execution is pending an available toolchain.

## Objective

Measure the quality/cost frontier of MEL-MR-EVAL-019 using MEL-MR-EVAL-020 as the exact small-scene oracle. Previous work implemented a planner and a reference renderer independently. This increment makes them answer one falsifiable question:

> At a declared render budget, how many non-null contextual pair effects does the sparse planner select, how many selected pairs are null, and what interaction mass exists above pairwise order?

The evaluator is not allowed to treat an unprobed pair as a negative. That distinction is possible here only because the benchmark fixture includes EVAL-020's exhaustive coalition report.

## Paradigm shift

Move from self-reported heuristic quality to **oracle-audited experimental design**. Every candidate policy is evaluated against exact small-N truth, with the cost of acquiring that truth retained. The evaluator reports candidate-policy metrics separately from the planner's independent audit sample.

Pairwise probe effects are contextual, not an unconditional decomposition by interaction order. A pure three-way interaction can produce nonzero contextual four-coalition differences when the third event remains present, even though every unconditional order-2 Harsanyi dividend is zero. The evaluator reports both the contextual probe target and the global Harsanyi order decomposition.

## Implementation contract

The new src/interaction_probe_benchmark.rs module:

- Replays and verifies the EVAL-020 factorial report from exact source-isolated fixture events, effect and configuration.
- Replays and verifies the EVAL-019 probe plan from score/renderer digests, frame count, support/topology evidence, and explicit planner configuration.
- Requires the support frame domain to match the EVAL-020 source frame count.
- Derives the probe target from EVAL-018's *contextual four-coalition finite difference*: for each pair A/B, retain all other events and compare full, without A, without B, and without both. This makes the evaluator measure the same contextual effect that a real EVAL-018 probe measures.
- Computes candidate-only precision/recall and false-positive/false-negative counts.
- Reports audit-only discoveries separately; selected-plan recall includes candidate pairs plus audit pairs.
- Separately reports contextual pair-effect magnitude, unconditional global order-2 Harsanyi structure, and order-3+ Harsanyi structure. These are not interchangeable: a three-way effect can induce nonzero contextual pair probes while its global order-2 Harsanyi terms remain exactly zero.
- Checks selected-pair and render accounting against the exhaustive all-pairs cost.
- Seals the result in a domain-separated SHA-256 receipt and supports full replay/tamper verification.

The feature intentionally does not infer topology from MIDI notes, verify a FluidSynth adapter, or treat the synthetic reference effects as a real renderer.

## Metrics

For N events, the contextual pairwise reference contains N(N-1)/2 pair tests. For pair A/B, with all other events C retained, the PCM residual is:

    J_AB|C = Y_all - Y_(all without A) - Y_(all without B) + Y_(all without A and B)

The global Harsanyi decomposition remains a separate diagnostic: order-2 terms are unconditional pair structure, and order-3+ terms capture higher-order coalition structure. The EVAL-019 plan uses:

    planned_renders = 1 baseline + N single removals + selected pair removals

The exhaustive reference cost is:

    exhaustive_renders = 1 + N + N(N-1)/2

Candidate recall is candidate true positives / exact contextual-positive pairs. Candidate precision is candidate true positives / candidate pairs. Selected-plan recall combines candidate-policy pairs and the separately identified audit sample. Empty denominators are None, not 1.0.

The cost ratio is planned renders divided by exhaustive renders. Pair counts describe selected probes, not completed renderer measurements. It is an exact synthetic accounting comparison; it is not a measured wall-clock speedup.

Three separately versioned positive L2 floors govern contextual pair positives, global order-2 Harsanyi counts, and higher-order Harsanyi counts. The first two must not be conflated: the same fixture can have a contextual pair effect above its threshold while its global order-2 diagnostic is below its own threshold. Thresholds are part of the replay receipt.

## Known-answer fixtures authored

1. **Sparse pair case:** eight events; only events 0 and 1 interact through a quadratic shared processor. Their nonzero source samples exactly match their overlapping synthetic acoustic supports; all other event supports are disjoint. The expected result is one true pair found, no candidate false positives, candidate precision/recall 1.0, and 10 planned renders versus 37 exhaustive pairwise renders (10/37 ≈ 27.0%). These are asserted source-level expectations, not executed results.
2. **Context versus order:** three co-supported events with a pure third-order product. Every contextual pair finite difference is nonzero while all unconditional order-2 Harsanyi dividends are zero and the order-3 dividend is nonzero. The planner should select all three pair probes; the report must show contextual recall separately from the global interaction-order decomposition.
3. **Support-omission audit:** an intentionally understated support map hides a real contextual interaction from the candidate policy. The independent audit sample should detect it; candidate recall remains zero while selected-plan recall becomes one.
4. **Receipt hard negative:** changing a report count without recomputing the receipt must be rejected.

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

Five Rust tests are authored for sparse pair recall/cost, an independent audit catching a support omission, the distinction between contextual pair effects and a pure third-order Harsanyi term, independent contextual/global thresholds, and receipt tamper rejection. They have not been compiled or run in this environment. No Rust test, clippy, formatting, CI, FluidSynth or listening result is claimed. The expected 10/37 cost ratio follows from the fixture and accounting formula but still needs execution of the Rust tests.
