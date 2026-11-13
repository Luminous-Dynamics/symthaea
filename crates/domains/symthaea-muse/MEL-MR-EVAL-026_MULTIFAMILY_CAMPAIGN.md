# MEL-MR-EVAL-026 — Paired Multi-Family Interaction Probe Campaign

**Date:** 13 November 2026  
**Status:** first campaign layer and deterministic fixture generator authored; Rust execution pending exact-head qualification.  
**Tracking issue:** [#9240](https://github.com/Luminous-Dynamics/symthaea/issues/9240)  
**Branch:** \`research/mel-mr-eval-026-multifamily-campaign-20261113\`

## Objective

Turn the EVAL-024 single-fixture policy comparator into a preregistered multi-composition campaign with defensible uncertainty estimates, explicit negative controls and held-out generalization.

This module composes, and does not duplicate:

- EVAL-017 / PR #9128: replayable paired-render single-event influence.
- EVAL-018 / PR #9133: four-coalition contextual pair interactions.
- EVAL-019 / PR #9151: sparse, budgeted candidate selection.
- EVAL-020 / PR #9155: exhaustive small-N factorial oracle and Harsanyi decomposition.
- EVAL-023 / PR #9213: sparse-probe audit against exact factorial truth.
- EVAL-024 / PR #9217: matched-actual-render-cost policy comparison.
- EVAL-025 / PR #9218: replay-verified FluidSynth support/routing adapter (partial routing; incomplete topology must still widen).

No new renderer, event-influence estimator, probe-selection policy, interaction oracle or perceptual metric is introduced.

## Paradigm shift: benchmark the experimental design, not a hand-picked fixture

A sparse probe planner is useful only if its advantages survive changes in event density, acoustic support, routing topology, interaction order and source composition.

The campaign therefore pairs every policy on identical fixture, seed and requested-budget blocks. It reports actual selected-pair count and actual render cost; a policy that underfills its requested budget is not credited with spending work it did not perform.

Crucially, the independent resampling unit is the source-composition lineage:

- Different seeds on one composition are repeated policy randomness.
- Different budgets on one composition are repeated operating points.
- Pairs, windows, channels and frames are not independent compositions.
- Fixture variants derived from one composition lineage remain one statistical cluster.

The module reduces repeated rows to a per-lineage value before bootstrap resampling. It reports a point estimate even when there are too few independent lineages, but withholds a confidence interval when fewer than two clusters are available.

## Implemented first tranche

The new \`src/interaction_probe_campaign.rs\` module provides:

1. **Deterministic fixture generation** using a seed and source-composition lineage digest.
2. **Separate source oracle and adapter inputs.** Synthetic source signals/effect parameters determine exact EVAL-020 factorial truth. Event support and routing metadata are a separate input that can be deliberately corrupted for negative controls without mutating oracle signals.
3. **Ten fixture families:** additive null, sparse pairwise, hard clipping, pure higher-order three-way, linear FIR tail, dense overlap, shared routing with disjoint time support, understated support, incomplete routing, and mixed salience.
4. **Preregistered train/held-out separation** by family and routing topology.
5. **Lineage leakage checks:** a source-composition digest cannot appear in both development and held-out splits.
6. **Minimum independent composition counts** per family/split stratum.
7. **Paired EVAL-024 runs** over identical seeds and budgets for every fixture.
8. **Equal-actual-cost contrasts:** EVAL-019 is compared with UniformRandom and TemporalOnly only when both policies have the same exact total render count for the same fixture, seed and requested budget.
9. **Composition-clustered bootstrap intervals** for paired recall deltas, with deterministic resampling seed and explicit matched/unmatched block counts.
10. **Worst-family and routing-stratified diagnostics:** precision, recall, actual cost, false discoveries and missed interactions are not hidden in a single composite score.
11. **Content-addressed replay:** the report binds the complete fixture input set, configuration, per-fixture factorial and comparison receipts, aggregates, paired contrasts and confidence intervals. Verification rebuilds the campaign from exact inputs.

The first fixture generator is synthetic mathematical ground truth. Its routing and support declarations are not observations of a real DAW or renderer. The FluidSynth adapter and real-audio/human studies remain separate evidence requirements.

## Frozen experimental protocol

### Fixture families

| Family | Oracle behavior | Adapter/control question |
|---|---|---|
| Additive null | Linear sum | Does the planner avoid manufacturing interactions? |
| Sparse pairwise | Shared polynomial pair effect | Can support structure find sparse positives? |
| Hard clipping | Shared hard limiter | Does dense nonlinear coupling defeat temporal-only assumptions? |
| Pure higher-order | Explicit three-way product | Are contextual pair effects kept distinct from global order-2 Harsanyi truth? |
| Linear FIR tail | Linear convolution | Does tail support avoid false nonlinear attribution? |
| Dense overlap | Many overlapping events through shared nonlinear effect | Does a dense cluster monopolize the candidate budget? |
| Shared routing / disjoint time | Events share a declared bus but do not overlap | Is routing treated as a candidate prior rather than proof of interaction? |
| Understated support | Oracle source overlap remains unchanged; declared support is shortened | Does the planner/audit expose false negatives caused by support corruption? |
| Incomplete routing | Oracle source/effect is unchanged; route completeness is false | Does the planner widen scope rather than prune unknown routes? |
| Mixed salience | Unequal event amplitudes | Does performance vary with source salience and density? |

The source signals and effect parameters are generated before support/topology metadata. In the UnderstatedSupport family, only the declared support intervals are shortened; the exact source signals and factorial oracle remain unchanged. In the IncompleteRouting family, routing completeness is false and no independence inference is permitted.

### Splits and leakage

- Freeze source-composition lineage digests before campaign execution.
- Development and held-out lineages must be disjoint.
- At least one interaction family and one routing topology are held out in the default protocol.
- Each family/split stratum requires at least two independent composition lineages; the publication protocol should use materially more.
- Budgets, seeds, effect thresholds, metric reducers and fixture generator versions are included in the report receipt.
- Held-out results are read-only. They cannot be used to tune support thresholds, budgets, fixture amplitudes or policy selection.

### Paired policy comparison

Every fixture is evaluated with the same seed and requested-budget grid through EVAL-024. The campaign retains the EVAL-024 per-policy rows and reports paired contrasts only at identical actual render count.

Primary comparisons:
- EVAL-019 versus seeded UniformRandom.
- EVAL-019 versus TemporalOnly.

Secondary diagnostics:
- RoutingOnly, NoProbe, OracleTopK and Exhaustive retain their EVAL-024 meanings.
- OracleTopK is an unattainable ceiling, not a deployable policy.
- Exhaustive is a finite-scene reference, not a scalable system.

### Statistical analysis

For each split, family, policy and requested budget:

1. Average repeated seeds and fixture variants within each source-composition lineage.
2. Compute the family-level mean by giving each independent lineage equal weight.
3. For policy contrasts, match identical fixture, seed, requested budget and actual render count.
4. Average matched paired recall differences within each composition lineage.
5. Bootstrap composition-lineage clusters, not seed rows, pairs or windows.
6. Report matched composition count, matched block count and unmatched/undefined block count alongside the 95% interval.
7. If fewer than two independent composition clusters remain, publish the point estimate only and omit the interval.

The campaign emits family-specific results and worst-family performance. It does not treat confidence intervals over a single synthetic scene's seeds as population-level uncertainty.

## Prospective qualification targets (not achieved results)

The targets in #9240 remain research objectives:

- EVAL-019 candidate-pair recall at least 0.95 at no more than 0.30 of exhaustive pairwise render cost on the preregistered sparse family suite.
- Candidate precision at least 0.80 on that same suite.
- EVAL-019 should outperform UniformRandom on paired recall at matched actual render cost across held-out sparse and clustered families.
- Zero false-independent decisions under incomplete-routing and support-corruption controls.
- Publish 95% composition-clustered intervals, per-family metrics and worst-family bounds.
- No tuning on held-out families or routing topologies.
- Report actual render count, p50/p95 runtime and memory when the live campaign runner is available.

These are target thresholds, not measured results. Passing the synthetic campaign does not prove FluidSynth correctness, real-world support completeness, perceptual edit quality or human collaboration.

## Industry and benchmark context

The external benchmarks measure different axes:

- **SMEBench** (Cervera et al., 27 October 2026) contains 3,600 controlled paired music-edit examples across twelve operations plus a 416-pair real-audio complement. It evaluates paired target agreement, source preservation and text compatibility; it is not an event-interaction ground-truth oracle. https://arxiv.org/abs/2610.36715
- **ICASSP 2027 MMAE Audio Editing Challenge** has a 2,000-example public development set with 17,741 atomic rubrics and separate 500-example hidden Single Model and Agent tracks. It reports IFR, CR and EMR; its official leaderboard is ranked primarily by EMR. https://audio-editing-challenge.github.io/
- The ICASSP Agent Track explicitly allows planning, multiple tools/models, intermediate inspection and refinement. A planner's novelty is not the number of tools used; its value must be demonstrated through matched-cost reliability, preservation and exact execution.

As of 13 November 2026, the official challenge page says hidden test sets and the SDK were released on 10 November. Participant IDs and submission keys are distributed privately. Registration closed on 8 October, the final submission deadline is 25 November, and the official leaderboard requires its SDK and credentials. Do not claim entry eligibility or hidden-set results without authorized team status and the official run. The public 2,000-example development evaluation remains available independently.

## Tests authored

Nine deterministic unit tests are authored:
- generator reproducibility and oracle/support separation;
- paired multi-family campaign construction and replay;
- composition-lineage split leakage rejection;
- held-out split enforcement;
- same-actual-cost contrast accounting;
- no interval from fewer than two independent clusters;
- receipt tamper rejection;
- pure higher-order truth kept separate from global order-2 terms;
- incomplete-routing negative-control classification.

They are not reported as executed. No Rust compile, formatting, Clippy, CI pass, benchmark score, confidence interval or SOTA improvement is claimed until exact-head qualification completes.

## Run

\`\`\`sh
cargo fmt --manifest-path crates/domains/symthaea-muse/Cargo.toml -- --check
cargo test --locked --manifest-path crates/domains/symthaea-muse/Cargo.toml --features theory --lib interaction_probe_campaign::tests
cargo test --locked --manifest-path crates/domains/symthaea-muse/Cargo.toml --features theory --lib probe_policy_comparison::tests
cargo clippy --locked --manifest-path crates/domains/symthaea-muse/Cargo.toml --features theory --lib --tests -- -D warnings
\`\`\`
