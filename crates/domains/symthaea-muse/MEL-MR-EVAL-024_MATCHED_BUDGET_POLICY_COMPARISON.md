# MEL-MR-EVAL-024 — Matched-budget probe-policy comparison

**Status:** executable comparison design with deterministic synthetic fixtures; Rust execution is pending an available toolchain.

## Objective

Compare MEL-MR-EVAL-019 against simple probe-selection policies using the same source events and replay-verified MEL-MR-EVAL-020 factorial oracle.

The scientific question is whether structural information (acoustic-support overlap and renderer routing domains) improves contextual interaction discovery at the same actual pair-render cost, rather than whether one hand-picked EVAL-019 fixture can be solved.

## Paradigm shift

Evaluate the **experimental-design policy** rather than just the downstream edit. Publish seed-level recall-versus-render-cost observations, compare with simple baselines, and retain a non-deployable oracle top-K ceiling.

## Frame-domain contract

The benchmark has two distinct frame domains. Source-isolated event arrays and the EVAL-020 `input_frame_count` remain in the source/input domain. `EventInteractionSupport` intervals and the planner's `total_frames` are in the rendered PCM/output domain. The support horizon must equal EVAL-020's `output_frame_count`, including FIR/effect tails; interval ends are exclusive. A support interval is never clipped to the source duration. A real-render adapter must additionally bind frame origin, resampling and latency compensation.

This distinction is required for a 64-frame source whose fixed FIR output is 67 frames: support in frames 64–67 is valid rendered evidence, not an out-of-range source event.

A policy's requested budget is not its measured cost. Temporal-only and routing-only selection can legitimately return fewer pairs than requested. The evaluator records both the requested budget and selected pair count, then computes cost from the actual selected probes.

## Policies

- **NoProbe:** baseline + event-removal setup, no pair probes.
- **UniformRandom:** deterministic, seed-specific sampling without replacement over all possible event pairs.
- **TemporalOnly:** select pairs with intersecting verified acoustic-support intervals, ranked by maximum overlap.
- **RoutingOnly:** select pairs sharing a known routing domain. If any event has incomplete routing, broaden the candidate scope to every possible pair rather than pruning.
- **Eval019:** call the actual EVAL-019 planner with the same support/topology inputs and a zero audit budget.
- **OracleTopK:** select contextual-positive pairs ranked by exact EVAL-020 four-coalition PCM effect magnitude. This is an unattainable upper bound for deployable policies.
- **Exhaustive:** all possible pairs.

All pair-positive labels use the contextual finite difference that EVAL-018 measures with all other source events retained. They are not confused with unconditional global order-2 Harsanyi terms. The report records global order-2 and higher-order Harsanyi counts separately.

## Cost and metrics

For N source events and P = N(N-1)/2 possible pairs:

    total_render_count = 1 baseline + N single-removal renders + selected pair renders
    exhaustive_render_count = 1 + N + P

Each policy/seed/budget row reports:
- requested budget and actual selected pair count;
- true/false positives and false negatives against contextual factorial truth;
- precision and recall (undefined denominators are None);
- actual render count and fraction of exhaustive cost;
- unused budget.

The comparator also emits a recall-versus-actual-render-cost curve summary per policy and seed. Trapezoidal area is calculated only over observed cost points; there is no extrapolation to an unobserved range. Multiple requested budgets that underfill to the same actual cost are averaged at that x-coordinate, not cherry-picked. Cost to 95% contextual recall is reported when observed.

Seed-level rows are not independent fixture replications. This first implementation deliberately does not manufacture confidence intervals from repeated seeds on one scene. The campaign layer must pair policies on the same independently generated fixture families, then bootstrap at the fixture/composition cluster level.

## Known-answer fixture assertions authored

1. **Sparse pair:** eight event signals with only the first two sharing nonzero sample support. One contextual-positive pair is expected among 28 possible pairs.
2. **Matched budget:** at budget one, EVAL-019 and temporal-only select the true pair; routing-only underfills to zero because there are no shared routing domains; oracle top-K selects the true pair.
3. **Render accounting:** one selected pair costs 10 renders against 37 exhaustive renders. Exhaustive reaches a cost fraction of 1.0.
4. **Random reproducibility:** each seed produces deterministic nested prefixes as budgets increase; the comparison report replays exactly.
5. **Unknown routing:** any incomplete routing evidence causes routing-only selection to broaden rather than prune.
6. **Receipt hard negative:** a modified row count without a recomputed receipt is rejected.

These are source-level assertions, not executed benchmark results.

## Qualification targets — targets, not results

- EVAL-019 union + stratified round-robin should outperform uniform random on paired recall at equal actual pair-render cost across held-out sparse and clustered families.
- Candidate recall at least 0.95 on preregistered sparse pairwise families at no more than 0.30 of exhaustive pairwise render cost.
- Candidate precision at least 0.80 on those same families.
- Publish worst-family results, not only the pooled average.
- No unknown-routing case may be treated as independent.
- Publish the oracle ceiling and the gap to that ceiling.
- Hold out at least one interaction family and one routing topology from development.
- Human-perceptual superiority claims require EVAL-021 listening validation; synthetic interaction-discovery wins are not product-quality wins.

## Validation boundary

Five Rust tests are authored. They have not been compiled or executed in this environment. No CI pass, FluidSynth result, measured runtime speedup, confidence interval, or superiority over external systems is claimed. The evaluator's exact replay is limited to synthetic factorial scenes with 2–8 source events.
