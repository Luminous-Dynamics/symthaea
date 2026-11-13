# MEL-MR-EVAL-026 — Paired Multi-Family Probe Campaign and Clustered Analysis

**Tracking issue:** [#9240](https://github.com/Luminous-Dynamics/symthaea/issues/9240)  
**Implementation:** \`src/probe_policy_campaign.rs\`  
**Status:** deterministic synthetic campaign and analysis harness authored 13 November 2026; tests/CI pending.

## Why this tranche exists

EVAL-024 compares sparse-probe policies on one synthetic scene with repeated policy seeds and budgets. Repeated seeds on one composition are not independent composition replications. A policy that wins one convenient topology may fail on a different interaction family, event density, support geometry or routing completeness.

EVAL-026 creates independently generated source compositions, computes exact EVAL-020 factorial truth for each, derives a separate EVAL-019 support/routing view, and runs every EVAL-024 policy on the same composition, seeds and requested budget schedule.

The inferential unit is the **composition cluster**. Seeds and budgets are repeated measurements within a composition, never additional independent scenes.

## Paradigm

Evaluate probe policies as controlled experimental-design policies, not merely as downstream edit tools.

Every fixture has two separate evidence paths:

\`\`\`
synthetic source events
  -> EVAL-020 factorial oracle truth
  -> contextual pair labels + global Harsanyi structure

same source events
  -> separately declared support/routing adapter view
  -> EVAL-019 / EVAL-024 probe policies
  -> selected pair probes
\`\`\`

Support corruption changes only the planner input. It must never rewrite oracle source signals, coalition renders or factorial truth.

The campaign's default fixed split holds out:
- **PureThirdOrder** as an interaction-family holdout.
- **IncompleteRouting** as a routing-topology holdout.

These splits are determined by the versioned family map, not chosen after looking at the results. The default development families include additive null, sparse pairwise, saturation, fixed FIR tail, dense overlap, clustered overlap, shared-route/temporally-disjoint, and understated-support controls.

## Implemented

\`probe_policy_campaign.rs\` provides:

- Ten deterministic synthetic interaction/support families.
- Seven independent composition variants per family by default, covering event counts 2..8 (the pure-third-order family uses 3..8).
- Bounded deterministic event signals and exact source/effect/support digests.
- Separate source-oracle truth and potentially corrupted support/routing observations.
- Exact EVAL-020 factorial report generation per composition.
- Calls the existing EVAL-024 comparator for every fixture; it does not reimplement or fork the policy algorithms.
- Carries all per-policy/per-seed/per-budget rows and recall-cost curve summaries into the campaign report.
- Paired EVAL-019 versus uniform-random and temporal-only recall comparisons at the **same actual total render count**. Requested budgets with different actual costs are explicitly unmatched.
- Composition-equal weighting: first average the matched seed/budget deltas within each composition, then average composition means.
- Deterministic composition-cluster bootstrap intervals; no row-, seed-, pair- or frame-level pseudo-replication.
- Separate null-scene false-discovery summaries, preserving the distinction between undefined recall on zero-positive scenes and zero recall.
- Domain-separated SHA-256 campaign receipt and full replay by regenerating the frozen fixture matrix, oracle reports, policy rows and summaries.

The campaign defaults to 70 composition fixtures (10 families × 7 compositions), three policy seeds and seven pair budgets. The module's tests use a smaller four-family matrix while retaining seven independent compositions per included family.

## Important measurement semantics

- A policy comparison is called matched-cost only when candidate and baseline have equal **actual** render counts for the same fixture, seed and requested budget.
- An underfilled policy is not credited with the requested budget.
- Undefined recall on a zero-positive fixture is not coerced to zero; null-scene false discoveries are reported separately.
- Contextual four-coalition pair effects remain distinct from unconditional global order-2 and order-3+ Harsanyi structure.
- Incomplete routing is never treated as proof of independence.
- The FIR family uses the oracle's rendered output frame count (including the convolution tail) as the support coordinate horizon.
- A local deterministic bootstrap interval is an uncertainty summary for this frozen synthetic campaign, not a universal safety bound. Multi-family or multiple-comparison claims still require a preregistered primary contrast and independent held-out replication.

## Proposed qualification targets — not results

| Measure | Prospective target |
|---|---:|
| Sparse-family candidate recall | ≥0.95 |
| Sparse-family candidate precision | ≥0.80 |
| Actual pair-render cost at that recall | ≤0.30 of exhaustive pair cost |
| Incomplete-routing false-independent classification | 0 |
| Held-out interaction-family and routing-topology results | Report separately; no retuning |
| Cluster uncertainty | 95% composition-cluster bootstrap interval |
| Generalization | Worst-family performance and oracle-ceiling gap published |

The code preserves all raw policy rows and per-composition clusters so targets can be evaluated without pooling repeated seeds as independent observations.

## Tests authored

Seven tests cover:
1. Event-count/family coverage and the two frozen holdouts.
2. Paired same-composition / same-seed comparisons and cluster-bootstrap output.
3. Support-view corruption without changing the factorial oracle input truth.
4. Incomplete-routing holdout behavior.
5. Full campaign replay and receipt-tamper rejection.
6. Duplicate seed, budget and family rejection.

The tests have not yet been compiled or executed. No 0.95 recall, 0.30 cost, confidence interval, FluidSynth result, perceptual-quality result or SOTA improvement is claimed.

## External and product benchmark boundaries

This campaign is a synthetic causal-interaction discovery benchmark. It is not a substitute for:
- SMEBench paired-target music-edit evaluation;
- MMAE/ICASSP 2027 instruction-following, consistency and exact-match scoring;
- MEL-MR-EVAL-021 blinded listener calibration;
- held-out real FluidSynth/SoundFont routing experiments.

A synthetic probe-policy win does not establish improved music editing or listener preference. Product claims require the corresponding real-renderer and human evidence.
