# MACE Pairwise Evidence Interaction Benchmark Extension v0.1

**Status:** evaluator implementation authored; exact-head Rust 1.96 formatting, compile and negative-control run pending. This extension is research-only and does not change the existing v1/v2/v3 score schemas.

## Why leave-one-out is not enough

The current MACE evaluator estimates the effect of removing one returned evidence item under a frozen downstream policy. That is useful, but it cannot distinguish:

- **Redundancy:** two independently sufficient records each preserve the same decision when the other is removed.
- **Synergy:** neither record is sufficient alone, but the pair jointly changes the action to the lower-loss choice.
- **Conflict:** a pair causes a worse action or unresolved contradiction.
- **Irrelevance:** removing one or both records leaves the policy output and loss unchanged.

A retrieval system can therefore have good individual leave-one-out scores while hiding redundant storage, or poor individual scores while missing a crucial complementary pair. Pairwise intervention makes these interactions measurable without pretending that individual contribution is additive.

## Estimator

For a frozen candidate episode, policy digest and decision seed, record:
- (L(arnothing)): loss with the full retrieved evidence set;
- (L(-i)): loss after withholding exactly evidence item (i);
- (L(-i,-j)): loss after withholding exactly items (i) and (j).

The pair interaction is:

[
I_{ij}=L(-i,-j)-L(-i)-L(-j)+L(arnothing)
]

Interpretation:
- (I_{ij}>0): joint removal is more harmful than the sum of individual removals (complementarity / synergy);
- (I_{ij}<0): joint removal is less harmful than the sum of individual removals (substitution / redundancy);
- (I_{ij}=0): no second-order interaction under this policy and episode.

This is a finite-difference interaction for a fixed deterministic decision policy. It is not a Shapley value, causal effect outside the evaluated policy, or a guarantee of semantic necessity.

## Execution contract

The new `score_pairwise_interactions` reference requires:
1. A complete leave-one-out intervention for every returned evidence ID.
2. Exactly one pairwise intervention for every unordered pair in the retrieved set.
3. Exact identity of system, episode, downstream policy digest and decision seed across every rerun.
4. Each single intervention removes exactly one ID from both retrieval and decision-support sets.
5. Each pair intervention removes exactly its two named IDs from both sets.
6. Every output remains validated against evaluator-owned evidence IDs and action losses.

The evaluator calculates the interaction in signed integer micro-loss units. It does not expose evaluator losses to candidate systems. The complete pair design has (O(n^2)) reruns for (n) returned evidence items; benchmark harnesses should cap the evidence set and report intervention compute/latency.

## Qualification controls

The focused tests include:
- a constructed redundant/complementary pair where both single removals preserve the optimal action but removing both increases loss;
- incomplete pair sets;
- duplicate or reversed duplicate pairs;
- missing or identical pair targets;
- wrong episode/system/policy/seed;
- outputs that retain a removed item or alter unrelated evidence.

The first known-answer example is deliberately simple: each single removal preserves the correct action, while removing both causes abstention. Its interaction is exactly +2,000,000 micro-loss units. This is a test fixture, not a measured system result.

## Research metrics to publish

Per episode and aggregated by independent world cluster:
- distribution of pair interaction (I_{ij});
- fraction of redundant, synergistic, conflicting and neutral pairs under adjudicated labels;
- pairwise intervention coverage and skipped/invalid intervention rate;
- decision-loss reduction per byte retained and per millisecond of maintenance;
- action-change and harm rates;
- cost of complete (O(n^2)) evaluation;
- confidence intervals from cluster-level resampling, never from treating pair rows as independent episodes.

Avoid reporting the sum of all pair interactions as a standalone “value” score: higher-order interactions can remain and pair terms can double-count effects. Use the pair map to diagnose retrieval composition and identify candidates for more selective storage, compression or follow-up observation.

## Benchmark targets (proposed, not achieved)

- Reproduce the full no-memory, BM25, BM25+RRF and candidate-memory paired baselines under one frozen reader/policy.
- Report leave-one-out and pairwise interaction coverage on 100% of episodes where the intervention budget permits; publish the remainder as explicitly unmeasured.
- Keep intervention reruns on identical policy digest, seed, candidate world and action space.
- Use at least 1,000 independent world clusters for confirmatory estimates where feasible; report power and cluster counts rather than assuming episode count is sample size.
- Require no unexplained candidate/evaluator leakage and no pairwise positive-value claim without held-out confirmation.

The paradigm shift is from "which memory item was retrieved?" to "which combinations of evidence change a decision, under what policy, at what lifecycle cost?"
