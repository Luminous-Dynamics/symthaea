# MON-042 Benchmark Plan v0.1

## Hypothesis

Counting dependency clusters instead of raw replay runs reduces overconfident reproducibility claims when observations share common infrastructure failure domains.

## Proposed targets (not achieved)

| Dimension | Target |
|---|---:|
| Domain inventory completeness negative controls | 100% held indeterminate |
| Shared-domain overlap detection | 100% of seeded declared overlaps |
| Transitive dependency closure | 100% of seeded A-B-C overlap chains |
| Missing/incomplete replay denominator | 100% retained as cluster non-success |
| Duplicate units, seeds and replay IDs | 100% blocked |
| Receipt/domain manifest tampering | 100% blocked |
| Mixed-stratum pooling | 0 accepted |\n| Mixed cluster-size pooling | 0 accepted without a stratified plan |\n| Replay scope / seed schedule drift | 100% blocked |
| Exact confidence bounds | 100% known-answer arithmetic |
| Campaign authorization from hash-only input | 0 permitted |

## Adversarial scenario matrix

- All dependency domains distinct across 29 execution units: 29 clusters.
- All units share one cloud control-plane ID: one cluster.
- Pairwise host overlaps: every overlapping pair collapses into a shared cluster.
- Transitive overlap: A shares host with B; B shares storage with C; all three form one cluster.
- Different unit IDs but same underlying identity issuer or credential authority: cluster them.
- Missing external-service dependency class: hold; do not count as independent.
- Replayed inventory digest with mutated domain map: block.
- A unit has fewer observations than preregistered: retain its cluster as non-success; never reduce N.
- One observation has a different blocked signature: cluster non-success.
- One observation is qualified or indeterminate: cluster non-success.
- A new unit is appended after preregistration: block optional sample expansion.
- Mixed strata submitted to v1: block pooled estimate.

## Statistical reporting

Report both:
- MON-040 per-replay result and its iid assumption;
- MON-042 dependency-cluster result and its cluster-level all-runs-success estimand.

Never relabel the MON-042 cluster-level probability as a per-replay reliability. Publish the number of execution units, number of connected components, units per component, repeated runs per unit, shared domain labels, failed/incomplete clusters and the one-sided bound.

Deterministic graph mutations test evaluator sensitivity, not operational independence. Independent infrastructure review and authenticated inventory receipts are required.

## Matched comparisons

Compare:
1. MON-040 raw-replay Clopper–Pearson;
2. MON-042 dependency-cluster Clopper–Pearson;
3. naive per-host aggregation;
4. independent cluster-level confirmatory experiments;
5. an explicitly correlated simulation family with known common-cause structure.

Use matched seeds within units where useful, but do not confuse matched seeds with independence. Simulate intra-cluster correlation, missingness, unequal cluster sizes, shared service outages and adaptive stopping. Evaluate interval coverage and bias against the preregistered cluster-level estimand.

## Execution and non-claims

The v0 module is an offline evaluator. There is no live infrastructure inventory adapter, signed dependency attestation, runtime fault campaign, hierarchical estimator, cluster bootstrap or independent replication. The unit tests are synthetic controls. No production reliability or SOTA result is claimed.
