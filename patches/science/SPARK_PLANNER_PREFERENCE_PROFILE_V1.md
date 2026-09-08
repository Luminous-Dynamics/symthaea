# Spark Planner Preference Profile v1 — design / qualification contract

Status: staged architecture only. No product implementation. No scientific or action authority.

Tracks: #907

## Current inconsistency

The planner currently has two different preference implementations:

`rank_experiments()`:

```text
EIG per dollar descending
then raw EIG descending
```

`greedy_sequence()`:

```text
EIG per dollar strictly greater than current best
otherwise preserve first encountered candidate
```

Therefore the report ranking and the sequence selector can disagree when EIG/$ ties.

## Required invariant

Within one declared planner profile:

```text
candidate assessment
 -> one named preference relation
 -> ranking and selection consume that same relation
```

Incidental vector order is not a scientific preference.

## Compatibility profile

A first compatibility-safe profile may preserve the existing intended scalar objective:

```text
SparkEigPerDollarPreferenceV1

1. candidate must be structurally valid
2. candidate must be eligible now
3. EIG/$ descending
4. raw EIG descending
5. deterministic canonical candidate tie-break
```

Steps 1-2 become real only after the #885/#890 contracts are implemented. Until then the compatibility layer must not imply those gates are already enforced.

## Tie-break evidence

A future comparison result should be machine-readable, e.g. conceptually:

```text
PreferenceDecisionV1 {
    winner,
    loser,
    decisive_coordinate:
        EigPerDollar
        | RawEig
        | CanonicalTieBreak,
}
```

This lets #906 planning receipts explain why a winner beat a near alternative.

## Candidate identity boundary

Do not use candidate list position as final tie-break.

Before SCI-002 candidate identity is available, a validated unique Spark-local experiment identifier/name may be used as a diagnostic deterministic tie-break. Such a string is not yet a universal scientific identity.

## Floating-point boundary

The current implementation uses `f64::total_cmp` in ranking. Preserve one exact documented comparison rule for compatibility tests rather than silently introducing approximate equality.

A future uncertainty-aware/Pareto planner may use a different preference relation; it must have a different profile identity.

## Pareto boundary

SCI-009 supports multidimensional planning coordinates. This scalar profile is only a legacy/compatibility policy.

Do not later compute a Pareto frontier and then silently route exact ties through this scalar comparator while presenting the result as policy-neutral Pareto dominance.

Distinguish:

```text
Dominated
NonDominated
PreferredAmongNonDominatedUnderExplicitPolicy
```

## Required negative controls

1. Synthetic A/B with equal EIG/$ but B having higher raw EIG: ranking and one-step selection both choose B.
2. Reverse A/B input order: winner unchanged.
3. Exact tie on EIG/$ and raw EIG: final deterministic tie-break gives same winner under all input permutations.
4. Receipt records which comparator level decided each close comparison.
5. Changing the preference profile changes planning-snapshot identity/provenance.
6. A candidate blocked by prerequisites cannot win through high scalar utility after #890 is implemented.

## Non-claims

This contract does not establish EIG/$ as the correct universal objective, calibrate EIG, authorize an experiment, or define future Pareto preference policy.
