# SYM-RSI-SEM-004 — Preregistration Amendment 1

Status: **FROZEN BEFORE SEM-004 IMPLEMENTATION OR MEASUREMENT**

Parent preregistration:

`2a11924821831137a58253ec6013ba40a63f0f3d`

## Why this amendment exists

Pre-implementation review found a structural ambiguity in the parent protocol's relative-band calibration.

If the labelled clean source is usually the top-1 candidate, then the clean calibration gaps

`top1_similarity(query) - similarity(query, true_source)`

can all or nearly all equal zero. In that case the parent rule can produce `R_D = 0`, making the supposed ambiguity band unable to include two nearly tied but non-identical plausible parents.

That would fail to test the hypothesis SEM-004 was created to test.

No SEM-004 implementation, calibration output, held-out result, or disposition had been observed when this amendment was authored.

The thirteen primary success criteria, OOD rule, absolute unrelated-query gate, support boundary, maximum set size, and claim boundary are unchanged.

## New frozen ambiguous-calibration partition

Add, per dimension:

- ambiguous calibration queries: `16`.

These queries are deterministic mixtures/midpoints of two preregistered sibling candidate memories and record both intended parent identities before semantic encoding.

They are calibration-only and must be exact-disjoint from:

- labelled clean calibration queries;
- unrelated calibration-control queries;
- clean held-out queries;
- ambiguous held-out queries;
- unrelated held-out queries;
- OOD held-out queries;
- every frozen SEM-003 corpus identity.

Ambiguous calibration queries do not contribute to held-out metrics.

## Revised frozen relative-band calibration

For each dimension `D`, compute two calibration score families.

### Clean-source gap

For each labelled clean calibration query:

`clean_gap_i = top1_similarity(query_i) - similarity(query_i, true_source_i)`

with `clean_gap_i >= 0`.

Define:

`C_D = q95({clean_gap_i})`

using the parent protocol's deterministic nearest-rank rule.

### Ambiguous weaker-parent gap

For each ambiguous calibration query with intended parents `left_i` and `right_i`, define:

`ambiguous_gap_i = max(`
`    top1_similarity(query_i) - similarity(query_i, left_i),`
`    top1_similarity(query_i) - similarity(query_i, right_i)`
`)`

with `ambiguous_gap_i >= 0`.

This is the band width required to retain **both** intended parents for that calibration query.

Define:

`A_D = q95({ambiguous_gap_i})`

using the same deterministic nearest-rank rule.

### Operational relative band

Freeze:

`R_D = max(C_D, A_D)`

Candidate admission after absolute query admission remains:

`s* - similarity(q,c) <= R_D`

where `s*` is the query's top-1 same-dimensional similarity.

No held-out query contributes to `C_D`, `A_D`, or `R_D`.

## Absolute gate unchanged

The unrelated calibration-control split remains the sole calibrator of the absolute query-admission threshold:

`G_D = q95(unrelated_calibration_top1_similarity)`

Operational admission still requires strictly:

`top1_similarity(q) > G_D`

Ambiguous calibration data has zero authority over `G_D`.

## Frozen corpus counts after amendment

Per dimension:

- candidate memory bank: `96`;
- labelled clean calibration queries: `32`;
- ambiguous calibration queries: `16`;
- unrelated calibration-control queries: `32`;
- clean held-out queries: `32`;
- ambiguous held-out queries: `24`;
- unrelated held-out queries: `24`;
- OOD held-out queries: `16`.

## Additional calibration diagnostics

The SEM-004 receipt must report and bind, per dimension:

- `C_D` clean-source gap quantile;
- `A_D` ambiguous weaker-parent gap quantile;
- final `R_D = max(C_D, A_D)`;
- clean calibration true-source coverage under `R_D`;
- ambiguous calibration dual-parent coverage under `R_D`;
- unrelated calibration-control admission rate under `G_D`.

These calibration diagnostics are not held-out endpoints and cannot rescue a failed primary disposition.

## Generator requirements frozen before implementation

The SEM-004 generator must assign an explicit domain/version and deterministic equations before the first SEM-004 benchmark execution.

The implementation commit must not choose generator constants after inspecting SEM-004 held-out performance. If implementation review requires clarifying generator equations, that clarification must be frozen in a second preregistration amendment **before** executing the benchmark.

## Unchanged primary criteria

All thirteen primary success criteria from the parent preregistration remain unchanged.

In particular this amendment does not relax:

- clean coverage;
- ambiguity coverage;
- unrelated activation;
- overload rates;
- OOD rejection;
- returned set-size constraints.

## Claim boundary unchanged

`G_D`, `C_D`, `A_D`, `R_D`, set membership, overload state, similarity, margin, specificity, and support status remain semantic-search quantities only.

None grants factual correctness, empirical evidence authority, belief confidence, consciousness evidence, or confidence-promotion permission.
