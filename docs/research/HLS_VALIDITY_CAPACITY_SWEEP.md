# HLS validity-memory capacity sweep v0

This note freezes the first finite-dimensional capacity study for `ValidityIntervalMemory` before any `research_v0` result is interpreted.

## Status

`ValidityCapacityPlan::research_v0()` is an exploratory preregistration. It is intended to reveal the failure surface, not to guarantee a successful regime.

If the pilot motivates different dimensions, seeds, horizons, candidate counts, or other settings, those changes must be introduced under a new experiment version rather than silently editing `research_v0`.

## Primary load coordinate

The archive represents one value for every active relation key at every causal checkpoint. The primary proposed load coordinate is therefore

`rho = key_count * horizon / dimension`.

This is the represented key-checkpoint facts per temporal dimension.

The hypothesis is not that `rho` alone determines performance. Candidate cleanup competition, codebook crosstalk, and finite-dimensional effects may matter independently. The sweep varies those factors separately.

## Why span length is a separate axis

A value valid over `[start, end)` is archived with one analytic O(D) span write regardless of interval length.

Changing span length therefore changes the number of closed archive writes while **not changing** `key_count * horizon`, the number of represented key-checkpoint facts.

If retrieval accuracy changes strongly with span length at fixed `key_count`, `horizon`, and `dimension`, that is evidence that write segmentation/numerics matter beyond the simple fact-load hypothesis.

### Pre-execution limitation: segmentation is not guaranteed semantic mutation

Source audit found that `research_v0` assigns each span's candidate independently from the deterministic pseudo-random schedule. Adjacent spans can therefore occasionally select the same candidate. Under the intended approximately uniform assignment, the repeat probability is roughly `1 / candidate_count`.

Consequently, the span-length axis in `research_v0` must be interpreted as a **span-segmentation / write-density axis**, not as a guaranteed semantic-mutation-density axis.

This limitation was identified before result interpretation. The frozen cases and seeds are intentionally left unchanged rather than silently editing the preregistered schedule. A later confirmatory version may add a separate forced-change mutation axis in which every adjacent span is required to choose a different value.

## Fixed replicate seeds

Every research case is run on exactly five deterministic replicates:

`31001, 31002, 31003, 31004, 31005`.

Seeds may not be added, removed, or selected after observing outcomes while calling the run `research_v0`.

## Axis A — dimension

Fixed:

- keys: 8;
- candidates: 8;
- horizon: 128 checkpoints;
- span length: 8.

Dimensions:

`512, 1024, 2048, 4096, 8192`.

## Axis B — relation-key count

Fixed:

- dimension: 4096;
- candidates: 8;
- horizon: 128;
- span length: 8.

Key counts:

`2, 4, 8, 16, 32`.

## Axis C — candidate count

Fixed:

- dimension: 4096;
- keys: 8;
- horizon: 128;
- span length: 8.

Candidate counts:

`2, 4, 8, 16, 32`.

## Axis D — checkpoint horizon

Fixed:

- dimension: 4096;
- keys: 8;
- candidates: 8;
- span length: 8.

Horizons:

`32, 64, 128, 256, 512`.

## Axis E — span length / archive segmentation

Fixed:

- dimension: 4096;
- keys: 8;
- candidates: 8;
- horizon: 256.

Span lengths:

`1, 2, 4, 8, 16, 32, 64`.

This axis controls scheduled span segmentation and write count. It does not guarantee that every boundary changes the semantic candidate value.

## Total study

The plan contains 27 axis-labelled cases and five replicate seeds, for 135 observations.

Repeated baseline configurations across different axes are intentional: they keep each axis independently interpretable rather than deduplicating away its reference point.

## Deterministic synthetic histories

For each observation:

- temporal frequencies are generated from the replicate seed;
- key roles and candidate roles are generated from disjoint deterministic seed ranges;
- each key receives a deterministic piecewise-constant candidate sequence;
- every key is queried at every checkpoint;
- cleanup is performed against the complete candidate set.

No best seed, best checkpoint, or best candidate subset is selected.

## Reported metrics

Each observation reports:

- correct queries;
- total queries;
- accuracy;
- mean cleanup margin;
- smallest cleanup margin;
- closed spans written;
- represented key-checkpoint facts;
- facts per dimension (`rho`);
- candidate-score evaluations;
- complex history payload bytes;
- temporal-axis payload bytes;
- key/value codebook payload bytes.

Payload figures deliberately remain separate. They are not presented as a single misleading total because history state, temporal basis, and semantic codebook have different reuse semantics.

## Interpretation rules

1. Null and failure regimes are valid results.
2. The first dimension or horizon that fails is not to be hidden by averaging only successful cases.
3. Accuracy and cleanup margin must both be examined; high accuracy with near-zero margin is fragile.
4. Candidate-count effects must not be attributed to temporal capacity without comparison to the fixed-load candidate axis.
5. Span-length effects in `research_v0` are segmentation/write-density effects, not pure semantic-mutation effects.
6. `research_v0` is descriptive/exploratory. Any confirmatory threshold must be frozen later on untouched seeds.
7. Wall-clock throughput is not part of this first deterministic core result; operation counts are reported instead. A dedicated benchmark can measure timing later.

## Next decision

If results show a stable regime with useful cleanup margins, freeze a separate historical state-tracking experiment that uses untouched benchmark seeds and compares:

- explicit event-log oracle;
- validity archive;
- integrated HLS current-state + historical archive;
- appropriate recurrent/SSM/attention/external-memory baselines.

If capacity collapses early, characterize that failure directly before integrating the archive with HLS.
