# HLS validity-memory capacity sweep v0

This note freezes the first finite-dimensional capacity study for `ValidityIntervalMemory` before any `research_v0` result is interpreted.

## Status

`ValidityCapacityPlan::research_v0()` is an exploratory preregistration. It is intended to reveal the failure surface, not to guarantee a successful regime.

If the pilot motivates different dimensions, seeds, horizons, candidate counts, or other settings, those changes must be introduced under a new experiment version rather than silently editing `research_v0`.

## Primary load coordinate

The archive represents one value for every active relation key at every causal checkpoint. The primary proposed load coordinate is therefore

`rho = key_count * horizon / dimension`.

This is the represented key-checkpoint facts per temporal dimension.

The hypothesis is not that `rho` alone determines performance. Candidate cleanup competition, codebook coherence, semantic-change density, and finite-dimensional temporal crosstalk may matter independently. The sweep records diagnostics for those alternatives rather than attributing every failure to `rho`.

## Why span length is a separate axis

A value valid over `[start, end)` is archived with one analytic O(D) span write regardless of interval length.

Changing span length keeps `key_count * horizon` fixed, but it changes both:

- the number of closed span writes;
- the number of opportunities for the independently resampled span value to change.

Because adjacent spans can draw the same candidate, span length is **not** a guaranteed one-semantic-change-per-boundary axis. Conversely, because values are independently resampled at each span, it is also **not** a pure numerical/write-segmentation axis.

`research_v0` therefore records the realized semantic-change count for every observation. Span-length results must be interpreted as a joint segmentation + stochastic mutation-density manipulation, with the realized change count used to disambiguate the two effects.

A later confirmatory plan may add two cleaner controls:

1. a forced-change mutation axis where every adjacent span must select a different value;
2. a pure-segmentation axis where a fixed semantic history is split into different numbers of mathematically equivalent writes.

## Fixed replicate seeds

Every research case is run on exactly five deterministic replicates:

`31001, 31002, 31003, 31004, 31005`.

Seeds may not be added, removed, or selected after observing outcomes while calling the run `research_v0`.

## Axis A — dimension

Fixed: keys 8, candidates 8, horizon 128 checkpoints, span length 8.

Dimensions: `512, 1024, 2048, 4096, 8192`.

## Axis B — relation-key count

Fixed: dimension 4096, candidates 8, horizon 128, span length 8.

Key counts: `2, 4, 8, 16, 32`.

## Axis C — candidate count

Fixed: dimension 4096, keys 8, horizon 128, span length 8.

Candidate counts: `2, 4, 8, 16, 32`.

## Axis D — checkpoint horizon

Fixed: dimension 4096, keys 8, candidates 8, span length 8.

Horizons: `32, 64, 128, 256, 512`.

## Axis E — span length / stochastic mutation density

Fixed: dimension 4096, keys 8, candidates 8, horizon 256.

Span lengths: `1, 2, 4, 8, 16, 32, 64`.

The runner records both nominal span writes and realized semantic changes, so an observed effect can be related to the history that was actually generated rather than inferred from span length alone.

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

- correct queries, total queries, and accuracy;
- mean cleanup margin and smallest cleanup margin;
- closed spans written;
- realized semantic changes;
- number of candidate values actually used;
- represented key-checkpoint facts and facts per dimension (`rho`);
- candidate-score evaluations;
- maximum absolute pairwise similarity within the key codebook;
- maximum absolute pairwise similarity within the candidate codebook;
- maximum absolute key-to-candidate codebook similarity;
- complex history payload bytes;
- temporal-axis payload bytes;
- key/value codebook payload bytes.

The codebook-coherence diagnostics are important because a low-dimensional failure should not automatically be attributed to temporal capacity if that replicate also produced unusually correlated bipolar roles.

Payload figures deliberately remain separate. They are not presented as a single misleading total because history state, temporal basis, and semantic codebook have different reuse semantics.

## Query-compute contract

Validity cleanup constructs the temporal query phasor once per checkpoint query, then reuses it across all candidate correlations. The deterministic operation accounting therefore separates one temporal-role construction from the `candidate_count` score evaluations rather than hiding repeated trigonometric work inside each candidate.

## Interpretation rules

1. Null and failure regimes are valid results.
2. The first dimension or horizon that fails is not to be hidden by averaging only successful cases.
3. Accuracy and cleanup margin must both be examined; high accuracy with near-zero margin is fragile.
4. Candidate-count effects must not be attributed to temporal capacity without comparison to the fixed-load candidate axis.
5. Span-length effects must be interpreted with realized semantic-change count; they are neither pure write-density nor guaranteed mutation-density effects.
6. Replicates with unusually high codebook coherence must be visible rather than silently discarded.
7. `research_v0` is descriptive/exploratory. Any confirmatory threshold must be frozen later on untouched seeds.
8. Wall-clock throughput is not part of this first deterministic core result; a dedicated benchmark can measure timing later.

## Next decision

If results show a stable regime with useful cleanup margins after controlling for codebook coherence and realized mutation density, freeze a separate historical state-tracking experiment that uses untouched benchmark seeds and compares:

- explicit event-log oracle;
- validity archive;
- integrated HLS current-state + historical archive;
- appropriate recurrent/SSM/attention/external-memory baselines.

If capacity collapses early, characterize that failure directly before integrating the archive with HLS.
