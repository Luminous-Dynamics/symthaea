# HLS validity numeric-condition alias contract

Status: pre-result interpretation contract. This document changes no frozen `research_v0` case or seed and contains no outcome interpretation.

## Condition identity

For cross-condition analysis, the scientific condition identity is the numeric tuple

`(dimension, key_count, candidate_count, horizon, span_length)`.

`ValidityCapacityAxis` is a presentation/sweep label. It is retained for per-axis tables and plots but does not create an additional independent condition when two labeled cases share the same numeric tuple.

## Frozen research_v0 census

`ValidityCapacityPlan::research_v0()` contains 27 labeled cases but 23 unique numeric conditions.

Two alias groups are frozen:

1. `(4096, 8, 8, 128, 8)` appears under `Dimension`, `KeyCount`, `CandidateCount`, and `Horizon`.
2. `(4096, 8, 8, 256, 8)` appears under `Horizon` and `SpanLength`.

These repeated labels are common anchors/crossover points. They are not extra replication.

## Required evidence check

After the retained v1/v2 artifact pair has independently verified, run `validity_capacity_condition_alias_verify` on retained v2.

The verifier removes only `case.axis` from each falsification payload. For a fixed numeric condition and replicate seed, all repeated labeled copies must then be exact JSON equals. Because v2 floats carry exact IEEE-754 bit encodings, any differing raw/falsification metric causes fail-closed rejection.

A disagreement is a pipeline defect, not an experimental result.

## Replication and pooling

The preregistered seed remains the independent replication unit within each unique numeric condition. Repeated axis labels contribute one condition, not multiple conditions, to any global sample-size statement, model-fit statistic, or pooled cross-condition summary.

Per-axis presentations may preserve each labeled anchor copy so readers can see the common center/crossover point used by each sweep.

## Non-claims

This contract defines no favorable sign, threshold, significance test, capacity law, or HLS superiority claim. It exists solely to prevent pseudoreplication and to make repeated anchors useful as deterministic pipeline-consistency sentinels.
