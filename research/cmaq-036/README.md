
# CMAQ-036 — Common-cause closure for independent calibration blocks

## Problem

A calibration set with 100 episode rows is not a 100-unit calibration sample when multiple episodes share a physical asset, collection session, scenario seed, external shock, raw source trace or temporal block. Treating them as independent pseudo-replicates can make conformal order-statistic ranks too optimistic and predictive-rank monitor null assumptions invalid.

## Implemented rule

The build_common_cause_closure function accepts a source-manifest digest and evidence units. Each unit has a unique episode ID, declared independent-cluster identity and receipt, split label, dangerous score and score receipt, and typed dependence keys with evidence-receipt digests.

The crate adds the declared cluster identity as a closure key, then computes deterministic transitive connected components over shared (kind, value) keys. The key receipt must be identical wherever that typed key appears in the same manifest. If any connected component spans two or more splits, the entire manifest is rejected.

Each component becomes one independent block with the maximum member dangerous score. Every member ID, score receipt and dependence-key receipt is retained. Calibration effective sample count is the number of closure blocks, never the raw row count. The manifest also reports the exact raw/effective shrinkage ratio.

## Why maximum aggregation?

For one-sided calibration of dangerous optimism, each connected component represents evidence that may share a common cause. Taking the maximum score in that component prevents repeated benign episodes from diluting one dangerous member and prevents multiple dependent episodes from receiving multiple order-statistic votes. The resulting unit matches the conservative block-maximum approach used in CMAQ-029 and CMAQ-030.

The choice is task-specific: if larger scores do not mean more dangerous outcomes, the data must be transformed under a reviewed score contract before it enters this API.

## Trust boundary

This catches declared common-cause overlap, transitive dependencies, shared-session leakage and pseudo-replication. It does not discover undeclared shared causes, prove that disjoint declared keys are physically independent, authenticate signatures, infer semantic similarity, or make environment-family transfer exchangeable.

The external ingestion authority must authenticate dependence-key inventories and receipts. TemporalBlock values must come from a canonical interval/blocking policy; overlapping intervals assigned different keys will not be discovered automatically. Unknown common-cause relationships must be merged or blocked upstream, not labelled independent by omission.

## Adversarial test matrix

Tests cover transitive chains (asset A-B, session B-C), cross-split leakage through one key, transitive seed-to-raw-trace split leakage, duplicate unit IDs, contradictory shared-key receipts, missing calibration partitions, malformed digests, deterministic output under input-order permutations, and max-score preservation.

## Benchmark fixture: pseudo-replication

The unit-test fixture has four calibration episodes:
- three episodes share a physical asset and collection session through a transitive chain;
- one separate episode has a distinct scenario seed.

Raw row count is 4; closure-block count is 2; effective sample size is reduced by 50%; the shared block score is the maximum of 90, not the mean of 43.3 or three independent votes.

## Literature and positioning

Recent grouped conformal studies demonstrate that row-pooled calibration can materially under-cover at the cluster-weighted level. A 2026 clustered-data study reports nominal 90% pooled coverage of 87.2% cell-weighted but 79.4% economy-weighted; equal-cluster and subsampling rules improve coverage at greater interval width. This motivates careful definition of the calibration unit but is not evidence for this exact closure algorithm.

See:
- Allohibi (2026), Grouped Validation and Conformal Calibration for Clustered Cross-National Data, Mathematics 14(21), 4055.
- Ramos et al. (2026), Conformal Prediction via Transported Beta Laws, PMLR 329, 133–168: dependence can deform the calibration-conditional order-statistic law.
- Bhattacharyya & Barber (2026), Group-weighted conformal prediction, Electronic Journal of Statistics 20(1).

CMAQ-036 is a software evidence-closure mechanism, not a new conformal theorem. Its integration step is to make CMAQ-029/030 accept only an authenticated closure manifest and use resulting block-max scores directly.
