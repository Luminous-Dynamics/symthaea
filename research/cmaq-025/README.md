# CMAQ-025 — Identifiability Under Observational Aliasing

**Status:** Rust evaluator and deterministic tests authored; exact-head formatting, tests, and Clippy remain pending. Local Rust execution was unavailable in the authoring environment.  
**Relationship:** continuation of CMAQ-020–024; isolated evaluator crate, not yet a NixWard runtime dependency.  
**Claim ceiling:** finite identification diagnostics under a frozen response contract. No arbitrary hidden-mechanism discovery, calibrated probability, external benchmark, or physical-safety claim.

## Research question

CMAQ-024 tests retrieval across unseen family identifiers when observable context carries the same response map as training. It does not show that a model can detect a hidden mechanism change that leaves every candidate-visible input unchanged.

CMAQ-025 scores this boundary. Records are observationally aliased when their complete candidate-visible input digests are identical. If evaluator-only intervention responses differ, that view does not identify one scalar response across those worlds.

The candidate must run before evaluator-only family IDs and reference responses are attached. Under one frozen checkpoint, identical candidate-view digests must produce identical outputs. A mismatch is rejected as view leakage or nondeterministic inference.

## Output semantics

- Point(value): one scalar response is asserted.
- Set(values): a finite, canonical set of possible responses is asserted.
- Unknown(reason): the candidate withholds a point forecast.

An aliased group is safely represented when the candidate returns Unknown, or a finite set that covers every evaluator-observed response in that group. A point forecast on an aliased group cannot be universally correct; the evaluator reports actual errors and the irreducible point-error floor.

## Information-flow boundary

The candidate channel is represented by CandidatePrediction: candidate-view digest plus Point, Set, or Unknown. The evaluator channel is represented separately by OracleAnnotation: candidate-view digest, family ID, independent-cluster ID, and reference response. EvaluationCustody binds one contract, source manifest, and candidate checkpoint for the complete batch.

The scorer is the only join point. Candidate inference is not given the oracle annotation type. Every prediction must match at least one annotation, every annotation must match a prediction, and repeated predictions for the same view must agree. This is an API-level information-flow separation, not a proof that an external caller has implemented a secure process boundary.

The candidate-view digest must cover state/context, realized action, query, horizon, authorized evidence, and every permitted probe result. It must exclude evaluator-only labels and reference values.

This crate syntax-checks SHA-256 strings; it does not recompute or authenticate their underlying bytes. That remains the responsibility of the independent evidence-ingestion boundary.

## Metrics

- point forecast coverage and selective risk;
- abstention on unidentifiable views and unnecessary abstention on identifiable views;
- finite response-set miss rate, width, and complete coverage of ambiguous groups;
- identifiable and unidentifiable view-group counts;
- independent family and cluster counts;
- irreducible scalar point-error floor, computed as group size minus the modal evaluator-response count for each aliased view.

The statistical unit is the independent environment/mechanism cluster, not a raw record or repeated transition. View-cluster cells are descriptive and are not independent when a cluster appears in multiple views.

## Deterministic witnesses

1. Same candidate view, two hidden family IDs, conflicting responses: a scalar forecast exposes the irreducible error floor.
2. A finite set covering both observed outcomes is scored as a set, not a point forecast.
3. Unknown on aliased views is safe; Unknown on identifiable views is measured as over-abstention.
4. New family/cluster IDs with an unchanged view-to-response map remain identifiable (benign covariate shift).
5. An informative permitted probe changes the candidate-visible view and separates previously aliased cases.
6. An uninformative probe leaves the group unidentifiable.
7. Repeated records in one cluster do not increase independent support.
8. Cluster reuse across families, mixed checkpoints, malformed digests, within-cluster response conflicts, non-canonical sets, and inconsistent outputs for one view are rejected.

## Paradigm implications

A predictor cannot detect an arbitrary mechanism change if that change leaves all candidate-visible observations identical. The benchmark should not reward pretending otherwise. Runtime prediction requires at least one of:

- an independently authenticated mechanism/invariance warrant that scopes the response map;
- a permitted observation or probe that changes the candidate-visible evidence;
- a robust response set containing all admissible outcomes; or
- an explicit Unknown coupled to a safe fallback or action shield.

The evaluator diagnoses this limit; it cannot create information the candidate never received.

## Next integration

Serialize candidate predictions from the actual NixWard HDC transition path before evaluator-only fields are attached. Compare contextual HDC, pooled delta, identity/no-op, full-state, symbolic/SCM, and recurrent baselines under the same contract, clusters, and probe budget. Then connect CMAQ-016 to test whether planner search exploits optimistic scalar forecasts that CMAQ-025 marks as unidentifiable.
