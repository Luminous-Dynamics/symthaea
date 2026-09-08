# Spark Outcome-Class Order Invariance v1 — Frozen Candidate Plan

Status: queue-neutral branch artifact only. NOT QUALIFIED. NOT MATERIALIZED INTO PRODUCT SOURCE.

Issue: #857.

Base: `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`.

Candidate patch: `patches/science/spark-outcome-class-order-invariance-v1.patch`.

## Demonstrated defect

Current `OutcomeClasses::from_design()` greedily groups predictions by overlap with the first representative of an existing class.

Because interval overlap is not transitive, equivalent semantic prediction sets can produce different class partitions under different input orderings.

Synthetic chain:

- A: rate `[0,2]`, energy `2.45`;
- B: rate `[1,3]`, energy `2.45`;
- C: rate `[2.5,4]`, energy `2.45`.

Relations:

- A overlaps B;
- B overlaps C;
- A does not overlap C.

Current representative-greedy results:

- A/B/C -> `{A,B}`, `{C}`;
- B/A/C -> `{A,B,C}`;
- C/B/A -> `{C,B}`, `{A}`.

Thus experiment EIG can depend on vector order.

`matching_class()` also checks only the representative prediction for each class, so an observation that matches a non-representative member can be missed.

## Candidate theorem

The candidate aims to establish only:

- current signature-level ambiguity grouping is deterministic under permutation of the same prediction set;
- class membership is derived from the complete pairwise overlap graph, not one arbitrary representative;
- member order and component order are canonical under `ALL_HYPOTHESES`;
- observation matching examines every member of the ambiguity component;
- EIG under the same belief and semantic prediction set is invariant to prediction-vector permutation.

## Candidate mechanism

The patch replaces representative-greedy grouping with connected components of the undirected pairwise signature-overlap graph.

This is explicitly a **conservative ambiguity closure**.

If A overlaps B and B overlaps C while A and C are directly distinguishable, transitive closure places all three into one ambiguity component. That can understate available information. It is preferable to order-dependent scientific ranking, but it is not the final observation model.

## Required future qualification

Before applying the patch, an exact hosted/local qualification should execute current-main negative controls proving:

1. the A/B/C permutations yield different class partitions;
2. at least two permutations yield different EIG under the same belief;
3. representative-only observation matching misses a non-representative member.

Then apply the exact patch artifact and require:

1. identical canonical classes under all six permutations of A/B/C;
2. identical EIG under all six permutations;
3. observation matching succeeds for any member support in a component;
4. existing Spark Bayesian/EIG tests continue to pass;
5. Rust formatting/check/strict Clippy pass under the pinned toolchain;
6. no widening of scientific claims from signature-level approximation to calibrated physical likelihood.

## Follow-on scientific boundary

The connected-component model should remain a baseline.

A later SCI-008/SCI-009 adapter should replace hard ambiguity classes with an explicit observation likelihood such as `p(y | h, E, measurement_model)` that can retain detector resolution, uncertainty, missing observation channels, background, calibration/OOD state, censoring, and other scientific measurement semantics.

## Queue governance

Do not open a PR or add a workflow for this branch while the focused verifier/soundness queue remains starved.

The branch exists to preserve an exact candidate and qualification plan without consuming another PR-triggered Actions surface.

No qualification, correctness, merge-readiness, scientific-calibration, or action-authority claim follows from this branch artifact.