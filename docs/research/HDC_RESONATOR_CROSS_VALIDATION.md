# HDC Resonator Cross-Validation Harness

This research harness compares the existing direct associative cleanup path with the
iterative seeded resonator path on the same deterministic key/value memories.

## Purpose

The comparison is mechanism-level, not a dimension leaderboard. Each cell holds
resolution, memory load, and query corruption constant while both retrieval paths
consume the same generated fixtures.

The harness reports separately:

- direct cleanup accuracy and top-1/top-2 margin;
- resonator decoded accuracy and margin;
- correct convergence;
- spurious convergence;
- non-convergence;
- mean iterations across all resonator trials;
- mean iterations conditioned on convergence;
- solver-seed identity.

No composite score is defined.

## Matched fixture

The key/value generation and query corruption are inherited from
`associative_cleanup_harness`, so this experiment does not silently introduce a
second representation protocol. Direct cleanup performs the existing unbind +
nearest-codebook retrieval. Resonator solving uses the equivalent single unknown
constraint `key ⊛ X ≈ memory`, with the value codebook registered for cleanup.

Three deterministic solver seeds are used by default. They vary resonator
initialization while leaving the input representation and query fixed.

## Terminal outcome semantics

A resonator trial is classified as:

- **correct convergence** — converged and the decoded value is the target;
- **spurious convergence** — converged but decoded to a non-target value;
- **non-convergence** — did not satisfy the solver's convergence condition within
  the configured iteration budget.

Decoded accuracy is retained separately because convergence status and correctness
answer different questions.

## Research boundary

This does not establish an optimal HDC dimension, a universal resonator advantage,
or a production default. The purpose is to expose where direct cleanup and
iterative dynamics differ, including whether an apparent accuracy change is
associated with correct convergence, spurious convergence, or failure to converge.

The default matrix is 6 dimensions × 4 memory loads × 4 noise weights = 96 cells,
with 3 solver seeds per query. It is intentionally smaller than the broader
dimension ladders so the cross-validation remains an inspectable mechanism study.

## Literature basis

Recent resonator research explicitly separates correct convergence, spurious
convergence, and non-convergence because accuracy alone can hide materially
different failure dynamics. It also finds that cleanup nonlinearities and
representation choices alter convergence and failure modes. This harness follows
that measurement discipline rather than collapsing terminal behavior into one
number.

## Reproducibility

The experiment specification, fixture revision, solver seed ladder, and artifact
digest are part of the evidence object. The seeded resonator entry point controls
initialization without changing the production stochastic wrapper.

Focused validation:

`cargo test -p symthaea-core --lib resonator_cross_validation_harness`
