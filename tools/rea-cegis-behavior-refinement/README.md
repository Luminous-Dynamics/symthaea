# REA CEGIS Behavioral Refinement v0.1

A Rust-first, dependency-free finite-domain kernel for counterexample-guided behavioral hypothesis refinement.

## Current scope

- Validate competing, exact-target and exact-environment behavioral hypotheses.
- Rank finite inputs by the number of distinct predicted outcomes and separated hypothesis pairs.
- Eliminate only hypotheses contradicted by complete observations.
- Preserve partial/unavailable observations as unresolved.
- Detect repeated complete observations that disagree and avoid using them to eliminate deterministic hypotheses.
- Group hypotheses observationally equivalent over an explicitly named finite input set.
- Greedily minimize a counterexample to one-deletion minimality using a caller-supplied discrepancy predicate.

## Qualification ceiling

This v0 is pure and synthetic. It does not execute target programs, admit or verify REA Evidence, authenticate an observation producer, prove capture completeness, compute EIG/EVSI, authorize experiments, or establish behavior outside the supplied finite domain. The minimizer calls only the predicate supplied by its caller; it does not execute a target. Its result is one-deletion-minimal, not globally minimum.

Real observations must enter through the separate #11257 integration after raw Evidence admission and capture-to-Evidence correspondence are qualified. The crate intentionally has no API that upgrades a synthetic result into admitted Evidence.
