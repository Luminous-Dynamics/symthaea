# symthaea-pareto

Small domain-neutral Pareto kernel extracted as a common target for Symthaea's existing domain-specific multi-objective implementations.

## Scope

The crate provides only:

- named objective specifications;
- explicit minimize/maximize direction;
- finite objective vectors;
- deterministic non-dominated ranks;
- deterministic Pareto fronts preserving input order;
- NSGA-II-style crowding distance within each rank.

It does not know about energy, consciousness, neuroevolution, fusion, candidates, feasibility, weights, or deployment policy.

## Why a shared kernel

The repository currently contains several valid but hard-coded Pareto implementations, including:

- neuroevolution fitness dimensions;
- Spark/core design-space mass/cost/dose/lifetime;
- consciousness-profile dimensions.

Those APIs cannot safely be reused for arbitrary discovery objectives without pretending one domain's semantics are universal. This crate is the extraction point; existing callers can migrate only after equivalence tests show their old and new fronts agree.

## Determinism and validation

- objective names must be non-empty and unique;
- at least one objective is required;
- every point must have exactly one finite value per objective;
- dominance is direction-aware and strict in at least one objective;
- fronts are emitted in original input order;
- crowding uses normalized objective ranges and `f64::INFINITY` for boundary points;
- constant-valued objectives contribute zero interior crowding distance;
- NaN/Inf are rejected rather than producing accidental nondomination.

## Non-goals

This v0 does not implement weighted scoring, hypervolume, epsilon dominance, constraint handling, target-value semantics, evolutionary selection, or candidate mutation. Those belong in callers/adapters unless they become demonstrably shared primitives.
