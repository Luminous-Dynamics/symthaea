# EKM-036 — Belief Revision Snapshot Schema

## Purpose

EKM-034 intentionally leaves `BeliefRevisionPolicy` and complete decision failure semantics opaque in the first restart wire envelope because those live types were not designed as persistence schemas. EKM-036 introduces an explicit, read-only semantic target without exposing private authority-bearing internals.

## Added contracts

- `BeliefRevisionPolicySchemaV1`
  - explicit policy fields
  - numeric validation delegated to the existing `BeliefRevisionPolicy` constructor/builders
  - uncertainty caps canonicalized by stable uncertainty-dimension tag
  - deterministic `build_policy()` back into the existing private policy object
- `BeliefRevisionDecisionSnapshotV1`
  - eligibility
  - declared provenance-root count
  - complete typed failure list
- `BeliefRevisionFailureSnapshotV1`
  - typed payloads for every current EKM-024 failure variant
- `KnowledgeWeightRoutingFailureSnapshotV1`
  - typed nested authority-routing failures
- stable tags for uncertainty dimensions, knowledge-weight dimensions, and knowledge-weight sources

## Central invariant

**Future persisted policy semantics should originate from an explicit schema rather than be reverse-engineered from a private policy object after the decision was made.**

The schema builds the existing policy; it does not authorize a revision, evaluate evidence by itself, mutate support, or activate restored state.

## Canonicalization

`strengthen_uncertainty_caps` are stored in deterministic uncertainty-dimension order. Re-applying a cap for the same dimension replaces the previous value rather than creating duplicate policy semantics.

Stable V1 tags are explicit and are intended for a later wire migration. They do not depend on Rust enum discriminants or `Debug` formatting.

## Decision snapshots

The decision projection uses only public read-only methods on `BeliefRevisionDecision` and exhaustive matching over the public `BeliefRevisionFailure` variants. Numeric and typed payloads are retained; failures are not collapsed into strings.

## Authority boundary

This PR adds no belief mutation, evidence mutation, causal promotion, external action, restart hydration, activation, file I/O, or network I/O.

`BeliefRevisionPolicySchemaV1::build_policy()` constructs the same proposal-only policy object already used by the gate. It does not run the gate or grant mutation authority.

The schema does **not** retroactively recover policy fields from legacy receipts. That would reintroduce dependence on `Debug` parsing. The next tranche should record the explicit schema alongside a receipt at evaluation time and keep legacy opaque receipts visibly legacy.

## Qualification boundary

This stack remains unqualified until exact-head CI executes. Queued jobs are not format, compilation, Clippy, test, or runtime evidence.
