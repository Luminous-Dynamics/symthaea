# Migrating to 0.4

Version 0.4 is additive. The v0.3 direct resolver remains available. Recursive
inference is a separate named semantic profile and does not silently replace
`resolve_literal`.

## Preserve factual provenance

Prefer `FactBase` and `FactAssertion` when results will be reviewed or bound as
evidence. Classify each input as observed, stipulated, or assumed and attach a
source or asserting authority when available.

## Choose ambiguity behavior explicitly

Use `InferenceProfile::grounded_blocking_v1()` when a literal supported in both
signs must not satisfy downstream premises. Use
`InferenceProfile::grounded_propagating_v1()` when each sign may propagate
independently. Explicitly supported exceptions block rules under both profiles.

Do not infer a result after `InferenceError::Oscillation` or
`InferenceError::RoundLimit`. Those errors mean the selected profile did not
produce a grounded final state within its declared bounds.

## Move recursive queries to `infer`

Pass a validated `RulePack`, `FactBase`, and `InferenceProfile` to `infer`.
Use `InferenceResult::status` for four-valued queries, `explain_query` for local
blockage diagnostics, and `ProofGraph::slice_for` for transitive support.

The direct `resolve_literal` API remains appropriate when the caller already
possesses a closed fact set and wants only one layer of conflict resolution.

## Select temporal revisions before inference

Represent versioned rule packs as `TemporalRevision<RulePack>` and use
`infer_at`. No governing revision returns `None`; overlapping governing
revisions remain an explicit error. Run `overlapping_revisions` during rule-pack
approval to detect schedule overlap before a live query.

## Resolve same-day lifecycle sequence explicitly

The original `assess_lifecycle` deliberately reports same-day action/waiver
ambiguity. Supply a validated `EventOrder` to
`assess_lifecycle_with_order` only when upstream formalization has established a
real within-day sequence. Event identifiers are never treated as hidden time.

## Use transactional sessions for changing facts

`EvaluationSession::apply` commits additions and removals only when the new
inference state succeeds. The current implementation recomputes the grounded
model; it does not claim an incremental optimization. `RuleDependencyIndex`
provides deterministic conservative impact analysis for future verified
incremental evaluators.

## Update evidence bindings

Version 0.4 adds canonical encodings for fact bases, recursive inference,
query explanations, proof graphs, temporal results, and evaluation deltas.
Recursive applications and proof graphs also bind the explicit exceptions that
were checked absent. Treat their schema tags as new payload types and update
allowlists accordingly.
