# EKM-029 — Belief Mutation Authority Surface Closure

## Purpose

EKM-026 introduced the first writer for explicit epistemic-support state. EKM-028 then wrapped that writer in a sealed decision/evidence transaction with independent verification. Before this tranche, however, the raw mutation firewall and transaction coordinator were still publicly reachable from `knowledge::*`.

EKM-029 closes that API-level bypass.

## Public authority surface

Public callers now use one facade:

1. `BeliefMutationAuthority::prepare(...)`
   - evaluates the EKM-024 belief-revision policy,
   - records the EKM-025 immutable decision receipt,
   - validates decision-time evidence/provenance temporality,
   - seals the complete EKM-028 claim/evidence census,
   - returns an opaque `PreparedBeliefMutation` containing the exact receipt+seal pair.

2. A higher authority separately creates `BeliefMutationAuthorization` against the prepared receipt and exact current `EpistemicSupportState`.

3. `BeliefMutationAuthority::apply(...)`
   - rechecks the prepared receipt/seal binding,
   - invokes the sealed EKM-028 transaction,
   - reaches the EKM-026 writer only through its internally owned firewall,
   - returns the EKM-027 independent verification report with the mutation outcome.

`PreparedBeliefMutation` has private fields. A public caller may inspect its receipt and seal but cannot assemble an arbitrary receipt/seal pair.

## Visibility closure

The following modules are now private implementation details of `knowledge`:

- `belief_mutation_decision_guard`
- `belief_mutation_firewall`
- `belief_mutation_transaction`
- `belief_mutation_verifier`

The raw mutation-capable symbols are no longer publicly re-exported:

- `BeliefMutationFirewall`
- `BeliefMutationTransactionCoordinator`
- `BeliefMutationDecisionGuard`

They retain crate-private aliases so internal EKM tests and future internal integration can use them without reopening the external authority surface.

Read-only/audit types and the separately supplied authorization/state types remain public where useful.

## Invariants

- Public callers cannot invoke the raw epistemic-support writer through the `knowledge` API.
- Public callers cannot invoke the raw transaction coordinator through the `knowledge` API.
- Public callers cannot forge a `PreparedBeliefMutation` from mismatched public pieces.
- Preparation always uses same-call decision + full evidence/provenance sealing.
- Application always uses the sealed transaction path and independent post-mutation verifier.
- Authorization remains separate from preparation and is bound to the exact decision and exact pre-state.
- Rebuilding the facade does not make a previously applied decision apply twice while the support store/history survives.

## Non-claims

This tranche does **not** establish:

- cryptographic signer authentication,
- persistence across process restart,
- safe rollback execution,
- correctness of any scientific belief-update policy,
- migration of legacy `TemporalFact::confidence`,
- causal truth,
- action authority.

The existing authorization object remains a typed approval record, not a signature or identity proof.

## Qualification boundary

This PR is stacked on EKM-028 / PR #3715. It changes the API authority surface and adds facade-level controls, but no CI PASS is inferred until the exact head executes the repository qualification workflow.

Queued, absent, or unexecuted workflow jobs are not qualification evidence.
