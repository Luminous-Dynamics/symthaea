# EKM-028 — Sealed belief mutation transaction

## Purpose

EKM-028 closes the freshness gap between a belief-revision decision and the EKM-026 mutation firewall.

A timestamp-only freshness rule is insufficient because an evidence record may be inserted *after* a decision while carrying an older `observed_at_cycle`. EKM-028 therefore seals the complete claim/evidence census at decision time and requires exact equality immediately before mutation.

## Architecture

The intended path is:

1. `BeliefMutationDecisionGuard::evaluate_and_seal`
   - evaluates and records the EKM-025 revision decision;
   - retains an immutable borrow of the ledger for the whole operation;
   - validates the complete claim provenance ancestry against the decision cycle;
   - captures a `BeliefRevisionEvidenceSeal` over the exact claim and evidence census.
2. A separate EKM-026 `BeliefMutationAuthorization` binds the eligible receipt to an exact support pre-state.
3. `BeliefMutationTransactionCoordinator::apply`
   - verifies the live claim/evidence census is byte-semantically equivalent to the seal;
   - captures the EKM-027 pre-mutation snapshot;
   - delegates the only mutation to the existing EKM-026 firewall;
   - independently verifies the resulting transition with EKM-027.

## Invariants

- Decision and seal use the same logical cycle.
- The decision/seal guard borrows the ledger immutably across both operations.
- Every claim evidence ID must resolve exactly once.
- Every sealed evidence record preserves ID, claim, kind, polarity, provenance, observation cycle, context, and method.
- The receipt basis must be a semantic subset of the full sealed census.
- Any added/removed evidence ID invalidates the seal, even when a late insertion carries an older observation timestamp.
- Any semantic change to a sealed evidence record invalidates the seal.
- Any claim metadata change invalidates the seal.
- Any provenance node or declared provenance ancestor recorded after the decision cycle blocks same-call sealing.
- Evidence observed after the decision cycle blocks sealing, including non-basis contextual evidence.
- Seal verification occurs before the EKM-026 mutation call.
- EKM-027 verification remains independent and observational after mutation.
- A post-mutation verification failure is returned as data; this layer does not pretend that the already-attempted mutation did not happen.

## Non-claims and authority boundary

EKM-028 does not establish that an accepted belief is true, calibrated, causally identified, transferable, or safe to act upon. Those claims remain governed by the earlier evidence, calibration, uncertainty, and causal-admission layers.

This PR adds no new belief-update primitive. The only support-state mutation remains `BeliefMutationFirewall::apply` from EKM-026.

It does not mutate or migrate:

- `EnhancedKnowledgeGraph`;
- `TemporalFact::confidence`;
- legacy HDC corroboration;
- retrieval recency;
- dream/consolidation strengthening;
- the causal DAG;
- world-model state;
- action selection;
- external systems.

Rollback execution, restart persistence for the support store/history, and cryptographic authorization remain separate boundaries.

## Qualification boundary

The implementation is stacked on EKM-027 and remains draft until executable CI evidence exists. Mergeability, static inspection, expected test behavior, and queued jobs are not qualification evidence.

A future qualification PASS for EKM-028 should establish only that the sealed decision→authorization→mutation→verification protocol executes as specified. It does not establish scientific correctness of the underlying evidence or belief-revision policy.
