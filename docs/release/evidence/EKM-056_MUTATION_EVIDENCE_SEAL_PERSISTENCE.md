# EKM-056 — Mutation-Time Evidence Seal Persistence

## Purpose

EKM-055 exposed a real restart limitation: the final ledger may contain evidence added after an already-applied belief revision, while EKM-026 correctly refuses to replay that historical mutation against the later evidence census.

No new timestamp should be invented to solve that problem.

EKM-028 already captures the exact complete claim/evidence census in `BeliefRevisionEvidenceSeal` at the same logical cycle as the revision decision, and verifies exact census equality immediately before mutation.

EKM-056 therefore treats the problem as **retention of an existing authoritative observation**, not inference of historical availability.

The central invariant is:

**every persisted applied belief mutation must retain exactly one canonical EKM-028 evidence seal bound to that exact mutation receipt.**

## Record contract

`PersistedBeliefMutationEvidenceSealV1` is created from:

- the exact `PreparedBeliefMutation` decision+seal pair produced by EKM-029; and
- the exact `BeliefMutationOutcome` returned by the authority application.

The record binds:

- mutation receipt ID;
- source revision receipt ID;
- claim ID;
- seal/evaluation cycle;
- applied cycle;
- a canonical digest over every mutation-receipt field;
- the complete EKM-028 sealed claim snapshot;
- every evidence snapshot present in the claim census at decision time;
- a domain-separated record digest.

The mutation binding digest covers the proposal delta, support before/after, state revisions, authorization identity and authority label, authorization cycle and application cycle in addition to the stable IDs.

## Complete-history capsule

`BeliefMutationEvidenceSealCapsuleV1` links these records to one EKM-030 `BeliefMutationPersistenceCapsuleV1`.

Capture fails unless:

- the seal count exactly equals the persisted mutation count;
- every mutation has exactly one seal record;
- mutation IDs are unique and ordered canonically;
- source revision receipt IDs are unique;
- each record's mutation-binding digest exactly matches the persisted mutation;
- each seal is bound to the same revision receipt and claim as its mutation;
- seal/evaluation precedes authorization, which precedes application;
- every sealed evidence record belongs to the sealed claim and does not postdate the seal;
- sealed evidence IDs are canonical and strictly ordered;
- the sealed claim still exists with identical immutable metadata in the final ledger;
- every sealed evidence record still exists with identical semantics in the final ledger.

Later evidence attached to the same claim is intentionally allowed.

That last rule is essential: EKM-056 preserves **what existed for the historical decision** without requiring the final append-only ledger to stop growing.

## Why this is better than `available_at_cycle`

The current ledger does not record an explicit insertion/availability cycle separate from `observed_at_cycle`. Adding one only at restart time would fabricate history.

The EKM-028 seal is stronger for historical mutation replay because it records the exact complete evidence census that was actually inspected at the decision boundary and checked again immediately before mutation.

Therefore EKM-056 reuses that evidence instead of creating a second chronology model.

## Regression case

A focused integration test now covers:

1. create claim evidence A;
2. prepare and apply an epistemic-support mutation using the sealed EKM-028 path;
3. retain the mutation/seal record;
4. append later evidence B to the same claim;
5. capture the final EKM-030 mutation state;
6. prove the EKM-056 seal capsule still validates against the evolved final ledger;
7. prove a persisted mutation history with a missing seal fails closed.

This is the exact case that EKM-055 cannot replay safely from the final ledger alone.

## Current boundary

EKM-056 does **not yet** place the seal capsule in EKM-032/039 restart manifests or EKM-042 wire V2.

It establishes the versioned, canonical retention object first. A later restart schema should carry this capsule explicitly and bind its digest into the restart manifest.

Once that exists, isolated hydration can reconstruct a mutation-time ledger projection for the target claim from the persisted EKM-028 census and replay the existing firewall without guessing what evidence was historically available.

## Non-authorities

EKM-056 does not:

- apply or authorize a belief mutation;
- alter EKM-029 authority routing;
- alter support state;
- alter revision decisions;
- modify the evidence ledger;
- activate restart state;
- export EKM-055 writable state;
- advance restart/verifier/protected checkpoints;
- change legacy `TemporalFact::confidence`;
- change the causal DAG, world model or action policy;
- perform file/network I/O or key custody.

## Qualification status

This tranche is stacked on EKM-055. GitHub Actions remains the executable authority. At creation time EKM-055 CI #7256 is queued, so no format, compile, Clippy, test or runtime PASS is inferred from the code or static review.
