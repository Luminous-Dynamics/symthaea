# EKM-026 — Belief Mutation Firewall

## Status

Draft / unqualified until executable CI evidence exists.

## Purpose

EKM-026 introduces the first EKM layer permitted to mutate an explicit epistemic-support value. It does **not** modify `EnhancedKnowledgeGraph`, `TemporalFact::confidence`, causal DAG state, world-model parameters, action selection, or any external system.

The mutation target is a separate `EpistemicSupportStore` whose state is explicitly initialized and versioned.

## Authority chain

A mutation requires all of the following:

1. an EKM-025 `BeliefRevisionReceipt` whose frozen EKM-024 decision is eligible;
2. no duplicate evidence-ID attempts in that receipt;
3. an explicit `BeliefMutationAuthorization` bound to the exact revision receipt, claim, delta, expected state revision, expected support, authority label, and authorization cycle;
4. mutation time not earlier than either evaluation or authorization;
5. exact live equality between snapshotted basis evidence and the current ledger records;
6. no evidence for the claim observed after the revision decision;
7. re-evaluation of the frozen revision proposal/policy/calibration/uncertainty against the current ledger with an unchanged eligible decision;
8. an exact bounded state transition that fits in `[0,1]` without clamping.

## Replay and stale-state behavior

- A revision receipt is single-use.
- Replaying an already-applied receipt returns the original mutation receipt and performs no second mutation.
- Authorization IDs are bound in the retained store as well as the firewall instance, so rebuilding the firewall cannot make a previously acknowledged authorization ID available for a different mutation.
- Authorization is optimistic-concurrency-bound to an exact pre-state revision and support value.
- A revision receipt older than the current support state fails closed.
- Any evidence observed after the revision decision makes that decision stale, even when the later evidence points in the same direction.

The store is still in-memory. Restart durability across process loss is **not established** until a separately qualified persistence format exists.

## Reversibility

Every successful mutation records exact support-before/support-after and state revision-before/revision-after values. The receipt can produce a typed rollback plan containing the exact value to restore and expected current state.

The rollback plan is deliberately non-executing. Rollback authority and rollback mutation remain a separate future boundary.

## Non-claims

This PR does not establish that:

- a belief is true;
- a causal claim is identified;
- provenance roots are statistically independent;
- calibration generalizes out of distribution;
- an epistemic-support scalar is sufficient to represent all belief semantics;
- live legacy confidence should yet be replaced;
- any external action is authorized.

## Qualification requirements

At minimum, executable qualification should cover:

- format and Clippy;
- unit tests for exact one-time application and replay;
- firewall reconstruction with retained store history;
- rejected receipt and denied authorization controls;
- duplicate-evidence rejection;
- post-decision evidence invalidation;
- stale-state authorization rejection;
- out-of-range transition rejection without clamping;
- exact rollback-plan generation;
- no mutation of the legacy knowledge graph.

Queued or unexecuted CI is not qualification evidence.
