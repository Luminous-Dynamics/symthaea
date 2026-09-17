# EKM-027 — Belief Mutation Verifier

## Status

Draft / unqualified until executable CI evidence exists.

## Purpose

EKM-027 independently checks the EKM-026 isolated epistemic-support mutation after it occurs. A successful firewall call is not treated as sufficient evidence that the resulting state transition was correctly scoped.

The verifier is read-only. It cannot mutate, repair, roll back, authorize, or retry anything.

## Pre-mutation snapshot

`BeliefMutationSnapshot` captures:

- target claim and complete target support state;
- isolated support-store state count;
- mutation-history length;
- consumed-authorization count;
- epistemic ledger claim/evidence/provenance counts.

## Post-mutation checks

For a newly applied revision, the verifier checks:

- source revision receipt identity;
- target claim identity;
- exact proposed delta;
- applied authorization ID;
- unchanged ledger claim/evidence/provenance counts;
- unchanged support-store state count;
- exactly one appended mutation history entry;
- exactly one newly consumed authorization binding;
- exact support-before value;
- exact support-after = support-before + approved delta;
- monotonic state revision +1;
- current target support equals the mutation receipt's after value;
- current `last_mutation_id` and update cycle match the receipt;
- exactly one stored mutation receipt with that mutation ID.

For an idempotent replay, no second history entry may appear. The live state may legitimately be at a later revision due to subsequent separately authorized changes, but it must never regress behind the original receipt's applied revision.

## Negative controls

The initial fixture set includes detection of:

- unrelated support-store registration between snapshot and verification;
- unrelated ledger claim/provenance/evidence mutation between snapshot and verification;
- duplicate mutation history on replay;
- state regression relative to the returned receipt.

## Evidence boundary

EKM-027 does not establish that the original belief-revision policy was scientifically correct. That remains the responsibility of EKM-024/025 and their evidence/calibration/uncertainty inputs.

It also does not persist state, authenticate authority cryptographically, execute rollback, mutate the live legacy confidence path, promote causal claims, change world-model state, or authorize external action.

A verifier PASS means only that the observed isolated mutation matched the recorded scope and state-transition invariants.
