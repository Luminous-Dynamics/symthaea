# EKM-025 — Immutable Belief Revision Decision Receipts

Status: audit/history layer only. No epistemic-support value is changed by this PR.

## Purpose

EKM-024 can evaluate whether a proposed `EpistemicSupport` delta is eligible under an explicit policy. Before any future mutation path exists, EKM-025 records the exact inputs and result of that evaluation in immutable in-memory receipts.

The history records both eligible and rejected decisions. Rejection history is epistemically useful: a failed proposal should not disappear merely because it was not authorized to update belief state.

## Receipt contents

Each receipt binds:

- target claim ID;
- proposed signed delta;
- proposal rationale;
- unique basis evidence IDs in first-seen order;
- immutable snapshots of evidence records that resolved at evaluation time;
- unknown basis IDs as unresolved references rather than silently dropping them;
- duplicate basis-ID attempts as a separate diagnostic;
- the complete `BeliefRevisionPolicy` value;
- calibration snapshot supplied to the gate;
- uncertainty assessment supplied to the gate;
- the gate decision and every typed failure reason;
- evaluation cycle.

The receipt recomputes `BeliefRevisionGate::evaluate` internally. It does not accept a caller-supplied verdict that may have been calculated against different inputs.

## Temporal integrity

A receipt fails closed if its evaluation cycle predates:

- creation of a known target claim;
- any known snapshotted basis evidence;
- the supplied uncertainty assessment.

Unknown evidence can still be preserved in a rejected audit receipt because the absence itself may be the reason the gate rejected the proposal.

## Duplicate evidence IDs

Repeated occurrences of one basis `EvidenceId` are snapshotted once. The receipt separately records which IDs were duplicated. This prevents a revision audit record from representing one ledger item as several independent pieces of evidence while preserving the fact that the proposal attempted to repeat it.

## Non-authority boundary

EKM-025 does not:

- apply the proposed delta;
- modify `TemporalFact::confidence`;
- modify a `KnowledgeWeightVector`;
- insert, delete, or rewrite evidence;
- resolve contradictions;
- update calibration or uncertainty;
- promote causal hypotheses;
- affect world-model state or action selection.

The current history is in-memory only. Persistence, replay protection for revision receipts, cryptographic authorization, and an actual belief-state mutation path remain separate future boundaries.

## Qualification rule

Do not call EKM-025 qualified until the exact PR head executes repository CI. Static inspection, mergeability, queued jobs, and expected fixture behavior are not PASS evidence.
