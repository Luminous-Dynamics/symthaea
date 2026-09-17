# EKM-024 — Evidence-Grounded Belief Revision Gate

Status: shadow/proposal-only gate. This PR contains no live belief mutation.

## Purpose

EKM-023 separates `EpistemicSupport` from memory accessibility, retention, and consolidation. EKM-024 evaluates whether a proposed epistemic-support delta is eligible under an explicit caller policy before any future mutation layer may exist.

## Required checks

The gate evaluates the proposal against the actual `EpistemicLedger` and can require:

- a bounded absolute update magnitude;
- basis evidence that exists and belongs to the target claim;
- supporting evidence for positive deltas and contradicting evidence for negative deltas;
- a minimum number of distinct **declared provenance roots**;
- a calibration snapshot with a minimum real sample count and maximum ECE;
- an uncertainty assessment, optionally required to postdate all current claim evidence;
- dimension-specific uncertainty ceilings before strengthening;
- no unresolved contradictory evidence before strengthening;
- intervention/replication evidence before strengthening a causal claim.

## Important epistemic boundaries

### Declared roots are not independence

Two distinct ultimate provenance roots satisfy only a lineage-diversity condition. They do not establish statistical, institutional, methodological, or causal independence.

### ECE = 0 with zero samples is not calibration

The existing `CalibrationAudit` returns `0.0` ECE when no samples exist. EKM-024 therefore carries `sample_count` separately and can require a minimum sample count. A zero-sample calibration snapshot cannot pass a positive minimum-sample policy merely because its ECE is numerically zero.

### Direction matters

A positive revision cannot be justified by a basis containing contradicting evidence, and a negative revision cannot be justified by a basis containing supporting evidence. Contextual evidence does not by itself satisfy either direction.

### Causal language is not causal evidence

When enabled by policy, strengthening a causal claim requires supporting `Intervention` or `Replication` evidence in the proposal basis. A report asserting causality remains a report.

### Uncertainty freshness is explicit

`require_current_uncertainty = true` independently requires an uncertainty assessment. If newer evidence exists after that assessment, the revision fails closed as stale.

### Eligibility is not truth

`eligible = true` means only that the explicit caller policy accepted this proposed bounded update. It does not establish that the claim is true, calibrated in all environments, causally identified, transferable, or safe to act upon.

## Non-authority boundary

EKM-024 has no function that changes:

- `TemporalFact::confidence`;
- any `KnowledgeWeightVector` value;
- evidence history or polarity;
- uncertainty assessments;
- the causal DAG;
- entity identity;
- world-model state;
- action selection;
- external systems.

A separately qualified future mutation boundary would need immutable revision receipts/history, exact pre/post state binding, replay/idempotency protection, rollback/audit semantics, and independent post-mutation verification before live epistemic-support updates should be considered.

## Qualification rule

Do not call EKM-024 qualified until the exact PR head executes repository CI. Static inspection, expected fixture behavior, mergeability, or queued jobs are not PASS evidence.
