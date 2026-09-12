# symthaea-energy-material-pareto-cohort

Strict pre-Pareto eligibility bridge for Tier-1 energy-material screening.

This crate does **not** implement non-dominated sorting. It converts screening assessments into a canonical cohort that can later be handed to the shared domain-neutral discovery Pareto bridge (#1729).

## Why a separate boundary exists

A Pareto algorithm should never have to decide whether evidence is complete or whether hard constraints passed. Those are upstream epistemic/feasibility questions.

This bridge therefore admits an evaluation to the downstream Pareto cohort only when it is:

- bound to the exact supplied screening policy;
- evidence-complete across all seven Tier-1 dimensions;
- internally consistent with the canonical policy schema;
- free of any pre-existing Pareto rank;
- `Feasible` under its hard constraints.

Everything else remains unranked.

## Revalidation, not label trust

`EnergyMaterialScreeningAssessment` is serializable and has public fields. A caller could therefore deserialize or directly construct a value whose labels do not match its evidence.

Before eligibility is considered, this bridge revalidates:

- candidate identity;
- exact policy id and SHA-256;
- screening capability classification;
- canonical seven-dimension order and metric/unit schema;
- available dimensions have selected predictions;
- unavailable dimensions do not carry selected predictions;
- selected prediction metric/unit, fidelity and accepted evidence-kind requirements;
- completeness flag agrees with dimension statuses;
- incomplete assessments carry no generic evaluation;
- complete assessments carry a valid generic evaluation;
- evaluation candidate id matches the assessment;
- objective schema/order matches the exact policy;
- constraint schema/order matches the exact policy;
- evaluation predictions equal the canonical selected dimension evidence;
- `pareto_rank` is still `None`.

A mismatch is an integrity error, not an exclusion score.

## Exclusion ledger

Candidates that are valid assessments but not rankable remain visible as:

- `IncompleteEvidence`, including each unavailable dimension/status;
- `HardConstraintInfeasible`;
- `UnknownFeasibility` (defensive case).

They do not disappear from the evidence record and are not assigned a worst synthetic rank.

## Canonical cohort identity

Input assessments are sorted by candidate id before processing. The resulting eligible evaluations and exclusions therefore have stable ordering independent of caller input order.

The full report can be content-addressed with a domain-separated SHA-256.

## Downstream Pareto relationship

The intended downstream implementation is the existing strict discovery Pareto bridge (#1729), which already provides:

- cohort objective-schema isolation;
- unique highest-fidelity objective selection;
- minimize/maximize/target transforms;
- deterministic non-dominated ranks;
- atomic rank assignment;
- no rank for missing/ambiguous/infeasible evaluations.

This crate intentionally does not duplicate that logic while #1729 remains a separate draft stack.

## Authority boundary

Cohort eligibility means only that the candidate has sufficient declared evidence and satisfies the policy's hard constraints to participate in comparative ranking.

It does not mean the candidate is Pareto-optimal, scientifically validated, novel, synthesizable, safe for use, manufacturable at scale, certified, or approved for deployment.
