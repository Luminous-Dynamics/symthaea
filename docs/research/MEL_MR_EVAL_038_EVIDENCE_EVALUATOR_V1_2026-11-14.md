# MEL-MR-EVAL-038 — Source-bound evidence evaluator for musical edits

**Date:** 14 November 2026  
**Status:** First executable reducer tranche; exact-head qualification pending  
**Tracking:** [Issue #12065](https://github.com/Luminous-Dynamics/symthaea/issues/12065)  
**Depends on:** [EVAL-037 / PR #12050](https://github.com/Luminous-Dynamics/symthaea/pull/12050)

## Why this is the next boundary

EVAL-037 defines what a valid edit means: requested musical obligations, protected invariants, permitted variation, forbidden changes and reference adequacy. A static envelope alone cannot decide whether a candidate actually met those obligations. EVAL-038 adds a deterministic reducer over typed, source-bound observation records.

The reducer does not create or authenticate measurements. It ensures that the observations it receives are bound to the exact envelope, candidate output and source lineage, and then applies the declared relations without silently converting missing or ambiguous evidence into success.

## Decision model

Each declared constraint receives exactly one outcome:

- **Satisfied:** a typed observation of the required evidence family supports the relation.
- **Violated:** a sufficiently resolved observation contradicts the relation.
- **Indeterminate:** evidence is missing, malformed, insufficient, unit-incompatible, or uncertainty overlaps the decision boundary.

The aggregate is deliberately non-compensatory:
- Acceptable only if every declared constraint is Satisfied.
- Rejected if any hard relation is Violated.
- Abstain if no violation is proven but at least one constraint remains Indeterminate.

No weighted average or strong easy-edit result can cover a failed hard invariant.

## Implemented contract

- Typed source families: Symbolic, Acoustic, HumanAdjudication and RendererBound.
- Every observation binds constraint ID, property, scope, envelope digest, candidate-output digest, source-lineage digest, evidence-source records and a typed payload.
- Typed payloads: text, scalar with unit/uncertainty, presence, before/after text, before/after numeric and numeric series.
- Relation support: EqualTo, DifferentFrom, WithinRange, MemberOfSet, NotMemberOfSet, Present, Absent, NoChange, MonotonicIncrease and MonotonicDecrease.
- Evidence-family matching; Combined requires at least two distinct families.
- Duplicate/extra observations, stale envelope/candidate/source binding, wrong property/scope and non-canonical observation order fail structurally.
- Missing observations create Indeterminate results and an overall Abstain.
- Numeric range uncertainty is conservative: the entire measurement interval must lie within the declared range. A boundary-straddling interval abstains; a fully out-of-range interval violates.
- Monotonicity requires a resolved direction and at least one actual increase/decrease. Reversals and flatlines do not qualify.
- Deterministic receipt commits envelope, candidate, source lineage, observation bundle, per-constraint outcomes and overall decision.
- Receipt replay re-runs the deterministic reducer and verifies all commitments. It does not rerun the external measurement adapters.
- Twelve authored unit tests cover acceptance/replay, no-op false-preservation, missing evidence, protected-lineage mutation, candidate/source substitution, duplicate/extra evidence, uncertainty boundary, out-of-range values, forbidden changes, monotonic reversal/flatline, Combined source families and receipt sensitivity.

## Important limits

A content digest proves that a particular evidence record was committed; it does not prove that the measurement method is accurate, independent, calibrated or perceptually valid. A Symbolic or Acoustic source-family label is also not an authentication of the tool that produced it. Human adjudication remains a distinct evidence family and requires a separate consented, blinded protocol.

The observation's uncertainty is not a license to widen an envelope. Range checks require the whole uncertainty interval to fit within the envelope; uncertainty crossing the boundary yields Abstain.

This tranche does not read candidate audio, invoke a renderer, decide musical quality, validate rights, or qualify any of the twelve operations. The synthetic fixture contains metadata and placeholder digests only; it is not a benchmark run.

## Prospective acceptance targets

- 100% deterministic relation/section decision coverage.
- Zero false acceptance on no-op, protected-content mutation, forbidden-change, source-substitution and missing-evidence controls.
- Exactly one result per declared constraint, no extras.
- No Acceptable outcome with an Indeterminate observation.
- Exact replay of each evaluation receipt from the same envelope and observation bundle.
- Independent validation of each observation adapter before any perceptual or production claim.

These are prospective goals, not results.
