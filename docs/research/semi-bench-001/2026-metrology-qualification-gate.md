# SEMI-CAL-001 — Fail-closed metrology qualification gate

## Why this exists

The evidence chain must not treat a numeric instrument result as ground truth simply because it was emitted by an instrument or attached to a wafer/MES record. A measurement's eligibility depends on instrument identity, measurand, unit, method revision, calibration validity, revocation state, reference-standard lineage and uncertainty.

This package inserts an explicit qualification decision before a measurement can be promoted into:
- a VM training/update label;
- an adaptive-metrology reward or policy outcome;
- a benchmark truth label;
- an engineering alert or recommendation.

The gate is intentionally separate from VM prediction and selection. It qualifies the evidence; it does not predict a physical value, decide which wafer to inspect, recalibrate an instrument or actuate equipment.

## Typed contract

The first implementation contains:
- CalibrationCertificateV1: certificate identity/digest, issuing authority reference, instrument/measurand/unit/method identity, reference artifact/certificate, traceability declaration, performed/issued/validity/revocation times, correction, standard uncertainty and certificate coverage factor;
- MeasurementEvidenceV1: event/source digest, instrument/measurand/unit/method identity, timestamp, raw/corrected values, applied correction and uncertainty components;
- MeasurementQualificationPolicyV1: policy version, maximum standard-uncertainty bound, reporting coverage factor and required traceability/reference evidence;
- MeasurementQualificationReportV1: status, stable reason codes, canonical measurement digest, external and parsed-certificate digests, policy digest, corrected value and uncertainty bounds.

The decision is three-valued:

| State | Meaning | Downstream treatment |
|---|---|---|
| Qualified | Supplied evidence satisfies this versioned policy | Eligible for a separately authorized downstream update |
| Unknown | Missing/unverified/insufficient evidence | Hold; do not promote |
| Rejected | Contradictory identity, expired/revoked calibration, invalid correction or out-of-policy uncertainty | Reject; do not promote |

A missing calibration certificate is UNKNOWN, not a default calibration. Unverified or limited traceability is UNKNOWN when the policy requires a claimed traceability declaration. Identity mismatch, invalid chronology, expired/revoked certificates and uncertainty above the configured limit are REJECTED.

## Uncertainty semantics

The V1 policy does not silently assume independent uncertainty components. In the absence of a validated covariance model it uses:

u_upper = u_calibration + sum_i(u_i)

This is a fully positively correlated upper bound. It is conservative and may be wider than root-sum-square, but cannot understate standard uncertainty merely because common-cause dependence is unknown. The expanded upper bound is U_upper = k × u_upper.

The next version may accept an explicit covariance/correlation matrix only with symmetry, unit-diagonal, bounded-correlation and positive-semidefinite validation. If the matrix is absent or invalid, V1's conservative bound or UNKNOWN must remain the fallback—never an implicit RSS assumption.

## Integrity and replay

The gate binds:
- the externally supplied measurement source-record digest;
- the canonical typed measurement record, with uncertainty components sorted by component_id;
- the externally issued calibration certificate digest;
- the structured parsed certificate record digest;
- the qualification policy digest;
- the final qualification report digest.

The structured certificate digest is important: mutating parsed fields such as calibration uncertainty while leaving the external certificate digest unchanged must still change the qualification report.

These digests provide content integrity and deterministic replay relative to trusted inputs. They do not authenticate the issuing authority, prove source-certificate authenticity or establish a traceability chain.

## Adversarial controls

The Rust tests cover:
- missing calibration certificate → UNKNOWN;
- instrument identity mismatch → REJECTED;
- expired and revoked calibration → REJECTED;
- unverified traceability → UNKNOWN;
- missing reference-standard certificate → UNKNOWN;
- inconsistent correction → REJECTED;
- missing uncertainty components → UNKNOWN;
- fully correlated upper-bound threshold breach → REJECTED;
- component-order permutation → identical canonical measurement/report digests;
- structured certificate mutation → changed parsed-record/report digest.

## Relationship to the other SEMI-BENCH lanes

- #13328 — policy replay, delayed labels and decision EVSI. The gate's QUALIFIED/UNKNOWN/REJECTED result should be a prerequisite for label promotion or reward updates.
- #13336 — independently replayable selection receipt V2. The selection receipt explains why a wafer was selected; the qualification receipt explains whether its measurement may be consumed.
- #13356 — MSA and variance components. Physical MSA evidence can eventually supply repeatability/reproducibility uncertainty inputs to this gate.
- #13273 — preregistered competitive and industrial qualification scorecard.

## Claim ceiling

A passing report proves only that the supplied evidence passes the encoded policy. ClaimedTraceable is an issuer/source declaration, not independently authenticated proof. Qualification does not prove physical instrument capability, genuine certificate origin, accreditation, standards conformity, causal validity or safe fab operation.

Industrial use requires trusted certificate bytes, a validated traceability chain, authorized source systems, approved uncertainty procedures and metrology-owner sign-off. No equipment-control or process-actuation authority is granted by this package.
