# PIE-009T — Metrology traceability pyramids and reference-standard aging

Status: independent structural reference only.

## Claim under test
A measurement chain can be internally consistent yet still fail a decision because the traceability path is missing, expired, import-dependent, or too imprecise after conservative bounded-error accumulation.

The reference distinguishes:

- `QualifiedLocalTraceability`: complete path terminates at a declared local primary anchor and the total conservative bound fits the decision allowance;
- `QualifiedImportTraceability`: complete path terminates at a declared imported primary anchor and fits the allowance;
- `InsufficientPrecision`: a complete path exists but the accumulated bound exceeds the decision allowance;
- `Expired`: at least one path element is beyond its declared validity interval;
- `Unreachable`: no complete primary-anchor path exists.

## Conservative reference semantics
For a reference element evaluated at step `t`:

`bound(t) = bound_at_calibration + drift_bound_per_step * (t - calibrated_at)`

The decision-level path bound is the sum of all reference-element bounds from the working/shop reference through its primary anchor. No probability distribution, covariance cancellation, or uncertainty reduction is invented.

Primary anchors have no parent. Derived references require an explicit parent. Missing parents, unsupported cycles, direct self-reference, malformed timing, non-finite values, and negative bounds fail closed.

Exact equality with the decision allowance is accepted using a fixed comparison epsilon (`1e-12`) so ordinary floating representation cannot turn a mathematically equal synthetic fixture into a false rejection.

## Independent execution evidence
The final Python candidate was executed locally on 2026-09-13 and returned:

`ok`

The executed fixture covers imported-anchor qualification, local-anchor qualification, locally anchored but insufficiently precise traceability, conservative aging, exact-bound acceptance, expiry, missing-root rejection, cycle rejection, deterministic metadata-sensitive evidence digests, and malformed numeric rejection.

## Important non-claims
This reference does not prove any declared local primary anchor is physically valid, SI-traceable, environmentally stable, manufacturable, or suitable for lunar/Mars operations. It does not implement GUM covariance analysis, calibration physics, certification, cyber authenticity, artifact-aging prediction, or hardware control.

Its purpose is narrower: freeze the bookkeeping theorem that local measurement autonomy requires a complete, temporally valid, sufficiently precise traceability path—not merely a working sensor or calibration bench.
