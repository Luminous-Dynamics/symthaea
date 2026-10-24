# HABITAT-MEASUREMENT-GATE-001 — uncertainty-aware material acceptance

**Status:** synthetic decision-rule benchmark; no physical weighing instrument, certificate, or real material sample is connected.

## Paradigm shift

Exact ledger arithmetic is necessary but insufficient: a digital quantity may balance perfectly while the source measurement is missing, out of calibration, incorrectly converted, or too uncertain to justify a conformity decision. A robust engineering workflow needs four states, not binary pass/fail: `Conforming`, `Nonconforming`, `Indeterminate`, and `Blocked` (with missing evidence represented as `NotEvaluable`).

## Research basis

- [ISO 10012:2026](https://www.iso.org/standard/10012), published February 2026, defines a measurement-management framework intended to provide confidence in the validity and reliability of measurement results used in production, testing and service delivery.
- [NIST Handbook 44, 2026 edition](https://www.nist.gov/publications/nist-handbook-44-specifications-tolerances-and-other-technical-requirements-weighing-18), published January 5, 2026, covers specifications and tolerances for weighing/measuring devices, including accuracy and repeatability.
- GS1 EPCIS supports event sharing, sensor data and interoperable supply-chain visibility ([GS1 EPCIS/CBV](https://www.gs1.org/standards/epcis)). This fixture does not implement EPCIS or metrology-standard conformance.

These sources motivate the problem framing; they do not certify this prototype.

## Implemented synthetic decision contract

Fixture: `docs/research/fixtures/habitat-measurement-gate-v0.1.json`. Rust qualifier: `crates/domains/symthaea-habitat-vertical-bench/src/measurement_gate.rs`.

For an observation x and a declared deterministic uncertainty bound u, build [x-u, x+u] using checked integer arithmetic. Compare that interval against an inclusive specification interval [L,U]:

- `Conforming`: the entire observation interval is contained inside [L,U].
- `Nonconforming`: the observation interval and specification interval are disjoint.
- `Indeterminate`: the intervals overlap, but the observation interval is not fully contained.
- `Blocked`: incompatible units, inverted specification, out-of-validity calibration dates, or explicitly broken traceability chain.
- `NotEvaluable`: missing/invalid observation or uncertainty, missing certificate metadata, invalid dates, or arithmetic overflow/underflow.

The interval is a deterministic challenge bound, **not a probability distribution**, not automatically a statistical confidence interval, and not a claim that the bound is calibrated. Certificate and traceability labels are synthetic oracle state only. The qualifier does not verify instruments, calibration certificates, traceability to SI, uncertainty budgets or real measurements.

The 13 challenge cases cover conforming, nonconforming, interval-boundary indeterminate, expired calibration, absent uncertainty, mismatched units, missing certificate metadata, broken traceability, upper overflow, lower underflow, impossible measurement date, inverted limits, and calibration not yet valid. There are also 5 declared mutation controls.

## Proposed benchmarks (not achieved)

1. 100% correct disposition against independently calculated known answers.
2. Zero false-conforming decisions in adversarial boundary, invalid calibration, missing-evidence and unit-error controls.
3. 100% of missing/out-of-date/broken evidence surfaced as `NotEvaluable` or `Blocked`, never numeric zero or pass.
4. No conversion of uncertainty intervals into probabilities unless a separately specified statistical model and calibration evidence justify that model.
5. Exact deterministic output under case-array permutation; checked integer overflow/underflow must never be clamped into a pass.
6. After real metrology is available, pre-register false-accept/false-reject rate targets by application risk, test repeatability/reproducibility, and validate the decision rule against traceable reference standards. Targets depend on the risk and measurement process; do not invent one universal acceptance rate.

## Next integration gate

Map a real signed/calibrated weighment record to one ledger opening, transfer, or closing quantity. Include instrument identity, certificate and validity, measurement procedure, unit and conversion authority, raw reading, tare, resolution, uncertainty budget, and source custody. Compare the admitted interval with the ledger quantity; preserve all ambiguous cases as indeterminate until an authorised review or additional measurement resolves them.
