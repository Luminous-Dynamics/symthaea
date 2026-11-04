# Household Service Evaluator

A small, offline Rust evaluation core for two bounded household-service questions:

1. Is a metered water balance compatible with its declared residual error bound, calibration evidence and topology-audit reference?
2. Does a repair attempt meet its declared prerequisites, post-repair qualification and follow-up gate?

It is intentionally separate from the fixture-integrity validator in tools/household-bench-fixtures.

## Water balance

The signed equation is:

    inlet - tank_storage_delta - delivered_outlet - purge_and_overflow

A positive storage delta means the tank gained water; a negative delta means it lost water. The evaluator checks that four readings share one time window and use distinct stream and boundary identifiers. Each reading must have a qualified calibration state and a calibration-record digest reference before the result can be treated as calibration-qualified.

The input carries a conservative aggregate absolute error bound for the complete residual. This crate does not assume independent errors, calculate covariance, or infer a residual uncertainty from a single meter's uncertainty. The supplied bound must come from the applicable metrology procedure.

If zero lies inside the residual interval, the result is ZeroNotExcludedByDeclaredBound, not a physical pass. If it lies outside, the result is ContradictionBeyondDeclaredBound. A topology-audit report reference is separately required; a digest string is not proof that the topology is physically independent or that the report/auditor is trustworthy.

## Deliverable reserve

The duration estimator accepts only a bounded usable volume at the delivery boundary, a bounded positive demand rate, a qualified delivery flow/pressure envelope, an available pump/electrical path and a separate end-use quality qualification.

It computes:

    lower_hours = usable_volume_lower_l / demand_upper_l_per_hour
    upper_hours = usable_volume_upper_l / demand_lower_l_per_hour

It never accepts gross tank level as a substitute for usable deliverable volume. If the pump, electrical path, measured pressure, delivery qualification or end-use quality gate is missing, no duration is returned.

## Repair

The repair evaluator keeps blocked access/parts, incomplete requalification, recurrence, initial restoration and no recurrence during the observed follow-up interval distinct. It always prohibits a lifetime-reliability claim. For example, a 30-day follow-up supports only a statement about the specified 30-day observation window.

## Evidence and safety limits

- The core performs deterministic arithmetic and disposition logic only.
- SHA-256 fields are syntactically checked references; this crate does not authenticate the referenced bytes or signer.
- Identifier uniqueness does not prove physical measurement independence. The topology report must be independently audited.
- A calibration flag does not establish that a calibration procedure was scientifically suitable.
- The reserve estimator does not establish potability, pathogen safety, plumbing compliance or safe installation.
- A repair disposition is not a safety certification or authorization to work on energized, pressurized, hot or otherwise hazardous equipment.
- Synthetic tests establish only software behavior. They are not physical product evidence.

No network access, cloud service, live actuator, or safety-control authority is present.
