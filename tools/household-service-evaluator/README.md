# Household Service Evaluator

A small, offline Rust evaluation core with Serde-serializable evidence inputs and result receipts for two bounded household-service questions:

1. Is a metered water balance compatible with its declared residual error bound, calibration evidence and topology-audit reference?
2. Does a repair attempt meet its declared prerequisites, post-repair qualification and follow-up gate?

It is intentionally separate from the fixture-integrity validator in tools/household-bench-fixtures.

## Water balance

The signed equation is:

    inlet - tank_storage_delta - delivered_outlet - purge_and_overflow

A positive storage delta means the tank gained water; a negative delta means it lost water. The evaluator checks that four readings share one time window and use distinct stream and boundary identifiers. Each reading must have a qualified calibration state and a calibration-record digest reference before the result can be treated as calibration-qualified.

The input carries a conservative aggregate absolute error bound for the complete residual. This crate does not assume independent errors, calculate covariance, or infer a residual uncertainty from a single meter's uncertainty. The supplied bound must come from the applicable metrology procedure. The WCONT-008 fixture's single `meter_uncertainty_l` field has ambiguous scope; its integration test conservatively sums four 5 L per-term bounds to a 20 L residual bound and records that interpretation explicitly.

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

Public input and result types use explicit Serde contracts; enum dispositions serialize in snake_case for downstream report tooling. This is a serialization contract, not an authenticated evidence receipt.

No network access, cloud service, live actuator, or safety-control authority is present.


## Batch evaluation of the 20 frozen scenarios

Build and evaluate both v0.1 corpora in one offline command:

~~~bash
cargo run --locked --manifest-path tools/household-service-evaluator/Cargo.toml -- \
  --water docs/research/household-services/fixtures/water-continuity-scenarios-v0.1.json \
  --repair docs/research/household-services/fixtures/repairability-scenarios-v0.1.json \
  --output -
~~~

Use a file path instead of --output - to write a deterministic JSON report:

~~~bash
cargo run --locked --manifest-path tools/household-service-evaluator/Cargo.toml -- \
  --water docs/research/household-services/fixtures/water-continuity-scenarios-v0.1.json \
  --repair docs/research/household-services/fixtures/repairability-scenarios-v0.1.json \
  --output target/household-service-batch-report.json
~~~

The report contains 10 water receipts and 10 repair receipts, corpus IDs, the fixture-declared disposition as reference metadata, evaluator dispositions, deterministic derived metrics, evidence notes, unresolved claims, per-disposition counts and the source claim ceilings. The expected fixture disposition is never copied into the evaluator result.

The batch evaluator rejects a corpus with the wrong schema version/status/ID, an empty claim ceiling, missing/duplicate/unknown scenario IDs, wrong scenario count or missing contract fields. It has no clock, network, cloud, actuator or product-safety authority.

Selected software-only calculations:
- WCONT-001 integrates its declared constant 8 L/min over 24 h to 11,520 L, while withholding service qualification because no outlet/demand/pressure/quality evidence exists.
- WCONT-003 integrates 0.8 L/min over 12 h to 576 L only under the explicit constant-flow synthetic assumption; low-pressure time series and delivered volume remain unresolved.
- WCONT-004 reports the 82 L difference between the two fixture readings without selecting either as ground truth.
- WCONT-005 integrates 0.5 L/min over 8 h to 240 L as unexplained flow, not proven leakage.
- WCONT-008 recomputes -200 L and uses the conservative 20 L aggregate error bound (four 5 L per-term bounds) because the source fixture does not define the uncertainty field's scope. Its result remains synthetic and unqualified.
- REPAIR-006/007/009 remain blocked; REPAIR-008 preserves observed recurrence; REPAIR-010 remains held; REPAIR-005 records that offline artifacts are available but does not claim a rollback was performed.

The CI workflow runs this command and uploads the resulting JSON as a synthetic batch-report artifact. Artifact existence is not physical validation.
