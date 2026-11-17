# Robot Lightshell qualification contract v0.1

**Owner:** Symthaea ROBOT-LIGHTSHELL-001 #12688  
**Implementation:** `symthaea-engineering::lightshell_qualification`  
**Status:** authored Rust admission logic and unit-test intentions; exact-head CI not yet passed  
**Authority:** evidence/benchmark admission only; never robot actuation, safety-control or certification authority

## 1. Problem this contract prevents

A benchmark can appear complete while hiding a failure if it stores one global gate state for several different candidate constructions. Lightshell compares materially different configurations: bare robot, rigid cover, commercial textile cover, textile-only, textile+spacer and textile+spacer+local guard.

Therefore every measurement and every hard gate is scoped to the exact candidate. One candidate's PASS cannot discharge another candidate's gate. One failed candidate must not be removed from the record merely because another variant performed well.

## 2. Data model change

The benchmark fixture now has:
- 6 distinct candidate records: B0, B1, B2, L1, L2 and L3;
- 42 metric rows: 7 metrics × 6 candidates;
- 54 hard-gate rows: 9 gates × 6 candidates;
- null observed metric values, targets and baselines while the benchmark is still inputs-pending;
- unknown gate states for every candidate;
- no test runs.

The schema now requires `candidate_id` in each metric row and each gate row. Duplicate candidate/metric and candidate/gate pairs are rejected.

## 3. Rust API

Module:
`crates/domains/symthaea-engineering/src/lightshell_qualification.rs`

Public entry point:

```rust
use symthaea_engineering::lightshell_qualification::parse_and_validate;

let report = parse_and_validate(manifest_json)?;
if !report.is_valid() {
    for finding in report.findings {
        eprintln!("{} {}: {}", finding.code, finding.path, finding.message);
    }
    return Err("Lightshell benchmark manifest rejected".into());
}
```

The returned `rankable_candidates` list is a **qualification eligibility gate**, not an ordering, score or statement of comparative superiority. A separate analysis stage would compute Pareto dominance or confidence-aware comparisons from qualified observed values.

## 4. Admission invariants

### Identity and completeness
- Only B0/B1/B2/L1/L2/L3 candidate IDs are admitted.
- Candidate IDs are unique and all six are present in v0.1.
- Required comparison metrics have a row for every candidate.
- All nine hard gates have a row for every candidate.
- Metric units and measurement boundaries are explicit.
- Unknown or duplicate metric/gate records are not silently accepted.

### Frozen protocol
A frozen, running or complete benchmark requires:
- exact CAD revision-bound or as-built-metrology-bound geometry;
- a non-empty operating domain, task profile and coverage boundary;
- an available and matched B0 bare-platform baseline;
- every candidate's comparability explicitly marked matched, partially matched or not comparable.

An unassessed comparator blocks protocol freeze. An unavailable or not-comparable candidate remains visible but is not silently treated as a matched competitor.

### Hard-gate evidence
Each candidate must resolve all nine gate identities. States are typed:
- unknown;
- not_run;
- pass;
- fail;
- not_applicable.

A PASS requires an acceptance-rule reference and reviewer reference. Unknown, not-run or failed states block ranking. A candidate needs at least one actual PASS; an all-not-applicable gate set cannot qualify a candidate.

### Physical-test evidence
A test marked passed requires:
- exact protocol revision;
- one or more instrument references;
- one or more calibration references;
- a SHA-256 raw-data digest;
- an acceptance-rule reference;
- exact candidate article marked available.

A failed or inconclusive run remains in the manifest and blocks that candidate from ranking. Invalidated runs remain recorded but do not count as physical qualification.

### Metric evidence
To become rankable, both B0 and the candidate require observed values for:
- added mass;
- range-of-motion delta;
- matched-trajectory joint energy delta;
- peak temperature delta.

Each of these observed values must bind a method and uncertainty-model reference. No target or baseline is invented in the seed fixture.

### Complete state
A complete benchmark additionally requires each available/matched candidate to have:
- all candidate-scoped hard gates passed or explicitly not applicable, with at least one pass;
- a passed integrated-robot test;
- required observed metrics with method and uncertainty references.

A completed coupon or module test is not a complete robot qualification.

## 5. Ranking eligibility

A candidate can enter comparative analysis only when:
1. B0 is available, matched, gate-qualified and physically tested;
2. candidate is available and matched;
3. candidate and B0 refer to the same declared coverage boundary;
4. all candidate-specific hard gates are resolved acceptably;
5. at least one passed module or integrated-robot test is admitted;
6. no failed or inconclusive test remains for that candidate;
7. required observed metrics are present with methods and uncertainty references;
8. the overall manifest has no structural or evidence-policy findings.

The validator does not calculate improvements, infer unobserved measurements, rank Pareto fronts, or certify a protection level.

## 6. Unit-test coverage authored

The module contains tests for:
- the seed fixture being valid but not rankable;
- candidate-specific failure not being hidden by a different candidate's PASS;
- ranking only against an exact matched B0 control and common coverage boundary;
- a passed test without calibration being rejected;
- frozen benchmark rejection when geometry/task requirements are absent;
- adverse failed test evidence preventing candidate ranking even if another run passed.

These tests are authored in source, but a source review or file fetch is not equivalent to running `cargo test`. CI status must be read at the exact current head; skipped/queued is not PASS.

## 7. Claim and authority boundary

This module:
- validates evidence structure and benchmark eligibility;
- does not evaluate textile mechanical performance;
- does not determine whether any design is safe;
- does not issue protection or IP/IK ratings;
- does not authorize a robot, actuator, machine or human-contact task;
- does not turn simulation, supplier declarations or synthetic fixtures into physical evidence.

The application safety owner, calibrated test system and independent engineering review remain authoritative.

## 8. Next implementation increments

1. Run the focused Rust tests on an exact branch head and repair any compile/test failures without weakening assertions.
2. Add mutation tests for duplicate/missing candidate rows, unknown states, invalid digest, stale calibration, failed/ambiguous results and coverage-boundary mismatch.
3. Add exact matched-trajectory measurement ingestion with raw-data digest, sample-clock and uncertainty records.
4. Add a separate confidence-aware Pareto analysis function, only after metric method and uncertainty schemas are approved.
5. Bind a real CAD/ODD revision from JointLab #4814 before moving the benchmark out of inputs-pending.
