//! Executable, fail-closed dual-axis RF assessment contract.
//!
//! Acquisition qualification and recognition are separate inputs. A recognition
//! candidate never repairs an unqualified capture chain. This module reuses the
//! EMSR evidence crate's dataset, split, dependency and receipt semantics; it is
//! not a receiver driver or RF classifier.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;

use super::{
    assess_common_mode_independence, sha256_hex, EvidenceSourceProfile,
    IndependenceSummary, PairwiseIndependence, PairwiseRelation, ValidationError,
};

pub const FIXTURE_SCHEMA: &str = "emsr.rf-dual-axis-fixture-pack";
pub const FIXTURE_VERSION: &str = "0.1.0";
pub const FIXTURE_STATE: &str = "EXECUTABLE_SYNTHETIC_FIXTURES";
pub const PROFILE_ID: &str = "EMSR-RF-OPENSET-001";
pub const REPORT_SCHEMA: &str = "emsr-rf-dual-axis-conformance-report/0.1";

const EXPECTED_FIXTURE_IDS: [&str; 12] = [
    "RF-AXIS-001",
    "RF-AXIS-002",
    "RF-AXIS-003",
    "RF-AXIS-004",
    "RF-AXIS-005",
    "RF-AXIS-006",
    "RF-AXIS-007",
    "RF-AXIS-008",
    "RF-AXIS-009",
    "RF-AXIS-010",
    "RF-AXIS-011",
    "RF-AXIS-012",
];

const CLAIM_CEILING: &str = "Synthetic dual-axis semantic conformance only. The evaluator does not qualify physical RF hardware or measurements, authenticate evidence issuers, establish emitter identity/intent/location, establish GNSS/PNT integrity, or authorize operational actions.";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum AcquisitionQualification {
    Qualified,
    Degraded,
    Unqualified,
    Unknown,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ClockQuality {
    Qualified,
    Degraded,
    Unknown,
    NotApplicable,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum AcquisitionCheckResult {
    Pass,
    Fail,
    Unknown,
    NotRun,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum RecognitionDisposition {
    KnownClass,
    UnknownClass,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ConsumerDisposition {
    KnownClass,
    UnknownClass,
    Indeterminate,
    DegradedAcquisition,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcquisitionCheck {
    check_id: String,
    result: AcquisitionCheckResult,
    evidence_ref: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcquisitionAxis {
    qualification: AcquisitionQualification,
    acquisition_profile_id: String,
    receiver_id: Option<String>,
    front_end_id: Option<String>,
    firmware_digest: String,
    configuration_digest: String,
    clock_quality: ClockQuality,
    checks: Vec<AcquisitionCheck>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecognitionAxis {
    disposition: RecognitionDisposition,
    class_vocabulary_digest: String,
    model_digest: String,
    input_window_digest: String,
    candidate_class: Option<String>,
    confidence: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DerivedAxis {
    consumer_disposition: ConsumerDisposition,
    candidate_admitted: bool,
    reason_codes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceBindings {
    acquisition_evidence_refs: Vec<String>,
    recognition_evidence_refs: Vec<String>,
    as_of: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Assessment {
    assessment_id: String,
    profile_id: String,
    acquisition: AcquisitionAxis,
    recognition: RecognitionAxis,
    derived: DerivedAxis,
    evidence: EvidenceBindings,
    claim_ceiling: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedOutcome {
    consumer_disposition: ConsumerDisposition,
    candidate_admitted: bool,
    reason_code: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    id: String,
    assessment: Assessment,
    expected: ExpectedOutcome,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixturePack {
    schema: String,
    version: String,
    state: String,
    fixtures: Vec<Fixture>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DerivedOutcome {
    consumer_disposition: ConsumerDisposition,
    candidate_admitted: bool,
    reason_code: &'static str,
}

impl DerivedOutcome {
    fn as_value(&self) -> Value {
        json!({
            "consumer_disposition": self.consumer_disposition,
            "candidate_admitted": self.candidate_admitted,
            "reason_code": self.reason_code,
        })
    }
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn digest_with_prefix(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|digest| super::is_lower_hex(digest, 64))
}

fn unique_nonempty(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| nonempty(value) && seen.insert(value))
}

fn validate_assessment(assessment: &Assessment, fixture_id: &str) -> Result<(), ValidationError> {
    let path = format!("$.fixtures[{fixture_id}].assessment");
    if !nonempty(&assessment.assessment_id) {
        return Err(ValidationError::new(
            format!("{path}.assessment_id"),
            "assessment ID must be non-empty",
        ));
    }
    if assessment.profile_id != PROFILE_ID {
        return Err(ValidationError::new(
            format!("{path}.profile_id"),
            format!("expected {PROFILE_ID:?}"),
        ));
    }
    if !nonempty(&assessment.acquisition.acquisition_profile_id) {
        return Err(ValidationError::new(
            format!("{path}.acquisition.acquisition_profile_id"),
            "acquisition profile ID must be non-empty",
        ));
    }
    if assessment
        .acquisition
        .receiver_id
        .as_deref()
        .is_some_and(|value| !nonempty(value))
        || assessment
            .acquisition
            .front_end_id
            .as_deref()
            .is_some_and(|value| !nonempty(value))
    {
        return Err(ValidationError::new(
            format!("{path}.acquisition"),
            "receiver and front-end IDs must be non-empty when present",
        ));
    }
    for (field, digest) in [
        (
            "firmware_digest",
            assessment.acquisition.firmware_digest.as_str(),
        ),
        (
            "configuration_digest",
            assessment.acquisition.configuration_digest.as_str(),
        ),
        (
            "class_vocabulary_digest",
            assessment.recognition.class_vocabulary_digest.as_str(),
        ),
        ("model_digest", assessment.recognition.model_digest.as_str()),
        (
            "input_window_digest",
            assessment.recognition.input_window_digest.as_str(),
        ),
    ] {
        if !digest_with_prefix(digest) {
            return Err(ValidationError::new(
                format!("{path}.{field}"),
                "expected sha256: followed by 64 lowercase hexadecimal characters",
            ));
        }
    }

    if assessment.recognition.confidence.is_some_and(|value| {
        !value.is_finite() || !(0.0..=1.0).contains(&value)
    }) {
        return Err(ValidationError::new(
            format!("{path}.recognition.confidence"),
            "confidence must be null or a finite number in [0, 1]",
        ));
    }
    match assessment.recognition.disposition {
        RecognitionDisposition::KnownClass => {
            if assessment
                .recognition
                .candidate_class
                .as_deref()
                .is_none_or(|value| !nonempty(value))
            {
                return Err(ValidationError::new(
                    format!("{path}.recognition.candidate_class"),
                    "KNOWN_CLASS requires a non-empty candidate class",
                ));
            }
        }
        RecognitionDisposition::UnknownClass | RecognitionDisposition::Indeterminate => {
            if assessment.recognition.candidate_class.is_some() {
                return Err(ValidationError::new(
                    format!("{path}.recognition.candidate_class"),
                    "UNKNOWN_CLASS and INDETERMINATE must not carry an admitted candidate class",
                ));
            }
        }
    }

    if assessment.acquisition.checks.is_empty() {
        return Err(ValidationError::new(
            format!("{path}.acquisition.checks"),
            "at least one acquisition check is required",
        ));
    }
    let required_checks = ["calibration_current", "sample_integrity", "clock_quality"];
    let mut check_ids = BTreeSet::new();
    for (index, check) in assessment.acquisition.checks.iter().enumerate() {
        if !nonempty(&check.check_id) || !check_ids.insert(check.check_id.as_str()) {
            return Err(ValidationError::new(
                format!("{path}.acquisition.checks[{index}].check_id"),
                "check IDs must be non-empty and unique",
            ));
        }
        if check
            .evidence_ref
            .as_deref()
            .is_some_and(|value| !nonempty(value))
        {
            return Err(ValidationError::new(
                format!("{path}.acquisition.checks[{index}].evidence_ref"),
                "evidence reference must be null or non-empty",
            ));
        }
        if matches!(
            check.result,
            AcquisitionCheckResult::Pass | AcquisitionCheckResult::Fail
        ) && check
            .evidence_ref
            .as_deref()
            .is_none_or(|value| !assessment.evidence.acquisition_evidence_refs.contains(&value.to_owned()))
        {
            return Err(ValidationError::new(
                format!("{path}.acquisition.checks[{index}].evidence_ref"),
                "PASS/FAIL checks require an evidence reference present in acquisition_evidence_refs",
            ));
        }
    }
    for required in required_checks {
        if !check_ids.contains(required) {
            return Err(ValidationError::new(
                format!("{path}.acquisition.checks"),
                format!("required acquisition check {required:?} is missing"),
            ));
        }
    }

    let failed_checks = assessment
        .acquisition
        .checks
        .iter()
        .any(|check| check.result == AcquisitionCheckResult::Fail);
    let unknown_checks = assessment.acquisition.checks.iter().any(|check| {
        matches!(
            check.result,
            AcquisitionCheckResult::Unknown | AcquisitionCheckResult::NotRun
        )
    });
    match assessment.acquisition.qualification {
        AcquisitionQualification::Qualified => {
            if !matches!(
                assessment.acquisition.clock_quality,
                ClockQuality::Qualified | ClockQuality::NotApplicable
            ) || failed_checks
                || unknown_checks
                || assessment
                    .acquisition
                    .checks
                    .iter()
                    .any(|check| check.result != AcquisitionCheckResult::Pass)
                || assessment.evidence.acquisition_evidence_refs.is_empty()
            {
                return Err(ValidationError::new(
                    format!("{path}.acquisition.qualification"),
                    "QUALIFIED requires qualified/applicable clock quality, all required checks PASS, and acquisition evidence",
                ));
            }
        }
        AcquisitionQualification::Degraded => {
            if assessment.acquisition.clock_quality == ClockQuality::Qualified
                || !failed_checks
                || assessment.evidence.acquisition_evidence_refs.is_empty()
            {
                return Err(ValidationError::new(
                    format!("{path}.acquisition.qualification"),
                    "DEGRADED requires a non-qualified clock state, at least one failed check, and acquisition evidence",
                ));
            }
        }
        AcquisitionQualification::Unqualified => {
            if assessment.acquisition.clock_quality == ClockQuality::Qualified
                || !failed_checks
                || assessment.evidence.acquisition_evidence_refs.is_empty()
            {
                return Err(ValidationError::new(
                    format!("{path}.acquisition.qualification"),
                    "UNQUALIFIED requires a non-qualified clock state, at least one failed check, and acquisition evidence",
                ));
            }
        }
        AcquisitionQualification::Unknown => {
            if assessment.acquisition.clock_quality == ClockQuality::Qualified
                || failed_checks
                || !unknown_checks
            {
                return Err(ValidationError::new(
                    format!("{path}.acquisition.qualification"),
                    "UNKNOWN requires non-qualified clock quality, no asserted FAIL, and at least one UNKNOWN/NOT_RUN check",
                ));
            }
        }
    }

    if !unique_nonempty(&assessment.evidence.acquisition_evidence_refs)
        || !unique_nonempty(&assessment.evidence.recognition_evidence_refs)
    {
        return Err(ValidationError::new(
            format!("{path}.evidence"),
            "evidence references must be non-empty and unique",
        ));
    }
    if assessment.evidence.recognition_evidence_refs.is_empty() {
        return Err(ValidationError::new(
            format!("{path}.evidence.recognition_evidence_refs"),
            "recognition output requires at least one evidence reference",
        ));
    }
    if chrono::DateTime::parse_from_rfc3339(&assessment.evidence.as_of).is_err() {
        return Err(ValidationError::new(
            format!("{path}.evidence.as_of"),
            "as_of must be a valid RFC 3339 timestamp",
        ));
    }
    if assessment.claim_ceiling.len() < 3 || !unique_nonempty(&assessment.claim_ceiling) {
        return Err(ValidationError::new(
            format!("{path}.claim_ceiling"),
            "claim-ceiling entries must contain at least three non-empty unique statements",
        ));
    }
    Ok(())
}

/// Primary derivation: acquisition is checked first and is non-compensatory.
fn derive_primary(
    acquisition: AcquisitionQualification,
    recognition: RecognitionDisposition,
) -> DerivedOutcome {
    if acquisition != AcquisitionQualification::Qualified {
        let reason_code = match (acquisition, recognition) {
            (AcquisitionQualification::Degraded, RecognitionDisposition::KnownClass) => {
                "degraded_acquisition_overrides_known_candidate"
            }
            (AcquisitionQualification::Degraded, RecognitionDisposition::UnknownClass) => {
                "degraded_acquisition_overrides_unknown_candidate"
            }
            (AcquisitionQualification::Degraded, RecognitionDisposition::Indeterminate) => {
                "degraded_acquisition_overrides_indeterminate"
            }
            (AcquisitionQualification::Unqualified, RecognitionDisposition::KnownClass) => {
                "unqualified_acquisition_blocks_candidate"
            }
            (AcquisitionQualification::Unqualified, RecognitionDisposition::UnknownClass) => {
                "unqualified_acquisition_blocks_unknown_claim"
            }
            (AcquisitionQualification::Unqualified, RecognitionDisposition::Indeterminate) => {
                "unqualified_acquisition_takes_precedence"
            }
            (AcquisitionQualification::Unknown, RecognitionDisposition::KnownClass) => {
                "missing_acquisition_evidence_blocks_class_admission"
            }
            (AcquisitionQualification::Unknown, RecognitionDisposition::UnknownClass) => {
                "missing_acquisition_evidence_blocks_unknown_admission"
            }
            (AcquisitionQualification::Unknown, RecognitionDisposition::Indeterminate) => {
                "unknown_acquisition_precedes_recognition"
            }
            (AcquisitionQualification::Qualified, _) => unreachable!("qualified handled above"),
        };
        return DerivedOutcome {
            consumer_disposition: ConsumerDisposition::DegradedAcquisition,
            candidate_admitted: false,
            reason_code,
        };
    }

    match recognition {
        RecognitionDisposition::KnownClass => DerivedOutcome {
            consumer_disposition: ConsumerDisposition::KnownClass,
            candidate_admitted: true,
            reason_code: "qualified_known_class",
        },
        RecognitionDisposition::UnknownClass => DerivedOutcome {
            consumer_disposition: ConsumerDisposition::UnknownClass,
            candidate_admitted: false,
            reason_code: "qualified_unknown_class",
        },
        RecognitionDisposition::Indeterminate => DerivedOutcome {
            consumer_disposition: ConsumerDisposition::Indeterminate,
            candidate_admitted: false,
            reason_code: "qualified_indeterminate",
        },
    }
}

/// Reference derivation: a complete 4x3 decision table, separate from the
/// primary acquisition-first branch structure.
fn derive_reference(
    acquisition: AcquisitionQualification,
    recognition: RecognitionDisposition,
) -> DerivedOutcome {
    use AcquisitionQualification as A;
    use ConsumerDisposition as C;
    use RecognitionDisposition as R;

    let (consumer_disposition, candidate_admitted, reason_code) = match (acquisition, recognition) {
        (A::Qualified, R::KnownClass) => (C::KnownClass, true, "qualified_known_class"),
        (A::Qualified, R::UnknownClass) => (C::UnknownClass, false, "qualified_unknown_class"),
        (A::Qualified, R::Indeterminate) => (C::Indeterminate, false, "qualified_indeterminate"),
        (A::Degraded, R::KnownClass) => (C::DegradedAcquisition, false, "degraded_acquisition_overrides_known_candidate"),
        (A::Degraded, R::UnknownClass) => (C::DegradedAcquisition, false, "degraded_acquisition_overrides_unknown_candidate"),
        (A::Degraded, R::Indeterminate) => (C::DegradedAcquisition, false, "degraded_acquisition_overrides_indeterminate"),
        (A::Unqualified, R::KnownClass) => (C::DegradedAcquisition, false, "unqualified_acquisition_blocks_candidate"),
        (A::Unqualified, R::UnknownClass) => (C::DegradedAcquisition, false, "unqualified_acquisition_blocks_unknown_claim"),
        (A::Unqualified, R::Indeterminate) => (C::DegradedAcquisition, false, "unqualified_acquisition_takes_precedence"),
        (A::Unknown, R::KnownClass) => (C::DegradedAcquisition, false, "missing_acquisition_evidence_blocks_class_admission"),
        (A::Unknown, R::UnknownClass) => (C::DegradedAcquisition, false, "missing_acquisition_evidence_blocks_unknown_admission"),
        (A::Unknown, R::Indeterminate) => (C::DegradedAcquisition, false, "unknown_acquisition_precedes_recognition"),
    };

    DerivedOutcome {
        consumer_disposition,
        candidate_admitted,
        reason_code,
    }
}

fn outcome_matches_derived(outcome: &DerivedOutcome, supplied: &DerivedAxis) -> bool {
    supplied.consumer_disposition == outcome.consumer_disposition
        && supplied.candidate_admitted == outcome.candidate_admitted
        && supplied.reason_codes.as_slice() == [outcome.reason_code]
}

fn outcome_matches_expected(outcome: &DerivedOutcome, expected: &ExpectedOutcome) -> bool {
    expected.consumer_disposition == outcome.consumer_disposition
        && expected.candidate_admitted == outcome.candidate_admitted
        && expected.reason_code == outcome.reason_code
}

/// Execute the frozen 12-case pack, requiring exact case discovery and
/// agreement between the primary evaluator, the independent decision table,
/// the source-declared derived result and the frozen expected outcome.
pub fn validate_rf_dual_axis_fixture_pack(source_bytes: &[u8]) -> Result<Value, ValidationError> {
    let pack: FixturePack = serde_json::from_slice(source_bytes).map_err(|error| {
        ValidationError::new("$", format!("invalid RF dual-axis fixture pack: {error}"))
    })?;

    if pack.schema != FIXTURE_SCHEMA
        || pack.version != FIXTURE_VERSION
        || pack.state != FIXTURE_STATE
    {
        return Err(ValidationError::new(
            "$",
            "unsupported RF dual-axis fixture schema, version, or execution state",
        ));
    }

    let expected_ids: BTreeSet<&str> = EXPECTED_FIXTURE_IDS.into_iter().collect();
    let mut seen = BTreeSet::new();
    let mut duplicate_ids = Vec::new();
    let mut discovered = Vec::new();
    let mut per_fixture = Vec::new();
    let mut failure_count = 0usize;
    let mut reference_disagreement_count = 0usize;
    let mut false_accept_count = 0usize;
    let mut high_confidence_fault_count = 0usize;

    for fixture in &pack.fixtures {
        discovered.push(fixture.id.clone());
        if !seen.insert(fixture.id.as_str()) {
            duplicate_ids.push(fixture.id.clone());
        }
        if !expected_ids.contains(fixture.id.as_str()) {
            failure_count += 1;
            per_fixture.push(json!({
                "id": fixture.id,
                "status": "unexpected_fixture_id",
                "passed": false
            }));
            continue;
        }

        validate_assessment(&fixture.assessment, &fixture.id)?;
        let primary = derive_primary(
            fixture.assessment.acquisition.qualification,
            fixture.assessment.recognition.disposition,
        );
        let reference = derive_reference(
            fixture.assessment.acquisition.qualification,
            fixture.assessment.recognition.disposition,
        );
        let reference_agrees = primary == reference;
        let submitted_derived_matches = outcome_matches_derived(&primary, &fixture.assessment.derived);
        let expected_matches = outcome_matches_expected(&primary, &fixture.expected);
        let false_accept = primary.candidate_admitted
            && (fixture.assessment.acquisition.qualification != AcquisitionQualification::Qualified
                || fixture.assessment.recognition.disposition != RecognitionDisposition::KnownClass);
        if false_accept {
            false_accept_count += 1;
        }
        if fixture.assessment.recognition.confidence.unwrap_or(0.0) >= 0.90
            && fixture.assessment.acquisition.qualification != AcquisitionQualification::Qualified
            && primary.consumer_disposition == ConsumerDisposition::DegradedAcquisition
        {
            high_confidence_fault_count += 1;
        }
        if !reference_agrees {
            reference_disagreement_count += 1;
        }

        let passed = reference_agrees
            && submitted_derived_matches
            && expected_matches
            && !false_accept;
        if !passed {
            failure_count += 1;
        }

        per_fixture.push(json!({
            "id": fixture.id,
            "acquisition_qualification": fixture.assessment.acquisition.qualification,
            "recognition_disposition": fixture.assessment.recognition.disposition,
            "recognition_confidence": fixture.assessment.recognition.confidence,
            "primary": primary.as_value(),
            "reference": reference.as_value(),
            "submitted_derived": {
                "consumer_disposition": fixture.assessment.derived.consumer_disposition,
                "candidate_admitted": fixture.assessment.derived.candidate_admitted,
                "reason_codes": fixture.assessment.derived.reason_codes,
            },
            "expected": {
                "consumer_disposition": fixture.expected.consumer_disposition,
                "candidate_admitted": fixture.expected.candidate_admitted,
                "reason_code": fixture.expected.reason_code,
            },
            "reference_agrees": reference_agrees,
            "submitted_derived_matches": submitted_derived_matches,
            "expected_matches": expected_matches,
            "false_accept": false_accept,
            "passed": passed
        }));
    }

    let discovery_complete = pack.fixtures.len() == EXPECTED_FIXTURE_IDS.len()
        && seen == expected_ids
        && duplicate_ids.is_empty();
    if !discovery_complete {
        failure_count += 1;
    }

    let passed = discovery_complete
        && duplicate_ids.is_empty()
        && failure_count == 0
        && reference_disagreement_count == 0
        && false_accept_count == 0;

    Ok(json!({
        "report_contract": REPORT_SCHEMA,
        "status": if passed { "passed" } else { "failed" },
        "authority": "synthetic_rf_dual_axis_semantic_conformance_only",
        "input_sha256": sha256_hex(source_bytes),
        "profile_id": PROFILE_ID,
        "fixture_version": pack.version,
        "fixture_count_expected": EXPECTED_FIXTURE_IDS.len(),
        "fixture_count_discovered": pack.fixtures.len(),
        "fixture_ids": discovered,
        "discovery_complete": discovery_complete,
        "duplicate_fixture_ids": duplicate_ids,
        "passed_count": per_fixture.iter().filter(|case| case["passed"] == true).count(),
        "failure_count": failure_count,
        "reference_disagreement_count": reference_disagreement_count,
        "false_accept_count": false_accept_count,
        "high_confidence_acquisition_fault_cases_blocked": high_confidence_fault_count,
        "verification": {
            "status": if passed { "passed" } else { "failed" },
            "method": "separate_4x3_reference_decision_table",
            "claim_ceiling": "Algorithmically distinct rule table in the same Rust package; not cross-language, organizational, custody or physical independence."
        },
        "per_fixture": per_fixture,
        "claim_ceiling": CLAIM_CEILING,
        "non_claims": [
            "not_physical_receiver_or_front_end_qualification",
            "not_RF_classification_performance",
            "not_emitter_identity_or_attribution",
            "not_GNSS_or_PNT_integrity",
            "not_ground_truth_authentication",
            "not_operational_authorization"
        ]
    }))
}

/// Reuse the EMSR/Mycelix common-mode implementation rather than introducing
/// a second RF-specific dependency graph or source-independence algorithm.
pub fn assess_rf_source_independence(
    left: &EvidenceSourceProfile,
    right: &EvidenceSourceProfile,
) -> Result<PairwiseIndependence, ValidationError> {
    let summary: IndependenceSummary =
        assess_common_mode_independence(&[left.clone(), right.clone()])?;
    summary
        .pairs
        .into_iter()
        .next()
        .ok_or_else(|| ValidationError::new("$.sources", "two source profiles are required"))
}

#[cfg(test)]
mod tests {
    use super::{
        assess_rf_source_independence, validate_rf_dual_axis_fixture_pack, ConsumerDisposition,
    };
    use crate::{EvidenceSourceProfile, PairwiseRelation};
    use serde_json::{json, Value};

    const FIXTURES: &str =
        include_str!("../../../../docs/research/emsr-v0.1/emsr-rf-dual-axis-fixtures-v0.1.json");

    fn fixture_document() -> Value {
        serde_json::from_str(FIXTURES).expect("RF dual-axis fixtures parse")
    }

    #[test]
    fn exact_twelve_case_pack_executes_and_both_axes_agree() {
        let report = validate_rf_dual_axis_fixture_pack(FIXTURES.as_bytes())
            .expect("strict RF fixture pack validates");
        assert_eq!(report["status"], "passed", "{report:#}");
        assert_eq!(report["fixture_count_expected"], 12);
        assert_eq!(report["fixture_count_discovered"], 12);
        assert_eq!(report["passed_count"], 12);
        assert_eq!(report["failure_count"], 0);
        assert_eq!(report["reference_disagreement_count"], 0);
        assert_eq!(report["false_accept_count"], 0);
        assert_eq!(report["high_confidence_acquisition_fault_cases_blocked"], 1);
        assert_eq!(report["verification"]["status"], "passed");
    }

    #[test]
    fn unknown_fields_fail_closed() {
        let mut document = fixture_document();
        document["fixtures"][0]["assessment"]["acquisition"]["unexpected"] = json!("must-reject");
        let bytes = serde_json::to_vec(&document).unwrap();
        let error = validate_rf_dual_axis_fixture_pack(&bytes)
            .expect_err("unknown acquisition fields must not be silently ignored");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn high_confidence_known_candidate_cannot_override_degraded_acquisition() {
        let mut document = fixture_document();
        let assessment = &mut document["fixtures"][3]["assessment"];
        assessment["recognition"]["confidence"] = json!(0.999999);
        let bytes = serde_json::to_vec(&document).unwrap();
        let report = validate_rf_dual_axis_fixture_pack(&bytes)
            .expect("mutated confidence remains structurally valid");
        assert_eq!(report["status"], "passed", "{report:#}");
        assert_eq!(
            report["per_fixture"][3]["primary"]["consumer_disposition"],
            json!(ConsumerDisposition::DegradedAcquisition)
        );
        assert_eq!(report["per_fixture"][3]["primary"]["candidate_admitted"], false);
    }

    #[test]
    fn forged_derived_admission_on_degraded_capture_is_detected() {
        let mut document = fixture_document();
        let assessment = &mut document["fixtures"][3]["assessment"];
        assessment["derived"]["consumer_disposition"] = json!("KNOWN_CLASS");
        assessment["derived"]["candidate_admitted"] = json!(true);
        assessment["derived"]["reason_codes"] = json!(["high_confidence_class"]);
        let bytes = serde_json::to_vec(&document).unwrap();
        let report = validate_rf_dual_axis_fixture_pack(&bytes)
            .expect("forged derived output is reported as a conformance failure");
        assert_eq!(report["status"], "failed");
        assert_eq!(report["per_fixture"][3]["submitted_derived_matches"], false);
        assert_eq!(report["per_fixture"][3]["false_accept"], false);
    }

    #[test]
    fn receiver_ids_do_not_make_shared_clock_or_pipeline_domains_independent() {
        let left = EvidenceSourceProfile {
            snapshot_id: "rf-dependency-snapshot-1".to_owned(),
            as_of: "2026-11-08T07:00:00Z".to_owned(),
            source_id: "receiver-a/front-end-a".to_owned(),
            failure_domains: vec![
                "clock-distribution:shared-clock-1".to_owned(),
                "capture-pipeline:collector-1".to_owned(),
            ],
            dependencies_complete: true,
        };
        let right = EvidenceSourceProfile {
            snapshot_id: "rf-dependency-snapshot-1".to_owned(),
            as_of: "2026-11-08T07:00:00Z".to_owned(),
            source_id: "receiver-b/front-end-b".to_owned(),
            failure_domains: vec![
                "clock-distribution:shared-clock-1".to_owned(),
                "capture-pipeline:collector-1".to_owned(),
            ],
            dependencies_complete: true,
        };

        let result = assess_rf_source_independence(&left, &right)
            .expect("common-mode check accepts same snapshot and different source IDs");
        assert_eq!(result.relation, PairwiseRelation::SharedFailureDomain);
        assert_eq!(
            result.shared_failure_domains,
            vec![
                "capture-pipeline:collector-1".to_owned(),
                "clock-distribution:shared-clock-1".to_owned()
            ]
        );
    }

    #[test]
    fn different_ids_without_complete_inventory_remain_unknown() {
        let left = EvidenceSourceProfile {
            snapshot_id: "rf-dependency-snapshot-1".to_owned(),
            as_of: "2026-11-08T07:00:00Z".to_owned(),
            source_id: "receiver-a".to_owned(),
            failure_domains: vec!["clock-a".to_owned()],
            dependencies_complete: false,
        };
        let right = EvidenceSourceProfile {
            snapshot_id: "rf-dependency-snapshot-1".to_owned(),
            as_of: "2026-11-08T07:00:00Z".to_owned(),
            source_id: "receiver-b".to_owned(),
            failure_domains: vec!["clock-b".to_owned()],
            dependencies_complete: true,
        };

        let result = assess_rf_source_independence(&left, &right)
            .expect("same-snapshot profiles can be compared");
        assert_eq!(result.relation, PairwiseRelation::Unknown);
        assert!(result.shared_failure_domains.is_empty());
    }

    #[test]
    fn distinct_complete_domains_are_only_no_declared_common_domain() {
        let left = EvidenceSourceProfile {
            snapshot_id: "rf-dependency-snapshot-1".to_owned(),
            as_of: "2026-11-08T07:00:00Z".to_owned(),
            source_id: "receiver-a".to_owned(),
            failure_domains: vec!["clock-a".to_owned(), "pipeline-a".to_owned()],
            dependencies_complete: true,
        };
        let right = EvidenceSourceProfile {
            snapshot_id: "rf-dependency-snapshot-1".to_owned(),
            as_of: "2026-11-08T07:00:00Z".to_owned(),
            source_id: "receiver-b".to_owned(),
            failure_domains: vec!["clock-b".to_owned(), "pipeline-b".to_owned()],
            dependencies_complete: true,
        };

        let result = assess_rf_source_independence(&left, &right)
            .expect("same-snapshot profiles can be compared");
        assert_eq!(result.relation, PairwiseRelation::NoDeclaredCommonDomain);
    }

    #[test]
    fn mismatched_dependency_snapshots_cannot_be_compared() {
        let left = EvidenceSourceProfile {
            snapshot_id: "snapshot-a".to_owned(),
            as_of: "2026-11-08T07:00:00Z".to_owned(),
            source_id: "receiver-a".to_owned(),
            failure_domains: vec!["clock-a".to_owned()],
            dependencies_complete: true,
        };
        let right = EvidenceSourceProfile {
            snapshot_id: "snapshot-b".to_owned(),
            as_of: "2026-11-08T07:00:00Z".to_owned(),
            source_id: "receiver-b".to_owned(),
            failure_domains: vec!["clock-b".to_owned()],
            dependencies_complete: true,
        };

        let error = assess_rf_source_independence(&left, &right)
            .expect_err("mixed snapshot versions must be rejected");
        assert!(error.to_string().contains("same dependency snapshot"));
    }

    #[test]
    fn confidence_is_not_used_as_a_calibration_or_acquisition_predicate() {
        let mut document = fixture_document();
        document["fixtures"][9]["assessment"]["recognition"]["confidence"] = json!(1.0);
        let bytes = serde_json::to_vec(&document).unwrap();
        let report = validate_rf_dual_axis_fixture_pack(&bytes)
            .expect("confidence mutation is structurally valid");
        assert_eq!(report["status"], "passed");
        assert_eq!(
            report["per_fixture"][9]["primary"]["consumer_disposition"],
            json!("DEGRADED_ACQUISITION")
        );
    }
}
