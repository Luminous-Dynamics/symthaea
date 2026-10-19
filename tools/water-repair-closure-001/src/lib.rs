use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const INPUT_SCHEMA_VERSION: &str = "water-repair-closure-001-inputs-v1";
pub const ORACLE_SCHEMA_VERSION: &str = "water-repair-closure-001-oracle-v1";
pub const CLAIM_CEILING: &str = "SyntheticFieldRepairLifecycleOnly";
pub const EXPECTED_CASE_COUNT: usize = 22;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputCorpus {
    pub schema_version: String,
    pub qualification_status: String,
    pub claim_ceiling: String,
    pub cases: Vec<RepairCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RepairCase {
    pub case_id: String,
    pub profile: Profile,
    pub profile_requirements_reviewed: bool,
    pub asset_id: String,
    pub configuration_id: String,
    pub event: EventEvidence,
    pub authorization_state: AuthorizationState,
    pub as_found: MeasurementEvidence,
    pub as_left: MeasurementEvidence,
    pub component: ComponentEvidence,
    pub media: MediaEvidence,
    pub commissioning: CommissioningEvidence,
    pub water_quality: WaterQualityEvidence,
    pub redundancy: RedundancyEvidence,
    pub residual: ResidualEvidence,
    pub service_outcome: ServiceOutcomeEvidence,
    pub privacy: PrivacyEvidence,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    NonPotableModule,
    MunicipalDistribution,
    BuildingPlumbing,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EventEvidence {
    pub event_id: String,
    pub content_sha256: String,
    pub delivered_copies: u32,
    pub content_conflict: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AuthorizationState {
    Valid,
    Missing,
    Expired,
    Revoked,
    ScopeMismatch,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementEvidence {
    pub present: bool,
    pub asset_matches: bool,
    pub kind: MeasurementKind,
    pub current: bool,
    pub instrument_qualified: bool,
    pub unit_status: UnitStatus,
    pub within_window: bool,
    pub passed: bool,
    pub uncertainty: MeasurementUncertainty,
    pub conflicting: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum MeasurementKind {
    Observed,
    Simulated,
    Missing,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnitStatus {
    Valid,
    Missing,
    Unknown,
    ConversionInvalid,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementUncertainty {
    BoundedAcceptable,
    Wide,
    Unknown,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentEvidence {
    pub required: bool,
    pub identity_known: bool,
    pub compatible: bool,
    pub qualified: bool,
    pub reconditioned: bool,
    pub requalified: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MediaEvidence {
    pub required: bool,
    pub identity_known: bool,
    pub current: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommissioningEvidence {
    pub required: bool,
    pub complete: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterQualityEvidence {
    pub required: bool,
    pub clearance_present: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RedundancyEvidence {
    pub independence_claimed: bool,
    pub independent_failure_domain_evidenced: bool,
    pub common_mode_present: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualEvidence {
    pub required: bool,
    pub disposition_known: bool,
    pub recovery_claimed: bool,
    pub recovery_qualified: bool,
    pub input_mass_g: u64,
    pub recovered_mass_g: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceOutcomeEvidence {
    pub required: bool,
    pub observed: bool,
    pub model_digest_matches: bool,
    pub time_aligned: bool,
    pub protected_floor_met: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrivacyEvidence {
    pub public_projection_requested: bool,
    pub location_restricted: bool,
    pub access_policy_allows: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum Disposition {
    Blocked,
    Incomplete,
    Indeterminate,
    ReadyForReview,
    VerifiedWithinProfile,
    Rejected,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CaseResult {
    pub case_id: String,
    pub disposition: Disposition,
    pub reason_codes: Vec<String>,
    pub semantic_effect_count: u8,
    pub duplicate_delivery_idempotent: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OracleCorpus {
    pub schema_version: String,
    pub independence_note: String,
    pub qualification_status: String,
    pub claim_ceiling: String,
    pub results: Vec<CaseResult>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunReceipt {
    pub schema_version: String,
    pub qualification_status: String,
    pub claim_ceiling: String,
    pub implementation_sha: String,
    pub inputs_sha256: String,
    pub oracle_sha256: String,
    pub case_count: usize,
    pub mismatches: Vec<String>,
    pub results: Vec<CaseResult>,
}

pub fn validate_input_corpus(corpus: &InputCorpus) -> Result<(), String> {
    if corpus.schema_version != INPUT_SCHEMA_VERSION {
        return Err(format!("unsupported input schema: {}", corpus.schema_version));
    }
    if corpus.qualification_status != "NotRun" {
        return Err("frozen input fixture must remain NotRun".to_owned());
    }
    if corpus.claim_ceiling != CLAIM_CEILING {
        return Err("input claim ceiling mismatch".to_owned());
    }
    if corpus.cases.len() != EXPECTED_CASE_COUNT {
        return Err(format!(
            "expected {EXPECTED_CASE_COUNT} input cases, found {}",
            corpus.cases.len()
        ));
    }

    let mut case_ids = BTreeSet::new();
    let mut asset_ids = BTreeSet::new();
    let mut event_ids = BTreeSet::new();
    for case in &corpus.cases {
        if !case_ids.insert(case.case_id.as_str()) {
            return Err(format!("duplicate case ID: {}", case.case_id));
        }
        if !asset_ids.insert(case.asset_id.as_str()) {
            return Err(format!("duplicate asset ID: {}", case.asset_id));
        }
        if !event_ids.insert(case.event.event_id.as_str()) {
            return Err(format!("duplicate event ID: {}", case.event.event_id));
        }
        if case.asset_id.trim().is_empty() || case.configuration_id.trim().is_empty() {
            return Err(format!("empty asset/configuration identity in {}", case.case_id));
        }
        if case.event.event_id.trim().is_empty() {
            return Err(format!("empty event ID in {}", case.case_id));
        }
        if case.event.delivered_copies == 0 {
            return Err(format!("event delivered zero times in {}", case.case_id));
        }
        if !is_sha256_hex(&case.event.content_sha256) {
            return Err(format!("invalid event content SHA-256 in {}", case.case_id));
        }
    }
    Ok(())
}

pub fn validate_oracle(oracle: &OracleCorpus) -> Result<(), String> {
    if oracle.schema_version != ORACLE_SCHEMA_VERSION {
        return Err(format!("unsupported oracle schema: {}", oracle.schema_version));
    }
    if oracle.qualification_status != "NotRun" {
        return Err("frozen expected-output oracle must remain NotRun".to_owned());
    }
    if oracle.claim_ceiling != CLAIM_CEILING {
        return Err("oracle claim ceiling mismatch".to_owned());
    }
    if oracle.independence_note.trim().is_empty() {
        return Err("oracle independence note is required".to_owned());
    }
    if oracle.results.len() != EXPECTED_CASE_COUNT {
        return Err(format!(
            "expected {EXPECTED_CASE_COUNT} oracle rows, found {}",
            oracle.results.len()
        ));
    }
    let mut ids = BTreeSet::new();
    for expected in &oracle.results {
        if !ids.insert(expected.case_id.as_str()) {
            return Err(format!("duplicate oracle case ID: {}", expected.case_id));
        }
        if expected.reason_codes.is_empty() {
            return Err(format!("oracle reason codes are empty for {}", expected.case_id));
        }
        if expected.semantic_effect_count > 1 {
            return Err(format!("oracle effect count exceeds one for {}", expected.case_id));
        }
    }
    Ok(())
}

pub fn evaluate_all(corpus: &InputCorpus) -> Result<Vec<CaseResult>, String> {
    validate_input_corpus(corpus)?;
    let mut results: Vec<CaseResult> = corpus.cases.iter().map(evaluate_case).collect();
    results.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    Ok(results)
}

pub fn evaluate_case(case: &RepairCase) -> CaseResult {
    let duplicate_delivery_idempotent =
        case.event.delivered_copies > 1 && !case.event.content_conflict;
    let semantic_effect_count = if case.event.content_conflict { 0 } else { 1 };

    let finish = |disposition, reason: &str| CaseResult {
        case_id: case.case_id.clone(),
        disposition,
        reason_codes: vec![reason.to_owned()],
        semantic_effect_count,
        duplicate_delivery_idempotent,
    };

    if case.event.content_conflict {
        return finish(Disposition::Rejected, "EVENT_ID_CONTENT_CONFLICT");
    }
    if case.privacy.public_projection_requested
        && case.privacy.location_restricted
        && !case.privacy.access_policy_allows
    {
        return finish(Disposition::Rejected, "ACCESS_POLICY_VIOLATION");
    }

    match case.authorization_state {
        AuthorizationState::Missing => {
            return finish(Disposition::Rejected, "AUTHORIZATION_MISSING_OR_MISMATCHED");
        }
        AuthorizationState::Expired
        | AuthorizationState::Revoked
        | AuthorizationState::ScopeMismatch => {
            return finish(Disposition::Blocked, "AUTHORIZATION_NOT_CURRENT_OR_IN_SCOPE");
        }
        AuthorizationState::Valid => {}
    }

    if !case.as_found.asset_matches || !case.as_left.asset_matches {
        return finish(Disposition::Blocked, "ASSET_IDENTITY_MISMATCH");
    }
    if !case.as_found.present || case.as_found.kind != MeasurementKind::Observed {
        return finish(Disposition::Incomplete, "AS_FOUND_EVIDENCE_MISSING");
    }
    if case.as_left.conflicting || case.as_found.conflicting {
        return finish(Disposition::Indeterminate, "CONFLICTING_OBSERVATIONS");
    }
    if !case.as_left.present || case.as_left.kind == MeasurementKind::Missing {
        return finish(Disposition::Incomplete, "MISSING_AS_LEFT_EVIDENCE");
    }
    if case.as_left.kind != MeasurementKind::Observed {
        return finish(Disposition::Incomplete, "PHYSICAL_OBSERVATION_MISSING");
    }

    for measurement in [&case.as_found, &case.as_left] {
        match measurement.unit_status {
            UnitStatus::Missing | UnitStatus::Unknown => {
                return finish(Disposition::Rejected, "MISSING_OR_UNKNOWN_UNIT");
            }
            UnitStatus::ConversionInvalid => {
                return finish(Disposition::Rejected, "INVALID_UNIT_CONVERSION");
            }
            UnitStatus::Valid => {}
        }
        if !measurement.current || !measurement.instrument_qualified || !measurement.within_window {
            return finish(Disposition::Indeterminate, "STALE_OR_UNQUALIFIED_MEASUREMENT");
        }
        if measurement.uncertainty != MeasurementUncertainty::BoundedAcceptable {
            return finish(Disposition::Indeterminate, "MEASUREMENT_UNCERTAINTY_TOO_WIDE");
        }
        if !measurement.passed {
            return finish(Disposition::Rejected, "FUNCTION_TEST_FAILED");
        }
    }

    if case.water_quality.required {
        if !case.water_quality.clearance_present {
            return finish(
                Disposition::Blocked,
                "REQUIRED_WATER_QUALITY_CLEARANCE_MISSING",
            );
        }
        if !case.water_quality.passed {
            return finish(
                Disposition::Blocked,
                "REQUIRED_WATER_QUALITY_CLEARANCE_FAILED",
            );
        }
    }

    if case.media.required && (!case.media.identity_known || !case.media.current) {
        return finish(
            Disposition::Indeterminate,
            "MEDIA_IDENTITY_OR_CURRENTNESS_UNKNOWN",
        );
    }

    if case.component.required {
        if !case.component.identity_known {
            return finish(Disposition::Indeterminate, "PART_IDENTITY_UNKNOWN");
        }
        if !case.component.compatible || !case.component.qualified {
            return finish(Disposition::Blocked, "PART_COMPATIBILITY_UNQUALIFIED");
        }
        if case.component.reconditioned && !case.component.requalified {
            return finish(
                Disposition::Blocked,
                "RECONDITIONED_COMPONENT_NOT_REQUALIFIED",
            );
        }
    }

    if case.commissioning.required && !case.commissioning.complete {
        return finish(Disposition::Incomplete, "COMMISSIONING_MISSING");
    }

    if case.redundancy.independence_claimed
        && (!case.redundancy.independent_failure_domain_evidenced
            || case.redundancy.common_mode_present)
    {
        return finish(
            Disposition::Indeterminate,
            "REDUNDANCY_INDEPENDENCE_UNPROVEN",
        );
    }

    if case.residual.required && !case.residual.disposition_known {
        return finish(Disposition::Incomplete, "RESIDUAL_DISPOSITION_MISSING");
    }
    if case.residual.recovery_claimed {
        if !case.residual.recovery_qualified {
            return finish(Disposition::Indeterminate, "RECOVERY_NOT_QUALIFIED");
        }
        if case.residual.recovered_mass_g > case.residual.input_mass_g {
            return finish(Disposition::Rejected, "RECOVERED_MASS_EXCEEDS_INPUT_BOUNDARY");
        }
    }

    if case.service_outcome.required {
        if !case.service_outcome.observed {
            return finish(Disposition::Incomplete, "SERVICE_OUTCOME_NOT_OBSERVED");
        }
        if !case.service_outcome.model_digest_matches || !case.service_outcome.time_aligned {
            return finish(Disposition::Indeterminate, "TIME_ALIGNMENT_FAILURE");
        }
        if !case.service_outcome.protected_floor_met {
            return finish(Disposition::Rejected, "SERVICE_FLOOR_NOT_MET");
        }
    }

    if case.profile == Profile::BuildingPlumbing && !case.profile_requirements_reviewed {
        return finish(Disposition::ReadyForReview, "PROFILE_AUTHORITY_REVIEW_REQUIRED");
    }

    if duplicate_delivery_idempotent {
        finish(
            Disposition::VerifiedWithinProfile,
            "DUPLICATE_DELIVERY_IDEMPOTENT",
        )
    } else {
        finish(Disposition::VerifiedWithinProfile, "PROFILE_SCOPED_SUCCESS")
    }
}

pub fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUTS: &str = include_str!("../fixtures/inputs-v1.json");
    const ORACLE: &str = include_str!("../fixtures/oracle-v1.json");

    fn frozen_inputs() -> InputCorpus {
        serde_json::from_str(INPUTS).expect("input fixture must parse")
    }

    fn frozen_oracle() -> OracleCorpus {
        serde_json::from_str(ORACLE).expect("oracle fixture must parse")
    }

    #[test]
    fn all_frozen_cases_match_separate_expected_oracle() {
        let inputs = frozen_inputs();
        let oracle = frozen_oracle();
        validate_oracle(&oracle).expect("oracle contract valid");
        let actual = evaluate_all(&inputs).expect("input corpus valid");
        let mut expected = oracle.results;
        expected.sort_by(|left, right| left.case_id.cmp(&right.case_id));
        assert_eq!(actual, expected, "per-case outputs must match the independent oracle");
    }

    #[test]
    fn frozen_inputs_have_unique_ids_and_expected_count() {
        let corpus = frozen_inputs();
        validate_input_corpus(&corpus).expect("input corpus valid");
        assert_eq!(corpus.cases.len(), EXPECTED_CASE_COUNT);
    }

    #[test]
    fn input_order_does_not_change_canonical_results() {
        let mut corpus = frozen_inputs();
        let baseline = evaluate_all(&corpus).expect("baseline valid");
        corpus.cases.reverse();
        let permuted = evaluate_all(&corpus).expect("permuted valid");
        assert_eq!(baseline, permuted);
    }

    #[test]
    fn removing_required_as_left_evidence_cannot_improve_disposition() {
        let mut case = frozen_inputs().cases.into_iter().find(|c| c.case_id == "WR-01").unwrap();
        case.as_left.present = false;
        assert_eq!(evaluate_case(&case).disposition, Disposition::Incomplete);
    }

    #[test]
    fn widening_measurement_uncertainty_cannot_strengthen_claim() {
        let mut case = frozen_inputs().cases.into_iter().find(|c| c.case_id == "WR-01").unwrap();
        case.as_left.uncertainty = MeasurementUncertainty::Wide;
        assert_eq!(evaluate_case(&case).disposition, Disposition::Indeterminate);
    }

    #[test]
    fn duplicate_delivery_has_one_semantic_effect() {
        let case = frozen_inputs().cases.into_iter().find(|c| c.case_id == "WR-04").unwrap();
        let result = evaluate_case(&case);
        assert_eq!(result.disposition, Disposition::VerifiedWithinProfile);
        assert_eq!(result.semantic_effect_count, 1);
        assert!(result.duplicate_delivery_idempotent);
    }

    #[test]
    fn conflicted_event_has_no_semantic_effect() {
        let case = frozen_inputs().cases.into_iter().find(|c| c.case_id == "WR-05").unwrap();
        let result = evaluate_case(&case);
        assert_eq!(result.disposition, Disposition::Rejected);
        assert_eq!(result.semantic_effect_count, 0);
    }

    #[test]
    fn building_plumbing_profile_without_review_cannot_be_verified() {
        let mut case = frozen_inputs().cases.into_iter().find(|c| c.case_id == "WR-01").unwrap();
        case.profile = Profile::BuildingPlumbing;
        case.profile_requirements_reviewed = false;
        assert_eq!(evaluate_case(&case).disposition, Disposition::ReadyForReview);
    }

    #[test]
    fn missing_water_quality_clearance_blocks_required_profile() {
        let mut case = frozen_inputs().cases.into_iter().find(|c| c.case_id == "WR-01").unwrap();
        case.water_quality.required = true;
        case.water_quality.clearance_present = false;
        assert_eq!(evaluate_case(&case).disposition, Disposition::Blocked);
    }

    #[test]
    fn failed_protected_service_floor_is_not_verified() {
        let mut case = frozen_inputs().cases.into_iter().find(|c| c.case_id == "WR-02").unwrap();
        case.service_outcome.protected_floor_met = false;
        assert_eq!(evaluate_case(&case).disposition, Disposition::Rejected);
    }

    #[test]
    fn out_of_boundary_recovery_mass_is_rejected() {
        let mut case = frozen_inputs().cases.into_iter().find(|c| c.case_id == "WR-01").unwrap();
        case.residual.recovery_claimed = true;
        case.residual.recovery_qualified = true;
        case.residual.input_mass_g = 100;
        case.residual.recovered_mass_g = 101;
        assert_eq!(evaluate_case(&case).disposition, Disposition::Rejected);
    }

    #[test]
    fn malformed_content_digest_fails_corpus_validation() {
        let mut corpus = frozen_inputs();
        corpus.cases[0].event.content_sha256 = "not-a-digest".to_owned();
        assert!(validate_input_corpus(&corpus).is_err());
    }

    #[test]
    fn wrong_schema_version_fails_closed() {
        let mut corpus = frozen_inputs();
        corpus.schema_version = "future-or-foreign-schema".to_owned();
        assert!(validate_input_corpus(&corpus).is_err());
    }
}
