use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApproachOutcome {
    Docked,
    Aborted,
    TimedOut,
    DeltaVExhausted,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LatchState {
    Engaged,
    NotEngaged,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RetentionState {
    Passed,
    Failed,
    NotRun,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApproachRecord {
    pub outcome: ApproachOutcome,
    pub distance_m: Option<f64>,
    pub speed_mps: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContactEvidence {
    pub frame_id: Option<String>,
    pub epoch_tick: Option<u64>,
    pub client_id: Option<String>,
    pub interface_id: Option<String>,
    pub interface_revision: Option<u64>,
    pub position_error_m: Option<f64>,
    pub attitude_error_rad: Option<f64>,
    pub relative_speed_mps: Option<f64>,
    pub relative_angular_rate_rad_s: Option<f64>,
    pub contact_impulse_ns: Option<f64>,
    pub end_effector_error_m: Option<f64>,
    pub collision_clearance_m: Option<f64>,
    pub joint_angles_rad: Option<Vec<f64>>,
    pub bus_angular_velocity_rad_s: Option<Vec<f64>>,
    pub latch_state: Option<LatchState>,
    pub retention_state: Option<RetentionState>,
    pub approach_receipt: Option<String>,
    pub contact_receipt: Option<String>,
    pub arm_receipt: Option<String>,
    pub latch_receipt: Option<String>,
    pub retention_receipt: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    #[serde(rename = "class")]
    pub case_class: String,
    pub expected: String,
    pub approach: ApproachRecord,
    pub contact: ContactEvidence,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub frame_id: String,
    pub client_id: String,
    pub interface_id: String,
    pub interface_revision: u64,
    pub epoch_tick: u64,
    pub max_epoch_skew_ticks: u64,
    pub max_approach_distance_m: f64,
    pub max_approach_speed_mps: f64,
    pub max_contact_position_error_m: f64,
    pub max_attitude_error_rad: f64,
    pub max_contact_speed_mps: f64,
    pub max_relative_angular_rate_rad_s: f64,
    pub max_contact_impulse_ns: f64,
    pub max_end_effector_error_m: f64,
    pub min_collision_clearance_m: f64,
    pub max_joint_abs_rad: f64,
    pub min_joint_limit_margin_rad: f64,
    pub max_bus_angular_rate_rad_s: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub suite_id: String,
    pub suite_version: String,
    pub evidence_class: String,
    pub claim_ceiling: String,
    pub profile: Profile,
    pub cases: Vec<Case>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GateStatus {
    AcceptedForHandoff,
    Rejected,
    Indeterminate,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct CaseResult {
    pub case_id: String,
    pub case_class: String,
    pub client_id: Option<String>,
    pub interface_id: Option<String>,
    pub interface_revision: Option<u64>,
    pub frame_id: Option<String>,
    pub epoch_tick: Option<u64>,
    pub expected: String,
    pub status: GateStatus,
    pub reason_codes: Vec<String>,
    pub authority_granted: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Summary {
    pub case_count: u64,
    pub accepted_for_handoff: u64,
    pub rejected: u64,
    pub indeterminate: u64,
    pub proximity_only_false_accepts: u64,
    pub expected_result_mismatches: u64,
    pub hard_gate_violations: u64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Report {
    pub suite_id: String,
    pub suite_version: String,
    pub evidence_class: String,
    pub claim_ceiling: String,
    pub result: String,
    pub summary: Summary,
    pub cases: Vec<CaseResult>,
}

fn positive(value: f64, label: &str) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        Err(format!("profile_invalid_{label}"))
    } else {
        Ok(())
    }
}

fn nonnegative(value: f64, label: &str) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        Err(format!("profile_invalid_{label}"))
    } else {
        Ok(())
    }
}

fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

fn missing(reason: &str, reasons: &mut Vec<String>) {
    reasons.push(format!("missing_{reason}"));
}

fn violates(reason: &str, reasons: &mut Vec<String>) {
    reasons.push(format!("limit_{reason}"));
}

fn check_profile(profile: &Profile) -> Result<(), String> {
    if profile.frame_id.trim().is_empty()
        || profile.client_id.trim().is_empty()
        || profile.interface_id.trim().is_empty()
    {
        return Err("profile_identity_empty".to_owned());
    }
    positive(profile.max_approach_distance_m, "approach_distance")?;
    positive(profile.max_approach_speed_mps, "approach_speed")?;
    positive(profile.max_contact_position_error_m, "contact_position_error")?;
    positive(profile.max_attitude_error_rad, "attitude_error")?;
    positive(profile.max_contact_speed_mps, "contact_speed")?;
    positive(
        profile.max_relative_angular_rate_rad_s,
        "relative_angular_rate",
    )?;
    positive(profile.max_contact_impulse_ns, "contact_impulse")?;
    positive(profile.max_end_effector_error_m, "end_effector_error")?;
    nonnegative(profile.min_collision_clearance_m, "collision_clearance")?;
    positive(profile.max_joint_abs_rad, "joint_angle")?;
    nonnegative(
        profile.min_joint_limit_margin_rad,
        "joint_limit_margin",
    )?;
    if profile.min_joint_limit_margin_rad >= profile.max_joint_abs_rad {
        return Err("profile_invalid_joint_limit_margin".to_owned());
    }
    positive(profile.max_bus_angular_rate_rad_s, "bus_angular_rate")?;
    Ok(())
}

pub fn evaluate_case(profile: &Profile, case: &Case) -> CaseResult {
    let mut reasons = Vec::new();

    if case.approach.outcome != ApproachOutcome::Docked {
        violates("approach_not_docked", &mut reasons);
    }
    match finite(case.approach.distance_m) {
        Some(value) if value >= 0.0 && value <= profile.max_approach_distance_m => {}
        Some(_) => violates("approach_distance", &mut reasons),
        None => missing("approach_distance", &mut reasons),
    }
    match finite(case.approach.speed_mps) {
        Some(value) if value >= 0.0 && value <= profile.max_approach_speed_mps => {}
        Some(_) => violates("approach_speed", &mut reasons),
        None => missing("approach_speed", &mut reasons),
    }

    let contact = &case.contact;

    match contact.frame_id.as_deref() {
        Some(value) if value == profile.frame_id => {}
        Some(_) => violates("frame_mismatch", &mut reasons),
        None => missing("frame_id", &mut reasons),
    }
    match contact.epoch_tick {
        Some(value)
            if value.abs_diff(profile.epoch_tick) <= profile.max_epoch_skew_ticks => {}
        Some(_) => violates("stale_or_future_epoch", &mut reasons),
        None => missing("epoch_tick", &mut reasons),
    }
    match contact.client_id.as_deref() {
        Some(value) if value == profile.client_id => {}
        Some(_) => violates("client_id_mismatch", &mut reasons),
        None => missing("client_id", &mut reasons),
    }
    match contact.interface_id.as_deref() {
        Some(value) if value == profile.interface_id => {}
        Some(_) => violates("interface_mismatch", &mut reasons),
        None => missing("interface_id", &mut reasons),
    }
    match contact.interface_revision {
        Some(value) if value == profile.interface_revision => {}
        Some(_) => violates("interface_revision_mismatch", &mut reasons),
        None => missing("interface_revision", &mut reasons),
    }

    let scalar_checks = [
        (
            "contact_position_error",
            finite(contact.position_error_m),
            profile.max_contact_position_error_m,
        ),
        (
            "attitude_error",
            finite(contact.attitude_error_rad),
            profile.max_attitude_error_rad,
        ),
        (
            "contact_speed",
            finite(contact.relative_speed_mps),
            profile.max_contact_speed_mps,
        ),
        (
            "relative_angular_rate",
            finite(contact.relative_angular_rate_rad_s),
            profile.max_relative_angular_rate_rad_s,
        ),
        (
            "contact_impulse",
            finite(contact.contact_impulse_ns),
            profile.max_contact_impulse_ns,
        ),
        (
            "end_effector_error",
            finite(contact.end_effector_error_m),
            profile.max_end_effector_error_m,
        ),
    ];
    for (name, value, limit) in scalar_checks {
        match value {
            Some(actual) if actual >= 0.0 && actual <= limit => {}
            Some(_) => violates(name, &mut reasons),
            None => missing(name, &mut reasons),
        }
    }

    match finite(contact.collision_clearance_m) {
        Some(actual) if actual >= profile.min_collision_clearance_m => {}
        Some(_) => violates("collision_clearance", &mut reasons),
        None => missing("collision_clearance", &mut reasons),
    }

    match contact.joint_angles_rad.as_ref() {
        Some(values) if values.len() == 7 && values.iter().all(|v| v.is_finite()) => {
            if values.iter().any(|value| {
                profile.max_joint_abs_rad - value.abs()
                    < profile.min_joint_limit_margin_rad
            }) {
                violates("joint_limit_margin", &mut reasons);
            }
        }
        Some(_) => missing("valid_seven_axis_joint_state", &mut reasons),
        None => missing("joint_state", &mut reasons),
    }

    match contact.bus_angular_velocity_rad_s.as_ref() {
        Some(values) if values.len() == 3 && values.iter().all(|v| v.is_finite()) => {
            let angular_rate_magnitude =
                values.iter().map(|value| value * value).sum::<f64>().sqrt();
            if angular_rate_magnitude > profile.max_bus_angular_rate_rad_s {
                violates("bus_angular_rate_magnitude", &mut reasons);
            }
        }
        Some(_) => missing("valid_three_axis_bus_rate", &mut reasons),
        None => missing("bus_angular_velocity", &mut reasons),
    }

    match contact.latch_state {
        Some(LatchState::Engaged) => {}
        Some(LatchState::NotEngaged) => violates("latch_not_engaged", &mut reasons),
        None => missing("latch_state", &mut reasons),
    }
    match contact.retention_state {
        Some(RetentionState::Passed) => {}
        Some(RetentionState::Failed) => violates("retention_failed", &mut reasons),
        Some(RetentionState::NotRun) | None => missing("retention_test", &mut reasons),
    }

    let receipts = [
        ("approach_receipt", contact.approach_receipt.as_deref()),
        ("contact_receipt", contact.contact_receipt.as_deref()),
        ("arm_receipt", contact.arm_receipt.as_deref()),
        ("latch_receipt", contact.latch_receipt.as_deref()),
        ("retention_receipt", contact.retention_receipt.as_deref()),
    ];
    for (name, receipt) in receipts {
        match receipt {
            Some(value) if !value.trim().is_empty() => {}
            _ => missing(name, &mut reasons),
        }
    }

    reasons.sort();
    reasons.dedup();
    let has_hard_violation = reasons.iter().any(|reason| reason.starts_with("limit_"));
    let has_missing = reasons.iter().any(|reason| reason.starts_with("missing_"));
    let status = if has_hard_violation {
        GateStatus::Rejected
    } else if has_missing {
        GateStatus::Indeterminate
    } else {
        GateStatus::AcceptedForHandoff
    };

    CaseResult {
        case_id: case.id.clone(),
        case_class: case.case_class.clone(),
        client_id: contact.client_id.clone(),
        interface_id: contact.interface_id.clone(),
        interface_revision: contact.interface_revision,
        frame_id: contact.frame_id.clone(),
        epoch_tick: contact.epoch_tick,
        expected: case.expected.clone(),
        status,
        reason_codes: reasons,
        authority_granted: false,
    }
}

pub fn evaluate(fixture: &Fixture) -> Result<Report, String> {
    if fixture.suite_id != "orbital-capture-handoff" || fixture.suite_version != "0.1.0" {
        return Err("unsupported_suite_identity_or_version".to_owned());
    }
    if fixture.evidence_class != "synthetic" {
        return Err("fixture_must_be_explicitly_synthetic".to_owned());
    }
    if fixture.claim_ceiling.trim().is_empty() {
        return Err("claim_ceiling_missing".to_owned());
    }
    if fixture.cases.is_empty() {
        return Err("fixture_cases_empty".to_owned());
    }
    check_profile(&fixture.profile)?;

    let mut ids = BTreeSet::new();
    let mut results = Vec::with_capacity(fixture.cases.len());
    let mut proximity_only_false_accepts = 0_u64;
    let mut mismatches = 0_u64;
    let mut hard_gate_violations = 0_u64;

    for case in &fixture.cases {
        if case.id.trim().is_empty() || !ids.insert(case.id.as_str()) {
            return Err("case_id_empty_or_duplicate".to_owned());
        }
        if !["positive_control", "negative_control"].contains(&case.case_class.as_str()) {
            return Err(format!("case_{}_has_unknown_class", case.id));
        }
        let result = evaluate_case(&fixture.profile, case);
        if case.approach.outcome == ApproachOutcome::Docked
            && result.status != GateStatus::AcceptedForHandoff
        {
            proximity_only_false_accepts += 1;
        }
        let status_label = match &result.status {
            GateStatus::AcceptedForHandoff => "accepted_for_handoff",
            GateStatus::Rejected => "rejected",
            GateStatus::Indeterminate => "indeterminate",
        };
        if status_label != case.expected {
            mismatches += 1;
        }
        if result
            .reason_codes
            .iter()
            .any(|reason| reason.starts_with("limit_"))
        {
            hard_gate_violations += 1;
        }
        results.push(result);
    }

    let accepted = results
        .iter()
        .filter(|case| case.status == GateStatus::AcceptedForHandoff)
        .count() as u64;
    let rejected = results
        .iter()
        .filter(|case| case.status == GateStatus::Rejected)
        .count() as u64;
    let indeterminate = results
        .iter()
        .filter(|case| case.status == GateStatus::Indeterminate)
        .count() as u64;

    let result = if mismatches == 0 && results.iter().all(|case| !case.authority_granted) {
        "pass"
    } else {
        "fail"
    };

    Ok(Report {
        suite_id: fixture.suite_id.clone(),
        suite_version: fixture.suite_version.clone(),
        evidence_class: fixture.evidence_class.clone(),
        claim_ceiling: fixture.claim_ceiling.clone(),
        result: result.to_owned(),
        summary: Summary {
            case_count: results.len() as u64,
            accepted_for_handoff: accepted,
            rejected,
            indeterminate,
            proximity_only_false_accepts,
            expected_result_mismatches: mismatches,
            hard_gate_violations,
        },
        cases: results,
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("String writes cannot fail");
    }
    output
}

pub fn is_valid_source_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Fixture {
        serde_json::from_str(include_str!("../fixtures/v0.1/cases.json"))
            .expect("frozen fixture must parse")
    }

    #[test]
    fn complete_capture_fixture_matches_all_expected_results() {
        let report = evaluate(&fixture()).expect("fixture should validate");
        assert_eq!(report.result, "pass");
        assert_eq!(report.summary.case_count, 19);
        assert_eq!(report.summary.accepted_for_handoff, 1);
        assert_eq!(report.summary.rejected, 14);
        assert_eq!(report.summary.indeterminate, 4);
        assert_eq!(report.summary.expected_result_mismatches, 0);
        assert_eq!(report.summary.proximity_only_false_accepts, 17);
        assert!(report.cases.iter().all(|case| !case.authority_granted));
    }

    #[test]
    fn approach_docked_without_capture_evidence_is_indeterminate() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "APPROACH_ONLY_NOT_CAPTURE").unwrap();
        let result = evaluate_case(&data.profile, case);
        assert_eq!(result.status, GateStatus::Indeterminate);
    }

    #[test]
    fn nominal_capture_can_only_pass_handoff_not_authority() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "NOMINAL_CAPTURE").unwrap();
        let result = evaluate_case(&data.profile, case);
        assert_eq!(result.status, GateStatus::AcceptedForHandoff);
        assert_eq!(result.client_id.as_deref(), Some("serviced-client-001"));
        assert_eq!(result.interface_id.as_deref(), Some(data.profile.interface_id.as_str()));
        assert_eq!(result.interface_revision, Some(data.profile.interface_revision));
        assert_eq!(result.frame_id.as_deref(), Some(data.profile.frame_id.as_str()));
        assert_eq!(result.epoch_tick, Some(data.profile.epoch_tick));
        assert!(!result.authority_granted);
    }

    #[test]
    fn wrong_serviced_client_identity_is_rejected() {
        let data = fixture();
        let mut case = data.cases.iter().find(|case| case.id == "NOMINAL_CAPTURE").unwrap().clone();
        case.contact.client_id = Some("unrelated-client".to_owned());
        assert_eq!(evaluate_case(&data.profile, &case).status, GateStatus::Rejected);
    }

    #[test]
    fn missing_serviced_client_identity_is_indeterminate() {
        let data = fixture();
        let mut case = data.cases.iter().find(|case| case.id == "NOMINAL_CAPTURE").unwrap().clone();
        case.contact.client_id = None;
        assert_eq!(evaluate_case(&data.profile, &case).status, GateStatus::Indeterminate);
    }

    #[test]
    fn interface_mismatch_is_rejected() {
        let data = fixture();
        let case = data
            .cases
            .iter()
            .find(|case| case.id == "INTERFACE_REVISION_MISMATCH")
            .unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Rejected);
    }

    #[test]
    fn missing_frame_is_indeterminate() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "FRAME_NOT_OBSERVED").unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Indeterminate);
    }

    #[test]
    fn contact_impulse_above_envelope_is_rejected() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "CONTACT_IMPULSE_TOO_HIGH").unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Rejected);
    }

    #[test]
    fn negative_magnitude_observation_is_rejected() {
        let data = fixture();
        let mut case = data
            .cases
            .iter()
            .find(|case| case.id == "NOMINAL_CAPTURE")
            .unwrap()
            .clone();
        case.contact.position_error_m = Some(-0.001);
        assert_eq!(evaluate_case(&data.profile, &case).status, GateStatus::Rejected);
    }

    #[test]
    fn failed_latch_is_not_promoted_to_capture() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "LATCH_NOT_ENGAGED").unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Rejected);
    }

    #[test]
    fn unrun_retention_check_is_indeterminate() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "RETENTION_NOT_RUN").unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Indeterminate);
    }

    #[test]
    fn missing_evidence_receipt_is_indeterminate() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "MISSING_CONTACT_EVIDENCE").unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Indeterminate);
    }

    #[test]
    fn high_joint_angle_is_rejected() {
        let data = fixture();
        let case = data
            .cases
            .iter()
            .find(|case| case.id == "JOINT_LIMIT_MARGIN_VIOLATION")
            .unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Rejected);
    }

    #[test]
    fn insufficient_joint_limit_margin_is_rejected() {
        let data = fixture();
        let case = data
            .cases
            .iter()
            .find(|case| case.id == "JOINT_LIMIT_MARGIN_VIOLATION")
            .unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Rejected);
    }

    #[test]
    fn attitude_misalignment_is_rejected() {
        let data = fixture();
        let case = data
            .cases
            .iter()
            .find(|case| case.id == "ATTITUDE_MISALIGNMENT")
            .unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Rejected);
    }

    #[test]
    fn high_bus_rate_is_rejected() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "BUS_RATE_LIMIT_VIOLATION").unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Rejected);
    }

    #[test]
    fn bus_rate_vector_magnitude_is_checked_not_only_axis_components() {
        let data = fixture();
        let mut case = data
            .cases
            .iter()
            .find(|case| case.id == "NOMINAL_CAPTURE")
            .unwrap()
            .clone();
        case.contact.bus_angular_velocity_rad_s = Some([0.015, 0.015, 0.015].to_vec());
        assert_eq!(evaluate_case(&data.profile, &case).status, GateStatus::Rejected);
    }

    #[test]
    fn stale_epoch_is_rejected() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "STALE_EPOCH").unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Rejected);
    }

    #[test]
    fn false_approach_success_is_rejected_even_if_contact_data_is_good() {
        let data = fixture();
        let case = data.cases.iter().find(|case| case.id == "APPROACH_ABORTED").unwrap();
        assert_eq!(evaluate_case(&data.profile, case).status, GateStatus::Rejected);
    }

    #[test]
    fn wrong_joint_or_bus_vector_dimensions_are_indeterminate() {
        let data = fixture();
        let mut case = data.cases[0].clone();
        case.contact.joint_angles_rad = Some(vec![0.0; 6]);
        assert_eq!(evaluate_case(&data.profile, &case).status, GateStatus::Indeterminate);
    }

    #[test]
    fn duplicate_case_identifiers_are_rejected() {
        let mut data = fixture();
        data.cases[1].id = data.cases[0].id.clone();
        assert!(evaluate(&data).unwrap_err().contains("duplicate"));
    }

    #[test]
    fn source_commit_validator_rejects_non_sha_values() {
        assert!(is_valid_source_commit("0123456789abcdef0123456789abcdef01234567"));
        assert!(!is_valid_source_commit("main"));
        assert!(!is_valid_source_commit("0123456789abcdef0123456789abcdef0123456z"));
    }

    #[test]
    fn profile_rejects_nonpositive_thresholds() {
        let mut data = fixture();
        data.profile.max_contact_speed_mps = 0.0;
        assert!(evaluate(&data).unwrap_err().contains("profile_invalid"));
    }
}
