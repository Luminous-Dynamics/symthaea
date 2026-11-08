//! Exact-subject evidence composition for marine liveaboard mission cases.
//!
//! This crate joins owner-declared outcomes. It does not calculate physics,
//! re-evaluate evidence, validate the owners' methods, or issue approval.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum EvidenceKind {
    Synthetic,
    Assumed,
    ModelEstimate,
    ExternalSource,
    Measured,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum OwnerOutcome {
    SupportsDeclaredScope,
    DoesNotSupportDeclaredScope,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Currentness {
    Current,
    Expired,
    Unknown,
    NotRequired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum OwnerDisposition {
    Supports,
    DoesNotSupport,
    Indeterminate,
    Invalid,
    Missing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum MissionDisposition {
    AllRequiredOwnersSupportDeclaredScope,
    AtLeastOneRequiredOwnerDoesNotSupportScope,
    Indeterminate,
    Invalid,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Finding {
    MissingProfileId,
    MissingProfileManifestDigest,
    InvalidProfileManifestDigest,
    MissingConfigurationDigest,
    InvalidConfigurationDigest,
    MissingMissionContractDigest,
    InvalidMissionContractDigest,
    MissingOperatingCaseId,
    MissingConditionsDigest,
    InvalidConditionsDigest,
    ZeroTargetDuration,
    EmptyRequiredOwnerRoster,
    EmptyRequiredOwnerId,
    DuplicateRequiredOwnerId { owner_id: String },
    MissingOwnerId,
    DuplicateOwnerEvidence { owner_id: String },
    UnexpectedOwnerEvidence { owner_id: String },
    MissingRequiredOwnerEvidence { owner_id: String },
    MissingReportReference { owner_id: String },
    MissingResultDigest { owner_id: String },
    InvalidResultDigest { owner_id: String },
    MissingMethodRevision { owner_id: String },
    MissingProfileBinding { owner_id: String },
    StaleProfileBinding { owner_id: String },
    MissingProfileDigestBinding { owner_id: String },
    InvalidProfileDigestBinding { owner_id: String },
    StaleProfileDigestBinding { owner_id: String },
    MissingConfigurationBinding { owner_id: String },
    InvalidConfigurationBinding { owner_id: String },
    StaleConfigurationBinding { owner_id: String },
    MissingMissionContractBinding { owner_id: String },
    InvalidMissionContractBinding { owner_id: String },
    StaleMissionContractBinding { owner_id: String },
    MissingOperatingCaseBinding { owner_id: String },
    StaleOperatingCaseBinding { owner_id: String },
    MissingConditionsBinding { owner_id: String },
    InvalidConditionsBinding { owner_id: String },
    StaleConditionsBinding { owner_id: String },
    MissingTargetDurationBinding { owner_id: String },
    StaleTargetDurationBinding { owner_id: String },
    MissingEvidenceKind { owner_id: String },
    MissingCurrentness { owner_id: String },
    MissingCurrentnessReference { owner_id: String },
    ExpiredEvidence { owner_id: String },
    UnknownEvidenceCurrentness { owner_id: String },
    MissingOwnerOutcome { owner_id: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MissionCase {
    pub profile_id: String,
    pub profile_manifest_digest: String,
    pub configuration_digest: String,
    pub mission_contract_digest: String,
    pub operating_case_id: String,
    pub conditions_digest: String,
    pub target_duration_seconds: u64,
    /// The profile freezes this roster. Input order has no meaning.
    pub required_owner_ids: Vec<String>,
    pub evidence: Vec<MissionEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MissionEvidence {
    pub owner_id: String,
    pub report_ref: Option<String>,
    pub result_digest: Option<String>,
    pub method_revision_ref: Option<String>,
    pub profile_id: Option<String>,
    pub profile_manifest_digest: Option<String>,
    pub configuration_digest: Option<String>,
    pub mission_contract_digest: Option<String>,
    pub operating_case_id: Option<String>,
    pub conditions_digest: Option<String>,
    pub target_duration_seconds: Option<u64>,
    pub evidence_kind: Option<EvidenceKind>,
    pub currentness: Option<Currentness>,
    pub currentness_ref: Option<String>,
    pub outcome: Option<OwnerOutcome>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct OwnerAssessment {
    pub owner_id: String,
    pub report_ref: Option<String>,
    pub result_digest: Option<String>,
    pub evidence_kind: Option<EvidenceKind>,
    pub currentness: Option<Currentness>,
    pub outcome: Option<OwnerOutcome>,
    pub disposition: OwnerDisposition,
    pub findings: Vec<Finding>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MissionReport {
    pub profile_id: String,
    pub profile_manifest_digest: String,
    pub configuration_digest: String,
    pub mission_contract_digest: String,
    pub operating_case_id: String,
    pub conditions_digest: String,
    pub target_duration_seconds: u64,
    pub disposition: MissionDisposition,
    pub owners: Vec<OwnerAssessment>,
    pub findings: Vec<Finding>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn sort_findings(findings: &mut Vec<Finding>) {
    findings.sort();
    findings.dedup();
}

fn check_required_text(
    value: &str,
    missing: Finding,
    findings: &mut Vec<Finding>,
    indeterminate: &mut bool,
) {
    if !nonempty(value) {
        findings.push(missing);
        *indeterminate = true;
    }
}

fn check_case_digest(
    value: &str,
    missing: Finding,
    malformed: Finding,
    findings: &mut Vec<Finding>,
    invalid: &mut bool,
    indeterminate: &mut bool,
) {
    if !nonempty(value) {
        findings.push(missing);
        *indeterminate = true;
    } else if !is_sha256(value) {
        findings.push(malformed);
        *invalid = true;
    }
}

fn check_subject_text(
    value: Option<&str>,
    expected: &str,
    missing: Finding,
    stale: Finding,
    findings: &mut Vec<Finding>,
    invalid: &mut bool,
    indeterminate: &mut bool,
) {
    match value {
        Some(value) if nonempty(value) => {
            if value != expected {
                findings.push(stale);
                *invalid = true;
            }
        }
        _ => {
            findings.push(missing);
            *indeterminate = true;
        }
    }
}

fn check_subject_digest(
    value: Option<&str>,
    expected: &str,
    missing: Finding,
    malformed: Finding,
    stale: Finding,
    findings: &mut Vec<Finding>,
    invalid: &mut bool,
    indeterminate: &mut bool,
) {
    match value {
        Some(value) if nonempty(value) => {
            if !is_sha256(value) {
                findings.push(malformed);
                *invalid = true;
            } else if value != expected {
                findings.push(stale);
                *invalid = true;
            }
        }
        _ => {
            findings.push(missing);
            *indeterminate = true;
        }
    }
}

fn assess_owner(evidence: &MissionEvidence, case: &MissionCase, duplicate: bool) -> OwnerAssessment {
    let owner_id = if nonempty(&evidence.owner_id) {
        evidence.owner_id.clone()
    } else {
        "<missing>".to_owned()
    };
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut indeterminate = false;

    if owner_id == "<missing>" {
        findings.push(Finding::MissingOwnerId);
        invalid = true;
    }
    if duplicate {
        findings.push(Finding::DuplicateOwnerEvidence { owner_id: owner_id.clone() });
        invalid = true;
    }
    if owner_id != "<missing>" && !case.required_owner_ids.contains(&owner_id) {
        findings.push(Finding::UnexpectedOwnerEvidence { owner_id: owner_id.clone() });
        invalid = true;
    }

    if evidence.report_ref.as_deref().is_none_or(|value| !nonempty(value)) {
        findings.push(Finding::MissingReportReference { owner_id: owner_id.clone() });
        indeterminate = true;
    }
    match evidence.result_digest.as_deref() {
        Some(value) if nonempty(value) && is_sha256(value) => {}
        Some(value) if nonempty(value) => {
            findings.push(Finding::InvalidResultDigest { owner_id: owner_id.clone() });
            invalid = true;
        }
        _ => {
            findings.push(Finding::MissingResultDigest { owner_id: owner_id.clone() });
            indeterminate = true;
        }
    }
    if evidence.method_revision_ref.as_deref().is_none_or(|value| !nonempty(value)) {
        findings.push(Finding::MissingMethodRevision { owner_id: owner_id.clone() });
        indeterminate = true;
    }

    check_subject_text(
        evidence.profile_id.as_deref(),
        &case.profile_id,
        Finding::MissingProfileBinding { owner_id: owner_id.clone() },
        Finding::StaleProfileBinding { owner_id: owner_id.clone() },
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    check_subject_digest(
        evidence.profile_manifest_digest.as_deref(),
        &case.profile_manifest_digest,
        Finding::MissingProfileDigestBinding { owner_id: owner_id.clone() },
        Finding::InvalidProfileDigestBinding { owner_id: owner_id.clone() },
        Finding::StaleProfileDigestBinding { owner_id: owner_id.clone() },
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    check_subject_digest(
        evidence.configuration_digest.as_deref(),
        &case.configuration_digest,
        Finding::MissingConfigurationBinding { owner_id: owner_id.clone() },
        Finding::InvalidConfigurationBinding { owner_id: owner_id.clone() },
        Finding::StaleConfigurationBinding { owner_id: owner_id.clone() },
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    check_subject_digest(
        evidence.mission_contract_digest.as_deref(),
        &case.mission_contract_digest,
        Finding::MissingMissionContractBinding { owner_id: owner_id.clone() },
        Finding::InvalidMissionContractBinding { owner_id: owner_id.clone() },
        Finding::StaleMissionContractBinding { owner_id: owner_id.clone() },
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    check_subject_text(
        evidence.operating_case_id.as_deref(),
        &case.operating_case_id,
        Finding::MissingOperatingCaseBinding { owner_id: owner_id.clone() },
        Finding::StaleOperatingCaseBinding { owner_id: owner_id.clone() },
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    check_subject_digest(
        evidence.conditions_digest.as_deref(),
        &case.conditions_digest,
        Finding::MissingConditionsBinding { owner_id: owner_id.clone() },
        Finding::InvalidConditionsBinding { owner_id: owner_id.clone() },
        Finding::StaleConditionsBinding { owner_id: owner_id.clone() },
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    match evidence.target_duration_seconds {
        Some(value) if value == case.target_duration_seconds => {}
        Some(_) => {
            findings.push(Finding::StaleTargetDurationBinding { owner_id: owner_id.clone() });
            invalid = true;
        }
        None => {
            findings.push(Finding::MissingTargetDurationBinding { owner_id: owner_id.clone() });
            indeterminate = true;
        }
    }

    if evidence.evidence_kind.is_none() {
        findings.push(Finding::MissingEvidenceKind { owner_id: owner_id.clone() });
        indeterminate = true;
    }

    match evidence.currentness {
        Some(Currentness::Current) => {
            if evidence.currentness_ref.as_deref().is_none_or(|value| !nonempty(value)) {
                findings.push(Finding::MissingCurrentnessReference { owner_id: owner_id.clone() });
                indeterminate = true;
            }
        }
        Some(Currentness::NotRequired) => {}
        Some(Currentness::Expired) => {
            findings.push(Finding::ExpiredEvidence { owner_id: owner_id.clone() });
            indeterminate = true;
        }
        Some(Currentness::Unknown) => {
            findings.push(Finding::UnknownEvidenceCurrentness { owner_id: owner_id.clone() });
            indeterminate = true;
        }
        None => {
            findings.push(Finding::MissingCurrentness { owner_id: owner_id.clone() });
            indeterminate = true;
        }
    }

    let outcome = evidence.outcome;
    match outcome {
        Some(OwnerOutcome::SupportsDeclaredScope) => {}
        Some(OwnerOutcome::DoesNotSupportDeclaredScope) => {}
        Some(OwnerOutcome::Indeterminate) => indeterminate = true,
        None => {
            findings.push(Finding::MissingOwnerOutcome { owner_id: owner_id.clone() });
            indeterminate = true;
        }
    }

    let disposition = if invalid {
        OwnerDisposition::Invalid
    } else if indeterminate {
        OwnerDisposition::Indeterminate
    } else {
        match outcome.expect("outcome checked above") {
            OwnerOutcome::SupportsDeclaredScope => OwnerDisposition::Supports,
            OwnerOutcome::DoesNotSupportDeclaredScope => OwnerDisposition::DoesNotSupport,
            OwnerOutcome::Indeterminate => OwnerDisposition::Indeterminate,
        }
    };

    sort_findings(&mut findings);
    OwnerAssessment {
        owner_id,
        report_ref: evidence.report_ref.clone(),
        result_digest: evidence.result_digest.clone(),
        evidence_kind: evidence.evidence_kind,
        currentness: evidence.currentness,
        outcome,
        disposition,
        findings,
    }
}

/// Join owner-declared outcomes for one frozen mission case.
///
/// Digests are validated as 64-character hexadecimal identifiers and compared
/// exactly. This function does not recompute them, authenticate their sources,
/// or validate the engineering method that emitted each report.
pub fn assess_case(case: &MissionCase) -> MissionReport {
    let mut findings = Vec::new();
    let mut invalid = false;
    let mut indeterminate = false;

    check_required_text(
        &case.profile_id,
        Finding::MissingProfileId,
        &mut findings,
        &mut indeterminate,
    );
    check_case_digest(
        &case.profile_manifest_digest,
        Finding::MissingProfileManifestDigest,
        Finding::InvalidProfileManifestDigest,
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    check_case_digest(
        &case.configuration_digest,
        Finding::MissingConfigurationDigest,
        Finding::InvalidConfigurationDigest,
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    check_case_digest(
        &case.mission_contract_digest,
        Finding::MissingMissionContractDigest,
        Finding::InvalidMissionContractDigest,
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    check_required_text(
        &case.operating_case_id,
        Finding::MissingOperatingCaseId,
        &mut findings,
        &mut indeterminate,
    );
    check_case_digest(
        &case.conditions_digest,
        Finding::MissingConditionsDigest,
        Finding::InvalidConditionsDigest,
        &mut findings,
        &mut invalid,
        &mut indeterminate,
    );
    if case.target_duration_seconds == 0 {
        findings.push(Finding::ZeroTargetDuration);
        invalid = true;
    }
    if case.required_owner_ids.is_empty() {
        findings.push(Finding::EmptyRequiredOwnerRoster);
        invalid = true;
    }

    let mut required_counts = BTreeMap::<String, usize>::new();
    for owner_id in &case.required_owner_ids {
        if !nonempty(owner_id) {
            findings.push(Finding::EmptyRequiredOwnerId);
            invalid = true;
        } else {
            *required_counts.entry(owner_id.clone()).or_default() += 1;
        }
    }
    for (owner_id, count) in &required_counts {
        if *count > 1 {
            findings.push(Finding::DuplicateRequiredOwnerId { owner_id: owner_id.clone() });
            invalid = true;
        }
    }
    let required_ids: Vec<String> = required_counts.keys().cloned().collect();

    let mut evidence_counts = BTreeMap::<String, usize>::new();
    for evidence in &case.evidence {
        if nonempty(&evidence.owner_id) {
            *evidence_counts.entry(evidence.owner_id.clone()).or_default() += 1;
        }
    }
    for owner_id in &required_ids {
        if !evidence_counts.contains_key(owner_id) {
            findings.push(Finding::MissingRequiredOwnerEvidence { owner_id: owner_id.clone() });
            indeterminate = true;
        }
    }

    // MissionEvidence derives Ord, so report ordering is independent of caller
    // input order even when multiple invalid rows share one owner.
    let mut evidence_rows = case.evidence.clone();
    evidence_rows.sort();

    let mut owners = evidence_rows
        .iter()
        .map(|row| {
            let duplicate = nonempty(&row.owner_id)
                && evidence_counts.get(&row.owner_id).copied().unwrap_or(0) > 1;
            assess_owner(row, case, duplicate)
        })
        .collect::<Vec<_>>();

    for owner_id in &required_ids {
        if !evidence_counts.contains_key(owner_id) {
            owners.push(OwnerAssessment {
                owner_id: owner_id.clone(),
                report_ref: None,
                result_digest: None,
                evidence_kind: None,
                currentness: None,
                outcome: None,
                disposition: OwnerDisposition::Missing,
                findings: vec![Finding::MissingRequiredOwnerEvidence { owner_id: owner_id.clone() }],
            });
        }
    }

    owners.sort();
    for owner in &owners {
        findings.extend(owner.findings.clone());
        match owner.disposition {
            OwnerDisposition::Invalid => invalid = true,
            OwnerDisposition::Indeterminate | OwnerDisposition::Missing => indeterminate = true,
            OwnerDisposition::DoesNotSupport | OwnerDisposition::Supports => {}
        }
    }
    sort_findings(&mut findings);

    let disposition = if invalid {
        MissionDisposition::Invalid
    } else if owners
        .iter()
        .any(|owner| owner.disposition == OwnerDisposition::DoesNotSupport)
    {
        MissionDisposition::AtLeastOneRequiredOwnerDoesNotSupportScope
    } else if indeterminate {
        MissionDisposition::Indeterminate
    } else {
        MissionDisposition::AllRequiredOwnersSupportDeclaredScope
    };

    MissionReport {
        profile_id: case.profile_id.clone(),
        profile_manifest_digest: case.profile_manifest_digest.clone(),
        configuration_digest: case.configuration_digest.clone(),
        mission_contract_digest: case.mission_contract_digest.clone(),
        operating_case_id: case.operating_case_id.clone(),
        conditions_digest: case.conditions_digest.clone(),
        target_duration_seconds: case.target_duration_seconds,
        disposition,
        owners,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CONFIG: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const MISSION: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const CONDITIONS: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const RESULT: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn evidence(owner_id: &str) -> MissionEvidence {
        MissionEvidence {
            owner_id: owner_id.to_owned(),
            report_ref: Some(format!("fixture:report:{owner_id}")),
            result_digest: Some(RESULT.to_owned()),
            method_revision_ref: Some(format!("fixture:method:{owner_id}:v1")),
            profile_id: Some("liveaboard-offshore-v1".to_owned()),
            profile_manifest_digest: Some(PROFILE.to_owned()),
            configuration_digest: Some(CONFIG.to_owned()),
            mission_contract_digest: Some(MISSION.to_owned()),
            operating_case_id: Some("offshore-30d-sea-state-4-v1".to_owned()),
            conditions_digest: Some(CONDITIONS.to_owned()),
            target_duration_seconds: Some(30 * 24 * 60 * 60),
            evidence_kind: Some(EvidenceKind::Synthetic),
            currentness: Some(Currentness::Current),
            currentness_ref: Some(format!("fixture:current:{owner_id}")),
            outcome: Some(OwnerOutcome::SupportsDeclaredScope),
        }
    }

    fn case() -> MissionCase {
        MissionCase {
            profile_id: "liveaboard-offshore-v1".to_owned(),
            profile_manifest_digest: PROFILE.to_owned(),
            configuration_digest: CONFIG.to_owned(),
            mission_contract_digest: MISSION.to_owned(),
            operating_case_id: "offshore-30d-sea-state-4-v1".to_owned(),
            conditions_digest: CONDITIONS.to_owned(),
            target_duration_seconds: 30 * 24 * 60 * 60,
            required_owner_ids: vec![
                "energy_budget".to_owned(),
                "resource_autonomy".to_owned(),
                "habitability_motion".to_owned(),
            ],
            evidence: vec![
                evidence("energy_budget"),
                evidence("resource_autonomy"),
                evidence("habitability_motion"),
            ],
        }
    }

    #[test]
    fn all_required_owners_support_same_frozen_case() {
        let report = assess_case(&case());
        assert_eq!(report.disposition, MissionDisposition::AllRequiredOwnersSupportDeclaredScope);
        assert_eq!(report.owners.len(), 3);
        assert!(report.owners.iter().all(|owner| owner.disposition == OwnerDisposition::Supports));
    }

    #[test]
    fn one_owner_contradiction_cannot_be_compensated_by_other_domains() {
        let mut input = case();
        input.evidence[2].outcome = Some(OwnerOutcome::DoesNotSupportDeclaredScope);
        let report = assess_case(&input);
        assert_eq!(report.disposition, MissionDisposition::AtLeastOneRequiredOwnerDoesNotSupportScope);
        let habitability = report.owners.iter().find(|owner| owner.owner_id == "habitability_motion").unwrap();
        assert_eq!(habitability.disposition, OwnerDisposition::DoesNotSupport);
    }

    #[test]
    fn missing_owner_report_is_indeterminate_not_supported() {
        let mut input = case();
        input.evidence.retain(|row| row.owner_id != "resource_autonomy");
        let report = assess_case(&input);
        assert_eq!(report.disposition, MissionDisposition::Indeterminate);
        assert!(report.findings.contains(&Finding::MissingRequiredOwnerEvidence {
            owner_id: "resource_autonomy".to_owned()
        }));
    }

    #[test]
    fn configuration_revision_mismatch_invalidates_join() {
        let mut input = case();
        input.evidence[0].configuration_digest = Some("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_owned());
        assert_eq!(assess_case(&input).disposition, MissionDisposition::Invalid);
    }

    #[test]
    fn sea_state_or_condition_mismatch_invalidates_join() {
        let mut input = case();
        input.evidence[1].conditions_digest = Some("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_owned());
        assert_eq!(assess_case(&input).disposition, MissionDisposition::Invalid);
    }

    #[test]
    fn target_duration_mismatch_invalidates_join() {
        let mut input = case();
        input.evidence[0].target_duration_seconds = Some(14 * 24 * 60 * 60);
        assert_eq!(assess_case(&input).disposition, MissionDisposition::Invalid);
    }

    #[test]
    fn duplicate_evidence_rows_are_invalid_even_when_both_support() {
        let mut input = case();
        input.evidence.push(input.evidence[0].clone());
        let report = assess_case(&input);
        assert_eq!(report.disposition, MissionDisposition::Invalid);
        assert!(report.findings.iter().any(|finding| matches!(
            finding,
            Finding::DuplicateOwnerEvidence { owner_id } if owner_id == "energy_budget"
        )));
    }

    #[test]
    fn extra_owner_cannot_expand_the_frozen_roster() {
        let mut input = case();
        input.evidence.push(evidence("unregistered_bonus_domain"));
        assert_eq!(assess_case(&input).disposition, MissionDisposition::Invalid);
    }

    #[test]
    fn duplicate_required_owner_is_invalid() {
        let mut input = case();
        input.required_owner_ids.push("energy_budget".to_owned());
        assert_eq!(assess_case(&input).disposition, MissionDisposition::Invalid);
    }

    #[test]
    fn missing_metadata_and_unknown_currentness_remain_indeterminate() {
        let mut input = case();
        input.evidence[0].result_digest = None;
        input.evidence[1].currentness = Some(Currentness::Unknown);
        let report = assess_case(&input);
        assert_eq!(report.disposition, MissionDisposition::Indeterminate);
    }

    #[test]
    fn expired_evidence_remains_indeterminate() {
        let mut input = case();
        input.evidence[0].currentness = Some(Currentness::Expired);
        let report = assess_case(&input);
        assert_eq!(report.disposition, MissionDisposition::Indeterminate);
        assert!(report.findings.contains(&Finding::ExpiredEvidence {
            owner_id: "energy_budget".to_owned()
        }));
    }

    #[test]
    fn synthetic_or_assumed_labels_are_preserved_not_upgraded() {
        let mut input = case();
        input.evidence[0].evidence_kind = Some(EvidenceKind::Assumed);
        let report = assess_case(&input);
        let energy = report.owners.iter().find(|owner| owner.owner_id == "energy_budget").unwrap();
        assert_eq!(energy.evidence_kind, Some(EvidenceKind::Assumed));
        assert_eq!(report.disposition, MissionDisposition::AllRequiredOwnersSupportDeclaredScope);
    }

    #[test]
    fn evidence_permutation_does_not_change_report() {
        let forward = case();
        let mut reverse = case();
        reverse.evidence.reverse();
        reverse.required_owner_ids.reverse();
        assert_eq!(assess_case(&forward), assess_case(&reverse));
    }

    #[test]
    fn malformed_subject_digest_is_invalid() {
        let mut input = case();
        input.evidence[0].mission_contract_digest = Some("not-a-digest".to_owned());
        assert_eq!(assess_case(&input).disposition, MissionDisposition::Invalid);
    }

    #[test]
    fn missing_subject_binding_is_indeterminate() {
        let mut input = case();
        input.evidence[0].configuration_digest = None;
        assert_eq!(assess_case(&input).disposition, MissionDisposition::Indeterminate);
    }

    #[test]
    fn owner_declared_not_required_currentness_is_explicit() {
        let mut input = case();
        input.evidence[0].currentness = Some(Currentness::NotRequired);
        input.evidence[0].currentness_ref = None;
        let report = assess_case(&input);
        assert_eq!(report.disposition, MissionDisposition::AllRequiredOwnersSupportDeclaredScope);
    }

    #[test]
    fn missing_outcome_never_passes() {
        let mut input = case();
        input.evidence[0].outcome = None;
        assert_eq!(assess_case(&input).disposition, MissionDisposition::Indeterminate);
    }
}
