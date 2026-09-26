use serde::{Deserialize, Serialize};
use symthaea_manufacturing_process::ProcessDefinitionId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum CapabilityEvidenceClassV1 {
    UnknownOrUnavailable,
    Declared,
    ManufacturerSpecified,
    ObservedCapability,
    QualifiedUnderProfile,
    ProductionQualifiedUnderProfile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct CapabilityValidityV1 {
    revision: String,
    valid_from_unix_s: u64,
    valid_until_unix_s: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ProcessCapabilityProfileV1 {
    resource_subject_ref: String,
    process_id: ProcessDefinitionId,
    envelope_profile_ref: String,
    configuration_profile_ref: String,
    evidence_class: CapabilityEvidenceClassV1,
    evidence_ref: String,
    validity: CapabilityValidityV1,
    display_label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CapabilityMatchV1 {
    ExactAdmitted,
    CompatibleButWeakerEvidence,
    UnresolvedExternalRefs,
    ProcessMismatch,
    EnvelopeMismatch,
    ExpiredOrStale,
    Unknown,
}

fn canonical(value: &str) -> bool {
    !value.is_empty() && value.trim() == value && !value.chars().any(char::is_control)
}

fn validate_profile(profile: &ProcessCapabilityProfileV1) -> Result<(), &'static str> {
    for value in [
        &profile.resource_subject_ref,
        &profile.process_id.0,
        &profile.envelope_profile_ref,
        &profile.configuration_profile_ref,
        &profile.evidence_ref,
        &profile.validity.revision,
    ] {
        if !canonical(value) {
            return Err("capability profile contains non-canonical identity/ref");
        }
    }
    if profile.validity.valid_from_unix_s >= profile.validity.valid_until_unix_s {
        return Err("capability validity window must be ordered");
    }
    Ok(())
}

fn capability_identity(profile: &ProcessCapabilityProfileV1) -> Result<String, &'static str> {
    validate_profile(profile)?;
    let mut hasher = blake3::Hasher::new();
    for value in [
        profile.resource_subject_ref.as_str(),
        profile.process_id.0.as_str(),
        profile.envelope_profile_ref.as_str(),
        profile.configuration_profile_ref.as_str(),
        match profile.evidence_class {
            CapabilityEvidenceClassV1::UnknownOrUnavailable => "unknown-or-unavailable",
            CapabilityEvidenceClassV1::Declared => "declared",
            CapabilityEvidenceClassV1::ManufacturerSpecified => "manufacturer-specified",
            CapabilityEvidenceClassV1::ObservedCapability => "observed-capability",
            CapabilityEvidenceClassV1::QualifiedUnderProfile => "qualified-under-profile",
            CapabilityEvidenceClassV1::ProductionQualifiedUnderProfile => {
                "production-qualified-under-profile"
            }
        },
        profile.evidence_ref.as_str(),
        profile.validity.revision.as_str(),
    ] {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(&profile.validity.valid_from_unix_s.to_le_bytes());
    hasher.update(&profile.validity.valid_until_unix_s.to_le_bytes());
    Ok(hasher.finalize().to_hex().to_string())
}

fn match_capability(
    required_process: &ProcessDefinitionId,
    required_envelope_ref: &str,
    minimum_evidence: CapabilityEvidenceClassV1,
    candidate: &ProcessCapabilityProfileV1,
    evaluation_time_unix_s: u64,
    external_refs_resolved: bool,
) -> CapabilityMatchV1 {
    if validate_profile(candidate).is_err() {
        return CapabilityMatchV1::Unknown;
    }
    if candidate.process_id != *required_process {
        return CapabilityMatchV1::ProcessMismatch;
    }
    if candidate.envelope_profile_ref != required_envelope_ref {
        return CapabilityMatchV1::EnvelopeMismatch;
    }
    if evaluation_time_unix_s < candidate.validity.valid_from_unix_s
        || evaluation_time_unix_s >= candidate.validity.valid_until_unix_s
    {
        return CapabilityMatchV1::ExpiredOrStale;
    }
    if !external_refs_resolved {
        return CapabilityMatchV1::UnresolvedExternalRefs;
    }
    if candidate.evidence_class < minimum_evidence {
        return CapabilityMatchV1::CompatibleButWeakerEvidence;
    }
    CapabilityMatchV1::ExactAdmitted
}

fn candidate(class: CapabilityEvidenceClassV1) -> ProcessCapabilityProfileV1 {
    ProcessCapabilityProfileV1 {
        resource_subject_ref: "eng-catalog:machine:abc".into(),
        process_id: ProcessDefinitionId("process:cnc-milling-v1".into()),
        envelope_profile_ref: "se-sem:envelope:al-6061-mill-v1".into(),
        configuration_profile_ref: "mfg:fixture:vise-v2".into(),
        evidence_class: class,
        evidence_ref: "evidence:qualification:123".into(),
        validity: CapabilityValidityV1 {
            revision: "rev-4".into(),
            valid_from_unix_s: 100,
            valid_until_unix_s: 200,
        },
        display_label: Some("Mill A".into()),
    }
}

#[test]
fn declared_capability_cannot_satisfy_qualified_requirement() {
    let c = candidate(CapabilityEvidenceClassV1::Declared);
    assert_eq!(
        match_capability(
            &c.process_id,
            &c.envelope_profile_ref,
            CapabilityEvidenceClassV1::QualifiedUnderProfile,
            &c,
            150,
            true,
        ),
        CapabilityMatchV1::CompatibleButWeakerEvidence
    );
}

#[test]
fn qualified_capability_can_satisfy_weaker_minimum_when_refs_match() {
    let c = candidate(CapabilityEvidenceClassV1::QualifiedUnderProfile);
    assert_eq!(
        match_capability(
            &c.process_id,
            &c.envelope_profile_ref,
            CapabilityEvidenceClassV1::Declared,
            &c,
            150,
            true,
        ),
        CapabilityMatchV1::ExactAdmitted
    );
}

#[test]
fn unresolved_external_envelope_never_becomes_false_success() {
    let c = candidate(CapabilityEvidenceClassV1::ProductionQualifiedUnderProfile);
    assert_eq!(
        match_capability(
            &c.process_id,
            &c.envelope_profile_ref,
            CapabilityEvidenceClassV1::QualifiedUnderProfile,
            &c,
            150,
            false,
        ),
        CapabilityMatchV1::UnresolvedExternalRefs
    );
}

#[test]
fn expired_capability_is_not_admitted() {
    let c = candidate(CapabilityEvidenceClassV1::ProductionQualifiedUnderProfile);
    assert_eq!(
        match_capability(
            &c.process_id,
            &c.envelope_profile_ref,
            CapabilityEvidenceClassV1::QualifiedUnderProfile,
            &c,
            250,
            true,
        ),
        CapabilityMatchV1::ExpiredOrStale
    );
}

#[test]
fn wrong_process_and_envelope_are_distinct_failures() {
    let c = candidate(CapabilityEvidenceClassV1::QualifiedUnderProfile);
    assert_eq!(
        match_capability(
            &ProcessDefinitionId("process:turning-v1".into()),
            &c.envelope_profile_ref,
            CapabilityEvidenceClassV1::Declared,
            &c,
            150,
            true,
        ),
        CapabilityMatchV1::ProcessMismatch
    );
    assert_eq!(
        match_capability(
            &c.process_id,
            "se-sem:envelope:wrong",
            CapabilityEvidenceClassV1::Declared,
            &c,
            150,
            true,
        ),
        CapabilityMatchV1::EnvelopeMismatch
    );
}

#[test]
fn display_label_is_excluded_from_engineering_capability_identity() {
    let mut a = candidate(CapabilityEvidenceClassV1::QualifiedUnderProfile);
    let mut b = a.clone();
    a.display_label = Some("Machine 1".into());
    b.display_label = Some("Renamed UI label".into());
    assert_eq!(capability_identity(&a).unwrap(), capability_identity(&b).unwrap());
}

#[test]
fn semantic_ref_changes_modify_capability_identity() {
    let a = candidate(CapabilityEvidenceClassV1::QualifiedUnderProfile);
    let mut b = a.clone();
    b.envelope_profile_ref = "se-sem:envelope:different".into();
    assert_ne!(capability_identity(&a).unwrap(), capability_identity(&b).unwrap());
}

#[test]
fn serialization_preserves_evidence_and_validity() {
    let c = candidate(CapabilityEvidenceClassV1::ObservedCapability);
    let encoded = serde_json::to_string(&c).unwrap();
    let decoded: ProcessCapabilityProfileV1 = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, c);
    assert_eq!(capability_identity(&decoded), capability_identity(&c));
}
