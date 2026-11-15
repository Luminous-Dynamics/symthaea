//! Evidence-backed capability manifests for candidate CAD backends.
//!
//! A feature claim is not treated as a qualified capability. Each PASS must
//! reference both a frozen fixture manifest and a test report digest.

use serde::{Deserialize, Serialize};

use crate::cad_exchange::{
    BackendIdentity, EvidenceStatus, StepSchemaEdition,
};

const CAPABILITY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CadCapability {
    StepAp214Import,
    StepAp214Export,
    StepAp2422025Edition4Import,
    StepAp2422025Edition4Export,
    ExactBrepImport,
    ExactBrepExport,
    TessellatedGeometryImport,
    TessellatedGeometryExport,
    NurbsCurveSurfaceSupport,
    ClosedSolidValidation,
    BrepTopologyValidation,
    SelfIntersectionDetection,
    GeometryValidationProperties,
    PmiGdtPreservation,
    PersistentIdPreservation,
    ProductAssemblyStructure,
    MaterialMetadataPreservation,
    UnitFramePreservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityEvidence {
    pub capability: CadCapability,
    pub status: EvidenceStatus,
    /// Frozen corpus identity used to exercise this capability.
    pub fixture_manifest_digest_blake3: Option<String>,
    /// Exact report produced by the test or independent validator.
    pub test_report_digest_blake3: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCapabilityPayload {
    pub capability_schema_version: u32,
    pub backend: BackendIdentity,
    /// This is descriptive metadata. A schema is qualified only when the
    /// corresponding import/export capability has PASS evidence.
    pub declared_default_step_schema: Option<StepSchemaEdition>,
    pub evidence: Vec<CapabilityEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCapabilityManifest {
    pub payload: BackendCapabilityPayload,
    /// BLAKE3 of the normalized typed payload; not a signature.
    pub manifest_digest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityRequirementProfile {
    pub id: String,
    pub required_capabilities: Vec<CadCapability>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapabilityDisposition {
    QualifiedWithinProfile,
    Rejected,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityViolation {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityEvaluation {
    pub backend_id: String,
    pub backend_version: String,
    pub profile_id: String,
    pub disposition: CapabilityDisposition,
    pub missing: Vec<CadCapability>,
    pub unknown: Vec<CadCapability>,
    pub failed: Vec<CadCapability>,
}

fn is_blake3_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn add_violation(
    violations: &mut Vec<CapabilityViolation>,
    code: &str,
    detail: impl Into<String>,
) {
    violations.push(CapabilityViolation {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

pub fn compute_backend_capability_digest(
    payload: &BackendCapabilityPayload,
) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(payload)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Verify manifest integrity and evidence quality.
///
/// PASS without both a fixture-manifest digest and a test-report digest is
/// invalid. A prose capability claim cannot qualify a backend.
pub fn verify_backend_capability_manifest(
    manifest: &BackendCapabilityManifest,
) -> Result<(), Vec<CapabilityViolation>> {
    let mut violations = Vec::new();
    let payload = &manifest.payload;

    if payload.capability_schema_version != CAPABILITY_SCHEMA_VERSION {
        add_violation(
            &mut violations,
            "capability_schema_version",
            format!(
                "capability manifest version {} is unsupported",
                payload.capability_schema_version
            ),
        );
    }
    if payload.backend.backend_family.trim().is_empty()
        || payload.backend.backend_id.trim().is_empty()
        || payload.backend.version.trim().is_empty()
    {
        add_violation(
            &mut violations,
            "backend_identity",
            "backend family, ID and version must be explicit",
        );
    }
    for (name, value) in [
        ("backend build", &payload.backend.build_digest_blake3),
        (
            "backend configuration",
            &payload.backend.configuration_digest_blake3,
        ),
        ("manifest", &manifest.manifest_digest_blake3),
    ] {
        if !is_blake3_digest(value) {
            add_violation(
                &mut violations,
                "digest_format",
                format!("{name} digest must be a 64-character hexadecimal BLAKE3 value"),
            );
        }
    }

    let mut seen = std::collections::BTreeSet::new();
    for evidence in &payload.evidence {
        if !seen.insert(evidence.capability) {
            add_violation(
                &mut violations,
                "duplicate_capability",
                format!("capability {:?} appears more than once", evidence.capability),
            );
        }
        if evidence.status == EvidenceStatus::Pass || evidence.status == EvidenceStatus::Fail {
            match &evidence.fixture_manifest_digest_blake3 {
                Some(value) if is_blake3_digest(value) => {}
                _ => add_violation(
                    &mut violations,
                    "pass_without_fixture",
                    format!(
                        "capability {:?} has a PASS/FAIL status but no valid fixture-manifest digest",
                        evidence.capability
                    ),
                ),
            }
            match &evidence.test_report_digest_blake3 {
                Some(value) if is_blake3_digest(value) => {}
                _ => add_violation(
                    &mut violations,
                    "pass_without_test_report",
                    format!(
                        "capability {:?} has a PASS/FAIL status but no valid test-report digest",
                        evidence.capability
                    ),
                ),
            }
        }
    }

    match compute_backend_capability_digest(payload) {
        Ok(actual) if actual.eq_ignore_ascii_case(&manifest.manifest_digest_blake3) => {}
        Ok(_) => add_violation(
            &mut violations,
            "manifest_digest_mismatch",
            "backend capability payload does not match its manifest digest",
        ),
        Err(error) => add_violation(
            &mut violations,
            "manifest_serialization",
            format!("capability payload cannot be serialized: {error}"),
        ),
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Evaluate an independently defined requirement profile against the
/// evidence manifest. Missing or UNKNOWN capabilities remain INCOMPLETE.
pub fn evaluate_backend_capabilities(
    manifest: &BackendCapabilityManifest,
    profile: &CapabilityRequirementProfile,
) -> Result<CapabilityEvaluation, Vec<CapabilityViolation>> {
    verify_backend_capability_manifest(manifest)?;

    let mut missing = Vec::new();
    let mut unknown = Vec::new();
    let mut failed = Vec::new();
    let mut profile_seen = std::collections::BTreeSet::new();
    let mut profile_violations = Vec::new();

    if profile.id.trim().is_empty() || profile.required_capabilities.is_empty() {
        add_violation(
            &mut profile_violations,
            "profile_identity",
            "capability profile ID and required capability set must be non-empty",
        );
    }
    for capability in &profile.required_capabilities {
        if !profile_seen.insert(*capability) {
            add_violation(
                &mut profile_violations,
                "duplicate_required_capability",
                format!("required capability {:?} appears more than once", capability),
            );
            continue;
        }
        match manifest
            .payload
            .evidence
            .iter()
            .find(|item| item.capability == *capability)
        {
            None => missing.push(*capability),
            Some(item) if item.status == EvidenceStatus::Pass => {}
            Some(item) if item.status == EvidenceStatus::Fail => failed.push(*capability),
            Some(_) => unknown.push(*capability),
        }
    }

    if !profile_violations.is_empty() {
        return Err(profile_violations);
    }

    let disposition = if !failed.is_empty() {
        CapabilityDisposition::Rejected
    } else if !missing.is_empty() || !unknown.is_empty() {
        CapabilityDisposition::Incomplete
    } else {
        CapabilityDisposition::QualifiedWithinProfile
    };

    Ok(CapabilityEvaluation {
        backend_id: manifest.payload.backend.backend_id.clone(),
        backend_version: manifest.payload.backend.version.clone(),
        profile_id: profile.id.clone(),
        disposition,
        missing,
        unknown,
        failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        blake3::hash(label.as_bytes()).to_hex().to_string()
    }

    fn backend() -> BackendIdentity {
        BackendIdentity {
            backend_family: "occt".into(),
            backend_id: "opencascade-step".into(),
            version: "8.0.1".into(),
            build_digest_blake3: digest("occt-build"),
            configuration_digest_blake3: digest("occt-config"),
        }
    }

    fn evidence(capability: CadCapability, status: EvidenceStatus) -> CapabilityEvidence {
        CapabilityEvidence {
            capability,
            status,
            fixture_manifest_digest_blake3: (status == EvidenceStatus::Pass)
                .then(|| digest("marine-cad-fixtures-v1")),
            test_report_digest_blake3: (status == EvidenceStatus::Pass)
                .then(|| digest(&format!("{capability:?}-report"))),
            detail: "recorded backend test result".into(),
        }
    }

    fn manifest() -> BackendCapabilityManifest {
        let payload = BackendCapabilityPayload {
            capability_schema_version: 1,
            backend: backend(),
            declared_default_step_schema: Some(StepSchemaEdition::Ap2422025Edition4),
            evidence: vec![
                evidence(
                    CadCapability::StepAp2422025Edition4Import,
                    EvidenceStatus::Pass,
                ),
                evidence(
                    CadCapability::StepAp2422025Edition4Export,
                    EvidenceStatus::Pass,
                ),
                evidence(CadCapability::ExactBrepImport, EvidenceStatus::Pass),
                evidence(CadCapability::ExactBrepExport, EvidenceStatus::Pass),
                evidence(
                    CadCapability::NurbsCurveSurfaceSupport,
                    EvidenceStatus::Pass,
                ),
                evidence(
                    CadCapability::BrepTopologyValidation,
                    EvidenceStatus::Pass,
                ),
                evidence(
                    CadCapability::GeometryValidationProperties,
                    EvidenceStatus::Pass,
                ),
            ],
        };
        let manifest_digest_blake3 = compute_backend_capability_digest(&payload).unwrap();
        BackendCapabilityManifest {
            payload,
            manifest_digest_blake3,
        }
    }

    fn ap242_profile() -> CapabilityRequirementProfile {
        CapabilityRequirementProfile {
            id: "marine-ap242e4-full-brep-v1".into(),
            required_capabilities: vec![
                CadCapability::StepAp2422025Edition4Import,
                CadCapability::StepAp2422025Edition4Export,
                CadCapability::ExactBrepImport,
                CadCapability::ExactBrepExport,
                CadCapability::NurbsCurveSurfaceSupport,
                CadCapability::BrepTopologyValidation,
                CadCapability::GeometryValidationProperties,
            ],
        }
    }

    #[test]
    fn explicit_ap242e4_evidence_qualifies_a_backend_profile() {
        let evaluation = evaluate_backend_capabilities(&manifest(), &ap242_profile()).unwrap();
        assert_eq!(
            evaluation.disposition,
            CapabilityDisposition::QualifiedWithinProfile
        );
        assert!(evaluation.missing.is_empty());
        assert!(evaluation.unknown.is_empty());
    }

    #[test]
    fn ap214_export_does_not_substitute_for_ap242e4_export() {
        let mut item = manifest();
        item.payload.evidence.retain(|evidence| {
            evidence.capability != CadCapability::StepAp2422025Edition4Export
        });
        item.payload
            .evidence
            .push(evidence(CadCapability::StepAp214Export, EvidenceStatus::Pass));
        item.manifest_digest_blake3 =
            compute_backend_capability_digest(&item.payload).unwrap();
        let evaluation = evaluate_backend_capabilities(&item, &ap242_profile()).unwrap();
        assert_eq!(evaluation.disposition, CapabilityDisposition::Incomplete);
        assert!(evaluation
            .missing
            .contains(&CadCapability::StepAp2422025Edition4Export));
    }

    #[test]
    fn unknown_self_intersection_capability_remains_incomplete() {
        let mut item = manifest();
        item.payload.evidence.push(evidence(
            CadCapability::SelfIntersectionDetection,
            EvidenceStatus::Unknown,
        ));
        item.manifest_digest_blake3 =
            compute_backend_capability_digest(&item.payload).unwrap();
        let profile = CapabilityRequirementProfile {
            id: "marine-full-surface-v1".into(),
            required_capabilities: vec![CadCapability::SelfIntersectionDetection],
        };
        let evaluation = evaluate_backend_capabilities(&item, &profile).unwrap();
        assert_eq!(evaluation.disposition, CapabilityDisposition::Incomplete);
        assert!(evaluation
            .unknown
            .contains(&CadCapability::SelfIntersectionDetection));
    }

    #[test]
    fn failed_topology_validation_rejects_the_backend_profile() {
        let mut item = manifest();
        let topology = item
            .payload
            .evidence
            .iter_mut()
            .find(|evidence| evidence.capability == CadCapability::BrepTopologyValidation)
            .unwrap();
        topology.status = EvidenceStatus::Fail;
        item.manifest_digest_blake3 =
            compute_backend_capability_digest(&item.payload).unwrap();
        let evaluation = evaluate_backend_capabilities(&item, &ap242_profile()).unwrap();
        assert_eq!(evaluation.disposition, CapabilityDisposition::Rejected);
        assert!(evaluation
            .failed
            .contains(&CadCapability::BrepTopologyValidation));
    }

    #[test]
    fn a_prose_claim_without_fixture_and_report_cannot_be_pass() {
        let mut item = manifest();
        let capability = item
            .payload
            .evidence
            .iter_mut()
            .find(|evidence| evidence.capability == CadCapability::StepAp2422025Edition4Export)
            .unwrap();
        capability.fixture_manifest_digest_blake3 = None;
        capability.test_report_digest_blake3 = None;
        item.manifest_digest_blake3 =
            compute_backend_capability_digest(&item.payload).unwrap();
        let error = verify_backend_capability_manifest(&item).unwrap_err();
        assert!(error.iter().any(|issue| issue.code == "pass_without_fixture"));
        assert!(error
            .iter()
            .any(|issue| issue.code == "pass_without_test_report"));
    }

    #[test]
    fn failed_capability_without_test_report_is_not_accepted_as_evidence() {
        let mut item = manifest();
        let topology = item
            .payload
            .evidence
            .iter_mut()
            .find(|evidence| evidence.capability == CadCapability::BrepTopologyValidation)
            .unwrap();
        topology.status = EvidenceStatus::Fail;
        topology.test_report_digest_blake3 = None;
        item.manifest_digest_blake3 =
            compute_backend_capability_digest(&item.payload).unwrap();
        let error = verify_backend_capability_manifest(&item).unwrap_err();
        assert!(error
            .iter()
            .any(|issue| issue.code == "pass_without_test_report"));
    }

    #[test]
    fn duplicate_capability_evidence_is_rejected() {
        let mut item = manifest();
        item.payload
            .evidence
            .push(evidence(CadCapability::ExactBrepImport, EvidenceStatus::Pass));
        item.manifest_digest_blake3 =
            compute_backend_capability_digest(&item.payload).unwrap();
        let error = verify_backend_capability_manifest(&item).unwrap_err();
        assert!(error.iter().any(|issue| issue.code == "duplicate_capability"));
    }

    #[test]
    fn manifest_payload_tampering_is_detected() {
        let mut item = manifest();
        item.payload.backend.version = "8.0.2".into();
        let error = verify_backend_capability_manifest(&item).unwrap_err();
        assert!(error
            .iter()
            .any(|issue| issue.code == "manifest_digest_mismatch"));
    }
}
