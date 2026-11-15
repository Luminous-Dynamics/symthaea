//! G3a evidence contract for neutral CAD translation.
//!
//! This module does not parse STEP or validate a B-rep. It qualifies the
//! structure and provenance of a translation receipt. A receipt is only as
//! trustworthy as the independent backends and evidence supplied to it.

use serde::{Deserialize, Serialize};

const RECEIPT_SCHEMA_VERSION: u32 = 1;
const BLAKE3_HEX_LEN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GeometrySemantic {
    DisplacementVolumeShell,
    FullExternalHull,
    LegacyReconstruction,
    ComponentOrSubassembly,
    GenericSurface,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RepresentationKind {
    PolyhedralMesh,
    TessellatedBrep,
    ExactBrep,
    ParametricSurfaceDefinition,
    SectionCurves,
    PointCloud,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StepSchemaEdition {
    Ap214,
    #[serde(rename = "ap242-2025-edition-4")]
    Ap2422025Edition4,
    #[serde(rename = "ap242-edition-5-committee-draft")]
    Ap242Edition5CommitteeDraft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvidenceStatus {
    Pass,
    Fail,
    Unknown,
    NotRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TranslationDisposition {
    QualifiedWithinProfile,
    Rejected,
    Incomplete,
    Incomparable,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CadViolation {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactReference {
    pub artifact_id: String,
    pub format: String,
    pub digest_blake3: String,
    pub source_commit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendIdentity {
    /// Product/backend implementation family, not an arbitrary executable label.
    pub backend_family: String,
    pub backend_id: String,
    pub version: String,
    pub build_digest_blake3: String,
    pub configuration_digest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslationToleranceProfile {
    pub id: String,
    pub max_absolute_length_deviation_m: f64,
    pub max_relative_volume_deviation: f64,
    pub max_relative_area_deviation: f64,
    pub max_centroid_deviation_m: f64,
    pub max_surface_deviation_m: f64,
    pub max_edge_gap_m: f64,
    pub required_checks: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslationCheck {
    pub code: String,
    pub status: EvidenceStatus,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslationMetrics {
    pub max_absolute_length_deviation_m: Option<f64>,
    pub relative_volume_deviation: Option<f64>,
    pub relative_area_deviation: Option<f64>,
    pub centroid_deviation_m: Option<f64>,
    pub max_surface_deviation_m: Option<f64>,
    pub max_edge_gap_m: Option<f64>,
    pub source_solid_count: Option<u64>,
    pub imported_solid_count: Option<u64>,
    pub source_shell_count: Option<u64>,
    pub imported_shell_count: Option<u64>,
    pub source_face_count: Option<u64>,
    pub imported_face_count: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslationEvidence {
    pub checks: Vec<TranslationCheck>,
    pub metrics: TranslationMetrics,
    pub disposition: TranslationDisposition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CadTranslationPayload {
    pub receipt_schema_version: u32,
    pub candidate_id: String,
    pub geometry_semantic: GeometrySemantic,
    /// Explicit, human-readable statement of what the artifact represents.
    pub semantic_scope: String,
    /// True only when the external deck/freeboard geometry is actually modelled.
    pub has_deck_freeboard_surface: bool,
    /// A displacement shell's waterplane cap must never be labelled as deck/freeboard.
    pub waterplane_cap_is_deck: bool,
    pub source_artifact: ArtifactReference,
    pub source_representation: RepresentationKind,
    pub target_representation: RepresentationKind,
    pub step_schema_edition: StepSchemaEdition,
    pub step_file_digest_blake3: String,
    pub exporter: BackendIdentity,
    pub importer: BackendIdentity,
    pub tolerance_profile: TranslationToleranceProfile,
    pub evidence: TranslationEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CadTranslationReceipt {
    pub payload: CadTranslationPayload,
    /// BLAKE3 of the normalized typed payload. Integrity aid, not a signature.
    pub receipt_digest_blake3: String,
}

fn is_blake3_digest(value: &str) -> bool {
    value.len() == BLAKE3_HEX_LEN && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_git_commit_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn add_violation(
    violations: &mut Vec<CadViolation>,
    code: &str,
    detail: impl Into<String>,
) {
    violations.push(CadViolation {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

pub fn compute_receipt_digest(
    payload: &CadTranslationPayload,
) -> Result<String, serde_json::Error> {
    let normalized = serde_json::to_vec(payload)?;
    Ok(blake3::hash(&normalized).to_hex().to_string())
}

fn backends_are_independent(exporter: &BackendIdentity, importer: &BackendIdentity) -> bool {
    !exporter.backend_family.trim().is_empty()
        && !importer.backend_family.trim().is_empty()
        && exporter.backend_family != importer.backend_family
        && exporter.backend_id != importer.backend_id
        && exporter.build_digest_blake3 != importer.build_digest_blake3
}

fn disposition_from_checks(
    profile: &TranslationToleranceProfile,
    checks: &[TranslationCheck],
) -> TranslationDisposition {
    let mut incomplete = false;
    let mut failed = false;
    for required in &profile.required_checks {
        match checks.iter().find(|check| check.code == *required) {
            Some(check) if check.status == EvidenceStatus::Pass => {}
            Some(check) if check.status == EvidenceStatus::Fail => failed = true,
            Some(_) | None => incomplete = true,
        }
    }
    if failed {
        TranslationDisposition::Rejected
    } else if incomplete {
        TranslationDisposition::Incomplete
    } else {
        TranslationDisposition::QualifiedWithinProfile
    }
}

/// Validate a translation receipt's evidence and integrity envelope.
///
/// This function cannot determine whether the STEP bytes encode a valid B-rep.
/// It verifies that the receipt claims a supported release, declares its
/// geometry semantic, binds source/export/import identities and supplies
/// independent evidence for every required check in its frozen profile.
pub fn verify_translation_receipt(
    receipt: &CadTranslationReceipt,
) -> Result<(), Vec<CadViolation>> {
    let mut violations = Vec::new();
    let payload = &receipt.payload;

    if payload.receipt_schema_version != RECEIPT_SCHEMA_VERSION {
        add_violation(
            &mut violations,
            "receipt_schema_version",
            format!(
                "receipt schema {} is unsupported; expected {}",
                payload.receipt_schema_version, RECEIPT_SCHEMA_VERSION
            ),
        );
    }
    if payload.candidate_id.trim().is_empty()
        || payload.source_artifact.artifact_id.trim().is_empty()
        || payload.semantic_scope.trim().is_empty()
        || payload.tolerance_profile.id.trim().is_empty()
    {
        add_violation(
            &mut violations,
            "required_identity",
            "candidate, source artifact, semantic scope and tolerance profile IDs must be present",
        );
    }

    if !is_git_commit_id(&payload.source_artifact.source_commit) {
        add_violation(
            &mut violations,
            "source_commit",
            "source commit must be a 40- or 64-character hexadecimal object ID",
        );
    }

    if !is_blake3_digest(&payload.source_artifact.digest_blake3)
        || !is_blake3_digest(&payload.step_file_digest_blake3)
        || !is_blake3_digest(&payload.exporter.build_digest_blake3)
        || !is_blake3_digest(&payload.exporter.configuration_digest_blake3)
        || !is_blake3_digest(&payload.importer.build_digest_blake3)
        || !is_blake3_digest(&payload.importer.configuration_digest_blake3)
        || !is_blake3_digest(&receipt.receipt_digest_blake3)
    {
        add_violation(
            &mut violations,
            "digest_format",
            "source, STEP, backend build/configuration and receipt digests must be 64-character hexadecimal BLAKE3 values",
        );
    }

    if payload.exporter.backend_family.trim().is_empty()
        || payload.exporter.backend_id.trim().is_empty()
        || payload.exporter.version.trim().is_empty()
        || payload.importer.backend_family.trim().is_empty()
        || payload.importer.backend_id.trim().is_empty()
        || payload.importer.version.trim().is_empty()
    {
        add_violation(
            &mut violations,
            "backend_identity",
            "exporter and importer backend IDs and versions must be present",
        );
    }

    if !backends_are_independent(&payload.exporter, &payload.importer) {
        add_violation(
            &mut violations,
            "independent_backend_required",
            "exporter and importer must be independently identified; same-kernel round-trip is only a diagnostic control",
        );
    }

    if payload.step_schema_edition != StepSchemaEdition::Ap2422025Edition4 {
        add_violation(
            &mut violations,
            "step_schema_edition",
            "the initial qualified exchange profile requires released ISO 10303-242:2025 Edition 4; AP214 and the Edition 5 committee draft are not interchangeable",
        );
    }

    if payload.geometry_semantic == GeometrySemantic::DisplacementVolumeShell
        && (payload.waterplane_cap_is_deck || payload.has_deck_freeboard_surface)
    {
        add_violation(
            &mut violations,
            "geometry_semantic_mismatch",
            "a displacement-volume waterplane cap is not a deck/freeboard surface",
        );
    }

    if payload.geometry_semantic == GeometrySemantic::FullExternalHull
        && (!payload.has_deck_freeboard_surface
            || payload.semantic_scope.to_ascii_lowercase().contains("waterplane-capped displacement"))
    {
        add_violation(
            &mut violations,
            "full_hull_semantic_mismatch",
            "a waterplane-capped displacement shell cannot be labelled a full external hull",
        );
    }

    let tolerance_values = [
        payload.tolerance_profile.max_absolute_length_deviation_m,
        payload.tolerance_profile.max_relative_volume_deviation,
        payload.tolerance_profile.max_relative_area_deviation,
        payload.tolerance_profile.max_centroid_deviation_m,
        payload.tolerance_profile.max_surface_deviation_m,
        payload.tolerance_profile.max_edge_gap_m,
    ];
    if tolerance_values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        add_violation(
            &mut violations,
            "tolerance_profile",
            "all tolerance bounds must be finite and non-negative",
        );
    }
    if payload.tolerance_profile.required_checks.is_empty() {
        add_violation(
            &mut violations,
            "required_checks",
            "a qualification profile must declare at least one required check",
        );
    }

    let mut required_seen = std::collections::BTreeSet::new();
    for required in &payload.tolerance_profile.required_checks {
        if required.trim().is_empty() || !required_seen.insert(required.as_str()) {
            add_violation(
                &mut violations,
                "required_check_identity",
                "required check codes must be non-empty and unique",
            );
        }
    }

    let mut seen = std::collections::BTreeSet::new();
    for check in &payload.evidence.checks {
        if check.code.trim().is_empty() || !seen.insert(check.code.as_str()) {
            add_violation(
                &mut violations,
                "check_identity",
                "evidence check codes must be non-empty and unique",
            );
        }
    }

    for required in &payload.tolerance_profile.required_checks {
        match payload.evidence.checks.iter().find(|check| check.code == *required) {
            Some(check) if check.status == EvidenceStatus::Pass => {}
            Some(check) if check.status == EvidenceStatus::Fail => add_violation(
                &mut violations,
                "required_check_failed",
                format!("required translation check '{}' failed", required),
            ),
            Some(_) => add_violation(
                &mut violations,
                "required_check_unknown",
                format!("required translation check '{}' is unknown or not required", required),
            ),
            None => add_violation(
                &mut violations,
                "required_check_missing",
                format!("required translation check '{}' is missing", required),
            ),
        }
    }

    let metric_requirements = [
        (
            "length-deviation",
            payload.evidence.metrics.max_absolute_length_deviation_m,
            payload.tolerance_profile.max_absolute_length_deviation_m,
        ),
        (
            "volume-deviation",
            payload.evidence.metrics.relative_volume_deviation,
            payload.tolerance_profile.max_relative_volume_deviation,
        ),
        (
            "area-deviation",
            payload.evidence.metrics.relative_area_deviation,
            payload.tolerance_profile.max_relative_area_deviation,
        ),
        (
            "centroid-deviation",
            payload.evidence.metrics.centroid_deviation_m,
            payload.tolerance_profile.max_centroid_deviation_m,
        ),
        (
            "surface-deviation",
            payload.evidence.metrics.max_surface_deviation_m,
            payload.tolerance_profile.max_surface_deviation_m,
        ),
        (
            "edge-gap",
            payload.evidence.metrics.max_edge_gap_m,
            payload.tolerance_profile.max_edge_gap_m,
        ),
    ];
    for (check_code, measured, limit) in metric_requirements {
        if payload.tolerance_profile.required_checks.iter().any(|required| required == check_code) {
            match measured {
                Some(value) if value.is_finite() && value >= 0.0 && value <= limit => {}
                Some(value) if !value.is_finite() || value < 0.0 || value > limit => add_violation(
                    &mut violations,
                    "metric_outside_tolerance",
                    format!("metric '{}' value {} exceeds or invalidates its declared tolerance {}", check_code, value, limit),
                ),
                None => add_violation(
                    &mut violations,
                    "required_metric_missing",
                    format!("required metric '{}' has no measured value", check_code),
                ),
            }
        }
    }

    if payload.geometry_semantic == GeometrySemantic::FullExternalHull
        && payload.target_representation != RepresentationKind::ExactBrep
    {
        add_violation(
            &mut violations,
            "full_hull_requires_exact_brep",
            "the initial full-external-hull exchange profile requires an exact B-rep target; a tessellation alone is not a CAD hull surface",
        );
    }

    if payload.evidence.disposition != TranslationDisposition::QualifiedWithinProfile
        || disposition_from_checks(
            &payload.tolerance_profile,
            &payload.evidence.checks,
        ) != TranslationDisposition::QualifiedWithinProfile
    {
        add_violation(
            &mut violations,
            "disposition_not_qualified",
            "receipt cannot be qualified unless every required check is explicitly PASS",
        );
    }

    match compute_receipt_digest(payload) {
        Ok(actual) if actual.eq_ignore_ascii_case(&receipt.receipt_digest_blake3) => {}
        Ok(_) => add_violation(
            &mut violations,
            "receipt_digest_mismatch",
            "receipt payload does not match receipt_digest_blake3",
        ),
        Err(error) => add_violation(
            &mut violations,
            "receipt_digest",
            format!("receipt payload cannot be serialized: {error}"),
        ),
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Derive a conservative disposition from an evidence payload without
/// overriding the supplied check statuses.
pub fn evaluate_translation_disposition(
    profile: &TranslationToleranceProfile,
    checks: &[TranslationCheck],
) -> TranslationDisposition {
    disposition_from_checks(profile, checks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        blake3::hash(label.as_bytes()).to_hex().to_string()
    }

    fn profile() -> TranslationToleranceProfile {
        TranslationToleranceProfile {
            id: "marine-ap242-exchange-v1".into(),
            max_absolute_length_deviation_m: 1.0e-5,
            max_relative_volume_deviation: 1.0e-8,
            max_relative_area_deviation: 1.0e-8,
            max_centroid_deviation_m: 1.0e-5,
            max_surface_deviation_m: 1.0e-5,
            max_edge_gap_m: 1.0e-6,
            required_checks: vec![
                "step-schema".into(),
                "independent-backend".into(),
                "brep-validity".into(),
                "geometry-equivalence".into(),
                "units-frame".into(),
                "semantic-preservation".into(),
            ],
        }
    }

    fn backend(id: &str) -> BackendIdentity {
        BackendIdentity {
            backend_family: id.into(),
            backend_id: id.into(),
            version: "1.0.0".into(),
            build_digest_blake3: digest(&format!("{id}-build")),
            configuration_digest_blake3: digest(&format!("{id}-config")),
        }
    }

    fn receipt() -> CadTranslationReceipt {
        let tolerance_profile = profile();
        let checks = tolerance_profile
            .required_checks
            .iter()
            .map(|code| TranslationCheck {
                code: code.clone(),
                status: EvidenceStatus::Pass,
                detail: "fixture evidence".into(),
            })
            .collect();
        let payload = CadTranslationPayload {
            receipt_schema_version: 1,
            candidate_id: "cad-fixture-001".into(),
            geometry_semantic: GeometrySemantic::FullExternalHull,
            semantic_scope: "full external hull including deck/freeboard surfaces".into(),
            has_deck_freeboard_surface: true,
            waterplane_cap_is_deck: false,
            source_artifact: ArtifactReference {
                artifact_id: "source-geometry-001".into(),
                format: "symthaea-marine-geometry-v1".into(),
                digest_blake3: digest("source"),
                source_commit: "a".repeat(40),
            },
            source_representation: RepresentationKind::ParametricSurfaceDefinition,
            target_representation: RepresentationKind::ExactBrep,
            step_schema_edition: StepSchemaEdition::Ap242_2025_Edition4,
            step_file_digest_blake3: digest("step-file"),
            exporter: backend("occt-export"),
            importer: backend("independent-cad-import"),
            tolerance_profile,
            evidence: TranslationEvidence {
                checks,
                metrics: TranslationMetrics {
                    max_absolute_length_deviation_m: Some(1.0e-8),
                    relative_volume_deviation: Some(1.0e-10),
                    relative_area_deviation: Some(1.0e-10),
                    centroid_deviation_m: Some(1.0e-8),
                    max_surface_deviation_m: Some(1.0e-8),
                    max_edge_gap_m: Some(1.0e-9),
                    source_solid_count: Some(1),
                    imported_solid_count: Some(1),
                    source_shell_count: Some(1),
                    imported_shell_count: Some(1),
                    source_face_count: Some(256),
                    imported_face_count: Some(260),
                },
                disposition: TranslationDisposition::QualifiedWithinProfile,
            },
        };
        let receipt_digest_blake3 = compute_receipt_digest(&payload).unwrap();
        CadTranslationReceipt {
            payload,
            receipt_digest_blake3,
        }
    }

    #[test]
    fn independent_ap242_2025_receipt_passes_envelope_verification() {
        assert!(verify_translation_receipt(&receipt()).is_ok());
    }

    #[test]
    fn ap214_is_not_silently_promoted_to_ap242() {
        let mut item = receipt();
        item.payload.step_schema_edition = StepSchemaEdition::Ap214;
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error.iter().any(|issue| issue.code == "step_schema_edition"));
    }

    #[test]
    fn ap242_edition_five_draft_is_not_treated_as_released_edition_four() {
        let mut item = receipt();
        item.payload.step_schema_edition = StepSchemaEdition::Ap242Edition5CommitteeDraft;
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error.iter().any(|issue| issue.code == "step_schema_edition"));
    }

    #[test]
    fn same_backend_round_trip_is_not_independent_validation() {
        let mut item = receipt();
        item.payload.importer = item.payload.exporter.clone();
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error
            .iter()
            .any(|issue| issue.code == "independent_backend_required"));
    }

    #[test]
    fn unknown_required_check_forces_incomplete() {
        let mut item = receipt();
        item.payload.evidence.checks[2].status = EvidenceStatus::Unknown;
        item.payload.evidence.disposition = TranslationDisposition::Incomplete;
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error
            .iter()
            .any(|issue| issue.code == "required_check_unknown"));
    }

    #[test]
    fn changed_semantics_cannot_relabel_waterplane_cap_as_deck() {
        let mut item = receipt();
        item.payload.geometry_semantic = GeometrySemantic::DisplacementVolumeShell;
        item.payload.semantic_scope = "waterplane-capped displacement volume".into();
        item.payload.has_deck_freeboard_surface = false;
        item.payload.waterplane_cap_is_deck = true;
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error
            .iter()
            .any(|issue| issue.code == "geometry_semantic_mismatch"));
    }

    #[test]
    fn full_external_hull_cannot_be_a_waterplane_capped_displacement_shell() {
        let mut item = receipt();
        item.payload.has_deck_freeboard_surface = false;
        item.payload.semantic_scope = "waterplane-capped displacement shell".into();
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error
            .iter()
            .any(|issue| issue.code == "full_hull_semantic_mismatch"));
    }

    #[test]
    fn translation_metric_cannot_exceed_its_frozen_tolerance() {
        let mut item = receipt();
        item.payload.evidence.metrics.relative_volume_deviation = Some(1.0e-3);
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error.iter().any(|issue| issue.code == "metric_outside_tolerance"));
    }

    #[test]
    fn full_hull_cannot_be_qualified_by_tessellation_only() {
        let mut item = receipt();
        item.payload.target_representation = RepresentationKind::TessellatedBrep;
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error.iter().any(|issue| issue.code == "full_hull_requires_exact_brep"));
    }

    #[test]
    fn required_metric_missing_cannot_be_promoted_to_pass() {
        let mut item = receipt();
        item.payload.evidence.metrics.max_surface_deviation_m = None;
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        item.payload.tolerance_profile.required_checks.push("surface-deviation".into());
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error.iter().any(|issue| issue.code == "required_metric_missing"));
    }

    #[test]
    fn duplicate_required_checks_are_rejected() {
        let mut item = receipt();
        item.payload.tolerance_profile.required_checks.push("brep-validity".into());
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error.iter().any(|issue| issue.code == "required_check_identity"));
    }

    #[test]
    fn malformed_source_commit_identity_is_rejected() {
        let mut item = receipt();
        item.payload.source_artifact.source_commit = "not-a-commit".into();
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error.iter().any(|issue| issue.code == "source_commit"));
    }

    #[test]
    fn receipt_payload_tampering_is_detected() {
        let mut item = receipt();
        item.payload.semantic_scope = "modified after translation".into();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error
            .iter()
            .any(|issue| issue.code == "receipt_digest_mismatch"));
    }

    #[test]
    fn missing_required_check_is_not_defaulted_to_pass() {
        let mut item = receipt();
        item.payload.evidence.checks.pop();
        item.payload.evidence.disposition = TranslationDisposition::Incomplete;
        item.receipt_digest_blake3 = compute_receipt_digest(&item.payload).unwrap();
        let error = verify_translation_receipt(&item).unwrap_err();
        assert!(error
            .iter()
            .any(|issue| issue.code == "required_check_missing"));
    }
}
