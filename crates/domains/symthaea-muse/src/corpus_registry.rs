// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rights-aware, artifact-level corpus provenance for Melothaea.
//!
//! A composition, edition, performance, recording, transcription and
//! annotation are separate artifacts with separate rights. Registration is
//! not ingestion approval. Eligibility is evaluated for an explicit use and
//! jurisdiction, and unknown rights or cultural protocol status fail closed.
//! This module records evidence; it is not legal advice or a substitute for
//! rights-holder / community authorization.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const CORPUS_REGISTRY_SCHEMA: &str = "muse-corpus-registry-v1";
pub const WORLDWIDE_JURISDICTION: &str = "WORLDWIDE";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactModality {
    ScoreImage,
    ScorePdf,
    MusicXml,
    Mei,
    SymbolicMidi,
    PerformedMidi,
    AudioRecording,
    VideoRecording,
    ScholarlyAnnotation,
    Metadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsLayer {
    UnderlyingComposition,
    ScoreEdition,
    Performance,
    SoundRecording,
    AudiovisualRecording,
    Annotation,
    Metadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntendedUse {
    InternalAnalysis,
    NonCommercialResearch,
    ModelTraining,
    CommercialTraining,
    DerivativeCreation,
    Redistribution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsDisposition {
    /// No usable rights conclusion has been reached.
    Unknown,
    /// Evidence supports public-domain status for this specific rights layer
    /// and stated jurisdiction(s); this is not inherited by other layers.
    VerifiedPublicDomain,
    /// An identified licence was checked for this rights layer.
    VerifiedOpenLicense,
    /// Direct permission was documented for this rights layer.
    PermissionGranted,
    /// Use is restricted or prohibited by the recorded terms.
    Restricted,
    /// This artifact/layer is explicitly excluded from ingestion.
    Excluded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RightsGrant {
    pub layer: RightsLayer,
    pub disposition: RightsDisposition,
    /// Exact ISO jurisdiction code or WORLDWIDE; do not infer one country's
    /// status from another country's copyright term.
    pub jurisdictions: Vec<String>,
    /// Explicit uses allowed by the evidence. No use is implied by a licence
    /// label or by a public-domain assertion alone.
    pub permitted_uses: Vec<IntendedUse>,
    pub licence_identifier: Option<String>,
    pub attribution_text: Option<String>,
    pub evidence_uri: Option<String>,
    pub reviewer: Option<String>,
    /// ISO-8601 UTC timestamp supplied by the rights review process.
    pub verified_at_utc: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsentRequirement {
    Unknown,
    NotRequired,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsentDisposition {
    Unknown,
    NotApplicable,
    Obtained,
    Pending,
    Denied,
    Restricted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CulturalProtocolEvidence {
    pub requirement: ConsentRequirement,
    pub disposition: ConsentDisposition,
    pub community_or_rights_holder: Option<String>,
    pub protocol_reference: Option<String>,
    pub evidence_uri: Option<String>,
    pub reviewer: Option<String>,
    pub reviewed_at_utc: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestStage {
    Discovered,
    MetadataChecked,
    RightsEvidenceChecked,
    ArtifactDigestVerified,
    Excluded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelKind {
    Composer,
    Performer,
    Instrument,
    Region,
    Period,
    Tradition,
    Style,
    Technique,
    WorkTitle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceLabel {
    pub kind: LabelKind,
    pub value: String,
    pub source_uri: String,
    /// Confidence belongs to the annotation source, not a model's taste score.
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkRelationKind {
    SameWorkDifferentEdition,
    SameWorkDifferentPerformance,
    ArrangementOf,
    TranscriptionOf,
    DerivativeOf,
    PossibleDuplicate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkRelation {
    pub related_artifact_id: String,
    pub relation: WorkRelationKind,
    pub evidence_uri: Option<String>,
    pub confidence_basis: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentEvidence {
    pub method: String,
    pub reference_modality: ArtifactModality,
    pub paired_item_count: usize,
    pub median_onset_error_ms: Option<f64>,
    pub p95_onset_error_ms: Option<f64>,
    pub median_duration_error_ms: Option<f64>,
    pub p95_duration_error_ms: Option<f64>,
    /// Method-specific uncertainty, e.g. alignment posterior, confidence
    /// interval, unmatched-event count or curator caveat.
    pub uncertainty_description: String,
    pub evidence_uri: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusArtifactRecord {
    pub schema: String,
    pub corpus_id: String,
    pub artifact_id: String,
    pub work_id: Option<String>,
    pub title: String,
    pub upstream_uri: String,
    pub publisher_or_maintainer: String,
    pub release: String,
    pub revision: Option<String>,
    pub publication_date: Option<String>,
    pub artifact_sha256: Option<String>,
    pub modality: ArtifactModality,
    pub item_count: usize,
    /// Citation or upstream path explaining where the count came from.
    pub item_count_provenance: String,
    pub duration_seconds: Option<f64>,
    pub source_jurisdiction: Option<String>,
    pub labels: Vec<SourceLabel>,
    pub work_relations: Vec<WorkRelation>,
    pub alignment: Option<AlignmentEvidence>,
    /// Rights are deliberately separate by layer; a score's public-domain
    /// composition does not make a modern edition or recording free to use.
    pub rights: Vec<RightsGrant>,
    pub cultural_protocol: CulturalProtocolEvidence,
    pub ingest_stage: IngestStage,
    pub exclusion_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusRegistrySnapshot {
    pub schema: String,
    pub artifact_count: usize,
    pub artifacts: Vec<CorpusArtifactRecord>,
    /// Domain-separated digest of the snapshot with this field empty.
    pub registry_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EligibilityIssue {
    pub code: String,
    pub layer: Option<RightsLayer>,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EligibilityRequest {
    pub intended_use: IntendedUse,
    /// Jurisdiction for the planned use, not necessarily the source country.
    pub jurisdiction: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusEligibility {
    pub artifact_id: String,
    pub eligible: bool,
    pub intended_use: IntendedUse,
    pub jurisdiction: String,
    pub required_rights_layers: Vec<RightsLayer>,
    pub issues: Vec<EligibilityIssue>,
}

pub fn required_rights_layers(modality: ArtifactModality) -> Vec<RightsLayer> {
    use ArtifactModality as M;
    use RightsLayer as R;
    let mut layers = match modality {
        M::ScoreImage | M::ScorePdf | M::MusicXml | M::Mei | M::SymbolicMidi => {
            vec![R::UnderlyingComposition, R::ScoreEdition]
        }
        M::PerformedMidi => vec![R::UnderlyingComposition, R::Performance],
        M::AudioRecording => vec![
            R::UnderlyingComposition,
            R::Performance,
            R::SoundRecording,
        ],
        M::VideoRecording => vec![
            R::UnderlyingComposition,
            R::Performance,
            R::SoundRecording,
            R::AudiovisualRecording,
        ],
        M::ScholarlyAnnotation => vec![R::Annotation],
        M::Metadata => Vec::new(),
    };
    // Catalogue metadata (labels, provenance, counts and descriptive fields)
    // is itself a rights-bearing layer and must not be swept in with the
    // underlying score or recording.
    if !layers.contains(&R::Metadata) {
        layers.push(R::Metadata);
    }
    layers
}

pub fn validate_artifact_record(record: &CorpusArtifactRecord) -> Vec<EligibilityIssue> {
    let mut issues = Vec::new();
    if record.schema != CORPUS_REGISTRY_SCHEMA {
        issues.push(issue(
            "schema_version_mismatch",
            None,
            format!("expected schema {CORPUS_REGISTRY_SCHEMA}, got {}", record.schema),
        ));
    }
    for (field, value) in [
        ("corpus_id", record.corpus_id.as_str()),
        ("artifact_id", record.artifact_id.as_str()),
        ("title", record.title.as_str()),
        ("upstream_uri", record.upstream_uri.as_str()),
        ("publisher_or_maintainer", record.publisher_or_maintainer.as_str()),
        ("release", record.release.as_str()),
        ("item_count_provenance", record.item_count_provenance.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(issue(
                "required_field_missing",
                None,
                format!("{field} must not be empty"),
            ));
        }
    }
    if record.item_count == 0 {
        issues.push(issue(
            "zero_item_count",
            None,
            "item_count must be positive".into(),
        ));
    }
    if let Some(duration) = record.duration_seconds
        && (!duration.is_finite() || duration <= 0.0)
    {
        issues.push(issue(
            "invalid_duration",
            None,
            "duration_seconds must be finite and positive when provided".into(),
        ));
    }
    if let Some(digest) = &record.artifact_sha256
        && !is_sha256_hex(digest)
    {
        issues.push(issue(
            "invalid_artifact_digest",
            None,
            "artifact_sha256 must be exactly 64 lowercase hexadecimal characters".into(),
        ));
    }
    if matches!(
        record.ingest_stage,
        IngestStage::ArtifactDigestVerified
    ) && record.artifact_sha256.is_none()
    {
        issues.push(issue(
            "digest_stage_without_digest",
            None,
            "ArtifactDigestVerified requires artifact_sha256".into(),
        ));
    }
    if record.ingest_stage == IngestStage::Excluded
        && record.exclusion_reason.as_deref().unwrap_or("").trim().is_empty()
    {
        issues.push(issue(
            "excluded_without_reason",
            None,
            "Excluded records require an explicit exclusion_reason".into(),
        ));
    }
    if record.ingest_stage != IngestStage::Excluded && record.exclusion_reason.is_some() {
        issues.push(issue(
            "exclusion_reason_on_active_record",
            None,
            "active records must not carry exclusion_reason; use Excluded stage".into(),
        ));
    }

    for label in &record.labels {
        if label.value.trim().is_empty() || label.source_uri.trim().is_empty() {
            issues.push(issue(
                "label_without_provenance",
                None,
                format!("{:?} labels require a value and source_uri", label.kind),
            ));
        }
        if label.confidence.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            issues.push(issue(
                "invalid_label_confidence",
                None,
                format!("{:?} label confidence must be finite and in [0,1]", label.kind),
            ));
        }
    }

    if let Some(alignment) = &record.alignment {
        if alignment.paired_item_count == 0 {
            issues.push(issue(
                "empty_alignment_evidence",
                None,
                "alignment evidence requires at least one paired item".into(),
            ));
        }
        for (field, value) in [
            ("median_onset_error_ms", alignment.median_onset_error_ms),
            ("p95_onset_error_ms", alignment.p95_onset_error_ms),
            ("median_duration_error_ms", alignment.median_duration_error_ms),
            ("p95_duration_error_ms", alignment.p95_duration_error_ms),
        ] {
            if value.is_some_and(|error| !error.is_finite() || error < 0.0) {
                issues.push(issue(
                    "invalid_alignment_error",
                    None,
                    format!("{field} must be finite and non-negative"),
                ));
            }
        }
        if alignment.uncertainty_description.trim().is_empty() {
            issues.push(issue(
                "alignment_uncertainty_missing",
                None,
                "alignment evidence must describe its uncertainty".into(),
            ));
        }
    }

    let mut seen_layers = BTreeSet::new();
    for grant in &record.rights {
        if !seen_layers.insert(grant.layer) {
            issues.push(issue(
                "duplicate_rights_layer",
                Some(grant.layer),
                "rights layer must have one consolidated grant record per artifact".into(),
            ));
        }
        if grant.jurisdictions.is_empty()
            || grant.jurisdictions.iter().any(|jurisdiction| jurisdiction.trim().is_empty())
        {
            issues.push(issue(
                "rights_jurisdiction_missing",
                Some(grant.layer),
                "rights grants require non-empty explicit jurisdiction(s)".into(),
            ));
        }
        if grant.disposition != RightsDisposition::Unknown {
            if grant.evidence_uri.as_deref().unwrap_or("").trim().is_empty()
                || grant.reviewer.as_deref().unwrap_or("").trim().is_empty()
                || grant.verified_at_utc.as_deref().unwrap_or("").trim().is_empty()
            {
                issues.push(issue(
                    "verified_rights_without_review_evidence",
                    Some(grant.layer),
                    "a non-Unknown rights disposition requires evidence_uri, reviewer and verified_at_utc".into(),
                ));
            }
        }
        if matches!(
            grant.disposition,
            RightsDisposition::VerifiedOpenLicense | RightsDisposition::PermissionGranted
        ) && grant.licence_identifier.as_deref().unwrap_or("").trim().is_empty()
        {
            issues.push(issue(
                "rights_licence_identifier_missing",
                Some(grant.layer),
                "open-license or permission grants require a licence/permission identifier".into(),
            ));
        }
        if grant.disposition == RightsDisposition::VerifiedPublicDomain
            && grant.jurisdictions.iter().any(|jurisdiction| jurisdiction.trim().is_empty())
        {
            issues.push(issue(
                "public_domain_jurisdiction_blank",
                Some(grant.layer),
                "public-domain verification requires explicit jurisdiction(s)".into(),
            ));
        }
    }

    if matches!(
        record.ingest_stage,
        IngestStage::RightsEvidenceChecked | IngestStage::ArtifactDigestVerified
    ) {
        for layer in required_rights_layers(record.modality) {
            match record.rights.iter().find(|grant| grant.layer == layer) {
                None => issues.push(issue(
                    "rights_stage_missing_layer",
                    Some(layer),
                    "rights evidence stage requires a grant record for every modality layer".into(),
                )),
                Some(grant) if grant.disposition == RightsDisposition::Unknown => {
                    issues.push(issue(
                        "rights_stage_unresolved_layer",
                        Some(layer),
                        "rights evidence stage cannot be reached while a required layer is Unknown".into(),
                    ));
                }
                Some(_) => {}
            }
        }
    }

    validate_cultural_protocol(&record.cultural_protocol, &mut issues);
    issues
}

pub fn build_registry_snapshot(
    mut artifacts: Vec<CorpusArtifactRecord>,
) -> Result<CorpusRegistrySnapshot, Vec<EligibilityIssue>> {
    artifacts.sort_by(|left, right| left.artifact_id.cmp(&right.artifact_id));
    let mut issues = Vec::new();
    let mut seen = BTreeSet::new();
    for artifact in &artifacts {
        if !seen.insert(artifact.artifact_id.clone()) {
            issues.push(issue(
                "duplicate_artifact_id",
                None,
                format!("artifact_id {} occurs more than once", artifact.artifact_id),
            ));
        }
        issues.extend(validate_artifact_record(artifact));
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut snapshot = CorpusRegistrySnapshot {
        schema: CORPUS_REGISTRY_SCHEMA.into(),
        artifact_count: artifacts.len(),
        artifacts,
        registry_sha256: String::new(),
    };
    snapshot.registry_sha256 = snapshot_digest(&snapshot)?;
    Ok(snapshot)
}

fn snapshot_digest(snapshot: &CorpusRegistrySnapshot) -> Result<String, Vec<EligibilityIssue>> {
    let mut copy = snapshot.clone();
    copy.registry_sha256.clear();
    let bytes = serde_json::to_vec(&copy).map_err(|error| {
        vec![issue(
            "registry_serialization_failed",
            None,
            error.to_string(),
        )]
    })?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea:muse-corpus-registry-snapshot:v1\n");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

impl CorpusRegistrySnapshot {
    pub fn verify_digest(&self) -> bool {
        snapshot_digest(self)
            .map(|expected| expected == self.registry_sha256)
            .unwrap_or(false)
    }
}

pub fn evaluate_eligibility(
    record: &CorpusArtifactRecord,
    request: &EligibilityRequest,
) -> CorpusEligibility {
    let mut issues = validate_artifact_record(record);
    if request.jurisdiction.trim().is_empty() {
        issues.push(issue(
            "requested_jurisdiction_missing",
            None,
            "eligibility requires an explicit jurisdiction".into(),
        ));
    }
    if record.ingest_stage == IngestStage::Excluded {
        issues.push(issue(
            "artifact_excluded",
            None,
            "artifact is explicitly excluded from ingestion".into(),
        ));
    }

    let required = required_rights_layers(record.modality);
    for layer in &required {
        let Some(grant) = record.rights.iter().find(|grant| grant.layer == *layer) else {
            issues.push(issue(
                "required_rights_layer_missing",
                Some(*layer),
                format!("no rights record for required layer {layer:?}"),
            ));
            continue;
        };

        if grant.disposition == RightsDisposition::Unknown {
            issues.push(issue(
                "rights_unknown",
                Some(*layer),
                format!("{layer:?} rights are unknown"),
            ));
            continue;
        }
        if matches!(
            grant.disposition,
            RightsDisposition::Restricted | RightsDisposition::Excluded
        ) {
            issues.push(issue(
                "rights_restrict_use",
                Some(*layer),
                format!("{layer:?} disposition is {:?}", grant.disposition),
            ));
            continue;
        }

        let jurisdiction = request.jurisdiction.trim().to_ascii_uppercase();
        let covers_jurisdiction = grant.jurisdictions.iter().any(|value| {
            value.trim().eq_ignore_ascii_case(WORLDWIDE_JURISDICTION)
                || value.trim().eq_ignore_ascii_case(&jurisdiction)
        });
        if !covers_jurisdiction {
            issues.push(issue(
                "rights_jurisdiction_not_covered",
                Some(*layer),
                format!(
                    "{layer:?} evidence does not cover requested jurisdiction {}",
                    request.jurisdiction
                ),
            ));
        }
        if !grant.permitted_uses.contains(&request.intended_use) {
            issues.push(issue(
                "intended_use_not_granted",
                Some(*layer),
                format!("{layer:?} does not explicitly permit {:?}", request.intended_use),
            ));
        }
    }

    if record.cultural_protocol.requirement != ConsentRequirement::NotRequired
        && record.cultural_protocol.disposition != ConsentDisposition::Obtained
    {
        issues.push(issue(
            "cultural_consent_not_cleared",
            None,
            "culturally sensitive material requires a documented Obtained disposition before use".into(),
        ));
    }

    CorpusEligibility {
        artifact_id: record.artifact_id.clone(),
        eligible: issues.is_empty(),
        intended_use: request.intended_use,
        jurisdiction: request.jurisdiction.clone(),
        required_rights_layers: required,
        issues,
    }
}

fn validate_cultural_protocol(
    evidence: &CulturalProtocolEvidence,
    issues: &mut Vec<EligibilityIssue>,
) {
    match evidence.requirement {
        ConsentRequirement::Unknown => issues.push(issue(
            "cultural_protocol_unknown",
            None,
            "cultural protocol applicability must be reviewed; Unknown fails closed".into(),
        )),
        ConsentRequirement::NotRequired => {
            if evidence.disposition != ConsentDisposition::NotApplicable {
                issues.push(issue(
                    "cultural_protocol_inconsistent",
                    None,
                    "NotRequired consent policy must use NotApplicable disposition".into(),
                ));
            }
        }
        ConsentRequirement::Required => {
            if evidence.disposition == ConsentDisposition::Obtained
                && (evidence.community_or_rights_holder.as_deref().unwrap_or("").trim().is_empty()
                    || evidence.protocol_reference.as_deref().unwrap_or("").trim().is_empty()
                    || evidence.evidence_uri.as_deref().unwrap_or("").trim().is_empty()
                    || evidence.reviewer.as_deref().unwrap_or("").trim().is_empty()
                    || evidence.reviewed_at_utc.as_deref().unwrap_or("").trim().is_empty())
            {
                issues.push(issue(
                    "consent_obtained_without_evidence",
                    None,
                    "Obtained consent requires holder/community, protocol reference, evidence URI, reviewer and timestamp".into(),
                ));
            }
        }
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn issue(code: &str, layer: Option<RightsLayer>, message: String) -> EligibilityIssue {
    EligibilityIssue {
        code: code.into(),
        layer,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grant(layer: RightsLayer, permitted_uses: Vec<IntendedUse>) -> RightsGrant {
        RightsGrant {
            layer,
            disposition: RightsDisposition::VerifiedOpenLicense,
            jurisdictions: vec!["WORLDWIDE".into()],
            permitted_uses,
            licence_identifier: Some("CC0-1.0".into()),
            attribution_text: None,
            evidence_uri: Some("https://example.org/rights".into()),
            reviewer: Some("rights-reviewer".into()),
            verified_at_utc: Some("2026-10-31T00:00:00Z".into()),
        }
    }

    fn audio_record() -> CorpusArtifactRecord {
        CorpusArtifactRecord {
            schema: CORPUS_REGISTRY_SCHEMA.into(),
            corpus_id: "example-corpus".into(),
            artifact_id: "example-audio-001".into(),
            work_id: Some("work-001".into()),
            title: "Example recording".into(),
            upstream_uri: "https://example.org/item/1".into(),
            publisher_or_maintainer: "Example archive".into(),
            release: "1.0".into(),
            revision: Some("abc123".into()),
            publication_date: Some("1900".into()),
            artifact_sha256: Some("a".repeat(64)),
            modality: ArtifactModality::AudioRecording,
            item_count: 1,
            item_count_provenance: "upstream metadata record".into(),
            duration_seconds: Some(120.0),
            source_jurisdiction: Some("US".into()),
            labels: vec![SourceLabel {
                kind: LabelKind::Composer,
                value: "Example Composer".into(),
                source_uri: "https://example.org/item/1".into(),
                confidence: Some(1.0),
            }],
            work_relations: Vec::new(),
            alignment: None,
            rights: vec![
                grant(
                    RightsLayer::UnderlyingComposition,
                    vec![IntendedUse::NonCommercialResearch, IntendedUse::InternalAnalysis],
                ),
                grant(
                    RightsLayer::Performance,
                    vec![IntendedUse::NonCommercialResearch, IntendedUse::InternalAnalysis],
                ),
                grant(
                    RightsLayer::SoundRecording,
                    vec![IntendedUse::NonCommercialResearch, IntendedUse::InternalAnalysis],
                ),
                grant(
                    RightsLayer::Metadata,
                    vec![IntendedUse::NonCommercialResearch, IntendedUse::InternalAnalysis],
                ),
            ],
            cultural_protocol: CulturalProtocolEvidence {
                requirement: ConsentRequirement::NotRequired,
                disposition: ConsentDisposition::NotApplicable,
                community_or_rights_holder: None,
                protocol_reference: None,
                evidence_uri: None,
                reviewer: None,
                reviewed_at_utc: None,
            },
            ingest_stage: IngestStage::ArtifactDigestVerified,
            exclusion_reason: None,
        }
    }

    fn research_request() -> EligibilityRequest {
        EligibilityRequest {
            intended_use: IntendedUse::NonCommercialResearch,
            jurisdiction: "US".into(),
        }
    }

    #[test]
    fn requires_distinct_rights_for_composition_performance_and_recording() {
        let record = audio_record();
        assert_eq!(
            required_rights_layers(record.modality),
            vec![
                RightsLayer::UnderlyingComposition,
                RightsLayer::Performance,
                RightsLayer::SoundRecording,
                RightsLayer::Metadata
            ]
        );
        assert!(evaluate_eligibility(&record, &research_request()).eligible);
    }

    #[test]
    fn public_domain_composition_does_not_clear_a_restricted_recording() {
        let mut record = audio_record();
        record.rights[0].disposition = RightsDisposition::VerifiedPublicDomain;
        record.rights[0].licence_identifier = None;
        record.rights[0].jurisdictions = vec!["US".into()];
        record.rights[2].disposition = RightsDisposition::Restricted;
        let result = evaluate_eligibility(&record, &research_request());
        assert!(!result.eligible);
        assert!(result.issues.iter().any(|issue| {
            issue.code == "rights_restrict_use"
                && issue.layer == Some(RightsLayer::SoundRecording)
        }));
    }

    #[test]
    fn missing_or_unknown_rights_never_fall_through() {
        let mut record = audio_record();
        record.rights.pop();
        let missing = evaluate_eligibility(&record, &research_request());
        assert!(!missing.eligible);
        assert!(missing.issues.iter().any(|issue| issue.code == "required_rights_layer_missing"));

        let mut record = audio_record();
        record.rights[1].disposition = RightsDisposition::Unknown;
        let unknown = evaluate_eligibility(&record, &research_request());
        assert!(!unknown.eligible);
        assert!(unknown.issues.iter().any(|issue| issue.code == "rights_unknown"));
    }

    #[test]
    fn catalog_metadata_rights_are_required_even_when_audio_rights_are_clear() {
        let mut record = audio_record();
        record.rights.retain(|grant| grant.layer != RightsLayer::Metadata);
        let result = evaluate_eligibility(&record, &research_request());
        assert!(!result.eligible);
        assert!(result.issues.iter().any(|issue| {
            issue.code == "required_rights_layer_missing"
                && issue.layer == Some(RightsLayer::Metadata)
        }));
    }

    #[test]
    fn eligibility_is_use_and_jurisdiction_specific() {
        let record = audio_record();
        let commercial = EligibilityRequest {
            intended_use: IntendedUse::CommercialTraining,
            jurisdiction: "US".into(),
        };
        assert!(!evaluate_eligibility(&record, &commercial).eligible);

        let foreign = EligibilityRequest {
            intended_use: IntendedUse::NonCommercialResearch,
            jurisdiction: "ZA".into(),
        };
        assert!(evaluate_eligibility(&record, &foreign).eligible);
    }

    #[test]
    fn consent_required_fails_closed_without_documented_obtained_state() {
        let mut record = audio_record();
        record.cultural_protocol.requirement = ConsentRequirement::Required;
        record.cultural_protocol.disposition = ConsentDisposition::Pending;
        let pending = evaluate_eligibility(&record, &research_request());
        assert!(!pending.eligible);
        assert!(pending.issues.iter().any(|issue| issue.code == "cultural_consent_not_cleared"));

        record.cultural_protocol.disposition = ConsentDisposition::Obtained;
        record.cultural_protocol.community_or_rights_holder = Some("Rights holder".into());
        record.cultural_protocol.protocol_reference = Some("protocol-v1".into());
        record.cultural_protocol.evidence_uri = Some("https://example.org/consent".into());
        record.cultural_protocol.reviewer = Some("reviewer".into());
        record.cultural_protocol.reviewed_at_utc = Some("2026-10-31T00:00:00Z".into());
        assert!(evaluate_eligibility(&record, &research_request()).eligible);
    }

    #[test]
    fn bad_digests_and_unverified_cultural_scope_are_reported() {
        let mut record = audio_record();
        record.artifact_sha256 = Some("not-a-sha".into());
        record.cultural_protocol.requirement = ConsentRequirement::Unknown;
        let issues = validate_artifact_record(&record);
        assert!(issues.iter().any(|issue| issue.code == "invalid_artifact_digest"));
        assert!(issues.iter().any(|issue| issue.code == "cultural_protocol_unknown"));
    }

    #[test]
    fn score_and_annotation_modalities_require_their_own_rights_layers() {
        assert_eq!(
            required_rights_layers(ArtifactModality::ScorePdf),
            vec![
                RightsLayer::UnderlyingComposition,
                RightsLayer::ScoreEdition,
                RightsLayer::Metadata
            ]
        );
        assert_eq!(
            required_rights_layers(ArtifactModality::ScholarlyAnnotation),
            vec![RightsLayer::Annotation, RightsLayer::Metadata]
        );
    }

    #[test]
    fn rights_review_stage_cannot_be_reached_with_unknown_required_layer() {
        let mut record = audio_record();
        record.rights[1].disposition = RightsDisposition::Unknown;
        record.ingest_stage = IngestStage::RightsEvidenceChecked;
        let issues = validate_artifact_record(&record);
        assert!(issues.iter().any(|issue| {
            issue.code == "rights_stage_unresolved_layer"
                && issue.layer == Some(RightsLayer::Performance)
        }));
    }

    #[test]
    fn registry_snapshot_is_sorted_deduplicated_and_digest_bound() {
        let first = audio_record();
        let mut second = audio_record();
        second.artifact_id = "example-audio-002".into();
        second.title = "Second recording".into();
        let left = build_registry_snapshot(vec![second.clone(), first.clone()]).unwrap();
        let right = build_registry_snapshot(vec![first.clone(), second.clone()]).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.artifact_count, 2);
        assert!(left.verify_digest());

        let duplicate = build_registry_snapshot(vec![first.clone(), first]).unwrap_err();
        assert!(duplicate.iter().any(|issue| issue.code == "duplicate_artifact_id"));
    }

    #[test]
    fn eligible_output_is_bound_to_the_specific_request() {
        let record = audio_record();
        let result = evaluate_eligibility(&record, &research_request());
        assert_eq!(result.artifact_id, "example-audio-001");
        assert_eq!(result.intended_use, IntendedUse::NonCommercialResearch);
        assert_eq!(result.jurisdiction, "US");
        assert!(result.eligible);
        assert!(result.issues.is_empty());
    }
}
