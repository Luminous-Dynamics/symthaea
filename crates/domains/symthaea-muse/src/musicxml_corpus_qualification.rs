// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rights-aware, split-aware qualification for external music corpora.
//!
//! This module records publisher-declared rights and project use scope,
//! fingerprints samples and protocols, and enforces work/composer split rules.
//! It does not download data, give legal advice, or infer rights from titles.
//! A SHA-256 string in a manifest is a digest declaration, not proof that the
//! referenced bytes were actually read and hashed. V1 deliberately stops
//! before claiming byte verification or full legal/annotation qualification.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MUSICXML_CORPUS_REGISTRY_SCHEMA_V1: &str = "muse-musicxml-corpus-registry-v1";
pub const MUSICXML_CORPUS_REGISTRATION_SCHEMA_V1: &str = "muse-musicxml-corpus-registration-v1";
pub const MUSICXML_CORPUS_SAMPLE_MANIFEST_SCHEMA_V1: &str = "muse-musicxml-corpus-sample-manifest-v1";
pub const MUSICXML_CORPUS_PROTOCOL_SCHEMA_V1: &str = "muse-musicxml-corpus-protocol-v1";
pub const MUSICXML_CORPUS_QUALIFICATION_REPORT_SCHEMA_V1: &str = "muse-musicxml-corpus-qualification-report-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusLicenseV1 {
    Cc0_1_0,
    CcByNcSa_4_0,
    ExternalEvaluationTerms,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusUseScopeV1 {
    CommercialAndNonCommercialResearch,
    NonCommercialResearchOnly,
    ExternalEvaluationOnly,
    Unverified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsEvidenceStatusV1 {
    Unknown,
    SourceDeclared,
    HumanReviewed,
    LegalReviewed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusModalityV1 {
    ScoreImage,
    Audio,
    Midi,
    MusicXml,
    MuseScore,
    Kern,
    Lmxe,
    Abc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusTaskV1 {
    OpticalMusicRecognition,
    SymbolicScoreEvaluation,
    AudioToScore,
    ScorePerformanceAlignment,
    AudioToMidi,
    ExternalLeaderboard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusSplitRoleV1 {
    Train,
    Validation,
    Calibration,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusQualificationStageV1 {
    Candidate,
    MetadataRecorded,
    SampleDigestsDeclared,
    SplitQualified,
    ProtocolFrozen,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusSplitPolicyV1 {
    pub work_disjoint: bool,
    pub composer_disjoint: bool,
    pub exact_artifact_disjoint: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusTrackRegistrationV1 {
    pub track_id: String,
    pub task: CorpusTaskV1,
    pub input_modalities: Vec<CorpusModalityV1>,
    pub reference_modalities: Vec<CorpusModalityV1>,
    pub split_policy: CorpusSplitPolicyV1,
    pub external_protocol_label: Option<String>,
    pub training_allowed: bool,
    pub redistribution_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusRegistrationV1 {
    pub schema: String,
    pub corpus_id: String,
    pub release_id: String,
    pub source_uri: String,
    pub license: CorpusLicenseV1,
    pub use_scope: CorpusUseScopeV1,
    pub rights_evidence_status: RightsEvidenceStatusV1,
    pub rights_evidence_uri: String,
    pub rights_notes: String,
    pub tracks: Vec<CorpusTrackRegistrationV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusSampleV1 {
    pub sample_id: String,
    pub fold_id: String,
    pub split: CorpusSplitRoleV1,
    pub composer_group_id: Option<String>,
    pub work_group_id: Option<String>,
    pub input_sha256: String,
    pub reference_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusSampleManifestV1 {
    pub schema: String,
    pub corpus_id: String,
    pub release_id: String,
    pub track_id: String,
    pub samples: Vec<CorpusSampleV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusProtocolFingerprintV1 {
    pub schema: String,
    pub split_protocol_sha256: String,
    pub annotation_protocol_sha256: String,
    pub conversion_pipeline_sha256: String,
    pub parser_build_sha256: String,
    pub evaluator_build_sha256: String,
    pub evaluator_schema: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorpusQualificationIssueV1 {
    WrongRegistrationSchema,
    EmptyCorpusId,
    EmptyReleaseId,
    InvalidSourceUri,
    InvalidRightsEvidenceUri,
    MissingRightsNotes,
    UnknownRightsScope,
    IncompatibleLicenseAndUseScope,
    CommercialUseNotPermitted,
    NonCommercialUseNotEstablished,
    ExternalEvaluationPolicyMissing,
    EmptyTrackRegistry,
    DuplicateTrackId { track_id: String },
    EmptyTrackId,
    EmptyInputModalities { track_id: String },
    EmptyReferenceModalities { track_id: String },
    InvalidTrainingAndRedistributionPolicy { track_id: String },
    WrongSampleManifestSchema,
    SampleCorpusMismatch,
    SampleReleaseMismatch,
    SampleTrackMismatch,
    EmptySampleManifest,
    EmptySampleId,
    EmptyFoldId { sample_id: String },
    DuplicateSampleId { sample_id: String },
    InvalidInputDigest { sample_id: String },
    InvalidReferenceDigest { sample_id: String },
    MissingComposerGroup { sample_id: String },
    MissingWorkGroup { sample_id: String },
    ComposerLeakage { fold_id: String, composer_group_id: String },
    WorkLeakage { fold_id: String, work_group_id: String },
    ExactArtifactLeakage { fold_id: String, digest: String },
    WrongProtocolSchema,
    EmptyEvaluatorSchema,
    InvalidProtocolDigest { field: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusQualificationReportV1 {
    pub schema: String,
    pub corpus_id: String,
    pub release_id: String,
    pub track_id: String,
    pub stage: CorpusQualificationStageV1,
    pub registration_sha256: String,
    pub sample_manifest_sha256: Option<String>,
    pub protocol_sha256: Option<String>,
    pub sample_count: usize,
    pub fold_count: usize,
    pub composer_group_count: usize,
    pub work_group_count: usize,
    pub issues: Vec<CorpusQualificationIssueV1>,
    pub next_stage_requirements: Vec<String>,
    pub report_sha256: String,
}

#[derive(Serialize)]
struct QualificationReportCommitmentV1<'a> {
    schema: &'a str,
    corpus_id: &'a str,
    release_id: &'a str,
    track_id: &'a str,
    stage: CorpusQualificationStageV1,
    registration_sha256: &'a str,
    sample_manifest_sha256: &'a Option<String>,
    protocol_sha256: &'a Option<String>,
    sample_count: usize,
    fold_count: usize,
    composer_group_count: usize,
    work_group_count: usize,
    issues: &'a [CorpusQualificationIssueV1],
    next_stage_requirements: &'a [String],
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_https_uri(value: &str) -> bool {
    value.starts_with("https://") && value.len() > "https://".len()
}

pub fn validate_corpus_registration(
    registration: &CorpusRegistrationV1,
) -> Vec<CorpusQualificationIssueV1> {
    let mut issues = Vec::new();
    if registration.schema != MUSICXML_CORPUS_REGISTRATION_SCHEMA_V1 {
        issues.push(CorpusQualificationIssueV1::WrongRegistrationSchema);
    }
    if registration.corpus_id.trim().is_empty() {
        issues.push(CorpusQualificationIssueV1::EmptyCorpusId);
    }
    if registration.release_id.trim().is_empty() {
        issues.push(CorpusQualificationIssueV1::EmptyReleaseId);
    }
    if !is_https_uri(&registration.source_uri) {
        issues.push(CorpusQualificationIssueV1::InvalidSourceUri);
    }
    if !is_https_uri(&registration.rights_evidence_uri) {
        issues.push(CorpusQualificationIssueV1::InvalidRightsEvidenceUri);
    }
    if registration.rights_notes.trim().is_empty() {
        issues.push(CorpusQualificationIssueV1::MissingRightsNotes);
    }
    if registration.use_scope == CorpusUseScopeV1::Unverified {
        issues.push(CorpusQualificationIssueV1::UnknownRightsScope);
    }

    match (registration.license, registration.use_scope) {
        (CorpusLicenseV1::Cc0_1_0, CorpusUseScopeV1::CommercialAndNonCommercialResearch)
        | (CorpusLicenseV1::Cc0_1_0, CorpusUseScopeV1::NonCommercialResearchOnly)
        | (CorpusLicenseV1::Cc0_1_0, CorpusUseScopeV1::ExternalEvaluationOnly)
        | (CorpusLicenseV1::CcByNcSa_4_0, CorpusUseScopeV1::NonCommercialResearchOnly)
        | (CorpusLicenseV1::CcByNcSa_4_0, CorpusUseScopeV1::ExternalEvaluationOnly)
        | (CorpusLicenseV1::ExternalEvaluationTerms, CorpusUseScopeV1::ExternalEvaluationOnly)
        | (CorpusLicenseV1::Unknown, CorpusUseScopeV1::ExternalEvaluationOnly) => {}
        (CorpusLicenseV1::CcByNcSa_4_0, CorpusUseScopeV1::CommercialAndNonCommercialResearch) => {
            issues.push(CorpusQualificationIssueV1::CommercialUseNotPermitted);
        }
        (CorpusLicenseV1::ExternalEvaluationTerms, _) => {
            issues.push(CorpusQualificationIssueV1::ExternalEvaluationPolicyMissing);
        }
        (CorpusLicenseV1::Unknown, _) => {
            issues.push(CorpusQualificationIssueV1::NonCommercialUseNotEstablished);
        }
        _ => issues.push(CorpusQualificationIssueV1::IncompatibleLicenseAndUseScope),
    }
    if registration.rights_evidence_status == RightsEvidenceStatusV1::Unknown {
        issues.push(CorpusQualificationIssueV1::NonCommercialUseNotEstablished);
    }

    if registration.tracks.is_empty() {
        issues.push(CorpusQualificationIssueV1::EmptyTrackRegistry);
    }
    let mut seen_tracks = BTreeSet::new();
    for track in &registration.tracks {
        if track.track_id.trim().is_empty() {
            issues.push(CorpusQualificationIssueV1::EmptyTrackId);
        } else if !seen_tracks.insert(track.track_id.clone()) {
            issues.push(CorpusQualificationIssueV1::DuplicateTrackId {
                track_id: track.track_id.clone(),
            });
        }
        if track.input_modalities.is_empty() {
            issues.push(CorpusQualificationIssueV1::EmptyInputModalities {
                track_id: track.track_id.clone(),
            });
        }
        if track.reference_modalities.is_empty() {
            issues.push(CorpusQualificationIssueV1::EmptyReferenceModalities {
                track_id: track.track_id.clone(),
            });
        }
        if registration.use_scope == CorpusUseScopeV1::ExternalEvaluationOnly
            && (track.training_allowed || track.redistribution_allowed)
        {
            issues.push(CorpusQualificationIssueV1::InvalidTrainingAndRedistributionPolicy {
                track_id: track.track_id.clone(),
            });
        }
        if registration.use_scope == CorpusUseScopeV1::NonCommercialResearchOnly
            && track.redistribution_allowed
        {
            issues.push(CorpusQualificationIssueV1::InvalidTrainingAndRedistributionPolicy {
                track_id: track.track_id.clone(),
            });
        }
    }
    issues
}

pub fn validate_corpus_sample_manifest(
    registration: &CorpusRegistrationV1,
    track: &CorpusTrackRegistrationV1,
    manifest: &CorpusSampleManifestV1,
) -> Vec<CorpusQualificationIssueV1> {
    let mut issues = Vec::new();
    if manifest.schema != MUSICXML_CORPUS_SAMPLE_MANIFEST_SCHEMA_V1 {
        issues.push(CorpusQualificationIssueV1::WrongSampleManifestSchema);
    }
    if manifest.corpus_id != registration.corpus_id {
        issues.push(CorpusQualificationIssueV1::SampleCorpusMismatch);
    }
    if manifest.release_id != registration.release_id {
        issues.push(CorpusQualificationIssueV1::SampleReleaseMismatch);
    }
    if manifest.track_id != track.track_id {
        issues.push(CorpusQualificationIssueV1::SampleTrackMismatch);
    }
    if manifest.samples.is_empty() {
        issues.push(CorpusQualificationIssueV1::EmptySampleManifest);
    }

    let mut sample_ids = BTreeSet::new();
    let mut composer_splits: BTreeMap<(String, String), CorpusSplitRoleV1> = BTreeMap::new();
    let mut work_splits: BTreeMap<(String, String), CorpusSplitRoleV1> = BTreeMap::new();
    let mut artifact_splits: BTreeMap<(String, String), CorpusSplitRoleV1> = BTreeMap::new();

    for sample in &manifest.samples {
        if sample.sample_id.trim().is_empty() {
            issues.push(CorpusQualificationIssueV1::EmptySampleId);
        }
        if !sample_ids.insert(sample.sample_id.clone()) {
            issues.push(CorpusQualificationIssueV1::DuplicateSampleId {
                sample_id: sample.sample_id.clone(),
            });
        }
        if sample.fold_id.trim().is_empty() {
            issues.push(CorpusQualificationIssueV1::EmptyFoldId {
                sample_id: sample.sample_id.clone(),
            });
        }
        if !is_sha256(&sample.input_sha256) {
            issues.push(CorpusQualificationIssueV1::InvalidInputDigest {
                sample_id: sample.sample_id.clone(),
            });
        }
        if !is_sha256(&sample.reference_sha256) {
            issues.push(CorpusQualificationIssueV1::InvalidReferenceDigest {
                sample_id: sample.sample_id.clone(),
            });
        }

        if track.split_policy.composer_disjoint {
            match sample.composer_group_id.as_deref().filter(|value| !value.trim().is_empty()) {
                Some(composer) => {
                    let key = (sample.fold_id.clone(), composer.to_string());
                    if composer_splits
                        .insert(key.clone(), sample.split)
                        .is_some_and(|previous| previous != sample.split)
                    {
                        issues.push(CorpusQualificationIssueV1::ComposerLeakage {
                            fold_id: key.0,
                            composer_group_id: key.1,
                        });
                    }
                }
                None => issues.push(CorpusQualificationIssueV1::MissingComposerGroup {
                    sample_id: sample.sample_id.clone(),
                }),
            }
        }
        if track.split_policy.work_disjoint {
            match sample.work_group_id.as_deref().filter(|value| !value.trim().is_empty()) {
                Some(work) => {
                    let key = (sample.fold_id.clone(), work.to_string());
                    if work_splits
                        .insert(key.clone(), sample.split)
                        .is_some_and(|previous| previous != sample.split)
                    {
                        issues.push(CorpusQualificationIssueV1::WorkLeakage {
                            fold_id: key.0,
                            work_group_id: key.1,
                        });
                    }
                }
                None => issues.push(CorpusQualificationIssueV1::MissingWorkGroup {
                    sample_id: sample.sample_id.clone(),
                }),
            }
        }
        if track.split_policy.exact_artifact_disjoint {
            for digest in [&sample.input_sha256, &sample.reference_sha256] {
                if !is_sha256(digest) {
                    continue;
                }
                let key = (sample.fold_id.clone(), digest.clone());
                if artifact_splits
                    .insert(key.clone(), sample.split)
                    .is_some_and(|previous| previous != sample.split)
                {
                    issues.push(CorpusQualificationIssueV1::ExactArtifactLeakage {
                        fold_id: key.0,
                        digest: key.1,
                    });
                }
            }
        }
    }
    issues
}

pub fn validate_corpus_protocol(
    protocol: &CorpusProtocolFingerprintV1,
) -> Vec<CorpusQualificationIssueV1> {
    let mut issues = Vec::new();
    if protocol.schema != MUSICXML_CORPUS_PROTOCOL_SCHEMA_V1 {
        issues.push(CorpusQualificationIssueV1::WrongProtocolSchema);
    }
    if protocol.evaluator_schema.trim().is_empty() {
        issues.push(CorpusQualificationIssueV1::EmptyEvaluatorSchema);
    }
    for (field, digest) in [
        ("split_protocol_sha256", &protocol.split_protocol_sha256),
        ("annotation_protocol_sha256", &protocol.annotation_protocol_sha256),
        ("conversion_pipeline_sha256", &protocol.conversion_pipeline_sha256),
        ("parser_build_sha256", &protocol.parser_build_sha256),
        ("evaluator_build_sha256", &protocol.evaluator_build_sha256),
    ] {
        if !is_sha256(digest) {
            issues.push(CorpusQualificationIssueV1::InvalidProtocolDigest {
                field: field.into(),
            });
        }
    }
    issues
}

/// Determine the strongest stage supported by declared metadata and receipts.
///
/// V1 deliberately stops at ProtocolFrozen. It does not read source files,
/// verify their bytes, authenticate annotations, or provide legal review. A
/// SHA-256 string supplied by a manifest is only a digest declaration.
pub fn qualify_corpus_track(
    registration: &CorpusRegistrationV1,
    manifest: Option<&CorpusSampleManifestV1>,
    protocol: Option<&CorpusProtocolFingerprintV1>,
) -> CorpusQualificationReportV1 {
    let registration_issues = validate_corpus_registration(registration);
    let track = manifest.and_then(|manifest| {
        registration
            .tracks
            .iter()
            .find(|track| track.track_id == manifest.track_id)
    });
    let mut sample_issues = Vec::new();
    if let (Some(manifest), Some(track)) = (manifest, track) {
        sample_issues.extend(validate_corpus_sample_manifest(registration, track, manifest));
    }
    if manifest.is_some() && track.is_none() {
        sample_issues.push(CorpusQualificationIssueV1::SampleTrackMismatch);
    }
    let protocol_issues = protocol.map(validate_corpus_protocol).unwrap_or_default();
    let mut issues = registration_issues.clone();
    issues.extend(sample_issues.clone());
    issues.extend(protocol_issues.clone());

    let registration_sha256 = canonical_json_sha256(registration).unwrap_or_default();
    let sample_manifest_sha256 = manifest.and_then(|value| canonical_json_sha256(value).ok());
    let protocol_sha256 = protocol.and_then(|value| canonical_json_sha256(value).ok());

    // A declared manifest is not yet split-qualified. Keep the stage useful
    // even when its hashes parse but leakage is found: callers can inspect the
    // exact blocker instead of seeing an uninformative Candidate downgrade.
    let mut stage = if !registration_issues.is_empty() {
        CorpusQualificationStageV1::Candidate
    } else if manifest.is_none() {
        CorpusQualificationStageV1::MetadataRecorded
    } else if !sample_issues.is_empty() {
        CorpusQualificationStageV1::SampleDigestsDeclared
    } else {
        CorpusQualificationStageV1::SplitQualified
    };
    if protocol.is_some()
        && issues.is_empty()
        && sample_manifest_sha256.is_some()
    {
        stage = CorpusQualificationStageV1::ProtocolFrozen;
    }

    let mut next_stage_requirements = Vec::new();
    if !registration_issues.is_empty() {
        next_stage_requirements.push("Resolve registration and rights-scope issues; retain source evidence and obtain any needed legal review".into());
    }
    if manifest.is_none() {
        next_stage_requirements.push("Declare sample IDs, fold assignments and input/reference SHA-256 digests".into());
    } else if !sample_issues.is_empty() {
        next_stage_requirements.push("Resolve sample identity, checksum syntax and work/composer/artifact split leakage".into());
    }
    if protocol.is_none() || !protocol_issues.is_empty() {
        next_stage_requirements.push("Freeze split, annotation, conversion, parser and evaluator fingerprints".into());
    }
    if stage == CorpusQualificationStageV1::ProtocolFrozen {
        next_stage_requirements.push("Run a separate byte verifier and attach its output; V1 does not claim byte verification".into());
        next_stage_requirements.push("Obtain independent annotation adjudication and any required legal review before public qualification".into());
    }

    let mut report = CorpusQualificationReportV1 {
        schema: MUSICXML_CORPUS_QUALIFICATION_REPORT_SCHEMA_V1.into(),
        corpus_id: registration.corpus_id.clone(),
        release_id: registration.release_id.clone(),
        track_id: manifest.map(|value| value.track_id.clone()).unwrap_or_default(),
        stage,
        registration_sha256,
        sample_manifest_sha256,
        protocol_sha256,
        sample_count: manifest.map_or(0, |value| value.samples.len()),
        fold_count: manifest.map_or(0, |value| value.samples.iter().map(|sample| sample.fold_id.as_str()).collect::<BTreeSet<_>>().len()),
        composer_group_count: manifest.map_or(0, |value| value.samples.iter().filter_map(|sample| sample.composer_group_id.as_deref()).collect::<BTreeSet<_>>().len()),
        work_group_count: manifest.map_or(0, |value| value.samples.iter().filter_map(|sample| sample.work_group_id.as_deref()).collect::<BTreeSet<_>>().len()),
        issues,
        next_stage_requirements,
        report_sha256: String::new(),
    };
    report.report_sha256 = qualification_report_commitment(&report).unwrap_or_default();
    report
}

pub fn qualification_report_commitment(
    report: &CorpusQualificationReportV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&QualificationReportCommitmentV1 {
        schema: &report.schema,
        corpus_id: &report.corpus_id,
        release_id: &report.release_id,
        track_id: &report.track_id,
        stage: report.stage,
        registration_sha256: &report.registration_sha256,
        sample_manifest_sha256: &report.sample_manifest_sha256,
        protocol_sha256: &report.protocol_sha256,
        sample_count: report.sample_count,
        fold_count: report.fold_count,
        composer_group_count: report.composer_group_count,
        work_group_count: report.work_group_count,
        issues: &report.issues,
        next_stage_requirements: &report.next_stage_requirements,
    })
}

/// Metadata-only candidate registry. No sample bytes, SHA-256 manifests or
/// frozen conversion/evaluation protocols are embedded here, so these entries
/// cannot be promoted beyond MetadataRecorded until independent manifests
/// and fingerprints are supplied.
pub fn candidate_musicxml_corpus_registry_v1() -> Vec<CorpusRegistrationV1> {
    let split = |work_disjoint, composer_disjoint| CorpusSplitPolicyV1 {
        work_disjoint,
        composer_disjoint,
        exact_artifact_disjoint: true,
    };
    vec![
        CorpusRegistrationV1 {
            schema: MUSICXML_CORPUS_REGISTRATION_SCHEMA_V1.into(),
            corpus_id: "ossq-omr".into(),
            release_id: "ismir-2026-ossq-omr".into(),
            source_uri: "https://github.com/MALerLab/ossq-omr".into(),
            license: CorpusLicenseV1::Cc0_1_0,
            use_scope: CorpusUseScopeV1::CommercialAndNonCommercialResearch,
            rights_evidence_status: RightsEvidenceStatusV1::SourceDeclared,
            rights_evidence_uri: "https://github.com/MALerLab/ossq-omr".into(),
            rights_notes: "The dataset maintainer states CC0-1.0 for source annotations and derived formats. Preserve requested attribution/credit. Verify exact release and all acquired artifacts before use.".into(),
            tracks: vec![
                CorpusTrackRegistrationV1 {
                    track_id: "omr-synthetic-score-image".into(),
                    task: CorpusTaskV1::OpticalMusicRecognition,
                    input_modalities: vec![CorpusModalityV1::ScoreImage],
                    reference_modalities: vec![CorpusModalityV1::MusicXml],
                    split_policy: split(true, false),
                    external_protocol_label: Some("OSSQ-OMR synthetic; official score-level split".into()),
                    training_allowed: true,
                    redistribution_allowed: true,
                },
                CorpusTrackRegistrationV1 {
                    track_id: "omr-scanned-score-image".into(),
                    task: CorpusTaskV1::OpticalMusicRecognition,
                    input_modalities: vec![CorpusModalityV1::ScoreImage],
                    reference_modalities: vec![CorpusModalityV1::MusicXml],
                    split_policy: split(true, false),
                    external_protocol_label: Some("OSSQ-OMR scanned; official score-level split".into()),
                    training_allowed: true,
                    redistribution_allowed: true,
                },
                CorpusTrackRegistrationV1 {
                    track_id: "omr-composer-heldout".into(),
                    task: CorpusTaskV1::OpticalMusicRecognition,
                    input_modalities: vec![CorpusModalityV1::ScoreImage],
                    reference_modalities: vec![CorpusModalityV1::MusicXml],
                    split_policy: split(true, true),
                    external_protocol_label: Some("MUSE proposed composer-disjoint OOD track; not an official OSSQ split".into()),
                    training_allowed: true,
                    redistribution_allowed: true,
                },
            ],
        },
        CorpusRegistrationV1 {
            schema: MUSICXML_CORPUS_REGISTRATION_SCHEMA_V1.into(),
            corpus_id: "asap".into(),
            release_id: "v1.1".into(),
            source_uri: "https://github.com/fosfrancesco/asap-dataset".into(),
            license: CorpusLicenseV1::CcByNcSa_4_0,
            use_scope: CorpusUseScopeV1::NonCommercialResearchOnly,
            rights_evidence_status: RightsEvidenceStatusV1::SourceDeclared,
            rights_evidence_uri: "https://github.com/fosfrancesco/asap-dataset".into(),
            rights_notes: "CC BY-NC-SA 4.0. Noncommercial research only; attribution and share-alike obligations apply. Maintainers report 31 performances that are not score-aligned and notation/beat-position limits around embellishments, pickups and rubato; preserve those statuses.".into(),
            tracks: vec![CorpusTrackRegistrationV1 {
                track_id: "score-performance-alignment".into(),
                task: CorpusTaskV1::ScorePerformanceAlignment,
                input_modalities: vec![CorpusModalityV1::Audio, CorpusModalityV1::Midi],
                reference_modalities: vec![CorpusModalityV1::MusicXml, CorpusModalityV1::Midi],
                split_policy: split(true, false),
                external_protocol_label: Some("ASAP v1.1; aligned audio/MIDI subset only where declared".into()),
                training_allowed: true,
                redistribution_allowed: false,
            }],
        },
        CorpusRegistrationV1 {
            schema: MUSICXML_CORPUS_REGISTRATION_SCHEMA_V1.into(),
            corpus_id: "maestro".into(),
            release_id: "v3.0.0".into(),
            source_uri: "https://magenta.withgoogle.com/datasets/maestro".into(),
            license: CorpusLicenseV1::CcByNcSa_4_0,
            use_scope: CorpusUseScopeV1::NonCommercialResearchOnly,
            rights_evidence_status: RightsEvidenceStatusV1::SourceDeclared,
            rights_evidence_uri: "https://magenta.withgoogle.com/datasets/maestro".into(),
            rights_notes: "CC BY-NC-SA 4.0. Piano audio/MIDI event corpus, not a multi-part source MusicXML score corpus. Preserve attribution/share-alike obligations; do not use in a commercial-reuse training pool.".into(),
            tracks: vec![CorpusTrackRegistrationV1 {
                track_id: "piano-audio-midi-performance".into(),
                task: CorpusTaskV1::AudioToMidi,
                input_modalities: vec![CorpusModalityV1::Audio],
                reference_modalities: vec![CorpusModalityV1::Midi],
                split_policy: split(true, false),
                external_protocol_label: Some("MAESTRO v3.0.0 official train/validation/test splits".into()),
                training_allowed: true,
                redistribution_allowed: false,
            }],
        },
        CorpusRegistrationV1 {
            schema: MUSICXML_CORPUS_REGISTRATION_SCHEMA_V1.into(),
            corpus_id: "mirex-2026-quartets".into(),
            release_id: "mirex-2026-audio-to-score".into(),
            source_uri: "https://music-ir.org/mirex/wiki/2026:Audio-to-Score_Transcription".into(),
            license: CorpusLicenseV1::Unknown,
            use_scope: CorpusUseScopeV1::ExternalEvaluationOnly,
            rights_evidence_status: RightsEvidenceStatusV1::SourceDeclared,
            rights_evidence_uri: "https://music-ir.org/mirex/wiki/2026:Audio-to-Score_Transcription".into(),
            rights_notes: "External benchmark/task terms only. Corpus asset license is not asserted. Do not redistribute or train/tune on evaluation excerpts; preserve Blind and Staves-Informed tracks and Quartets versus held-out QuartetsMIREX.".into(),
            tracks: vec![
                CorpusTrackRegistrationV1 {
                    track_id: "blind-audio-to-score".into(),
                    task: CorpusTaskV1::ExternalLeaderboard,
                    input_modalities: vec![CorpusModalityV1::Audio],
                    reference_modalities: vec![CorpusModalityV1::Kern],
                    split_policy: split(true, true),
                    external_protocol_label: Some("MIREX 2026 Blind A2S; Quartets and QuartetsMIREX reported separately".into()),
                    training_allowed: false,
                    redistribution_allowed: false,
                },
                CorpusTrackRegistrationV1 {
                    track_id: "staves-informed-audio-to-score".into(),
                    task: CorpusTaskV1::ExternalLeaderboard,
                    input_modalities: vec![CorpusModalityV1::Audio, CorpusModalityV1::ScoreImage],
                    reference_modalities: vec![CorpusModalityV1::Kern],
                    split_policy: split(true, true),
                    external_protocol_label: Some("MIREX 2026 Staves-Informed A2S; report separately from Blind".into()),
                    training_allowed: false,
                    redistribution_allowed: false,
                },
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registration(id: &str) -> CorpusRegistrationV1 {
        candidate_musicxml_corpus_registry_v1()
            .into_iter()
            .find(|corpus| corpus.corpus_id == id)
            .unwrap()
    }

    fn sample(
        id: &str,
        fold: &str,
        split: CorpusSplitRoleV1,
        composer: &str,
        work: &str,
        input_digest: char,
        reference_digest: char,
    ) -> CorpusSampleV1 {
        CorpusSampleV1 {
            sample_id: id.into(),
            fold_id: fold.into(),
            split,
            composer_group_id: Some(composer.into()),
            work_group_id: Some(work.into()),
            input_sha256: input_digest.to_string().repeat(64),
            reference_sha256: reference_digest.to_string().repeat(64),
        }
    }

    fn sample_manifest(
        registration: &CorpusRegistrationV1,
        track_id: &str,
        samples: Vec<CorpusSampleV1>,
    ) -> CorpusSampleManifestV1 {
        CorpusSampleManifestV1 {
            schema: MUSICXML_CORPUS_SAMPLE_MANIFEST_SCHEMA_V1.into(),
            corpus_id: registration.corpus_id.clone(),
            release_id: registration.release_id.clone(),
            track_id: track_id.into(),
            samples,
        }
    }

    fn protocol() -> CorpusProtocolFingerprintV1 {
        CorpusProtocolFingerprintV1 {
            schema: MUSICXML_CORPUS_PROTOCOL_SCHEMA_V1.into(),
            split_protocol_sha256: "1".repeat(64),
            annotation_protocol_sha256: "2".repeat(64),
            conversion_pipeline_sha256: "3".repeat(64),
            parser_build_sha256: "4".repeat(64),
            evaluator_build_sha256: "5".repeat(64),
            evaluator_schema: "muse-musicxml-notation-evaluation-v1".into(),
        }
    }

    #[test]
    fn candidate_registry_records_four_distinct_usage_scopes() {
        let registry = candidate_musicxml_corpus_registry_v1();
        assert_eq!(registry.len(), 4);
        assert!(registry.iter().all(|entry| validate_corpus_registration(entry).is_empty()));
        let ossq = registry.iter().find(|entry| entry.corpus_id == "ossq-omr").unwrap();
        assert_eq!(ossq.license, CorpusLicenseV1::Cc0_1_0);
        assert_eq!(
            ossq.use_scope,
            CorpusUseScopeV1::CommercialAndNonCommercialResearch
        );
        let asap = registry.iter().find(|entry| entry.corpus_id == "asap").unwrap();
        let maestro = registry.iter().find(|entry| entry.corpus_id == "maestro").unwrap();
        assert_eq!(asap.license, CorpusLicenseV1::CcByNcSa_4_0);
        assert_eq!(maestro.license, CorpusLicenseV1::CcByNcSa_4_0);
        assert_eq!(asap.use_scope, CorpusUseScopeV1::NonCommercialResearchOnly);
        assert_eq!(maestro.use_scope, CorpusUseScopeV1::NonCommercialResearchOnly);
        let mirex = registry.iter().find(|entry| entry.corpus_id == "mirex-2026-quartets").unwrap();
        assert_eq!(mirex.license, CorpusLicenseV1::Unknown);
        assert_eq!(mirex.use_scope, CorpusUseScopeV1::ExternalEvaluationOnly);
        assert!(mirex.tracks.iter().all(|track| !track.training_allowed && !track.redistribution_allowed));
    }

    #[test]
    fn noncommercial_license_cannot_be_registered_for_commercial_reuse() {
        let mut registration = registration("asap");
        registration.use_scope = CorpusUseScopeV1::CommercialAndNonCommercialResearch;
        assert!(validate_corpus_registration(&registration).contains(
            &CorpusQualificationIssueV1::CommercialUseNotPermitted
        ));
    }

    #[test]
    fn unknown_license_is_not_treated_as_unrestricted() {
        let mut registration = registration("mirex-2026-quartets");
        registration.use_scope = CorpusUseScopeV1::CommercialAndNonCommercialResearch;
        assert!(validate_corpus_registration(&registration).contains(
            &CorpusQualificationIssueV1::NonCommercialUseNotEstablished
        ));
    }

    #[test]
    fn metadata_only_registration_does_not_claim_sample_or_byte_verification() {
        let registration = registration("ossq-omr");
        let report = qualify_corpus_track(&registration, None, None);
        assert_eq!(report.stage, CorpusQualificationStageV1::MetadataRecorded);
        assert_eq!(report.sample_count, 0);
        assert!(report.sample_manifest_sha256.is_none());
        assert!(report.next_stage_requirements.iter().any(|item| item.contains("sample IDs")));
        assert!(report.next_stage_requirements.iter().any(|item| item.contains("fingerprints")));
        assert!(report.report_sha256.len() == 64);
    }

    #[test]
    fn composer_disjoint_track_rejects_same_composer_in_train_and_test() {
        let registration = registration("ossq-omr");
        let manifest = sample_manifest(
            &registration,
            "omr-composer-heldout",
            vec![
                sample("train-a", "fold-0", CorpusSplitRoleV1::Train, "composer-a", "work-a", 'a', 'b'),
                sample("test-b", "fold-0", CorpusSplitRoleV1::Test, "composer-a", "work-b", 'c', 'd'),
            ],
        );
        let track = registration.tracks.iter().find(|track| track.track_id == manifest.track_id).unwrap();
        let issues = validate_corpus_sample_manifest(&registration, track, &manifest);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            CorpusQualificationIssueV1::ComposerLeakage { composer_group_id, .. }
                if composer_group_id == "composer-a"
        )));
    }

    #[test]
    fn work_disjoint_track_rejects_same_score_in_two_splits() {
        let registration = registration("ossq-omr");
        let manifest = sample_manifest(
            &registration,
            "omr-synthetic-score-image",
            vec![
                sample("train-a", "fold-0", CorpusSplitRoleV1::Train, "composer-a", "work-a", 'a', 'b'),
                sample("test-a", "fold-0", CorpusSplitRoleV1::Test, "composer-b", "work-a", 'c', 'd'),
            ],
        );
        let track = registration.tracks.iter().find(|track| track.track_id == manifest.track_id).unwrap();
        let issues = validate_corpus_sample_manifest(&registration, track, &manifest);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            CorpusQualificationIssueV1::WorkLeakage { work_group_id, .. }
                if work_group_id == "work-a"
        )));
    }

    #[test]
    fn exact_input_or_reference_bytes_cannot_cross_splits() {
        let registration = registration("ossq-omr");
        let manifest = sample_manifest(
            &registration,
            "omr-synthetic-score-image",
            vec![
                sample("train-a", "fold-0", CorpusSplitRoleV1::Train, "composer-a", "work-a", 'a', 'b'),
                sample("test-b", "fold-0", CorpusSplitRoleV1::Test, "composer-b", "work-b", 'c', 'b'),
            ],
        );
        let track = registration.tracks.iter().find(|track| track.track_id == manifest.track_id).unwrap();
        let issues = validate_corpus_sample_manifest(&registration, track, &manifest);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            CorpusQualificationIssueV1::ExactArtifactLeakage { digest, .. }
                if digest == &"b".repeat(64)
        )));
    }

    #[test]
    fn protocol_frozen_requires_clean_manifest_and_all_protocol_fingerprints() {
        let registration = registration("ossq-omr");
        let track_id = "omr-composer-heldout";
        let manifest = sample_manifest(
            &registration,
            track_id,
            vec![
                sample("train-a", "fold-0", CorpusSplitRoleV1::Train, "composer-a", "work-a", 'a', 'b'),
                sample("test-b", "fold-0", CorpusSplitRoleV1::Test, "composer-b", "work-b", 'c', 'd'),
            ],
        );
        let report = qualify_corpus_track(&registration, Some(&manifest), Some(&protocol()));
        assert_eq!(report.stage, CorpusQualificationStageV1::ProtocolFrozen);
        assert!(report.issues.is_empty());
        assert_eq!(report.sample_count, 2);
        assert_eq!(report.fold_count, 1);
        assert_eq!(report.composer_group_count, 2);
        assert_eq!(report.work_group_count, 2);
        assert!(report.next_stage_requirements.iter().any(|item| item.contains("byte verifier")));
    }

    #[test]
    fn leakage_keeps_declared_manifest_below_split_qualified() {
        let registration = registration("ossq-omr");
        let manifest = sample_manifest(
            &registration,
            "omr-composer-heldout",
            vec![
                sample("train-a", "fold-0", CorpusSplitRoleV1::Train, "composer-a", "work-a", 'a', 'b'),
                sample("test-b", "fold-0", CorpusSplitRoleV1::Test, "composer-a", "work-b", 'c', 'd'),
            ],
        );
        let report = qualify_corpus_track(&registration, Some(&manifest), Some(&protocol()));
        assert_eq!(report.stage, CorpusQualificationStageV1::SampleDigestsDeclared);
        assert!(!report.issues.is_empty());
        assert_ne!(report.stage, CorpusQualificationStageV1::ProtocolFrozen);
    }

    #[test]
    fn malformed_digest_duplicate_id_and_wrong_track_fail_closed() {
        let registration = registration("ossq-omr");
        let mut first = sample("same", "fold-0", CorpusSplitRoleV1::Train, "composer-a", "work-a", 'a', 'b');
        first.input_sha256 = "not-a-sha".into();
        let second = sample("same", "fold-0", CorpusSplitRoleV1::Train, "composer-a", "work-b", 'c', 'd');
        let mut manifest = sample_manifest(&registration, "missing-track", vec![first, second]);
        let issues = validate_corpus_sample_manifest(
            &registration,
            &registration.tracks[0],
            &manifest,
        );
        assert!(issues.contains(&CorpusQualificationIssueV1::SampleTrackMismatch));
        assert!(issues.contains(&CorpusQualificationIssueV1::DuplicateSampleId { sample_id: "same".into() }));
        assert!(issues.contains(&CorpusQualificationIssueV1::InvalidInputDigest { sample_id: "same".into() }));
        manifest.samples.clear();
        assert!(validate_corpus_sample_manifest(&registration, &registration.tracks[0], &manifest)
            .contains(&CorpusQualificationIssueV1::EmptySampleManifest));
    }

    #[test]
    fn report_commitment_changes_when_stage_or_issue_changes() {
        let registration = registration("ossq-omr");
        let mut report = qualify_corpus_track(&registration, None, None);
        let digest = report.report_sha256.clone();
        assert_eq!(qualification_report_commitment(&report).unwrap(), digest);
        report.stage = CorpusQualificationStageV1::ProtocolFrozen;
        assert_ne!(qualification_report_commitment(&report).unwrap(), digest);
    }

    #[test]
    fn metadata_candidate_tracks_are_not_mislabelled_as_audio_to_score() {
        let registry = candidate_musicxml_corpus_registry_v1();
        let ossq = registry.iter().find(|entry| entry.corpus_id == "ossq-omr").unwrap();
        assert!(ossq.tracks.iter().all(|track| track.task == CorpusTaskV1::OpticalMusicRecognition));
        let maestro = registry.iter().find(|entry| entry.corpus_id == "maestro").unwrap();
        assert!(maestro.tracks.iter().all(|track| track.task == CorpusTaskV1::AudioToMidi));
        let mirex = registry.iter().find(|entry| entry.corpus_id == "mirex-2026-quartets").unwrap();
        assert_eq!(mirex.tracks.len(), 2);
        assert_ne!(
            mirex.tracks[0].external_protocol_label,
            mirex.tracks[1].external_protocol_label
        );
    }
}
