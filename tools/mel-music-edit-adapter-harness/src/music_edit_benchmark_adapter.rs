// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dataset/track adapter contracts for external music-editing benchmarks.
//!
//! This module validates task manifests and comparison scopes. It does not
//! download benchmark data, execute editors, score audio, or grant dataset
//! rights. Every example binds a source-lineage identity so paired edits from
//! one source composition cannot leak across splits.
//!
//! All twelve SMEBench operation families are represented explicitly. A
//! NotYetQualified status means a capability has not been demonstrated under
//! the exact operation/scope/renderer contract; it is not a product claim that
//! the operation is impossible. Unsupported and unqualified tasks remain in
//! the attempted denominator as abstentions/failures in downstream run
//! manifests. They must never be silently replaced with identity output.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MUSIC_EDIT_DATASET_MANIFEST_SCHEMA_V1: &str =
    "mel-music-edit-dataset-manifest-v1";
pub const MUSIC_EDIT_COMPARISON_SCOPE_SCHEMA_V1: &str =
    "mel-music-edit-comparison-scope-v1";

/// The exact twelve controlled SMEBench operations. The benchmark operation
/// identifier is kept independent of any local implementation operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditOperationV1 {
    RemoveInstrument,
    ReplaceInstrument,
    ExtractInstrument,
    DoubleInstrument,
    RemoveDrums,
    ReplaceDrums,
    ExtractDrums,
    Staccato,
    Legato,
    Crescendo,
    Decrescendo,
    TuningCorrection440,
}

impl MusicEditOperationV1 {
    pub const ALL: [Self; 12] = [
        Self::RemoveInstrument,
        Self::ReplaceInstrument,
        Self::ExtractInstrument,
        Self::DoubleInstrument,
        Self::RemoveDrums,
        Self::ReplaceDrums,
        Self::ExtractDrums,
        Self::Staccato,
        Self::Legato,
        Self::Crescendo,
        Self::Decrescendo,
        Self::TuningCorrection440,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditTargetScopeV1 {
    Instrument,
    DrumKit,
    Mixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditDatasetStratumV1 {
    MidiSynthesized,
    RealMultitrack,
    MmaeMusicSubset,
    MmaeFullMultimodal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditSplitV1 {
    Train,
    Development,
    Test,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditAccessTierV1 {
    Public,
    OrganizerRestricted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditRightsStateV1 {
    Cleared,
    Restricted,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditPermittedUseV1 {
    EvaluationOnly,
    Training,
    Redistribution,
    PublicRelease,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditRightsV1 {
    pub state: MusicEditRightsStateV1,
    /// License URL, permission record, or rights-owner statement reference.
    pub rights_reference: String,
    /// Digest of the permission/license evidence where required.
    pub rights_evidence_sha256: Option<String>,
    pub permitted_uses: Vec<MusicEditPermittedUseV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditDatasetRowV1 {
    pub sample_id: String,
    /// Stable grouping key for the original composition/recording lineage.
    /// Source and paired target versions must share this lineage identity.
    pub source_lineage_sha256: String,
    pub split: MusicEditSplitV1,
    pub stratum: MusicEditDatasetStratumV1,
    pub operation: MusicEditOperationV1,
    pub target_scope: MusicEditTargetScopeV1,
    pub source_audio_sha256: String,
    pub instruction_sha256: String,
    pub target_audio_sha256: String,
    /// Optional exact MIDI/score source artifact; absent for audio-only items.
    pub source_symbolic_sha256: Option<String>,
    /// Commits the target selector (instrument identity, event set, or mixture).
    pub target_selector_sha256: String,
    /// Commits benchmark annotations not represented by the fields above.
    pub metadata_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditDatasetManifestV1 {
    pub schema_version: String,
    pub dataset_id: String,
    /// Immutable release/tag/commit identifier, never latest/current/stable.
    pub dataset_version: String,
    pub benchmark_track_id: String,
    pub protocol_sha256: String,
    pub scorer_sha256: String,
    pub access_tier: MusicEditAccessTierV1,
    /// Required for organizer-restricted data; does not imply registration.
    pub access_evidence_sha256: Option<String>,
    pub intended_use: MusicEditPermittedUseV1,
    pub rights: MusicEditRightsV1,
    pub rows: Vec<MusicEditDatasetRowV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationQualificationV1 {
    Qualified,
    NotYetQualified,
    Unsupported,
    AbstainOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditOperationCapabilityV1 {
    pub operation: MusicEditOperationV1,
    pub qualification: OperationQualificationV1,
    /// A Qualified operation must point to a frozen exact-head qualification
    /// record covering this operation, target scope and renderer profile.
    pub qualification_evidence_sha256: Option<String>,
    /// Human-readable constraint boundary; never used as machine evidence.
    pub boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditCapabilityMatrixV1 {
    pub schema_version: String,
    pub capabilities: Vec<MusicEditOperationCapabilityV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicEditAdapterIssueV1 {
    WrongSchema,
    EmptyField(&'static str),
    MutableDatasetVersion,
    InvalidDigest(&'static str),
    UnresolvedRights,
    MissingRightsEvidence,
    IntendedUseNotPermitted,
    MissingAccessEvidence,
    DuplicatePermittedUse,
    EmptyPermittedUses,
    EmptyRows,
    DuplicateSampleId,
    LineageCrossesSplits,
    InvalidCapabilityCoverage,
    DuplicateCapability,
    MissingCapability(&'static str),
    QualifiedWithoutEvidence,
    NonQualifiedWithEvidence,
    EmptyCapabilityBoundary,
    UnknownOperation,
}

impl std::fmt::Display for MusicEditAdapterIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MusicEditAdapterIssueV1 {}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(
    issues: &mut Vec<MusicEditAdapterIssueV1>,
    field: &'static str,
    value: &str,
) {
    if !is_lower_sha256(value) {
        issues.push(MusicEditAdapterIssueV1::InvalidDigest(field));
    }
}

fn required(
    issues: &mut Vec<MusicEditAdapterIssueV1>,
    field: &'static str,
    value: &str,
) {
    if value.trim().is_empty() {
        issues.push(MusicEditAdapterIssueV1::EmptyField(field));
    }
}

/// Validate a dataset manifest without accessing the referenced audio files.
/// File existence, file digests and native scorer validation are performed by
/// the later materialization/runner stage; this layer validates commitments.
pub fn validate_music_edit_dataset_manifest_v1(
    manifest: &MusicEditDatasetManifestV1,
) -> Vec<MusicEditAdapterIssueV1> {
    let mut issues = Vec::new();
    if manifest.schema_version != MUSIC_EDIT_DATASET_MANIFEST_SCHEMA_V1 {
        issues.push(MusicEditAdapterIssueV1::WrongSchema);
    }
    required(&mut issues, "dataset_id", &manifest.dataset_id);
    required(&mut issues, "dataset_version", &manifest.dataset_version);
    required(&mut issues, "benchmark_track_id", &manifest.benchmark_track_id);
    required(&mut issues, "rights_reference", &manifest.rights.rights_reference);

    if matches!(
        manifest.dataset_version.to_ascii_lowercase().as_str(),
        "latest" | "current" | "stable"
    ) {
        issues.push(MusicEditAdapterIssueV1::MutableDatasetVersion);
    }
    validate_digest(&mut issues, "protocol_sha256", &manifest.protocol_sha256);
    validate_digest(&mut issues, "scorer_sha256", &manifest.scorer_sha256);

    if manifest.rights.state == MusicEditRightsStateV1::Unresolved {
        issues.push(MusicEditAdapterIssueV1::UnresolvedRights);
    }
    if manifest.rights.state != MusicEditRightsStateV1::Unresolved
        && manifest.rights.rights_evidence_sha256.is_none()
    {
        issues.push(MusicEditAdapterIssueV1::MissingRightsEvidence);
    }
    if let Some(digest) = manifest.rights.rights_evidence_sha256.as_deref() {
        validate_digest(&mut issues, "rights_evidence_sha256", digest);
    }
    if manifest.rights.permitted_uses.is_empty() {
        issues.push(MusicEditAdapterIssueV1::EmptyPermittedUses);
    }
    let mut permitted_uses = BTreeSet::new();
    if manifest
        .rights
        .permitted_uses
        .iter()
        .any(|usage| !permitted_uses.insert(*usage))
    {
        issues.push(MusicEditAdapterIssueV1::DuplicatePermittedUse);
    }
    if !manifest.rights.permitted_uses.contains(&manifest.intended_use) {
        issues.push(MusicEditAdapterIssueV1::IntendedUseNotPermitted);
    }
    if manifest.access_tier == MusicEditAccessTierV1::OrganizerRestricted
        && manifest.access_evidence_sha256.is_none()
    {
        issues.push(MusicEditAdapterIssueV1::MissingAccessEvidence);
    }
    if let Some(digest) = manifest.access_evidence_sha256.as_deref() {
        validate_digest(&mut issues, "access_evidence_sha256", digest);
    }

    if manifest.rows.is_empty() {
        issues.push(MusicEditAdapterIssueV1::EmptyRows);
    }
    let mut sample_ids = BTreeSet::new();
    let mut lineage_splits = BTreeMap::<String, MusicEditSplitV1>::new();
    for row in &manifest.rows {
        required(&mut issues, "sample_id", &row.sample_id);
        if !sample_ids.insert(row.sample_id.clone()) {
            issues.push(MusicEditAdapterIssueV1::DuplicateSampleId);
        }
        required(&mut issues, "source_lineage_sha256", &row.source_lineage_sha256);
        validate_digest(
            &mut issues,
            "source_lineage_sha256",
            &row.source_lineage_sha256,
        );
        validate_digest(&mut issues, "source_audio_sha256", &row.source_audio_sha256);
        validate_digest(&mut issues, "instruction_sha256", &row.instruction_sha256);
        validate_digest(&mut issues, "target_audio_sha256", &row.target_audio_sha256);
        validate_digest(&mut issues, "target_selector_sha256", &row.target_selector_sha256);
        validate_digest(&mut issues, "metadata_sha256", &row.metadata_sha256);
        if let Some(digest) = row.source_symbolic_sha256.as_deref() {
            validate_digest(&mut issues, "source_symbolic_sha256", digest);
        }
        match lineage_splits.get(&row.source_lineage_sha256) {
            Some(existing) if *existing != row.split => {
                issues.push(MusicEditAdapterIssueV1::LineageCrossesSplits);
            }
            Some(_) => {}
            None => {
                lineage_splits.insert(row.source_lineage_sha256.clone(), row.split);
            }
        }
    }
    issues
}

/// Require exactly one status for each of the twelve benchmark operations.
/// A claimed Qualified status is not accepted without a pinned qualification
/// evidence digest. Other states remain explicit and scoreable as abstentions.
pub fn validate_music_edit_capability_matrix_v1(
    matrix: &MusicEditCapabilityMatrixV1,
) -> Vec<MusicEditAdapterIssueV1> {
    let mut issues = Vec::new();
    if matrix.schema_version != "mel-music-edit-capability-matrix-v1" {
        issues.push(MusicEditAdapterIssueV1::WrongSchema);
    }
    let mut seen = BTreeSet::new();
    for capability in &matrix.capabilities {
        if !seen.insert(capability.operation) {
            issues.push(MusicEditAdapterIssueV1::DuplicateCapability);
        }
        if capability.boundary.trim().is_empty() {
            issues.push(MusicEditAdapterIssueV1::EmptyCapabilityBoundary);
        }
        match (
            capability.qualification,
            capability.qualification_evidence_sha256.as_deref(),
        ) {
            (OperationQualificationV1::Qualified, None) => {
                issues.push(MusicEditAdapterIssueV1::QualifiedWithoutEvidence);
            }
            (OperationQualificationV1::Qualified, Some(digest)) => {
                validate_digest(&mut issues, "qualification_evidence_sha256", digest);
            }
            (_, Some(_)) => issues.push(MusicEditAdapterIssueV1::NonQualifiedWithEvidence),
            (_, None) => {}
        }
    }
    for operation in MusicEditOperationV1::ALL {
        if !seen.contains(&operation) {
            issues.push(MusicEditAdapterIssueV1::InvalidCapabilityCoverage);
            break;
        }
    }
    if matrix.capabilities.len() != MusicEditOperationV1::ALL.len() {
        issues.push(MusicEditAdapterIssueV1::InvalidCapabilityCoverage);
    }
    issues
}

pub fn qualification_for(
    matrix: &MusicEditCapabilityMatrixV1,
    operation: MusicEditOperationV1,
) -> Option<OperationQualificationV1> {
    matrix
        .capabilities
        .iter()
        .find(|entry| entry.operation == operation)
        .map(|entry| entry.qualification)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditComparisonScopeV1 {
    pub schema_version: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub native_track_id: String,
    pub dataset_stratum: MusicEditDatasetStratumV1,
    pub split_id: String,
    pub protocol_sha256: String,
    pub scorer_sha256: String,
    pub native_metric_ids: Vec<String>,
    pub sample_manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicEditScopeMismatchV1 {
    Schema,
    BenchmarkId,
    BenchmarkVersion,
    Track,
    DatasetStratum,
    Split,
    Protocol,
    Scorer,
    NativeMetricSet,
    SampleManifest,
    EmptyScopeField(&'static str),
    InvalidScopeDigest(&'static str),
    DuplicateNativeMetric,
    MissingNativeMetric,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MusicEditScopeComparisonV1 {
    pub directly_comparable: bool,
    pub mismatches: Vec<MusicEditScopeMismatchV1>,
}

/// Direct comparisons require exact track, split, scorer and sample-manifest
/// identity. Similar metric names or a shared benchmark family are insufficient.
pub fn compare_music_edit_scopes_v1(
    left: &MusicEditComparisonScopeV1,
    right: &MusicEditComparisonScopeV1,
) -> MusicEditScopeComparisonV1 {
    let mut mismatches = Vec::new();
    for scope in [left, right] {
        for (field, value) in [
            ("benchmark_id", scope.benchmark_id.as_str()),
            ("benchmark_version", scope.benchmark_version.as_str()),
            ("native_track_id", scope.native_track_id.as_str()),
            ("split_id", scope.split_id.as_str()),
        ] {
            if value.trim().is_empty() {
                mismatches.push(MusicEditScopeMismatchV1::EmptyScopeField(field));
            }
        }
        for (field, digest) in [
            ("protocol_sha256", scope.protocol_sha256.as_str()),
            ("scorer_sha256", scope.scorer_sha256.as_str()),
            ("sample_manifest_sha256", scope.sample_manifest_sha256.as_str()),
        ] {
            if !is_lower_sha256(digest) {
                mismatches.push(MusicEditScopeMismatchV1::InvalidScopeDigest(field));
            }
        }
        if scope.native_metric_ids.is_empty()
            || scope.native_metric_ids.iter().any(|metric| metric.trim().is_empty())
        {
            mismatches.push(MusicEditScopeMismatchV1::MissingNativeMetric);
        }
        let mut unique_metrics = BTreeSet::new();
        if scope
            .native_metric_ids
            .iter()
            .any(|metric| !unique_metrics.insert(metric))
        {
            mismatches.push(MusicEditScopeMismatchV1::DuplicateNativeMetric);
        }
    }
    if left.schema_version != right.schema_version
        || left.schema_version != MUSIC_EDIT_COMPARISON_SCOPE_SCHEMA_V1
        || right.schema_version != MUSIC_EDIT_COMPARISON_SCOPE_SCHEMA_V1
    {
        mismatches.push(MusicEditScopeMismatchV1::Schema);
    }
    if left.benchmark_id != right.benchmark_id {
        mismatches.push(MusicEditScopeMismatchV1::BenchmarkId);
    }
    if left.benchmark_version != right.benchmark_version {
        mismatches.push(MusicEditScopeMismatchV1::BenchmarkVersion);
    }
    if left.native_track_id != right.native_track_id {
        mismatches.push(MusicEditScopeMismatchV1::Track);
    }
    if left.dataset_stratum != right.dataset_stratum {
        mismatches.push(MusicEditScopeMismatchV1::DatasetStratum);
    }
    if left.split_id != right.split_id {
        mismatches.push(MusicEditScopeMismatchV1::Split);
    }
    if left.protocol_sha256 != right.protocol_sha256 {
        mismatches.push(MusicEditScopeMismatchV1::Protocol);
    }
    if left.scorer_sha256 != right.scorer_sha256 {
        mismatches.push(MusicEditScopeMismatchV1::Scorer);
    }
    let mut left_metrics = left.native_metric_ids.clone();
    let mut right_metrics = right.native_metric_ids.clone();
    left_metrics.sort();
    right_metrics.sort();
    if left_metrics != right_metrics {
        mismatches.push(MusicEditScopeMismatchV1::NativeMetricSet);
    }
    if left.sample_manifest_sha256 != right.sample_manifest_sha256 {
        mismatches.push(MusicEditScopeMismatchV1::SampleManifest);
    }
    MusicEditScopeComparisonV1 {
        directly_comparable: mismatches.is_empty(),
        mismatches,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn row(sample: &str, lineage: &str, split: MusicEditSplitV1) -> MusicEditDatasetRowV1 {
        MusicEditDatasetRowV1 {
            sample_id: sample.to_owned(),
            source_lineage_sha256: lineage.to_owned(),
            split,
            stratum: MusicEditDatasetStratumV1::MidiSynthesized,
            operation: MusicEditOperationV1::RemoveInstrument,
            target_scope: MusicEditTargetScopeV1::Instrument,
            source_audio_sha256: A.to_owned(),
            instruction_sha256: B.to_owned(),
            target_audio_sha256: C.to_owned(),
            source_symbolic_sha256: Some(D.to_owned()),
            target_selector_sha256: E.to_owned(),
            metadata_sha256: A.to_owned(),
        }
    }

    fn manifest(rows: Vec<MusicEditDatasetRowV1>) -> MusicEditDatasetManifestV1 {
        MusicEditDatasetManifestV1 {
            schema_version: MUSIC_EDIT_DATASET_MANIFEST_SCHEMA_V1.to_owned(),
            dataset_id: "synthetic-adapter-fixture".to_owned(),
            dataset_version: "synthetic-v1".to_owned(),
            benchmark_track_id: "music-edit-development".to_owned(),
            protocol_sha256: A.to_owned(),
            scorer_sha256: B.to_owned(),
            access_tier: MusicEditAccessTierV1::Public,
            access_evidence_sha256: None,
            intended_use: MusicEditPermittedUseV1::EvaluationOnly,
            rights: MusicEditRightsV1 {
                state: MusicEditRightsStateV1::Cleared,
                rights_reference: "generated-only".to_owned(),
                rights_evidence_sha256: Some(C.to_owned()),
                permitted_uses: vec![MusicEditPermittedUseV1::EvaluationOnly],
            },
            rows,
        }
    }

    fn default_matrix() -> MusicEditCapabilityMatrixV1 {
        MusicEditCapabilityMatrixV1 {
            schema_version: "mel-music-edit-capability-matrix-v1".to_owned(),
            capabilities: MusicEditOperationV1::ALL
                .into_iter()
                .map(|operation| MusicEditOperationCapabilityV1 {
                    operation,
                    qualification: OperationQualificationV1::NotYetQualified,
                    qualification_evidence_sha256: None,
                    boundary: "not qualified on a frozen external benchmark protocol".to_owned(),
                })
                .collect(),
        }
    }

    fn scope(track: &str, stratum: MusicEditDatasetStratumV1) -> MusicEditComparisonScopeV1 {
        MusicEditComparisonScopeV1 {
            schema_version: MUSIC_EDIT_COMPARISON_SCOPE_SCHEMA_V1.to_owned(),
            benchmark_id: "synthetic".to_owned(),
            benchmark_version: "v1".to_owned(),
            native_track_id: track.to_owned(),
            dataset_stratum: stratum,
            split_id: "development".to_owned(),
            protocol_sha256: A.to_owned(),
            scorer_sha256: B.to_owned(),
            native_metric_ids: vec!["target_agreement".to_owned(), "preservation".to_owned()],
            sample_manifest_sha256: C.to_owned(),
        }
    }

    #[test]
    fn all_twelve_smbench_operations_have_exactly_one_mapping_slot() {
        let matrix = default_matrix();
        assert!(validate_music_edit_capability_matrix_v1(&matrix).is_empty());
        assert_eq!(matrix.capabilities.len(), 12);
        assert_eq!(
            qualification_for(&matrix, MusicEditOperationV1::TuningCorrection440),
            Some(OperationQualificationV1::NotYetQualified)
        );
    }

    #[test]
    fn duplicate_sample_ids_reject() {
        let manifest = manifest(vec![
            row("sample-1", A, MusicEditSplitV1::Development),
            row("sample-1", A, MusicEditSplitV1::Development),
        ]);
        assert!(
            validate_music_edit_dataset_manifest_v1(&manifest)
                .contains(&MusicEditAdapterIssueV1::DuplicateSampleId)
        );
    }

    #[test]
    fn source_lineage_cannot_cross_data_splits() {
        let manifest = manifest(vec![
            row("sample-train", A, MusicEditSplitV1::Train),
            row("sample-test", A, MusicEditSplitV1::Test),
        ]);
        assert!(
            validate_music_edit_dataset_manifest_v1(&manifest)
                .contains(&MusicEditAdapterIssueV1::LineageCrossesSplits)
        );
    }

    #[test]
    fn restricted_rights_without_permission_evidence_reject() {
        let mut manifest = manifest(vec![row("sample-1", A, MusicEditSplitV1::Development)]);
        manifest.rights.state = MusicEditRightsStateV1::Restricted;
        manifest.rights.rights_evidence_sha256 = None;
        assert!(
            validate_music_edit_dataset_manifest_v1(&manifest)
                .contains(&MusicEditAdapterIssueV1::MissingRightsEvidence)
        );
    }

    #[test]
    fn organizer_restricted_data_without_access_evidence_rejects() {
        let mut manifest = manifest(vec![row("sample-1", A, MusicEditSplitV1::Development)]);
        manifest.access_tier = MusicEditAccessTierV1::OrganizerRestricted;
        assert!(
            validate_music_edit_dataset_manifest_v1(&manifest)
                .contains(&MusicEditAdapterIssueV1::MissingAccessEvidence)
        );
    }

    #[test]
    fn mutable_dataset_alias_rejects() {
        let mut manifest = manifest(vec![row("sample-1", A, MusicEditSplitV1::Development)]);
        manifest.dataset_version = "latest".to_owned();
        assert!(
            validate_music_edit_dataset_manifest_v1(&manifest)
                .contains(&MusicEditAdapterIssueV1::MutableDatasetVersion)
        );
    }

    #[test]
    fn empty_and_duplicate_permitted_uses_reject() {
        let mut manifest = manifest(vec![row("sample-1", A, MusicEditSplitV1::Development)]);
        manifest.rights.permitted_uses.clear();
        assert!(
            validate_music_edit_dataset_manifest_v1(&manifest)
                .contains(&MusicEditAdapterIssueV1::EmptyPermittedUses)
        );
        manifest.rights.permitted_uses = vec![
            MusicEditPermittedUseV1::EvaluationOnly,
            MusicEditPermittedUseV1::EvaluationOnly,
        ];
        assert!(
            validate_music_edit_dataset_manifest_v1(&manifest)
                .contains(&MusicEditAdapterIssueV1::DuplicatePermittedUse)
        );
    }

    #[test]
    fn qualified_operation_requires_frozen_evidence_digest() {
        let mut matrix = default_matrix();
        matrix.capabilities[0].qualification = OperationQualificationV1::Qualified;
        assert!(
            validate_music_edit_capability_matrix_v1(&matrix)
                .contains(&MusicEditAdapterIssueV1::QualifiedWithoutEvidence)
        );
        matrix.capabilities[0].qualification_evidence_sha256 = Some(D.to_owned());
        assert!(validate_music_edit_capability_matrix_v1(&matrix).is_empty());
    }

    #[test]
    fn direct_comparison_rejects_track_and_stratum_mismatch() {
        let left = scope("mmae-agent-track", MusicEditDatasetStratumV1::MmaeMusicSubset);
        let right = scope("mmae-single-model-track", MusicEditDatasetStratumV1::MmaeMusicSubset);
        let result = compare_music_edit_scopes_v1(&left, &right);
        assert!(!result.directly_comparable);
        assert!(result.mismatches.contains(&MusicEditScopeMismatchV1::Track));
    }

    #[test]
    fn identical_valid_scopes_are_directly_comparable() {
        let left = scope("smebench-test", MusicEditDatasetStratumV1::MidiSynthesized);
        let right = scope("smebench-test", MusicEditDatasetStratumV1::MidiSynthesized);
        let result = compare_music_edit_scopes_v1(&left, &right);
        assert!(result.directly_comparable);
        assert!(result.mismatches.is_empty());
    }

    #[test]
    fn invalid_or_duplicate_scope_metrics_prevent_direct_comparison() {
        let left = scope("smebench-test", MusicEditDatasetStratumV1::MidiSynthesized);
        let mut right = left.clone();
        right.protocol_sha256 = "not-a-digest".to_owned();
        right.native_metric_ids.push("target_agreement".to_owned());
        let result = compare_music_edit_scopes_v1(&left, &right);
        assert!(!result.directly_comparable);
        assert!(result.mismatches.contains(&MusicEditScopeMismatchV1::Protocol));
        assert!(result.mismatches.contains(&MusicEditScopeMismatchV1::NativeMetricSet));
    }

    #[test]
    fn synthetic_and_real_multitrack_strata_are_not_pooled() {
        let left = scope("smebench-test", MusicEditDatasetStratumV1::MidiSynthesized);
        let right = scope("smebench-test", MusicEditDatasetStratumV1::RealMultitrack);
        let result = compare_music_edit_scopes_v1(&left, &right);
        assert!(!result.directly_comparable);
        assert!(result.mismatches.contains(&MusicEditScopeMismatchV1::DatasetStratum));
    }
}
