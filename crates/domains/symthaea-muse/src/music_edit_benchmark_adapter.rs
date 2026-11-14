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
    MutableBenchmarkVersion,
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
        if matches!(
            scope.benchmark_version.to_ascii_lowercase().as_str(),
            "latest" | "current" | "stable"
        ) {
            mismatches.push(MusicEditScopeMismatchV1::MutableBenchmarkVersion);
        }
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
    fn mutable_comparison_scope_version_is_never_directly_comparable() {
        let mut scope = scope("smebench-test", MusicEditDatasetStratumV1::MidiSynthesized);
        scope.benchmark_version = "latest".to_owned();
        let result = compare_music_edit_scopes_v1(&scope, &scope);
        assert!(!result.directly_comparable);
        assert!(
            result
                .mismatches
                .contains(&MusicEditScopeMismatchV1::MutableBenchmarkVersion)
        );
    }

    #[test]
    fn invalid_or_duplicate_scope_metrics_prevent_direct_comparison() {
        let left = scope("smebench-test", MusicEditDatasetStratumV1::MidiSynthesized);
        let mut right = left.clone();
        right.protocol_sha256 = "not-a-digest".to_owned();
        right.native_metric_ids.push("target_agreement".to_owned());
        let result = compare_music_edit_scopes_v1(&left, &right);
        assert!(!result.directly_comparable);
        assert!(result.mismatches.contains(
            &MusicEditScopeMismatchV1::InvalidScopeDigest("protocol_sha256")
        ));
        assert!(result.mismatches.contains(
            &MusicEditScopeMismatchV1::DuplicateNativeMetric
        ));
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


// -----------------------------------------------------------------------------
// Offline, rights-gated byte materialization (MEL-MR-EVAL-031).
//
// The metadata adapter above validates declarations. This layer opens the
// referenced local files, hashes their actual bytes, and binds the observations
// into a deterministic receipt. It performs no network access, archive
// extraction, audio decoding, scoring, legal review, or acoustic near-duplicate
// analysis.
// -----------------------------------------------------------------------------

use sha2::{Digest as ShaDigest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

pub const MUSIC_EDIT_ARTIFACT_MAP_SCHEMA_V1: &str = "mel-music-edit-artifact-map-v1";
pub const MUSIC_EDIT_MATERIALIZATION_RECEIPT_SCHEMA_V1: &str =
    "mel-music-edit-materialization-receipt-v1";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicEditMaterializedArtifactRoleV1 {
    RightsEvidence,
    AccessEvidence,
    SourceAudio,
    Instruction,
    TargetAudio,
    SourceSymbolic,
    TargetSelector,
    Metadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditArtifactPathsV1 {
    pub sample_id: String,
    pub source_audio_path: String,
    pub instruction_path: String,
    pub target_audio_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_symbolic_path: Option<String>,
    pub target_selector_path: String,
    pub metadata_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicEditArtifactMapV1 {
    pub schema_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rights_evidence_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_evidence_path: Option<String>,
    pub rows: Vec<MusicEditArtifactPathsV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicEditMaterializedArtifactV1 {
    /// None only for dataset-level rights/access evidence.
    pub sample_id: Option<String>,
    pub source_lineage_sha256: Option<String>,
    pub split: Option<MusicEditSplitV1>,
    pub role: MusicEditMaterializedArtifactRoleV1,
    /// Root-relative path only. Absolute host paths are never written to receipts.
    pub relative_path: String,
    /// SHA-256 computed from bytes actually read through the opened file handle.
    pub sha256: String,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MusicEditMaterializationReceiptV1 {
    pub schema_version: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub benchmark_track_id: String,
    /// Commitment to the typed manifest serialization, not a hash supplied by it.
    pub dataset_manifest_sha256: String,
    /// Commitment to the typed artifact-map serialization.
    pub artifact_map_sha256: String,
    pub sample_count: usize,
    pub artifact_count: usize,
    pub total_bytes: u64,
    pub artifacts: Vec<MusicEditMaterializedArtifactV1>,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct MusicEditMaterializationReceiptCommitmentV1<'a> {
    schema_version: &'a str,
    dataset_id: &'a str,
    dataset_version: &'a str,
    benchmark_track_id: &'a str,
    dataset_manifest_sha256: &'a str,
    artifact_map_sha256: &'a str,
    sample_count: usize,
    artifact_count: usize,
    total_bytes: u64,
    artifacts: &'a [MusicEditMaterializedArtifactV1],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MusicEditMaterializationLimitsV1 {
    pub max_artifact_bytes: u64,
    pub max_total_bytes: u64,
}

impl Default for MusicEditMaterializationLimitsV1 {
    fn default() -> Self {
        Self {
            max_artifact_bytes: 2 * 1024 * 1024 * 1024,
            max_total_bytes: 64 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicEditMaterializationIssueV1 {
    InvalidDatasetManifest { issues: Vec<String> },
    WrongArtifactMapSchema,
    EmptyArtifactMap,
    DuplicateArtifactMapSample { sample_id: String },
    MissingArtifactMapSample { sample_id: String },
    UnexpectedArtifactMapSample { sample_id: String },
    SymbolicArtifactPresenceMismatch { sample_id: String },
    RightsEvidencePathMismatch,
    AccessEvidencePathMismatch,
    InvalidRelativePath {
        sample_id: Option<String>,
        role: MusicEditMaterializedArtifactRoleV1,
        path: String,
    },
    SymlinkDisallowed { path: String },
    PathEscapesRoot { path: String },
    MissingArtifact {
        sample_id: Option<String>,
        role: MusicEditMaterializedArtifactRoleV1,
        path: String,
    },
    NotDirectory { path: String },
    NotRegularFile { path: String },
    FilesystemError { path: String, operation: String },
    ArtifactTooLarge { path: String, observed: u64, limit: u64 },
    TotalBytesExceeded { observed: u64, limit: u64 },
    EmptyArtifact { path: String },
    FileChangedDuringRead { path: String },
    DigestMismatch {
        sample_id: Option<String>,
        role: MusicEditMaterializedArtifactRoleV1,
        path: String,
        expected: String,
        actual: String,
    },
    CrossLineageExactAudioDuplicate {
        digest: String,
        first_lineage: String,
        second_lineage: String,
    },
    SerializationError,
    WrongReceiptSchema,
    ReceiptManifestMismatch,
    ReceiptArtifactMapMismatch,
    ReceiptSampleCountMismatch,
    ReceiptArtifactCountMismatch,
    ReceiptByteCountMismatch,
    ReceiptMissingOrUnexpectedArtifact,
    ReceiptArtifactMismatch {
        sample_id: Option<String>,
        role: MusicEditMaterializedArtifactRoleV1,
    },
    ReceiptDuplicateArtifact {
        sample_id: Option<String>,
        role: MusicEditMaterializedArtifactRoleV1,
    },
    ReceiptNotCanonicalOrder,
    ReceiptDigestMismatch,
}

fn materialization_domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous.mel.music-edit-materialization.v1\0");
    hasher.update(domain);
    hasher.update(b"\0");
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn materialization_json_digest<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, MusicEditMaterializationIssueV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| MusicEditMaterializationIssueV1::SerializationError)?;
    Ok(materialization_domain_hash(domain, &bytes))
}

fn is_materialization_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn safe_materialization_relative_path(value: &str) -> bool {
    if value.is_empty()
        || value.trim() != value
        || value.starts_with('/')
        || value.contains('\\')
        || value.contains(':')
        || value.contains('\0')
        || !value
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
    {
        return false;
    }
    let path = Path::new(value);
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn validate_music_edit_artifact_map_v1(
    manifest: &MusicEditDatasetManifestV1,
    artifact_map: &MusicEditArtifactMapV1,
) -> Vec<MusicEditMaterializationIssueV1> {
    let mut issues = Vec::new();
    if artifact_map.schema_version != MUSIC_EDIT_ARTIFACT_MAP_SCHEMA_V1 {
        issues.push(MusicEditMaterializationIssueV1::WrongArtifactMapSchema);
    }
    if artifact_map.rows.is_empty() {
        issues.push(MusicEditMaterializationIssueV1::EmptyArtifactMap);
    }

    let expected_ids: BTreeSet<&str> =
        manifest.rows.iter().map(|row| row.sample_id.as_str()).collect();
    let mut mapped = BTreeMap::<&str, &MusicEditArtifactPathsV1>::new();
    for row in &artifact_map.rows {
        if mapped.insert(row.sample_id.as_str(), row).is_some() {
            issues.push(MusicEditMaterializationIssueV1::DuplicateArtifactMapSample {
                sample_id: row.sample_id.clone(),
            });
        }
    }
    for sample_id in &expected_ids {
        if !mapped.contains_key(sample_id) {
            issues.push(MusicEditMaterializationIssueV1::MissingArtifactMapSample {
                sample_id: (*sample_id).to_owned(),
            });
        }
    }
    for sample_id in mapped.keys() {
        if !expected_ids.contains(sample_id) {
            issues.push(MusicEditMaterializationIssueV1::UnexpectedArtifactMapSample {
                sample_id: (*sample_id).to_owned(),
            });
        }
    }

    for row in &manifest.rows {
        if let Some(paths) = mapped.get(row.sample_id.as_str()) {
            if row.source_symbolic_sha256.is_some() != paths.source_symbolic_path.is_some() {
                issues.push(
                    MusicEditMaterializationIssueV1::SymbolicArtifactPresenceMismatch {
                        sample_id: row.sample_id.clone(),
                    },
                );
            }
        }
    }

    if manifest.rights.rights_evidence_sha256.is_some()
        != artifact_map.rights_evidence_path.is_some()
    {
        issues.push(MusicEditMaterializationIssueV1::RightsEvidencePathMismatch);
    }
    if manifest.access_evidence_sha256.is_some() != artifact_map.access_evidence_path.is_some() {
        issues.push(MusicEditMaterializationIssueV1::AccessEvidencePathMismatch);
    }
    issues
}

fn resolve_materialization_path(
    canonical_root: &Path,
    relative_path: &str,
    sample_id: Option<&str>,
    role: MusicEditMaterializedArtifactRoleV1,
) -> Result<PathBuf, MusicEditMaterializationIssueV1> {
    let fail_path = |path: &str| MusicEditMaterializationIssueV1::InvalidRelativePath {
        sample_id: sample_id.map(str::to_owned),
        role: role.clone(),
        path: path.to_owned(),
    };
    if !safe_materialization_relative_path(relative_path) {
        return Err(fail_path(relative_path));
    }

    let mut candidate = canonical_root.to_path_buf();
    let components: Vec<_> = Path::new(relative_path).components().collect();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(part) = component else {
            return Err(fail_path(relative_path));
        };
        candidate.push(part);
        let metadata = fs::symlink_metadata(&candidate).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                MusicEditMaterializationIssueV1::MissingArtifact {
                    sample_id: sample_id.map(str::to_owned),
                    role: role.clone(),
                    path: relative_path.to_owned(),
                }
            } else {
                MusicEditMaterializationIssueV1::FilesystemError {
                    path: relative_path.to_owned(),
                    operation: "symlink_metadata".to_owned(),
                }
            }
        })?;
        if metadata.file_type().is_symlink() {
            return Err(MusicEditMaterializationIssueV1::SymlinkDisallowed {
                path: relative_path.to_owned(),
            });
        }
        if index + 1 < components.len() && !metadata.is_dir() {
            return Err(MusicEditMaterializationIssueV1::NotDirectory {
                path: relative_path.to_owned(),
            });
        }
        if index + 1 == components.len() && !metadata.is_file() {
            return Err(MusicEditMaterializationIssueV1::NotRegularFile {
                path: relative_path.to_owned(),
            });
        }
    }

    let canonical_candidate = fs::canonicalize(&candidate).map_err(|_| {
        MusicEditMaterializationIssueV1::FilesystemError {
            path: relative_path.to_owned(),
            operation: "canonicalize".to_owned(),
        }
    })?;
    if !canonical_candidate.starts_with(canonical_root) {
        return Err(MusicEditMaterializationIssueV1::PathEscapesRoot {
            path: relative_path.to_owned(),
        });
    }
    Ok(canonical_candidate)
}

fn verify_and_hash_materialization_artifact(
    canonical_root: &Path,
    sample_id: Option<&str>,
    source_lineage_sha256: Option<&str>,
    split: Option<MusicEditSplitV1>,
    role: MusicEditMaterializedArtifactRoleV1,
    relative_path: &str,
    expected_sha256: &str,
    limits: MusicEditMaterializationLimitsV1,
    total_bytes: &mut u64,
) -> Result<MusicEditMaterializedArtifactV1, MusicEditMaterializationIssueV1> {
    if !is_materialization_sha256(expected_sha256) {
        return Err(MusicEditMaterializationIssueV1::DigestMismatch {
            sample_id: sample_id.map(str::to_owned),
            role,
            path: relative_path.to_owned(),
            expected: expected_sha256.to_owned(),
            actual: String::new(),
        });
    }

    let path = resolve_materialization_path(
        canonical_root,
        relative_path,
        sample_id,
        role.clone(),
    )?;
    let mut file = File::open(&path).map_err(|_| {
        MusicEditMaterializationIssueV1::FilesystemError {
            path: relative_path.to_owned(),
            operation: "open".to_owned(),
        }
    })?;
    let before = file
        .metadata()
        .map_err(|_| MusicEditMaterializationIssueV1::FilesystemError {
            path: relative_path.to_owned(),
            operation: "metadata_before_read".to_owned(),
        })?;
    if !before.is_file() {
        return Err(MusicEditMaterializationIssueV1::NotRegularFile {
            path: relative_path.to_owned(),
        });
    }
    if before.len() == 0 {
        return Err(MusicEditMaterializationIssueV1::EmptyArtifact {
            path: relative_path.to_owned(),
        });
    }
    if before.len() > limits.max_artifact_bytes {
        return Err(MusicEditMaterializationIssueV1::ArtifactTooLarge {
            path: relative_path.to_owned(),
            observed: before.len(),
            limit: limits.max_artifact_bytes,
        });
    }
    let projected_total = total_bytes.checked_add(before.len()).ok_or(
        MusicEditMaterializationIssueV1::TotalBytesExceeded {
            observed: u64::MAX,
            limit: limits.max_total_bytes,
        },
    )?;
    if projected_total > limits.max_total_bytes {
        return Err(MusicEditMaterializationIssueV1::TotalBytesExceeded {
            observed: projected_total,
            limit: limits.max_total_bytes,
        });
    }

    let before_modified: Option<SystemTime> = before.modified().ok();
    let mut hasher = Sha256::new();
    let mut byte_len = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|_| {
            MusicEditMaterializationIssueV1::FilesystemError {
                path: relative_path.to_owned(),
                operation: "read".to_owned(),
            }
        })?;
        if count == 0 {
            break;
        }
        byte_len = byte_len.checked_add(count as u64).ok_or(
            MusicEditMaterializationIssueV1::ArtifactTooLarge {
                path: relative_path.to_owned(),
                observed: u64::MAX,
                limit: limits.max_artifact_bytes,
            },
        )?;
        if byte_len > limits.max_artifact_bytes {
            return Err(MusicEditMaterializationIssueV1::ArtifactTooLarge {
                path: relative_path.to_owned(),
                observed: byte_len,
                limit: limits.max_artifact_bytes,
            });
        }
        hasher.update(&buffer[..count]);
    }
    let after = file
        .metadata()
        .map_err(|_| MusicEditMaterializationIssueV1::FilesystemError {
            path: relative_path.to_owned(),
            operation: "metadata_after_read".to_owned(),
        })?;
    let after_modified: Option<SystemTime> = after.modified().ok();
    if byte_len != before.len()
        || after.len() != before.len()
        || (before_modified.is_some()
            && after_modified.is_some()
            && before_modified != after_modified)
    {
        return Err(MusicEditMaterializationIssueV1::FileChangedDuringRead {
            path: relative_path.to_owned(),
        });
    }
    let actual_sha256 = format!("{:x}", hasher.finalize());
    if actual_sha256 != expected_sha256 {
        return Err(MusicEditMaterializationIssueV1::DigestMismatch {
            sample_id: sample_id.map(str::to_owned),
            role,
            path: relative_path.to_owned(),
            expected: expected_sha256.to_owned(),
            actual: actual_sha256,
        });
    }
    *total_bytes = projected_total;
    Ok(MusicEditMaterializedArtifactV1 {
        sample_id: sample_id.map(str::to_owned),
        source_lineage_sha256: source_lineage_sha256.map(str::to_owned),
        split,
        role,
        relative_path: relative_path.to_owned(),
        sha256: actual_sha256,
        byte_len,
    })
}

type ExpectedMaterializationArtifactV1 = BTreeMap<
    (Option<String>, MusicEditMaterializedArtifactRoleV1),
    (String, String, Option<String>, Option<MusicEditSplitV1>),
>;

fn push_expected_artifact(
    expected: &mut ExpectedMaterializationArtifactV1,
    sample_id: Option<String>,
    role: MusicEditMaterializedArtifactRoleV1,
    relative_path: &str,
    digest: &str,
    lineage: Option<String>,
    split: Option<MusicEditSplitV1>,
) {
    expected.insert(
        (sample_id, role),
        (relative_path.to_owned(), digest.to_owned(), lineage, split),
    );
}

fn expected_materialization_artifacts(
    manifest: &MusicEditDatasetManifestV1,
    artifact_map: &MusicEditArtifactMapV1,
) -> ExpectedMaterializationArtifactV1 {
    let mut expected = BTreeMap::new();
    if let (Some(path), Some(digest)) = (
        artifact_map.rights_evidence_path.as_deref(),
        manifest.rights.rights_evidence_sha256.as_deref(),
    ) {
        push_expected_artifact(
            &mut expected,
            None,
            MusicEditMaterializedArtifactRoleV1::RightsEvidence,
            path,
            digest,
            None,
            None,
        );
    }
    if let (Some(path), Some(digest)) = (
        artifact_map.access_evidence_path.as_deref(),
        manifest.access_evidence_sha256.as_deref(),
    ) {
        push_expected_artifact(
            &mut expected,
            None,
            MusicEditMaterializedArtifactRoleV1::AccessEvidence,
            path,
            digest,
            None,
            None,
        );
    }
    let map_rows: BTreeMap<&str, &MusicEditArtifactPathsV1> = artifact_map
        .rows
        .iter()
        .map(|row| (row.sample_id.as_str(), row))
        .collect();
    for row in &manifest.rows {
        let Some(paths) = map_rows.get(row.sample_id.as_str()) else {
            continue;
        };
        let sample_id = Some(row.sample_id.clone());
        let lineage = Some(row.source_lineage_sha256.clone());
        let split = Some(row.split);
        let mut add = |role, path: &str, digest: &str| {
            push_expected_artifact(
                &mut expected,
                sample_id.clone(),
                role,
                path,
                digest,
                lineage.clone(),
                split,
            );
        };
        add(
            MusicEditMaterializedArtifactRoleV1::SourceAudio,
            &paths.source_audio_path,
            &row.source_audio_sha256,
        );
        add(
            MusicEditMaterializedArtifactRoleV1::Instruction,
            &paths.instruction_path,
            &row.instruction_sha256,
        );
        add(
            MusicEditMaterializedArtifactRoleV1::TargetAudio,
            &paths.target_audio_path,
            &row.target_audio_sha256,
        );
        if let (Some(path), Some(digest)) = (
            paths.source_symbolic_path.as_deref(),
            row.source_symbolic_sha256.as_deref(),
        ) {
            add(
                MusicEditMaterializedArtifactRoleV1::SourceSymbolic,
                path,
                digest,
            );
        }
        add(
            MusicEditMaterializedArtifactRoleV1::TargetSelector,
            &paths.target_selector_path,
            &row.target_selector_sha256,
        );
        add(
            MusicEditMaterializedArtifactRoleV1::Metadata,
            &paths.metadata_path,
            &row.metadata_sha256,
        );
    }
    expected
}

fn verify_no_cross_lineage_exact_audio_duplicates(
    artifacts: &[MusicEditMaterializedArtifactV1],
) -> Result<(), MusicEditMaterializationIssueV1> {
    let mut digest_lineage = BTreeMap::<&str, &str>::new();
    for artifact in artifacts {
        if !matches!(
            artifact.role,
            MusicEditMaterializedArtifactRoleV1::SourceAudio
                | MusicEditMaterializedArtifactRoleV1::TargetAudio
        ) {
            continue;
        }
        let Some(lineage) = artifact.source_lineage_sha256.as_deref() else {
            continue;
        };
        if let Some(first_lineage) = digest_lineage.get(artifact.sha256.as_str()) {
            if *first_lineage != lineage {
                return Err(
                    MusicEditMaterializationIssueV1::CrossLineageExactAudioDuplicate {
                        digest: artifact.sha256.clone(),
                        first_lineage: (*first_lineage).to_owned(),
                        second_lineage: lineage.to_owned(),
                    },
                );
            }
        } else {
            digest_lineage.insert(artifact.sha256.as_str(), lineage);
        }
    }
    Ok(())
}

fn receipt_digest_v1(
    receipt: &MusicEditMaterializationReceiptV1,
) -> Result<String, MusicEditMaterializationIssueV1> {
    let commitment = MusicEditMaterializationReceiptCommitmentV1 {
        schema_version: &receipt.schema_version,
        dataset_id: &receipt.dataset_id,
        dataset_version: &receipt.dataset_version,
        benchmark_track_id: &receipt.benchmark_track_id,
        dataset_manifest_sha256: &receipt.dataset_manifest_sha256,
        artifact_map_sha256: &receipt.artifact_map_sha256,
        sample_count: receipt.sample_count,
        artifact_count: receipt.artifact_count,
        total_bytes: receipt.total_bytes,
        artifacts: &receipt.artifacts,
    };
    materialization_json_digest(b"receipt", &commitment)
}

/// Open and hash every declared local artifact. This function is intentionally
/// offline: it never downloads or extracts benchmark content. The caller must
/// already possess the files and explicit use authorization.
///
/// A successful receipt proves byte/digest correspondence for the files read
/// during this invocation. It does not authenticate the supplier, adjudicate
/// legal rights, validate audio decodability, detect acoustic near-duplicates,
/// or qualify any editing operation.
pub fn materialize_music_edit_dataset_v1(
    root: &Path,
    manifest: &MusicEditDatasetManifestV1,
    artifact_map: &MusicEditArtifactMapV1,
    limits: MusicEditMaterializationLimitsV1,
) -> Result<MusicEditMaterializationReceiptV1, Vec<MusicEditMaterializationIssueV1>> {
    let manifest_issues = validate_music_edit_dataset_manifest_v1(manifest);
    if !manifest_issues.is_empty() {
        return Err(vec![
            MusicEditMaterializationIssueV1::InvalidDatasetManifest {
                issues: manifest_issues
                    .iter()
                    .map(|issue| format!("{issue:?}"))
                    .collect(),
            },
        ]);
    }
    let map_issues = validate_music_edit_artifact_map_v1(manifest, artifact_map);
    if !map_issues.is_empty() {
        return Err(map_issues);
    }
    if limits.max_artifact_bytes == 0 || limits.max_total_bytes == 0 {
        return Err(vec![
            MusicEditMaterializationIssueV1::TotalBytesExceeded {
                observed: 0,
                limit: limits.max_total_bytes,
            },
        ]);
    }

    let root_metadata = fs::symlink_metadata(root).map_err(|_| {
        vec![MusicEditMaterializationIssueV1::FilesystemError {
            path: root.display().to_string(),
            operation: "root_symlink_metadata".to_owned(),
        }]
    })?;
    if root_metadata.file_type().is_symlink() {
        return Err(vec![
            MusicEditMaterializationIssueV1::SymlinkDisallowed {
                path: root.display().to_string(),
            },
        ]);
    }
    if !root_metadata.is_dir() {
        return Err(vec![MusicEditMaterializationIssueV1::NotDirectory {
            path: root.display().to_string(),
        }]);
    }
    let canonical_root = fs::canonicalize(root).map_err(|_| {
        vec![MusicEditMaterializationIssueV1::FilesystemError {
            path: root.display().to_string(),
            operation: "canonicalize_root".to_owned(),
        }]
    })?;

    let dataset_manifest_sha256 =
        materialization_json_digest(b"dataset-manifest", manifest).map_err(|issue| vec![issue])?;
    let artifact_map_sha256 =
        materialization_json_digest(b"artifact-map", artifact_map).map_err(|issue| vec![issue])?;
    let map_rows: BTreeMap<&str, &MusicEditArtifactPathsV1> = artifact_map
        .rows
        .iter()
        .map(|row| (row.sample_id.as_str(), row))
        .collect();

    let mut artifacts = Vec::new();
    let mut total_bytes = 0_u64;

    if let (Some(path), Some(digest)) = (
        artifact_map.rights_evidence_path.as_deref(),
        manifest.rights.rights_evidence_sha256.as_deref(),
    ) {
        artifacts.push(
            verify_and_hash_materialization_artifact(
                &canonical_root,
                None,
                None,
                None,
                MusicEditMaterializedArtifactRoleV1::RightsEvidence,
                path,
                digest,
                limits,
                &mut total_bytes,
            )
            .map_err(|issue| vec![issue])?,
        );
    }
    if let (Some(path), Some(digest)) = (
        artifact_map.access_evidence_path.as_deref(),
        manifest.access_evidence_sha256.as_deref(),
    ) {
        artifacts.push(
            verify_and_hash_materialization_artifact(
                &canonical_root,
                None,
                None,
                None,
                MusicEditMaterializedArtifactRoleV1::AccessEvidence,
                path,
                digest,
                limits,
                &mut total_bytes,
            )
            .map_err(|issue| vec![issue])?,
        );
    }

    for row in &manifest.rows {
        let Some(paths) = map_rows.get(row.sample_id.as_str()) else {
            return Err(vec![
                MusicEditMaterializationIssueV1::MissingArtifactMapSample {
                    sample_id: row.sample_id.clone(),
                },
            ]);
        };
        let sample_id = Some(row.sample_id.as_str());
        let lineage = Some(row.source_lineage_sha256.as_str());
        let split = Some(row.split);
        let mut add_artifact = |role, path: &str, digest: &str| {
            verify_and_hash_materialization_artifact(
                &canonical_root,
                sample_id,
                lineage,
                split,
                role,
                path,
                digest,
                limits,
                &mut total_bytes,
            )
        };

        artifacts.push(
            add_artifact(
                MusicEditMaterializedArtifactRoleV1::SourceAudio,
                &paths.source_audio_path,
                &row.source_audio_sha256,
            )
            .map_err(|issue| vec![issue])?,
        );
        artifacts.push(
            add_artifact(
                MusicEditMaterializedArtifactRoleV1::Instruction,
                &paths.instruction_path,
                &row.instruction_sha256,
            )
            .map_err(|issue| vec![issue])?,
        );
        artifacts.push(
            add_artifact(
                MusicEditMaterializedArtifactRoleV1::TargetAudio,
                &paths.target_audio_path,
                &row.target_audio_sha256,
            )
            .map_err(|issue| vec![issue])?,
        );
        if let (Some(path), Some(digest)) = (
            paths.source_symbolic_path.as_deref(),
            row.source_symbolic_sha256.as_deref(),
        ) {
            artifacts.push(
                add_artifact(
                    MusicEditMaterializedArtifactRoleV1::SourceSymbolic,
                    path,
                    digest,
                )
                .map_err(|issue| vec![issue])?,
            );
        }
        artifacts.push(
            add_artifact(
                MusicEditMaterializedArtifactRoleV1::TargetSelector,
                &paths.target_selector_path,
                &row.target_selector_sha256,
            )
            .map_err(|issue| vec![issue])?,
        );
        artifacts.push(
            add_artifact(
                MusicEditMaterializedArtifactRoleV1::Metadata,
                &paths.metadata_path,
                &row.metadata_sha256,
            )
            .map_err(|issue| vec![issue])?,
        );
    }

    verify_no_cross_lineage_exact_audio_duplicates(&artifacts).map_err(|issue| vec![issue])?;
    artifacts.sort_by(|left, right| {
        (&left.sample_id, &left.role, &left.relative_path).cmp(&(
            &right.sample_id,
            &right.role,
            &right.relative_path,
        ))
    });

    let mut receipt = MusicEditMaterializationReceiptV1 {
        schema_version: MUSIC_EDIT_MATERIALIZATION_RECEIPT_SCHEMA_V1.to_owned(),
        dataset_id: manifest.dataset_id.clone(),
        dataset_version: manifest.dataset_version.clone(),
        benchmark_track_id: manifest.benchmark_track_id.clone(),
        dataset_manifest_sha256,
        artifact_map_sha256,
        sample_count: manifest.rows.len(),
        artifact_count: artifacts.len(),
        total_bytes,
        artifacts,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_digest_v1(&receipt).map_err(|issue| vec![issue])?;
    Ok(receipt)
}

/// Verify that a serialized receipt is internally complete, canonical and bound
/// to the supplied manifest and artifact map. This is a receipt-integrity check;
/// it does not reopen files. Re-running materialize_music_edit_dataset_v1 is
/// required to re-establish that the current bytes still match the declarations.
pub fn verify_music_edit_materialization_receipt_v1(
    receipt: &MusicEditMaterializationReceiptV1,
    manifest: &MusicEditDatasetManifestV1,
    artifact_map: &MusicEditArtifactMapV1,
) -> Vec<MusicEditMaterializationIssueV1> {
    let mut issues = Vec::new();
    if receipt.schema_version != MUSIC_EDIT_MATERIALIZATION_RECEIPT_SCHEMA_V1 {
        issues.push(MusicEditMaterializationIssueV1::WrongReceiptSchema);
    }
    if !validate_music_edit_dataset_manifest_v1(manifest).is_empty()
        || !validate_music_edit_artifact_map_v1(manifest, artifact_map).is_empty()
    {
        issues.push(MusicEditMaterializationIssueV1::ReceiptMissingOrUnexpectedArtifact);
        return issues;
    }
    if receipt.dataset_id != manifest.dataset_id
        || receipt.dataset_version != manifest.dataset_version
        || receipt.benchmark_track_id != manifest.benchmark_track_id
    {
        issues.push(MusicEditMaterializationIssueV1::ReceiptManifestMismatch);
    }
    let manifest_digest = match materialization_json_digest(b"dataset-manifest", manifest) {
        Ok(value) => value,
        Err(issue) => {
            issues.push(issue);
            return issues;
        }
    };
    let map_digest = match materialization_json_digest(b"artifact-map", artifact_map) {
        Ok(value) => value,
        Err(issue) => {
            issues.push(issue);
            return issues;
        }
    };
    if receipt.dataset_manifest_sha256 != manifest_digest {
        issues.push(MusicEditMaterializationIssueV1::ReceiptManifestMismatch);
    }
    if receipt.artifact_map_sha256 != map_digest {
        issues.push(MusicEditMaterializationIssueV1::ReceiptArtifactMapMismatch);
    }
    if receipt.sample_count != manifest.rows.len() {
        issues.push(MusicEditMaterializationIssueV1::ReceiptSampleCountMismatch);
    }

    let expected = expected_materialization_artifacts(manifest, artifact_map);
    if receipt.artifact_count != receipt.artifacts.len() {
        issues.push(MusicEditMaterializationIssueV1::ReceiptArtifactCountMismatch);
    }
    if receipt.artifacts.len() != expected.len() {
        issues.push(MusicEditMaterializationIssueV1::ReceiptMissingOrUnexpectedArtifact);
    }

    let mut seen = BTreeSet::new();
    let mut observed_total = 0_u64;
    let mut previous_key: Option<(
        Option<String>,
        MusicEditMaterializedArtifactRoleV1,
        String,
    )> = None;

    for artifact in &receipt.artifacts {
        let key = (artifact.sample_id.clone(), artifact.role.clone());
        if !seen.insert(key.clone()) {
            issues.push(MusicEditMaterializationIssueV1::ReceiptDuplicateArtifact {
                sample_id: artifact.sample_id.clone(),
                role: artifact.role.clone(),
            });
        }
        let order_key = (
            artifact.sample_id.clone(),
            artifact.role.clone(),
            artifact.relative_path.clone(),
        );
        if previous_key.as_ref().is_some_and(|previous| previous > &order_key) {
            issues.push(MusicEditMaterializationIssueV1::ReceiptNotCanonicalOrder);
        }
        previous_key = Some(order_key);

        if !is_materialization_sha256(&artifact.sha256) || artifact.byte_len == 0 {
            issues.push(MusicEditMaterializationIssueV1::ReceiptArtifactMismatch {
                sample_id: artifact.sample_id.clone(),
                role: artifact.role.clone(),
            });
        }
        match expected.get(&key) {
            Some((path, digest, lineage, split))
                if path == &artifact.relative_path
                    && digest == &artifact.sha256
                    && lineage == &artifact.source_lineage_sha256
                    && split == &artifact.split => {}
            _ => issues.push(MusicEditMaterializationIssueV1::ReceiptArtifactMismatch {
                sample_id: artifact.sample_id.clone(),
                role: artifact.role.clone(),
            }),
        }
        observed_total = match observed_total.checked_add(artifact.byte_len) {
            Some(value) => value,
            None => {
                issues.push(MusicEditMaterializationIssueV1::ReceiptByteCountMismatch);
                u64::MAX
            }
        };
    }
    if observed_total != receipt.total_bytes {
        issues.push(MusicEditMaterializationIssueV1::ReceiptByteCountMismatch);
    }
    if verify_no_cross_lineage_exact_audio_duplicates(&receipt.artifacts).is_err() {
        issues.push(MusicEditMaterializationIssueV1::ReceiptMissingOrUnexpectedArtifact);
    }
    match receipt_digest_v1(receipt) {
        Ok(digest) if digest == receipt.receipt_sha256 => {}
        _ => issues.push(MusicEditMaterializationIssueV1::ReceiptDigestMismatch),
    }
    issues
}

#[cfg(test)]
mod materialization_tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new() -> Self {
            let serial = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "mel-music-edit-materializer-{}-{serial}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, relative: &str, bytes: &[u8]) -> String {
            let path = self.0.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut file = File::create(path).unwrap();
            file.write_all(bytes).unwrap();
            format!("{:x}", Sha256::digest(bytes))
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn manifest_and_map(root: &TestRoot) -> (MusicEditDatasetManifestV1, MusicEditArtifactMapV1) {
        let rights = b"synthetic rights evidence v1";
        let source = b"synthetic source audio bytes";
        let instruction = b"remove the violin";
        let target = b"synthetic target audio bytes";
        let symbolic = b"synthetic symbolic score bytes";
        let selector = b"{\"instrument\":\"violin\"}";
        let metadata = b"{\"synthetic\":true}";
        let rights_digest = root.write("dataset/rights.txt", rights);
        let source_digest = root.write("samples/one/source.wav", source);
        let instruction_digest = root.write("samples/one/instruction.txt", instruction);
        let target_digest = root.write("samples/one/target.wav", target);
        let symbolic_digest = root.write("samples/one/source.mid", symbolic);
        let selector_digest = root.write("samples/one/selector.json", selector);
        let metadata_digest = root.write("samples/one/metadata.json", metadata);
        let lineage = format!("{:x}", Sha256::digest(b"lineage-one"));

        let manifest = MusicEditDatasetManifestV1 {
            schema_version: MUSIC_EDIT_DATASET_MANIFEST_SCHEMA_V1.to_owned(),
            dataset_id: "synthetic-materializer-fixture".to_owned(),
            dataset_version: "synthetic-v1".to_owned(),
            benchmark_track_id: "synthetic-development".to_owned(),
            protocol_sha256: "a".repeat(64),
            scorer_sha256: "b".repeat(64),
            access_tier: MusicEditAccessTierV1::Public,
            access_evidence_sha256: None,
            intended_use: MusicEditPermittedUseV1::EvaluationOnly,
            rights: MusicEditRightsV1 {
                state: MusicEditRightsStateV1::Cleared,
                rights_reference: "synthetic local fixture".to_owned(),
                rights_evidence_sha256: Some(rights_digest),
                permitted_uses: vec![MusicEditPermittedUseV1::EvaluationOnly],
            },
            rows: vec![MusicEditDatasetRowV1 {
                sample_id: "sample-one".to_owned(),
                source_lineage_sha256: lineage,
                split: MusicEditSplitV1::Development,
                stratum: MusicEditDatasetStratumV1::MidiSynthesized,
                operation: MusicEditOperationV1::RemoveInstrument,
                target_scope: MusicEditTargetScopeV1::Instrument,
                source_audio_sha256: source_digest,
                instruction_sha256: instruction_digest,
                target_audio_sha256: target_digest,
                source_symbolic_sha256: Some(symbolic_digest),
                target_selector_sha256: selector_digest,
                metadata_sha256: metadata_digest,
            }],
        };
        let artifact_map = MusicEditArtifactMapV1 {
            schema_version: MUSIC_EDIT_ARTIFACT_MAP_SCHEMA_V1.to_owned(),
            rights_evidence_path: Some("dataset/rights.txt".to_owned()),
            access_evidence_path: None,
            rows: vec![MusicEditArtifactPathsV1 {
                sample_id: "sample-one".to_owned(),
                source_audio_path: "samples/one/source.wav".to_owned(),
                instruction_path: "samples/one/instruction.txt".to_owned(),
                target_audio_path: "samples/one/target.wav".to_owned(),
                source_symbolic_path: Some("samples/one/source.mid".to_owned()),
                target_selector_path: "samples/one/selector.json".to_owned(),
                metadata_path: "samples/one/metadata.json".to_owned(),
            }],
        };
        (manifest, artifact_map)
    }

    #[test]
    fn actual_bytes_materialize_and_receipt_replays() {
        let root = TestRoot::new();
        let (manifest, artifact_map) = manifest_and_map(&root);
        let receipt = materialize_music_edit_dataset_v1(
            &root.0,
            &manifest,
            &artifact_map,
            MusicEditMaterializationLimitsV1::default(),
        )
        .unwrap();
        assert_eq!(receipt.sample_count, 1);
        assert_eq!(receipt.artifact_count, 7);
        assert!(receipt.total_bytes > 0);
        assert!(verify_music_edit_materialization_receipt_v1(
            &receipt,
            &manifest,
            &artifact_map
        )
        .is_empty());
    }

    #[test]
    fn single_byte_source_mutation_rejects() {
        let root = TestRoot::new();
        let (manifest, artifact_map) = manifest_and_map(&root);
        fs::write(root.0.join("samples/one/source.wav"), b"mutated source bytes").unwrap();
        let result = materialize_music_edit_dataset_v1(
            &root.0,
            &manifest,
            &artifact_map,
            MusicEditMaterializationLimitsV1::default(),
        );
        assert!(matches!(
            result,
            Err(issues) if issues.iter().any(|issue| matches!(
                issue,
                MusicEditMaterializationIssueV1::DigestMismatch {
                    role: MusicEditMaterializedArtifactRoleV1::SourceAudio,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn path_traversal_and_missing_mapping_reject() {
        let root = TestRoot::new();
        let (manifest, mut artifact_map) = manifest_and_map(&root);
        artifact_map.rows[0].source_audio_path = "../outside.wav".to_owned();
        let result = materialize_music_edit_dataset_v1(
            &root.0,
            &manifest,
            &artifact_map,
            MusicEditMaterializationLimitsV1::default(),
        );
        assert!(matches!(
            result,
            Err(issues) if issues.iter().any(|issue| matches!(
                issue,
                MusicEditMaterializationIssueV1::InvalidRelativePath { .. }
            ))
        ));

        let (manifest, mut artifact_map) = manifest_and_map(&root);
        artifact_map.rows.clear();
        let result = materialize_music_edit_dataset_v1(
            &root.0,
            &manifest,
            &artifact_map,
            MusicEditMaterializationLimitsV1::default(),
        );
        assert!(matches!(
            result,
            Err(issues) if issues.iter().any(|issue| matches!(
                issue,
                MusicEditMaterializationIssueV1::MissingArtifactMapSample { .. }
            ))
        ));
    }

    #[test]
    fn symlinks_are_rejected() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let root = TestRoot::new();
            let (manifest, mut artifact_map) = manifest_and_map(&root);
            let outside = root.0.join("outside.wav");
            fs::write(&outside, b"synthetic source audio bytes").unwrap();
            let link = root.0.join("samples/one/source-link.wav");
            symlink(&outside, &link).unwrap();
            artifact_map.rows[0].source_audio_path = "samples/one/source-link.wav".to_owned();
            let result = materialize_music_edit_dataset_v1(
                &root.0,
                &manifest,
                &artifact_map,
                MusicEditMaterializationLimitsV1::default(),
            );
            assert!(matches!(
                result,
                Err(issues) if issues.iter().any(|issue| matches!(
                    issue,
                    MusicEditMaterializationIssueV1::SymlinkDisallowed { .. }
                ))
            ));
        }
    }

    #[test]
    fn rights_evidence_bytes_must_match_the_manifest() {
        let root = TestRoot::new();
        let (manifest, artifact_map) = manifest_and_map(&root);
        fs::write(root.0.join("dataset/rights.txt"), b"different permission text").unwrap();
        let result = materialize_music_edit_dataset_v1(
            &root.0,
            &manifest,
            &artifact_map,
            MusicEditMaterializationLimitsV1::default(),
        );
        assert!(matches!(
            result,
            Err(issues) if issues.iter().any(|issue| matches!(
                issue,
                MusicEditMaterializationIssueV1::DigestMismatch {
                    role: MusicEditMaterializedArtifactRoleV1::RightsEvidence,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn receipt_tampering_and_map_substitution_reject() {
        let root = TestRoot::new();
        let (manifest, artifact_map) = manifest_and_map(&root);
        let mut receipt = materialize_music_edit_dataset_v1(
            &root.0,
            &manifest,
            &artifact_map,
            MusicEditMaterializationLimitsV1::default(),
        )
        .unwrap();
        receipt.artifacts[0].byte_len += 1;
        assert!(
            verify_music_edit_materialization_receipt_v1(&receipt, &manifest, &artifact_map)
                .iter()
                .any(|issue| matches!(
                    issue,
                    MusicEditMaterializationIssueV1::ReceiptByteCountMismatch
                        | MusicEditMaterializationIssueV1::ReceiptDigestMismatch
                ))
        );

        let (manifest, artifact_map) = manifest_and_map(&root);
        let receipt = materialize_music_edit_dataset_v1(
            &root.0,
            &manifest,
            &artifact_map,
            MusicEditMaterializationLimitsV1::default(),
        )
        .unwrap();
        let mut changed_map = artifact_map.clone();
        changed_map.rows[0].instruction_path = "samples/one/metadata.json".to_owned();
        assert!(
            verify_music_edit_materialization_receipt_v1(&receipt, &manifest, &changed_map)
                .contains(&MusicEditMaterializationIssueV1::ReceiptArtifactMapMismatch)
        );
    }

    #[test]
    fn byte_identical_audio_cannot_cross_source_lineage() {
        let root = TestRoot::new();
        let (mut manifest, mut artifact_map) = manifest_and_map(&root);
        let duplicate_source = b"synthetic source audio bytes";
        let source_digest = format!("{:x}", Sha256::digest(duplicate_source));
        let lineage_two = format!("{:x}", Sha256::digest(b"lineage-two"));
        let target_two = b"another target audio payload";
        let target_two_digest = root.write("samples/two/target.wav", target_two);
        let instruction_two = b"remove the trumpet";
        let instruction_two_digest = root.write("samples/two/instruction.txt", instruction_two);
        let selector_two = b"{\"instrument\":\"trumpet\"}";
        let selector_two_digest = root.write("samples/two/selector.json", selector_two);
        let metadata_two = b"{\"synthetic\":true,\"row\":2}";
        let metadata_two_digest = root.write("samples/two/metadata.json", metadata_two);
        let symbolic_two = b"another symbolic score";
        let symbolic_two_digest = root.write("samples/two/source.mid", symbolic_two);
        root.write("samples/two/source.wav", duplicate_source);

        let first = manifest.rows[0].clone();
        manifest.rows.push(MusicEditDatasetRowV1 {
            sample_id: "sample-two".to_owned(),
            source_lineage_sha256: lineage_two,
            split: MusicEditSplitV1::Development,
            stratum: MusicEditDatasetStratumV1::MidiSynthesized,
            operation: MusicEditOperationV1::RemoveInstrument,
            target_scope: MusicEditTargetScopeV1::Instrument,
            source_audio_sha256: source_digest,
            instruction_sha256: instruction_two_digest,
            target_audio_sha256: target_two_digest,
            source_symbolic_sha256: Some(symbolic_two_digest),
            target_selector_sha256: selector_two_digest,
            metadata_sha256: metadata_two_digest,
        });
        artifact_map.rows.push(MusicEditArtifactPathsV1 {
            sample_id: "sample-two".to_owned(),
            source_audio_path: "samples/two/source.wav".to_owned(),
            instruction_path: "samples/two/instruction.txt".to_owned(),
            target_audio_path: "samples/two/target.wav".to_owned(),
            source_symbolic_path: Some("samples/two/source.mid".to_owned()),
            target_selector_path: "samples/two/selector.json".to_owned(),
            metadata_path: "samples/two/metadata.json".to_owned(),
        });
        assert_ne!(first.source_lineage_sha256, manifest.rows[1].source_lineage_sha256);
        let result = materialize_music_edit_dataset_v1(
            &root.0,
            &manifest,
            &artifact_map,
            MusicEditMaterializationLimitsV1::default(),
        );
        assert!(matches!(
            result,
            Err(issues) if issues.iter().any(|issue| matches!(
                issue,
                MusicEditMaterializationIssueV1::CrossLineageExactAudioDuplicate { .. }
            ))
        ));
    }

    #[test]
    fn size_limits_fail_closed() {
        let root = TestRoot::new();
        let (manifest, artifact_map) = manifest_and_map(&root);
        let result = materialize_music_edit_dataset_v1(
            &root.0,
            &manifest,
            &artifact_map,
            MusicEditMaterializationLimitsV1 {
                max_artifact_bytes: 4,
                max_total_bytes: 100,
            },
        );
        assert!(matches!(
            result,
            Err(issues) if issues.iter().any(|issue| matches!(
                issue,
                MusicEditMaterializationIssueV1::ArtifactTooLarge { .. }
            ))
        ));
    }
}
