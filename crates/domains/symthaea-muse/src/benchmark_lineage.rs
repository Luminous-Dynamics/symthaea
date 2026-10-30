// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed lineage auditing for paired-target music/audio benchmarks.
//!
//! A sample-level random split is not sufficient: alternate renders, stems,
//! MIDI exports, edits, and paired targets can leak the same composition across
//! train/validation/test. This module audits the declared identity graph before
//! an editor is run or an external score is reported. It does not prove that
//! upstream lineage labels are truthful; it only verifies the supplied claims.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

const MANIFEST_DOMAIN: &[u8] = b"luminous-dynamics-muse-benchmark-lineage-manifest-v1\0";
const AUDIT_DOMAIN: &[u8] = b"luminous-dynamics-muse-benchmark-lineage-audit-v1\0";
const AUDIT_SCHEMA_VERSION: &str = "muse-benchmark-lineage-audit-v1";

/// Dataset partition. Ordering is stable and used in canonical audit output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
}

impl DatasetSplit {
    fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

/// One paired source/instruction/target benchmark example.
///
/// All SHA-256 values are lowercase hexadecimal. The optional score digest is
/// absent for real-audio examples without a symbolic source. A composition
/// lineage digest is still mandatory for every modality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkLineageRecord {
    pub sample_id: String,
    pub split: DatasetSplit,
    pub operation_family: String,
    pub source_audio_sha256: String,
    pub target_audio_sha256: String,
    pub instruction_sha256: String,
    pub source_score_sha256: Option<String>,
    pub composition_lineage_sha256: String,
}

/// Versioned benchmark manifest audited before external scoring.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkLineageManifest {
    pub benchmark_id: String,
    pub dataset_revision: String,
    pub records: Vec<BenchmarkLineageRecord>,
}

/// Auditable failure class. Cross-split checks are independent by identity
/// domain: audio assets, symbolic sources, and composition lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LineageViolationKind {
    EmptyBenchmarkId,
    EmptyDatasetRevision,
    EmptySampleId,
    DuplicateSampleId,
    EmptyOperationFamily,
    InvalidSourceAudioDigest,
    InvalidTargetAudioDigest,
    InvalidInstructionDigest,
    InvalidSourceScoreDigest,
    InvalidCompositionLineageDigest,
    CrossSplitAudioAsset,
    CrossSplitSymbolicSource,
    CrossSplitCompositionLineage,
}

impl LineageViolationKind {
    fn code(self) -> &'static str {
        match self {
            Self::EmptyBenchmarkId => "empty_benchmark_id",
            Self::EmptyDatasetRevision => "empty_dataset_revision",
            Self::EmptySampleId => "empty_sample_id",
            Self::DuplicateSampleId => "duplicate_sample_id",
            Self::EmptyOperationFamily => "empty_operation_family",
            Self::InvalidSourceAudioDigest => "invalid_source_audio_digest",
            Self::InvalidTargetAudioDigest => "invalid_target_audio_digest",
            Self::InvalidInstructionDigest => "invalid_instruction_digest",
            Self::InvalidSourceScoreDigest => "invalid_source_score_digest",
            Self::InvalidCompositionLineageDigest => "invalid_composition_lineage_digest",
            Self::CrossSplitAudioAsset => "cross_split_audio_asset",
            Self::CrossSplitSymbolicSource => "cross_split_symbolic_source",
            Self::CrossSplitCompositionLineage => "cross_split_composition_lineage",
        }
    }
}

/// One deterministic violation, with only stable sample IDs and fingerprints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageViolation {
    pub kind: LineageViolationKind,
    pub identity_sha256: Option<String>,
    pub sample_ids: Vec<String>,
    pub splits: Vec<DatasetSplit>,
}

/// Replay-verifiable result of the benchmark lineage preflight.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkLineageAudit {
    pub schema_version: String,
    pub benchmark_id: String,
    pub dataset_revision: String,
    pub record_count: usize,
    pub unique_audio_asset_count: usize,
    pub unique_symbolic_source_count: usize,
    pub unique_composition_lineage_count: usize,
    pub manifest_sha256: String,
    pub violations: Vec<LineageViolation>,
    pub audit_sha256: String,
}

impl BenchmarkLineageAudit {
    /// True only when every required identity and split invariant is satisfied.
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }
}

type IdentityIndex = BTreeMap<String, BTreeMap<DatasetSplit, BTreeSet<String>>>;

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn add_identity(
    index: &mut IdentityIndex,
    digest: &str,
    split: DatasetSplit,
    sample_id: &str,
) {
    index
        .entry(digest.to_owned())
        .or_default()
        .entry(split)
        .or_default()
        .insert(sample_id.to_owned());
}

fn cross_split_violations(
    index: &IdentityIndex,
    kind: LineageViolationKind,
) -> Vec<LineageViolation> {
    index
        .iter()
        .filter_map(|(identity, by_split)| {
            if by_split.len() < 2 {
                return None;
            }
            let splits = by_split.keys().copied().collect();
            let sample_ids = by_split
                .values()
                .flat_map(|ids| ids.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            Some(LineageViolation {
                kind,
                identity_sha256: Some(identity.clone()),
                sample_ids,
                splits,
            })
        })
        .collect()
}

fn record_violation(
    violations: &mut Vec<LineageViolation>,
    kind: LineageViolationKind,
    record: &BenchmarkLineageRecord,
) {
    let sample_ids = if record.sample_id.is_empty() {
        Vec::new()
    } else {
        vec![record.sample_id.clone()]
    };
    violations.push(LineageViolation {
        kind,
        identity_sha256: None,
        sample_ids,
        splits: vec![record.split],
    });
}

fn compare_records(left: &BenchmarkLineageRecord, right: &BenchmarkLineageRecord) -> Ordering {
    (
        left.sample_id.as_str(),
        left.split,
        left.operation_family.as_str(),
        left.source_audio_sha256.as_str(),
        left.target_audio_sha256.as_str(),
        left.instruction_sha256.as_str(),
        left.source_score_sha256.as_deref(),
        left.composition_lineage_sha256.as_str(),
    )
        .cmp(&(
            right.sample_id.as_str(),
            right.split,
            right.operation_family.as_str(),
            right.source_audio_sha256.as_str(),
            right.target_audio_sha256.as_str(),
            right.instruction_sha256.as_str(),
            right.source_score_sha256.as_deref(),
            right.composition_lineage_sha256.as_str(),
        ))
}

fn hash_component(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_optional_component(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1_u8]);
            hash_component(hasher, value.as_bytes());
        }
        None => hasher.update([0_u8]),
    }
}

fn manifest_digest(manifest: &BenchmarkLineageManifest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(MANIFEST_DOMAIN);
    hash_component(&mut hasher, manifest.benchmark_id.as_bytes());
    hash_component(&mut hasher, manifest.dataset_revision.as_bytes());

    let mut records: Vec<&BenchmarkLineageRecord> = manifest.records.iter().collect();
    records.sort_by(|left, right| compare_records(left, right));
    hasher.update((records.len() as u64).to_le_bytes());
    for record in records {
        hash_component(&mut hasher, record.sample_id.as_bytes());
        hash_component(&mut hasher, record.split.as_str().as_bytes());
        hash_component(&mut hasher, record.operation_family.as_bytes());
        hash_component(&mut hasher, record.source_audio_sha256.as_bytes());
        hash_component(&mut hasher, record.target_audio_sha256.as_bytes());
        hash_component(&mut hasher, record.instruction_sha256.as_bytes());
        hash_optional_component(&mut hasher, record.source_score_sha256.as_deref());
        hash_component(
            &mut hasher,
            record.composition_lineage_sha256.as_bytes(),
        );
    }
    format!("{:x}", hasher.finalize())
}

fn audit_digest(audit: &BenchmarkLineageAudit) -> String {
    let mut hasher = Sha256::new();
    hasher.update(AUDIT_DOMAIN);
    hash_component(&mut hasher, audit.schema_version.as_bytes());
    hash_component(&mut hasher, audit.benchmark_id.as_bytes());
    hash_component(&mut hasher, audit.dataset_revision.as_bytes());
    hasher.update((audit.record_count as u64).to_le_bytes());
    hasher.update((audit.unique_audio_asset_count as u64).to_le_bytes());
    hasher.update((audit.unique_symbolic_source_count as u64).to_le_bytes());
    hasher.update((audit.unique_composition_lineage_count as u64).to_le_bytes());
    hash_component(&mut hasher, audit.manifest_sha256.as_bytes());
    hasher.update((audit.violations.len() as u64).to_le_bytes());
    for violation in &audit.violations {
        hash_component(&mut hasher, violation.kind.code().as_bytes());
        hash_optional_component(&mut hasher, violation.identity_sha256.as_deref());
        hasher.update((violation.sample_ids.len() as u64).to_le_bytes());
        for sample_id in &violation.sample_ids {
            hash_component(&mut hasher, sample_id.as_bytes());
        }
        hasher.update((violation.splits.len() as u64).to_le_bytes());
        for split in &violation.splits {
            hash_component(&mut hasher, split.as_str().as_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Audit split isolation and return deterministic, content-addressed evidence.
///
/// Rows with invalid identifiers/digests remain part of the manifest
/// commitment and produce violations; they are never silently filtered out.
pub fn audit_benchmark_lineage(
    manifest: &BenchmarkLineageManifest,
) -> BenchmarkLineageAudit {
    let mut violations = Vec::new();
    if manifest.benchmark_id.trim().is_empty() {
        violations.push(LineageViolation {
            kind: LineageViolationKind::EmptyBenchmarkId,
            identity_sha256: None,
            sample_ids: Vec::new(),
            splits: Vec::new(),
        });
    }
    if manifest.dataset_revision.trim().is_empty() {
        violations.push(LineageViolation {
            kind: LineageViolationKind::EmptyDatasetRevision,
            identity_sha256: None,
            sample_ids: Vec::new(),
            splits: Vec::new(),
        });
    }

    let mut sample_ids: BTreeMap<String, (usize, BTreeSet<DatasetSplit>)> = BTreeMap::new();
    let mut audio_assets = IdentityIndex::new();
    let mut symbolic_sources = IdentityIndex::new();
    let mut composition_lineages = IdentityIndex::new();
    let mut unique_audio_assets = BTreeSet::new();
    let mut unique_symbolic_sources = BTreeSet::new();
    let mut unique_composition_lineages = BTreeSet::new();

    for record in &manifest.records {
        if record.sample_id.trim().is_empty() {
            record_violation(
                &mut violations,
                LineageViolationKind::EmptySampleId,
                record,
            );
        } else {
            let entry = sample_ids
                .entry(record.sample_id.clone())
                .or_insert_with(|| (0, BTreeSet::new()));
            entry.0 += 1;
            entry.1.insert(record.split);
        }

        if record.operation_family.trim().is_empty() {
            record_violation(
                &mut violations,
                LineageViolationKind::EmptyOperationFamily,
                record,
            );
        }

        if is_sha256(&record.source_audio_sha256) {
            unique_audio_assets.insert(record.source_audio_sha256.clone());
            add_identity(
                &mut audio_assets,
                &record.source_audio_sha256,
                record.split,
                &record.sample_id,
            );
        } else {
            record_violation(
                &mut violations,
                LineageViolationKind::InvalidSourceAudioDigest,
                record,
            );
        }

        if is_sha256(&record.target_audio_sha256) {
            unique_audio_assets.insert(record.target_audio_sha256.clone());
            add_identity(
                &mut audio_assets,
                &record.target_audio_sha256,
                record.split,
                &record.sample_id,
            );
        } else {
            record_violation(
                &mut violations,
                LineageViolationKind::InvalidTargetAudioDigest,
                record,
            );
        }

        if is_sha256(&record.instruction_sha256) {
            // The instruction digest is committed by the manifest. It is not
            // a split key because the same instruction template is legitimate.
        } else {
            record_violation(
                &mut violations,
                LineageViolationKind::InvalidInstructionDigest,
                record,
            );
        }

        if let Some(score_digest) = record.source_score_sha256.as_deref() {
            if is_sha256(score_digest) {
                unique_symbolic_sources.insert(score_digest.to_owned());
                add_identity(
                    &mut symbolic_sources,
                    score_digest,
                    record.split,
                    &record.sample_id,
                );
            } else {
                record_violation(
                    &mut violations,
                    LineageViolationKind::InvalidSourceScoreDigest,
                    record,
                );
            }
        }

        if is_sha256(&record.composition_lineage_sha256) {
            unique_composition_lineages.insert(record.composition_lineage_sha256.clone());
            add_identity(
                &mut composition_lineages,
                &record.composition_lineage_sha256,
                record.split,
                &record.sample_id,
            );
        } else {
            record_violation(
                &mut violations,
                LineageViolationKind::InvalidCompositionLineageDigest,
                record,
            );
        }
    }

    for (sample_id, (count, splits)) in sample_ids {
        if count > 1 {
            violations.push(LineageViolation {
                kind: LineageViolationKind::DuplicateSampleId,
                identity_sha256: None,
                sample_ids: vec![sample_id],
                splits: splits.into_iter().collect(),
            });
        }
    }

    violations.extend(cross_split_violations(
        &audio_assets,
        LineageViolationKind::CrossSplitAudioAsset,
    ));
    violations.extend(cross_split_violations(
        &symbolic_sources,
        LineageViolationKind::CrossSplitSymbolicSource,
    ));
    violations.extend(cross_split_violations(
        &composition_lineages,
        LineageViolationKind::CrossSplitCompositionLineage,
    ));

    violations.sort_by(|left, right| {
        (
            left.kind,
            left.identity_sha256.as_deref().unwrap_or(""),
            &left.sample_ids,
            &left.splits,
        )
            .cmp(&(
                right.kind,
                right.identity_sha256.as_deref().unwrap_or(""),
                &right.sample_ids,
                &right.splits,
            ))
    });

    let mut audit = BenchmarkLineageAudit {
        schema_version: AUDIT_SCHEMA_VERSION.to_owned(),
        benchmark_id: manifest.benchmark_id.clone(),
        dataset_revision: manifest.dataset_revision.clone(),
        record_count: manifest.records.len(),
        unique_audio_asset_count: unique_audio_assets.len(),
        unique_symbolic_source_count: unique_symbolic_sources.len(),
        unique_composition_lineage_count: unique_composition_lineages.len(),
        manifest_sha256: manifest_digest(manifest),
        violations,
        audit_sha256: String::new(),
    };
    audit.audit_sha256 = audit_digest(&audit);
    audit
}

/// Recompute the entire audit; any changed record, count, violation, or digest
/// makes verification fail.
pub fn verify_benchmark_lineage_audit(
    manifest: &BenchmarkLineageManifest,
    supplied: &BenchmarkLineageAudit,
) -> bool {
    audit_benchmark_lineage(manifest) == *supplied
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String {
        std::iter::repeat_n(format!("{byte:02x}"), 32).collect()
    }

    fn record(
        sample_id: &str,
        split: DatasetSplit,
        source_audio: u8,
        target_audio: u8,
        source_score: Option<u8>,
        lineage: u8,
    ) -> BenchmarkLineageRecord {
        BenchmarkLineageRecord {
            sample_id: sample_id.to_owned(),
            split,
            operation_family: "remove_instrument".to_owned(),
            source_audio_sha256: digest(source_audio),
            target_audio_sha256: digest(target_audio),
            instruction_sha256: digest(90),
            source_score_sha256: source_score.map(digest),
            composition_lineage_sha256: digest(lineage),
        }
    }

    fn manifest(records: Vec<BenchmarkLineageRecord>) -> BenchmarkLineageManifest {
        BenchmarkLineageManifest {
            benchmark_id: "smebench".to_owned(),
            dataset_revision: "test-revision-1".to_owned(),
            records,
        }
    }

    #[test]
    fn repeated_source_within_one_split_is_valid() {
        let audit = audit_benchmark_lineage(&manifest(vec![
            record("train-1", DatasetSplit::Train, 1, 2, Some(11), 21),
            record("train-2", DatasetSplit::Train, 1, 3, Some(11), 21),
        ]));
        assert!(audit.is_clean(), "{:?}", audit.violations);
        assert_eq!(audit.record_count, 2);
        assert_eq!(audit.unique_audio_asset_count, 3);
        assert_eq!(audit.unique_symbolic_source_count, 1);
    }

    #[test]
    fn source_audio_reused_as_target_across_splits_is_rejected() {
        let audit = audit_benchmark_lineage(&manifest(vec![
            record("train-1", DatasetSplit::Train, 1, 2, None, 21),
            record("test-1", DatasetSplit::Test, 3, 1, None, 31),
        ]));
        assert!(audit.violations.iter().any(|violation| {
            violation.kind == LineageViolationKind::CrossSplitAudioAsset
                && violation.identity_sha256.as_deref() == Some(digest(1).as_str())
        }));
    }

    #[test]
    fn symbolic_source_fingerprint_cannot_cross_splits() {
        let audit = audit_benchmark_lineage(&manifest(vec![
            record("train-1", DatasetSplit::Train, 1, 2, Some(44), 21),
            record("valid-1", DatasetSplit::Validation, 3, 4, Some(44), 31),
        ]));
        assert!(audit.violations.iter().any(|violation| {
            violation.kind == LineageViolationKind::CrossSplitSymbolicSource
                && violation.identity_sha256.as_deref() == Some(digest(44).as_str())
        }));
    }

    #[test]
    fn composition_lineage_cannot_cross_splits_with_different_renders() {
        let audit = audit_benchmark_lineage(&manifest(vec![
            record("train-1", DatasetSplit::Train, 1, 2, None, 55),
            record("test-1", DatasetSplit::Test, 3, 4, None, 55),
        ]));
        assert!(audit.violations.iter().any(|violation| {
            violation.kind == LineageViolationKind::CrossSplitCompositionLineage
                && violation.identity_sha256.as_deref() == Some(digest(55).as_str())
        }));
    }

    #[test]
    fn duplicate_ids_and_invalid_required_digests_are_reported() {
        let mut invalid = record("same-id", DatasetSplit::Train, 1, 2, None, 21);
        invalid.instruction_sha256 = "not-a-digest".to_owned();
        let mut duplicate = record("same-id", DatasetSplit::Test, 3, 4, None, 31);
        duplicate.target_audio_sha256.clear();
        let audit = audit_benchmark_lineage(&manifest(vec![invalid, duplicate]));
        assert!(audit.violations.iter().any(|v| v.kind == LineageViolationKind::DuplicateSampleId));
        assert!(audit.violations.iter().any(|v| v.kind == LineageViolationKind::InvalidInstructionDigest));
        assert!(audit.violations.iter().any(|v| v.kind == LineageViolationKind::InvalidTargetAudioDigest));
        assert!(!audit.is_clean());
    }

    #[test]
    fn row_order_does_not_change_manifest_or_audit_commitments() {
        let rows = vec![
            record("train-b", DatasetSplit::Train, 1, 2, Some(11), 21),
            record("test-a", DatasetSplit::Test, 3, 4, Some(22), 31),
        ];
        let forward = audit_benchmark_lineage(&manifest(rows.clone()));
        let reverse = audit_benchmark_lineage(&manifest(rows.into_iter().rev().collect()));
        assert_eq!(forward, reverse);
    }

    #[test]
    fn replay_verification_rejects_a_tampered_audit() {
        let input = manifest(vec![record("train-1", DatasetSplit::Train, 1, 2, None, 21)]);
        let mut audit = audit_benchmark_lineage(&input);
        assert!(verify_benchmark_lineage_audit(&input, &audit));
        audit.unique_audio_asset_count += 1;
        assert!(!verify_benchmark_lineage_audit(&input, &audit));
    }
}
