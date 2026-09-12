// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Matbench v0.1 positional fold binding for experimental-gap Benchmark Zero.
//!
//! This crate accepts only the exact compressed `matbench_expt_gap` artifact
//! pinned by `symthaea-matbench-gap`, reproduces Matbench v0.1's published
//! regression split procedure over dataframe row positions, and removes
//! compositions present in Symthaea's currently exposed band-gap training table
//! before producing a Benchmark Zero truth slice.
//!
//! A leakage-clean slice is not an official Matbench leaderboard test set.

#![forbid(unsafe_code)]

mod legacy_randomstate;

use legacy_randomstate::shuffled_indices;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use symthaea_energy_benchmark_zero::{BandgapTruthSet, DataSliceId, DatasetProvenance};
use symthaea_matbench_gap::{
    parse_official_matbench_expt_gap, AdapterError, MatbenchGapDataset,
    MATBENCH_EXPT_GAP_DATASET_ID, MATBENCH_EXPT_GAP_LICENSE_DISCLOSURE,
    MATBENCH_EXPT_GAP_ROWS, MATBENCH_EXPT_GAP_SHA256, MATBENCH_EXPT_GAP_URL,
};
use thiserror::Error;

pub const MATBENCH_V01_VALIDATION_COMMIT: &str =
    "936176db18ca4cd7b38cbd957c017a5bac770c6b";
pub const MATBENCH_V01_VALIDATION_BLOB_SHA1: &str =
    "ca9d0d157b31cadb7d99ba7c404cd96cfa09cad9";
pub const MATBENCH_V01_N_SPLITS: u8 = 5;
pub const MATBENCH_V01_RANDOM_STATE: u32 = 18_012_019;
pub const MATBENCH_V01_SHUFFLE: bool = true;
pub const MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256: &str =
    "03a37eb4876e836878507c09559fefc55b1ff5f08db0c2229ac7dd80c0bffd7c";

pub const FOLD_DERIVATION_DISCLOSURE: &str =
    "Fold assignments are reproduced over dataframe row positions from the published Matbench v0.1 regression KFold procedure and pinned to the upstream validation commit/blob identity. Direct byte-for-byte extraction from the 46 MB validation JSON is a separate audit gate.";

pub const INDEX_IDENTITY_DISCLOSURE: &str =
    "This crate deliberately does not synthesize official mb-expt-gap-* IDs. Matbench derives those labels from the dataframe source index, while KFold partitions by row position. Exact source-index-to-Matbench-ID parity remains a separate artifact audit.";

pub const RESIDUAL_LEAKAGE_DISCLOSURE: &str =
    "Composition overlap with Symthaea's currently exposed band-gap training table was removed. This does not prove historical blindness: hand-written baselines or prior model choices may still reflect public semiconductor knowledge.";

const MANIFEST_DIGEST_DOMAIN: &[u8] = b"symthaea.matbench-expt-gap.fold-manifest.v0\0";
const TRAINING_DIGEST_DOMAIN: &[u8] = b"symthaea.bandgap.training-table.v0\0";
const TRUTH_SLICE_DIGEST_DOMAIN: &[u8] = b"symthaea.matbench-expt-gap.fold-truth.v0\0";
const OVERLAP_MASK_DIGEST_DOMAIN: &[u8] = b"symthaea.matbench-expt-gap.overlap-mask.v0\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.matbench-expt-gap.leakage-qualification.v0\0";

fn derive_fold_by_position() -> [u8; MATBENCH_EXPT_GAP_ROWS] {
    let permutation = shuffled_indices::<MATBENCH_EXPT_GAP_ROWS>(MATBENCH_V01_RANDOM_STATE);

    let mut fold_by_position = [u8::MAX; MATBENCH_EXPT_GAP_ROWS];
    let base_size = MATBENCH_EXPT_GAP_ROWS / usize::from(MATBENCH_V01_N_SPLITS);
    let larger_fold_count = MATBENCH_EXPT_GAP_ROWS % usize::from(MATBENCH_V01_N_SPLITS);
    let mut cursor = 0usize;

    for fold in 0..MATBENCH_V01_N_SPLITS {
        let fold_size = base_size + usize::from(usize::from(fold) < larger_fold_count);
        let stop = cursor + fold_size;
        for &row_position in &permutation[cursor..stop] {
            fold_by_position[row_position] = fold;
        }
        cursor = stop;
    }

    debug_assert_eq!(cursor, MATBENCH_EXPT_GAP_ROWS);
    debug_assert!(
        fold_by_position
            .iter()
            .all(|&fold| fold < MATBENCH_V01_N_SPLITS)
    );
    fold_by_position
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FoldIndex(u8);

impl FoldIndex {
    pub fn new(value: u8) -> Result<Self, FoldError> {
        if value < MATBENCH_V01_N_SPLITS {
            Ok(Self(value))
        } else {
            Err(FoldError::InvalidFold(value))
        }
    }

    pub const fn value(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeakageExclusion {
    /// Zero-based position in the exact pinned dataframe artifact.
    pub row_position: usize,
    pub candidate_id: String,
    pub symthaea_training_labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LeakageCleanFold {
    pub fold: FoldIndex,
    pub canonical_test_count: usize,
    pub retained_test_count: usize,
    pub excluded_training_overlap: Vec<LeakageExclusion>,
    /// SHA-256 of the exact official compressed source artifact accepted by the
    /// parent adapter.
    pub source_artifact_sha256: String,
    /// Content identity of the complete Symthaea band-gap training table used
    /// to decide overlap/non-overlap for this audit.
    pub symthaea_training_table_sha256: String,
    pub symthaea_training_entry_count: usize,
    pub fold_manifest_sha256: String,
    pub overlap_mask_sha256: String,
    pub truth_slice_sha256: String,
    /// Single durable identity binding source, fold, training table, exclusions,
    /// and retained truth for downstream benchmark receipts.
    pub qualification_sha256: String,
    pub fold_derivation_disclosure: &'static str,
    pub index_identity_disclosure: &'static str,
    pub residual_leakage_disclosure: &'static str,
    pub truth: BandgapTruthSet,
}

#[derive(Debug, Error)]
pub enum FoldError {
    #[error("fold index {0} is outside canonical Matbench v0.1 range 0..4")]
    InvalidFold(u8),
    #[error("row position {0} is outside matbench_expt_gap range")]
    RowPositionOutOfRange(usize),
    #[error("fold manifest invariant failed: {0}")]
    ManifestInvariant(String),
    #[error("leakage cleaning removed every candidate from fold {0}")]
    EmptyLeakageCleanFold(u8),
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error(transparent)]
    Benchmark(#[from] symthaea_energy_benchmark_zero::BenchmarkError),
}

/// Return the published-procedure test fold for one zero-based dataframe row
/// position. This is intentionally positional and does not claim an official
/// Matbench ID for the row.
pub fn fold_for_row_position(row_position: usize) -> Result<FoldIndex, FoldError> {
    if row_position >= MATBENCH_EXPT_GAP_ROWS {
        return Err(FoldError::RowPositionOutOfRange(row_position));
    }
    FoldIndex::new(derive_fold_by_position()[row_position])
}

pub fn canonical_fold_counts() -> [usize; MATBENCH_V01_N_SPLITS as usize] {
    let mut counts = [0usize; MATBENCH_V01_N_SPLITS as usize];
    for fold in derive_fold_by_position() {
        counts[usize::from(fold)] += 1;
    }
    counts
}

pub fn canonical_fold_manifest_sha256() -> String {
    let folds = derive_fold_by_position();
    let mut hasher = Sha256::new();
    hasher.update(MANIFEST_DIGEST_DOMAIN);
    hasher.update(MATBENCH_EXPT_GAP_DATASET_ID.as_bytes());
    hasher.update([0]);
    hasher.update(MATBENCH_V01_VALIDATION_COMMIT.as_bytes());
    hasher.update([0]);
    hasher.update(MATBENCH_V01_VALIDATION_BLOB_SHA1.as_bytes());
    hasher.update([0]);
    hasher.update([MATBENCH_V01_N_SPLITS]);
    hasher.update(MATBENCH_V01_RANDOM_STATE.to_le_bytes());
    hasher.update([u8::from(MATBENCH_V01_SHUFFLE)]);
    hasher.update(folds);
    let digest = hasher.finalize();
    hex_lower(&digest)
}

/// Content identity of the complete curated training table currently exposed by
/// `symthaea-bandgap`.
///
/// This digest is intentionally broader than the overlap list. A training-table
/// edit changes leakage qualification identity even if it happens not to add or
/// remove an overlapping Matbench composition.
pub fn symthaea_training_table_sha256() -> String {
    let training = symthaea_bandgap::training_data::load_training_data();
    let mut hasher = Sha256::new();
    hasher.update(TRAINING_DIGEST_DOMAIN);
    hasher.update((training.len() as u64).to_le_bytes());

    for entry in training {
        update_text(&mut hasher, entry.formula);
        let mut composition = entry.composition;
        composition.sort_by_key(|(atomic_number, _)| *atomic_number);
        hasher.update((composition.len() as u64).to_le_bytes());
        for (atomic_number, fraction) in composition {
            hasher.update([atomic_number]);
            hasher.update(fraction.to_bits().to_le_bytes());
        }
        hasher.update(entry.experimental_gap.to_bits().to_le_bytes());
        hasher.update([entry.crystal_system.ordinal()]);
    }

    let digest = hasher.finalize();
    hex_lower(&digest)
}

pub fn validate_canonical_manifest() -> Result<(), FoldError> {
    let counts = canonical_fold_counts();
    let expected = [921usize, 921, 921, 921, 920];
    if counts != expected {
        return Err(FoldError::ManifestInvariant(format!(
            "expected fold counts {expected:?}, got {counts:?}"
        )));
    }

    let actual = canonical_fold_manifest_sha256();
    if actual != MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256 {
        return Err(FoldError::ManifestInvariant(format!(
            "fold digest mismatch: expected {}, got {actual}",
            MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256
        )));
    }
    Ok(())
}

/// Parse the exact pinned truth artifact, select one published-procedure test
/// fold by dataframe row position, remove compositions overlapping Symthaea's
/// currently exposed band-gap training table, and return a content-addressed
/// Benchmark Zero truth slice.
///
/// No Matbench train-fold rows are used for fitting here.
pub fn build_leakage_clean_test_fold_from_official_bytes(
    compressed_bytes: &[u8],
    fold: FoldIndex,
) -> Result<LeakageCleanFold, FoldError> {
    validate_canonical_manifest()?;
    let dataset = parse_official_matbench_expt_gap(compressed_bytes)?;
    build_leakage_clean_test_fold(&dataset, fold)
}

fn build_leakage_clean_test_fold(
    dataset: &MatbenchGapDataset,
    fold: FoldIndex,
) -> Result<LeakageCleanFold, FoldError> {
    // Private helper: public callers must enter through exact-byte parsing.
    if !dataset
        .compressed_sha256
        .eq_ignore_ascii_case(MATBENCH_EXPT_GAP_SHA256)
        || dataset.ordered_records.len() != MATBENCH_EXPT_GAP_ROWS
    {
        return Err(FoldError::ManifestInvariant(
            "parsed dataset does not match the exact official artifact identity".into(),
        ));
    }

    let training = symthaea_bandgap::training_data::load_training_data();
    let training_entry_count = training.len();
    // Re-load through the canonical digest function so both the digest and the
    // parent's overlap detector bind to the same public training-table API.
    let training_table_sha256 = symthaea_training_table_sha256();

    let overlap = dataset.symthaea_training_overlap()?;
    let mut overlap_by_candidate: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for item in overlap {
        overlap_by_candidate.insert(item.candidate_id, item.symthaea_training_labels);
    }

    let fold_by_position = derive_fold_by_position();
    let mut retained = BTreeMap::new();
    let mut retained_rows = Vec::new();
    let mut exclusions = Vec::new();
    let mut canonical_test_count = 0usize;

    for (row_position, record) in dataset.ordered_records.iter().enumerate() {
        if fold_by_position[row_position] != fold.value() {
            continue;
        }
        canonical_test_count += 1;

        if let Some(labels) = overlap_by_candidate.get(&record.candidate_id) {
            exclusions.push(LeakageExclusion {
                row_position,
                candidate_id: record.candidate_id.clone(),
                symthaea_training_labels: labels.clone(),
            });
            continue;
        }

        if retained
            .insert(record.candidate_id.clone(), record.experimental_gap_ev)
            .is_some()
        {
            return Err(FoldError::ManifestInvariant(format!(
                "duplicate retained candidate {:?}",
                record.candidate_id
            )));
        }
        retained_rows.push((row_position, record));
    }

    if retained.is_empty() {
        return Err(FoldError::EmptyLeakageCleanFold(fold.value()));
    }

    let overlap_mask_sha256 =
        digest_overlap_mask(fold, &training_table_sha256, &exclusions);
    let truth_slice_sha256 = digest_truth_slice(
        fold,
        &dataset.row_order_sha256,
        &training_table_sha256,
        &overlap_mask_sha256,
        &retained_rows,
    );
    let retained_test_count = retained.len();
    let qualification_sha256 = digest_qualification(
        fold,
        &dataset.row_order_sha256,
        &training_table_sha256,
        training_entry_count,
        &overlap_mask_sha256,
        &truth_slice_sha256,
        canonical_test_count,
        retained_test_count,
    );

    let split_id = format!(
        "matbench_v0.1/fold_{}/test/leakage-clean-positional@qualsha256:{}",
        fold.value(), qualification_sha256,
    );

    let truth = BandgapTruthSet {
        provenance: DatasetProvenance {
            slice: DataSliceId::new(MATBENCH_EXPT_GAP_DATASET_ID, split_id)?,
            source_uri: MATBENCH_EXPT_GAP_URL.to_owned(),
            content_digest: format!("sha256:{truth_slice_sha256}"),
            license: MATBENCH_EXPT_GAP_LICENSE_DISCLOSURE.to_owned(),
        },
        experimental_gap_ev: retained,
    };
    truth.validate()?;

    Ok(LeakageCleanFold {
        fold,
        canonical_test_count,
        retained_test_count,
        excluded_training_overlap: exclusions,
        source_artifact_sha256: MATBENCH_EXPT_GAP_SHA256.to_owned(),
        symthaea_training_table_sha256: training_table_sha256,
        symthaea_training_entry_count: training_entry_count,
        fold_manifest_sha256: MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256.to_owned(),
        overlap_mask_sha256,
        truth_slice_sha256,
        qualification_sha256,
        fold_derivation_disclosure: FOLD_DERIVATION_DISCLOSURE,
        index_identity_disclosure: INDEX_IDENTITY_DISCLOSURE,
        residual_leakage_disclosure: RESIDUAL_LEAKAGE_DISCLOSURE,
        truth,
    })
}

fn digest_overlap_mask(
    fold: FoldIndex,
    training_table_sha256: &str,
    exclusions: &[LeakageExclusion],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(OVERLAP_MASK_DIGEST_DOMAIN);
    hasher.update([fold.value()]);
    update_text(&mut hasher, MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256);
    update_text(&mut hasher, training_table_sha256);
    hasher.update((exclusions.len() as u64).to_le_bytes());

    for exclusion in exclusions {
        hasher.update((exclusion.row_position as u64).to_le_bytes());
        update_text(&mut hasher, &exclusion.candidate_id);
        hasher.update((exclusion.symthaea_training_labels.len() as u64).to_le_bytes());
        for label in &exclusion.symthaea_training_labels {
            update_text(&mut hasher, label);
        }
    }

    let digest = hasher.finalize();
    hex_lower(&digest)
}

fn digest_truth_slice(
    fold: FoldIndex,
    parent_row_order_sha256: &str,
    training_table_sha256: &str,
    overlap_mask_sha256: &str,
    retained_rows: &[(usize, &symthaea_matbench_gap::MatbenchGapRecord)],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(TRUTH_SLICE_DIGEST_DOMAIN);
    hasher.update([fold.value()]);
    update_text(&mut hasher, MATBENCH_EXPT_GAP_SHA256);
    update_text(&mut hasher, parent_row_order_sha256);
    update_text(&mut hasher, MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256);
    update_text(&mut hasher, training_table_sha256);
    update_text(&mut hasher, overlap_mask_sha256);
    hasher.update((retained_rows.len() as u64).to_le_bytes());

    for (row_position, record) in retained_rows {
        hasher.update((*row_position as u64).to_le_bytes());
        update_text(&mut hasher, &record.candidate_id);
        hasher.update(record.experimental_gap_ev.to_bits().to_le_bytes());
    }

    let digest = hasher.finalize();
    hex_lower(&digest)
}

#[allow(clippy::too_many_arguments)]
fn digest_qualification(
    fold: FoldIndex,
    parent_row_order_sha256: &str,
    training_table_sha256: &str,
    training_entry_count: usize,
    overlap_mask_sha256: &str,
    truth_slice_sha256: &str,
    canonical_test_count: usize,
    retained_test_count: usize,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    hasher.update([fold.value()]);
    update_text(&mut hasher, MATBENCH_EXPT_GAP_SHA256);
    update_text(&mut hasher, parent_row_order_sha256);
    update_text(&mut hasher, MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256);
    update_text(&mut hasher, training_table_sha256);
    hasher.update((training_entry_count as u64).to_le_bytes());
    update_text(&mut hasher, overlap_mask_sha256);
    update_text(&mut hasher, truth_slice_sha256);
    hasher.update((canonical_test_count as u64).to_le_bytes());
    hasher.update((retained_test_count as u64).to_le_bytes());
    let digest = hasher.finalize();
    hex_lower(&digest)
}

fn update_text(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_randomstate::shuffled_indices;

    #[test]
    fn legacy_randomstate_shuffle_matches_frozen_prefix() {
        let permutation =
            shuffled_indices::<MATBENCH_EXPT_GAP_ROWS>(MATBENCH_V01_RANDOM_STATE);

        assert_eq!(
            &permutation[..20],
            &[
                3185, 3147, 3551, 2181, 3039, 3976, 1579, 3937, 2128, 2710, 3923, 587,
                1345, 3946, 1449, 3432, 2770, 693, 2931, 1275,
            ]
        );
    }

    #[test]
    fn manifest_digest_and_fold_sizes_are_frozen() {
        validate_canonical_manifest().expect("frozen manifest must validate");
        assert_eq!(
            canonical_fold_manifest_sha256(),
            MATBENCH_EXPT_GAP_FOLD_MANIFEST_SHA256
        );
        assert_eq!(canonical_fold_counts(), [921, 921, 921, 921, 920]);
    }

    #[test]
    fn positional_fold_anchors_are_stable() {
        assert_eq!(fold_for_row_position(7).unwrap().value(), 0);
        assert_eq!(fold_for_row_position(2).unwrap().value(), 1);
        assert_eq!(fold_for_row_position(0).unwrap().value(), 2);
        assert_eq!(fold_for_row_position(1).unwrap().value(), 3);
        assert_eq!(fold_for_row_position(4).unwrap().value(), 4);
        assert!(fold_for_row_position(MATBENCH_EXPT_GAP_ROWS).is_err());
    }

    #[test]
    fn training_table_identity_is_deterministic() {
        let first = symthaea_training_table_sha256();
        let second = symthaea_training_table_sha256();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn official_id_synthesis_is_intentionally_absent() {
        assert!(INDEX_IDENTITY_DISCLOSURE.contains("does not synthesize official"));
        assert!(INDEX_IDENTITY_DISCLOSURE.contains("source index"));
    }
}
