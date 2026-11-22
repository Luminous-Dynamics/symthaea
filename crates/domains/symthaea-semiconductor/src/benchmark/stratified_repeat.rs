// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic repeated stratified sample splits for published-protocol
//! comparability. These splits do not establish temporal or group independence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const INPUT_SCHEMA_V1: &str = "luminous.semi-bench.stratified-repeat-subject.v1";
pub const MANIFEST_SCHEMA_V1: &str = "luminous.semi-bench.stratified-repeat-manifest.v1";
pub const REPORT_SCHEMA_V1: &str = "luminous.semi-bench.stratified-repeat-report.v1";
pub const PRNG_ID_V1: &str = "splitmix64-fisher-yates-v1";

const PASS_SEED_SALT: u64 = 0xA076_1D64_78BD_642F;
const FAIL_SEED_SALT: u64 = 0xE703_7ED1_A0B4_28DB;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StratifiedSampleV1 {
    pub sample_id: String,
    pub positive: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StratifiedSplitSubjectV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_snapshot_sha256: String,
    pub source_identity_manifest_sha256: String,
    pub test_fraction: f64,
    pub seeds: Vec<u64>,
    pub seed_provenance: String,
    pub samples: Vec<StratifiedSampleV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StratifiedSplitAssignmentV1 {
    pub sample_id: String,
    pub partition: String,
    pub positive: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StratifiedSplitCountsV1 {
    pub samples: usize,
    pub positive: usize,
    pub negative: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StratifiedRepeatManifestV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_snapshot_sha256: String,
    pub source_identity_manifest_sha256: String,
    pub split_scheme: String,
    pub repeat_index: usize,
    pub seed: u64,
    pub random_algorithm: String,
    pub test_fraction: f64,
    pub seed_provenance: String,
    pub assignments: Vec<StratifiedSplitAssignmentV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StratifiedRepeatReportV1 {
    pub repeat_index: usize,
    pub seed: u64,
    pub split_manifest_sha256: String,
    pub train: StratifiedSplitCountsV1,
    pub test: StratifiedSplitCountsV1,
    pub positive_test_fraction: f64,
    pub negative_test_fraction: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StratifiedRepeatClaimLimitsV1 {
    pub stratification_preserves_class_proportions_per_repeat: bool,
    pub sample_identity_is_canonicalized_before_shuffle: bool,
    pub source_row_order_is_not_used_for_partition_assignment: bool,
    pub temporal_independence_not_established: bool,
    pub group_or_lot_independence_not_established: bool,
    pub seeds_are_reconstruction_not_authors_original_seeds: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StratifiedRepeatReportV1 {
    pub schema: String,
    pub dataset_id: String,
    pub dataset_snapshot_sha256: String,
    pub source_identity_manifest_sha256: String,
    pub prng: String,
    pub test_fraction: f64,
    pub repeat_count: usize,
    pub total_samples: usize,
    pub total_positive: usize,
    pub total_negative: usize,
    pub repeats: Vec<StratifiedRepeatReportV1>,
    pub manifests: Vec<StratifiedRepeatManifestV1>,
    pub claim_limits: StratifiedRepeatClaimLimitsV1,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct StratifiedSplitError(pub String);

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper_exclusive: usize) -> usize {
        let bound = upper_exclusive as u64;
        let rejection_threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= rejection_threshold {
                return (value % bound) as usize;
            }
        }
    }
}

fn require_text(value: &str, field: &str) -> Result<(), StratifiedSplitError> {
    if value.trim().is_empty() {
        Err(StratifiedSplitError(format!(
            "{field}: expected non-empty string"
        )))
    } else {
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_subject(subject: &StratifiedSplitSubjectV1) -> Result<(), StratifiedSplitError> {
    if subject.schema != INPUT_SCHEMA_V1 {
        return Err(StratifiedSplitError(
            "schema: unsupported or missing stratified split subject".into(),
        ));
    }
    require_text(&subject.dataset_id, "dataset_id")?;
    if !valid_sha256(&subject.dataset_snapshot_sha256) {
        return Err(StratifiedSplitError(
            "dataset_snapshot_sha256: expected lowercase SHA-256 hex".into(),
        ));
    }
    if !valid_sha256(&subject.source_identity_manifest_sha256) {
        return Err(StratifiedSplitError(
            "source_identity_manifest_sha256: expected lowercase SHA-256 hex".into(),
        ));
    }
    if !subject.test_fraction.is_finite()
        || subject.test_fraction <= 0.0
        || subject.test_fraction >= 0.5
    {
        return Err(StratifiedSplitError(
            "test_fraction: expected finite value in (0, 0.5)".into(),
        ));
    }
    require_text(&subject.seed_provenance, "seed_provenance")?;
    if subject.seeds.is_empty() {
        return Err(StratifiedSplitError(
            "seeds: expected non-empty array".into(),
        ));
    }
    let mut unique_seeds = BTreeSet::new();
    for seed in &subject.seeds {
        if !unique_seeds.insert(*seed) {
            return Err(StratifiedSplitError(format!(
                "seeds: duplicate seed {seed}"
            )));
        }
    }
    if subject.samples.len() < 4 {
        return Err(StratifiedSplitError(
            "samples: expected at least four rows".into(),
        ));
    }

    let mut seen_ids = BTreeSet::new();
    let mut counts = [0usize; 2];
    for (index, sample) in subject.samples.iter().enumerate() {
        require_text(&sample.sample_id, &format!("samples[{index}].sample_id"))?;
        if !seen_ids.insert(sample.sample_id.as_str()) {
            return Err(StratifiedSplitError(format!(
                "samples[{index}].sample_id: duplicate"
            )));
        }
        counts[if sample.positive { 1 } else { 0 }] += 1;
    }
    if counts.contains(&0) {
        return Err(StratifiedSplitError(
            "samples: both positive and negative classes are required".into(),
        ));
    }
    for (name, count) in [("negative", counts[0]), ("positive", counts[1])] {
        let test_count = (count as f64 * subject.test_fraction).round() as usize;
        if test_count == 0 || test_count >= count {
            return Err(StratifiedSplitError(format!(
                "{name}: test_fraction leaves an empty train or test class partition"
            )));
        }
    }
    Ok(())
}

fn shuffle<T>(values: &mut [T], rng: &mut SplitMix64) {
    for index in (1..values.len()).rev() {
        let other = rng.below(index + 1);
        values.swap(index, other);
    }
}

fn class_counts(
    assignments: &[StratifiedSplitAssignmentV1],
    partition: &str,
) -> StratifiedSplitCountsV1 {
    let selected: Vec<&StratifiedSplitAssignmentV1> = assignments
        .iter()
        .filter(|assignment| assignment.partition == partition)
        .collect();
    let positive = selected.iter().filter(|row| row.positive).count();
    StratifiedSplitCountsV1 {
        samples: selected.len(),
        positive,
        negative: selected.len() - positive,
    }
}

fn build_repeat(
    subject: &StratifiedSplitSubjectV1,
    repeat_index: usize,
    seed: u64,
) -> Result<(StratifiedRepeatManifestV1, StratifiedRepeatReportV1), StratifiedSplitError> {
    let mut negative_ids: Vec<String> = subject
        .samples
        .iter()
        .filter(|sample| !sample.positive)
        .map(|sample| sample.sample_id.clone())
        .collect();
    let mut positive_ids: Vec<String> = subject
        .samples
        .iter()
        .filter(|sample| sample.positive)
        .map(|sample| sample.sample_id.clone())
        .collect();
    negative_ids.sort();
    positive_ids.sort();

    let negative_test_count = (negative_ids.len() as f64 * subject.test_fraction).round() as usize;
    let positive_test_count = (positive_ids.len() as f64 * subject.test_fraction).round() as usize;

    let mut negative_rng = SplitMix64::new(seed ^ PASS_SEED_SALT);
    let mut positive_rng = SplitMix64::new(seed ^ FAIL_SEED_SALT);
    shuffle(&mut negative_ids, &mut negative_rng);
    shuffle(&mut positive_ids, &mut positive_rng);

    let mut partition_by_id = BTreeMap::new();
    for (index, sample_id) in negative_ids.iter().enumerate() {
        partition_by_id.insert(
            sample_id.clone(),
            if index < negative_test_count {
                "test"
            } else {
                "train"
            },
        );
    }
    for (index, sample_id) in positive_ids.iter().enumerate() {
        partition_by_id.insert(
            sample_id.clone(),
            if index < positive_test_count {
                "test"
            } else {
                "train"
            },
        );
    }

    let mut samples = subject.samples.clone();
    samples.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let assignments: Vec<StratifiedSplitAssignmentV1> = samples
        .iter()
        .map(|sample| StratifiedSplitAssignmentV1 {
            sample_id: sample.sample_id.clone(),
            partition: partition_by_id[sample.sample_id.as_str()].into(),
            positive: sample.positive,
        })
        .collect();

    let manifest = StratifiedRepeatManifestV1 {
        schema: MANIFEST_SCHEMA_V1.into(),
        dataset_id: subject.dataset_id.clone(),
        dataset_snapshot_sha256: subject.dataset_snapshot_sha256.clone(),
        source_identity_manifest_sha256: subject.source_identity_manifest_sha256.clone(),
        split_scheme: "stratified_sample_80_20".into(),
        repeat_index,
        seed,
        random_algorithm: PRNG_ID_V1.into(),
        test_fraction: subject.test_fraction,
        seed_provenance: subject.seed_provenance.clone(),
        assignments,
    };
    let serialized = serde_json::to_vec(&manifest)
        .map_err(|error| StratifiedSplitError(format!("manifest serialization failed: {error}")))?;
    let manifest_sha256 = sha256_hex(&serialized);
    let train = class_counts(&manifest.assignments, "train");
    let test = class_counts(&manifest.assignments, "test");
    let total_positive = manifest
        .assignments
        .iter()
        .filter(|row| row.positive)
        .count();
    let total_negative = manifest.assignments.len() - total_positive;
    let report = StratifiedRepeatReportV1 {
        repeat_index,
        seed,
        split_manifest_sha256: manifest_sha256,
        train,
        test: test.clone(),
        positive_test_fraction: rounded(test.positive as f64 / total_positive as f64),
        negative_test_fraction: rounded(test.negative as f64 / total_negative as f64),
    };
    Ok((manifest, report))
}

fn rounded(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

/// Builds deterministic class-stratified train/test manifests for each seed.
///
/// Input ordering is discarded and sample IDs are canonicalized before the
/// seeded per-class Fisher-Yates shuffle. The emitted seeds are explicit
/// protocol-reconstruction seeds, not an assertion about any paper's hidden
/// random state.
pub fn generate_repeated_stratified_splits_v1(
    subject: &StratifiedSplitSubjectV1,
) -> Result<StratifiedRepeatReportV1, StratifiedSplitError> {
    validate_subject(subject)?;
    let total_positive = subject
        .samples
        .iter()
        .filter(|sample| sample.positive)
        .count();
    let total_negative = subject.samples.len() - total_positive;
    let mut manifests = Vec::with_capacity(subject.seeds.len());
    let mut repeats = Vec::with_capacity(subject.seeds.len());

    for (index, seed) in subject.seeds.iter().copied().enumerate() {
        let (manifest, report) = build_repeat(subject, index + 1, seed)?;
        manifests.push(manifest);
        repeats.push(report);
    }

    Ok(StratifiedRepeatReportV1 {
        schema: REPORT_SCHEMA_V1.into(),
        dataset_id: subject.dataset_id.clone(),
        dataset_snapshot_sha256: subject.dataset_snapshot_sha256.clone(),
        source_identity_manifest_sha256: subject.source_identity_manifest_sha256.clone(),
        prng: PRNG_ID_V1.into(),
        test_fraction: subject.test_fraction,
        repeat_count: repeats.len(),
        total_samples: subject.samples.len(),
        total_positive,
        total_negative,
        repeats,
        manifests,
        claim_limits: StratifiedRepeatClaimLimitsV1 {
            stratification_preserves_class_proportions_per_repeat: true,
            sample_identity_is_canonicalized_before_shuffle: true,
            source_row_order_is_not_used_for_partition_assignment: true,
            temporal_independence_not_established: true,
            group_or_lot_independence_not_established: true,
            seeds_are_reconstruction_not_authors_original_seeds: true,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn subject(samples: Vec<StratifiedSampleV1>) -> StratifiedSplitSubjectV1 {
        StratifiedSplitSubjectV1 {
            schema: INPUT_SCHEMA_V1.into(),
            dataset_id: "SECOM-test".into(),
            dataset_snapshot_sha256: SHA.into(),
            source_identity_manifest_sha256: SHA.into(),
            test_fraction: 0.2,
            seeds: vec![42, 43, 44, 45, 46],
            seed_provenance: "Luminous reconstruction; not authors' original seeds".into(),
            samples,
        }
    }

    fn synthetic_samples() -> Vec<StratifiedSampleV1> {
        (0..100)
            .map(|index| StratifiedSampleV1 {
                sample_id: format!("sample-{index:04}"),
                positive: index < 10,
            })
            .collect()
    }

    #[test]
    fn five_repeats_preserve_expected_stratified_class_support() {
        let result = generate_repeated_stratified_splits_v1(&subject(synthetic_samples()))
            .expect("stratified repeat generation succeeds");
        assert_eq!(result.repeat_count, 5);
        assert_eq!(result.total_samples, 100);
        assert_eq!(result.total_positive, 10);
        for repeat in result.repeats {
            assert_eq!(repeat.train.samples, 80);
            assert_eq!(repeat.train.positive, 8);
            assert_eq!(repeat.test.samples, 20);
            assert_eq!(repeat.test.positive, 2);
            assert_eq!(repeat.test.negative, 18);
        }
    }

    #[test]
    fn sample_input_order_does_not_change_any_repeat_manifest() {
        let samples = synthetic_samples();
        let original = generate_repeated_stratified_splits_v1(&subject(samples.clone()))
            .expect("original split generation succeeds");
        let mut reversed = samples;
        reversed.reverse();
        let permuted = generate_repeated_stratified_splits_v1(&subject(reversed))
            .expect("permuted split generation succeeds");
        assert_eq!(original, permuted);
    }

    #[test]
    fn different_seeds_produce_distinct_repeat_manifests() {
        let result = generate_repeated_stratified_splits_v1(&subject(synthetic_samples()))
            .expect("split generation succeeds");
        let hashes: BTreeSet<String> = result
            .repeats
            .iter()
            .map(|repeat| repeat.split_manifest_sha256.clone())
            .collect();
        assert_eq!(hashes.len(), 5);
    }

    #[test]
    fn canonical_secom_support_is_approximately_paper_80_20() {
        let mut samples = Vec::with_capacity(1567);
        for index in 0..1567 {
            samples.push(StratifiedSampleV1 {
                sample_id: format!("secom-mirror-row-{index:04}"),
                positive: index < 104,
            });
        }
        let result = generate_repeated_stratified_splits_v1(&subject(samples))
            .expect("full-size split generation succeeds");
        for repeat in result.repeats {
            assert_eq!(repeat.train.samples, 1253);
            assert_eq!(repeat.train.positive, 83);
            assert_eq!(repeat.test.samples, 314);
            assert_eq!(repeat.test.positive, 21);
            assert_eq!(repeat.test.negative, 293);
            assert!((repeat.positive_test_fraction - 0.2).abs() < 0.01);
            assert!((repeat.negative_test_fraction - 0.2).abs() < 0.01);
        }
    }

    #[test]
    fn duplicate_sample_identity_is_rejected() {
        let mut samples = synthetic_samples();
        samples[1].sample_id = samples[0].sample_id.clone();
        assert!(generate_repeated_stratified_splits_v1(&subject(samples)).is_err());
    }

    #[test]
    fn invalid_test_fraction_and_duplicate_seeds_reject() {
        let mut invalid_fraction = subject(synthetic_samples());
        invalid_fraction.test_fraction = 0.8;
        assert!(generate_repeated_stratified_splits_v1(&invalid_fraction).is_err());

        let mut duplicate_seeds = subject(synthetic_samples());
        duplicate_seeds.seeds[1] = duplicate_seeds.seeds[0];
        assert!(generate_repeated_stratified_splits_v1(&duplicate_seeds).is_err());
    }
}
