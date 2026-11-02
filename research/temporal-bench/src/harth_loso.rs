//! Subject-held-out HARTH evaluation receipts.
//!
//! This module does not run a classifier. It validates the fold boundary and
//! computes participant-level scores from evaluator-owned confusion counts.
//! A candidate process must not receive the held-out labels or evaluator data.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use super::digest::CanonicalHasher;

const LOSO_SCHEMA: &str = "symthaea.temporal.harth-loso-receipt";
const LOSO_VERSION: &str = "0.1.0";
const HARTH_SUBJECT_COUNT: usize = 22;
const PROBABILITY_SCALE: u128 = 1_000_000;

const NATIVE_CODES: [u16; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140];
const SELFPAB_CODES: [u16; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthLosoError {
    pub field: String,
    pub message: String,
}

impl HarthLosoError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self { field: field.into(), message: message.into() }
    }
}

impl fmt::Display for HarthLosoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}
impl Error for HarthLosoError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
#[repr(u8)]
pub enum HarthLabelTrack {
    Native12 = 0,
    SelfPabComparable10 = 1,
}

impl HarthLabelTrack {
    pub const fn class_codes(self) -> &'static [u16] {
        match self {
            Self::Native12 => &NATIVE_CODES,
            Self::SelfPabComparable10 => &SELFPAB_CODES,
        }
    }

    pub const fn class_count(self) -> usize {
        self.class_codes().len()
    }

    /// Map an original HARTH label into the chosen evaluation track.
    ///
    /// SelfPABComparable10 merges cycling-sitting-inactive (130) into 13 and
    /// cycling-standing-inactive (140) into 14. Native12 preserves all codes.
    pub fn class_index(self, raw_code: u16) -> Option<usize> {
        let mapped = match (self, raw_code) {
            (Self::SelfPabComparable10, 130) => 13,
            (Self::SelfPabComparable10, 140) => 14,
            _ => raw_code,
        };
        self.class_codes().iter().position(|code| *code == mapped)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthCandidateIdentity {
    pub candidate_id: String,
    /// Git commit, immutable release ID, or other pinned source revision.
    pub source_revision: String,
    pub dependency_lock_sha256: [u8; 32],
    pub artifact_sha256: [u8; 32],
    pub configuration_sha256: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthSubjectFold {
    pub test_subject_sha256: [u8; 32],
    /// Sorted identities of the 21 training subjects for this LOSO fold.
    pub training_subject_sha256: Vec<[u8; 32]>,
    /// Row-major confusion counts: true class × predicted class.
    pub confusion_counts: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthFoldMetrics {
    pub test_subject_sha256: [u8; 32],
    pub sample_count: u64,
    /// Macro-F1 over classes present in truth or prediction for this subject.
    /// A class absent from both truth and prediction is excluded, matching
    /// macro-F1 over the observed label union; a class predicted but never
    /// true remains in the denominator with F1=0.
    pub macro_f1_ppm: u32,
    pub supported_class_count: u8,
    /// F1 for every declared class. None means absent from truth and prediction.
    pub class_f1_ppm: Vec<Option<u32>>,
    pub class_true_support: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthLosoAggregate {
    pub subject_fold_count: u32,
    pub total_test_samples: u64,
    /// Arithmetic mean of the 22 participant macro-F1 values (not pooled frames).
    pub mean_subject_macro_f1_ppm: u32,
    /// Descriptive standard error across the 22 participant-fold macro-F1 values.
    /// Paired candidate-vs-baseline inference must still use fold-matched deltas.
    pub standard_error_subject_macro_f1_ppm: u32,
    pub worst_subject_macro_f1_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthLosoReceipt {
    pub schema: String,
    pub version: String,
    pub dataset_manifest_sha256: [u8; 32],
    pub preprocessing_sha256: [u8; 32],
    pub label_track: HarthLabelTrack,
    pub candidate: HarthCandidateIdentity,
    /// Exactly 22 folds, sorted by test subject digest.
    pub folds: Vec<HarthSubjectFold>,
    pub aggregate: HarthLosoAggregate,
    pub receipt_sha256: [u8; 32],
}

fn digest_is_nonzero(digest: &[u8; 32]) -> bool {
    digest.iter().any(|byte| *byte != 0)
}

fn confusion_metrics(
    track: HarthLabelTrack,
    counts: &[u64],
    test_subject_sha256: [u8; 32],
) -> Result<HarthFoldMetrics, HarthLosoError> {
    let class_count = track.class_count();
    let expected_len = class_count.checked_mul(class_count).ok_or_else(|| {
        HarthLosoError::new("confusion_counts", "matrix dimension overflow")
    })?;
    if counts.len() != expected_len {
        return Err(HarthLosoError::new(
            "confusion_counts",
            format!("expected {expected_len} counts for a {class_count}x{class_count} matrix, got {}", counts.len()),
        ));
    }

    let mut class_f1_ppm = Vec::with_capacity(class_count);
    let mut class_true_support = Vec::with_capacity(class_count);
    let mut supported_f1_sum = 0_u128;
    let mut supported_class_count = 0_u8;
    let mut total_samples = 0_u64;

    for true_class in 0..class_count {
        let true_positive = u128::from(counts[true_class * class_count + true_class]);
        let mut false_negative = 0_u128;
        let mut false_positive = 0_u128;
        let mut true_support = 0_u64;

        for predicted_class in 0..class_count {
            let count = counts[true_class * class_count + predicted_class];
            true_support = true_support.checked_add(count).ok_or_else(|| {
                HarthLosoError::new("confusion_counts", "row support overflow")
            })?;
            if predicted_class != true_class {
                false_negative = false_negative.checked_add(u128::from(count)).ok_or_else(|| {
                    HarthLosoError::new("confusion_counts", "false-negative count overflow")
                })?;
            }
        }
        for other_true_class in 0..class_count {
            if other_true_class != true_class {
                false_positive = false_positive
                    .checked_add(u128::from(counts[other_true_class * class_count + true_class]))
                    .ok_or_else(|| {
                        HarthLosoError::new("confusion_counts", "false-positive count overflow")
                    })?;
            }
        }

        total_samples = total_samples.checked_add(true_support).ok_or_else(|| {
            HarthLosoError::new("confusion_counts", "total sample count overflow")
        })?;
        class_true_support.push(true_support);

        let denominator = true_positive * 2 + false_positive + false_negative;
        if denominator == 0 {
            class_f1_ppm.push(None);
        } else {
            let score = ((true_positive * 2 * PROBABILITY_SCALE) / denominator)
                .min(PROBABILITY_SCALE) as u32;
            supported_f1_sum = supported_f1_sum.checked_add(u128::from(score)).ok_or_else(|| {
                HarthLosoError::new("confusion_counts", "macro-F1 accumulation overflow")
            })?;
            supported_class_count = supported_class_count.saturating_add(1);
            class_f1_ppm.push(Some(score));
        }
    }

    if total_samples == 0 || supported_class_count == 0 {
        return Err(HarthLosoError::new(
            "confusion_counts",
            "each held-out subject must contain at least one scored sample",
        ));
    }

    let macro_f1_ppm = ((supported_f1_sum + u128::from(supported_class_count / 2))
        / u128::from(supported_class_count)) as u32;

    Ok(HarthFoldMetrics {
        test_subject_sha256,
        sample_count: total_samples,
        macro_f1_ppm,
        supported_class_count,
        class_f1_ppm,
        class_true_support,
    })
}

fn validate_fold_partition(
    folds: &[HarthSubjectFold],
) -> Result<(), HarthLosoError> {
    if folds.len() != HARTH_SUBJECT_COUNT {
        return Err(HarthLosoError::new(
            "folds",
            format!("HARTH LOSO requires exactly {HARTH_SUBJECT_COUNT} held-out subjects, got {}", folds.len()),
        ));
    }

    let mut all_subjects = BTreeSet::new();
    let mut previous_test: Option<[u8; 32]> = None;
    for (fold_index, fold) in folds.iter().enumerate() {
        if !digest_is_nonzero(&fold.test_subject_sha256) {
            return Err(HarthLosoError::new(
                format!("folds[{fold_index}].test_subject_sha256"),
                "held-out subject identity must be nonzero",
            ));
        }
        if previous_test.is_some_and(|previous| previous >= fold.test_subject_sha256) {
            return Err(HarthLosoError::new(
                format!("folds[{fold_index}].test_subject_sha256"),
                "folds must be sorted by unique held-out subject digest",
            ));
        }
        previous_test = Some(fold.test_subject_sha256);
        all_subjects.insert(fold.test_subject_sha256);

        if fold.training_subject_sha256.len() != HARTH_SUBJECT_COUNT - 1 {
            return Err(HarthLosoError::new(
                format!("folds[{fold_index}].training_subject_sha256"),
                "each fold must train on exactly the other 21 subjects",
            ));
        }
        let mut previous_training: Option<[u8; 32]> = None;
        for subject in &fold.training_subject_sha256 {
            if !digest_is_nonzero(subject) {
                return Err(HarthLosoError::new(
                    format!("folds[{fold_index}].training_subject_sha256"),
                    "training subject identity must be nonzero",
                ));
            }
            if *subject == fold.test_subject_sha256 {
                return Err(HarthLosoError::new(
                    format!("folds[{fold_index}].training_subject_sha256"),
                    "held-out subject is present in its own training set",
                ));
            }
            if previous_training.is_some_and(|previous| previous >= *subject) {
                return Err(HarthLosoError::new(
                    format!("folds[{fold_index}].training_subject_sha256"),
                    "training subject digests must be sorted and unique",
                ));
            }
            previous_training = Some(*subject);
            all_subjects.insert(*subject);
        }
    }

    if all_subjects.len() != HARTH_SUBJECT_COUNT {
        return Err(HarthLosoError::new(
            "folds.subject_universe",
            format!("expected exactly {HARTH_SUBJECT_COUNT} unique subject families, got {}", all_subjects.len()),
        ));
    }

    let subject_universe: BTreeSet<[u8; 32]> = folds
        .iter()
        .map(|fold| fold.test_subject_sha256)
        .collect();
    for (fold_index, fold) in folds.iter().enumerate() {
        let training: BTreeSet<[u8; 32]> =
            fold.training_subject_sha256.iter().copied().collect();
        let expected_training: BTreeSet<[u8; 32]> = subject_universe
            .iter()
            .copied()
            .filter(|subject| *subject != fold.test_subject_sha256)
            .collect();
        if training != expected_training {
            return Err(HarthLosoError::new(
                format!("folds[{fold_index}].training_subject_sha256"),
                "training set must contain exactly the other 21 held-out subject families",
            ));
        }
    }

    Ok(())
}

fn compute_aggregate(
    track: HarthLabelTrack,
    folds: &[HarthSubjectFold],
) -> Result<HarthLosoAggregate, HarthLosoError> {
    let mut total_samples = 0_u64;
    let mut score_sum = 0_u128;
    let mut subject_scores = Vec::with_capacity(folds.len());
    let mut worst_score = u32::MAX;

    for (fold_index, fold) in folds.iter().enumerate() {
        let metrics = confusion_metrics(track, &fold.confusion_counts, fold.test_subject_sha256)
            .map_err(|error| {
                HarthLosoError::new(
                    format!("folds[{fold_index}].{}", error.field),
                    error.message,
                )
            })?;
        total_samples = total_samples.checked_add(metrics.sample_count).ok_or_else(|| {
            HarthLosoError::new("aggregate.total_test_samples", "sample count overflow")
        })?;
        score_sum = score_sum.checked_add(u128::from(metrics.macro_f1_ppm)).ok_or_else(|| {
            HarthLosoError::new("aggregate.mean_subject_macro_f1_ppm", "score sum overflow")
        })?;
        subject_scores.push(f64::from(metrics.macro_f1_ppm));
        worst_score = worst_score.min(metrics.macro_f1_ppm);
    }

    let fold_count = u32::try_from(folds.len()).map_err(|_| {
        HarthLosoError::new("aggregate.subject_fold_count", "fold count overflow")
    })?;
    let mean_score = ((score_sum + u128::from(fold_count / 2)) / u128::from(fold_count)) as u32;
    let mean_f64 = subject_scores.iter().sum::<f64>() / subject_scores.len() as f64;
    let squared_deviation_sum = subject_scores
        .iter()
        .map(|score| {
            let difference = *score - mean_f64;
            difference * difference
        })
        .sum::<f64>();
    let sample_variance = squared_deviation_sum / (subject_scores.len() - 1) as f64;
    let standard_error = (sample_variance / subject_scores.len() as f64).sqrt().round()
        .clamp(0.0, f64::from(u32::MAX)) as u32;

    Ok(HarthLosoAggregate {
        subject_fold_count: fold_count,
        total_test_samples: total_samples,
        mean_subject_macro_f1_ppm: mean_score,
        standard_error_subject_macro_f1_ppm: standard_error,
        worst_subject_macro_f1_ppm: worst_score,
    })
}

fn receipt_digest(receipt: &HarthLosoReceipt) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.harth-loso-receipt.v1");
    hash.bytes(receipt.schema.as_bytes());
    hash.bytes(receipt.version.as_bytes());
    hash.bytes(&receipt.dataset_manifest_sha256);
    hash.bytes(&receipt.preprocessing_sha256);
    hash.u8(receipt.label_track as u8);
    hash.bytes(receipt.candidate.candidate_id.as_bytes());
    hash.bytes(receipt.candidate.source_revision.as_bytes());
    hash.bytes(&receipt.candidate.dependency_lock_sha256);
    hash.bytes(&receipt.candidate.artifact_sha256);
    hash.bytes(&receipt.candidate.configuration_sha256);
    hash.usize(receipt.folds.len());
    for fold in &receipt.folds {
        hash.bytes(&fold.test_subject_sha256);
        hash.usize(fold.training_subject_sha256.len());
        for subject in &fold.training_subject_sha256 {
            hash.bytes(subject);
        }
        hash.usize(fold.confusion_counts.len());
        for count in &fold.confusion_counts {
            hash.u64(*count);
        }
    }
    hash.u32(receipt.aggregate.subject_fold_count);
    hash.u64(receipt.aggregate.total_test_samples);
    hash.u32(receipt.aggregate.mean_subject_macro_f1_ppm);
    hash.u32(receipt.aggregate.standard_error_subject_macro_f1_ppm);
    hash.u32(receipt.aggregate.worst_subject_macro_f1_ppm);
    hash.finish()
}

/// Build a canonical participant-LOSO receipt from evaluator-owned fold counts.
/// Folds and each training-subject list are sorted before the receipt is frozen.
pub fn build_harth_loso_receipt(
    dataset_manifest_sha256: [u8; 32],
    preprocessing_sha256: [u8; 32],
    label_track: HarthLabelTrack,
    candidate: HarthCandidateIdentity,
    mut folds: Vec<HarthSubjectFold>,
) -> Result<HarthLosoReceipt, HarthLosoError> {
    folds.sort_by_key(|fold| fold.test_subject_sha256);
    for fold in &mut folds {
        fold.training_subject_sha256.sort();
    }

    validate_fold_partition(&folds)?;
    let aggregate = compute_aggregate(label_track, &folds)?;
    let mut receipt = HarthLosoReceipt {
        schema: LOSO_SCHEMA.to_owned(),
        version: LOSO_VERSION.to_owned(),
        dataset_manifest_sha256,
        preprocessing_sha256,
        label_track,
        candidate,
        folds,
        aggregate,
        receipt_sha256: [0; 32],
    };
    receipt.receipt_sha256 = receipt_digest(&receipt);
    validate_harth_loso_receipt(&receipt)?;
    Ok(receipt)
}

pub fn harth_fold_metrics(
    track: HarthLabelTrack,
    fold: &HarthSubjectFold,
) -> Result<HarthFoldMetrics, HarthLosoError> {
    confusion_metrics(track, &fold.confusion_counts, fold.test_subject_sha256)
}

/// Recompute the complete LOSO fold contract, all confusion-derived metrics,
/// participant mean/worst summaries, and the canonical receipt digest.
pub fn validate_harth_loso_receipt(
    receipt: &HarthLosoReceipt,
) -> Result<(), HarthLosoError> {
    if receipt.schema != LOSO_SCHEMA || receipt.version != LOSO_VERSION {
        return Err(HarthLosoError::new(
            "schema",
            "unsupported HARTH LOSO receipt schema or version",
        ));
    }
    if !digest_is_nonzero(&receipt.dataset_manifest_sha256)
        || !digest_is_nonzero(&receipt.preprocessing_sha256)
        || !digest_is_nonzero(&receipt.candidate.dependency_lock_sha256)
        || !digest_is_nonzero(&receipt.candidate.artifact_sha256)
        || !digest_is_nonzero(&receipt.candidate.configuration_sha256)
    {
        return Err(HarthLosoError::new(
            "receipt.identity",
            "dataset, preprocessing and candidate digests must be nonzero",
        ));
    }
    if receipt.candidate.candidate_id.trim().is_empty()
        || receipt.candidate.source_revision.trim().is_empty()
    {
        return Err(HarthLosoError::new(
            "candidate",
            "candidate ID and source revision must be non-empty",
        ));
    }

    validate_fold_partition(&receipt.folds)?;
    let expected_aggregate = compute_aggregate(receipt.label_track, &receipt.folds)?;
    if receipt.aggregate != expected_aggregate {
        return Err(HarthLosoError::new(
            "aggregate",
            "aggregate does not reproduce from the 22 subject-fold confusion matrices",
        ));
    }
    if receipt.receipt_sha256 != receipt_digest(receipt) {
        return Err(HarthLosoError::new(
            "receipt_sha256",
            "canonical receipt digest mismatch",
        ));
    }
    Ok(())
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthSubjectF1Delta {
    pub test_subject_sha256: [u8; 32],
    pub baseline_macro_f1_ppm: u32,
    pub candidate_macro_f1_ppm: u32,
    /// Candidate minus baseline; positive is an improvement.
    pub delta_macro_f1_ppm: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthPairedComparisonAggregate {
    pub subject_count: u32,
    pub mean_delta_macro_f1_ppm: i64,
    pub standard_error_delta_ppm: u32,
    /// Two-sided 95% paired Student-t interval, df=21, t*=2.0796.
    /// Bounds are rounded outward and clamped to the possible F1-difference range.
    pub lower_95_ci_delta_ppm: i64,
    pub upper_95_ci_delta_ppm: i64,
    pub improved_subjects: u32,
    pub unchanged_subjects: u32,
    pub worsened_subjects: u32,
    /// Exact two-sided binomial sign test on non-tied participant folds.
    /// Ties are excluded; numerator/denominator preserve the exact rational value.
    pub sign_test_extreme_outcomes: u64,
    pub sign_test_total_outcomes: u64,
    /// Rounded upward to ppm so a non-zero exact p-value cannot be rounded to zero.
    pub sign_test_two_sided_p_value_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HarthPairedComparisonReceipt {
    pub schema: String,
    pub version: String,
    pub baseline_receipt_sha256: [u8; 32],
    pub candidate_receipt_sha256: [u8; 32],
    pub label_track: HarthLabelTrack,
    pub subject_deltas: Vec<HarthSubjectF1Delta>,
    pub aggregate: HarthPairedComparisonAggregate,
    pub comparison_sha256: [u8; 32],
}

fn paired_comparison_digest(
    receipt: &HarthPairedComparisonReceipt,
) -> [u8; 32] {
    let mut hash = CanonicalHasher::new(b"symthaea.temporal.harth-paired-losocomparison.v1");
    hash.bytes(receipt.schema.as_bytes());
    hash.bytes(receipt.version.as_bytes());
    hash.bytes(&receipt.baseline_receipt_sha256);
    hash.bytes(&receipt.candidate_receipt_sha256);
    hash.u8(receipt.label_track as u8);
    hash.usize(receipt.subject_deltas.len());
    for delta in &receipt.subject_deltas {
        hash.bytes(&delta.test_subject_sha256);
        hash.u32(delta.baseline_macro_f1_ppm);
        hash.u32(delta.candidate_macro_f1_ppm);
        hash.u64(delta.delta_macro_f1_ppm as u64);
    }
    hash.u32(receipt.aggregate.subject_count);
    hash.u64(receipt.aggregate.mean_delta_macro_f1_ppm as u64);
    hash.u32(receipt.aggregate.standard_error_delta_ppm);
    hash.u64(receipt.aggregate.lower_95_ci_delta_ppm as u64);
    hash.u64(receipt.aggregate.upper_95_ci_delta_ppm as u64);
    hash.u32(receipt.aggregate.improved_subjects);
    hash.u32(receipt.aggregate.unchanged_subjects);
    hash.u32(receipt.aggregate.worsened_subjects);
    hash.u64(receipt.aggregate.sign_test_extreme_outcomes);
    hash.u64(receipt.aggregate.sign_test_total_outcomes);
    hash.u32(receipt.aggregate.sign_test_two_sided_p_value_ppm);
    hash.finish()
}


fn binomial_coefficient(n: u32, k: u32) -> u64 {
    let k = k.min(n.saturating_sub(k));
    let mut value = 1_u64;
    for index in 0..k {
        value = value
            .saturating_mul(u64::from(n - index))
            / u64::from(index + 1);
    }
    value
}

fn exact_two_sided_sign_test(
    improved: u32,
    worsened: u32,
) -> (u64, u64, u32) {
    let non_tied = improved.saturating_add(worsened);
    if non_tied == 0 {
        return (1, 1, 1_000_000);
    }
    let lower_tail = improved.min(worsened);
    let mut tail_sum = 0_u64;
    for k in 0..=lower_tail {
        tail_sum = tail_sum.saturating_add(binomial_coefficient(non_tied, k));
    }
    let total = 1_u64 << non_tied;
    let extreme = tail_sum.saturating_mul(2).min(total);
    let p_ppm = ((u128::from(extreme) * 1_000_000
        + u128::from(total) - 1)
        / u128::from(total)) as u32;
    (extreme, total, p_ppm)
}

fn calculate_paired_comparison(
    baseline: &HarthLosoReceipt,
    candidate: &HarthLosoReceipt,
) -> Result<HarthPairedComparisonReceipt, HarthLosoError> {
    validate_harth_loso_receipt(baseline)?;
    validate_harth_loso_receipt(candidate)?;

    if baseline.dataset_manifest_sha256 != candidate.dataset_manifest_sha256 {
        return Err(HarthLosoError::new(
            "comparison.dataset_manifest_sha256",
            "paired comparison requires the same dataset manifest",
        ));
    }
    if baseline.preprocessing_sha256 != candidate.preprocessing_sha256 {
        return Err(HarthLosoError::new(
            "comparison.preprocessing_sha256",
            "paired comparison requires identical preprocessing",
        ));
    }
    if baseline.label_track != candidate.label_track {
        return Err(HarthLosoError::new(
            "comparison.label_track",
            "paired comparison requires the same label track",
        ));
    }
    if baseline.folds.len() != candidate.folds.len()
        || baseline.folds.len() != HARTH_SUBJECT_COUNT
    {
        return Err(HarthLosoError::new(
            "comparison.folds",
            "paired comparison requires exactly the same 22 subject folds",
        ));
    }

    let mut subject_deltas = Vec::with_capacity(HARTH_SUBJECT_COUNT);
    let mut delta_values = Vec::with_capacity(HARTH_SUBJECT_COUNT);
    let mut improved_subjects = 0_u32;
    let mut unchanged_subjects = 0_u32;
    let mut worsened_subjects = 0_u32;

    for index in 0..HARTH_SUBJECT_COUNT {
        let baseline_fold = &baseline.folds[index];
        let candidate_fold = &candidate.folds[index];
        if baseline_fold.test_subject_sha256 != candidate_fold.test_subject_sha256 {
            return Err(HarthLosoError::new(
                format!("comparison.subject_deltas[{index}]"),
                "candidate and baseline folds must use the same held-out participant",
            ));
        }

        let baseline_metrics = harth_fold_metrics(baseline.label_track, baseline_fold)?;
        let candidate_metrics = harth_fold_metrics(candidate.label_track, candidate_fold)?;
        let delta = i64::from(candidate_metrics.macro_f1_ppm)
            - i64::from(baseline_metrics.macro_f1_ppm);
        if delta > 0 {
            improved_subjects = improved_subjects.saturating_add(1);
        } else if delta < 0 {
            worsened_subjects = worsened_subjects.saturating_add(1);
        } else {
            unchanged_subjects = unchanged_subjects.saturating_add(1);
        }
        delta_values.push(delta as f64);
        subject_deltas.push(HarthSubjectF1Delta {
            test_subject_sha256: baseline_fold.test_subject_sha256,
            baseline_macro_f1_ppm: baseline_metrics.macro_f1_ppm,
            candidate_macro_f1_ppm: candidate_metrics.macro_f1_ppm,
            delta_macro_f1_ppm: delta,
        });
    }

    let subject_count = subject_deltas.len() as u32;
    let mean = delta_values.iter().sum::<f64>() / delta_values.len() as f64;
    let mean_delta = mean.round() as i64;
    let squared_deviation_sum = delta_values
        .iter()
        .map(|delta| {
            let difference = *delta - mean;
            difference * difference
        })
        .sum::<f64>();
    let sample_variance = squared_deviation_sum / (delta_values.len() - 1) as f64;
    let standard_error_f64 = (sample_variance / delta_values.len() as f64).sqrt();
    let standard_error = standard_error_f64
        .round()
        .clamp(0.0, f64::from(u32::MAX)) as u32;
    // Student-t 97.5th percentile for 21 degrees of freedom.
    const T_975_DF21: f64 = 2.079_613_844_727_662;
    let half_width = T_975_DF21 * standard_error_f64;
    let lower = (mean - half_width).floor().max(-1_000_000.0) as i64;
    let upper = (mean + half_width).ceil().min(1_000_000.0) as i64;

    let (sign_test_extreme_outcomes, sign_test_total_outcomes, sign_test_two_sided_p_value_ppm) =
        exact_two_sided_sign_test(improved_subjects, worsened_subjects);

    let mut receipt = HarthPairedComparisonReceipt {
        schema: "symthaea.temporal.harth-paired-loso-comparison".to_owned(),
        version: "0.1.0".to_owned(),
        baseline_receipt_sha256: baseline.receipt_sha256,
        candidate_receipt_sha256: candidate.receipt_sha256,
        label_track: baseline.label_track,
        subject_deltas,
        aggregate: HarthPairedComparisonAggregate {
            subject_count,
            mean_delta_macro_f1_ppm: mean_delta,
            standard_error_delta_ppm: standard_error,
            lower_95_ci_delta_ppm: lower,
            upper_95_ci_delta_ppm: upper,
            improved_subjects,
            unchanged_subjects,
            worsened_subjects,
            sign_test_extreme_outcomes,
            sign_test_total_outcomes,
            sign_test_two_sided_p_value_ppm,
        },
        comparison_sha256: [0; 32],
    };
    receipt.comparison_sha256 = paired_comparison_digest(&receipt);
    Ok(receipt)
}

/// Compare two LOSO receipts only when all 22 test subjects, dataset bytes,
/// preprocessing and label-track semantics match exactly.
pub fn build_harth_paired_comparison(
    baseline: &HarthLosoReceipt,
    candidate: &HarthLosoReceipt,
) -> Result<HarthPairedComparisonReceipt, HarthLosoError> {
    calculate_paired_comparison(baseline, candidate)
}

/// Recompute every paired fold delta, the mean/standard error/95% interval,
/// subject win counts and comparison digest.
pub fn validate_harth_paired_comparison(
    receipt: &HarthPairedComparisonReceipt,
    baseline: &HarthLosoReceipt,
    candidate: &HarthLosoReceipt,
) -> Result<(), HarthLosoError> {
    let expected = calculate_paired_comparison(baseline, candidate)?;
    if *receipt != expected {
        return Err(HarthLosoError::new(
            "comparison_receipt",
            "paired comparison does not reproduce from the two LOSO receipts",
        ));
    }
    if receipt.comparison_sha256 != paired_comparison_digest(receipt) {
        return Err(HarthLosoError::new(
            "comparison_sha256",
            "paired comparison digest mismatch",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate() -> HarthCandidateIdentity {
        HarthCandidateIdentity {
            candidate_id: "fixture-baseline".to_owned(),
            source_revision: "git:0123456789abcdef".to_owned(),
            dependency_lock_sha256: [1; 32],
            artifact_sha256: [2; 32],
            configuration_sha256: [3; 32],
        }
    }

    fn fold_set(track: HarthLabelTrack) -> Vec<HarthSubjectFold> {
        let mut subjects: Vec<[u8; 32]> = (1_u8..=22).map(|value| [value; 32]).collect();
        subjects.sort();
        let classes = track.class_count();
        let mut folds = Vec::new();
        for test in &subjects {
            let mut training: Vec<[u8; 32]> = subjects
                .iter()
                .copied()
                .filter(|subject| subject != test)
                .collect();
            training.sort();
            let mut confusion_counts = vec![0_u64; classes * classes];
            for class in 0..classes {
                confusion_counts[class * classes + class] = 100;
            }
            folds.push(HarthSubjectFold {
                test_subject_sha256: *test,
                training_subject_sha256: training,
                confusion_counts,
            });
        }
        folds
    }

    #[test]
    fn native_and_selfpab_label_tracks_map_all_twelve_codes() {
        for (index, code) in NATIVE_CODES.iter().enumerate() {
            assert_eq!(HarthLabelTrack::Native12.class_index(*code), Some(index));
        }
        for (index, code) in SELFPAB_CODES.iter().enumerate() {
            assert_eq!(HarthLabelTrack::SelfPabComparable10.class_index(*code), Some(index));
        }
        assert_eq!(HarthLabelTrack::SelfPabComparable10.class_index(130), Some(8));
        assert_eq!(HarthLabelTrack::SelfPabComparable10.class_index(140), Some(9));
        assert_eq!(HarthLabelTrack::Native12.class_index(99), None);
        assert_eq!(HarthLabelTrack::SelfPabComparable10.class_index(99), None);
    }

    #[test]
    fn exactly_twenty_two_subject_loso_receipt_recomputes() {
        let receipt = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            fold_set(HarthLabelTrack::Native12),
        )
        .unwrap();
        assert_eq!(receipt.aggregate.subject_fold_count, 22);
        assert_eq!(receipt.aggregate.mean_subject_macro_f1_ppm, 1_000_000);
        assert_eq!(receipt.aggregate.worst_subject_macro_f1_ppm, 1_000_000);
        validate_harth_loso_receipt(&receipt).unwrap();
    }

    #[test]
    fn aggregate_standard_error_reflects_subject_level_variation() {
        let mut folds = fold_set(HarthLabelTrack::Native12);
        // Make one subject fold imperfect while leaving the other 21 perfect.
        let classes = HarthLabelTrack::Native12.class_count();
        folds[0].confusion_counts[0] = 90;
        folds[0].confusion_counts[1] = 10;
        let receipt = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            folds,
        )
        .unwrap();
        assert!(receipt.aggregate.standard_error_subject_macro_f1_ppm > 0);
        assert!(receipt.aggregate.worst_subject_macro_f1_ppm
            < receipt.aggregate.mean_subject_macro_f1_ppm);
        assert_eq!(classes, 12);
    }

    #[test]
    fn selfpab_comparable_receipt_uses_ten_class_confusion_matrices() {
        let receipt = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::SelfPabComparable10,
            candidate(),
            fold_set(HarthLabelTrack::SelfPabComparable10),
        )
        .unwrap();
        assert_eq!(receipt.folds[0].confusion_counts.len(), 100);
        assert_eq!(receipt.aggregate.mean_subject_macro_f1_ppm, 1_000_000);
    }

    #[test]
    fn held_out_subject_cannot_leak_into_training() {
        let mut folds = fold_set(HarthLabelTrack::Native12);
        let held_out = folds[0].test_subject_sha256;
        folds[0].training_subject_sha256[0] = held_out;
        assert!(build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            folds,
        )
        .is_err());
    }

    #[test]
    fn every_subject_must_be_held_out_exactly_once() {
        let mut folds = fold_set(HarthLabelTrack::Native12);
        folds[1].test_subject_sha256 = folds[0].test_subject_sha256;
        assert!(build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            folds,
        )
        .is_err());
    }

    #[test]
    fn each_fold_must_contain_the_other_twenty_one_subjects() {
        let mut folds = fold_set(HarthLabelTrack::Native12);
        folds[0].training_subject_sha256.pop();
        assert!(build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            folds,
        )
        .is_err());
    }

    #[test]
    fn confusion_matrix_shape_and_nonempty_denominators_are_required() {
        let mut folds = fold_set(HarthLabelTrack::Native12);
        folds[0].confusion_counts.pop();
        assert!(build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            folds,
        )
        .is_err());

        let mut empty = fold_set(HarthLabelTrack::Native12);
        empty[0].confusion_counts.fill(0);
        assert!(build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            empty,
        )
        .is_err());
    }

    #[test]
    fn fold_macro_f1_is_recomputed_from_confusion_counts() {
        let mut fold = fold_set(HarthLabelTrack::SelfPabComparable10).remove(0);
        // class 0: TP=8, FN=2, FP=2 => F1 = 8/10 = 800,000 ppm.
        fold.confusion_counts[0] = 8;
        fold.confusion_counts[1] = 2;
        fold.confusion_counts[10] = 2;
        let metrics = harth_fold_metrics(HarthLabelTrack::SelfPabComparable10, &fold).unwrap();
        assert_eq!(metrics.class_f1_ppm[0], Some(800_000));
        assert_eq!(metrics.class_true_support[0], 10);
        assert_eq!(metrics.supported_class_count, 10);
    }

    #[test]
    fn receipt_digest_detects_tampered_aggregate_or_fold_counts() {
        let receipt = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            fold_set(HarthLabelTrack::Native12),
        )
        .unwrap();
        let mut forged_aggregate = receipt.clone();
        forged_aggregate.aggregate.worst_subject_macro_f1_ppm -= 1;
        assert!(validate_harth_loso_receipt(&forged_aggregate).is_err());

        let mut forged_fold = receipt;
        forged_fold.folds[0].confusion_counts[0] += 1;
        assert!(validate_harth_loso_receipt(&forged_fold).is_err());
    }

    #[test]
    fn paired_loso_comparison_requires_same_protocol_and_reports_subject_deltas() {
        fn folds_with_score(diagonal: u64, error: u64) -> Vec<HarthSubjectFold> {
            let track = HarthLabelTrack::Native12;
            fold_set(track)
                .into_iter()
                .map(|mut fold| {
                    let classes = track.class_count();
                    fold.confusion_counts.fill(0);
                    for class in 0..classes {
                        fold.confusion_counts[class * classes + class] = diagonal;
                        fold.confusion_counts[class * classes + (class + 1) % classes] = error;
                    }
                    fold
                })
                .collect()
        }

        let baseline = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            folds_with_score(80, 20),
        )
        .unwrap();
        let candidate_receipt = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            HarthCandidateIdentity {
                candidate_id: "candidate-vs-baseline".to_owned(),
                source_revision: "git:fedcba9876543210".to_owned(),
                dependency_lock_sha256: [6; 32],
                artifact_sha256: [7; 32],
                configuration_sha256: [8; 32],
            },
            folds_with_score(90, 10),
        )
        .unwrap();

        let paired = build_harth_paired_comparison(&baseline, &candidate_receipt).unwrap();
        assert_eq!(paired.aggregate.subject_count, 22);
        assert_eq!(paired.aggregate.mean_delta_macro_f1_ppm, 100_000);
        assert_eq!(paired.aggregate.standard_error_delta_ppm, 0);
        assert_eq!(paired.aggregate.lower_95_ci_delta_ppm, 100_000);
        assert_eq!(paired.aggregate.upper_95_ci_delta_ppm, 100_000);
        assert_eq!(paired.aggregate.improved_subjects, 22);
        assert_eq!(paired.aggregate.worsened_subjects, 0);
        validate_harth_paired_comparison(&paired, &baseline, &candidate_receipt).unwrap();
    }

    #[test]
    fn exact_sign_test_uses_non_ties_and_preserves_small_p_values() {
        let (extreme, total, p_ppm) = exact_two_sided_sign_test(22, 0);
        assert_eq!(extreme, 2);
        assert_eq!(total, 1 << 22);
        assert_eq!(p_ppm, 1);

        let (extreme_with_ties, total_with_ties, p_with_ties) =
            exact_two_sided_sign_test(5, 0);
        assert_eq!(extreme_with_ties, 2);
        assert_eq!(total_with_ties, 32);
        assert_eq!(p_with_ties, 62_500);

        let (extreme_none, total_none, p_none) = exact_two_sided_sign_test(0, 0);
        assert_eq!((extreme_none, total_none, p_none), (1, 1, 1_000_000));
    }

    #[test]
    fn paired_comparison_rejects_dataset_or_preprocessing_mismatch() {
        let baseline = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            fold_set(HarthLabelTrack::Native12),
        )
        .unwrap();
        let candidate_receipt = build_harth_loso_receipt(
            [9; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            HarthCandidateIdentity {
                candidate_id: "different-dataset".to_owned(),
                source_revision: "git:fedcba9876543210".to_owned(),
                dependency_lock_sha256: [6; 32],
                artifact_sha256: [7; 32],
                configuration_sha256: [8; 32],
            },
            fold_set(HarthLabelTrack::Native12),
        )
        .unwrap();
        assert!(build_harth_paired_comparison(&baseline, &candidate_receipt).is_err());
    }

    #[test]
    fn paired_comparison_rejects_label_track_mismatch() {
        let baseline = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            fold_set(HarthLabelTrack::Native12),
        )
        .unwrap();
        let candidate_receipt = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::SelfPabComparable10,
            HarthCandidateIdentity {
                candidate_id: "ten-class".to_owned(),
                source_revision: "git:fedcba9876543210".to_owned(),
                dependency_lock_sha256: [6; 32],
                artifact_sha256: [7; 32],
                configuration_sha256: [8; 32],
            },
            fold_set(HarthLabelTrack::SelfPabComparable10),
        )
        .unwrap();
        assert!(build_harth_paired_comparison(&baseline, &candidate_receipt).is_err());
    }

    #[test]
    fn paired_comparison_tampering_is_rejected() {
        let baseline = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            fold_set(HarthLabelTrack::Native12),
        )
        .unwrap();
        let candidate_receipt = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            HarthCandidateIdentity {
                candidate_id: "candidate".to_owned(),
                source_revision: "git:fedcba9876543210".to_owned(),
                dependency_lock_sha256: [6; 32],
                artifact_sha256: [7; 32],
                configuration_sha256: [8; 32],
            },
            fold_set(HarthLabelTrack::Native12),
        )
        .unwrap();
        let mut paired = build_harth_paired_comparison(&baseline, &candidate_receipt).unwrap();
        paired.aggregate.mean_delta_macro_f1_ppm = 1;
        assert!(validate_harth_paired_comparison(&paired, &baseline, &candidate_receipt).is_err());
    }

    #[test]
    fn training_subjects_are_sorted_canonically_by_builder() {
        let mut folds = fold_set(HarthLabelTrack::Native12);
        folds[0].training_subject_sha256.reverse();
        let receipt = build_harth_loso_receipt(
            [4; 32],
            [5; 32],
            HarthLabelTrack::Native12,
            candidate(),
            folds,
        )
        .unwrap();
        assert!(receipt.folds.iter().all(|fold| {
            fold.training_subject_sha256.windows(2).all(|pair| pair[0] < pair[1])
        }));
    }
}
