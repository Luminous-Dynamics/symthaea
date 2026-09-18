// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Preregistered SYM-RSI-SEM-001 semantic-calibration benchmark.
//!
//! This benchmark owns its own deterministic synthetic contexts. It does not use
//! the protected SYM-RSI-001/001D fixture seeds or measurement partitions.
//! Its claim is deliberately narrow: threshold portability for semantic retrieval.

use super::semantic_context::{ContextIdentity, SemanticContextEncoder, SemanticContextError};
use super::semantic_null_calibration::{
    SemanticNullCalibration, SemanticNullCalibrationError, SemanticNullConfig,
};

pub const SYM_RSI_SEM_001_SCHEMA: &str = "symthaea.sym-rsi-sem-001.v1";
pub const SYM_RSI_SEM_001_GENERATOR_VERSION: &str = "semantic-context-generator.v1";
pub const SYM_RSI_SEM_001_DIMENSIONS: [usize; 3] = [4, 8, 16];
pub const SYM_RSI_SEM_001_REFERENCE_COUNT_PER_DIMENSION: usize = 48;
pub const SYM_RSI_SEM_001_HELDOUT_ANCHOR_COUNT_PER_DIMENSION: usize = 24;
pub const SYM_RSI_SEM_001_OOD_ANCHOR_COUNT_PER_DIMENSION: usize = 16;
pub const SYM_RSI_SEM_001_CALIBRATED_SPECIFICITY_THRESHOLD: f32 = 0.95;
pub const SYM_RSI_SEM_001_RELATED_RETENTION_TOLERANCE: f64 = 0.10;
const DISPERSION_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticEvaluationRegime {
    HeldOutRelated,
    HeldOutUnrelated,
    OodClampSaturation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticPairObservation {
    pub context_dimension: usize,
    pub regime: SemanticEvaluationRegime,
    pub raw_similarity: f32,
    pub retrieval_specificity: f32,
    pub raw_activated: bool,
    pub calibrated_activated: bool,
    pub exact_identity_differs: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticDimensionMetrics {
    pub context_dimension: usize,
    pub related_count: usize,
    pub unrelated_count: usize,
    pub ood_count: usize,
    pub raw_related_retention: f64,
    pub calibrated_related_retention: f64,
    pub raw_unrelated_false_activation: f64,
    pub calibrated_unrelated_false_activation: f64,
    pub mean_related_similarity: f64,
    pub mean_unrelated_similarity: f64,
    pub mean_related_specificity: f64,
    pub mean_unrelated_specificity: f64,
    pub ood_exact_identity_differs_count: usize,
    pub ood_raw_activation_count: usize,
    pub ood_calibrated_activation_count: usize,
    pub mean_ood_similarity: f64,
    pub mean_ood_specificity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticCalibrationDisposition {
    /// Primary dispersion endpoint and both preregistered guards passed.
    Pass,
    /// Dispersion improved, but at least one utility/safety guard failed.
    Tradeoff,
    /// Primary dispersion improvement was not established.
    NotEstablished,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticCalibrationReceipt {
    pub schema: &'static str,
    pub generator_version: &'static str,
    pub subject_sha: String,
    pub calibration_digest: [u8; 32],
    pub reference_context_digest: [u8; 32],
    pub heldout_context_digest: [u8; 32],
    pub ood_context_digest: [u8; 32],
    pub raw_global_p95_threshold: f32,
    pub calibrated_specificity_threshold: f32,
    pub dimension_metrics: Vec<SemanticDimensionMetrics>,
    pub raw_unrelated_false_activation_dispersion: f64,
    pub calibrated_unrelated_false_activation_dispersion: f64,
    pub raw_related_retention: f64,
    pub calibrated_related_retention: f64,
    pub primary_dispersion_improved: bool,
    pub related_retention_guard_passed: bool,
    pub worst_dimension_false_activation_guard_passed: bool,
    pub disposition: SemanticCalibrationDisposition,
    pub evidence_digest: [u8; 32],
}

impl SemanticCalibrationReceipt {
    pub fn evidence_digest_hex(&self) -> String {
        hex::encode(self.evidence_digest)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticCalibrationExperimentError {
    Context(SemanticContextError),
    Calibration(SemanticNullCalibrationError),
    MissingDimensionSummary { context_dimension: usize },
    InvalidSubjectSha,
}

impl From<SemanticContextError> for SemanticCalibrationExperimentError {
    fn from(value: SemanticContextError) -> Self {
        Self::Context(value)
    }
}

impl From<SemanticNullCalibrationError> for SemanticCalibrationExperimentError {
    fn from(value: SemanticNullCalibrationError) -> Self {
        Self::Calibration(value)
    }
}

/// Execute the preregistered synthetic SEM-001 benchmark.
///
/// The function is deterministic and does not touch any protected SYM-RSI fixture
/// environment or seed partition. A returned receipt is measurement output, not
/// executable qualification of the Rust subject that produced it.
pub fn run_sym_rsi_sem_001(
    subject_sha: &str,
) -> Result<SemanticCalibrationReceipt, SemanticCalibrationExperimentError> {
    validate_subject_sha(subject_sha)?;

    let encoder = SemanticContextEncoder::default();
    let reference_contexts = reference_contexts();
    let calibration = SemanticNullCalibration::build(
        &reference_contexts,
        encoder,
        SemanticNullConfig {
            max_reference_contexts_per_dimension: SYM_RSI_SEM_001_REFERENCE_COUNT_PER_DIMENSION,
            min_null_pairs_per_dimension: 128,
        },
    )?;

    let summaries = calibration.summaries();
    let mut raw_threshold_sum = 0.0_f32;
    for dimension in SYM_RSI_SEM_001_DIMENSIONS {
        let summary = summaries
            .iter()
            .find(|summary| summary.context_dimension == dimension)
            .ok_or(SemanticCalibrationExperimentError::MissingDimensionSummary {
                context_dimension: dimension,
            })?;
        raw_threshold_sum += summary.p95_similarity;
    }
    let raw_global_p95_threshold =
        raw_threshold_sum / SYM_RSI_SEM_001_DIMENSIONS.len() as f32;

    let heldout_cases = heldout_cases();
    let ood_cases = ood_cases();
    let mut observations = Vec::with_capacity(heldout_cases.len() + ood_cases.len());
    for case in heldout_cases.iter().chain(ood_cases.iter()) {
        let raw_similarity = encoder
            .encode(&case.anchor)?
            .similarity(&encoder.encode(&case.query)?);
        let assessment = calibration.assess(case.dimension, raw_similarity)?;
        observations.push(SemanticPairObservation {
            context_dimension: case.dimension,
            regime: case.regime,
            raw_similarity,
            retrieval_specificity: assessment.retrieval_specificity,
            raw_activated: raw_similarity >= raw_global_p95_threshold,
            calibrated_activated: assessment.retrieval_specificity
                >= SYM_RSI_SEM_001_CALIBRATED_SPECIFICITY_THRESHOLD,
            exact_identity_differs: ContextIdentity::from_context(&case.anchor)?
                != ContextIdentity::from_context(&case.query)?,
        });
    }

    let dimension_metrics = SYM_RSI_SEM_001_DIMENSIONS
        .iter()
        .copied()
        .map(|dimension| dimension_metrics(dimension, &observations))
        .collect::<Vec<_>>();

    let raw_rates = dimension_metrics
        .iter()
        .map(|metrics| metrics.raw_unrelated_false_activation)
        .collect::<Vec<_>>();
    let calibrated_rates = dimension_metrics
        .iter()
        .map(|metrics| metrics.calibrated_unrelated_false_activation)
        .collect::<Vec<_>>();
    let raw_dispersion = dispersion(&raw_rates);
    let calibrated_dispersion = dispersion(&calibrated_rates);

    let related_total = dimension_metrics
        .iter()
        .map(|metrics| metrics.related_count)
        .sum::<usize>();
    let raw_related_activated = observations
        .iter()
        .filter(|observation| observation.regime == SemanticEvaluationRegime::HeldOutRelated)
        .filter(|observation| observation.raw_activated)
        .count();
    let calibrated_related_activated = observations
        .iter()
        .filter(|observation| observation.regime == SemanticEvaluationRegime::HeldOutRelated)
        .filter(|observation| observation.calibrated_activated)
        .count();
    let raw_related_retention = rate(raw_related_activated, related_total);
    let calibrated_related_retention = rate(calibrated_related_activated, related_total);

    let primary_dispersion_improved = raw_dispersion > DISPERSION_EPSILON
        && calibrated_dispersion + DISPERSION_EPSILON < raw_dispersion;
    let related_retention_guard_passed = calibrated_related_retention
        + SYM_RSI_SEM_001_RELATED_RETENTION_TOLERANCE
        >= raw_related_retention;
    let worst_dimension_false_activation_guard_passed = max_rate(&calibrated_rates)
        <= max_rate(&raw_rates) + DISPERSION_EPSILON;

    let disposition = if !primary_dispersion_improved {
        SemanticCalibrationDisposition::NotEstablished
    } else if related_retention_guard_passed && worst_dimension_false_activation_guard_passed {
        SemanticCalibrationDisposition::Pass
    } else {
        SemanticCalibrationDisposition::Tradeoff
    };

    let reference_context_digest = context_corpus_digest("reference", &reference_contexts);
    let heldout_context_digest = case_corpus_digest("heldout", &heldout_cases);
    let ood_context_digest = case_corpus_digest("ood", &ood_cases);
    let calibration_digest = calibration.identity().digest;
    let evidence_digest = receipt_digest(
        subject_sha,
        calibration_digest,
        reference_context_digest,
        heldout_context_digest,
        ood_context_digest,
        raw_global_p95_threshold,
        &dimension_metrics,
        raw_dispersion,
        calibrated_dispersion,
        raw_related_retention,
        calibrated_related_retention,
        primary_dispersion_improved,
        related_retention_guard_passed,
        worst_dimension_false_activation_guard_passed,
        disposition,
    );

    Ok(SemanticCalibrationReceipt {
        schema: SYM_RSI_SEM_001_SCHEMA,
        generator_version: SYM_RSI_SEM_001_GENERATOR_VERSION,
        subject_sha: subject_sha.to_owned(),
        calibration_digest,
        reference_context_digest,
        heldout_context_digest,
        ood_context_digest,
        raw_global_p95_threshold,
        calibrated_specificity_threshold: SYM_RSI_SEM_001_CALIBRATED_SPECIFICITY_THRESHOLD,
        dimension_metrics,
        raw_unrelated_false_activation_dispersion: raw_dispersion,
        calibrated_unrelated_false_activation_dispersion: calibrated_dispersion,
        raw_related_retention,
        calibrated_related_retention,
        primary_dispersion_improved,
        related_retention_guard_passed,
        worst_dimension_false_activation_guard_passed,
        disposition,
        evidence_digest,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct SemanticPairCase {
    dimension: usize,
    regime: SemanticEvaluationRegime,
    anchor: Vec<f32>,
    query: Vec<f32>,
}

fn reference_contexts() -> Vec<Vec<f32>> {
    let mut contexts = Vec::with_capacity(
        SYM_RSI_SEM_001_DIMENSIONS.len() * SYM_RSI_SEM_001_REFERENCE_COUNT_PER_DIMENSION,
    );
    for dimension in SYM_RSI_SEM_001_DIMENSIONS {
        for index in 0..SYM_RSI_SEM_001_REFERENCE_COUNT_PER_DIMENSION {
            contexts.push(base_context(dimension, index));
        }
    }
    contexts
}

fn heldout_cases() -> Vec<SemanticPairCase> {
    let mut cases = Vec::new();
    for dimension in SYM_RSI_SEM_001_DIMENSIONS {
        for offset in 0..SYM_RSI_SEM_001_HELDOUT_ANCHOR_COUNT_PER_DIMENSION {
            let index = 100 + offset;
            let anchor = base_context(dimension, index);
            for (variant, magnitude) in [0.04_f32, 0.08_f32].into_iter().enumerate() {
                cases.push(SemanticPairCase {
                    dimension,
                    regime: SemanticEvaluationRegime::HeldOutRelated,
                    anchor: anchor.clone(),
                    query: local_perturbation(&anchor, index, variant, magnitude),
                });
            }
            cases.push(SemanticPairCase {
                dimension,
                regime: SemanticEvaluationRegime::HeldOutUnrelated,
                anchor,
                query: base_context(dimension, 1000 + offset),
            });
        }
    }
    cases
}

fn ood_cases() -> Vec<SemanticPairCase> {
    let mut cases = Vec::new();
    for dimension in SYM_RSI_SEM_001_DIMENSIONS {
        for offset in 0..SYM_RSI_SEM_001_OOD_ANCHOR_COUNT_PER_DIMENSION {
            let index = 200 + offset;
            let anchor = base_context(dimension, index);
            let mut query = anchor.clone();
            let coordinate = (index * 7 + dimension) % dimension;
            query[coordinate] = if anchor[coordinate] >= 0.0 {
                1.5 + anchor[coordinate].abs()
            } else {
                -1.5 - anchor[coordinate].abs()
            };
            cases.push(SemanticPairCase {
                dimension,
                regime: SemanticEvaluationRegime::OodClampSaturation,
                anchor,
                query,
            });
        }
    }
    cases
}

fn base_context(dimension: usize, index: usize) -> Vec<f32> {
    (0..dimension)
        .map(|coordinate| {
            let mixed = dimension
                .wrapping_mul(97)
                .wrapping_add(index.wrapping_mul(53))
                .wrapping_add(coordinate.wrapping_mul(29))
                .wrapping_add(coordinate.wrapping_mul(coordinate).wrapping_mul(11));
            (mixed % 1801) as f32 / 900.0 - 1.0
        })
        .collect()
}

fn local_perturbation(
    anchor: &[f32],
    index: usize,
    variant: usize,
    magnitude: f32,
) -> Vec<f32> {
    let mut query = anchor.to_vec();
    let coordinate = (index + variant * 3) % query.len();
    if query[coordinate] >= 0.0 {
        query[coordinate] = (query[coordinate] - magnitude).max(-1.0);
    } else {
        query[coordinate] = (query[coordinate] + magnitude).min(1.0);
    }
    query
}

fn dimension_metrics(
    dimension: usize,
    observations: &[SemanticPairObservation],
) -> SemanticDimensionMetrics {
    let related = observations
        .iter()
        .filter(|observation| {
            observation.context_dimension == dimension
                && observation.regime == SemanticEvaluationRegime::HeldOutRelated
        })
        .collect::<Vec<_>>();
    let unrelated = observations
        .iter()
        .filter(|observation| {
            observation.context_dimension == dimension
                && observation.regime == SemanticEvaluationRegime::HeldOutUnrelated
        })
        .collect::<Vec<_>>();
    let ood = observations
        .iter()
        .filter(|observation| {
            observation.context_dimension == dimension
                && observation.regime == SemanticEvaluationRegime::OodClampSaturation
        })
        .collect::<Vec<_>>();

    SemanticDimensionMetrics {
        context_dimension: dimension,
        related_count: related.len(),
        unrelated_count: unrelated.len(),
        ood_count: ood.len(),
        raw_related_retention: rate(
            related.iter().filter(|observation| observation.raw_activated).count(),
            related.len(),
        ),
        calibrated_related_retention: rate(
            related
                .iter()
                .filter(|observation| observation.calibrated_activated)
                .count(),
            related.len(),
        ),
        raw_unrelated_false_activation: rate(
            unrelated
                .iter()
                .filter(|observation| observation.raw_activated)
                .count(),
            unrelated.len(),
        ),
        calibrated_unrelated_false_activation: rate(
            unrelated
                .iter()
                .filter(|observation| observation.calibrated_activated)
                .count(),
            unrelated.len(),
        ),
        mean_related_similarity: mean(related.iter().map(|observation| observation.raw_similarity)),
        mean_unrelated_similarity: mean(
            unrelated.iter().map(|observation| observation.raw_similarity),
        ),
        mean_related_specificity: mean(
            related
                .iter()
                .map(|observation| observation.retrieval_specificity),
        ),
        mean_unrelated_specificity: mean(
            unrelated
                .iter()
                .map(|observation| observation.retrieval_specificity),
        ),
        ood_exact_identity_differs_count: ood
            .iter()
            .filter(|observation| observation.exact_identity_differs)
            .count(),
        ood_raw_activation_count: ood
            .iter()
            .filter(|observation| observation.raw_activated)
            .count(),
        ood_calibrated_activation_count: ood
            .iter()
            .filter(|observation| observation.calibrated_activated)
            .count(),
        mean_ood_similarity: mean(ood.iter().map(|observation| observation.raw_similarity)),
        mean_ood_specificity: mean(
            ood.iter().map(|observation| observation.retrieval_specificity),
        ),
    }
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(values: impl Iterator<Item = f32>) -> f64 {
    let values = values.map(f64::from).collect::<Vec<_>>();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn dispersion(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    max_rate(values) - values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_rate(values: &[f64]) -> f64 {
    values.iter().copied().fold(0.0, f64::max)
}

fn validate_subject_sha(subject_sha: &str) -> Result<(), SemanticCalibrationExperimentError> {
    if subject_sha.len() != 40 || !subject_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(SemanticCalibrationExperimentError::InvalidSubjectSha);
    }
    Ok(())
}

fn context_corpus_digest(label: &str, contexts: &[Vec<f32>]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SYM_RSI_SEM_001_SCHEMA.as_bytes());
    hasher.update(label.as_bytes());
    hasher.update(&(contexts.len() as u64).to_le_bytes());
    for context in contexts {
        hasher.update(&(context.len() as u64).to_le_bytes());
        for value in context {
            hasher.update(&value.to_le_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

fn case_corpus_digest(label: &str, cases: &[SemanticPairCase]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SYM_RSI_SEM_001_SCHEMA.as_bytes());
    hasher.update(label.as_bytes());
    hasher.update(&(cases.len() as u64).to_le_bytes());
    for case in cases {
        hasher.update(&(case.dimension as u64).to_le_bytes());
        hasher.update(&[case.regime as u8]);
        for vector in [&case.anchor, &case.query] {
            hasher.update(&(vector.len() as u64).to_le_bytes());
            for value in vector {
                hasher.update(&value.to_le_bytes());
            }
        }
    }
    *hasher.finalize().as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn receipt_digest(
    subject_sha: &str,
    calibration_digest: [u8; 32],
    reference_context_digest: [u8; 32],
    heldout_context_digest: [u8; 32],
    ood_context_digest: [u8; 32],
    raw_global_p95_threshold: f32,
    dimension_metrics: &[SemanticDimensionMetrics],
    raw_dispersion: f64,
    calibrated_dispersion: f64,
    raw_related_retention: f64,
    calibrated_related_retention: f64,
    primary_dispersion_improved: bool,
    related_retention_guard_passed: bool,
    worst_dimension_guard_passed: bool,
    disposition: SemanticCalibrationDisposition,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SYM_RSI_SEM_001_SCHEMA.as_bytes());
    hasher.update(SYM_RSI_SEM_001_GENERATOR_VERSION.as_bytes());
    hasher.update(subject_sha.as_bytes());
    for digest in [
        calibration_digest,
        reference_context_digest,
        heldout_context_digest,
        ood_context_digest,
    ] {
        hasher.update(&digest);
    }
    hasher.update(&raw_global_p95_threshold.to_bits().to_le_bytes());
    hasher.update(&SYM_RSI_SEM_001_CALIBRATED_SPECIFICITY_THRESHOLD.to_bits().to_le_bytes());
    hasher.update(&(dimension_metrics.len() as u64).to_le_bytes());
    for metrics in dimension_metrics {
        hash_dimension_metrics(&mut hasher, metrics);
    }
    for value in [
        raw_dispersion,
        calibrated_dispersion,
        raw_related_retention,
        calibrated_related_retention,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&[
        primary_dispersion_improved as u8,
        related_retention_guard_passed as u8,
        worst_dimension_guard_passed as u8,
        disposition as u8,
    ]);
    *hasher.finalize().as_bytes()
}

fn hash_dimension_metrics(hasher: &mut blake3::Hasher, metrics: &SemanticDimensionMetrics) {
    for value in [
        metrics.context_dimension,
        metrics.related_count,
        metrics.unrelated_count,
        metrics.ood_count,
        metrics.ood_exact_identity_differs_count,
        metrics.ood_raw_activation_count,
        metrics.ood_calibrated_activation_count,
    ] {
        hasher.update(&(value as u64).to_le_bytes());
    }
    for value in [
        metrics.raw_related_retention,
        metrics.calibrated_related_retention,
        metrics.raw_unrelated_false_activation,
        metrics.calibrated_unrelated_false_activation,
        metrics.mean_related_similarity,
        metrics.mean_unrelated_similarity,
        metrics.mean_related_specificity,
        metrics.mean_unrelated_specificity,
        metrics.mean_ood_similarity,
        metrics.mean_ood_specificity,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn dataset_partitions_are_frozen_and_disjoint_by_construction() {
        let reference = reference_contexts();
        let heldout = heldout_cases();
        let ood = ood_cases();

        assert_eq!(
            reference.len(),
            SYM_RSI_SEM_001_DIMENSIONS.len() * SYM_RSI_SEM_001_REFERENCE_COUNT_PER_DIMENSION
        );
        assert_eq!(
            heldout
                .iter()
                .filter(|case| case.regime == SemanticEvaluationRegime::HeldOutRelated)
                .count(),
            SYM_RSI_SEM_001_DIMENSIONS.len()
                * SYM_RSI_SEM_001_HELDOUT_ANCHOR_COUNT_PER_DIMENSION
                * 2
        );
        assert_eq!(
            heldout
                .iter()
                .filter(|case| case.regime == SemanticEvaluationRegime::HeldOutUnrelated)
                .count(),
            SYM_RSI_SEM_001_DIMENSIONS.len()
                * SYM_RSI_SEM_001_HELDOUT_ANCHOR_COUNT_PER_DIMENSION
        );
        assert_eq!(
            ood.len(),
            SYM_RSI_SEM_001_DIMENSIONS.len() * SYM_RSI_SEM_001_OOD_ANCHOR_COUNT_PER_DIMENSION
        );

        let reference_digest = context_corpus_digest("reference", &reference);
        assert_ne!(reference_digest, case_corpus_digest("heldout", &heldout));
        assert_ne!(reference_digest, case_corpus_digest("ood", &ood));
    }

    #[test]
    fn related_cases_are_exactly_distinct_from_their_anchor() {
        for case in heldout_cases()
            .into_iter()
            .filter(|case| case.regime == SemanticEvaluationRegime::HeldOutRelated)
        {
            assert_ne!(
                ContextIdentity::from_context(&case.anchor).unwrap(),
                ContextIdentity::from_context(&case.query).unwrap()
            );
        }
    }

    #[test]
    fn ood_cases_cross_the_encoder_clamp_and_keep_distinct_exact_identity() {
        for case in ood_cases() {
            assert!(case.query.iter().any(|value| value.abs() > 1.0));
            assert_ne!(
                ContextIdentity::from_context(&case.anchor).unwrap(),
                ContextIdentity::from_context(&case.query).unwrap()
            );
        }
    }

    #[test]
    fn sem_001_receipt_is_deterministic() {
        let first = run_sym_rsi_sem_001(TEST_SHA).unwrap();
        let second = run_sym_rsi_sem_001(TEST_SHA).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.dimension_metrics.len(), 3);
    }

    #[test]
    fn disposition_requires_primary_and_both_guards() {
        let receipt = run_sym_rsi_sem_001(TEST_SHA).unwrap();
        match receipt.disposition {
            SemanticCalibrationDisposition::Pass => {
                assert!(receipt.primary_dispersion_improved);
                assert!(receipt.related_retention_guard_passed);
                assert!(receipt.worst_dimension_false_activation_guard_passed);
            }
            SemanticCalibrationDisposition::Tradeoff => {
                assert!(receipt.primary_dispersion_improved);
                assert!(
                    !receipt.related_retention_guard_passed
                        || !receipt.worst_dimension_false_activation_guard_passed
                );
            }
            SemanticCalibrationDisposition::NotEstablished => {
                assert!(!receipt.primary_dispersion_improved);
            }
        }
    }

    #[test]
    fn invalid_subject_identity_fails_closed() {
        assert_eq!(
            run_sym_rsi_sem_001("not-a-sha"),
            Err(SemanticCalibrationExperimentError::InvalidSubjectSha)
        );
    }
}