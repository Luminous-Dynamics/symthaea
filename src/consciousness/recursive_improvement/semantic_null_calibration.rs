// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Empirical null calibration for semantic HDC retrieval.
//!
//! Raw `BinaryHV` similarity is a matching-bit fraction. Random independent
//! 16,384-bit vectors center near 0.5, but Symthaea's semantic context encoder is
//! structured (level encoding + role binding + bundling), so an IID Bernoulli null
//! is not an adequate calibration model for encoded contexts.
//!
//! This module therefore learns a conservative, dimension-stratified empirical null
//! from *encoded, non-identical reference contexts*. The resulting tail rank is a
//! retrieval-specificity signal only. It is deliberately **not** called a p-value and
//! grants no evidence, validation, or confidence authority.

use std::collections::BTreeMap;

use super::semantic_context::{ContextIdentity, SemanticContextEncoder, SemanticContextError};
use symthaea_core::hdc::BinaryHV;

pub const SEMANTIC_NULL_CALIBRATION_SCHEMA: &str = "symthaea.semantic-null-calibration.v1";
pub const DEFAULT_MAX_NULL_CONTEXTS_PER_DIMENSION: usize = 96;
pub const DEFAULT_MIN_NULL_PAIRS_PER_DIMENSION: usize = 128;
const MAX_REFERENCE_CONTEXTS_HARD_LIMIT: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticNullConfig {
    /// Maximum exact-distinct reference contexts retained per context dimension.
    ///
    /// When a stratum is larger, references are sorted by collision-resistant digest
    /// before truncation. Digest order gives a deterministic content-independent
    /// subsample without introducing runtime RNG state.
    pub max_reference_contexts_per_dimension: usize,
    /// Minimum number of distinct-context pair similarities required before a
    /// dimensional stratum is considered calibrated.
    pub min_null_pairs_per_dimension: usize,
}

impl Default for SemanticNullConfig {
    fn default() -> Self {
        Self {
            max_reference_contexts_per_dimension: DEFAULT_MAX_NULL_CONTEXTS_PER_DIMENSION,
            min_null_pairs_per_dimension: DEFAULT_MIN_NULL_PAIRS_PER_DIMENSION,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticNullCalibrationError {
    Context(SemanticContextError),
    MaxReferenceContextsTooSmall,
    MaxReferenceContextsTooLarge,
    MinNullPairsTooSmall,
    MinNullPairsExceedCapacity,
    NoCalibratableStrata,
    InvalidSimilarity,
    MissingDimension { context_dimension: usize },
}

impl From<SemanticContextError> for SemanticNullCalibrationError {
    fn from(value: SemanticContextError) -> Self {
        Self::Context(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticNullCalibrationIdentity {
    pub digest: [u8; 32],
    pub strata_count: usize,
    pub null_pair_count: usize,
}

impl SemanticNullCalibrationIdentity {
    pub fn to_hex(self) -> String {
        hex::encode(self.digest)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticNullStratumSummary {
    pub context_dimension: usize,
    pub reference_context_count: usize,
    pub null_pair_count: usize,
    pub median_similarity: f32,
    pub p95_similarity: f32,
}

/// Conservative empirical location of one raw semantic similarity inside the
/// encoded reference-context null distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticNullAssessment {
    pub context_dimension: usize,
    pub similarity: f32,
    pub null_pair_count: usize,
    pub null_median_similarity: f32,
    pub null_p95_similarity: f32,
    /// Smoothed fraction of null pairs with similarity >= the observed similarity.
    ///
    /// `(exceedances + 1) / (n + 1)` prevents a finite reference corpus from ever
    /// claiming an impossible zero tail. This is an empirical rank, not a p-value.
    pub empirical_tail_fraction: f32,
    /// `1 - empirical_tail_fraction`, suitable only for attenuating/ranking semantic
    /// retrieval candidates. It is not epistemic confidence.
    pub retrieval_specificity: f32,
}

#[derive(Debug, Clone)]
struct SemanticNullStratum {
    context_dimension: usize,
    reference_digests: Vec<[u8; 32]>,
    sorted_similarities: Vec<f32>,
}

impl SemanticNullStratum {
    fn summary(&self) -> SemanticNullStratumSummary {
        SemanticNullStratumSummary {
            context_dimension: self.context_dimension,
            reference_context_count: self.reference_digests.len(),
            null_pair_count: self.sorted_similarities.len(),
            median_similarity: quantile(&self.sorted_similarities, 0.50),
            p95_similarity: quantile(&self.sorted_similarities, 0.95),
        }
    }

    fn assess(&self, similarity: f32) -> SemanticNullAssessment {
        let first_ge = self
            .sorted_similarities
            .partition_point(|candidate| *candidate < similarity);
        let exceedances = self.sorted_similarities.len() - first_ge;
        let n = self.sorted_similarities.len();
        let empirical_tail_fraction = (exceedances as f32 + 1.0) / (n as f32 + 1.0);
        let summary = self.summary();

        SemanticNullAssessment {
            context_dimension: self.context_dimension,
            similarity,
            null_pair_count: n,
            null_median_similarity: summary.median_similarity,
            null_p95_similarity: summary.p95_similarity,
            empirical_tail_fraction,
            retrieval_specificity: (1.0 - empirical_tail_fraction).clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone)]
struct EncodedReference {
    identity: ContextIdentity,
    key: BinaryHV,
}

/// Runtime-only empirical calibration bound to one semantic encoder configuration.
///
/// No serde traits are intentionally provided. A durable calibration should be
/// regenerated/bound through an explicit reproducibility capsule rather than silently
/// surviving an encoder change.
#[derive(Clone)]
pub struct SemanticNullCalibration {
    encoder: SemanticContextEncoder,
    config: SemanticNullConfig,
    identity: SemanticNullCalibrationIdentity,
    strata: Vec<SemanticNullStratum>,
}

impl SemanticNullCalibration {
    pub fn build(
        reference_contexts: &[Vec<f32>],
        encoder: SemanticContextEncoder,
        config: SemanticNullConfig,
    ) -> Result<Self, SemanticNullCalibrationError> {
        validate_config(config)?;

        let mut grouped: BTreeMap<usize, Vec<EncodedReference>> = BTreeMap::new();
        for context in reference_contexts {
            let identity = ContextIdentity::from_context(context)?;
            let key = encoder.encode(context)?;
            grouped
                .entry(context.len())
                .or_default()
                .push(EncodedReference { identity, key });
        }

        let mut strata = Vec::new();
        for (context_dimension, references) in grouped.iter_mut() {
            references.sort_by(|left, right| {
                left.identity
                    .exact_digest
                    .cmp(&right.identity.exact_digest)
            });
            references.dedup_by(|left, right| {
                left.identity.exact_digest == right.identity.exact_digest
            });
            references.truncate(config.max_reference_contexts_per_dimension);

            if references.len() < 2 {
                continue;
            }

            let pair_capacity = references.len() * (references.len() - 1) / 2;
            if pair_capacity < config.min_null_pairs_per_dimension {
                continue;
            }

            let mut similarities = Vec::with_capacity(pair_capacity);
            for left_index in 0..references.len() - 1 {
                for right_index in left_index + 1..references.len() {
                    similarities.push(
                        references[left_index]
                            .key
                            .similarity(&references[right_index].key),
                    );
                }
            }
            similarities.sort_by(f32::total_cmp);

            strata.push(SemanticNullStratum {
                context_dimension: *context_dimension,
                reference_digests: references
                    .iter()
                    .map(|reference| reference.identity.exact_digest)
                    .collect(),
                sorted_similarities: similarities,
            });
        }

        if strata.is_empty() {
            return Err(SemanticNullCalibrationError::NoCalibratableStrata);
        }

        let identity = calibration_identity(encoder, config, &strata);
        Ok(Self {
            encoder,
            config,
            identity,
            strata,
        })
    }

    pub fn build_default(
        reference_contexts: &[Vec<f32>],
    ) -> Result<Self, SemanticNullCalibrationError> {
        Self::build(
            reference_contexts,
            SemanticContextEncoder::default(),
            SemanticNullConfig::default(),
        )
    }

    pub fn encoder(&self) -> SemanticContextEncoder {
        self.encoder
    }

    pub fn config(&self) -> SemanticNullConfig {
        self.config
    }

    pub fn identity(&self) -> SemanticNullCalibrationIdentity {
        self.identity
    }

    pub fn matches_encoder(&self, encoder: SemanticContextEncoder) -> bool {
        self.encoder == encoder
    }

    pub fn strata_count(&self) -> usize {
        self.strata.len()
    }

    pub fn total_null_pairs(&self) -> usize {
        self.identity.null_pair_count
    }

    pub fn summaries(&self) -> Vec<SemanticNullStratumSummary> {
        self.strata.iter().map(SemanticNullStratum::summary).collect()
    }

    pub fn has_dimension(&self, context_dimension: usize) -> bool {
        self.strata
            .binary_search_by_key(&context_dimension, |stratum| stratum.context_dimension)
            .is_ok()
    }

    /// Assess a raw similarity against the same-dimensional encoded reference null.
    ///
    /// The returned specificity is a retrieval attenuation/ranking value only.
    pub fn assess(
        &self,
        context_dimension: usize,
        similarity: f32,
    ) -> Result<SemanticNullAssessment, SemanticNullCalibrationError> {
        if !similarity.is_finite() || !(0.0..=1.0).contains(&similarity) {
            return Err(SemanticNullCalibrationError::InvalidSimilarity);
        }
        let index = self
            .strata
            .binary_search_by_key(&context_dimension, |stratum| stratum.context_dimension)
            .map_err(|_| SemanticNullCalibrationError::MissingDimension {
                context_dimension,
            })?;
        Ok(self.strata[index].assess(similarity))
    }
}

fn validate_config(config: SemanticNullConfig) -> Result<(), SemanticNullCalibrationError> {
    if config.max_reference_contexts_per_dimension < 2 {
        return Err(SemanticNullCalibrationError::MaxReferenceContextsTooSmall);
    }
    if config.max_reference_contexts_per_dimension > MAX_REFERENCE_CONTEXTS_HARD_LIMIT {
        return Err(SemanticNullCalibrationError::MaxReferenceContextsTooLarge);
    }
    if config.min_null_pairs_per_dimension == 0 {
        return Err(SemanticNullCalibrationError::MinNullPairsTooSmall);
    }
    let max_pair_capacity = config.max_reference_contexts_per_dimension
        * (config.max_reference_contexts_per_dimension - 1)
        / 2;
    if config.min_null_pairs_per_dimension > max_pair_capacity {
        return Err(SemanticNullCalibrationError::MinNullPairsExceedCapacity);
    }
    Ok(())
}

fn calibration_identity(
    encoder: SemanticContextEncoder,
    config: SemanticNullConfig,
    strata: &[SemanticNullStratum],
) -> SemanticNullCalibrationIdentity {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SEMANTIC_NULL_CALIBRATION_SCHEMA.as_bytes());
    hasher.update(&(encoder.levels() as u64).to_le_bytes());
    hasher.update(&encoder.clamp_abs().to_bits().to_le_bytes());
    hasher.update(&(config.max_reference_contexts_per_dimension as u64).to_le_bytes());
    hasher.update(&(config.min_null_pairs_per_dimension as u64).to_le_bytes());
    hasher.update(&(strata.len() as u64).to_le_bytes());

    let mut total_null_pairs = 0usize;
    for stratum in strata {
        hasher.update(&(stratum.context_dimension as u64).to_le_bytes());
        hasher.update(&(stratum.reference_digests.len() as u64).to_le_bytes());
        for digest in &stratum.reference_digests {
            hasher.update(digest);
        }
        hasher.update(&(stratum.sorted_similarities.len() as u64).to_le_bytes());
        for similarity in &stratum.sorted_similarities {
            hasher.update(&similarity.to_bits().to_le_bytes());
        }
        total_null_pairs += stratum.sorted_similarities.len();
    }

    SemanticNullCalibrationIdentity {
        digest: *hasher.finalize().as_bytes(),
        strata_count: strata.len(),
        null_pair_count: total_null_pairs,
    }
}

fn quantile(sorted: &[f32], probability: f32) -> f32 {
    debug_assert!(!sorted.is_empty());
    let rank = ((probability.clamp(0.0, 1.0) * sorted.len() as f32).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[rank]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SemanticNullConfig {
        SemanticNullConfig {
            max_reference_contexts_per_dimension: 32,
            min_null_pairs_per_dimension: 3,
        }
    }

    fn one_dimensional_contexts() -> Vec<Vec<f32>> {
        vec![
            vec![-1.0],
            vec![-0.7],
            vec![-0.3],
            vec![0.1],
            vec![0.5],
            vec![0.9],
        ]
    }

    #[test]
    fn calibration_is_deterministic_under_input_reordering() {
        let contexts = one_dimensional_contexts();
        let mut reversed = contexts.clone();
        reversed.reverse();

        let first = SemanticNullCalibration::build(
            &contexts,
            SemanticContextEncoder::default(),
            config(),
        )
        .unwrap();
        let second = SemanticNullCalibration::build(
            &reversed,
            SemanticContextEncoder::default(),
            config(),
        )
        .unwrap();

        assert_eq!(first.identity(), second.identity());
        assert_eq!(first.summaries(), second.summaries());
    }

    #[test]
    fn reference_corpus_identity_changes_even_when_shape_is_unchanged() {
        let first_contexts = one_dimensional_contexts();
        let mut second_contexts = first_contexts.clone();
        second_contexts[0][0] = -0.95;

        let first = SemanticNullCalibration::build(
            &first_contexts,
            SemanticContextEncoder::default(),
            config(),
        )
        .unwrap();
        let second = SemanticNullCalibration::build(
            &second_contexts,
            SemanticContextEncoder::default(),
            config(),
        )
        .unwrap();

        assert_ne!(first.identity(), second.identity());
    }

    #[test]
    fn exact_duplicate_contexts_do_not_inflate_null_pairs() {
        let mut contexts = one_dimensional_contexts();
        contexts.push(vec![0.5]);
        contexts.push(vec![0.5]);
        let calibration = SemanticNullCalibration::build(
            &contexts,
            SemanticContextEncoder::default(),
            config(),
        )
        .unwrap();

        let summary = calibration.summaries()[0];
        assert_eq!(summary.reference_context_count, 6);
        assert_eq!(summary.null_pair_count, 15);
    }

    #[test]
    fn calibration_is_dimension_stratified() {
        let mut contexts = one_dimensional_contexts();
        contexts.extend([
            vec![-1.0, -1.0],
            vec![-0.5, 0.2],
            vec![0.0, 0.5],
            vec![0.7, -0.4],
        ]);
        let calibration = SemanticNullCalibration::build(
            &contexts,
            SemanticContextEncoder::default(),
            config(),
        )
        .unwrap();

        assert!(calibration.has_dimension(1));
        assert!(calibration.has_dimension(2));
        assert_eq!(calibration.strata_count(), 2);
    }

    #[test]
    fn higher_similarity_never_has_lower_retrieval_specificity() {
        let calibration = SemanticNullCalibration::build(
            &one_dimensional_contexts(),
            SemanticContextEncoder::default(),
            config(),
        )
        .unwrap();

        let lower = calibration.assess(1, 0.5).unwrap();
        let higher = calibration.assess(1, 1.0).unwrap();
        assert!(higher.retrieval_specificity >= lower.retrieval_specificity);
        assert!(higher.empirical_tail_fraction <= lower.empirical_tail_fraction);
    }

    #[test]
    fn finite_reference_null_never_claims_zero_tail() {
        let calibration = SemanticNullCalibration::build(
            &one_dimensional_contexts(),
            SemanticContextEncoder::default(),
            config(),
        )
        .unwrap();
        let assessment = calibration.assess(1, 1.0).unwrap();

        assert!(assessment.empirical_tail_fraction > 0.0);
        assert!(assessment.retrieval_specificity < 1.0);
    }

    #[test]
    fn missing_dimension_fails_closed() {
        let calibration = SemanticNullCalibration::build(
            &one_dimensional_contexts(),
            SemanticContextEncoder::default(),
            config(),
        )
        .unwrap();
        assert_eq!(
            calibration.assess(3, 0.8),
            Err(SemanticNullCalibrationError::MissingDimension {
                context_dimension: 3,
            })
        );
    }

    #[test]
    fn impossible_config_fails_before_corpus_processing() {
        let impossible = SemanticNullConfig {
            max_reference_contexts_per_dimension: 4,
            min_null_pairs_per_dimension: 7,
        };
        assert!(matches!(
            SemanticNullCalibration::build(
                &one_dimensional_contexts(),
                SemanticContextEncoder::default(),
                impossible,
            ),
            Err(SemanticNullCalibrationError::MinNullPairsExceedCapacity)
        ));
    }

    #[test]
    fn malformed_reference_context_fails_closed() {
        let mut contexts = one_dimensional_contexts();
        contexts.push(vec![f32::NAN]);
        assert!(matches!(
            SemanticNullCalibration::build(
                &contexts,
                SemanticContextEncoder::default(),
                config(),
            ),
            Err(SemanticNullCalibrationError::Context(
                SemanticContextError::NonFiniteValue { index: 0 }
            ))
        ));
    }
}
