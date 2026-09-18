// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runtime-only semantic retrieval for dream-derived action priors.
//!
//! This sidecar deliberately does not modify `DreamFeedbackBridge` persistence.
//! It indexes already-accepted `ActionPrior`s by collision-resistant context
//! identity and uses HDC similarity only to retrieve/attenuate candidate search
//! directions.
//!
//! The sidecar has no confidence API, evidence class, validation capability, or
//! serialization authority. Its outputs are search heuristics, not belief updates.

use std::collections::HashMap;

use super::dream_feedback::ActionPrior;
use super::semantic_context::{
    ContextIdentity, SemanticContextEncoder, SemanticContextError, SemanticContextIndex,
    SemanticContextMatch,
};
use super::semantic_null_calibration::{
    SemanticNullAssessment, SemanticNullCalibration, SemanticNullCalibrationError,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticPriorError {
    Context(SemanticContextError),
    NullCalibration(SemanticNullCalibrationError),
    CalibrationEncoderMismatch,
    ContextHashMismatch { expected: u64, observed: u64 },
    EmptyPreferredDirection,
    NonFiniteDirection { index: usize },
    InvalidPriorStrength,
}

impl From<SemanticContextError> for SemanticPriorError {
    fn from(value: SemanticContextError) -> Self {
        Self::Context(value)
    }
}

impl From<SemanticNullCalibrationError> for SemanticPriorError {
    fn from(value: SemanticNullCalibrationError) -> Self {
        Self::NullCalibration(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SemanticPriorRecord {
    identity: ContextIdentity,
    context_dimension: usize,
    preferred_direction: Vec<f32>,
    prior_strength: f64,
    generated_support_count: usize,
}

/// Retrieval-only candidate produced by semantic dream memory.
///
/// `effective_search_strength` may bias exploration/action search. It must not be
/// interpreted as epistemic confidence or empirical validation.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticPriorCandidate {
    pub identity: ContextIdentity,
    pub context_dimension: usize,
    pub preferred_direction: Vec<f32>,
    pub prior_strength: f64,
    pub generated_support_count: usize,
    pub similarity: f32,
    pub semantic_distance: f32,
    pub retrieval_weight: f32,
    pub effective_search_strength: f64,
}

/// Semantic prior candidate with an empirical encoded-context null assessment.
///
/// Calibration can only attenuate the pre-existing search strength. Neither the
/// tail rank nor the calibrated strength is evidence or epistemic confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedSemanticPriorCandidate {
    pub candidate: SemanticPriorCandidate,
    pub null_assessment: SemanticNullAssessment,
    pub calibrated_search_strength: f64,
}

/// Runtime semantic index over dream-derived action priors.
///
/// This type intentionally does not derive `Serialize`/`Deserialize`. Restarting
/// cannot promote or silently restore semantic retrieval state as evidence.
#[derive(Clone, Default)]
pub struct SemanticPriorMemory {
    contexts: SemanticContextIndex,
    priors: HashMap<ContextIdentity, SemanticPriorRecord>,
}

impl SemanticPriorMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.priors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.priors.is_empty()
    }

    pub fn context_encoder(&self) -> SemanticContextEncoder {
        self.contexts.encoder()
    }

    /// Index an already-accepted action prior under its exact numeric context.
    ///
    /// The context vector must reproduce the prior's historical fast hash. This
    /// prevents a caller from attaching an existing prior to an unrelated semantic
    /// context. Durable identity is still the BLAKE3 digest inside `ContextIdentity`.
    pub fn index_prior(
        &mut self,
        context: &[f32],
        prior: &ActionPrior,
    ) -> Result<ContextIdentity, SemanticPriorError> {
        validate_prior(prior)?;

        let identity = ContextIdentity::from_context(context)?;
        if identity.fast_hash != prior.context_hash {
            return Err(SemanticPriorError::ContextHashMismatch {
                expected: prior.context_hash,
                observed: identity.fast_hash,
            });
        }

        let indexed_identity = self.contexts.insert(context)?;
        debug_assert_eq!(identity, indexed_identity);

        self.priors.insert(
            identity,
            SemanticPriorRecord {
                identity,
                context_dimension: context.len(),
                preferred_direction: prior.preferred_direction.clone(),
                prior_strength: prior.strength,
                generated_support_count: prior.evidence_count,
            },
        );
        Ok(identity)
    }

    pub fn contains_exact(&self, identity: ContextIdentity) -> bool {
        self.priors.contains_key(&identity)
    }

    /// Retrieve semantically related action priors with a real `[0,1]` HDC cutoff.
    pub fn retrieve(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<SemanticPriorCandidate>, SemanticPriorError> {
        let matches = self.contexts.nearest(query, min_similarity, limit)?;
        Ok(self.materialize(matches))
    }

    /// Retrieve without a similarity cutoff.
    pub fn retrieve_unfiltered(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<SemanticPriorCandidate>, SemanticPriorError> {
        let matches = self.contexts.nearest_unfiltered(query, limit)?;
        Ok(self.materialize(matches))
    }

    /// Retrieve with empirical encoded-context null calibration.
    ///
    /// The calibration must use the exact same semantic encoder configuration and
    /// contain a stratum for `query.len()`. Cross-dimensional candidate contexts are
    /// excluded because a same-dimensional null does not calibrate their similarity.
    /// Existing raw retrieval remains available separately for callers that
    /// deliberately want uncalibrated behavior.
    pub fn retrieve_calibrated(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
        calibration: &SemanticNullCalibration,
    ) -> Result<Vec<CalibratedSemanticPriorCandidate>, SemanticPriorError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        self.validate_calibration(calibration)?;
        let matches = self
            .contexts
            .nearest(query, min_similarity, self.priors.len())?;
        self.materialize_calibrated(matches, query.len(), limit, calibration)
    }

    /// Calibrated retrieval without a raw-similarity cutoff.
    pub fn retrieve_calibrated_unfiltered(
        &self,
        query: &[f32],
        limit: usize,
        calibration: &SemanticNullCalibration,
    ) -> Result<Vec<CalibratedSemanticPriorCandidate>, SemanticPriorError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        self.validate_calibration(calibration)?;
        let matches = self
            .contexts
            .nearest_unfiltered(query, self.priors.len())?;
        self.materialize_calibrated(matches, query.len(), limit, calibration)
    }

    fn validate_calibration(
        &self,
        calibration: &SemanticNullCalibration,
    ) -> Result<(), SemanticPriorError> {
        if !calibration.matches_encoder(self.contexts.encoder()) {
            return Err(SemanticPriorError::CalibrationEncoderMismatch);
        }
        Ok(())
    }

    fn materialize(&self, matches: Vec<SemanticContextMatch>) -> Vec<SemanticPriorCandidate> {
        matches
            .into_iter()
            .filter_map(|matched| {
                let prior = self.priors.get(&matched.identity)?;
                let effective_search_strength =
                    (prior.prior_strength * f64::from(matched.retrieval_weight)).clamp(0.0, 1.0);
                Some(SemanticPriorCandidate {
                    identity: prior.identity,
                    context_dimension: prior.context_dimension,
                    preferred_direction: prior.preferred_direction.clone(),
                    prior_strength: prior.prior_strength,
                    generated_support_count: prior.generated_support_count,
                    similarity: matched.similarity,
                    semantic_distance: matched.semantic_distance,
                    retrieval_weight: matched.retrieval_weight,
                    effective_search_strength,
                })
            })
            .collect()
    }

    fn materialize_calibrated(
        &self,
        matches: Vec<SemanticContextMatch>,
        query_dimension: usize,
        limit: usize,
        calibration: &SemanticNullCalibration,
    ) -> Result<Vec<CalibratedSemanticPriorCandidate>, SemanticPriorError> {
        let mut calibrated = Vec::new();
        for candidate in self.materialize(matches) {
            if candidate.context_dimension != query_dimension {
                continue;
            }
            let null_assessment = calibration.assess(query_dimension, candidate.similarity)?;
            let calibrated_search_strength = (candidate.effective_search_strength
                * f64::from(null_assessment.retrieval_specificity))
            .clamp(0.0, candidate.effective_search_strength);
            calibrated.push(CalibratedSemanticPriorCandidate {
                candidate,
                null_assessment,
                calibrated_search_strength,
            });
        }

        calibrated.sort_by(|left, right| {
            right
                .calibrated_search_strength
                .total_cmp(&left.calibrated_search_strength)
                .then_with(|| {
                    right
                        .candidate
                        .similarity
                        .total_cmp(&left.candidate.similarity)
                })
                .then_with(|| {
                    left.candidate
                        .identity
                        .exact_digest
                        .cmp(&right.candidate.identity.exact_digest)
                })
        });
        calibrated.truncate(limit);
        Ok(calibrated)
    }
}

fn validate_prior(prior: &ActionPrior) -> Result<(), SemanticPriorError> {
    if !prior.strength.is_finite() || !(0.0..=1.0).contains(&prior.strength) {
        return Err(SemanticPriorError::InvalidPriorStrength);
    }
    if prior.preferred_direction.is_empty() {
        return Err(SemanticPriorError::EmptyPreferredDirection);
    }
    for (index, value) in prior.preferred_direction.iter().enumerate() {
        if !value.is_finite() {
            return Err(SemanticPriorError::NonFiniteDirection { index });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::semantic_null_calibration::SemanticNullConfig;
    use super::*;

    fn prior(context: &[f32], strength: f64) -> ActionPrior {
        ActionPrior {
            context_hash: super::super::dream_feedback::hash_context(context),
            preferred_direction: vec![0.25, -0.5, 0.75],
            strength,
            evidence_count: 3,
        }
    }

    fn calibration(encoder: SemanticContextEncoder) -> SemanticNullCalibration {
        let contexts = vec![
            vec![-1.0],
            vec![-0.7],
            vec![-0.3],
            vec![0.1],
            vec![0.5],
            vec![0.9],
        ];
        SemanticNullCalibration::build(
            &contexts,
            encoder,
            SemanticNullConfig {
                max_reference_contexts_per_dimension: 32,
                min_null_pairs_per_dimension: 3,
            },
        )
        .unwrap()
    }

    #[test]
    fn exact_context_preserves_prior_strength_for_search() {
        let context = [0.2, -0.4, 0.8];
        let mut memory = SemanticPriorMemory::new();
        let identity = memory.index_prior(&context, &prior(&context, 0.8)).unwrap();

        let candidates = memory.retrieve(&context, 1.0, 1).unwrap();
        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert_eq!(candidate.identity, identity);
        assert_eq!(candidate.context_dimension, context.len());
        assert_eq!(candidate.similarity, 1.0);
        assert_eq!(candidate.semantic_distance, 0.0);
        assert_eq!(candidate.retrieval_weight, 1.0);
        assert_eq!(candidate.effective_search_strength, 0.8);
    }

    #[test]
    fn semantic_retrieval_can_only_attenuate_generated_prior_strength() {
        let context = [0.0, 0.0, 0.0, 0.0];
        let query = [0.4, 0.0, 0.0, 0.0];
        let mut memory = SemanticPriorMemory::new();
        memory.index_prior(&context, &prior(&context, 0.9)).unwrap();

        let candidates = memory.retrieve_unfiltered(&query, 1).unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].effective_search_strength <= candidates[0].prior_strength);
        assert!((0.0..=1.0).contains(&candidates[0].retrieval_weight));
    }

    #[test]
    fn calibrated_retrieval_can_only_further_attenuate_search_strength() {
        let context = [0.2];
        let mut memory = SemanticPriorMemory::new();
        memory.index_prior(&context, &prior(&context, 0.9)).unwrap();
        let calibration = calibration(memory.context_encoder());

        let raw = memory.retrieve(&context, 0.0, 1).unwrap();
        let calibrated = memory
            .retrieve_calibrated(&context, 0.0, 1, &calibration)
            .unwrap();

        assert_eq!(calibrated.len(), 1);
        assert!(
            calibrated[0].calibrated_search_strength <= raw[0].effective_search_strength
        );
        assert_eq!(calibrated[0].candidate.identity, raw[0].identity);
        assert!(calibrated[0].null_assessment.empirical_tail_fraction > 0.0);
    }

    #[test]
    fn calibrated_retrieval_excludes_cross_dimensional_candidates() {
        let one_dimensional = [0.2];
        let two_dimensional = [0.2, 0.3];
        let mut memory = SemanticPriorMemory::new();
        memory
            .index_prior(&one_dimensional, &prior(&one_dimensional, 0.9))
            .unwrap();
        memory
            .index_prior(&two_dimensional, &prior(&two_dimensional, 0.9))
            .unwrap();
        let calibration = calibration(memory.context_encoder());

        let candidates = memory
            .retrieve_calibrated_unfiltered(&one_dimensional, 10, &calibration)
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].candidate.context_dimension, 1);
    }

    #[test]
    fn calibrated_retrieval_fails_closed_on_encoder_mismatch() {
        let context = [0.2];
        let mut memory = SemanticPriorMemory::new();
        memory.index_prior(&context, &prior(&context, 0.9)).unwrap();
        let different_encoder = SemanticContextEncoder::new(32, 1.0).unwrap();
        let calibration = calibration(different_encoder);

        assert_eq!(
            memory.retrieve_calibrated(&context, 0.0, 1, &calibration),
            Err(SemanticPriorError::CalibrationEncoderMismatch)
        );
    }

    #[test]
    fn calibrated_retrieval_fails_closed_without_matching_dimension() {
        let context = [0.2, 0.3];
        let mut memory = SemanticPriorMemory::new();
        memory.index_prior(&context, &prior(&context, 0.9)).unwrap();
        let calibration = calibration(memory.context_encoder());

        assert!(matches!(
            memory.retrieve_calibrated(&context, 0.0, 1, &calibration),
            Err(SemanticPriorError::NullCalibration(
                SemanticNullCalibrationError::MissingDimension {
                    context_dimension: 2
                }
            ))
        ));
    }

    #[test]
    fn context_hash_mismatch_fails_closed() {
        let context = [0.1, 0.2];
        let mut wrong_prior = prior(&context, 0.5);
        wrong_prior.context_hash ^= 1;
        let mut memory = SemanticPriorMemory::new();

        assert!(matches!(
            memory.index_prior(&context, &wrong_prior),
            Err(SemanticPriorError::ContextHashMismatch { .. })
        ));
        assert!(memory.is_empty());
    }

    #[test]
    fn malformed_prior_fails_closed() {
        let context = [0.1];
        let mut memory = SemanticPriorMemory::new();

        let mut nonfinite = prior(&context, 0.5);
        nonfinite.preferred_direction[1] = f32::NAN;
        assert_eq!(
            memory.index_prior(&context, &nonfinite),
            Err(SemanticPriorError::NonFiniteDirection { index: 1 })
        );

        let invalid_strength = prior(&context, 1.1);
        assert_eq!(
            memory.index_prior(&context, &invalid_strength),
            Err(SemanticPriorError::InvalidPriorStrength)
        );
    }

    #[test]
    fn exact_context_identity_remains_collision_resistant_sidecar_key() {
        let first_context = [0.1, 0.2, 0.3];
        let second_context = [0.1, 0.2, 0.31];
        let mut memory = SemanticPriorMemory::new();
        let first = memory
            .index_prior(&first_context, &prior(&first_context, 0.4))
            .unwrap();
        let second = memory
            .index_prior(&second_context, &prior(&second_context, 0.6))
            .unwrap();

        assert_ne!(first.exact_digest, second.exact_digest);
        assert_eq!(memory.len(), 2);
        assert!(memory.contains_exact(first));
        assert!(memory.contains_exact(second));
    }
}
