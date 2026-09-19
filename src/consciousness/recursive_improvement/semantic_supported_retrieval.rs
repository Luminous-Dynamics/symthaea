// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Support-aware wrappers around semantic dream-prior retrieval.
//!
//! Existing raw and calibrated retrieval APIs remain available for diagnostics and
//! experiments. This module adds two stricter operational layers:
//!
//! - query-only support checking for an ordinary `SemanticPriorMemory`;
//! - support-bound memory that captures candidate-source support at indexing time and
//!   requires both query and candidate source to be inside nominal encoder support.
//!
//! Support is not evidence, confidence, or validation. It only says whether the
//! numeric vectors being compared were represented without clamp saturation.

use std::collections::HashMap;

use super::dream_feedback::ActionPrior;
use super::semantic_context::{ContextIdentity, SemanticContextEncoder};
use super::semantic_null_calibration::SemanticNullCalibration;
use super::semantic_prior::{
    CalibratedSemanticPriorCandidate, SemanticPriorCandidate, SemanticPriorError,
    SemanticPriorMemory,
};
use super::semantic_support::{
    SemanticContextSupport, SemanticSupportError, assess_semantic_context_support,
};

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticQuerySupportError {
    Support(SemanticSupportError),
    Prior(SemanticPriorError),
    QueryOutsideNominalSupport { support: SemanticContextSupport },
}

impl From<SemanticSupportError> for SemanticQuerySupportError {
    fn from(value: SemanticSupportError) -> Self {
        Self::Support(value)
    }
}

impl From<SemanticPriorError> for SemanticQuerySupportError {
    fn from(value: SemanticPriorError) -> Self {
        Self::Prior(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuerySupportedSemanticRetrieval {
    pub query_support: SemanticContextSupport,
    pub candidates: Vec<CalibratedSemanticPriorCandidate>,
}

/// Add a query-support-aware operational retrieval path without changing the
/// historical raw/calibrated methods on `SemanticPriorMemory`.
pub trait QuerySupportedSemanticPriorRetrieval {
    fn retrieve_calibrated_with_supported_query(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
        calibration: &SemanticNullCalibration,
    ) -> Result<QuerySupportedSemanticRetrieval, SemanticQuerySupportError>;
}

impl QuerySupportedSemanticPriorRetrieval for SemanticPriorMemory {
    fn retrieve_calibrated_with_supported_query(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
        calibration: &SemanticNullCalibration,
    ) -> Result<QuerySupportedSemanticRetrieval, SemanticQuerySupportError> {
        let support = assess_semantic_context_support(self.context_encoder(), query)?;
        if !support.within_nominal_support() {
            return Err(SemanticQuerySupportError::QueryOutsideNominalSupport {
                support,
            });
        }

        let candidates = self.retrieve_calibrated(query, min_similarity, limit, calibration)?;
        Ok(QuerySupportedSemanticRetrieval {
            query_support: support,
            candidates,
        })
    }
}

/// Raw semantic candidate whose query and indexed source context were both inside
/// the encoder's nominal scalar support.
#[derive(Debug, Clone, PartialEq)]
pub struct FullySupportedSemanticPriorCandidate {
    pub candidate: SemanticPriorCandidate,
    pub query_support: SemanticContextSupport,
    pub candidate_source_support: SemanticContextSupport,
}

/// Calibrated semantic candidate whose query and indexed source context were both
/// inside the encoder's nominal scalar support.
#[derive(Debug, Clone, PartialEq)]
pub struct FullySupportedCalibratedSemanticPriorCandidate {
    pub candidate: CalibratedSemanticPriorCandidate,
    pub query_support: SemanticContextSupport,
    pub candidate_source_support: SemanticContextSupport,
}

/// Runtime-only semantic-prior memory that binds source-support provenance at the
/// same indexing event as each candidate prior.
///
/// The inner raw memory is intentionally not exposed mutably. A candidate therefore
/// cannot enter this strict path without corresponding source-support metadata.
#[derive(Clone, Default)]
pub struct SupportBoundSemanticPriorMemory {
    inner: SemanticPriorMemory,
    source_support: HashMap<ContextIdentity, SemanticContextSupport>,
}

impl SupportBoundSemanticPriorMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn context_encoder(&self) -> SemanticContextEncoder {
        self.inner.context_encoder()
    }

    pub fn source_support(&self, identity: ContextIdentity) -> Option<SemanticContextSupport> {
        self.source_support.get(&identity).copied()
    }

    /// Index an accepted prior and bind its source-support diagnostic atomically.
    ///
    /// Support assessment happens before mutation. The support map is updated only
    /// after the underlying semantic-prior index accepts the exact same context.
    pub fn index_prior(
        &mut self,
        context: &[f32],
        prior: &ActionPrior,
    ) -> Result<ContextIdentity, SemanticQuerySupportError> {
        let support = assess_semantic_context_support(self.inner.context_encoder(), context)?;
        let identity = self.inner.index_prior(context, prior)?;
        self.source_support.insert(identity, support);
        Ok(identity)
    }

    /// Strict raw retrieval requiring nominal support on both sides and the same
    /// numeric context dimensionality.
    pub fn retrieve_fully_supported(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
    ) -> Result<Vec<FullySupportedSemanticPriorCandidate>, SemanticQuerySupportError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query_support = assess_semantic_context_support(self.inner.context_encoder(), query)?;
        if !query_support.within_nominal_support() {
            return Err(SemanticQuerySupportError::QueryOutsideNominalSupport {
                support: query_support,
            });
        }

        let candidates = self
            .inner
            .retrieve(query, min_similarity, self.inner.len())?;
        Ok(self.filter_raw_candidates(candidates, query_support, limit))
    }

    /// Strict calibrated retrieval requiring nominal support on both sides.
    ///
    /// The inner query requests the full candidate set so an unsupported high-ranked
    /// candidate cannot consume the caller's final `limit` before filtering.
    pub fn retrieve_calibrated_fully_supported(
        &self,
        query: &[f32],
        min_similarity: f32,
        limit: usize,
        calibration: &SemanticNullCalibration,
    ) -> Result<Vec<FullySupportedCalibratedSemanticPriorCandidate>, SemanticQuerySupportError>
    {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query_support = assess_semantic_context_support(self.inner.context_encoder(), query)?;
        if !query_support.within_nominal_support() {
            return Err(SemanticQuerySupportError::QueryOutsideNominalSupport {
                support: query_support,
            });
        }

        let candidates = self.inner.retrieve_calibrated(
            query,
            min_similarity,
            self.inner.len(),
            calibration,
        )?;
        Ok(self.filter_calibrated_candidates(candidates, query_support, limit))
    }

    fn filter_raw_candidates(
        &self,
        candidates: Vec<SemanticPriorCandidate>,
        query_support: SemanticContextSupport,
        limit: usize,
    ) -> Vec<FullySupportedSemanticPriorCandidate> {
        candidates
            .into_iter()
            .filter_map(|candidate| {
                if candidate.context_dimension != query_support.context_dimension {
                    return None;
                }
                let source_support = self.source_support.get(&candidate.identity).copied()?;
                if source_support.context_dimension != candidate.context_dimension
                    || !source_support.within_nominal_support()
                {
                    return None;
                }
                Some(FullySupportedSemanticPriorCandidate {
                    candidate,
                    query_support,
                    candidate_source_support: source_support,
                })
            })
            .take(limit)
            .collect()
    }

    fn filter_calibrated_candidates(
        &self,
        candidates: Vec<CalibratedSemanticPriorCandidate>,
        query_support: SemanticContextSupport,
        limit: usize,
    ) -> Vec<FullySupportedCalibratedSemanticPriorCandidate> {
        candidates
            .into_iter()
            .filter_map(|candidate| {
                let source_support = self
                    .source_support
                    .get(&candidate.candidate.identity)
                    .copied()?;
                if source_support.context_dimension != query_support.context_dimension
                    || source_support.context_dimension != candidate.candidate.context_dimension
                    || !source_support.within_nominal_support()
                {
                    return None;
                }
                Some(FullySupportedCalibratedSemanticPriorCandidate {
                    candidate,
                    query_support,
                    candidate_source_support: source_support,
                })
            })
            .take(limit)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::dream_feedback::hash_context;
    use super::super::semantic_null_calibration::SemanticNullConfig;
    use super::*;

    fn prior(context: &[f32], strength: f64) -> ActionPrior {
        ActionPrior {
            context_hash: hash_context(context),
            preferred_direction: vec![0.2, -0.3, 0.4],
            strength,
            evidence_count: 1,
        }
    }

    fn calibration(memory: &SemanticPriorMemory) -> SemanticNullCalibration {
        let references = (0..8)
            .map(|index| vec![-0.9 + index as f32 * 0.25])
            .collect::<Vec<_>>();
        SemanticNullCalibration::build(
            &references,
            memory.context_encoder(),
            SemanticNullConfig {
                max_reference_contexts_per_dimension: 8,
                min_null_pairs_per_dimension: 3,
            },
        )
        .unwrap()
    }

    fn strict_calibration(memory: &SupportBoundSemanticPriorMemory) -> SemanticNullCalibration {
        let references = (0..8)
            .map(|index| vec![-0.9 + index as f32 * 0.25])
            .collect::<Vec<_>>();
        SemanticNullCalibration::build(
            &references,
            memory.context_encoder(),
            SemanticNullConfig {
                max_reference_contexts_per_dimension: 8,
                min_null_pairs_per_dimension: 3,
            },
        )
        .unwrap()
    }

    #[test]
    fn in_support_query_can_use_calibrated_retrieval() {
        let context = [0.2];
        let mut memory = SemanticPriorMemory::new();
        memory.index_prior(&context, &prior(&context, 0.8)).unwrap();
        let calibration = calibration(&memory);

        let result = memory
            .retrieve_calibrated_with_supported_query(&context, 0.0, 4, &calibration)
            .unwrap();
        assert!(result.query_support.within_nominal_support());
        assert_eq!(result.candidates.len(), 1);
    }

    #[test]
    fn clamp_saturated_query_fails_before_semantic_retrieval() {
        let context = [0.2];
        let mut memory = SemanticPriorMemory::new();
        memory.index_prior(&context, &prior(&context, 0.8)).unwrap();
        let calibration = calibration(&memory);

        let error = memory
            .retrieve_calibrated_with_supported_query(&[2.5], 0.0, 4, &calibration)
            .unwrap_err();
        match error {
            SemanticQuerySupportError::QueryOutsideNominalSupport { support } => {
                assert_eq!(support.clipped_value_count, 1);
                assert!(support.max_overflow_abs > 0.0);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn wrapper_does_not_change_underlying_raw_retrieval_availability() {
        let context = [0.2];
        let mut memory = SemanticPriorMemory::new();
        memory.index_prior(&context, &prior(&context, 0.8)).unwrap();

        let raw = memory.retrieve_unfiltered(&[2.5], 4).unwrap();
        assert_eq!(raw.len(), 1);
    }

    #[test]
    fn strict_memory_records_candidate_source_support() {
        let context = [0.2];
        let mut memory = SupportBoundSemanticPriorMemory::new();
        let identity = memory.index_prior(&context, &prior(&context, 0.8)).unwrap();
        let support = memory.source_support(identity).unwrap();
        assert!(support.within_nominal_support());
        assert_eq!(support.context_dimension, 1);
    }

    #[test]
    fn strict_memory_filters_unsupported_candidate_sources() {
        let supported = [0.2];
        let unsupported = [2.0];
        let mut memory = SupportBoundSemanticPriorMemory::new();
        memory
            .index_prior(&supported, &prior(&supported, 0.7))
            .unwrap();
        memory
            .index_prior(&unsupported, &prior(&unsupported, 0.9))
            .unwrap();

        let candidates = memory
            .retrieve_fully_supported(&supported, 0.0, 10)
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].candidate.identity,
            ContextIdentity::from_context(&supported).unwrap()
        );
        assert!(candidates[0].query_support.within_nominal_support());
        assert!(candidates[0].candidate_source_support.within_nominal_support());
    }

    #[test]
    fn unsupported_candidate_does_not_consume_final_limit() {
        let supported = [1.0];
        let unsupported = [2.0];
        let mut memory = SupportBoundSemanticPriorMemory::new();
        memory
            .index_prior(&unsupported, &prior(&unsupported, 1.0))
            .unwrap();
        memory
            .index_prior(&supported, &prior(&supported, 0.5))
            .unwrap();

        let candidates = memory
            .retrieve_fully_supported(&supported, 0.0, 1)
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].candidate.identity,
            ContextIdentity::from_context(&supported).unwrap()
        );
    }

    #[test]
    fn strict_raw_retrieval_excludes_cross_dimensional_candidates() {
        let one_dimensional = [0.2];
        let two_dimensional = [0.2, 0.3];
        let mut memory = SupportBoundSemanticPriorMemory::new();
        memory
            .index_prior(&one_dimensional, &prior(&one_dimensional, 0.8))
            .unwrap();
        memory
            .index_prior(&two_dimensional, &prior(&two_dimensional, 0.8))
            .unwrap();

        let candidates = memory
            .retrieve_fully_supported(&one_dimensional, 0.0, 10)
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].candidate.context_dimension, 1);
    }

    #[test]
    fn strict_calibrated_retrieval_preserves_both_support_receipts() {
        let context = [0.2];
        let mut memory = SupportBoundSemanticPriorMemory::new();
        memory.index_prior(&context, &prior(&context, 0.8)).unwrap();
        let calibration = strict_calibration(&memory);

        let candidates = memory
            .retrieve_calibrated_fully_supported(&context, 0.0, 4, &calibration)
            .unwrap();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].query_support.within_nominal_support());
        assert!(
            candidates[0]
                .candidate_source_support
                .within_nominal_support()
        );
    }
}