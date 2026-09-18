// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Query-support-aware wrapper around calibrated semantic dream-prior retrieval.
//!
//! Existing raw and calibrated retrieval APIs remain available for diagnostics and
//! experiments. This wrapper adds a stricter operational path: it refuses to treat a
//! clamp-saturated **query** as nominally supported semantic retrieval.
//!
//! The candidate's original numeric context is not retained by `SemanticPriorMemory`,
//! so this module deliberately makes no claim that candidate-source support is known.
//! Rejection here is not a truth judgment; it means only that the query lies outside
//! the numeric support represented faithfully by the selected encoder configuration.

use super::semantic_null_calibration::SemanticNullCalibration;
use super::semantic_prior::{
    CalibratedSemanticPriorCandidate, SemanticPriorError, SemanticPriorMemory,
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

#[cfg(test)]
mod tests {
    use super::super::dream_feedback::{ActionPrior, hash_context};
    use super::super::semantic_null_calibration::SemanticNullConfig;
    use super::*;

    fn prior(context: &[f32]) -> ActionPrior {
        ActionPrior {
            context_hash: hash_context(context),
            preferred_direction: vec![0.2, -0.3, 0.4],
            strength: 0.8,
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

    #[test]
    fn in_support_query_can_use_calibrated_retrieval() {
        let context = [0.2];
        let mut memory = SemanticPriorMemory::new();
        memory.index_prior(&context, &prior(&context)).unwrap();
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
        memory.index_prior(&context, &prior(&context)).unwrap();
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
        memory.index_prior(&context, &prior(&context)).unwrap();

        // Raw retrieval remains intentionally available for diagnostics / experiments.
        let raw = memory.retrieve_unfiltered(&[2.5], 4).unwrap();
        assert_eq!(raw.len(), 1);
    }
}