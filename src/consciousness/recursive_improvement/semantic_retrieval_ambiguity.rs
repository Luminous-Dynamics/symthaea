// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ambiguity diagnostics for operational HDC semantic retrieval.
//!
//! Absolute similarity and retrieval ambiguity are different geometric facts. This
//! module reports the top-1/top-2 similarity gap without treating that gap as
//! empirical evidence, calibrated probability, or confidence authority.

use super::semantic_support::{
    IntegrityBoundCalibratedSemanticPriorCandidate, IntegrityBoundSemanticPriorCandidate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticRetrievalAmbiguityError {
    NonFiniteSimilarity { index: usize },
    SimilarityOutOfRange { index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SemanticRetrievalAmbiguity {
    pub candidate_count: usize,
    pub best_similarity: Option<f32>,
    pub runner_up_similarity: Option<f32>,
    /// `best_similarity - runner_up_similarity` when at least two candidates exist.
    ///
    /// A larger margin means the nearest candidate is more geometrically separated
    /// from its closest rival. This is an ambiguity diagnostic only.
    pub top_two_margin: Option<f32>,
}

impl SemanticRetrievalAmbiguity {
    pub fn from_similarities(
        similarities: impl IntoIterator<Item = f32>,
    ) -> Result<Self, SemanticRetrievalAmbiguityError> {
        let mut count = 0usize;
        let mut best: Option<f32> = None;
        let mut second: Option<f32> = None;

        for (index, similarity) in similarities.into_iter().enumerate() {
            if !similarity.is_finite() {
                return Err(SemanticRetrievalAmbiguityError::NonFiniteSimilarity {
                    index,
                });
            }
            if !(0.0..=1.0).contains(&similarity) {
                return Err(SemanticRetrievalAmbiguityError::SimilarityOutOfRange {
                    index,
                });
            }
            count += 1;

            match best {
                None => best = Some(similarity),
                Some(best_value) if similarity > best_value => {
                    second = best;
                    best = Some(similarity);
                }
                Some(_) => match second {
                    None => second = Some(similarity),
                    Some(second_value) if similarity > second_value => {
                        second = Some(similarity);
                    }
                    Some(_) => {}
                },
            }
        }

        Ok(Self {
            candidate_count: count,
            best_similarity: best,
            runner_up_similarity: second,
            top_two_margin: best.zip(second).map(|(top, runner_up)| top - runner_up),
        })
    }
}

pub fn diagnose_integrity_bound_retrieval(
    candidates: &[IntegrityBoundSemanticPriorCandidate],
) -> Result<SemanticRetrievalAmbiguity, SemanticRetrievalAmbiguityError> {
    SemanticRetrievalAmbiguity::from_similarities(
        candidates
            .iter()
            .map(|candidate| candidate.candidate.candidate.similarity),
    )
}

pub fn diagnose_integrity_bound_calibrated_retrieval(
    candidates: &[IntegrityBoundCalibratedSemanticPriorCandidate],
) -> Result<SemanticRetrievalAmbiguity, SemanticRetrievalAmbiguityError> {
    SemanticRetrievalAmbiguity::from_similarities(candidates.iter().map(|candidate| {
        candidate.candidate.candidate.candidate.similarity
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_is_order_independent() {
        let left = SemanticRetrievalAmbiguity::from_similarities([0.61, 0.83, 0.72]).unwrap();
        let right = SemanticRetrievalAmbiguity::from_similarities([0.83, 0.72, 0.61]).unwrap();

        assert_eq!(left, right);
        assert_eq!(left.candidate_count, 3);
        assert!((left.best_similarity.unwrap() - 0.83).abs() < f32::EPSILON);
        assert!((left.runner_up_similarity.unwrap() - 0.72).abs() < f32::EPSILON);
        assert!((left.top_two_margin.unwrap() - 0.11).abs() < 1e-6);
    }

    #[test]
    fn exact_tie_has_zero_margin() {
        let ambiguity =
            SemanticRetrievalAmbiguity::from_similarities([0.8, 0.8, 0.4]).unwrap();
        assert_eq!(ambiguity.top_two_margin, Some(0.0));
    }

    #[test]
    fn one_candidate_does_not_imply_infinite_certainty() {
        let ambiguity = SemanticRetrievalAmbiguity::from_similarities([0.8]).unwrap();
        assert_eq!(ambiguity.candidate_count, 1);
        assert_eq!(ambiguity.best_similarity, Some(0.8));
        assert_eq!(ambiguity.runner_up_similarity, None);
        assert_eq!(ambiguity.top_two_margin, None);
    }

    #[test]
    fn empty_retrieval_has_no_margin() {
        let ambiguity =
            SemanticRetrievalAmbiguity::from_similarities(std::iter::empty()).unwrap();
        assert_eq!(ambiguity.candidate_count, 0);
        assert_eq!(ambiguity.best_similarity, None);
        assert_eq!(ambiguity.runner_up_similarity, None);
        assert_eq!(ambiguity.top_two_margin, None);
    }

    #[test]
    fn malformed_similarity_fails_closed() {
        assert_eq!(
            SemanticRetrievalAmbiguity::from_similarities([0.8, f32::NAN]),
            Err(SemanticRetrievalAmbiguityError::NonFiniteSimilarity { index: 1 })
        );
        assert_eq!(
            SemanticRetrievalAmbiguity::from_similarities([1.1]),
            Err(SemanticRetrievalAmbiguityError::SimilarityOutOfRange { index: 0 })
        );
    }
}
