// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tempered evidence fusion for hybrid owl-inspired perception regimes.
//!
//! This module combines per-candidate log-likelihood fields supplied by
//! independent modality-specific estimators (for example acoustic, visual, or
//! head-pose-conditioned estimates). It does not extract features from sensors.
//!
//! Each present modality contributes:
//!
//!     tempering_weight * log_likelihood(candidate | modality)
//!
//! to the candidate's prior log-weight. Weights in [0, 1] temper that
//! modality's influence; they are not learned reliabilities. Selecting them
//! requires validation on development data. A modality may be marked unavailable
//! only with an empty score array, so sensor loss is explicit and never silently
//! replaced by invented evidence.
//!
//! This product-of-experts style fusion assumes conditionally independent
//! modality likelihoods given a source candidate. Real modalities can be
//! correlated (e.g. a vision model conditioned on audio), so independence must
//! be justified or the fusion model revised. Normalized weights are only
//! relative weights over the discrete candidate set, not calibrated posterior
//! probabilities over continuous space.

use core::fmt;

/// Candidate-aligned log-likelihoods supplied by one modality-specific estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModalityEvidence<'a> {
    /// Stable identity such as "audio_tdoa", "visual_bearing", or "head_pose".
    pub modality_id: &'a str,
    /// One log-likelihood per candidate, in exactly the same candidate order.
    /// Negative infinity may mark a candidate as impossible under the model.
    pub log_likelihoods: &'a [f64],
    /// Explicit exponent/tempering weight in [0, 1]. This is not a calibrated
    /// confidence and must be selected / validated separately.
    pub tempering_weight: f64,
    /// Whether this modality contributed observations to this decision.
    pub available: bool,
}

/// One fused candidate score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedCandidateScore {
    /// Zero-based index into the candidate list shared by all input arrays.
    pub candidate_index: usize,
    /// Combined prior-plus-evidence log score. This is not a calibrated log probability.
    pub relative_log_score: f64,
    /// Normalized weight across the candidates supplied to this call.
    /// This depends on the candidate grid and is not a calibrated probability.
    pub normalized_weight: f64,
}

/// Input / numeric validation failures from hybrid evidence fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridFusionError {
    EmptyCandidateSet,
    EmptyModalitySet,
    EmptyModalityId,
    DuplicateModalityId,
    CandidateCountMismatch,
    UnavailableModalityHasScores,
    InvalidTemperingWeight,
    InvalidLogScore,
    NoActiveModality,
    NoFiniteCandidateScores,
    NumericalOverflow,
}

impl fmt::Display for HybridFusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet => write!(f, "candidate set must not be empty"),
            Self::EmptyModalitySet => write!(f, "at least one modality declaration is required"),
            Self::EmptyModalityId => write!(f, "modality identity must not be blank"),
            Self::DuplicateModalityId => write!(f, "modality identities must be unique"),
            Self::CandidateCountMismatch => {
                write!(f, "each available modality must score every candidate in the same order")
            }
            Self::UnavailableModalityHasScores => {
                write!(f, "an unavailable modality must provide no scores")
            }
            Self::InvalidTemperingWeight => {
                write!(f, "tempering weight must be finite and within [0, 1]")
            }
            Self::InvalidLogScore => {
                write!(f, "log scores may be finite or negative infinity, not NaN or positive infinity")
            }
            Self::NoActiveModality => {
                write!(f, "at least one available modality must have a positive weight")
            }
            Self::NoFiniteCandidateScores => {
                write!(f, "no candidate has a finite fused log score")
            }
            Self::NumericalOverflow => write!(f, "log-score fusion overflowed numerically"),
        }
    }
}

impl std::error::Error for HybridFusionError {}

fn valid_log_score(value: f64) -> bool {
    value.is_finite() || value == f64::NEG_INFINITY
}

/// Fuse modality-specific log-likelihood fields over the same discrete candidates.
///
/// The `prior_log_weights` slice defines candidate count and may contain finite
/// values or negative infinity. Each available modality must supply exactly one
/// valid log-likelihood per candidate. Unavailable modalities must supply an
/// empty likelihood slice; no imputation occurs in this kernel.
///
/// The fused unnormalized score is
///
///     prior_log_weight + sum_i(tempering_weight_i * log_likelihood_i)
///
/// A zero-weight modality is validated but contributes nothing. The function
/// uses max-subtracted exponentiation for normalization and returns an error if
/// all candidates are impossible under the combined model.
///
/// # Model limitations
///
/// This is a transparent scoring kernel, not a complete sensor-fusion system.
/// It does not determine candidate identity alignment, calibrate likelihoods,
/// account for correlated modalities, handle delayed observations, detect a
/// faulty sensor, or choose control actions. A candidate grid is discrete, so
/// normalized weights depend on candidate sampling density. Do not label them
/// as calibrated probabilities without held-out calibration and a suitable
/// spatial prior.
pub fn fuse_candidate_log_likelihoods(
    prior_log_weights: &[f64],
    modalities: &[ModalityEvidence<'_>],
) -> Result<Vec<FusedCandidateScore>, HybridFusionError> {
    if prior_log_weights.is_empty() {
        return Err(HybridFusionError::EmptyCandidateSet);
    }
    if modalities.is_empty() {
        return Err(HybridFusionError::EmptyModalitySet);
    }
    if prior_log_weights.iter().any(|score| !valid_log_score(*score)) {
        return Err(HybridFusionError::InvalidLogScore);
    }

    let candidate_count = prior_log_weights.len();
    let mut active_modality_count = 0usize;

    for (index, modality) in modalities.iter().enumerate() {
        if modality.modality_id.trim().is_empty() {
            return Err(HybridFusionError::EmptyModalityId);
        }
        if modalities[..index]
            .iter()
            .any(|previous| previous.modality_id == modality.modality_id)
        {
            return Err(HybridFusionError::DuplicateModalityId);
        }
        if !modality.tempering_weight.is_finite()
            || !(0.0..=1.0).contains(&modality.tempering_weight)
        {
            return Err(HybridFusionError::InvalidTemperingWeight);
        }

        if modality.available {
            if modality.log_likelihoods.len() != candidate_count {
                return Err(HybridFusionError::CandidateCountMismatch);
            }
            if modality
                .log_likelihoods
                .iter()
                .any(|score| !valid_log_score(*score))
            {
                return Err(HybridFusionError::InvalidLogScore);
            }
            if modality.tempering_weight > 0.0 {
                active_modality_count += 1;
            }
        } else if !modality.log_likelihoods.is_empty() {
            return Err(HybridFusionError::UnavailableModalityHasScores);
        }
    }

    if active_modality_count == 0 {
        return Err(HybridFusionError::NoActiveModality);
    }

    let mut fused_scores = Vec::with_capacity(candidate_count);
    for (candidate_index, prior) in prior_log_weights.iter().copied().enumerate() {
        let mut score = prior;
        if score.is_finite() {
            for modality in modalities {
                if !modality.available || modality.tempering_weight == 0.0 {
                    continue;
                }
                let evidence_score = modality.log_likelihoods[candidate_index];
                if evidence_score == f64::NEG_INFINITY {
                    score = f64::NEG_INFINITY;
                    break;
                }
                if score == f64::NEG_INFINITY {
                    break;
                }

                let contribution = modality.tempering_weight * evidence_score;
                if !contribution.is_finite() {
                    return Err(HybridFusionError::NumericalOverflow);
                }
                let next_score = score + contribution;
                if !next_score.is_finite() {
                    return Err(HybridFusionError::NumericalOverflow);
                }
                score = next_score;
            }
        }
        fused_scores.push(score);
    }

    let max_score = fused_scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_score.is_finite() {
        return Err(HybridFusionError::NoFiniteCandidateScores);
    }

    let weights: Vec<f64> = fused_scores
        .iter()
        .map(|score| {
            if *score == f64::NEG_INFINITY {
                0.0
            } else {
                (*score - max_score).exp()
            }
        })
        .collect();
    let weight_sum: f64 = weights.iter().sum();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(HybridFusionError::NoFiniteCandidateScores);
    }

    Ok(fused_scores
        .into_iter()
        .zip(weights)
        .enumerate()
        .map(
            |(candidate_index, (relative_log_score, weight))| FusedCandidateScore {
                candidate_index,
                relative_log_score,
                normalized_weight: weight / weight_sum,
            },
        )
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence<'a>(
        modality_id: &'a str,
        scores: &'a [f64],
        tempering_weight: f64,
    ) -> ModalityEvidence<'a> {
        ModalityEvidence {
            modality_id,
            log_likelihoods: scores,
            tempering_weight,
            available: true,
        }
    }

    #[test]
    fn agreeing_modalities_sharpen_the_same_candidate() {
        let prior = [0.0, 0.0];
        let audio = [0.0, -2.0];
        let vision = [0.0, -1.0];
        let fused = fuse_candidate_log_likelihoods(
            &prior,
            &[evidence("audio", &audio, 1.0), evidence("vision", &vision, 1.0)],
        )
        .unwrap();

        assert!(fused[0].normalized_weight > fused[1].normalized_weight);
        assert!((fused.iter().map(|score| score.normalized_weight).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn conflicting_modalities_do_not_get_silently_resolved_by_order() {
        let prior = [0.0, 0.0];
        let audio = [0.0, -2.0];
        let vision = [-2.0, 0.0];
        let forward = fuse_candidate_log_likelihoods(
            &prior,
            &[evidence("audio", &audio, 1.0), evidence("vision", &vision, 1.0)],
        )
        .unwrap();
        let reverse = fuse_candidate_log_likelihoods(
            &prior,
            &[evidence("vision", &vision, 1.0), evidence("audio", &audio, 1.0)],
        )
        .unwrap();

        assert!((forward[0].normalized_weight - forward[1].normalized_weight).abs() < 1e-12);
        assert_eq!(forward, reverse);
    }

    #[test]
    fn absent_modality_is_explicit_and_matches_single_modality_result() {
        let prior = [0.0, 0.0];
        let audio = [0.0, -1.0];
        let single = fuse_candidate_log_likelihoods(&prior, &[evidence("audio", &audio, 1.0)])
            .unwrap();
        let audio_only = ModalityEvidence {
            modality_id: "audio",
            log_likelihoods: &audio,
            tempering_weight: 1.0,
            available: true,
        };
        let absent_vision = ModalityEvidence {
            modality_id: "vision",
            log_likelihoods: &[],
            tempering_weight: 1.0,
            available: false,
        };
        let hybrid =
            fuse_candidate_log_likelihoods(&prior, &[audio_only, absent_vision]).unwrap();

        assert_eq!(single, hybrid);
    }

    #[test]
    fn zero_weight_disables_modality_without_changing_result() {
        let prior = [0.0, 0.0];
        let audio = [0.0, -1.0];
        let vision = [-100.0, 0.0];
        let baseline = fuse_candidate_log_likelihoods(&prior, &[evidence("audio", &audio, 1.0)])
            .unwrap();
        let fused = fuse_candidate_log_likelihoods(
            &prior,
            &[evidence("audio", &audio, 1.0), evidence("vision", &vision, 0.0)],
        )
        .unwrap();

        assert_eq!(baseline, fused);
    }

    #[test]
    fn candidate_count_mismatch_is_rejected() {
        let prior = [0.0, 0.0];
        let short = [0.0];
        assert_eq!(
            fuse_candidate_log_likelihoods(&prior, &[evidence("audio", &short, 1.0)]),
            Err(HybridFusionError::CandidateCountMismatch)
        );
    }

    #[test]
    fn duplicate_modality_names_are_rejected_to_prevent_double_counting() {
        let prior = [0.0, 0.0];
        let scores = [0.0, -1.0];
        assert_eq!(
            fuse_candidate_log_likelihoods(
                &prior,
                &[evidence("audio", &scores, 1.0), evidence("audio", &scores, 1.0)],
            ),
            Err(HybridFusionError::DuplicateModalityId)
        );
    }

    #[test]
    fn unavailable_modality_cannot_smuggle_stale_scores() {
        let prior = [0.0, 0.0];
        let scores = [0.0, -1.0];
        let missing = ModalityEvidence {
            modality_id: "vision",
            log_likelihoods: &scores,
            tempering_weight: 1.0,
            available: false,
        };
        assert_eq!(
            fuse_candidate_log_likelihoods(&prior, &[missing]),
            Err(HybridFusionError::UnavailableModalityHasScores)
        );
    }

    #[test]
    fn invalid_weights_are_rejected() {
        let prior = [0.0, 0.0];
        let scores = [0.0, -1.0];
        assert_eq!(
            fuse_candidate_log_likelihoods(&prior, &[evidence("audio", &scores, 1.01)]),
            Err(HybridFusionError::InvalidTemperingWeight)
        );
    }

    #[test]
    fn non_finite_likelihoods_are_rejected() {
        let prior = [0.0, 0.0];
        let scores = [0.0, f64::NAN];
        assert_eq!(
            fuse_candidate_log_likelihoods(&prior, &[evidence("audio", &scores, 1.0)]),
            Err(HybridFusionError::InvalidLogScore)
        );
    }

    #[test]
    fn all_impossible_candidates_are_rejected() {
        let prior = [0.0, 0.0];
        let audio = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        assert_eq!(
            fuse_candidate_log_likelihoods(&prior, &[evidence("audio", &audio, 1.0)]),
            Err(HybridFusionError::NoFiniteCandidateScores)
        );
    }

    #[test]
    fn missing_or_zero_weight_only_is_rejected() {
        let prior = [0.0, 0.0];
        let missing = ModalityEvidence {
            modality_id: "audio",
            log_likelihoods: &[],
            tempering_weight: 1.0,
            available: false,
        };
        assert_eq!(
            fuse_candidate_log_likelihoods(&prior, &[missing]),
            Err(HybridFusionError::NoActiveModality)
        );
    }

    #[test]
    fn nonzero_prior_is_preserved() {
        let prior = [0.0, -3.0];
        let neutral = [0.0, 0.0];
        let fused =
            fuse_candidate_log_likelihoods(&prior, &[evidence("audio", &neutral, 1.0)]).unwrap();
        assert!(fused[0].normalized_weight > fused[1].normalized_weight);
    }
}
