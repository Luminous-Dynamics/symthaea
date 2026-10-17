// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Synthetic Earth-interior model-discrimination fixtures for SenseBench.
//!
//! These likelihoods are deliberately hand-authored toy values for checking
//! experiment-ranking plumbing. They are not derived from geophysical data,
//! a calibrated forward model, or a claim about Earth's actual core dynamics.
//! Replace them only with frozen, independently reviewed likelihoods tied to
//! an explicit qualified scenario manifest.

use super::experiment_design::{
    ExperimentCandidate, ExperimentDesignError, ExperimentScore, HypothesisPrior,
    ModelLikelihoodScenario, RobustExperimentScore, rank_experiments, rank_experiments_robust,
};

/// Scenario identity for this small, synthetic reference fixture.
pub const SCENARIO_ID: &str = "earth-interior-model-discrimination-synthetic-v0";

/// Two abstract competing explanations, not asserted Earth states.
pub fn synthetic_hypotheses() -> Vec<HypothesisPrior> {
    vec![
        HypothesisPrior {
            id: "transient_component_present".into(),
            probability: 0.5,
        },
        HypothesisPrior {
            id: "transient_component_absent".into(),
            probability: 0.5,
        },
    ]
}

/// Candidate outcome likelihoods with deliberately simple, explicit costs.
/// 
/// The first candidate is more discriminating by construction; the third is
/// intentionally uninformative. This fixture tests ranking math and deterministic
/// integration only, not which geophysical measurements are valuable in reality.
pub fn synthetic_candidates() -> Vec<ExperimentCandidate> {
    vec![
        ExperimentCandidate {
            id: "targeted_magnetic_time_window".into(),
            cost: 1.0,
            likelihoods: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
        },
        ExperimentCandidate {
            id: "coarse_spatial_comparator".into(),
            cost: 1.5,
            likelihoods: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
        },
        ExperimentCandidate {
            id: "uninformative_control".into(),
            cost: 0.25,
            likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        },
    ]
}

/// Rank the frozen toy candidates through the shared, generic SenseBench API.
pub fn rank_synthetic_candidates(
) -> Result<Vec<ExperimentScore>, ExperimentDesignError> {
    rank_experiments(&synthetic_hypotheses(), &synthetic_candidates())
}


/// Alternative likelihood models for the same synthetic Earth-interior fixture.
///
/// Model A assumes the targeted magnetic time window is highly discriminating;
/// model B assumes that same measurement is non-discriminating. The coarse
/// comparator is moderately informative in both toy models. These invented
/// probabilities illustrate robust ranking under model uncertainty only.
pub fn synthetic_model_scenarios() -> Vec<ModelLikelihoodScenario> {
    vec![
        ModelLikelihoodScenario {
            scenario_id: "toy-forward-model-a".into(),
            weight: 0.5,
            candidates: vec![
                ExperimentCandidate {
                    id: "targeted_magnetic_time_window".into(),
                    cost: 1.0,
                    likelihoods: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
                },
                ExperimentCandidate {
                    id: "coarse_spatial_comparator".into(),
                    cost: 1.5,
                    likelihoods: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
                },
                ExperimentCandidate {
                    id: "uninformative_control".into(),
                    cost: 0.25,
                    likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
                },
            ],
        },
        ModelLikelihoodScenario {
            scenario_id: "toy-forward-model-b".into(),
            weight: 0.5,
            candidates: vec![
                ExperimentCandidate {
                    id: "targeted_magnetic_time_window".into(),
                    cost: 1.0,
                    likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
                },
                ExperimentCandidate {
                    id: "coarse_spatial_comparator".into(),
                    cost: 1.5,
                    likelihoods: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
                },
                ExperimentCandidate {
                    id: "uninformative_control".into(),
                    cost: 0.25,
                    likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
                },
            ],
        },
    ]
}

/// Rank the toy candidates by worst-case gain across two hand-authored models.
pub fn rank_robust_synthetic_candidates(
) -> Result<Vec<RobustExperimentScore>, ExperimentDesignError> {
    rank_experiments_robust(&synthetic_hypotheses(), &synthetic_model_scenarios())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_ranking_prefers_consistent_candidate_over_brittle_targeted_candidate() {
        let scores = rank_robust_synthetic_candidates().unwrap();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].candidate_id, "coarse_spatial_comparator");
        assert!(scores[0].worst_case_information_gain_bits > 0.0);
        assert_eq!(scores[0].model_sensitivity_stddev_bits, 0.0);

        let targeted = scores.iter()
            .find(|score| score.candidate_id == "targeted_magnetic_time_window")
            .unwrap();
        assert_eq!(targeted.worst_case_information_gain_bits, 0.0);
        assert!(targeted.model_sensitivity_stddev_bits > 0.0);
    }

    #[test]
    fn synthetic_discriminating_measurement_ranks_above_controls() {
        let scores = rank_synthetic_candidates().unwrap();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].candidate_id, "targeted_magnetic_time_window");
        assert_eq!(scores[1].candidate_id, "coarse_spatial_comparator");
        assert_eq!(scores[2].candidate_id, "uninformative_control");

        // Binary symmetric channels with equal priors:
        // I(H; O) = 1 - [-p log2(p) - (1-p) log2(1-p)].
        let binary_entropy = |p: f64| -p * p.log2() - (1.0 - p) * (1.0 - p).log2();
        let targeted_eig = 1.0 - binary_entropy(0.9);
        let coarse_eig = 1.0 - binary_entropy(0.75);
        assert!((scores[0].expected_information_gain_bits - targeted_eig).abs() < 1.0e-12);
        assert!((scores[0].information_gain_per_cost - targeted_eig).abs() < 1.0e-12);
        assert!((scores[1].expected_information_gain_bits - coarse_eig).abs() < 1.0e-12);
        assert!((scores[1].information_gain_per_cost - coarse_eig / 1.5).abs() < 1.0e-12);
        assert!(scores[2].expected_information_gain_bits.abs() < 1.0e-12);
    }

    #[test]
    fn repeated_ranking_is_deterministic_and_prior_is_preserved() {
        assert_eq!(synthetic_hypotheses(), synthetic_hypotheses());
        assert_eq!(synthetic_candidates(), synthetic_candidates());
        assert_eq!(rank_synthetic_candidates().unwrap(), rank_synthetic_candidates().unwrap());
    }

    #[test]
    fn likelihood_rows_represent_distinct_prediction_distributions() {
        let candidates = synthetic_candidates();
        assert_ne!(candidates[0].likelihoods[0], candidates[0].likelihoods[1]);
        assert_eq!(candidates[2].likelihoods[0], candidates[2].likelihoods[1]);
    }
}
