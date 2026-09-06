// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Counterfactual candidate portfolios and Pareto filtering.
//!
//! This module ranks *unqualified* proposals for further reasoning. It never
//! executes a simulator, selects an actuator command, or grants authority.

use serde::{Deserialize, Serialize};
use symthaea_physical_effects::{
    AbstentionReason, EffectValidationError, ProposedIntervention,
};
use thiserror::Error;

/// One independent model's prediction for the same candidate intervention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub model_id: String,
    pub success_probability: f64,
}

impl ModelPrediction {
    pub fn validate(&self) -> Result<(), PortfolioError> {
        if self.model_id.trim().is_empty() {
            return Err(PortfolioError::EmptyField("model_prediction.model_id"));
        }
        validate_unit_interval(
            "model_prediction.success_probability",
            self.success_probability,
        )
    }
}

/// Preserves model disagreement instead of averaging it away.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelEnsembleSummary {
    pub min_success_probability: f64,
    pub max_success_probability: f64,
    pub mean_success_probability: f64,
    /// Range of the independent success predictions, in [0, 1].
    pub disagreement: f64,
}

impl ModelEnsembleSummary {
    pub fn from_predictions(predictions: &[ModelPrediction]) -> Result<Self, PortfolioError> {
        if predictions.is_empty() {
            return Err(PortfolioError::EmptyModelEnsemble);
        }
        for prediction in predictions {
            prediction.validate()?;
        }

        let mut min = 1.0_f64;
        let mut max = 0.0_f64;
        let mut sum = 0.0_f64;
        for prediction in predictions {
            min = min.min(prediction.success_probability);
            max = max.max(prediction.success_probability);
            sum += prediction.success_probability;
        }

        Ok(Self {
            min_success_probability: min,
            max_success_probability: max,
            mean_success_probability: sum / predictions.len() as f64,
            disagreement: max - min,
        })
    }
}

/// Multi-objective assessment of an unqualified proposal.
///
/// Benefits are success probability, information gain, reversibility, and
/// safety margin. Costs are uncertainty, energy, duration, and model
/// disagreement. All fields are explicit so Pareto filtering does not hide a
/// weighting scheme inside one scalar reward.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateAssessment {
    pub proposal: ProposedIntervention,
    pub expected_energy_j: f64,
    pub expected_duration_ms: u64,
    pub information_gain: f64,
    pub reversibility_score: f64,
    pub safety_margin: f64,
    pub model_disagreement: f64,
}

impl CandidateAssessment {
    pub fn validate(&self) -> Result<(), PortfolioError> {
        self.proposal.validate().map_err(PortfolioError::Effect)?;
        if !self.expected_energy_j.is_finite() || self.expected_energy_j < 0.0 {
            return Err(PortfolioError::InvalidMetric {
                field: "candidate.expected_energy_j",
                value: self.expected_energy_j,
            });
        }
        if self.expected_duration_ms == 0 {
            return Err(PortfolioError::ZeroDuration);
        }
        validate_unit_interval("candidate.information_gain", self.information_gain)?;
        validate_unit_interval("candidate.reversibility_score", self.reversibility_score)?;
        validate_unit_interval("candidate.safety_margin", self.safety_margin)?;
        validate_unit_interval("candidate.model_disagreement", self.model_disagreement)?;
        Ok(())
    }

    /// Strict Pareto dominance across all currently represented objectives.
    ///
    /// No hidden weights are used. `self` dominates `other` only when it is no
    /// worse on every objective and strictly better on at least one.
    pub fn dominates(&self, other: &Self) -> bool {
        let a = &self.proposal.predicted_outcome;
        let b = &other.proposal.predicted_outcome;

        let no_worse = a.success_probability >= b.success_probability
            && a.epistemic_uncertainty <= b.epistemic_uncertainty
            && a.aleatoric_uncertainty <= b.aleatoric_uncertainty
            && self.expected_energy_j <= other.expected_energy_j
            && self.expected_duration_ms <= other.expected_duration_ms
            && self.information_gain >= other.information_gain
            && self.reversibility_score >= other.reversibility_score
            && self.safety_margin >= other.safety_margin
            && self.model_disagreement <= other.model_disagreement;

        let strictly_better = a.success_probability > b.success_probability
            || a.epistemic_uncertainty < b.epistemic_uncertainty
            || a.aleatoric_uncertainty < b.aleatoric_uncertainty
            || self.expected_energy_j < other.expected_energy_j
            || self.expected_duration_ms < other.expected_duration_ms
            || self.information_gain > other.information_gain
            || self.reversibility_score > other.reversibility_score
            || self.safety_margin > other.safety_margin
            || self.model_disagreement < other.model_disagreement;

        no_worse && strictly_better
    }
}

/// Conservative admission policy applied before Pareto filtering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PortfolioPolicy {
    pub min_success_probability: f64,
    pub max_epistemic_uncertainty: f64,
    pub max_aleatoric_uncertainty: f64,
    pub max_model_disagreement: f64,
    pub min_safety_margin: f64,
}

impl Default for PortfolioPolicy {
    fn default() -> Self {
        Self {
            min_success_probability: 0.0,
            max_epistemic_uncertainty: 1.0,
            max_aleatoric_uncertainty: 1.0,
            max_model_disagreement: 1.0,
            min_safety_margin: 0.0,
        }
    }
}

impl PortfolioPolicy {
    pub fn validate(&self) -> Result<(), PortfolioError> {
        validate_unit_interval(
            "policy.min_success_probability",
            self.min_success_probability,
        )?;
        validate_unit_interval(
            "policy.max_epistemic_uncertainty",
            self.max_epistemic_uncertainty,
        )?;
        validate_unit_interval(
            "policy.max_aleatoric_uncertainty",
            self.max_aleatoric_uncertainty,
        )?;
        validate_unit_interval(
            "policy.max_model_disagreement",
            self.max_model_disagreement,
        )?;
        validate_unit_interval("policy.min_safety_margin", self.min_safety_margin)?;
        Ok(())
    }

    fn admits(&self, candidate: &CandidateAssessment) -> bool {
        let outcome = &candidate.proposal.predicted_outcome;
        outcome.success_probability >= self.min_success_probability
            && outcome.epistemic_uncertainty <= self.max_epistemic_uncertainty
            && outcome.aleatoric_uncertainty <= self.max_aleatoric_uncertainty
            && candidate.model_disagreement <= self.max_model_disagreement
            && candidate.safety_margin >= self.min_safety_margin
    }
}

/// A collection of mechanism candidates for one desired transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidatePortfolio {
    pub transition_id: String,
    pub candidates: Vec<CandidateAssessment>,
}

impl CandidatePortfolio {
    pub fn validate(&self) -> Result<(), PortfolioError> {
        if self.transition_id.trim().is_empty() {
            return Err(PortfolioError::EmptyField("portfolio.transition_id"));
        }
        for candidate in &self.candidates {
            candidate.validate()?;
            if candidate.proposal.transition_id != self.transition_id {
                return Err(PortfolioError::TransitionMismatch {
                    portfolio: self.transition_id.clone(),
                    proposal: candidate.proposal.transition_id.clone(),
                });
            }
        }
        Ok(())
    }

    /// Filter candidates by explicit conservative thresholds and return the
    /// non-dominated frontier. Empty/insufficient portfolios abstain rather
    /// than forcing a single action.
    pub fn evaluate(&self, policy: PortfolioPolicy) -> Result<PortfolioOutcome, PortfolioError> {
        self.validate()?;
        policy.validate()?;

        let eligible = self
            .candidates
            .iter()
            .filter(|candidate| policy.admits(candidate))
            .cloned()
            .collect::<Vec<_>>();

        if eligible.is_empty() {
            return Ok(PortfolioOutcome::Abstain(
                AbstentionReason::NoQualifiedAction,
            ));
        }

        let frontier = eligible
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                !eligible
                    .iter()
                    .enumerate()
                    .any(|(other_index, other)| other_index != *index && other.dominates(candidate))
            })
            .map(|(_, candidate)| candidate.clone())
            .collect::<Vec<_>>();

        if frontier.is_empty() {
            // This should be unreachable for a finite non-empty set under
            // strict Pareto dominance, but retain fail-closed behaviour.
            return Ok(PortfolioOutcome::Abstain(
                AbstentionReason::NoQualifiedAction,
            ));
        }

        Ok(PortfolioOutcome::ParetoFrontier(frontier))
    }
}

/// Deliberative output. A frontier is intentionally not an execution decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PortfolioOutcome {
    ParetoFrontier(Vec<CandidateAssessment>),
    Abstain(AbstentionReason),
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum PortfolioError {
    #[error("required field is empty: {0}")]
    EmptyField(&'static str),
    #[error("model ensemble cannot be empty")]
    EmptyModelEnsemble,
    #[error("invalid metric for {field}: {value}")]
    InvalidMetric { field: &'static str, value: f64 },
    #[error("candidate expected duration must be greater than zero")]
    ZeroDuration,
    #[error("proposal transition {proposal:?} does not match portfolio {portfolio:?}")]
    TransitionMismatch { portfolio: String, proposal: String },
    #[error("invalid physical-effect proposal: {0}")]
    Effect(EffectValidationError),
}

fn validate_unit_interval(field: &'static str, value: f64) -> Result<(), PortfolioError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(PortfolioError::InvalidMetric { field, value });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_physical_effects::{
        AuthorityClass, MechanismRef, PhysicalModality, PredictedOutcome,
    };

    fn candidate(
        id: &str,
        modality: PhysicalModality,
        success: f64,
        energy: f64,
        info: f64,
        safety: f64,
        disagreement: f64,
    ) -> CandidateAssessment {
        CandidateAssessment {
            proposal: ProposedIntervention {
                id: id.into(),
                transition_id: "t-1".into(),
                mechanism: MechanismRef {
                    backend: "fixture".into(),
                    mechanism: format!("{id}-mechanism"),
                    modality,
                },
                required_authority: AuthorityClass::SimulationOnly,
                predicted_outcome: PredictedOutcome {
                    success_probability: success,
                    epistemic_uncertainty: 0.1,
                    aleatoric_uncertainty: 0.05,
                },
            },
            expected_energy_j: energy,
            expected_duration_ms: 100,
            information_gain: info,
            reversibility_score: 1.0,
            safety_margin: safety,
            model_disagreement: disagreement,
        }
    }

    #[test]
    fn model_disagreement_is_preserved_as_range() {
        let summary = ModelEnsembleSummary::from_predictions(&[
            ModelPrediction {
                model_id: "analytical".into(),
                success_probability: 0.9,
            },
            ModelPrediction {
                model_id: "numerical".into(),
                success_probability: 0.6,
            },
            ModelPrediction {
                model_id: "empirical".into(),
                success_probability: 0.75,
            },
        ])
        .unwrap();

        assert!((summary.disagreement - 0.3).abs() < 1e-12);
        assert!((summary.min_success_probability - 0.6).abs() < 1e-12);
        assert!((summary.max_success_probability - 0.9).abs() < 1e-12);
    }

    #[test]
    fn dominated_candidate_is_removed_without_scalar_reward() {
        let strong = candidate(
            "strong",
            PhysicalModality::Acoustic,
            0.9,
            5.0,
            0.8,
            0.9,
            0.05,
        );
        let weak = candidate(
            "weak",
            PhysicalModality::Photonic,
            0.8,
            10.0,
            0.7,
            0.8,
            0.1,
        );
        assert!(strong.dominates(&weak));

        let portfolio = CandidatePortfolio {
            transition_id: "t-1".into(),
            candidates: vec![strong.clone(), weak],
        };
        let outcome = portfolio.evaluate(PortfolioPolicy::default()).unwrap();
        assert_eq!(outcome, PortfolioOutcome::ParetoFrontier(vec![strong]));
    }

    #[test]
    fn genuine_tradeoff_preserves_multiple_frontier_candidates() {
        let efficient = candidate(
            "efficient",
            PhysicalModality::Acoustic,
            0.8,
            1.0,
            0.6,
            0.95,
            0.05,
        );
        let informative = candidate(
            "informative",
            PhysicalModality::Photonic,
            0.95,
            8.0,
            0.95,
            0.8,
            0.05,
        );

        assert!(!efficient.dominates(&informative));
        assert!(!informative.dominates(&efficient));

        let portfolio = CandidatePortfolio {
            transition_id: "t-1".into(),
            candidates: vec![efficient, informative],
        };
        match portfolio.evaluate(PortfolioPolicy::default()).unwrap() {
            PortfolioOutcome::ParetoFrontier(frontier) => assert_eq!(frontier.len(), 2),
            other => panic!("expected frontier, got {other:?}"),
        }
    }

    #[test]
    fn high_disagreement_can_force_abstention() {
        let uncertain = candidate(
            "uncertain",
            PhysicalModality::Coupled,
            0.95,
            2.0,
            0.8,
            0.9,
            0.7,
        );
        let portfolio = CandidatePortfolio {
            transition_id: "t-1".into(),
            candidates: vec![uncertain],
        };
        let policy = PortfolioPolicy {
            max_model_disagreement: 0.2,
            ..PortfolioPolicy::default()
        };

        assert_eq!(
            portfolio.evaluate(policy).unwrap(),
            PortfolioOutcome::Abstain(AbstentionReason::NoQualifiedAction)
        );
    }

    #[test]
    fn empty_portfolio_abstains_instead_of_inventing_action() {
        let portfolio = CandidatePortfolio {
            transition_id: "t-1".into(),
            candidates: vec![],
        };
        assert_eq!(
            portfolio.evaluate(PortfolioPolicy::default()).unwrap(),
            PortfolioOutcome::Abstain(AbstentionReason::NoQualifiedAction)
        );
    }

    #[test]
    fn mismatched_transition_is_rejected() {
        let mut wrong = candidate(
            "wrong",
            PhysicalModality::Thermal,
            0.9,
            2.0,
            0.5,
            0.9,
            0.1,
        );
        wrong.proposal.transition_id = "another-transition".into();
        let portfolio = CandidatePortfolio {
            transition_id: "t-1".into(),
            candidates: vec![wrong],
        };
        assert!(matches!(
            portfolio.evaluate(PortfolioPolicy::default()),
            Err(PortfolioError::TransitionMismatch { .. })
        ));
    }

    #[test]
    fn non_finite_metrics_fail_closed() {
        let mut bad = candidate(
            "bad",
            PhysicalModality::Mechanical,
            0.9,
            1.0,
            0.5,
            0.9,
            0.1,
        );
        bad.model_disagreement = f64::NAN;
        assert!(bad.validate().is_err());
    }
}
