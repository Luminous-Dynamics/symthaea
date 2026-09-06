// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Counterfactual candidate portfolios and Pareto filtering.
//!
//! This module ranks *unqualified* proposals for further reasoning. It never
//! executes a simulator, selects an actuator command, or grants authority.
//!
//! PA-06 hardening keeps the evidence used by deliberation structurally bound
//! to each candidate: model disagreement is derived from the supplied model
//! predictions rather than accepted as a caller-supplied scalar, and every
//! portfolio carries the full [`DesiredTransition`] contract rather than only
//! its string id.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_physical_effects::{
    AbstentionReason, DesiredTransition, EffectValidationError, ProposedIntervention,
};
use thiserror::Error;

/// One model's prediction for the same candidate intervention.
///
/// Distinct `model_id` values are required within an ensemble. That proves
/// identity separation only; it does not by itself prove statistical or
/// implementation independence between models.
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

/// Derived ensemble statistics. Callers cannot provide a separate disagreement
/// scalar to a candidate; the planner recomputes this summary from predictions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelEnsembleSummary {
    pub model_count: usize,
    pub min_success_probability: f64,
    pub max_success_probability: f64,
    pub mean_success_probability: f64,
    /// Range of the distinct model success predictions, in [0, 1].
    pub disagreement: f64,
}

impl ModelEnsembleSummary {
    pub fn from_predictions(predictions: &[ModelPrediction]) -> Result<Self, PortfolioError> {
        if predictions.is_empty() {
            return Err(PortfolioError::EmptyModelEnsemble);
        }

        let mut ids = BTreeSet::new();
        for prediction in predictions {
            prediction.validate()?;
            if !ids.insert(prediction.model_id.as_str()) {
                return Err(PortfolioError::DuplicateModelId(
                    prediction.model_id.clone(),
                ));
            }
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
            model_count: predictions.len(),
            min_success_probability: min,
            max_success_probability: max,
            mean_success_probability: sum / predictions.len() as f64,
            disagreement: max - min,
        })
    }
}

/// Multi-objective assessment of an unqualified proposal.
///
/// Benefits are conservative ensemble success, information gain,
/// reversibility, and safety margin. Costs are effective epistemic uncertainty,
/// aleatoric uncertainty, energy, power, duration, and model disagreement.
/// All fields remain explicit so Pareto filtering does not hide a weighting
/// scheme inside one scalar reward.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateAssessment {
    pub proposal: ProposedIntervention,
    /// Predictions used to derive disagreement and conservative success.
    pub model_predictions: Vec<ModelPrediction>,
    pub expected_energy_j: f64,
    /// Predicted power when available. If the transition specifies a power
    /// budget this field becomes mandatory and fails closed when absent.
    pub expected_power_w: Option<f64>,
    pub expected_duration_ms: u64,
    pub information_gain: f64,
    pub reversibility_score: f64,
    /// Planning estimate only; never formal safety evidence.
    pub safety_margin: f64,
}

impl CandidateAssessment {
    pub fn model_summary(&self) -> Result<ModelEnsembleSummary, PortfolioError> {
        ModelEnsembleSummary::from_predictions(&self.model_predictions)
    }

    /// Conservative success used for admission/dominance: the least optimistic
    /// prediction in the declared model ensemble.
    pub fn conservative_success_probability(&self) -> Result<f64, PortfolioError> {
        Ok(self.model_summary()?.min_success_probability)
    }

    pub fn model_disagreement(&self) -> Result<f64, PortfolioError> {
        Ok(self.model_summary()?.disagreement)
    }

    /// Model disagreement is treated as a lower bound on epistemic uncertainty
    /// instead of allowing a proposal-local uncertainty estimate to hide a
    /// conflict between models.
    pub fn effective_epistemic_uncertainty(&self) -> Result<f64, PortfolioError> {
        Ok(self
            .proposal
            .predicted_outcome
            .epistemic_uncertainty
            .max(self.model_summary()?.disagreement))
    }

    pub fn validate(&self) -> Result<(), PortfolioError> {
        self.proposal.validate().map_err(PortfolioError::Effect)?;
        self.model_summary()?;

        if !self.expected_energy_j.is_finite() || self.expected_energy_j < 0.0 {
            return Err(PortfolioError::InvalidMetric {
                field: "candidate.expected_energy_j",
                value: self.expected_energy_j,
            });
        }
        if let Some(power) = self.expected_power_w {
            if !power.is_finite() || power < 0.0 {
                return Err(PortfolioError::InvalidMetric {
                    field: "candidate.expected_power_w",
                    value: power,
                });
            }
        }
        if self.expected_duration_ms == 0 {
            return Err(PortfolioError::ZeroDuration);
        }
        validate_unit_interval("candidate.information_gain", self.information_gain)?;
        validate_unit_interval("candidate.reversibility_score", self.reversibility_score)?;
        validate_unit_interval("candidate.safety_margin", self.safety_margin)?;
        Ok(())
    }

    /// Strict Pareto dominance across all currently represented objectives.
    ///
    /// No hidden weights are used. `self` dominates `other` only when it is no
    /// worse on every objective and strictly better on at least one.
    pub fn dominates(&self, other: &Self) -> Result<bool, PortfolioError> {
        self.validate()?;
        other.validate()?;

        let a = &self.proposal.predicted_outcome;
        let b = &other.proposal.predicted_outcome;
        let a_success = self.conservative_success_probability()?;
        let b_success = other.conservative_success_probability()?;
        let a_epistemic = self.effective_epistemic_uncertainty()?;
        let b_epistemic = other.effective_epistemic_uncertainty()?;
        let a_disagreement = self.model_disagreement()?;
        let b_disagreement = other.model_disagreement()?;

        // Missing power cannot dominate a known power value. When both are
        // absent the objective is equal/unknown and contributes no advantage.
        let power_no_worse = match (self.expected_power_w, other.expected_power_w) {
            (Some(a), Some(b)) => a <= b,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => true,
        };
        let power_strictly_better = match (self.expected_power_w, other.expected_power_w) {
            (Some(a), Some(b)) => a < b,
            (Some(_), None) => false,
            (None, Some(_)) | (None, None) => false,
        };

        let no_worse = a_success >= b_success
            && a_epistemic <= b_epistemic
            && a.aleatoric_uncertainty <= b.aleatoric_uncertainty
            && self.expected_energy_j <= other.expected_energy_j
            && power_no_worse
            && self.expected_duration_ms <= other.expected_duration_ms
            && self.information_gain >= other.information_gain
            && self.reversibility_score >= other.reversibility_score
            && self.safety_margin >= other.safety_margin
            && a_disagreement <= b_disagreement;

        let strictly_better = a_success > b_success
            || a_epistemic < b_epistemic
            || a.aleatoric_uncertainty < b.aleatoric_uncertainty
            || self.expected_energy_j < other.expected_energy_j
            || power_strictly_better
            || self.expected_duration_ms < other.expected_duration_ms
            || self.information_gain > other.information_gain
            || self.reversibility_score > other.reversibility_score
            || self.safety_margin > other.safety_margin
            || a_disagreement < b_disagreement;

        Ok(no_worse && strictly_better)
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

    fn admits(&self, candidate: &CandidateAssessment) -> Result<bool, PortfolioError> {
        let outcome = &candidate.proposal.predicted_outcome;
        Ok(candidate.conservative_success_probability()? >= self.min_success_probability
            && candidate.effective_epistemic_uncertainty()? <= self.max_epistemic_uncertainty
            && outcome.aleatoric_uncertainty <= self.max_aleatoric_uncertainty
            && candidate.model_disagreement()? <= self.max_model_disagreement
            && candidate.safety_margin >= self.min_safety_margin)
    }
}

/// A collection of mechanism candidates for one complete desired transition.
///
/// Carrying the full transition prevents a candidate from entering deliberation
/// merely because it copied the right transition id while violating modality,
/// authority, or resource constraints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidatePortfolio {
    pub transition: DesiredTransition,
    pub candidates: Vec<CandidateAssessment>,
}

impl CandidatePortfolio {
    pub fn validate(&self) -> Result<(), PortfolioError> {
        self.transition
            .validate()
            .map_err(PortfolioError::Effect)?;

        for candidate in &self.candidates {
            candidate.validate()?;
            if candidate.proposal.transition_id != self.transition.id {
                return Err(PortfolioError::TransitionMismatch {
                    portfolio: self.transition.id.clone(),
                    proposal: candidate.proposal.transition_id.clone(),
                });
            }
            if !self
                .transition
                .allowed_modalities
                .contains(&candidate.proposal.mechanism.modality)
            {
                return Err(PortfolioError::ModalityNotAllowed {
                    proposal: candidate.proposal.id.clone(),
                });
            }
            if !self
                .transition
                .required_authority
                .allows(candidate.proposal.required_authority)
            {
                return Err(PortfolioError::AuthorityExceedsTransition {
                    proposal: candidate.proposal.id.clone(),
                });
            }

            if let Some(max_energy) = self.transition.resources.max_energy_j {
                if candidate.expected_energy_j > max_energy {
                    return Err(PortfolioError::EnergyBudgetExceeded {
                        proposal: candidate.proposal.id.clone(),
                        expected: candidate.expected_energy_j,
                        maximum: max_energy,
                    });
                }
            }

            if let Some(max_power) = self.transition.resources.max_power_w {
                let expected = candidate.expected_power_w.ok_or_else(|| {
                    PortfolioError::MissingPowerEstimate {
                        proposal: candidate.proposal.id.clone(),
                    }
                })?;
                if expected > max_power {
                    return Err(PortfolioError::PowerBudgetExceeded {
                        proposal: candidate.proposal.id.clone(),
                        expected,
                        maximum: max_power,
                    });
                }
            }

            if let Some(max_duration) = self.transition.resources.max_duration_ms {
                if candidate.expected_duration_ms > max_duration {
                    return Err(PortfolioError::DurationBudgetExceeded {
                        proposal: candidate.proposal.id.clone(),
                        expected: candidate.expected_duration_ms,
                        maximum: max_duration,
                    });
                }
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

        let mut eligible = Vec::new();
        for candidate in &self.candidates {
            if policy.admits(candidate)? {
                eligible.push(candidate.clone());
            }
        }

        if eligible.is_empty() {
            return Ok(PortfolioOutcome::Abstain(
                AbstentionReason::NoQualifiedAction,
            ));
        }

        let mut frontier = Vec::new();
        for (index, candidate) in eligible.iter().enumerate() {
            let mut dominated = false;
            for (other_index, other) in eligible.iter().enumerate() {
                if other_index != index && other.dominates(candidate)? {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                frontier.push(candidate.clone());
            }
        }

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
    #[error("duplicate model id in ensemble: {0:?}")]
    DuplicateModelId(String),
    #[error("invalid metric for {field}: {value}")]
    InvalidMetric { field: &'static str, value: f64 },
    #[error("candidate expected duration must be greater than zero")]
    ZeroDuration,
    #[error("proposal transition {proposal:?} does not match portfolio {portfolio:?}")]
    TransitionMismatch { portfolio: String, proposal: String },
    #[error("proposal {proposal:?} uses a modality not allowed by the transition")]
    ModalityNotAllowed { proposal: String },
    #[error("proposal {proposal:?} requires authority above the transition envelope")]
    AuthorityExceedsTransition { proposal: String },
    #[error("proposal {proposal:?} energy estimate {expected} J exceeds budget {maximum} J")]
    EnergyBudgetExceeded {
        proposal: String,
        expected: f64,
        maximum: f64,
    },
    #[error("proposal {proposal:?} lacks a power estimate required by the transition budget")]
    MissingPowerEstimate { proposal: String },
    #[error("proposal {proposal:?} power estimate {expected} W exceeds budget {maximum} W")]
    PowerBudgetExceeded {
        proposal: String,
        expected: f64,
        maximum: f64,
    },
    #[error("proposal {proposal:?} duration {expected} ms exceeds budget {maximum} ms")]
    DurationBudgetExceeded {
        proposal: String,
        expected: u64,
        maximum: u64,
    },
    #[error("invalid physical-effect value: {0}")]
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
        AuthorityClass, EffectKind, MechanismRef, PhysicalModality, PredictedOutcome, TargetRegion,
    };

    fn transition() -> DesiredTransition {
        DesiredTransition::simulation_only(
            "t-1",
            "compare diagnostic mechanisms",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![
                PhysicalModality::Acoustic,
                PhysicalModality::Photonic,
                PhysicalModality::Coupled,
                PhysicalModality::Thermal,
                PhysicalModality::Mechanical,
            ],
        )
    }

    fn predictions(id: &str, success: f64, disagreement: f64) -> Vec<ModelPrediction> {
        vec![
            ModelPrediction {
                model_id: format!("{id}-model-a"),
                success_probability: success,
            },
            ModelPrediction {
                model_id: format!("{id}-model-b"),
                success_probability: (success - disagreement).max(0.0),
            },
        ]
    }

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
            model_predictions: predictions(id, success, disagreement),
            expected_energy_j: energy,
            expected_power_w: None,
            expected_duration_ms: 100,
            information_gain: info,
            reversibility_score: 1.0,
            safety_margin: safety,
        }
    }

    #[test]
    fn model_disagreement_is_derived_as_range() {
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

        assert_eq!(summary.model_count, 3);
        assert!((summary.disagreement - 0.3).abs() < 1e-12);
        assert!((summary.min_success_probability - 0.6).abs() < 1e-12);
        assert!((summary.max_success_probability - 0.9).abs() < 1e-12);
    }

    #[test]
    fn duplicate_model_identity_is_rejected() {
        assert!(matches!(
            ModelEnsembleSummary::from_predictions(&[
                ModelPrediction {
                    model_id: "same".into(),
                    success_probability: 0.9,
                },
                ModelPrediction {
                    model_id: "same".into(),
                    success_probability: 0.8,
                },
            ]),
            Err(PortfolioError::DuplicateModelId(_))
        ));
    }

    #[test]
    fn model_disagreement_sets_epistemic_floor() {
        let candidate = candidate(
            "disputed",
            PhysicalModality::Acoustic,
            0.9,
            1.0,
            0.5,
            0.9,
            0.4,
        );
        assert!((candidate.effective_epistemic_uncertainty().unwrap() - 0.4).abs() < 1e-12);
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
        assert!(strong.dominates(&weak).unwrap());

        let portfolio = CandidatePortfolio {
            transition: transition(),
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

        assert!(!efficient.dominates(&informative).unwrap());
        assert!(!informative.dominates(&efficient).unwrap());

        let portfolio = CandidatePortfolio {
            transition: transition(),
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
            transition: transition(),
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
            transition: transition(),
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
            transition: transition(),
            candidates: vec![wrong],
        };
        assert!(matches!(
            portfolio.evaluate(PortfolioPolicy::default()),
            Err(PortfolioError::TransitionMismatch { .. })
        ));
    }

    #[test]
    fn disallowed_modality_is_rejected_before_pareto() {
        let only_acoustic = DesiredTransition::simulation_only(
            "t-1",
            "acoustic-only diagnostic",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic],
        );
        let portfolio = CandidatePortfolio {
            transition: only_acoustic,
            candidates: vec![candidate(
                "photonic",
                PhysicalModality::Photonic,
                0.9,
                1.0,
                0.5,
                0.9,
                0.05,
            )],
        };
        assert!(matches!(
            portfolio.evaluate(PortfolioPolicy::default()),
            Err(PortfolioError::ModalityNotAllowed { .. })
        ));
    }

    #[test]
    fn transition_resource_envelope_is_enforced() {
        let mut limited = transition();
        limited.resources.max_energy_j = Some(5.0);
        limited.resources.max_power_w = Some(10.0);
        limited.resources.max_duration_ms = Some(50);

        let mut over = candidate(
            "over-budget",
            PhysicalModality::Acoustic,
            0.9,
            6.0,
            0.5,
            0.9,
            0.05,
        );
        over.expected_power_w = Some(8.0);
        over.expected_duration_ms = 40;

        let portfolio = CandidatePortfolio {
            transition: limited,
            candidates: vec![over],
        };
        assert!(matches!(
            portfolio.evaluate(PortfolioPolicy::default()),
            Err(PortfolioError::EnergyBudgetExceeded { .. })
        ));
    }

    #[test]
    fn power_budget_requires_power_estimate() {
        let mut limited = transition();
        limited.resources.max_power_w = Some(10.0);
        let portfolio = CandidatePortfolio {
            transition: limited,
            candidates: vec![candidate(
                "unknown-power",
                PhysicalModality::Acoustic,
                0.9,
                1.0,
                0.5,
                0.9,
                0.05,
            )],
        };
        assert!(matches!(
            portfolio.evaluate(PortfolioPolicy::default()),
            Err(PortfolioError::MissingPowerEstimate { .. })
        ));
    }

    #[test]
    fn non_finite_model_prediction_fails_closed() {
        let mut bad = candidate(
            "bad",
            PhysicalModality::Mechanical,
            0.9,
            1.0,
            0.5,
            0.9,
            0.1,
        );
        bad.model_predictions[0].success_probability = f64::NAN;
        assert!(bad.validate().is_err());
    }
}
