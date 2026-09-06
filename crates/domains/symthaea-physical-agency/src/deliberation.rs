// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Non-serializable deliberation receipts for physical-agency portfolios.
//!
//! `CandidatePortfolio` and its report-oriented `PortfolioOutcome` are ordinary
//! planning data. This module adds a stricter runtime boundary: a
//! [`SelectedCandidate`] can only be minted by evaluating a portfolio and then
//! selecting an id that actually survived onto its Pareto frontier.
//!
//! The receipt grants no execution authority. It exists solely to keep
//! deliberation lineage from being replaced by a caller-assembled proposal at
//! the later simulation-qualification boundary.

use crate::portfolio::{
    CandidateAssessment, CandidatePortfolio, PortfolioError, PortfolioOutcome, PortfolioPolicy,
};
use symthaea_physical_effects::{AbstentionReason, DesiredTransition};

/// Pareto frontier produced by an actual portfolio evaluation.
///
/// Intentionally implements neither `Serialize` nor `Deserialize`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeliberatedFrontier {
    transition: DesiredTransition,
    policy: PortfolioPolicy,
    candidates: Vec<CandidateAssessment>,
}

impl DeliberatedFrontier {
    pub fn transition(&self) -> &DesiredTransition {
        &self.transition
    }

    pub fn policy(&self) -> PortfolioPolicy {
        self.policy
    }

    pub fn candidates(&self) -> &[CandidateAssessment] {
        &self.candidates
    }

    /// Select one candidate that actually survived Pareto filtering.
    ///
    /// The resulting receipt remains deliberative only; it carries no solver,
    /// HAL, actuator, or physical-execution capability.
    pub fn select(&self, proposal_id: &str) -> Option<SelectedCandidate> {
        self.candidates
            .iter()
            .find(|candidate| candidate.proposal.id == proposal_id)
            .cloned()
            .map(|assessment| SelectedCandidate {
                transition: self.transition.clone(),
                policy: self.policy,
                assessment,
            })
    }
}

/// Non-serializable receipt that a candidate came from a specific evaluated
/// transition/policy frontier.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedCandidate {
    transition: DesiredTransition,
    policy: PortfolioPolicy,
    assessment: CandidateAssessment,
}

impl SelectedCandidate {
    pub fn transition(&self) -> &DesiredTransition {
        &self.transition
    }

    pub fn policy(&self) -> PortfolioPolicy {
        self.policy
    }

    pub fn assessment(&self) -> &CandidateAssessment {
        &self.assessment
    }
}

/// Runtime deliberation result. A frontier is still not an execution decision.
#[derive(Debug, Clone, PartialEq)]
pub enum DeliberationOutcome {
    ParetoFrontier(DeliberatedFrontier),
    Abstain(AbstentionReason),
}

/// Evaluate an ordinary portfolio and convert a surviving frontier into a
/// non-serializable runtime receipt.
pub fn deliberate(
    portfolio: &CandidatePortfolio,
    policy: PortfolioPolicy,
) -> Result<DeliberationOutcome, PortfolioError> {
    match portfolio.evaluate(policy)? {
        PortfolioOutcome::ParetoFrontier(candidates) => {
            Ok(DeliberationOutcome::ParetoFrontier(DeliberatedFrontier {
                transition: portfolio.transition.clone(),
                policy,
                candidates,
            }))
        }
        PortfolioOutcome::Abstain(reason) => Ok(DeliberationOutcome::Abstain(reason)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portfolio::ModelPrediction;
    use symthaea_physical_effects::{
        AuthorityClass, EffectKind, MechanismRef, PhysicalModality, PredictedOutcome,
        ProposedIntervention, TargetRegion,
    };

    fn portfolio() -> CandidatePortfolio {
        let transition = DesiredTransition::simulation_only(
            "t-1",
            "test deliberation receipt",
            TargetRegion::new("world", "fixture"),
            EffectKind::Characterize,
            vec![PhysicalModality::Acoustic, PhysicalModality::Photonic],
        );
        let candidate = CandidateAssessment {
            proposal: ProposedIntervention {
                id: "acoustic".into(),
                transition_id: "t-1".into(),
                mechanism: MechanismRef {
                    backend: "fixture".into(),
                    mechanism: "diagnostic".into(),
                    modality: PhysicalModality::Acoustic,
                },
                required_authority: AuthorityClass::SimulationOnly,
                predicted_outcome: PredictedOutcome {
                    success_probability: 0.9,
                    epistemic_uncertainty: 0.1,
                    aleatoric_uncertainty: 0.05,
                },
            },
            model_predictions: vec![
                ModelPrediction {
                    model_id: "a".into(),
                    success_probability: 0.9,
                },
                ModelPrediction {
                    model_id: "b".into(),
                    success_probability: 0.88,
                },
            ],
            expected_energy_j: 1.0,
            expected_power_w: None,
            expected_duration_ms: 100,
            information_gain: 0.8,
            reversibility_score: 1.0,
            safety_margin: 0.9,
        };
        CandidatePortfolio {
            transition,
            candidates: vec![candidate],
        }
    }

    #[test]
    fn only_frontier_member_can_mint_selection_receipt() {
        let frontier = match deliberate(&portfolio(), PortfolioPolicy::default()).unwrap() {
            DeliberationOutcome::ParetoFrontier(frontier) => frontier,
            other => panic!("expected frontier, got {other:?}"),
        };

        assert!(frontier.select("acoustic").is_some());
        assert!(frontier.select("not-on-frontier").is_none());
    }
}
