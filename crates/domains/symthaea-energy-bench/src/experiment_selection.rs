//! Transparent, safety-gated Phase A ranking for energy-conversion experiments.
//!
//! This is a deterministic screening baseline, not Bayesian optimization or
//! expected information gain. It ranks only proposals that have a registered
//! decision question, measurable observable, competing hypotheses, prior
//! registration, a complete provenance plan, and an approved safety envelope.
//! All candidates in one queue must express cost in the same preregistered unit.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ResourceBasis {
    ComputeTime,
    RigTime,
    DirectCost,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ResourceCost {
    pub basis: ResourceBasis,
    /// Canonical unit registered by the queue policy, such as "s", "rig-hour",
    /// or "USD-cent". The label is compared exactly, not converted implicitly.
    pub unit_id: String,
    pub amount: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub enum DecisionRelevance {
    Exploratory = 0,
    CharacterizesNamedUncertainty = 1,
    DiscriminatesCompetingModels = 2,
    CanFalsifyFavoredHypothesis = 3,
    CanChangeFeasibilityOrDesignDecision = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
pub enum Reproducibility {
    OneOff = 0,
    ReplayableWithDeclaredArtifacts = 1,
    IndependentlyCrossCheckable = 2,
    IndependentReplicationPlanned = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum SafetyReview {
    Pending,
    Approved,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ExperimentProposal {
    pub experiment_id: String,
    pub decision_question: String,
    pub measurable_observable: String,
    pub competing_hypotheses: Vec<String>,
    pub decision_relevance: DecisionRelevance,
    /// Expected fractional reduction in uncertainty for the named observable,
    /// expressed as a value in (0, 1], not as an information-gain estimate.
    pub expected_uncertainty_reduction_fraction: f64,
    pub estimated_resource_cost: ResourceCost,
    pub reproducibility: Reproducibility,
    pub registered_before_observation: bool,
    pub safety_review: SafetyReview,
    pub operating_envelope_registered: bool,
    pub abort_criteria_registered: bool,
    pub provenance_plan_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SelectionPolicy {
    pub policy_id: String,
    pub registered_before_queue_generation: bool,
    pub resource_basis: ResourceBasis,
    /// Canonical cost unit (for example, "s", "rig-hour", or "USD-cent").
    /// Unit conversion must happen before proposals enter the queue.
    pub resource_unit_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedExperiment {
    pub proposal: ExperimentProposal,
    /// Fractional uncertainty reduction divided by the registered cost amount.
    /// This is a transparent screening ratio, not expected information gain.
    pub uncertainty_reduction_per_cost_unit: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct RejectedExperiment {
    pub experiment_id: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankingReport {
    pub policy_id: String,
    pub resource_basis: ResourceBasis,
    pub resource_unit_id: String,
    pub eligible: Vec<RankedExperiment>,
    pub rejected: Vec<RejectedExperiment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionError {
    pub reason: String,
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for SelectionError {}

/// Rank a registered batch with deterministic, published lexicographic rules.
///
/// Ordering is: decision relevance, expected fractional uncertainty reduction,
/// uncertainty reduction per registered resource unit, reproducibility, then
/// experiment ID ascending. Safety and evidence requirements are hard gates,
/// not weights. Duplicate IDs and malformed or non-comparable proposals are
/// rejected with explicit reasons instead of silently disappearing.
pub fn rank_experiments(
    policy: &SelectionPolicy,
    proposals: &[ExperimentProposal],
) -> Result<RankingReport, SelectionError> {
    if policy.policy_id.trim().is_empty() {
        return Err(SelectionError {
            reason: "selection policy ID must be non-empty".into(),
        });
    }
    if !policy.registered_before_queue_generation {
        return Err(SelectionError {
            reason: "selection policy must be registered before queue generation".into(),
        });
    }
    if policy.resource_unit_id.trim().is_empty() {
        return Err(SelectionError {
            reason: "selection policy resource unit must be non-empty".into(),
        });
    }

    let mut id_counts = BTreeMap::<&str, usize>::new();
    for proposal in proposals {
        *id_counts.entry(proposal.experiment_id.as_str()).or_default() += 1;
    }

    let mut eligible = Vec::new();
    let mut rejected = Vec::new();

    for proposal in proposals {
        let mut reasons = Vec::new();

        if proposal.experiment_id.trim().is_empty() {
            reasons.push("experiment ID must be non-empty".into());
        } else if id_counts.get(proposal.experiment_id.as_str()).copied().unwrap_or(0) > 1 {
            reasons.push("experiment ID is duplicated in this queue".into());
        }
        if proposal.decision_question.trim().is_empty() {
            reasons.push("decision question must be non-empty".into());
        }
        if proposal.measurable_observable.trim().is_empty() {
            reasons.push("measurable observable must be named".into());
        }
        if proposal.competing_hypotheses.len() < 2
            || proposal.competing_hypotheses.iter().any(|h| h.trim().is_empty())
        {
            reasons.push("at least two non-empty competing hypotheses are required".into());
        } else {
            let hypotheses: BTreeSet<&str> = proposal
                .competing_hypotheses
                .iter()
                .map(|hypothesis| hypothesis.trim())
                .collect();
            if hypotheses.len() != proposal.competing_hypotheses.len() {
                reasons.push("competing hypotheses must be distinct".into());
            }
        }
        if !proposal.registered_before_observation {
            reasons.push("proposal was not registered before observing results".into());
        }
        if proposal.safety_review != SafetyReview::Approved {
            reasons.push("safety review is not approved".into());
        }
        if !proposal.operating_envelope_registered {
            reasons.push("operating envelope is not registered".into());
        }
        if !proposal.abort_criteria_registered {
            reasons.push("abort criteria are not registered".into());
        }
        if !proposal.provenance_plan_complete {
            reasons.push("provenance and artifact-retention plan is incomplete".into());
        }
        let reduction = proposal.expected_uncertainty_reduction_fraction;
        if !reduction.is_finite() || reduction <= 0.0 || reduction > 1.0 {
            reasons.push(
                "expected uncertainty reduction must be finite and in (0, 1]".into(),
            );
        }

        let cost = &proposal.estimated_resource_cost;
        if cost.basis != policy.resource_basis {
            reasons.push("proposal resource basis differs from registered queue policy".into());
        }
        if cost.unit_id != policy.resource_unit_id {
            reasons.push("proposal resource unit differs from registered queue policy".into());
        }
        if !cost.amount.is_finite() || cost.amount <= 0.0 {
            reasons.push("estimated resource cost must be finite and positive".into());
        }

        if !reasons.is_empty() {
            rejected.push(RejectedExperiment {
                experiment_id: proposal.experiment_id.clone(),
                reasons,
            });
            continue;
        }

        let ratio = reduction / cost.amount;
        if !ratio.is_finite() {
            rejected.push(RejectedExperiment {
                experiment_id: proposal.experiment_id.clone(),
                reasons: vec!["uncertainty-reduction-per-cost arithmetic overflowed".into()],
            });
            continue;
        }

        eligible.push(RankedExperiment {
            proposal: proposal.clone(),
            uncertainty_reduction_per_cost_unit: ratio,
        });
    }

    eligible.sort_by(|left, right| {
        right
            .proposal
            .decision_relevance
            .cmp(&left.proposal.decision_relevance)
            .then_with(|| {
                right
                    .proposal
                    .expected_uncertainty_reduction_fraction
                    .total_cmp(&left.proposal.expected_uncertainty_reduction_fraction)
            })
            .then_with(|| {
                right
                    .uncertainty_reduction_per_cost_unit
                    .total_cmp(&left.uncertainty_reduction_per_cost_unit)
            })
            .then_with(|| {
                right
                    .proposal
                    .reproducibility
                    .cmp(&left.proposal.reproducibility)
            })
            .then_with(|| {
                left.proposal
                    .experiment_id
                    .cmp(&right.proposal.experiment_id)
            })
    });

    rejected.sort_by(|left, right| left.experiment_id.cmp(&right.experiment_id));

    Ok(RankingReport {
        policy_id: policy.policy_id.clone(),
        resource_basis: policy.resource_basis,
        resource_unit_id: policy.resource_unit_id.clone(),
        eligible,
        rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SelectionPolicy {
        SelectionPolicy {
            policy_id: "preregistered-v1".into(),
            registered_before_queue_generation: true,
            resource_basis: ResourceBasis::ComputeTime,
            resource_unit_id: "s".into(),
        }
    }

    fn proposal(id: &str, relevance: DecisionRelevance, reduction: f64, cost: f64) -> ExperimentProposal {
        ExperimentProposal {
            experiment_id: id.into(),
            decision_question: "Which loss mechanism controls feasibility?".into(),
            measurable_observable: "net useful energy per duty-cycle interval".into(),
            competing_hypotheses: vec![
                "dominant heat loss".into(),
                "dominant auxiliary load".into(),
            ],
            decision_relevance: relevance,
            expected_uncertainty_reduction_fraction: reduction,
            estimated_resource_cost: ResourceCost {
                basis: ResourceBasis::ComputeTime,
                unit_id: "s".into(),
                amount: cost,
            },
            reproducibility: Reproducibility::ReplayableWithDeclaredArtifacts,
            registered_before_observation: true,
            safety_review: SafetyReview::Approved,
            operating_envelope_registered: true,
            abort_criteria_registered: true,
            provenance_plan_complete: true,
        }
    }

    #[test]
    fn ranking_is_deterministic_and_prefers_decision_relevance_before_cost_ratio() {
        let expensive_decision_test = proposal(
            "decision-critical",
            DecisionRelevance::CanChangeFeasibilityOrDesignDecision,
            0.2,
            100.0,
        );
        let cheap_exploratory_test =
            proposal("cheap-exploration", DecisionRelevance::Exploratory, 0.9, 1.0);
        let batch = vec![cheap_exploratory_test, expensive_decision_test];

        let first = rank_experiments(&policy(), &batch).unwrap();
        let second = rank_experiments(&policy(), &batch).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.eligible[0].proposal.experiment_id, "decision-critical");
        assert_eq!(first.eligible[1].proposal.experiment_id, "cheap-exploration");
        assert!(first.rejected.is_empty());
    }

    #[test]
    fn safety_preregistration_and_falsifiability_are_hard_gates() {
        let mut unsafe = proposal("unsafe", DecisionRelevance::CanFalsifyFavoredHypothesis, 0.4, 3.0);
        unsafe.safety_review = SafetyReview::Pending;
        let mut late = proposal("late", DecisionRelevance::CanFalsifyFavoredHypothesis, 0.4, 3.0);
        late.registered_before_observation = false;
        let mut unfalsifiable =
            proposal("no-alternative", DecisionRelevance::CanFalsifyFavoredHypothesis, 0.4, 3.0);
        unfalsifiable.competing_hypotheses = vec!["same".into(), "same".into()];

        let report = rank_experiments(&policy(), &[unsafe, late, unfalsifiable]).unwrap();
        assert!(report.eligible.is_empty());
        assert_eq!(report.rejected.len(), 3);
        assert!(report.rejected.iter().all(|item| !item.reasons.is_empty()));
    }

    #[test]
    fn rejects_incomparable_cost_units_nonfinite_values_and_duplicate_ids() {
        let mut wrong_unit = proposal("wrong-unit", DecisionRelevance::CharacterizesNamedUncertainty, 0.4, 3.0);
        wrong_unit.estimated_resource_cost.unit_id = "USD-cent".into();
        let mut nan_cost = proposal("nan-cost", DecisionRelevance::CharacterizesNamedUncertainty, 0.4, 3.0);
        nan_cost.estimated_resource_cost.amount = f64::NAN;
        let duplicate_a = proposal("duplicate", DecisionRelevance::Exploratory, 0.2, 1.0);
        let duplicate_b = proposal("duplicate", DecisionRelevance::Exploratory, 0.2, 2.0);

        let report = rank_experiments(
            &policy(),
            &[wrong_unit, nan_cost, duplicate_a, duplicate_b],
        )
        .unwrap();
        assert!(report.eligible.is_empty());
        assert_eq!(report.rejected.len(), 4);
        assert!(report.rejected.iter().any(|item| item.reasons.iter().any(|r| r.contains("resource unit"))));
        assert!(report.rejected.iter().any(|item| item.reasons.iter().any(|r| r.contains("cost must be finite"))));
        assert_eq!(
            report.rejected.iter().filter(|item| item.experiment_id == "duplicate").count(),
            2
        );
    }

    #[test]
    fn tie_breaking_is_stable_by_experiment_id() {
        let a = proposal("alpha", DecisionRelevance::DiscriminatesCompetingModels, 0.5, 10.0);
        let b = proposal("beta", DecisionRelevance::DiscriminatesCompetingModels, 0.5, 10.0);
        let report = rank_experiments(&policy(), &[b, a]).unwrap();
        assert_eq!(report.eligible[0].proposal.experiment_id, "alpha");
        assert_eq!(report.eligible[1].proposal.experiment_id, "beta");
    }

    #[test]
    fn output_is_invariant_to_input_order_for_eligible_and_rejected_proposals() {
        let valid_a = proposal(
            "alpha",
            DecisionRelevance::DiscriminatesCompetingModels,
            0.5,
            10.0,
        );
        let valid_b = proposal(
            "beta",
            DecisionRelevance::CanFalsifyFavoredHypothesis,
            0.2,
            3.0,
        );
        let mut rejected = proposal(
            "rejected",
            DecisionRelevance::Exploratory,
            0.4,
            1.0,
        );
        rejected.provenance_plan_complete = false;

        let forward = rank_experiments(
            &policy(),
            &[valid_a.clone(), rejected.clone(), valid_b.clone()],
        )
        .unwrap();
        let reverse = rank_experiments(&policy(), &[valid_b, rejected, valid_a]).unwrap();

        assert_eq!(forward, reverse);
        assert_eq!(forward.eligible[0].proposal.experiment_id, "beta");
        assert_eq!(forward.rejected[0].experiment_id, "rejected");
    }

    #[test]
    fn every_hard_gate_is_independently_enforced() {
        let cases: Vec<(&str, Box<dyn Fn(&mut ExperimentProposal)>)> = vec![
            ("decision question", Box::new(|p| p.decision_question.clear())),
            ("observable", Box::new(|p| p.measurable_observable.clear())),
            ("competing hypotheses", Box::new(|p| p.competing_hypotheses.clear())),
            (
                "registered before observing",
                Box::new(|p| p.registered_before_observation = false),
            ),
            (
                "safety review",
                Box::new(|p| p.safety_review = SafetyReview::Rejected),
            ),
            (
                "operating envelope",
                Box::new(|p| p.operating_envelope_registered = false),
            ),
            (
                "abort criteria",
                Box::new(|p| p.abort_criteria_registered = false),
            ),
            (
                "provenance",
                Box::new(|p| p.provenance_plan_complete = false),
            ),
            (
                "uncertainty reduction",
                Box::new(|p| p.expected_uncertainty_reduction_fraction = f64::NAN),
            ),
            (
                "resource basis",
                Box::new(|p| p.estimated_resource_cost.basis = ResourceBasis::RigTime),
            ),
            (
                "resource unit",
                Box::new(|p| p.estimated_resource_cost.unit_id = "USD-cent".into()),
            ),
            (
                "resource cost",
                Box::new(|p| p.estimated_resource_cost.amount = 0.0),
            ),
        ];

        for (expected_reason, corrupt) in cases {
            let mut invalid = proposal(
                "invalid-case",
                DecisionRelevance::CanFalsifyFavoredHypothesis,
                0.4,
                3.0,
            );
            corrupt(&mut invalid);
            let report = rank_experiments(&policy(), &[invalid]).unwrap();

            assert!(report.eligible.is_empty(), "gate not enforced: {expected_reason}");
            assert_eq!(report.rejected.len(), 1);
            assert!(
                report.rejected[0]
                    .reasons
                    .iter()
                    .any(|reason| reason.to_lowercase().contains(expected_reason)),
                "expected reason {expected_reason:?}, got {:?}",
                report.rejected[0].reasons,
            );
        }
    }

    #[test]
    fn rejects_unregistered_queue_policy() {
        let mut unregistered = policy();
        unregistered.registered_before_queue_generation = false;
        assert!(rank_experiments(&unregistered, &[]).is_err());
    }
}
