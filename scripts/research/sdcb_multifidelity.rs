//! Decision-focused multi-fidelity experiment design over a finite ambiguity set.
//!
//! Build from repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_multifidelity.rs -o /tmp/sdcb-multifidelity
//!
//! This composes the exact finite robust experiment oracle from
//! sdcb_robust_pareto.rs with explicit fidelity-specific likelihoods and a
//! componentwise resource vector. It is not a general-purpose multi-fidelity
//! Gaussian-process optimizer.

#[path = "sdcb_robust_pareto.rs"]
mod robust_reference;

use robust_reference::{
    evaluate_robust_experiment, RobustDecisionModel, RobustExperimentCandidate,
    RobustExperimentEvaluation, RobustParetoError,
};

const EPS: f64 = 1e-12;
const MAX_FIDELITY_OPTIONS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiFidelityError {
    InvalidIdentifier,
    DuplicateFidelityOption,
    InvalidResourceVector,
    InvalidBudget,
    SearchSpaceTooLarge,
    NoCandidates,
    RobustReference(RobustParetoError),
}

impl From<RobustParetoError> for MultiFidelityError {
    fn from(value: RobustParetoError) -> Self {
        Self::RobustReference(value)
    }
}

/// Resource dimensions are kept separate and are never silently scalarized.
/// Units are fixed by the task contract: money units, instrument minutes,
/// compute seconds, and energy watt-hours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceVector {
    pub money_units: f64,
    pub instrument_minutes: f64,
    pub compute_seconds: f64,
    pub energy_watt_hours: f64,
}

impl ResourceVector {
    pub fn is_valid(&self) -> bool {
        [
            self.money_units,
            self.instrument_minutes,
            self.compute_seconds,
            self.energy_watt_hours,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    }

    pub fn is_within(&self, budget: &ResourceVector) -> bool {
        self.money_units <= budget.money_units + EPS
            && self.instrument_minutes <= budget.instrument_minutes + EPS
            && self.compute_seconds <= budget.compute_seconds + EPS
            && self.energy_watt_hours <= budget.energy_watt_hours + EPS
    }

    fn no_more_expensive_than(&self, other: &ResourceVector) -> bool {
        self.money_units <= other.money_units + EPS
            && self.instrument_minutes <= other.instrument_minutes + EPS
            && self.compute_seconds <= other.compute_seconds + EPS
            && self.energy_watt_hours <= other.energy_watt_hours + EPS
    }

    fn strictly_cheaper_on_any_axis(&self, other: &ResourceVector) -> bool {
        self.money_units < other.money_units - EPS
            || self.instrument_minutes < other.instrument_minutes - EPS
            || self.compute_seconds < other.compute_seconds - EPS
            || self.energy_watt_hours < other.energy_watt_hours - EPS
    }
}

/// One experiment design offered at one fidelity.
///
/// Each fidelity has its own likelihood model under every ambiguity scenario.
/// The scenario identity is hidden from the decision policy. Resource costs
/// are a vector, not a weighted scalar.
#[derive(Debug, Clone, PartialEq)]
pub struct FidelityOption {
    pub design_id: String,
    pub fidelity_id: String,
    pub resource_cost: ResourceVector,
    /// Scenario -> hypothesis -> observed outcome likelihood.
    pub likelihoods_by_scenario: Vec<Vec<Vec<f64>>>,
}

impl FidelityOption {
    fn candidate_id(&self) -> String {
        format!("{}::{}", self.design_id, self.fidelity_id)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FidelityAssessment {
    pub design_id: String,
    pub fidelity_id: String,
    pub resource_cost: ResourceVector,
    pub robust_evaluation: RobustExperimentEvaluation,
}

impl FidelityAssessment {
    pub fn candidate_id(&self) -> String {
        format!("{}::{}", self.design_id, self.fidelity_id)
    }

    pub fn robust_gain(&self) -> f64 {
        self.robust_evaluation.robust_gain
    }
}

/// A fidelity option dominates another if it gives at least as much robust
/// decision value and costs no more on every resource axis, with a strict
/// improvement on at least one axis. No exchange rate between resources is
/// assumed.
pub fn dominates(a: &FidelityAssessment, b: &FidelityAssessment) -> bool {
    let value_no_worse = a.robust_gain() >= b.robust_gain() - EPS;
    let resources_no_worse = a.resource_cost.no_more_expensive_than(&b.resource_cost);
    let strict_value = a.robust_gain() > b.robust_gain() + EPS;
    let strict_resource = a.resource_cost.strictly_cheaper_on_any_axis(&b.resource_cost);
    value_no_worse && resources_no_worse && (strict_value || strict_resource)
}

fn validate_options(
    model: &RobustDecisionModel,
    options: &[FidelityOption],
) -> Result<(), MultiFidelityError> {
    if options.is_empty() {
        return Err(MultiFidelityError::NoCandidates);
    }
    if options.len() > MAX_FIDELITY_OPTIONS {
        return Err(MultiFidelityError::SearchSpaceTooLarge);
    }

    for (index, option) in options.iter().enumerate() {
        if option.design_id.trim().is_empty() || option.fidelity_id.trim().is_empty() {
            return Err(MultiFidelityError::InvalidIdentifier);
        }
        if !option.resource_cost.is_valid() {
            return Err(MultiFidelityError::InvalidResourceVector);
        }
        let id = option.candidate_id();
        if options[..index].iter().any(|old| old.candidate_id() == id) {
            return Err(MultiFidelityError::DuplicateFidelityOption);
        }
        if option.likelihoods_by_scenario.len() != model.scenarios.len() {
            return Err(MultiFidelityError::RobustReference(
                RobustParetoError::InvalidLikelihoodMatrix,
            ));
        }
    }
    Ok(())
}

pub fn evaluate_fidelity_options(
    prior: &[f64],
    model: &RobustDecisionModel,
    options: &[FidelityOption],
) -> Result<Vec<FidelityAssessment>, MultiFidelityError> {
    validate_options(model, options)?;
    let mut assessments = Vec::with_capacity(options.len());

    for option in options {
        let candidate = RobustExperimentCandidate {
            experiment_id: option.candidate_id(),
            // The robust oracle only uses this field as reporting metadata.
            // Actual multi-dimensional costs remain separate below.
            cost_units: 1,
            likelihoods_by_scenario: option.likelihoods_by_scenario.clone(),
        };
        let robust_evaluation = evaluate_robust_experiment(prior, model, &candidate)?;
        assessments.push(FidelityAssessment {
            design_id: option.design_id.clone(),
            fidelity_id: option.fidelity_id.clone(),
            resource_cost: option.resource_cost,
            robust_evaluation,
        });
    }
    Ok(assessments)
}

/// Return the exact non-dominated fidelity options. A zero-gain option may
/// remain Pareto-optimal if it is uniquely cheap; this is intentional and
/// visible rather than silently discarding a low-cost fallback.
pub fn fidelity_pareto_frontier(
    assessments: &[FidelityAssessment],
) -> Vec<FidelityAssessment> {
    let mut frontier = Vec::new();

    'candidate: for (i, candidate) in assessments.iter().enumerate() {
        for (j, other) in assessments.iter().enumerate() {
            if i == j {
                continue;
            }
            if dominates(other, candidate) {
                continue 'candidate;
            }
            let equal = (other.robust_gain() - candidate.robust_gain()).abs() <= EPS
                && other.resource_cost == candidate.resource_cost;
            if equal && other.candidate_id() < candidate.candidate_id() {
                continue 'candidate;
            }
        }
        frontier.push(candidate.clone());
    }

    frontier.sort_by(|a, b| {
        a.design_id.cmp(&b.design_id)
            .then_with(|| a.fidelity_id.cmp(&b.fidelity_id))
    });
    frontier
}

/// Apply componentwise resource caps, then return the Pareto frontier.
pub fn feasible_fidelity_frontier(
    assessments: &[FidelityAssessment],
    budget: &ResourceVector,
) -> Result<Vec<FidelityAssessment>, MultiFidelityError> {
    if !budget.is_valid() {
        return Err(MultiFidelityError::InvalidBudget);
    }
    let feasible: Vec<FidelityAssessment> = assessments
        .iter()
        .filter(|assessment| assessment.resource_cost.is_within(budget))
        .cloned()
        .collect();
    Ok(fidelity_pareto_frontier(&feasible))
}

/// Choose the highest robust decision value under hard resource caps. Ties are
/// resolved lexicographically; resources are not collapsed into a scalar.
pub fn best_robust_value_within_budget(
    frontier: &[FidelityAssessment],
    budget: &ResourceVector,
) -> Result<Option<FidelityAssessment>, MultiFidelityError> {
    if !budget.is_valid() {
        return Err(MultiFidelityError::InvalidBudget);
    }
    let mut best: Option<&FidelityAssessment> = None;
    for assessment in frontier {
        if !assessment.resource_cost.is_within(budget) {
            continue;
        }
        let replace = match best {
            None => true,
            Some(old) if assessment.robust_gain() > old.robust_gain() + EPS => true,
            Some(old)
                if (assessment.robust_gain() - old.robust_gain()).abs() <= EPS =>
            {
                assessment.candidate_id() < old.candidate_id()
            }
            _ => false,
        };
        if replace {
            best = Some(assessment);
        }
    }
    Ok(best.cloned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use robust_reference::UtilityScenario;

    fn model() -> RobustDecisionModel {
        RobustDecisionModel {
            action_ids: vec!["choose-h1".into(), "choose-h2".into()],
            scenarios: vec![
                UtilityScenario {
                    scenario_id: "nominal".into(),
                    utility_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                },
                UtilityScenario {
                    scenario_id: "shifted".into(),
                    utility_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                },
            ],
        }
    }

    fn resource(money: f64, lab: f64, compute: f64, energy: f64) -> ResourceVector {
        ResourceVector {
            money_units: money,
            instrument_minutes: lab,
            compute_seconds: compute,
            energy_watt_hours: energy,
        }
    }

    fn signal(q: f64) -> Vec<Vec<f64>> {
        vec![vec![q, 1.0 - q], vec![1.0 - q, q]]
    }

    fn options() -> Vec<FidelityOption> {
        vec![
            FidelityOption {
                design_id: "calibration-target".into(),
                fidelity_id: "low".into(),
                resource_cost: resource(1.0, 2.0, 0.1, 0.01),
                // A low-fidelity proxy is calibrated in the nominal setting,
                // but its outcome channel reverses under a plausible shift.
                likelihoods_by_scenario: vec![signal(0.70), signal(0.30)],
            },
            FidelityOption {
                design_id: "calibration-target".into(),
                fidelity_id: "medium".into(),
                resource_cost: resource(2.0, 5.0, 0.5, 0.10),
                likelihoods_by_scenario: vec![signal(0.75), signal(0.70)],
            },
            FidelityOption {
                design_id: "calibration-target".into(),
                fidelity_id: "high".into(),
                resource_cost: resource(6.0, 20.0, 2.0, 0.50),
                likelihoods_by_scenario: vec![signal(0.90), signal(0.85)],
            },
            FidelityOption {
                design_id: "calibration-target".into(),
                fidelity_id: "medium-expensive-duplicate".into(),
                resource_cost: resource(4.0, 8.0, 1.0, 0.20),
                // Same likelihoods as medium, but strictly higher on every
                // resource axis: the medium option should dominate it.
                likelihoods_by_scenario: vec![signal(0.75), signal(0.70)],
            },
        ]
    }

    #[test]
    fn low_fidelity_reversal_is_not_hidden_by_the_nominal_model() {
        let assessments = evaluate_fidelity_options(&[0.5, 0.5], &model(), &options()).unwrap();
        let low = assessments.iter().find(|a| a.fidelity_id == "low").unwrap();
        assert!(low.robust_gain().abs() < EPS);
        assert_eq!(
            low.robust_evaluation.policy_actions_by_outcome,
            vec!["choose-h1", "choose-h1"]
        );
        assert_eq!(low.robust_evaluation.scenario_policy_values, vec![0.5, 0.5]);
    }

    #[test]
    fn medium_fidelity_has_positive_robust_gain() {
        let assessments = evaluate_fidelity_options(&[0.5, 0.5], &model(), &options()).unwrap();
        let medium = assessments.iter().find(|a| a.fidelity_id == "medium").unwrap();
        assert!((medium.robust_gain() - 0.20).abs() < EPS);
        assert_eq!(medium.robust_evaluation.scenario_policy_values, vec![0.75, 0.70]);
    }

    #[test]
    fn high_fidelity_has_larger_gain_but_more_resource_cost() {
        let assessments = evaluate_fidelity_options(&[0.5, 0.5], &model(), &options()).unwrap();
        let high = assessments.iter().find(|a| a.fidelity_id == "high").unwrap();
        assert!((high.robust_gain() - 0.35).abs() < EPS);
        assert!(high.resource_cost.money_units > assessments[1].resource_cost.money_units);
    }

    #[test]
    fn pareto_frontier_keeps_low_medium_and_high_tradeoffs() {
        let assessments = evaluate_fidelity_options(&[0.5, 0.5], &model(), &options()).unwrap();
        let frontier = fidelity_pareto_frontier(&assessments);
        let ids: Vec<String> = frontier.iter().map(|a| a.fidelity_id.clone()).collect();
        assert_eq!(ids, vec!["high", "low", "medium"]);
        assert!(!ids.contains(&"medium-expensive-duplicate".to_string()));
    }

    #[test]
    fn hard_resource_caps_select_medium_and_exclude_high() {
        let assessments = evaluate_fidelity_options(&[0.5, 0.5], &model(), &options()).unwrap();
        let budget = resource(3.0, 8.0, 1.0, 0.20);
        let frontier = feasible_fidelity_frontier(&assessments, &budget).unwrap();
        assert_eq!(
            frontier.iter().map(|a| a.fidelity_id.as_str()).collect::<Vec<_>>(),
            vec!["low", "medium"]
        );
        let best = best_robust_value_within_budget(&frontier, &budget).unwrap().unwrap();
        assert_eq!(best.fidelity_id, "medium");
    }

    #[test]
    fn all_fidelity_scenario_values_are_preserved() {
        let assessments = evaluate_fidelity_options(&[0.5, 0.5], &model(), &options()).unwrap();
        for assessment in assessments {
            assert_eq!(assessment.robust_evaluation.scenario_policy_values.len(), 2);
        }
    }

    #[test]
    fn no_candidate_set_is_rejected() {
        assert_eq!(
            evaluate_fidelity_options(&[0.5, 0.5], &model(), &[]),
            Err(MultiFidelityError::NoCandidates)
        );
    }

    #[test]
    fn duplicate_design_fidelity_pairs_are_rejected() {
        let mut candidates = options();
        candidates.push(candidates[0].clone());
        assert_eq!(
            evaluate_fidelity_options(&[0.5, 0.5], &model(), &candidates),
            Err(MultiFidelityError::DuplicateFidelityOption)
        );
    }

    #[test]
    fn malformed_resource_vector_is_rejected() {
        let mut candidates = options();
        candidates[0].resource_cost.energy_watt_hours = f64::NAN;
        assert_eq!(
            evaluate_fidelity_options(&[0.5, 0.5], &model(), &candidates),
            Err(MultiFidelityError::InvalidResourceVector)
        );
    }

    #[test]
    fn invalid_budget_is_rejected() {
        let assessments = evaluate_fidelity_options(&[0.5, 0.5], &model(), &options()).unwrap();
        let mut budget = resource(3.0, 8.0, 1.0, 0.20);
        budget.energy_watt_hours = -1.0;
        assert_eq!(
            feasible_fidelity_frontier(&assessments, &budget),
            Err(MultiFidelityError::InvalidBudget)
        );
    }

    #[test]
    fn scenario_count_mismatch_is_rejected() {
        let mut candidates = options();
        candidates[0].likelihoods_by_scenario.pop();
        assert_eq!(
            evaluate_fidelity_options(&[0.5, 0.5], &model(), &candidates),
            Err(MultiFidelityError::RobustReference(
                RobustParetoError::InvalidLikelihoodMatrix
            ))
        );
    }

    #[test]
    fn action_policy_is_shared_across_fidelity_scenarios() {
        let mut candidates = options();
        candidates[0].fidelity_id = "low-matched".into();
        candidates[0].likelihoods_by_scenario = vec![signal(0.70), signal(0.70)];
        let assessments = evaluate_fidelity_options(&[0.5, 0.5], &model(), &candidates[..1]).unwrap();
        assert_eq!(
            assessments[0].robust_evaluation.policy_actions_by_outcome,
            vec!["choose-h1", "choose-h2"]
        );
    }

    #[test]
    fn resource_budget_does_not_scalarize_dimensions() {
        let assessments = evaluate_fidelity_options(&[0.5, 0.5], &model(), &options()).unwrap();
        let budget = resource(2.0, 20.0, 0.3, 1.0);
        let frontier = feasible_fidelity_frontier(&assessments, &budget).unwrap();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].fidelity_id, "low");
    }
}
