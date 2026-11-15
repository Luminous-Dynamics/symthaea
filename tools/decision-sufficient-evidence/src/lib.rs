//! Decision-sufficient evidence acquisition (D3) prototype.
//!
//! Computes expected value of sample information (EVSI) from an explicit
//! hypothesis prior, action-loss matrix, and observation likelihoods. Authority,
//! scope, isolation, prerequisites, and resource limits are hard feasibility
//! gates; they are never blended into the utility calculation.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::json;

const EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionProblem {
    pub problem_id: String,
    pub hypotheses: Vec<Hypothesis>,
    pub actions: Vec<Action>,
    pub budgets: HardBudgets,
    pub evidence_options: Vec<EvidenceOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hypothesis {
    pub hypothesis_id: String,
    pub prior_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Action {
    pub action_id: String,
    /// Loss for each hypothesis, in the same decision-loss units as the prior.
    pub loss_by_hypothesis: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardBudgets {
    pub max_money_units: Option<f64>,
    pub max_latency_seconds: Option<f64>,
    pub max_reviewer_minutes: Option<f64>,
    pub max_compute_units: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceOption {
    pub evidence_id: String,
    /// These are explicit gates, not probabilistic confidence multipliers.
    pub authority_satisfied: bool,
    pub scope_satisfied: bool,
    pub isolation_satisfied: bool,
    pub prerequisites_satisfied: bool,
    pub costs: ResourceCosts,
    /// Must use the same explicit utility units as action loss.
    pub acquisition_cost_utility: f64,
    pub outcomes: Vec<ObservationOutcome>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceCosts {
    pub money_units: f64,
    pub latency_seconds: f64,
    pub reviewer_minutes: f64,
    pub compute_units: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationOutcome {
    pub outcome_id: String,
    /// P(this outcome | hypothesis_i), aligned with hypotheses array order.
    pub likelihood_by_hypothesis: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateAssessment {
    pub evidence_id: String,
    pub feasible: bool,
    pub rejection_reasons: Vec<String>,
    pub baseline_expected_loss: f64,
    pub expected_loss_after_observation: Option<f64>,
    pub gross_evsi: Option<f64>,
    pub acquisition_cost_utility: f64,
    pub net_evsi: Option<f64>,
    pub outcome_probabilities: Vec<OutcomeProbability>,
    pub costs: ResourceCosts,
    pub pareto_efficient: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeProbability {
    pub outcome_id: String,
    pub probability: f64,
    pub posterior: Vec<f64>,
    pub best_action_id: String,
    pub conditional_expected_loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerReport {
    pub problem_id: String,
    pub recommendation: String,
    pub selected_evidence_id: Option<String>,
    pub baseline_action_id: String,
    pub baseline_expected_loss: f64,
    pub assessments: Vec<CandidateAssessment>,
    pub pareto_frontier: Vec<String>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone)]
struct ActionRisk {
    action_id: String,
    expected_loss: f64,
}

fn is_finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn validate_problem(problem: &DecisionProblem) -> Result<(), String> {
    if problem.problem_id.trim().is_empty() {
        return Err("problem_id must not be empty".to_owned());
    }
    if problem.hypotheses.is_empty() || problem.actions.is_empty() {
        return Err("at least one hypothesis and one action are required".to_owned());
    }
    let mut hypothesis_ids = HashSet::new();
    let mut prior_sum = 0.0;
    for hypothesis in &problem.hypotheses {
        if hypothesis.hypothesis_id.trim().is_empty() || !hypothesis_ids.insert(&hypothesis.hypothesis_id) {
            return Err("hypothesis IDs must be non-empty and unique".to_owned());
        }
        if !hypothesis.prior_probability.is_finite() || hypothesis.prior_probability < 0.0 {
            return Err(format!("invalid prior probability for {}", hypothesis.hypothesis_id));
        }
        prior_sum += hypothesis.prior_probability;
    }
    if (prior_sum - 1.0).abs() > EPSILON {
        return Err(format!("hypothesis priors must sum to 1; found {prior_sum}"));
    }

    let mut action_ids = HashSet::new();
    for action in &problem.actions {
        if action.action_id.trim().is_empty() || !action_ids.insert(&action.action_id) {
            return Err("action IDs must be non-empty and unique".to_owned());
        }
        if action.loss_by_hypothesis.len() != problem.hypotheses.len()
            || action.loss_by_hypothesis.iter().any(|loss| !is_finite_nonnegative(*loss))
        {
            return Err(format!(
                "action {} must provide one finite non-negative loss per hypothesis",
                action.action_id
            ));
        }
    }

    for (name, limit) in [
        ("max_money_units", problem.budgets.max_money_units),
        ("max_latency_seconds", problem.budgets.max_latency_seconds),
        ("max_reviewer_minutes", problem.budgets.max_reviewer_minutes),
        ("max_compute_units", problem.budgets.max_compute_units),
    ] {
        if limit.is_some_and(|value| !is_finite_nonnegative(value)) {
            return Err(format!("{name} must be finite and non-negative"));
        }
    }
    let mut evidence_ids = HashSet::new();
    for option in &problem.evidence_options {
        if option.evidence_id.trim().is_empty() || !evidence_ids.insert(&option.evidence_id) {
            return Err("evidence option IDs must be non-empty and unique".to_owned());
        }
        if !is_finite_nonnegative(option.acquisition_cost_utility) {
            return Err(format!("{} acquisition_cost_utility must be finite and non-negative", option.evidence_id));
        }
        for (name, value) in [
            ("money_units", option.costs.money_units),
            ("latency_seconds", option.costs.latency_seconds),
            ("reviewer_minutes", option.costs.reviewer_minutes),
            ("compute_units", option.costs.compute_units),
        ] {
            if !is_finite_nonnegative(value) {
                return Err(format!("{} cost {name} must be finite and non-negative", option.evidence_id));
            }
        }
    }
    Ok(())
}

fn action_risks(problem: &DecisionProblem, distribution: &[f64]) -> Vec<ActionRisk> {
    problem.actions.iter().map(|action| ActionRisk {
        action_id: action.action_id.clone(),
        expected_loss: action.loss_by_hypothesis.iter()
            .zip(distribution)
            .map(|(loss, probability)| loss * probability)
            .sum(),
    }).collect()
}

fn best_action(risks: &[ActionRisk]) -> &ActionRisk {
    risks.iter().min_by(|left, right| {
        left.expected_loss.total_cmp(&right.expected_loss)
            .then_with(|| left.action_id.cmp(&right.action_id))
    }).expect("action list validated as non-empty")
}

fn candidate_gate_reasons(problem: &DecisionProblem, option: &EvidenceOption) -> Vec<String> {
    let mut reasons = Vec::new();
    if !option.authority_satisfied { reasons.push("authority_not_satisfied".to_owned()); }
    if !option.scope_satisfied { reasons.push("scope_not_satisfied".to_owned()); }
    if !option.isolation_satisfied { reasons.push("execution_isolation_not_satisfied".to_owned()); }
    if !option.prerequisites_satisfied { reasons.push("prerequisites_not_satisfied".to_owned()); }

    if problem.budgets.max_money_units.is_some_and(|limit| option.costs.money_units > limit) {
        reasons.push("money_budget_exceeded".to_owned());
    }
    if problem.budgets.max_latency_seconds.is_some_and(|limit| option.costs.latency_seconds > limit) {
        reasons.push("latency_budget_exceeded".to_owned());
    }
    if problem.budgets.max_reviewer_minutes.is_some_and(|limit| option.costs.reviewer_minutes > limit) {
        reasons.push("review_budget_exceeded".to_owned());
    }
    if problem.budgets.max_compute_units.is_some_and(|limit| option.costs.compute_units > limit) {
        reasons.push("compute_budget_exceeded".to_owned());
    }
    reasons
}

fn validate_likelihoods(option: &EvidenceOption, hypothesis_count: usize) -> Result<(), String> {
    if option.outcomes.is_empty() {
        return Err("at least one observation outcome is required".to_owned());
    }
    let mut outcome_ids = HashSet::new();
    let mut column_sums = vec![0.0; hypothesis_count];
    for outcome in &option.outcomes {
        if outcome.outcome_id.trim().is_empty() || !outcome_ids.insert(&outcome.outcome_id) {
            return Err("observation outcome IDs must be non-empty and unique".to_owned());
        }
        if outcome.likelihood_by_hypothesis.len() != hypothesis_count {
            return Err(format!(
                "outcome {} must provide one likelihood per hypothesis",
                outcome.outcome_id
            ));
        }
        for (index, likelihood) in outcome.likelihood_by_hypothesis.iter().enumerate() {
            if !likelihood.is_finite() || !(0.0..=1.0).contains(likelihood) {
                return Err(format!("outcome {} has an invalid likelihood", outcome.outcome_id));
            }
            column_sums[index] += likelihood;
        }
    }
    for (index, sum) in column_sums.iter().enumerate() {
        if (sum - 1.0).abs() > EPSILON {
            return Err(format!(
                "observation likelihoods for hypothesis {} must sum to 1; found {sum}",
                index
            ));
        }
    }
    Ok(())
}

fn assess_option(
    problem: &DecisionProblem,
    option: &EvidenceOption,
    baseline: &ActionRisk,
) -> CandidateAssessment {
    let rejection_reasons = candidate_gate_reasons(problem, option);
    if let Err(error) = validate_likelihoods(option, problem.hypotheses.len()) {
        return CandidateAssessment {
            evidence_id: option.evidence_id.clone(),
            feasible: false,
            rejection_reasons: {
                let mut reasons = rejection_reasons;
                reasons.push(format!("invalid_observation_model:{error}"));
                reasons
            },
            baseline_expected_loss: baseline.expected_loss,
            expected_loss_after_observation: None,
            gross_evsi: None,
            acquisition_cost_utility: option.acquisition_cost_utility,
            net_evsi: None,
            outcome_probabilities: Vec::new(),
            costs: option.costs.clone(),
            pareto_efficient: false,
        };
    }

    let mut expected_post_observation_loss = 0.0;
    let mut outcome_probabilities = Vec::new();
    for outcome in &option.outcomes {
        let unnormalized: Vec<f64> = problem.hypotheses.iter()
            .zip(&outcome.likelihood_by_hypothesis)
            .map(|(hypothesis, likelihood)| hypothesis.prior_probability * likelihood)
            .collect();
        let probability: f64 = unnormalized.iter().sum();
        if probability <= EPSILON {
            outcome_probabilities.push(OutcomeProbability {
                outcome_id: outcome.outcome_id.clone(),
                probability: 0.0,
                posterior: vec![0.0; problem.hypotheses.len()],
                best_action_id: baseline.action_id.clone(),
                conditional_expected_loss: 0.0,
            });
            continue;
        }
        let posterior: Vec<f64> = unnormalized.iter().map(|weight| weight / probability).collect();
        let risks = action_risks(problem, &posterior);
        let best = best_action(&risks);
        expected_post_observation_loss += probability * best.expected_loss;
        outcome_probabilities.push(OutcomeProbability {
            outcome_id: outcome.outcome_id.clone(),
            probability,
            posterior,
            best_action_id: best.action_id.clone(),
            conditional_expected_loss: best.expected_loss,
        });
    }

    // EVSI is the expected reduction in optimal Bayes risk before acquisition cost.
    let gross = (baseline.expected_loss - expected_post_observation_loss).max(0.0);
    let net = gross - option.acquisition_cost_utility;
    let feasible = rejection_reasons.is_empty();
    CandidateAssessment {
        evidence_id: option.evidence_id.clone(),
        feasible,
        rejection_reasons,
        baseline_expected_loss: baseline.expected_loss,
        expected_loss_after_observation: Some(expected_post_observation_loss),
        gross_evsi: Some(gross),
        acquisition_cost_utility: option.acquisition_cost_utility,
        net_evsi: Some(net),
        outcome_probabilities,
        costs: option.costs.clone(),
        pareto_efficient: false,
    }
}

fn dominates(left: &CandidateAssessment, right: &CandidateAssessment) -> bool {
    let left_benefit = left.gross_evsi.unwrap_or(f64::NEG_INFINITY);
    let right_benefit = right.gross_evsi.unwrap_or(f64::NEG_INFINITY);
    let benefit_no_worse = left_benefit >= right_benefit - EPSILON;
    let dimensions = [
        (left.costs.money_units, right.costs.money_units),
        (left.costs.latency_seconds, right.costs.latency_seconds),
        (left.costs.reviewer_minutes, right.costs.reviewer_minutes),
        (left.costs.compute_units, right.costs.compute_units),
        (left.acquisition_cost_utility, right.acquisition_cost_utility),
    ];
    let costs_no_worse = dimensions.iter().all(|(a,b)| a <= &(b + EPSILON));
    let strictly_better = left_benefit > right_benefit + EPSILON
        || dimensions.iter().any(|(a,b)| a + EPSILON < *b);
    benefit_no_worse && costs_no_worse && strictly_better
}

/// Evaluate candidate observations without treating authority or safety as a score.
pub fn plan_evidence(problem: &DecisionProblem) -> Result<PlannerReport, String> {
    validate_problem(problem)?;
    let prior: Vec<f64> = problem.hypotheses.iter().map(|item| item.prior_probability).collect();
    let risks = action_risks(problem, &prior);
    let baseline = best_action(&risks);

    let mut assessments: Vec<CandidateAssessment> = problem.evidence_options.iter()
        .map(|option| assess_option(problem, option, baseline))
        .collect();

    let feasible_indices: Vec<usize> = assessments.iter().enumerate()
        .filter_map(|(index, assessment)| assessment.feasible.then_some(index))
        .collect();
    for &index in &feasible_indices {
        let dominated = feasible_indices.iter().any(|&other| {
            other != index && dominates(&assessments[other], &assessments[index])
        });
        assessments[index].pareto_efficient = !dominated;
    }

    let mut selected: Vec<&CandidateAssessment> = assessments.iter()
        .filter(|assessment| assessment.feasible && assessment.net_evsi.is_some_and(|value| value > EPSILON))
        .collect();
    selected.sort_by(|left, right| {
        right.net_evsi.unwrap().total_cmp(&left.net_evsi.unwrap())
            .then_with(|| left.acquisition_cost_utility.total_cmp(&right.acquisition_cost_utility))
            .then_with(|| left.evidence_id.cmp(&right.evidence_id))
    });
    let selected_evidence_id = selected.first().map(|assessment| assessment.evidence_id.clone());
    let recommendation = if selected_evidence_id.is_some() {
        "acquire_evidence"
    } else {
        "no_positive_net_value_observation"
    };
    let pareto_frontier = assessments.iter()
        .filter(|assessment| assessment.feasible && assessment.pareto_efficient)
        .map(|assessment| assessment.evidence_id.clone())
        .collect();

    Ok(PlannerReport {
        problem_id: problem.problem_id.clone(),
        recommendation: recommendation.to_owned(),
        selected_evidence_id,
        baseline_action_id: baseline.action_id.clone(),
        baseline_expected_loss: baseline.expected_loss,
        assessments,
        pareto_frontier,
        claim_ceiling: "Decision-theoretic ranking over caller-supplied priors, losses, likelihoods and cost utility only. Authority/scope/isolation/prerequisites are caller-declared hard gates, not independently verified. No causal effect, observation quality, probability calibration, execution, or downstream outcome is authenticated.".to_owned(),
    })
}

pub fn plan_evidence_json(input: &str) -> Result<PlannerReport, String> {
    let problem: DecisionProblem = serde_json::from_str(input).map_err(|error| error.to_string())?;
    plan_evidence(&problem)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> DecisionProblem {
        DecisionProblem {
            problem_id: "contract-upgrade-example".to_owned(),
            hypotheses: vec![
                Hypothesis { hypothesis_id: "compatible".to_owned(), prior_probability: 0.5 },
                Hypothesis { hypothesis_id: "breaking".to_owned(), prior_probability: 0.5 },
            ],
            actions: vec![
                Action { action_id: "ship".to_owned(), loss_by_hypothesis: vec![0.0, 100.0] },
                Action { action_id: "hold".to_owned(), loss_by_hypothesis: vec![5.0, 5.0] },
            ],
            budgets: HardBudgets {
                max_money_units: Some(3.0),
                max_latency_seconds: Some(10.0),
                max_reviewer_minutes: Some(3.0),
                max_compute_units: Some(4.0),
            },
            evidence_options: vec![
                EvidenceOption {
                    evidence_id: "pinned-contract-test".to_owned(),
                    authority_satisfied: true,
                    scope_satisfied: true,
                    isolation_satisfied: true,
                    prerequisites_satisfied: true,
                    costs: ResourceCosts { money_units: 2.0, latency_seconds: 5.0, reviewer_minutes: 1.0, compute_units: 3.0 },
                    acquisition_cost_utility: 1.0,
                    outcomes: vec![
                        ObservationOutcome { outcome_id: "pass".to_owned(), likelihood_by_hypothesis: vec![1.0, 0.0] },
                        ObservationOutcome { outcome_id: "fail".to_owned(), likelihood_by_hypothesis: vec![0.0, 1.0] },
                    ],
                },
                EvidenceOption {
                    evidence_id: "uninformative-search".to_owned(),
                    authority_satisfied: true,
                    scope_satisfied: true,
                    isolation_satisfied: true,
                    prerequisites_satisfied: true,
                    costs: ResourceCosts { money_units: 0.5, latency_seconds: 1.0, reviewer_minutes: 0.5, compute_units: 0.5 },
                    acquisition_cost_utility: 0.2,
                    outcomes: vec![
                        ObservationOutcome { outcome_id: "hit".to_owned(), likelihood_by_hypothesis: vec![0.5, 0.5] },
                        ObservationOutcome { outcome_id: "miss".to_owned(), likelihood_by_hypothesis: vec![0.5, 0.5] },
                    ],
                },
            ],
        }
    }

    #[test] fn perfect_test_has_positive_evsi_and_is_selected() {
        let report = plan_evidence(&problem()).expect("plan");
        assert_eq!(report.baseline_action_id, "hold");
        assert!((report.baseline_expected_loss - 5.0).abs() < EPSILON);
        assert_eq!(report.recommendation, "acquire_evidence");
        assert_eq!(report.selected_evidence_id.as_deref(), Some("pinned-contract-test"));
        let candidate = report.assessments.iter().find(|item| item.evidence_id == "pinned-contract-test").unwrap();
        assert!((candidate.gross_evsi.unwrap() - 5.0).abs() < EPSILON);
        assert!((candidate.net_evsi.unwrap() - 4.0).abs() < EPSILON);
    }

    #[test] fn uninformative_observation_does_not_justify_acquisition() {
        let mut input = problem();
        input.evidence_options = vec![input.evidence_options[1].clone()];
        let report = plan_evidence(&input).expect("plan");
        assert_eq!(report.recommendation, "no_positive_net_value_observation");
        assert_eq!(report.selected_evidence_id, None);
        let assessment = &report.assessments[0];
        assert!(assessment.gross_evsi.unwrap().abs() < EPSILON);
        assert!(assessment.net_evsi.unwrap() < 0.0);
    }

    #[test] fn unauthorized_observation_is_rejected_even_when_perfectly_informative() {
        let mut input = problem();
        input.evidence_options[0].authority_satisfied = false;
        let report = plan_evidence(&input).expect("plan");
        assert_eq!(report.selected_evidence_id, None);
        assert!(!report.assessments[0].feasible);
        assert!(report.assessments[0].rejection_reasons.contains(&"authority_not_satisfied".to_owned()));
    }

    #[test] fn resource_budget_is_a_hard_constraint() {
        let mut input = problem();
        input.evidence_options[0].costs.money_units = 4.0;
        let report = plan_evidence(&input).expect("plan");
        assert_eq!(report.selected_evidence_id.as_deref(), Some("uninformative-search").filter(|_| false));
        assert!(!report.assessments[0].feasible);
        assert!(report.assessments[0].rejection_reasons.contains(&"money_budget_exceeded".to_owned()));
    }

    #[test] fn malformed_likelihood_distribution_is_rejected_as_infeasible() {
        let mut input = problem();
        input.evidence_options[0].outcomes[0].likelihood_by_hypothesis[0] = 0.4;
        let report = plan_evidence(&input).expect("plan report");
        assert!(!report.assessments[0].feasible);
        assert!(report.assessments[0].rejection_reasons.iter().any(|reason| reason.starts_with("invalid_observation_model:")));
    }

    #[test] fn invalid_prior_distribution_is_rejected() {
        let mut input = problem();
        input.hypotheses[0].prior_probability = 0.9;
        assert!(plan_evidence(&input).is_err());
    }

    #[test] fn hard_gates_are_not_hidden_inside_net_evsi() {
        let mut input = problem();
        input.evidence_options[0].isolation_satisfied = false;
        input.evidence_options[0].acquisition_cost_utility = 0.0;
        let report = plan_evidence(&input).expect("plan");
        assert!(!report.assessments[0].feasible);
        assert!(report.assessments[0].gross_evsi.unwrap() > 0.0);
        assert_eq!(report.selected_evidence_id, None);
    }
}
