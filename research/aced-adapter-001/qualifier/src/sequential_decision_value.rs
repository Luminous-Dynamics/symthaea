use crate::decision_value::{evaluate_experiment, Experiment};
use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const VALUE_TOLERANCE: f64 = 1e-10;

#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeBranch {
    pub outcome: String,
    pub probability: f64,
    pub continuation: Box<SequentialPlanNode>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SequentialAction {
    Stop,
    RunExperiment {
        experiment_id: String,
        outcomes: Vec<OutcomeBranch>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SequentialPlanNode {
    /// Expected downstream decision utility net of future experiment cost/risk,
    /// conditional on the evidence state at this node.
    pub value: f64,
    pub expected_experiments: f64,
    pub expected_experiment_cost: f64,
    pub expected_risk_penalty: f64,
    pub action: SequentialAction,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SequentialPlanReport {
    pub baseline_value: f64,
    pub planned_value: f64,
    pub net_value_of_plan: f64,
    pub expected_experiments: f64,
    pub expected_experiment_cost: f64,
    pub expected_risk_penalty: f64,
    pub max_depth: usize,
    pub tree: SequentialPlanNode,
}

fn best_expected_utility(prior: &[f64], action_utilities: &[Vec<f64>]) -> f64 {
    action_utilities
        .iter()
        .map(|row| prior.iter().zip(row).map(|(p, utility)| p * utility).sum())
        .fold(f64::NEG_INFINITY, f64::max)
}

fn validate_prior_and_actions(prior: &[f64], action_utilities: &[Vec<f64>]) -> Result<(), String> {
    if prior.is_empty() || action_utilities.is_empty() {
        return Err("prior and action utility table must be non-empty".into());
    }
    if prior.iter().any(|p| !p.is_finite() || *p < 0.0)
        || (prior.iter().sum::<f64>() - 1.0).abs() > PROBABILITY_TOLERANCE
    {
        return Err("prior probabilities must be finite, non-negative and sum to one".into());
    }
    if action_utilities.iter().any(|row| {
        row.len() != prior.len() || row.iter().any(|utility| !utility.is_finite())
    }) {
        return Err("each action utility row must match the hypothesis count and be finite".into());
    }
    Ok(())
}

fn posterior(prior: &[f64], likelihoods: &[f64]) -> Result<(f64, Vec<f64>), String> {
    if prior.len() != likelihoods.len() {
        return Err("posterior prior/likelihood dimensions differ".into());
    }
    let probability = prior
        .iter()
        .zip(likelihoods)
        .map(|(prior, likelihood)| prior * likelihood)
        .sum::<f64>();
    if probability <= PROBABILITY_TOLERANCE {
        return Ok((0.0, vec![0.0; prior.len()]));
    }
    let updated = prior
        .iter()
        .zip(likelihoods)
        .map(|(prior, likelihood)| prior * likelihood / probability)
        .collect();
    Ok((probability, updated))
}

fn solve_node(
    prior: &[f64],
    action_utilities: &[Vec<f64>],
    experiments: &[Experiment],
    available: &[usize],
    remaining_budget: f64,
    risk_aversion: f64,
    minimum_net_value: f64,
    remaining_depth: usize,
) -> Result<SequentialPlanNode, String> {
    let stop_value = best_expected_utility(prior, action_utilities);
    let stop = SequentialPlanNode {
        value: stop_value,
        expected_experiments: 0.0,
        expected_experiment_cost: 0.0,
        expected_risk_penalty: 0.0,
        action: SequentialAction::Stop,
    };

    if remaining_depth == 0 || available.is_empty() {
        return Ok(stop);
    }

    let mut candidates = available
        .iter()
        .copied()
        .filter(|index| experiments[*index].cost <= remaining_budget)
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| experiments[*left].id.cmp(&experiments[*right].id));

    let mut best = stop;
    for index in candidates {
        let experiment = &experiments[index];
        let mut expected_value = 0.0;
        let mut expected_followup_experiments = 0.0;
        let mut expected_followup_cost = 0.0;
        let mut expected_followup_risk = 0.0;
        let mut branches = Vec::with_capacity(experiment.outcomes.len());
        let next_available = available
            .iter()
            .copied()
            .filter(|candidate| *candidate != index)
            .collect::<Vec<_>>();

        for outcome in &experiment.outcomes {
            let (probability, updated_prior) =
                posterior(prior, &outcome.likelihood_by_hypothesis)?;
            if probability <= PROBABILITY_TOLERANCE {
                continue;
            }
            let continuation = solve_node(
                &updated_prior,
                action_utilities,
                experiments,
                &next_available,
                remaining_budget - experiment.cost,
                risk_aversion,
                minimum_net_value,
                remaining_depth - 1,
            )?;
            expected_value += probability * continuation.value;
            expected_followup_experiments += probability * continuation.expected_experiments;
            expected_followup_cost += probability * continuation.expected_experiment_cost;
            expected_followup_risk += probability * continuation.expected_risk_penalty;
            branches.push(OutcomeBranch {
                outcome: outcome.outcome.clone(),
                probability,
                continuation: Box::new(continuation),
            });
        }

        let risk_penalty = risk_aversion * experiment.risk;
        let value = expected_value - experiment.cost - risk_penalty;
        let net_value_over_stop = value - stop_value;
        let lexical_better = match &best.action {
            SequentialAction::Stop => true,
            SequentialAction::RunExperiment { experiment_id, .. } => {
                experiment.id.as_str() < experiment_id.as_str()
            }
        };
        if net_value_over_stop > minimum_net_value + VALUE_TOLERANCE
            && (value > best.value + VALUE_TOLERANCE
                || ((value - best.value).abs() <= VALUE_TOLERANCE && lexical_better))
        {
            best = SequentialPlanNode {
                value,
                expected_experiments: 1.0 + expected_followup_experiments,
                expected_experiment_cost: experiment.cost + expected_followup_cost,
                expected_risk_penalty: risk_penalty + expected_followup_risk,
                action: SequentialAction::RunExperiment {
                    experiment_id: experiment.id.clone(),
                    outcomes: branches,
                },
            };
        }
    }

    Ok(best)
}

/// Plan over a bounded finite horizon with adaptive follow-up decisions.
///
/// Every candidate is validated before budget filtering. An experiment may be
/// used at most once on each path. At every node the solver compares continuing
/// with stopping. This is exact enumeration over the supplied finite model,
/// not a general POMDP solver or a guarantee outside the supplied likelihoods.
pub fn plan_sequential_experiments(
    prior: &[f64],
    action_utilities: &[Vec<f64>],
    experiments: &[Experiment],
    available_budget: f64,
    risk_aversion: f64,
    minimum_net_value: f64,
    max_depth: usize,
) -> Result<SequentialPlanReport, String> {
    validate_prior_and_actions(prior, action_utilities)?;
    if !available_budget.is_finite() || available_budget < 0.0 {
        return Err("available budget must be finite and non-negative".into());
    }
    if !risk_aversion.is_finite() || risk_aversion < 0.0 {
        return Err("risk aversion must be finite and non-negative".into());
    }
    if !minimum_net_value.is_finite() || minimum_net_value < 0.0 {
        return Err("minimum net value must be finite and non-negative".into());
    }
    let mut identities = BTreeSet::new();
    for experiment in experiments {
        if experiment.id.trim().is_empty() || !identities.insert(experiment.id.as_str()) {
            return Err("experiment identities must be unique and non-empty".into());
        }
        evaluate_experiment(prior, action_utilities, experiment, risk_aversion)
            .map_err(|error| format!("experiment {} is invalid: {error}", experiment.id))?;
    }

    let baseline_value = best_expected_utility(prior, action_utilities);
    let available = (0..experiments.len()).collect::<Vec<_>>();
    let tree = solve_node(
        prior,
        action_utilities,
        experiments,
        &available,
        available_budget,
        risk_aversion,
        minimum_net_value,
        max_depth,
    )?;

    Ok(SequentialPlanReport {
        baseline_value,
        planned_value: tree.value,
        net_value_of_plan: tree.value - baseline_value,
        expected_experiments: tree.expected_experiments,
        expected_experiment_cost: tree.expected_experiment_cost,
        expected_risk_penalty: tree.expected_risk_penalty,
        max_depth,
        tree,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_value::{ObservationLikelihood, Experiment};

    fn probe(id: &str, cost: f64, partitions: &[(&str, &[f64])]) -> Experiment {
        Experiment {
            id: id.into(),
            cost,
            risk: 0.0,
            outcomes: partitions
                .iter()
                .map(|(name, likelihood)| ObservationLikelihood {
                    outcome: (*name).into(),
                    likelihood_by_hypothesis: likelihood.to_vec(),
                })
                .collect(),
        }
    }

    fn complementarity_problem() -> (Vec<f64>, Vec<Vec<f64>>, Vec<Experiment>) {
        let prior = vec![0.25, 0.25, 0.25, 0.25];
        let actions = vec![
            vec![10.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 10.0],
        ];
        // Each individual partition contains one A-state and one B-state,
        // so neither observation alone changes the optimal terminal action.
        // Together, the two orthogonal partitions identify the correct action.
        let first = probe(
            "partition_alpha",
            0.1,
            &[
                ("alpha_yes", &[1.0, 0.0, 1.0, 0.0]),
                ("alpha_no", &[0.0, 1.0, 0.0, 1.0]),
            ],
        );
        let second = probe(
            "partition_beta",
            0.1,
            &[
                ("beta_yes", &[1.0, 0.0, 0.0, 1.0]),
                ("beta_no", &[0.0, 1.0, 1.0, 0.0]),
            ],
        );
        (prior, actions, vec![first, second])
    }

    #[test]
    fn two_step_lookahead_finds_complementary_probes_greedy_evsi_rejects() {
        let (prior, actions, experiments) = complementarity_problem();
        for experiment in &experiments {
            let one_step =
                crate::decision_value::evaluate_experiment(&prior, &actions, experiment, 0.0)
                    .unwrap();
            assert!(one_step.expected_value_of_information < 1e-10);
        }

        let plan = plan_sequential_experiments(
            &prior,
            &actions,
            &experiments,
            1.0,
            0.0,
            0.0,
            2,
        )
        .unwrap();
        assert!(plan.net_value_of_plan > 4.0);
        assert!((plan.expected_experiments - 2.0).abs() < 1e-10);
        match &plan.tree.action {
            SequentialAction::RunExperiment {
                experiment_id,
                outcomes,
            } => {
                assert_eq!(experiment_id, "partition_alpha");
                assert_eq!(outcomes.len(), 2);
                assert!(outcomes.iter().all(|branch| {
                    matches!(
                        branch.continuation.action,
                        SequentialAction::RunExperiment { ref experiment_id, .. }
                            if experiment_id == "partition_beta"
                    )
                }));
            }
            SequentialAction::Stop => panic!("complementary sequence should not stop"),
        }
    }

    #[test]
    fn zero_budget_stops_without_dispatch() {
        let (prior, actions, experiments) = complementarity_problem();
        let plan = plan_sequential_experiments(&prior, &actions, &experiments, 0.0, 0.0, 0.0, 2)
            .unwrap();
        assert_eq!(plan.net_value_of_plan, 0.0);
        assert_eq!(plan.expected_experiments, 0.0);
        assert!(matches!(plan.tree.action, SequentialAction::Stop));
    }

    #[test]
    fn budget_allows_only_one_complementary_probe_then_stops() {
        let (prior, actions, experiments) = complementarity_problem();
        let plan = plan_sequential_experiments(&prior, &actions, &experiments, 0.15, 0.0, 0.0, 2)
            .unwrap();
        assert_eq!(plan.net_value_of_plan, 0.0);
        assert_eq!(plan.expected_experiments, 0.0);
        assert!(matches!(plan.tree.action, SequentialAction::Stop));
    }

    #[test]
    fn risk_penalty_can_reverse_the_best_sequence() {
        let (prior, actions, experiments) = complementarity_problem();
        let mut risky = experiments.clone();
        risky[0].risk = 3.0;
        risky[1].risk = 0.0;
        let plan = plan_sequential_experiments(&prior, &actions, &risky, 1.0, 2.0, 0.0, 2)
            .unwrap();
        assert!(plan.net_value_of_plan <= 0.0);
        assert!(matches!(plan.tree.action, SequentialAction::Stop));
    }

    #[test]
    fn empty_candidate_set_stops_cleanly() {
        let prior = vec![0.5, 0.5];
        let actions = vec![vec![10.0, 0.0], vec![0.0, 10.0]];
        let plan = plan_sequential_experiments(&prior, &actions, &[], 0.0, 0.0, 0.0, 2)
            .unwrap();
        assert_eq!(plan.net_value_of_plan, 0.0);
        assert_eq!(plan.expected_experiments, 0.0);
        assert!(matches!(plan.tree.action, SequentialAction::Stop));
    }

    #[test]
    fn horizon_may_exceed_available_candidates_and_stops_after_exhaustion() {
        let (prior, actions, experiments) = complementarity_problem();
        let plan = plan_sequential_experiments(&prior, &actions, &experiments[..1], 1.0, 0.0, 0.0, 2)
            .unwrap();
        assert_eq!(plan.net_value_of_plan, 0.0);
        assert_eq!(plan.expected_experiments, 0.0);
        assert!(matches!(plan.tree.action, SequentialAction::Stop));
    }

    #[test]
    fn malformed_unaffordable_candidate_fails_closed() {
        let (prior, actions, mut experiments) = complementarity_problem();
        experiments[0].cost = 10.0;
        experiments[0].outcomes[0].likelihood_by_hypothesis[0] = 0.25;
        assert!(plan_sequential_experiments(&prior, &actions, &experiments, 0.0, 0.0, 0.0, 2)
            .is_err());
    }

    #[test]
    fn duplicate_experiment_identity_fails_closed() {
        let (prior, actions, mut experiments) = complementarity_problem();
        experiments[1].id = experiments[0].id.clone();
        assert!(plan_sequential_experiments(&prior, &actions, &experiments, 1.0, 0.0, 0.0, 2)
            .is_err());
    }
}
