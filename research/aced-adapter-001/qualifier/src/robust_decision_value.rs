use crate::decision_value::{Experiment, evaluate_experiment};
use std::collections::{BTreeMap, BTreeSet};

const IDENTITY_TOLERANCE: f64 = 1e-9;

/// One independently specified plausible decision model.
/// Scenarios are a sensitivity set, not probability-weighted ensemble members.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionModelScenario {
    pub scenario_id: String,
    pub prior: Vec<f64>,
    pub action_utilities: Vec<Vec<f64>>,
    pub experiments: Vec<Experiment>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustExperimentValue {
    pub experiment_id: String,
    pub scenario_count: usize,
    pub minimum_evsi: f64,
    pub maximum_evsi: f64,
    /// Descriptive arithmetic mean only; no probability over scenarios is assumed.
    pub mean_scenario_net_value_descriptive_only: f64,
    pub minimum_scenario_net_value: f64,
    pub maximum_scenario_net_value: f64,
}

/// Compare the same proposed experiment under every declared plausible model.
///
/// This is a finite-set robust criterion, not a Bayesian posterior over models.
/// Scenario construction and coverage must be fixed before outcomes are observed.
pub fn select_robust_evsi(
    scenarios: &[DecisionModelScenario],
    available_budget: f64,
    risk_aversion: f64,
    minimum_worst_case_net_value: f64,
) -> Result<Option<RobustExperimentValue>, String> {
    if scenarios.len() < 2 {
        return Err("robust EVSI requires at least two declared plausible model scenarios".into());
    }
    if !available_budget.is_finite() || available_budget < 0.0 {
        return Err("available experiment budget must be finite and non-negative".into());
    }
    if !risk_aversion.is_finite() || risk_aversion < 0.0 {
        return Err("risk aversion must be finite and non-negative".into());
    }
    if !minimum_worst_case_net_value.is_finite() {
        return Err("minimum worst-case net value must be finite".into());
    }

    let mut scenario_ids = BTreeSet::new();
    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.as_str())
        {
            return Err("scenario identities must be non-empty and unique".into());
        }
    }

    let mut reference = BTreeMap::<String, (f64, f64)>::new();
    for experiment in &scenarios[0].experiments {
        if experiment.id.trim().is_empty()
            || reference
                .insert(experiment.id.clone(), (experiment.cost, experiment.risk))
                .is_some()
        {
            return Err("reference scenario contains an empty or duplicate experiment identity".into());
        }
    }

    for scenario in scenarios.iter().skip(1) {
        let mut observed = BTreeMap::<String, (f64, f64)>::new();
        for experiment in &scenario.experiments {
            if experiment.id.trim().is_empty()
                || observed
                    .insert(experiment.id.clone(), (experiment.cost, experiment.risk))
                    .is_some()
            {
                return Err(format!(
                    "scenario {} contains an empty or duplicate experiment identity",
                    scenario.scenario_id
                ));
            }
        }
        if observed.keys().collect::<Vec<_>>() != reference.keys().collect::<Vec<_>>() {
            return Err(format!(
                "scenario {} does not contain the exact reference experiment set",
                scenario.scenario_id
            ));
        }
        for (id, (cost, risk)) in &reference {
            let (other_cost, other_risk) = observed[id];
            let cost_scale = 1.0_f64.max(cost.abs()).max(other_cost.abs());
            let risk_scale = 1.0_f64.max(risk.abs()).max(other_risk.abs());
            if (cost - other_cost).abs() > IDENTITY_TOLERANCE * cost_scale
                || (risk - other_risk).abs() > IDENTITY_TOLERANCE * risk_scale
            {
                return Err(format!(
                    "scenario {} changes declared cost/risk for experiment {id}",
                    scenario.scenario_id
                ));
            }
        }
    }

    // Validate every proposed experiment in every scenario before applying the
    // budget filter. Unaffordable candidates are still part of the submitted
    // proposal set and must not be able to conceal malformed likelihoods.
    for scenario in scenarios {
        for experiment in &scenario.experiments {
            evaluate_experiment(
                &scenario.prior,
                &scenario.action_utilities,
                experiment,
                risk_aversion,
            )
            .map_err(|error| {
                format!(
                    "scenario {} experiment {} is invalid: {error}",
                    scenario.scenario_id, experiment.id
                )
            })?;
        }
    }

    if reference.is_empty() {
        return Ok(None);
    }

    let mut robust_scores = Vec::new();
    for (experiment_id, (cost, risk)) in &reference {
        if *cost > available_budget {
            continue;
        }

        let mut evsi_values = Vec::with_capacity(scenarios.len());
        let mut net_values = Vec::with_capacity(scenarios.len());
        for scenario in scenarios {
            let experiment = scenario
                .experiments
                .iter()
                .find(|experiment| experiment.id == *experiment_id)
                .ok_or_else(|| format!("scenario {} is missing {experiment_id}", scenario.scenario_id))?;
            let value = evaluate_experiment(
                &scenario.prior,
                &scenario.action_utilities,
                experiment,
                risk_aversion,
            )?;
            evsi_values.push(value.expected_value_of_information);
            net_values.push(value.expected_value_of_information - cost - risk_aversion * risk);
        }

        let minimum_evsi = evsi_values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum_evsi = evsi_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let minimum_scenario_net_value = net_values.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum_scenario_net_value = net_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_scenario_net_value_descriptive_only =
            net_values.iter().sum::<f64>() / net_values.len() as f64;

        robust_scores.push(RobustExperimentValue {
            experiment_id: experiment_id.clone(),
            scenario_count: scenarios.len(),
            minimum_evsi,
            maximum_evsi,
            mean_scenario_net_value_descriptive_only,
            minimum_scenario_net_value,
            maximum_scenario_net_value,
        });
    }

    robust_scores.sort_by(|left, right| {
        right
            .minimum_scenario_net_value
            .partial_cmp(&left.minimum_scenario_net_value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });

    match robust_scores.into_iter().next() {
        Some(best) if best.minimum_scenario_net_value > minimum_worst_case_net_value => {
            Ok(Some(best))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_value::{ObservationLikelihood, Experiment};

    fn signal(id: &str, accuracy: f64, cost: f64) -> Experiment {
        Experiment {
            id: id.into(),
            cost,
            risk: 0.0,
            outcomes: vec![
                ObservationLikelihood {
                    outcome: "positive".into(),
                    likelihood_by_hypothesis: vec![accuracy, 1.0 - accuracy],
                },
                ObservationLikelihood {
                    outcome: "negative".into(),
                    likelihood_by_hypothesis: vec![1.0 - accuracy, accuracy],
                },
            ],
        }
    }

    fn scenario(id: &str, stable_accuracy: f64, fragile_accuracy: f64) -> DecisionModelScenario {
        DecisionModelScenario {
            scenario_id: id.into(),
            prior: vec![0.5, 0.5],
            action_utilities: vec![vec![10.0, 0.0], vec![0.0, 10.0]],
            experiments: vec![
                signal("robust_signal", stable_accuracy, 0.1),
                signal("fragile_signal", fragile_accuracy, 0.1),
            ],
        }
    }

    #[test]
    fn lower_bound_evsi_prefers_signal_that_is_useful_across_models() {
        let scenarios = vec![
            scenario("model-a", 0.8, 1.0),
            scenario("model-b", 0.7, 0.5),
        ];
        let chosen = select_robust_evsi(&scenarios, 1.0, 0.0, 0.0)
            .unwrap()
            .unwrap();
        assert_eq!(chosen.experiment_id, "robust_signal");
        assert_eq!(chosen.scenario_count, 2);
        assert!(chosen.minimum_evsi > 0.0);
        assert!(chosen.minimum_scenario_net_value > 0.0);
        assert!(chosen.maximum_scenario_net_value > chosen.minimum_scenario_net_value);
    }

    #[test]
    fn optimistic_single_model_value_does_not_hide_negative_worst_case_value() {
        let scenarios = vec![
            scenario("model-a", 1.0, 1.0),
            scenario("model-b", 0.6, 0.5),
        ];
        let chosen = select_robust_evsi(&scenarios, 1.0, 0.0, 0.0)
            .unwrap()
            .unwrap();
        assert_eq!(chosen.experiment_id, "robust_signal");
        assert!(chosen.minimum_scenario_net_value > 0.0);
    }

    #[test]
    fn no_experiment_with_positive_worst_case_value_returns_stop() {
        let scenarios = vec![
            scenario("model-a", 0.5, 0.5),
            scenario("model-b", 0.5, 0.5),
        ];
        assert!(select_robust_evsi(&scenarios, 1.0, 0.0, 0.0)
            .unwrap()
            .is_none());
    }

    #[test]
    fn malformed_unaffordable_experiment_fails_closed() {
        let mut first = scenario("model-a", 0.8, 1.0);
        let mut second = scenario("model-b", 0.7, 0.5);
        first.experiments[1].cost = 10.0;
        second.experiments[1].cost = 10.0;
        // This malformed candidate is outside the available budget but remains
        // part of the submitted scenario set and therefore must be rejected.
        first.experiments[1].outcomes[0].likelihood_by_hypothesis[0] = 0.25;
        assert!(select_robust_evsi(&[first, second], 1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn scenario_experiment_set_mismatch_fails_closed() {
        let mut second = scenario("model-b", 0.7, 0.5);
        second.experiments.pop();
        assert!(select_robust_evsi(
            &[scenario("model-a", 0.8, 1.0), second],
            1.0,
            0.0,
            0.0
        )
        .is_err());
    }

    #[test]
    fn scenario_cost_or_risk_mismatch_fails_closed() {
        let first = scenario("model-a", 0.8, 1.0);
        let mut second = scenario("model-b", 0.7, 0.5);
        second.experiments[0].cost = 0.2;
        assert!(select_robust_evsi(&[first, second], 1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn duplicate_scenario_identity_fails_closed() {
        assert!(select_robust_evsi(
            &[scenario("same", 0.8, 1.0), scenario("same", 0.7, 0.5)],
            1.0,
            0.0,
            0.0
        )
        .is_err());
    }

    #[test]
    fn one_scenario_is_not_mislabeled_as_robust() {
        assert!(select_robust_evsi(&[scenario("only", 0.8, 1.0)], 1.0, 0.0, 0.0).is_err());
    }
}
