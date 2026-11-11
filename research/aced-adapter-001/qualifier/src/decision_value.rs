use std::cmp::Ordering;

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const NUMERIC_TOLERANCE: f64 = 1e-10;

#[derive(Clone, Debug, PartialEq)]
pub struct ObservationLikelihood {
    pub outcome: String,
    /// P(this observation | each hypothesis), in the same order as the prior.
    pub likelihood_by_hypothesis: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Experiment {
    pub id: String,
    /// Cost in the same utility units as the downstream action-value table.
    pub cost: f64,
    /// Declared non-negative risk exposure in a benchmark-defined unit.
    pub risk: f64,
    pub outcomes: Vec<ObservationLikelihood>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMethod {
    ExpectedInformationGain,
    ExpectedValueOfSampleInformation,
    RiskAdjustedExpectedValueOfSampleInformation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreUnits {
    Bits,
    Utility,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentValue {
    pub experiment_id: String,
    pub information_gain_bits: f64,
    pub expected_value_of_information: f64,
    pub cost: f64,
    pub risk_penalty: f64,
    /// Comparable only within a single SelectionMethod because EIG is measured in bits and EVSI in utility.
    pub selection_score: f64,
    pub selection_score_units: ScoreUnits,
}

fn validate_inputs(
    prior: &[f64],
    action_utilities: &[Vec<f64>],
    experiments: &[Experiment],
    risk_aversion: f64,
) -> Result<(), String> {
    if prior.is_empty() || action_utilities.is_empty() {
        return Err("prior and action table must be non-empty".into());
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
    if !risk_aversion.is_finite() || risk_aversion < 0.0 {
        return Err("risk aversion must be finite and non-negative".into());
    }

    let mut experiment_ids = std::collections::BTreeSet::new();
    for experiment in experiments {
        if experiment.id.trim().is_empty()
            || !experiment.cost.is_finite()
            || experiment.cost < 0.0
            || !experiment.risk.is_finite()
            || experiment.risk < 0.0
            || experiment.outcomes.is_empty()
            || !experiment_ids.insert(experiment.id.as_str())
        {
            return Err(format!(
                "experiment {} has invalid or duplicate identity, cost, risk or outcomes",
                experiment.id
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        let mut likelihood_sums = vec![0.0; prior.len()];
        for outcome in &experiment.outcomes {
            if outcome.outcome.trim().is_empty()
                || !names.insert(outcome.outcome.as_str())
                || outcome.likelihood_by_hypothesis.len() != prior.len()
                || outcome.likelihood_by_hypothesis.iter().any(|value| {
                    !value.is_finite() || *value < 0.0 || *value > 1.0
                })
            {
                return Err(format!(
                    "experiment {} has malformed or duplicate outcomes",
                    experiment.id
                ));
            }
            for (sum, likelihood) in likelihood_sums
                .iter_mut()
                .zip(&outcome.likelihood_by_hypothesis)
            {
                *sum += likelihood;
            }
        }
        if likelihood_sums
            .iter()
            .any(|sum| (*sum - 1.0).abs() > PROBABILITY_TOLERANCE)
        {
            return Err(format!(
                "experiment {} likelihoods must sum to one for every hypothesis",
                experiment.id
            ));
        }
    }
    Ok(())
}

fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.log2())
        .sum()
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
    if probability <= NUMERIC_TOLERANCE {
        return Ok((0.0, vec![0.0; prior.len()]));
    }
    let distribution = prior
        .iter()
        .zip(likelihoods)
        .map(|(prior, likelihood)| prior * likelihood / probability)
        .collect();
    Ok((probability, distribution))
}

fn best_expected_utility(prior: &[f64], action_utilities: &[Vec<f64>]) -> f64 {
    action_utilities
        .iter()
        .map(|row| prior.iter().zip(row).map(|(p, utility)| p * utility).sum())
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Evaluate one-step sample information for a fixed finite hypothesis and action set.
/// This does not implement a multi-stage policy, causal intervention optimizer or general POMDP solver.
pub fn evaluate_experiment(
    prior: &[f64],
    action_utilities: &[Vec<f64>],
    experiment: &Experiment,
    risk_aversion: f64,
) -> Result<ExperimentValue, String> {
    validate_inputs(prior, action_utilities, std::slice::from_ref(experiment), risk_aversion)?;
    let current_entropy = entropy_bits(prior);
    let current_best = best_expected_utility(prior, action_utilities);
    if !current_entropy.is_finite() || !current_best.is_finite() {
        return Err("current belief/action utility arithmetic is non-finite".into());
    }
    let mut expected_posterior_entropy = 0.0;
    let mut expected_post_observation_utility = 0.0;

    for outcome in &experiment.outcomes {
        let (outcome_probability, updated_prior) =
            posterior(prior, &outcome.likelihood_by_hypothesis)?;
        if outcome_probability <= NUMERIC_TOLERANCE {
            continue;
        }
        expected_posterior_entropy += outcome_probability * entropy_bits(&updated_prior);
        expected_post_observation_utility +=
            outcome_probability * best_expected_utility(&updated_prior, action_utilities);
    }

    let information_gain_bits = (current_entropy - expected_posterior_entropy).max(0.0);
    let expected_value_of_information =
        (expected_post_observation_utility - current_best).max(0.0);
    let risk_penalty = risk_aversion * experiment.risk;
    let net_value = expected_value_of_information - experiment.cost - risk_penalty;
    if !expected_posterior_entropy.is_finite()
        || !expected_post_observation_utility.is_finite()
        || !information_gain_bits.is_finite()
        || !expected_value_of_information.is_finite()
        || !risk_penalty.is_finite()
        || !net_value.is_finite()
    {
        return Err(format!(
            "experiment {} produced non-finite value arithmetic",
            experiment.id
        ));
    }

    Ok(ExperimentValue {
        experiment_id: experiment.id.clone(),
        information_gain_bits,
        expected_value_of_information,
        cost: experiment.cost,
        risk_penalty,
        selection_score: expected_value_of_information,
        selection_score_units: ScoreUnits::Utility,
    })
}

pub fn select_experiment(
    prior: &[f64],
    action_utilities: &[Vec<f64>],
    experiments: &[Experiment],
    method: SelectionMethod,
    risk_aversion: f64,
    minimum_selection_score: f64,
) -> Result<Option<ExperimentValue>, String> {
    if !minimum_selection_score.is_finite() {
        return Err("minimum net value must be finite".into());
    }
    validate_inputs(prior, action_utilities, experiments, risk_aversion)?;
    if experiments.is_empty() {
        return Ok(None);
    }

    let mut scored = experiments
        .iter()
        .map(|experiment| evaluate_experiment(prior, action_utilities, experiment, risk_aversion))
        .collect::<Result<Vec<_>, _>>()?;

    for score in &mut scored {
        match method {
            SelectionMethod::ExpectedInformationGain => {
                score.selection_score = score.information_gain_bits;
                score.selection_score_units = ScoreUnits::Bits;
            }
            SelectionMethod::ExpectedValueOfSampleInformation => {
                score.selection_score = score.expected_value_of_information - score.cost;
                score.selection_score_units = ScoreUnits::Utility;
            }
            SelectionMethod::RiskAdjustedExpectedValueOfSampleInformation => {
                score.selection_score =
                    score.expected_value_of_information - score.cost - score.risk_penalty;
                score.selection_score_units = ScoreUnits::Utility;
            }
        }
    }

    scored.sort_by(|left, right| {
        right
            .selection_score
            .partial_cmp(&left.selection_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });

    match scored.into_iter().next() {
        Some(best) if best.selection_score > minimum_selection_score => Ok(Some(best)),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn experiment(id: &str, cost: f64, risk: f64, rows: &[(&str, &[f64])]) -> Experiment {
        Experiment {
            id: id.into(),
            cost,
            risk,
            outcomes: rows
                .iter()
                .map(|(outcome, likelihood)| ObservationLikelihood {
                    outcome: (*outcome).into(),
                    likelihood_by_hypothesis: likelihood.to_vec(),
                })
                .collect(),
        }
    }

    fn problem() -> (Vec<f64>, Vec<Vec<f64>>, Vec<Experiment>) {
        let prior = vec![0.45, 0.45, 0.10];
        let actions = vec![vec![10.0, 10.0, 0.0], vec![0.0, 0.0, 12.0]];
        let high_entropy_low_decision_value = experiment(
            "split_redundant_hypotheses",
            0.01,
            0.0,
            &[("x1", &[1.0, 0.0, 0.5]), ("x2", &[0.0, 1.0, 0.5])],
        );
        let decision_relevant_signal = experiment(
            "signal_that_can_reverse_the_action",
            0.01,
            0.0,
            &[
                ("positive", &[0.1, 0.1, 0.9]),
                ("negative", &[0.9, 0.9, 0.1]),
            ],
        );
        (
            prior,
            actions,
            vec![high_entropy_low_decision_value, decision_relevant_signal],
        )
    }

    #[test]
    fn entropy_reduction_can_disagree_with_decision_value() {
        let (prior, actions, experiments) = problem();
        let eig = select_experiment(
            &prior,
            &actions,
            &experiments,
            SelectionMethod::ExpectedInformationGain,
            0.0,
            0.0,
        )
        .unwrap()
        .unwrap();
        let evsi = select_experiment(
            &prior,
            &actions,
            &experiments,
            SelectionMethod::ExpectedValueOfSampleInformation,
            0.0,
            0.0,
        )
        .unwrap()
        .unwrap();

        assert_eq!(eig.experiment_id, "split_redundant_hypotheses");
        assert_eq!(evsi.experiment_id, "signal_that_can_reverse_the_action");
        assert!(eig.information_gain_bits > evsi.information_gain_bits);
        assert!(evsi.expected_value_of_information > 0.0);
        assert!(eig.expected_value_of_information < NUMERIC_TOLERANCE);
        assert_eq!(eig.selection_score_units, ScoreUnits::Bits);
        assert_eq!(evsi.selection_score_units, ScoreUnits::Utility);
        assert!((eig.selection_score - eig.information_gain_bits).abs() < NUMERIC_TOLERANCE);
        assert!(
            (evsi.selection_score - (evsi.expected_value_of_information - evsi.cost)).abs()
                < NUMERIC_TOLERANCE
        );
    }

    #[test]
    fn risk_adjusted_evsi_avoids_high_value_but_high_risk_probe() {
        let (prior, actions, mut experiments) = problem();
        experiments.push(experiment(
            "unsafe_perfect_information",
            0.01,
            0.8,
            &[
                ("h1", &[1.0, 0.0, 0.0]),
                ("h2", &[0.0, 1.0, 0.0]),
                ("h3", &[0.0, 0.0, 1.0]),
            ],
        ));

        let ordinary = select_experiment(
            &prior,
            &actions,
            &experiments,
            SelectionMethod::ExpectedValueOfSampleInformation,
            2.0,
            0.0,
        )
        .unwrap()
        .unwrap();
        let risk_adjusted = select_experiment(
            &prior,
            &actions,
            &experiments,
            SelectionMethod::RiskAdjustedExpectedValueOfSampleInformation,
            2.0,
            0.0,
        )
        .unwrap()
        .unwrap();

        assert_eq!(ordinary.experiment_id, "unsafe_perfect_information");
        assert_eq!(risk_adjusted.experiment_id, "signal_that_can_reverse_the_action");
        assert!(risk_adjusted.risk_penalty == 0.0);
    }

    #[test]
    fn stop_is_selected_when_all_experiments_have_negative_net_value() {
        let (prior, actions, mut experiments) = problem();
        for experiment in &mut experiments {
            experiment.cost = 100.0;
        }
        assert!(select_experiment(
            &prior,
            &actions,
            &experiments,
            SelectionMethod::ExpectedValueOfSampleInformation,
            0.0,
            0.0,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn empty_experiment_set_stops_cleanly() {
        let prior = vec![0.5, 0.5];
        let actions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(select_experiment(
            &prior,
            &actions,
            &[],
            SelectionMethod::ExpectedValueOfSampleInformation,
            0.0,
            0.0,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn duplicate_experiment_identity_fails_closed() {
        let prior = vec![0.5, 0.5];
        let actions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let first = experiment(
            "same-id",
            0.0,
            0.0,
            &[("yes", &[1.0, 0.0]), ("no", &[0.0, 1.0])],
        );
        let second = first.clone();
        assert!(select_experiment(
            &prior,
            &actions,
            &[first, second],
            SelectionMethod::ExpectedValueOfSampleInformation,
            0.0,
            0.0,
        )
        .is_err());
    }

    #[test]
    fn malformed_probability_contract_fails_closed() {
        let (prior, actions, mut experiments) = problem();
        experiments[0].outcomes[0].likelihood_by_hypothesis[0] = 0.25;
        assert!(select_experiment(
            &prior,
            &actions,
            &experiments,
            SelectionMethod::ExpectedValueOfSampleInformation,
            0.0,
            0.0,
        )
        .is_err());
    }

    #[test]
    fn deterministic_tie_break_uses_experiment_identity() {
        let prior = vec![0.5, 0.5];
        let actions = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let one = experiment(
            "probe-b",
            0.0,
            0.0,
            &[("yes", &[1.0, 0.0]), ("no", &[0.0, 1.0])],
        );
        let two = experiment(
            "probe-a",
            0.0,
            0.0,
            &[("yes", &[1.0, 0.0]), ("no", &[0.0, 1.0])],
        );
        let chosen = select_experiment(
            &prior,
            &actions,
            &[one, two],
            SelectionMethod::ExpectedValueOfSampleInformation,
            0.0,
            0.0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(chosen.experiment_id, "probe-a");
    }
}
