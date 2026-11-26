//! Exact finite-horizon adaptive experiment oracle for DSSI.
//!
//! Build from repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_nonmyopic_oracle.rs -o /tmp/sdcb-nonmyopic
//!
//! This deliberately enumerative oracle is for small finite candidate sets. It
//! evaluates adaptive policies, not fixed experiment sequences: the next
//! experiment may depend on the observed outcome.

#[path = "sdcb_decision_value.rs"]
mod decision;

use decision::{
    best_action, evaluate_decision_experiment, DecisionError, DecisionExperimentCandidate,
    DecisionModel,
};

const EPS: f64 = 1e-12;
const MAX_EXACT_CANDIDATES: usize = 8;
const MAX_EXACT_HORIZON: usize = 4;
const MAX_EXACT_OUTCOMES: usize = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct NonMyopicChoice {
    pub first_experiment_id: String,
    pub stop_value: f64,
    pub expected_policy_value: f64,
    pub expected_improvement: f64,
    pub horizon: usize,
    pub budget_units: u64,
}

fn validate_candidates(
    posterior: &[f64],
    model: &DecisionModel,
    candidates: &[DecisionExperimentCandidate],
    cost_rate: f64,
) -> Result<(), DecisionError> {
    if !cost_rate.is_finite() || cost_rate < 0.0 {
        return Err(DecisionError::InvalidCostRate);
    }
    if candidates.len() > MAX_EXACT_CANDIDATES {
        return Err(DecisionError::InvalidExperiment);
    }
    for (index, candidate) in candidates.iter().enumerate() {
        if candidates[..index]
            .iter()
            .any(|earlier| earlier.experiment_id == candidate.experiment_id)
        {
            return Err(DecisionError::DuplicateExperimentId);
        }
        evaluate_decision_experiment(posterior, model, candidate)?;
        if candidate.outcome_likelihoods[0].len() > MAX_EXACT_OUTCOMES {
            return Err(DecisionError::InvalidExperiment);
        }
    }
    Ok(())
}

fn outcome_posterior(
    posterior: &[f64],
    candidate: &DecisionExperimentCandidate,
    outcome: usize,
) -> Option<(f64, Vec<f64>)> {
    let probability: f64 = posterior
        .iter()
        .enumerate()
        .map(|(hypothesis, p)| p * candidate.outcome_likelihoods[hypothesis][outcome])
        .sum();

    if probability <= 0.0 {
        return None;
    }

    let updated: Vec<f64> = posterior
        .iter()
        .enumerate()
        .map(|(hypothesis, p)| {
            p * candidate.outcome_likelihoods[hypothesis][outcome] / probability
        })
        .collect();

    Some((probability, updated))
}

fn solve(
    posterior: &[f64],
    model: &DecisionModel,
    candidates: &[DecisionExperimentCandidate],
    available: &[bool],
    budget_remaining: u64,
    steps_remaining: usize,
    cost_rate: f64,
) -> Result<f64, DecisionError> {
    let stop_value = best_action(posterior, model)?.expected_utility;
    if steps_remaining == 0 || !available.iter().any(|flag| *flag) {
        return Ok(stop_value);
    }

    let mut best_value = stop_value;

    for (index, candidate) in candidates.iter().enumerate() {
        if !available[index] || candidate.cost_units > budget_remaining {
            continue;
        }

        let cost = cost_rate * candidate.cost_units as f64;
        if !cost.is_finite() {
            return Err(DecisionError::NumericalInconsistency);
        }

        let mut expected_continuation = -cost;
        let outcome_count = candidate.outcome_likelihoods[0].len();

        for outcome in 0..outcome_count {
            if let Some((probability, updated)) =
                outcome_posterior(posterior, candidate, outcome)
            {
                let mut next_available = available.to_vec();
                next_available[index] = false;
                let continuation = solve(
                    &updated,
                    model,
                    candidates,
                    &next_available,
                    budget_remaining - candidate.cost_units,
                    steps_remaining - 1,
                    cost_rate,
                )?;
                expected_continuation += probability * continuation;
            }
        }

        if !expected_continuation.is_finite() {
            return Err(DecisionError::NumericalInconsistency);
        }
        best_value = best_value.max(expected_continuation);
    }

    Ok(best_value)
}

/// Exact finite-horizon value with adaptive experiment choice and optional stop.
/// Each experiment can be used at most once. Cost is expressed in the same
/// utility units as the declared decision utility, via cost_rate_per_unit.
pub fn finite_horizon_decision_value(
    posterior: &[f64],
    model: &DecisionModel,
    candidates: &[DecisionExperimentCandidate],
    budget_units: u64,
    max_experiments: usize,
    cost_rate_per_unit: f64,
) -> Result<f64, DecisionError> {
    validate_candidates(posterior, model, candidates, cost_rate_per_unit)?;
    if max_experiments > candidates.len() || max_experiments > MAX_EXACT_HORIZON {
        return Err(DecisionError::InvalidExperiment);
    }
    let available = vec![true; candidates.len()];
    solve(
        posterior,
        model,
        candidates,
        &available,
        budget_units,
        max_experiments,
        cost_rate_per_unit,
    )
}

/// Choose the first experiment of the exact optimal adaptive policy.
/// Returns None when stopping immediately is optimal.
pub fn choose_non_myopic_first_experiment(
    posterior: &[f64],
    model: &DecisionModel,
    candidates: &[DecisionExperimentCandidate],
    budget_units: u64,
    max_experiments: usize,
    cost_rate_per_unit: f64,
) -> Result<Option<NonMyopicChoice>, DecisionError> {
    validate_candidates(posterior, model, candidates, cost_rate_per_unit)?;
    if max_experiments > candidates.len() {
        return Err(DecisionError::InvalidExperiment);
    }

    let stop_value = best_action(posterior, model)?.expected_utility;
    if max_experiments == 0 {
        return Ok(None);
    }

    let mut best: Option<NonMyopicChoice> = None;

    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.cost_units > budget_units {
            continue;
        }
        let cost = cost_rate_per_unit * candidate.cost_units as f64;
        if !cost.is_finite() {
            return Err(DecisionError::NumericalInconsistency);
        }

        let mut available = vec![true; candidates.len()];
        available[index] = false;
        let mut expected_policy_value = -cost;
        let outcome_count = candidate.outcome_likelihoods[0].len();

        for outcome in 0..outcome_count {
            if let Some((probability, updated)) =
                outcome_posterior(posterior, candidate, outcome)
            {
                let continuation = solve(
                    &updated,
                    model,
                    candidates,
                    &available,
                    budget_units - candidate.cost_units,
                    max_experiments - 1,
                    cost_rate_per_unit,
                )?;
                expected_policy_value += probability * continuation;
            }
        }

        let improvement = expected_policy_value - stop_value;
        if !improvement.is_finite() {
            return Err(DecisionError::NumericalInconsistency);
        }
        if improvement <= EPS {
            continue;
        }

        let choice = NonMyopicChoice {
            first_experiment_id: candidate.experiment_id.clone(),
            stop_value,
            expected_policy_value,
            expected_improvement: improvement,
            horizon: max_experiments,
            budget_units,
        };

        let replace = match &best {
            None => true,
            Some(current)
                if choice.expected_policy_value > current.expected_policy_value + EPS =>
            {
                true
            }
            Some(current)
                if (choice.expected_policy_value - current.expected_policy_value).abs() <= EPS =>
            {
                choice.first_experiment_id < current.first_experiment_id
            }
            _ => false,
        };
        if replace {
            best = Some(choice);
        }
    }

    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> DecisionModel {
        DecisionModel {
            action_ids: vec!["act-a".into(), "act-b".into()],
            utility_by_hypothesis: vec![
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ],
        }
    }

    fn screen_a() -> DecisionExperimentCandidate {
        DecisionExperimentCandidate {
            experiment_id: "screen-a".into(),
            cost_units: 1,
            outcome_likelihoods: vec![
                vec![0.75, 0.25],
                vec![1.0, 0.0],
                vec![0.5, 0.5],
            ],
        }
    }

    fn screen_b() -> DecisionExperimentCandidate {
        DecisionExperimentCandidate {
            experiment_id: "screen-b".into(),
            cost_units: 1,
            outcome_likelihoods: vec![
                vec![0.5, 0.5],
                vec![0.5, 0.5],
                vec![1.0, 0.0],
            ],
        }
    }

    #[test]
    fn both_probes_have_zero_myopic_evsi() {
        let posterior = [0.4, 0.4, 0.2];
        let model = model();
        let a = evaluate_decision_experiment(&posterior, &model, &screen_a()).unwrap();
        let b = evaluate_decision_experiment(&posterior, &model, &screen_b()).unwrap();
        assert!(a.gross_evsi.abs() < EPS);
        assert!(b.gross_evsi.abs() < EPS);
    }

    #[test]
    fn exact_two_step_policy_finds_value_hidden_from_one_step_evsi() {
        let posterior = [0.4, 0.4, 0.2];
        let value = finite_horizon_decision_value(
            &posterior,
            &model(),
            &[screen_a(), screen_b()],
            2,
            2,
            0.0,
        )
        .unwrap();
        assert!((value - 0.85).abs() < EPS);
    }

    #[test]
    fn one_step_policy_abstains_but_two_step_policy_experiments() {
        let posterior = [0.4, 0.4, 0.2];
        let model = model();
        let candidates = vec![screen_a(), screen_b()];
        assert_eq!(
            decision::choose_decision_experiment(&posterior, &model, &candidates, 0.0).unwrap(),
            None
        );

        let choice = choose_non_myopic_first_experiment(
            &posterior,
            &model,
            &candidates,
            2,
            2,
            0.0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(choice.first_experiment_id, "screen-a");
        assert!((choice.stop_value - 0.8).abs() < EPS);
        assert!((choice.expected_policy_value - 0.85).abs() < EPS);
        assert!((choice.expected_improvement - 0.05).abs() < EPS);
    }

    #[test]
    fn budget_of_one_prevents_two_experiment_adaptive_gain() {
        let posterior = [0.4, 0.4, 0.2];
        let value = finite_horizon_decision_value(
            &posterior,
            &model(),
            &[screen_a(), screen_b()],
            1,
            2,
            0.0,
        )
        .unwrap();
        assert!((value - 0.8).abs() < EPS);
    }

    #[test]
    fn exact_oracle_charges_cost_at_each_experiment_node() {
        let posterior = [0.4, 0.4, 0.2];
        let value = finite_horizon_decision_value(
            &posterior,
            &model(),
            &[screen_a(), screen_b()],
            2,
            2,
            0.03,
        )
        .unwrap();
        assert!((value - 0.814).abs() < EPS);
    }

    #[test]
    fn horizon_zero_is_the_current_decision_value() {
        let value = finite_horizon_decision_value(
            &[0.4, 0.4, 0.2],
            &model(),
            &[screen_a(), screen_b()],
            2,
            0,
            0.0,
        )
        .unwrap();
        assert!((value - 0.8).abs() < EPS);
    }

    #[test]
    fn duplicate_experiment_ids_are_rejected() {
        let mut duplicate = screen_b();
        duplicate.experiment_id = "screen-a".into();
        assert_eq!(
            finite_horizon_decision_value(
                &[0.4, 0.4, 0.2],
                &model(),
                &[screen_a(), duplicate],
                2,
                2,
                0.0,
            ),
            Err(DecisionError::DuplicateExperimentId)
        );
    }

    #[test]
    fn exact_oracle_rejects_oversized_candidate_sets() {
        let candidates = (0..=MAX_EXACT_CANDIDATES)
            .map(|index| {
                let mut experiment = screen_a();
                experiment.experiment_id = format!("probe-{index}");
                experiment
            })
            .collect::<Vec<_>>();
        assert_eq!(
            finite_horizon_decision_value(
                &[0.4, 0.4, 0.2],
                &model(),
                &candidates,
                2,
                2,
                0.0,
            ),
            Err(DecisionError::InvalidExperiment)
        );
    }

    #[test]
    fn exact_oracle_rejects_horizon_above_reference_bound() {
        let candidates = (0..MAX_EXACT_HORIZON + 1)
            .map(|index| { let mut e = screen_a(); e.experiment_id = format!("probe-{index}"); e })
            .collect::<Vec<_>>();
        assert_eq!(
            finite_horizon_decision_value(&[0.4, 0.4, 0.2], &model(), &candidates, 8, MAX_EXACT_HORIZON + 1, 0.0),
            Err(DecisionError::InvalidExperiment)
        );
    }

    #[test]
    fn exact_oracle_rejects_outcome_branching_above_reference_bound() {
        let mut e = screen_a();
        e.outcome_likelihoods = vec![vec![0.2; 5], vec![0.2; 5], vec![0.2; 5]];
        assert_eq!(
            finite_horizon_decision_value(&[0.4, 0.4, 0.2], &model(), &[e], 1, 1, 0.0),
            Err(DecisionError::InvalidExperiment)
        );
    }
}
