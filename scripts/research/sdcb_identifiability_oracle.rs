//! Exact finite observational-equivalence and intervention-separation oracle.
//!
//! Build:
//!   rustc --edition=2024 --test scripts/research/sdcb_identifiability_oracle.rs -o /tmp/sdcb-identifiability-oracle
//! Run:
//!   /tmp/sdcb-identifiability-oracle
//!
//! This bounded oracle compares two fully specified causal/measurement worlds.
//! It does not infer the worlds from data, authenticate interventions, or prove
//! that a physical manipulation satisfies its declared likelihood model.

const PROB_TOL: f64 = 1e-9;
const IDENTIFIABILITY_TOL: f64 = 1e-12;
const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    Observation,
    Intervention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateAction {
    pub id: String,
    pub kind: ActionKind,
    pub cost_units: u64,
    /// Categorical outcome probabilities in world A.
    pub likelihood_world_a: Vec<f64>,
    /// Categorical outcome probabilities in world B.
    pub likelihood_world_b: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedWorldModel {
    pub world_a_id: String,
    pub world_b_id: String,
    /// Prior probability of world A. World B has probability 1 - prior_world_a.
    pub prior_world_a: f64,
    pub abstain_utility: f64,
    pub budget_units: u64,
    pub actions: Vec<CandidateAction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidWorldIdentity,
    IdenticalWorlds,
    InvalidPrior,
    InvalidAbstainUtility,
    NoActions,
    InvalidActionIdentity,
    DuplicateAction,
    ZeroCost,
    InvalidDistribution,
    OutcomeSpaceMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionScore {
    pub action_id: String,
    pub kind: ActionKind,
    pub cost_units: u64,
    /// Total-variation distance between the two worlds' outcome distributions.
    /// 0 means this action cannot distinguish this paired world.
    pub total_variation: f64,
    /// Bayes-optimal probability of identifying the correct world after this action,
    /// using only a forced A/B decision (no abstention).
    pub bayes_accuracy: f64,
    /// Bayes-optimal expected utility after the action, including abstention.
    pub expected_utility: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IdentifiabilityReport {
    pub world_a_id: String,
    pub world_b_id: String,
    pub observationally_equivalent: bool,
    pub max_observational_total_variation: f64,
    pub max_intervention_total_variation: f64,
    /// True when at least one declared intervention has non-zero separating power.
    /// This does not mean the intervention is affordable or physically valid.
    pub intervention_breaks_equivalence: bool,
    pub baseline_utility: f64,
    /// Highest expected utility among budget-feasible actions, including baseline.
    pub best_feasible_utility: f64,
    /// Present only when a feasible action strictly improves on the baseline.
    pub selected_action_id: Option<String>,
    pub scores: Vec<ActionScore>,
}

fn validate_distribution(values: &[f64]) -> Result<(), Error> {
    if values.len() < 2
        || values.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (values.iter().sum::<f64>() - 1.0).abs() > PROB_TOL
    {
        return Err(Error::InvalidDistribution);
    }
    Ok(())
}

fn total_variation(a: &[f64], b: &[f64]) -> f64 {
    0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
}

fn bayes_accuracy(prior_a: f64, a: &[f64], b: &[f64]) -> f64 {
    let prior_b = 1.0 - prior_a;
    a.iter()
        .zip(b)
        .map(|(pa, pb)| (prior_a * pa).max(prior_b * pb))
        .sum()
}

fn expected_utility(prior_a: f64, abstain_utility: f64, a: &[f64], b: &[f64]) -> f64 {
    let prior_b = 1.0 - prior_a;
    a.iter()
        .zip(b)
        .map(|(pa, pb)| {
            let mass_a = prior_a * pa;
            let mass_b = prior_b * pb;
            mass_a
                .max(mass_b)
                .max(abstain_utility * (mass_a + mass_b))
        })
        .sum()
}

fn validate(model: &PairedWorldModel) -> Result<(), Error> {
    if model.world_a_id.trim().is_empty() || model.world_b_id.trim().is_empty() {
        return Err(Error::InvalidWorldIdentity);
    }
    if model.world_a_id == model.world_b_id {
        return Err(Error::IdenticalWorlds);
    }
    if !model.prior_world_a.is_finite()
        || model.prior_world_a < 0.0
        || model.prior_world_a > 1.0
    {
        return Err(Error::InvalidPrior);
    }
    if !model.abstain_utility.is_finite()
        || model.abstain_utility < 0.0
        || model.abstain_utility > 1.0
    {
        return Err(Error::InvalidAbstainUtility);
    }
    if model.actions.is_empty() {
        return Err(Error::NoActions);
    }

    let mut has_observation = false;
    let mut action_ids: Vec<&str> = Vec::with_capacity(model.actions.len());
    let mut outcome_count: Option<usize> = None;
    for action in &model.actions {
        if action.id.trim().is_empty() {
            return Err(Error::InvalidActionIdentity);
        }
        if action_ids.contains(&action.id.as_str()) {
            return Err(Error::DuplicateAction);
        }
        action_ids.push(&action.id);
        if action.cost_units == 0 {
            return Err(Error::ZeroCost);
        }
        if action.kind == ActionKind::Observation {
            has_observation = true;
        }
        validate_distribution(&action.likelihood_world_a)?;
        validate_distribution(&action.likelihood_world_b)?;
        if action.likelihood_world_a.len() != action.likelihood_world_b.len() {
            return Err(Error::OutcomeSpaceMismatch);
        }
        match outcome_count {
            None => outcome_count = Some(action.likelihood_world_a.len()),
            Some(n) if n != action.likelihood_world_a.len() => {
                return Err(Error::OutcomeSpaceMismatch);
            }
            Some(_) => {}
        }
    }
    if !has_observation {
        return Err(Error::NoActions);
    }
    Ok(())
}

/// Evaluate exact one-action discrimination and decision value for a paired
/// set of causal/measurement worlds.
///
/// An observational-equivalence result is relative to the supplied candidate
/// observation set and tolerance. It is not a universal claim about all
/// possible observations. Interventions are scored separately.
pub fn evaluate(model: &PairedWorldModel) -> Result<IdentifiabilityReport, Error> {
    validate(model)?;

    let mut scores = Vec::with_capacity(model.actions.len());
    let mut max_observation_tv: f64 = 0.0;
    let mut max_intervention_tv: f64 = 0.0;

    for action in &model.actions {
        let tv = total_variation(
            &action.likelihood_world_a,
            &action.likelihood_world_b,
        );
        match action.kind {
            ActionKind::Observation => max_observation_tv = max_observation_tv.max(tv),
            ActionKind::Intervention => max_intervention_tv = max_intervention_tv.max(tv),
        }
        scores.push(ActionScore {
            action_id: action.id.clone(),
            kind: action.kind,
            cost_units: action.cost_units,
            total_variation: tv,
            bayes_accuracy: bayes_accuracy(
                model.prior_world_a,
                &action.likelihood_world_a,
                &action.likelihood_world_b,
            ),
            expected_utility: expected_utility(
                model.prior_world_a,
                model.abstain_utility,
                &action.likelihood_world_a,
                &action.likelihood_world_b,
            ),
        });
    }

    let baseline_utility = model
        .prior_world_a
        .max(1.0 - model.prior_world_a)
        .max(model.abstain_utility);

    let mut best_feasible_utility = baseline_utility;
    let mut selected: Option<&ActionScore> = None;
    for score in &scores {
        if score.cost_units > model.budget_units
            || score.expected_utility <= baseline_utility + EPS
        {
            continue;
        }
        let is_better = match selected {
            None => true,
            Some(current) => {
                score.expected_utility > current.expected_utility + EPS
                    || ((score.expected_utility - current.expected_utility).abs() <= EPS
                        && (score.cost_units < current.cost_units
                            || (score.cost_units == current.cost_units
                                && score.action_id < current.action_id)))
            }
        };
        if is_better {
            selected = Some(score);
        }
    }
    if let Some(score) = selected {
        best_feasible_utility = score.expected_utility;
    }

    Ok(IdentifiabilityReport {
        world_a_id: model.world_a_id.clone(),
        world_b_id: model.world_b_id.clone(),
        observationally_equivalent: max_observation_tv <= IDENTIFIABILITY_TOL,
        max_observational_total_variation: max_observation_tv,
        max_intervention_total_variation: max_intervention_tv,
        intervention_breaks_equivalence: max_intervention_tv > IDENTIFIABILITY_TOL,
        baseline_utility,
        best_feasible_utility,
        selected_action_id: selected.map(|score| score.action_id.clone()),
        scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equivalent_world_model(budget_units: u64) -> PairedWorldModel {
        PairedWorldModel {
            world_a_id: "science-present-common-offset-zero".into(),
            world_b_id: "science-absent-common-offset-one".into(),
            prior_world_a: 0.5,
            abstain_utility: 0.6,
            budget_units,
            actions: vec![
                CandidateAction {
                    id: "science-device-a".into(),
                    kind: ActionKind::Observation,
                    cost_units: 1,
                    likelihood_world_a: vec![0.95, 0.05],
                    likelihood_world_b: vec![0.95, 0.05],
                },
                CandidateAction {
                    id: "science-device-b".into(),
                    kind: ActionKind::Observation,
                    cost_units: 1,
                    likelihood_world_a: vec![0.80, 0.20],
                    likelihood_world_b: vec![0.80, 0.20],
                },
            ],
        }
    }

    #[test]
    fn different_device_ids_do_not_break_observational_equivalence() {
        let report = evaluate(&equivalent_world_model(2)).unwrap();
        assert!(report.observationally_equivalent);
        assert_eq!(report.max_observational_total_variation, 0.0);
        assert!(!report.intervention_breaks_equivalence);
        assert_eq!(report.baseline_utility, 0.6);
        assert_eq!(report.best_feasible_utility, 0.6);
        assert_eq!(report.selected_action_id, None);
        assert!(report.scores.iter().all(|s| (s.bayes_accuracy - 0.5).abs() < EPS));
        assert!(report.scores.iter().all(|s| (s.expected_utility - 0.6).abs() < EPS));
    }

    #[test]
    fn targeted_intervention_separates_the_paired_worlds() {
        let mut model = equivalent_world_model(2);
        model.actions.push(CandidateAction {
            id: "isolate-common-cause-then-repeat-science".into(),
            kind: ActionKind::Intervention,
            cost_units: 2,
            likelihood_world_a: vec![0.95, 0.05],
            likelihood_world_b: vec![0.05, 0.95],
        });
        let report = evaluate(&model).unwrap();
        assert!(report.observationally_equivalent);
        assert!(report.intervention_breaks_equivalence);
        assert!((report.max_intervention_total_variation - 0.9).abs() < EPS);
        assert_eq!(report.selected_action_id.as_deref(), Some("isolate-common-cause-then-repeat-science"));
        assert!((report.best_feasible_utility - 0.95).abs() < EPS);
        assert!((report.best_feasible_utility - report.baseline_utility - 0.35).abs() < EPS);
    }

    #[test]
    fn structurally_informative_but_unaffordable_intervention_is_not_selected() {
        let mut model = equivalent_world_model(1);
        model.actions.push(CandidateAction {
            id: "independent-reference-intervention".into(),
            kind: ActionKind::Intervention,
            cost_units: 2,
            likelihood_world_a: vec![0.95, 0.05],
            likelihood_world_b: vec![0.05, 0.95],
        });
        let report = evaluate(&model).unwrap();
        assert!(report.intervention_breaks_equivalence);
        assert_eq!(report.selected_action_id, None);
        assert_eq!(report.best_feasible_utility, report.baseline_utility);
    }

    #[test]
    fn prior_imbalance_is_respected_by_the_no_experiment_baseline() {
        let mut model = equivalent_world_model(2);
        model.prior_world_a = 0.9;
        let report = evaluate(&model).unwrap();
        assert_eq!(report.baseline_utility, 0.9);
        assert_eq!(report.selected_action_id, None);
    }

    #[test]
    fn invalid_probability_rows_reject() {
        let mut model = equivalent_world_model(2);
        model.actions[0].likelihood_world_a = vec![0.9, 0.2];
        assert_eq!(evaluate(&model), Err(Error::InvalidDistribution));
    }

    #[test]
    fn mismatched_outcome_spaces_reject() {
        let mut model = equivalent_world_model(2);
        model.actions[0].likelihood_world_b = vec![1.0, 0.0, 0.0];
        assert_eq!(evaluate(&model), Err(Error::OutcomeSpaceMismatch));
    }

    #[test]
    fn duplicate_action_ids_reject() {
        let mut model = equivalent_world_model(2);
        model.actions[1].id = model.actions[0].id.clone();
        assert_eq!(evaluate(&model), Err(Error::DuplicateAction));
    }

    #[test]
    fn zero_cost_actions_reject() {
        let mut model = equivalent_world_model(2);
        model.actions[0].cost_units = 0;
        assert_eq!(evaluate(&model), Err(Error::ZeroCost));
    }
}
