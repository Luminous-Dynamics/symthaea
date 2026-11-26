//! Decision-sufficient experimental design reference controls for SDCB.
//! Build: rustc --edition=2024 --test scripts/research/sdcb_decision_value.rs -o /tmp/sdcb-decision-value
//! This finite reference distinguishes expected information gain from EVSI.
//! It is not a general Bayesian optimizer or non-myopic sequential planner.

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const EPS: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionError {
    InvalidDistribution, InvalidActionSet, InvalidUtilityMatrix,
    InvalidLikelihoodMatrix, InvalidExperiment, DuplicateExperimentId,
    InvalidCostRate, NumericalInconsistency,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionModel {
    pub action_ids: Vec<String>,
    /// Row h contains utility for each action in action_ids.
    pub utility_by_hypothesis: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionExperimentCandidate {
    pub experiment_id: String,
    pub cost_units: u64,
    /// Row h contains P(outcome | hypothesis h, experiment).
    pub outcome_likelihoods: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BestAction {
    pub action_id: String,
    pub action_index: usize,
    pub expected_utility: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionClass {
    pub hypothesis_indices: Vec<usize>,
    pub utility_by_action: Vec<f64>,
    pub posterior_mass: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionExperimentEvaluation {
    pub experiment_id: String,
    pub current_action_id: String,
    pub current_decision_value: f64,
    pub expected_post_experiment_decision_value: f64,
    pub gross_evsi: f64,
    pub expected_information_gain_nats: f64,
    pub cost_units: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionExperimentChoice {
    pub evaluation: DecisionExperimentEvaluation,
    pub cost_rate_per_unit: f64,
    pub net_evsi: f64,
}

fn validate_distribution(p: &[f64]) -> Result<(), DecisionError> {
    if p.len() < 2 || p.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0) {
        return Err(DecisionError::InvalidDistribution);
    }
    let sum: f64 = p.iter().sum();
    if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(DecisionError::InvalidDistribution);
    }
    Ok(())
}

fn validate_model(p: &[f64], model: &DecisionModel) -> Result<(), DecisionError> {
    validate_distribution(p)?;
    if model.action_ids.is_empty()
        || model.utility_by_hypothesis.len() != p.len()
        || model.action_ids.iter().any(|id| id.trim().is_empty())
    {
        return Err(DecisionError::InvalidActionSet);
    }
    for (i, id) in model.action_ids.iter().enumerate() {
        if model.action_ids[..i].contains(id) {
            return Err(DecisionError::InvalidActionSet);
        }
    }
    let n = model.action_ids.len();
    for row in &model.utility_by_hypothesis {
        if row.len() != n || row.iter().any(|u| !u.is_finite()) {
            return Err(DecisionError::InvalidUtilityMatrix);
        }
    }
    Ok(())
}

fn validate_likelihoods(p: &[f64], l: &[Vec<f64>]) -> Result<usize, DecisionError> {
    validate_distribution(p)?;
    if l.len() != p.len() || l.is_empty() || l[0].is_empty() {
        return Err(DecisionError::InvalidLikelihoodMatrix);
    }
    let outcomes = l[0].len();
    for row in l {
        if row.len() != outcomes
            || row.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0)
        {
            return Err(DecisionError::InvalidLikelihoodMatrix);
        }
        let sum: f64 = row.iter().sum();
        if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(DecisionError::InvalidLikelihoodMatrix);
        }
    }
    Ok(outcomes)
}

fn entropy(p: &[f64]) -> f64 {
    p.iter().filter(|x| **x > 0.0).map(|x| -x * x.ln()).sum()
}

fn best_action_unchecked(p: &[f64], model: &DecisionModel) -> BestAction {
    let mut values = vec![0.0; model.action_ids.len()];
    for (h, probability) in p.iter().enumerate() {
        for (a, utility) in model.utility_by_hypothesis[h].iter().enumerate() {
            values[a] += probability * utility;
        }
    }
    let mut best = 0;
    for candidate in 1..values.len() {
        if values[candidate] > values[best] + EPS
            || ((values[candidate] - values[best]).abs() <= EPS
                && model.action_ids[candidate] < model.action_ids[best])
        {
            best = candidate;
        }
    }
    BestAction {
        action_id: model.action_ids[best].clone(),
        action_index: best,
        expected_utility: values[best],
    }
}

pub fn best_action(p: &[f64], model: &DecisionModel) -> Result<BestAction, DecisionError> {
    validate_model(p, model)?;
    Ok(best_action_unchecked(p, model))
}

/// Exact equality of utility rows defines decision-equivalent hypotheses for
/// this finite action set. This is not predictive or causal equivalence.
pub fn decision_sufficient_partition(
    p: &[f64],
    model: &DecisionModel,
) -> Result<Vec<DecisionClass>, DecisionError> {
    validate_model(p, model)?;
    let mut classes: Vec<DecisionClass> = Vec::new();
    for (i, row) in model.utility_by_hypothesis.iter().enumerate() {
        if let Some(class) = classes.iter_mut().find(|c| c.utility_by_action == *row) {
            class.hypothesis_indices.push(i);
            class.posterior_mass += p[i];
        } else {
            classes.push(DecisionClass {
                hypothesis_indices: vec![i],
                utility_by_action: row.clone(),
                posterior_mass: p[i],
            });
        }
    }
    Ok(classes)
}

pub fn expected_information_gain(p: &[f64], l: &[Vec<f64>]) -> Result<f64, DecisionError> {
    let outcomes = validate_likelihoods(p, l)?;
    let mut expected_entropy = 0.0;
    for o in 0..outcomes {
        let q: f64 = p.iter().enumerate().map(|(h, ph)| ph * l[h][o]).sum();
        if q <= 0.0 { continue; }
        let posterior: Vec<f64> = p.iter().enumerate()
            .map(|(h, ph)| ph * l[h][o] / q).collect();
        expected_entropy += q * entropy(&posterior);
    }
    let gain = entropy(p) - expected_entropy;
    if !gain.is_finite() || gain < -EPS {
        return Err(DecisionError::NumericalInconsistency);
    }
    Ok(gain.max(0.0))
}

/// EVSI = E_o[max_a E[U(a,H)|o,e]] - max_a E[U(a,H)].
/// Cost is kept separate and converted to utility units explicitly by the caller.
pub fn evaluate_decision_experiment(
    p: &[f64],
    model: &DecisionModel,
    candidate: &DecisionExperimentCandidate,
) -> Result<DecisionExperimentEvaluation, DecisionError> {
    validate_model(p, model)?;
    if candidate.experiment_id.trim().is_empty() || candidate.cost_units == 0 {
        return Err(DecisionError::InvalidExperiment);
    }
    let outcomes = validate_likelihoods(p, &candidate.outcome_likelihoods)?;
    let current = best_action_unchecked(p, model);
    let mut expected_after = 0.0;
    for o in 0..outcomes {
        let q: f64 = p.iter().enumerate()
            .map(|(h, ph)| ph * candidate.outcome_likelihoods[h][o]).sum();
        if q <= 0.0 { continue; }
        let posterior: Vec<f64> = p.iter().enumerate()
            .map(|(h, ph)| ph * candidate.outcome_likelihoods[h][o] / q).collect();
        expected_after += q * best_action_unchecked(&posterior, model).expected_utility;
    }
    let evsi = expected_after - current.expected_utility;
    if !expected_after.is_finite() || !evsi.is_finite() || evsi < -EPS {
        return Err(DecisionError::NumericalInconsistency);
    }
    Ok(DecisionExperimentEvaluation {
        experiment_id: candidate.experiment_id.clone(),
        current_action_id: current.action_id,
        current_decision_value: current.expected_utility,
        expected_post_experiment_decision_value: expected_after,
        gross_evsi: evsi.max(0.0),
        expected_information_gain_nats: expected_information_gain(p, &candidate.outcome_likelihoods)?,
        cost_units: candidate.cost_units,
    })
}

/// Maximize positive net EVSI. Abstain if no experiment's expected decision
/// improvement exceeds its cost. Ties resolve lexicographically by experiment ID.
pub fn choose_decision_experiment(
    p: &[f64],
    model: &DecisionModel,
    candidates: &[DecisionExperimentCandidate],
    cost_rate_per_unit: f64,
) -> Result<Option<DecisionExperimentChoice>, DecisionError> {
    if !cost_rate_per_unit.is_finite() || cost_rate_per_unit < 0.0 {
        return Err(DecisionError::InvalidCostRate);
    }
    validate_model(p, model)?;
    let mut best: Option<DecisionExperimentChoice> = None;
    for (i, candidate) in candidates.iter().enumerate() {
        if candidates[..i].iter().any(|x| x.experiment_id == candidate.experiment_id) {
            return Err(DecisionError::DuplicateExperimentId);
        }
        let evaluation = evaluate_decision_experiment(p, model, candidate)?;
        let net = evaluation.gross_evsi - cost_rate_per_unit * candidate.cost_units as f64;
        if !net.is_finite() { return Err(DecisionError::NumericalInconsistency); }
        if net <= EPS { continue; }
        let choice = DecisionExperimentChoice { evaluation, cost_rate_per_unit, net_evsi: net };
        let replace = match &best {
            None => true,
            Some(old) if net > old.net_evsi + EPS => true,
            Some(old) if (net - old.net_evsi).abs() <= EPS =>
                choice.evaluation.experiment_id < old.evaluation.experiment_id,
            _ => false,
        };
        if replace { best = Some(choice); }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> DecisionModel {
        DecisionModel {
            action_ids: vec!["act-a".into(), "act-b".into()],
            utility_by_hypothesis: vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        }
    }
    fn irrelevant() -> DecisionExperimentCandidate {
        DecisionExperimentCandidate {
            experiment_id: "irrelevant-hypothesis-split".into(),
            cost_units: 1,
            outcome_likelihoods: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]],
        }
    }
    fn relevant() -> DecisionExperimentCandidate {
        DecisionExperimentCandidate {
            experiment_id: "decision-boundary-probe".into(),
            cost_units: 1,
            outcome_likelihoods: vec![vec![0.8, 0.2], vec![0.8, 0.2], vec![0.2, 0.8]],
        }
    }

    #[test]
    fn current_action_uses_expected_utility() {
        let a = best_action(&[0.3, 0.3, 0.4], &model()).unwrap();
        assert_eq!(a.action_id, "act-a");
        assert!((a.expected_utility - 0.6).abs() < EPS);
    }
    #[test]
    fn raw_information_gain_prefers_decision_irrelevant_probe() {
        let p = [0.3, 0.3, 0.4];
        let a = expected_information_gain(&p, &irrelevant().outcome_likelihoods).unwrap();
        let b = expected_information_gain(&p, &relevant().outcome_likelihoods).unwrap();
        assert!(a > b);
    }
    #[test]
    fn decision_value_prefers_lower_information_decision_relevant_probe() {
        let p = [0.3, 0.3, 0.4];
        let a = evaluate_decision_experiment(&p, &model(), &irrelevant()).unwrap();
        let b = evaluate_decision_experiment(&p, &model(), &relevant()).unwrap();
        assert!(a.gross_evsi.abs() < EPS);
        assert!((b.gross_evsi - 0.2).abs() < EPS);
        assert!(a.expected_information_gain_nats > b.expected_information_gain_nats);
    }
    #[test]
    fn planner_selects_decision_relevant_experiment() {
        let c = choose_decision_experiment(
            &[0.3, 0.3, 0.4], &model(), &[irrelevant(), relevant()], 0.0
        ).unwrap().unwrap();
        assert_eq!(c.evaluation.experiment_id, "decision-boundary-probe");
        assert!((c.net_evsi - 0.2).abs() < EPS);
    }
    #[test]
    fn planner_abstains_on_information_without_decision_value() {
        assert_eq!(
            choose_decision_experiment(&[0.3, 0.3, 0.4], &model(), &[irrelevant()], 0.0).unwrap(),
            None
        );
    }
    #[test]
    fn planner_abstains_when_cost_exceeds_evsi() {
        assert_eq!(
            choose_decision_experiment(&[0.3, 0.3, 0.4], &model(), &[relevant()], 0.21).unwrap(),
            None
        );
    }
    #[test]
    fn decision_partition_merges_identical_utility_rows() {
        let c = decision_sufficient_partition(&[0.3, 0.3, 0.4], &model()).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].hypothesis_indices, vec![0, 1]);
        assert!((c[0].posterior_mass - 0.6).abs() < EPS);
    }
    #[test]
    fn moving_probability_inside_class_preserves_action_value() {
        let first = best_action(&[0.3, 0.3, 0.4], &model()).unwrap();
        let second = best_action(&[0.5, 0.1, 0.4], &model()).unwrap();
        assert_eq!(first.action_id, second.action_id);
        assert!((first.expected_utility - second.expected_utility).abs() < EPS);
    }
    #[test]
    fn same_decision_class_can_still_have_predictive_information() {
        let e = evaluate_decision_experiment(&[0.3, 0.3, 0.4], &model(), &irrelevant()).unwrap();
        assert!(e.expected_information_gain_nats > 0.0);
        assert!(e.gross_evsi.abs() < EPS);
    }
    #[test]
    fn cost_rate_must_be_finite_and_nonnegative() {
        assert_eq!(
            choose_decision_experiment(&[0.3, 0.3, 0.4], &model(), &[], f64::NAN),
            Err(DecisionError::InvalidCostRate)
        );
        assert_eq!(
            choose_decision_experiment(&[0.3, 0.3, 0.4], &model(), &[], -1.0),
            Err(DecisionError::InvalidCostRate)
        );
    }
    #[test]
    fn duplicate_experiment_ids_are_rejected() {
        let mut d = relevant();
        d.experiment_id = irrelevant().experiment_id;
        assert_eq!(
            choose_decision_experiment(&[0.3, 0.3, 0.4], &model(), &[irrelevant(), d], 0.0),
            Err(DecisionError::DuplicateExperimentId)
        );
    }
    #[test]
    fn invalid_probabilities_and_utility_matrix_are_rejected() {
        assert_eq!(
            best_action(&[0.5, 0.6], &model()),
            Err(DecisionError::InvalidDistribution)
        );
        let mut m = model();
        m.utility_by_hypothesis[0][0] = f64::NAN;
        assert_eq!(
            best_action(&[0.3, 0.3, 0.4], &m),
            Err(DecisionError::InvalidUtilityMatrix)
        );
    }
    #[test]
    fn invalid_likelihood_rows_are_rejected() {
        let mut d = relevant();
        d.outcome_likelihoods[0] = vec![0.7, 0.7];
        assert_eq!(
            evaluate_decision_experiment(&[0.3, 0.3, 0.4], &model(), &d),
            Err(DecisionError::InvalidLikelihoodMatrix)
        );
    }
    #[test]
    fn ties_are_resolved_by_experiment_id() {
        let mut a = relevant();
        a.experiment_id = "zeta".into();
        let mut b = relevant();
        b.experiment_id = "alpha".into();
        let chosen = choose_decision_experiment(&[0.3, 0.3, 0.4], &model(), &[a, b], 0.0)
            .unwrap().unwrap();
        assert_eq!(chosen.evaluation.experiment_id, "alpha");
    }
    #[test]
    fn expected_value_after_optional_observation_is_not_lower() {
        let p = [0.3, 0.3, 0.4];
        let base = best_action(&p, &model()).unwrap().expected_utility;
        for d in [irrelevant(), relevant()] {
            let e = evaluate_decision_experiment(&p, &model(), &d).unwrap();
            assert!(e.expected_post_experiment_decision_value + EPS >= base);
        }
    }
}
