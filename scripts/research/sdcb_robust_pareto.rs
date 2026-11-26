//! Robust and Pareto-aware experimental decision reference for SDCB.
//!
//! Build from repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_robust_pareto.rs -o /tmp/sdcb-robust-pareto
//!
//! This module adds two deliberately separate contracts:
//! 1. finite ambiguity-set robust EVSI under multiple plausible model scenarios;
//! 2. Pareto action / experiment-policy frontiers with non-compensatory
//!    per-hypothesis constraints.
//!
//! It is an exact small-instance reference, not a scalable optimizer.

const EPS: f64 = 1e-12;
const PROB_TOL: f64 = 1e-9;
const MAX_ROBUST_SCENARIOS: usize = 8;
const MAX_ROBUST_ACTIONS: usize = 5;
const MAX_ROBUST_OUTCOMES: usize = 4;
const MAX_ROBUST_CANDIDATES: usize = 8;
const MAX_PARETO_ACTIONS: usize = 8;
const MAX_PARETO_OUTCOMES: usize = 4;
const MAX_PARETO_OBJECTIVES: usize = 6;
const MAX_PARETO_CANDIDATES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobustParetoError {
    InvalidDistribution,
    InvalidModel,
    InvalidScenario,
    InvalidActionSet,
    InvalidObjectiveSet,
    InvalidUtilityMatrix,
    InvalidLikelihoodMatrix,
    InvalidExperiment,
    DuplicateIdentifier,
    InvalidCostRate,
    NoFeasibleAction,
    NumericalInconsistency,
    SearchSpaceTooLarge,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UtilityScenario {
    pub scenario_id: String,
    /// Rows are hypotheses, columns are actions. All values are maximized.
    pub utility_by_hypothesis: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustDecisionModel {
    pub action_ids: Vec<String>,
    /// Finite ambiguity set. These scenarios are not assigned probabilities.
    pub scenarios: Vec<UtilityScenario>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustExperimentCandidate {
    pub experiment_id: String,
    pub cost_units: u64,
    /// Scenario -> hypothesis -> outcome likelihood.
    pub likelihoods_by_scenario: Vec<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustAction {
    pub action_id: String,
    pub action_index: usize,
    pub scenario_values: Vec<f64>,
    pub worst_case_value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustExperimentEvaluation {
    pub experiment_id: String,
    pub baseline_action_id: String,
    pub baseline_worst_case_value: f64,
    pub policy_actions_by_outcome: Vec<String>,
    pub scenario_policy_values: Vec<f64>,
    pub robust_post_experiment_value: f64,
    pub robust_gain: f64,
    pub cost_units: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustExperimentChoice {
    pub evaluation: RobustExperimentEvaluation,
    pub cost_rate_per_unit: f64,
    pub net_robust_gain: f64,
}

fn validate_probabilities(p: &[f64]) -> Result<(), RobustParetoError> {
    if p.is_empty() || p.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0) {
        return Err(RobustParetoError::InvalidDistribution);
    }
    let total: f64 = p.iter().sum();
    if !total.is_finite() || (total - 1.0).abs() > PROB_TOL {
        return Err(RobustParetoError::InvalidDistribution);
    }
    Ok(())
}

fn validate_likelihood_matrix(
    p: &[f64],
    likelihoods: &[Vec<f64>],
) -> Result<usize, RobustParetoError> {
    if likelihoods.len() != p.len() || likelihoods.is_empty() {
        return Err(RobustParetoError::InvalidLikelihoodMatrix);
    }
    let outcomes = likelihoods[0].len();
    if outcomes == 0 || outcomes > MAX_ROBUST_OUTCOMES.max(MAX_PARETO_OUTCOMES) {
        return Err(RobustParetoError::SearchSpaceTooLarge);
    }
    for row in likelihoods {
        if row.len() != outcomes
            || row.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 1.0)
        {
            return Err(RobustParetoError::InvalidLikelihoodMatrix);
        }
        let sum: f64 = row.iter().sum();
        if !sum.is_finite() || (sum - 1.0).abs() > PROB_TOL {
            return Err(RobustParetoError::InvalidLikelihoodMatrix);
        }
    }
    Ok(outcomes)
}

fn validate_robust_model(
    posterior: &[f64],
    model: &RobustDecisionModel,
) -> Result<(), RobustParetoError> {
    validate_probabilities(posterior)?;
    let actions = model.action_ids.len();
    if actions == 0 || actions > MAX_ROBUST_ACTIONS {
        return Err(RobustParetoError::InvalidActionSet);
    }
    if model.scenarios.is_empty() || model.scenarios.len() > MAX_ROBUST_SCENARIOS {
        return Err(RobustParetoError::InvalidModel);
    }
    for (i, id) in model.action_ids.iter().enumerate() {
        if id.trim().is_empty() || model.action_ids[..i].contains(id) {
            return Err(RobustParetoError::InvalidActionSet);
        }
    }
    for (s, scenario) in model.scenarios.iter().enumerate() {
        if scenario.scenario_id.trim().is_empty()
            || model.scenarios[..s]
                .iter()
                .any(|old| old.scenario_id == scenario.scenario_id)
            || scenario.utility_by_hypothesis.len() != posterior.len()
            || scenario.utility_by_hypothesis.iter().any(|row| {
                row.len() != actions || row.iter().any(|utility| !utility.is_finite())
            })
        {
            return Err(RobustParetoError::InvalidScenario);
        }
    }
    Ok(())
}

fn validate_robust_candidate(
    posterior: &[f64],
    model: &RobustDecisionModel,
    candidate: &RobustExperimentCandidate,
) -> Result<usize, RobustParetoError> {
    if candidate.experiment_id.trim().is_empty() || candidate.cost_units == 0 {
        return Err(RobustParetoError::InvalidExperiment);
    }
    if candidate.likelihoods_by_scenario.len() != model.scenarios.len() {
        return Err(RobustParetoError::InvalidLikelihoodMatrix);
    }
    let mut common_outcomes = None;
    for matrix in &candidate.likelihoods_by_scenario {
        let outcomes = validate_likelihood_matrix(posterior, matrix)?;
        if outcomes > MAX_ROBUST_OUTCOMES {
            return Err(RobustParetoError::SearchSpaceTooLarge);
        }
        if common_outcomes.is_some_and(|n| n != outcomes) {
            return Err(RobustParetoError::InvalidLikelihoodMatrix);
        }
        common_outcomes = Some(outcomes);
    }
    common_outcomes.ok_or(RobustParetoError::InvalidLikelihoodMatrix)
}

fn enumerate_policies(action_count: usize, outcome_count: usize) -> Result<Vec<Vec<usize>>, RobustParetoError> {
    let mut count = 1usize;
    for _ in 0..outcome_count {
        count = count.checked_mul(action_count).ok_or(RobustParetoError::SearchSpaceTooLarge)?;
        if count > 4096 {
            return Err(RobustParetoError::SearchSpaceTooLarge);
        }
    }

    let mut policies = Vec::with_capacity(count);
    for mut code in 0..count {
        let mut policy = vec![0; outcome_count];
        for action in &mut policy {
            *action = code % action_count;
            code /= action_count;
        }
        policies.push(policy);
    }
    Ok(policies)
}

fn scenario_expected_action_value(
    posterior: &[f64],
    utilities: &[Vec<f64>],
    action: usize,
) -> f64 {
    posterior
        .iter()
        .enumerate()
        .map(|(h, p)| p * utilities[h][action])
        .sum()
}

/// Maximize the worst-case expected utility over the supplied finite ambiguity set.
pub fn robust_current_action(
    posterior: &[f64],
    model: &RobustDecisionModel,
) -> Result<RobustAction, RobustParetoError> {
    validate_robust_model(posterior, model)?;
    let mut best: Option<RobustAction> = None;

    for (action_index, action_id) in model.action_ids.iter().enumerate() {
        let values: Vec<f64> = model
            .scenarios
            .iter()
            .map(|scenario| {
                scenario_expected_action_value(
                    posterior,
                    &scenario.utility_by_hypothesis,
                    action_index,
                )
            })
            .collect();
        let worst = values.iter().copied().fold(f64::INFINITY, f64::min);
        let candidate = RobustAction {
            action_id: action_id.clone(),
            action_index,
            scenario_values: values,
            worst_case_value: worst,
        };
        let replace = match &best {
            None => true,
            Some(old) if worst > old.worst_case_value + EPS => true,
            Some(old)
                if (worst - old.worst_case_value).abs() <= EPS =>
            {
                action_id < &old.action_id
            }
            _ => false,
        };
        if replace {
            best = Some(candidate);
        }
    }
    best.ok_or(RobustParetoError::NoFeasibleAction)
}

fn policy_value_in_scenario(
    posterior: &[f64],
    scenario: &UtilityScenario,
    likelihoods: &[Vec<f64>],
    policy: &[usize],
) -> f64 {
    let mut value = 0.0;
    for h in 0..posterior.len() {
        for outcome in 0..policy.len() {
            value += posterior[h]
                * likelihoods[h][outcome]
                * scenario.utility_by_hypothesis[h][policy[outcome]];
        }
    }
    value
}

/// Exact maximin policy for one experiment over a finite set of complete
/// likelihood/utility scenarios. The policy maps outcomes to actions and is
/// shared across scenarios: it cannot observe which ambiguity scenario is true.
pub fn evaluate_robust_experiment(
    posterior: &[f64],
    model: &RobustDecisionModel,
    candidate: &RobustExperimentCandidate,
) -> Result<RobustExperimentEvaluation, RobustParetoError> {
    validate_robust_model(posterior, model)?;
    let outcomes = validate_robust_candidate(posterior, model, candidate)?;
    let baseline = robust_current_action(posterior, model)?;
    let policies = enumerate_policies(model.action_ids.len(), outcomes)?;
    let mut best_policy: Option<Vec<usize>> = None;
    let mut best_scenario_values = Vec::new();
    let mut best_worst = f64::NEG_INFINITY;

    for policy in policies {
        let scenario_values: Vec<f64> = model
            .scenarios
            .iter()
            .zip(candidate.likelihoods_by_scenario.iter())
            .map(|(scenario, likelihoods)| {
                policy_value_in_scenario(posterior, scenario, likelihoods, &policy)
            })
            .collect();
        let worst = scenario_values.iter().copied().fold(f64::INFINITY, f64::min);
        let replace = match &best_policy {
            None => true,
            Some(old) if worst > best_worst + EPS => true,
            Some(old) if (worst - best_worst).abs() <= EPS => policy.as_slice() < old.as_slice(),
            _ => false,
        };
        if replace {
            best_worst = worst;
            best_policy = Some(policy);
            best_scenario_values = scenario_values;
        }
    }

    let selected = best_policy.ok_or(RobustParetoError::NoFeasibleAction)?;
    let gain = best_worst - baseline.worst_case_value;
    if !gain.is_finite() || gain < -EPS {
        return Err(RobustParetoError::NumericalInconsistency);
    }
    Ok(RobustExperimentEvaluation {
        experiment_id: candidate.experiment_id.clone(),
        baseline_action_id: baseline.action_id,
        baseline_worst_case_value: baseline.worst_case_value,
        policy_actions_by_outcome: selected
            .iter()
            .map(|index| model.action_ids[*index].clone())
            .collect(),
        scenario_policy_values: best_scenario_values,
        robust_post_experiment_value: best_worst,
        robust_gain: gain.max(0.0),
        cost_units: candidate.cost_units,
    })
}

/// Select by worst-case net EVSI. Scenarios are ambiguity-set members, not a
/// probability distribution. The experiment's outcome-contingent action policy
/// must be the same for every scenario. This reference enumerates deterministic
/// policies only; mixed policies are not included.
pub fn choose_robust_experiment(
    posterior: &[f64],
    model: &RobustDecisionModel,
    candidates: &[RobustExperimentCandidate],
    cost_rate_per_unit: f64,
) -> Result<Option<RobustExperimentChoice>, RobustParetoError> {
    if !cost_rate_per_unit.is_finite() || cost_rate_per_unit < 0.0 {
        return Err(RobustParetoError::InvalidCostRate);
    }
    validate_robust_model(posterior, model)?;
    if candidates.len() > MAX_ROBUST_CANDIDATES {
        return Err(RobustParetoError::SearchSpaceTooLarge);
    }
    let mut best: Option<RobustExperimentChoice> = None;

    for (index, candidate) in candidates.iter().enumerate() {
        if candidates[..index]
            .iter()
            .any(|old| old.experiment_id == candidate.experiment_id)
        {
            return Err(RobustParetoError::DuplicateIdentifier);
        }
        let evaluation = evaluate_robust_experiment(posterior, model, candidate)?;
        let cost = cost_rate_per_unit * candidate.cost_units as f64;
        let net = evaluation.robust_gain - cost;
        if !net.is_finite() {
            return Err(RobustParetoError::NumericalInconsistency);
        }
        if net <= EPS {
            continue;
        }
        let choice = RobustExperimentChoice {
            evaluation,
            cost_rate_per_unit,
            net_robust_gain: net,
        };
        let replace = match &best {
            None => true,
            Some(old) if net > old.net_robust_gain + EPS => true,
            Some(old) if (net - old.net_robust_gain).abs() <= EPS =>
                choice.evaluation.experiment_id < old.evaluation.experiment_id,
            _ => false,
        };
        if replace {
            best = Some(choice);
        }
    }
    Ok(best)
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiObjectiveModel {
    pub action_ids: Vec<String>,
    pub objective_ids: Vec<String>,
    /// [hypothesis][action][objective]. All objectives are maximized.
    /// For risk/safety, use an explicitly defined safety margin or negative loss.
    pub utility_by_hypothesis: Vec<Vec<Vec<f64>>>,
    /// Non-compensatory lower bounds on objective values. Each constrained
    /// action must satisfy the bound under every hypothesis with nonzero prior.
    pub minimum_by_objective: Vec<Option<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoAction {
    pub action_id: String,
    pub expected_objectives: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoExperimentCandidate {
    pub experiment_id: String,
    pub cost_units: u64,
    /// Row h contains P(outcome | hypothesis h, experiment).
    pub outcome_likelihoods: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoExperimentPolicy {
    pub experiment_id: String,
    pub actions_by_outcome: Vec<String>,
    pub expected_objectives: Vec<f64>,
    pub cost_units: u64,
}

fn validate_multiobjective_model(
    posterior: &[f64],
    model: &MultiObjectiveModel,
) -> Result<(), RobustParetoError> {
    validate_probabilities(posterior)?;
    let actions = model.action_ids.len();
    let objectives = model.objective_ids.len();
    if actions == 0 || actions > MAX_PARETO_ACTIONS {
        return Err(RobustParetoError::InvalidActionSet);
    }
    if objectives == 0 || objectives > MAX_PARETO_OBJECTIVES
        || model.minimum_by_objective.len() != objectives
    {
        return Err(RobustParetoError::InvalidObjectiveSet);
    }
    if model.utility_by_hypothesis.len() != posterior.len() {
        return Err(RobustParetoError::InvalidUtilityMatrix);
    }
    for (i, id) in model.action_ids.iter().enumerate() {
        if id.trim().is_empty() || model.action_ids[..i].contains(id) {
            return Err(RobustParetoError::InvalidActionSet);
        }
    }
    for (i, id) in model.objective_ids.iter().enumerate() {
        if id.trim().is_empty() || model.objective_ids[..i].contains(id) {
            return Err(RobustParetoError::InvalidObjectiveSet);
        }
    }
    for threshold in &model.minimum_by_objective {
        if threshold.is_some_and(|x| !x.is_finite()) {
            return Err(RobustParetoError::InvalidObjectiveSet);
        }
    }
    for row in &model.utility_by_hypothesis {
        if row.len() != actions
            || row.iter().any(|action| {
                action.len() != objectives || action.iter().any(|x| !x.is_finite())
            })
        {
            return Err(RobustParetoError::InvalidUtilityMatrix);
        }
    }
    Ok(())
}

fn action_is_feasible(model: &MultiObjectiveModel, action: usize, posterior: &[f64]) -> bool {
    for (objective, threshold) in model.minimum_by_objective.iter().enumerate() {
        if let Some(minimum) = threshold {
            for (h, probability) in posterior.iter().enumerate() {
                if *probability > 0.0
                    && model.utility_by_hypothesis[h][action][objective] < *minimum - EPS
                {
                    return false;
                }
            }
        }
    }
    true
}

fn expected_action_objectives(
    posterior: &[f64],
    model: &MultiObjectiveModel,
    action: usize,
) -> Vec<f64> {
    let mut values = vec![0.0; model.objective_ids.len()];
    for (h, probability) in posterior.iter().enumerate() {
        for (objective, utility) in model.utility_by_hypothesis[h][action].iter().enumerate() {
            values[objective] += probability * utility;
        }
    }
    values
}

fn dominates_values(a: &[f64], a_cost: u64, b: &[f64], b_cost: u64) -> bool {
    let no_worse = a.iter().zip(b).all(|(x, y)| *x >= *y - EPS) && a_cost <= b_cost;
    let strictly_better =
        a.iter().zip(b).any(|(x, y)| *x > *y + EPS) || a_cost < b_cost;
    no_worse && strictly_better
}

/// Pareto frontier of feasible actions. No scalar weights are used. An action
/// violating a hard lower bound for any plausible hypothesis is removed before
/// Pareto comparison; expected utility cannot compensate for that violation.
pub fn pareto_actions(
    posterior: &[f64],
    model: &MultiObjectiveModel,
) -> Result<Vec<ParetoAction>, RobustParetoError> {
    validate_multiobjective_model(posterior, model)?;
    let mut candidates = Vec::new();
    for (action, action_id) in model.action_ids.iter().enumerate() {
        if action_is_feasible(model, action, posterior) {
            candidates.push(ParetoAction {
                action_id: action_id.clone(),
                expected_objectives: expected_action_objectives(posterior, model, action),
            });
        }
    }
    if candidates.is_empty() {
        return Err(RobustParetoError::NoFeasibleAction);
    }

    let mut frontier = Vec::new();
    for i in 0..candidates.len() {
        let dominated = candidates.iter().enumerate().any(|(j, other)| {
            i != j && dominates_values(
                &other.expected_objectives,
                0,
                &candidates[i].expected_objectives,
                0,
            )
        });
        if !dominated {
            frontier.push(candidates[i].clone());
        }
    }
    frontier.sort_by(|a, b| a.action_id.cmp(&b.action_id));
    Ok(frontier)
}

/// Pareto frontier over all outcome-contingent deterministic policies for the
/// supplied experiment candidates. The comparison maximizes every declared
/// objective and minimizes cost; no weighted scalar score is constructed.
pub fn pareto_experiment_policies(
    posterior: &[f64],
    model: &MultiObjectiveModel,
    candidates: &[ParetoExperimentCandidate],
) -> Result<Vec<ParetoExperimentPolicy>, RobustParetoError> {
    validate_multiobjective_model(posterior, model)?;
    if candidates.len() > MAX_PARETO_CANDIDATES {
        return Err(RobustParetoError::SearchSpaceTooLarge);
    }
    for (i, candidate) in candidates.iter().enumerate() {
        if candidates[..i].iter().any(|old| old.experiment_id == candidate.experiment_id) {
            return Err(RobustParetoError::DuplicateIdentifier);
        }
    }

    let mut all = Vec::new();
    for candidate in candidates {
        if candidate.experiment_id.trim().is_empty()
            || candidate.experiment_id == "NO_EXPERIMENT_LOTTERY"
            || candidate.cost_units == 0
        {
            return Err(RobustParetoError::InvalidExperiment);
        }
        let outcomes = validate_likelihood_matrix(posterior, &candidate.outcome_likelihoods)?;
        if outcomes > MAX_PARETO_OUTCOMES {
            return Err(RobustParetoError::SearchSpaceTooLarge);
        }
        let policies = enumerate_policies(model.action_ids.len(), outcomes)?;
        let outcome_mass: Vec<f64> = (0..outcomes)
            .map(|o| posterior.iter().enumerate()
                .map(|(h, p)| p * candidate.outcome_likelihoods[h][o]).sum())
            .collect();

        for policy in policies {
            let mut feasible = true;
            for (outcome, action) in policy.iter().enumerate() {
                if outcome_mass[outcome] > 0.0
                    && !action_is_feasible(model, *action, posterior)
                {
                    feasible = false;
                    break;
                }
            }
            if !feasible {
                continue;
            }

            let mut expected = vec![0.0; model.objective_ids.len()];
            let mut no_experiment_lottery = vec![0.0; model.objective_ids.len()];
            for outcome in 0..outcomes {
                let q = outcome_mass[outcome];
                let action = policy[outcome];
                let unconditional_action_value =
                    expected_action_objectives(posterior, model, action);
                for objective in 0..model.objective_ids.len() {
                    no_experiment_lottery[objective] += q * unconditional_action_value[objective];
                }
            }
            for (h, probability) in posterior.iter().enumerate() {
                for outcome in 0..outcomes {
                    let likelihood = candidate.outcome_likelihoods[h][outcome];
                    let action = policy[outcome];
                    for objective in 0..model.objective_ids.len() {
                        expected[objective] += probability * likelihood
                            * model.utility_by_hypothesis[h][action][objective];
                    }
                }
            }
            // The outcome distribution itself can act as a randomizer. Add
            // the same action lottery without observing the experiment, at
            // zero experiment cost, so randomization is not miscredited as
            // information gained from a measurement.
            all.push(ParetoExperimentPolicy {
                experiment_id: "NO_EXPERIMENT_LOTTERY".into(),
                actions_by_outcome: policy.iter().map(|a| model.action_ids[*a].clone()).collect(),
                expected_objectives: no_experiment_lottery,
                cost_units: 0,
            });
            all.push(ParetoExperimentPolicy {
                experiment_id: candidate.experiment_id.clone(),
                actions_by_outcome: policy.iter().map(|a| model.action_ids[*a].clone()).collect(),
                expected_objectives: expected,
                cost_units: candidate.cost_units,
            });
        }
    }

    if all.is_empty() {
        return Err(RobustParetoError::NoFeasibleAction);
    }

    let mut frontier = Vec::new();
    'candidate: for i in 0..all.len() {
        for j in 0..all.len() {
            if i == j {
                continue;
            }
            let other = &all[j];
            let current = &all[i];
            if dominates_values(
                &other.expected_objectives,
                other.cost_units,
                &current.expected_objectives,
                current.cost_units,
            ) {
                continue 'candidate;
            }
            let equal = other.expected_objectives.iter().zip(&current.expected_objectives)
                .all(|(a, b)| (*a - *b).abs() <= EPS)
                && other.cost_units == current.cost_units;
            if equal {
                let other_key = (&other.experiment_id, &other.actions_by_outcome);
                let current_key = (&current.experiment_id, &current.actions_by_outcome);
                if other_key < current_key {
                    continue 'candidate;
                }
            }
        }
        frontier.push(all[i].clone());
    }

    frontier.sort_by(|a, b| {
        a.experiment_id.cmp(&b.experiment_id)
            .then_with(|| a.actions_by_outcome.cmp(&b.actions_by_outcome))
    });
    Ok(frontier)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn robust_model() -> RobustDecisionModel {
        RobustDecisionModel {
            action_ids: vec!["act-a".into(), "act-b".into()],
            scenarios: vec![
                UtilityScenario {
                    scenario_id: "nominal".into(),
                    utility_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                },
                UtilityScenario {
                    scenario_id: "alternative-likelihood".into(),
                    utility_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                },
            ],
        }
    }

    fn brittle() -> RobustExperimentCandidate {
        RobustExperimentCandidate {
            experiment_id: "brittle".into(),
            cost_units: 1,
            likelihoods_by_scenario: vec![
                vec![vec![0.95, 0.05], vec![0.05, 0.95]],
                vec![vec![0.05, 0.95], vec![0.95, 0.05]],
            ],
        }
    }

    fn robust_probe() -> RobustExperimentCandidate {
        RobustExperimentCandidate {
            experiment_id: "robust-probe".into(),
            cost_units: 1,
            likelihoods_by_scenario: vec![
                vec![vec![0.7, 0.3], vec![0.3, 0.7]],
                vec![vec![0.7, 0.3], vec![0.3, 0.7]],
            ],
        }
    }

    fn pareto_model(constrained: bool) -> MultiObjectiveModel {
        MultiObjectiveModel {
            action_ids: vec!["fast".into(), "balanced".into(), "safe".into()],
            objective_ids: vec!["throughput".into(), "safety_margin".into()],
            utility_by_hypothesis: vec![
                vec![vec![10.0, -4.0], vec![8.0, -1.0], vec![6.0, 0.0]],
                vec![vec![10.0, 1.0], vec![8.0, -1.0], vec![6.0, 0.0]],
            ],
            minimum_by_objective: if constrained {
                vec![None, Some(-2.0)]
            } else {
                vec![None, None]
            },
        }
    }

    #[test]
    fn robust_baseline_uses_worst_case_scenario() {
        let action = robust_current_action(&[0.5, 0.5], &robust_model()).unwrap();
        assert!((action.worst_case_value - 0.5).abs() < EPS);
    }

    #[test]
    fn robust_probe_beats_a_misspecified_brittle_probe() {
        let model = robust_model();
        let posterior = [0.5, 0.5];
        let brittle_eval = evaluate_robust_experiment(&posterior, &model, &brittle()).unwrap();
        let robust_eval = evaluate_robust_experiment(&posterior, &model, &robust_probe()).unwrap();
        assert!((brittle_eval.robust_gain).abs() < EPS);
        assert!((robust_eval.robust_gain - 0.2).abs() < EPS);
    }

    #[test]
    fn robust_selector_prefers_positive_worst_case_net_value() {
        let selected = choose_robust_experiment(
            &[0.5, 0.5],
            &robust_model(),
            &[brittle(), robust_probe()],
            0.1,
        ).unwrap().unwrap();
        assert_eq!(selected.evaluation.experiment_id, "robust-probe");
        assert!((selected.net_robust_gain - 0.1).abs() < EPS);
    }

    #[test]
    fn robust_selector_abstains_when_only_brittle_probe_exists() {
        assert_eq!(
            choose_robust_experiment(&[0.5, 0.5], &robust_model(), &[brittle()], 0.0).unwrap(),
            None
        );
    }

    #[test]
    fn robust_policy_is_shared_across_scenarios() {
        let evaluation = evaluate_robust_experiment(
            &[0.5, 0.5],
            &robust_model(),
            &brittle(),
        ).unwrap();
        // The scenario-specific reversal cannot be exploited because the policy
        // does not receive the hidden scenario identity.
        assert_eq!(evaluation.policy_actions_by_outcome, vec!["act-a", "act-a"]);
        assert!((evaluation.robust_post_experiment_value - 0.5).abs() < EPS);
    }

    #[test]
    fn invalid_robust_scenario_dimensions_are_rejected() {
        let mut candidate = robust_probe();
        candidate.likelihoods_by_scenario.pop();
        assert_eq!(
            evaluate_robust_experiment(&[0.5, 0.5], &robust_model(), &candidate),
            Err(RobustParetoError::InvalidLikelihoodMatrix)
        );
    }

    #[test]
    fn robust_duplicate_candidate_ids_are_rejected() {
        let mut second = robust_probe();
        second.experiment_id = "brittle".into();
        assert_eq!(
            choose_robust_experiment(&[0.5, 0.5], &robust_model(), &[brittle(), second], 0.0),
            Err(RobustParetoError::DuplicateIdentifier)
        );
    }

    #[test]
    fn pareto_action_frontier_respects_per_hypothesis_hard_bounds() {
        let frontier = pareto_actions(&[0.5, 0.5], &pareto_model(true)).unwrap();
        let ids: Vec<_> = frontier.iter().map(|x| x.action_id.as_str()).collect();
        assert_eq!(ids, vec!["balanced", "safe"]);
    }

    #[test]
    fn expected_utility_cannot_hide_a_single_hypothesis_safety_violation() {
        let frontier = pareto_actions(&[0.5, 0.5], &pareto_model(true)).unwrap();
        assert!(!frontier.iter().any(|x| x.action_id == "fast"));
        // fast's mean safety_margin equals -1.5, which is above the -2 floor,
        // but its h1 value is -4 and therefore violates the hard floor.
    }

    #[test]
    fn unconstrained_pareto_frontier_preserves_throughput_safety_tradeoff() {
        let frontier = pareto_actions(&[0.5, 0.5], &pareto_model(false)).unwrap();
        assert_eq!(frontier.len(), 3);
    }

    #[test]
    fn pareto_frontier_does_not_create_a_weighted_scalar_score() {
        let frontier = pareto_actions(&[0.5, 0.5], &pareto_model(false)).unwrap();
        assert!(frontier.iter().all(|item| item.expected_objectives.len() == 2));
        assert!(frontier.iter().any(|item| item.expected_objectives == vec![8.0, -1.0]));
        assert!(frontier.iter().any(|item| item.expected_objectives == vec![6.0, 0.0]));
    }

    #[test]
    fn pareto_policy_frontier_preserves_feasible_tradeoffs() {
        let candidate = ParetoExperimentCandidate {
            experiment_id: "uninformative-randomized-measurement".into(),
            cost_units: 1,
            outcome_likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        };
        let frontier = pareto_experiment_policies(
            &[0.5, 0.5],
            &pareto_model(true),
            &[candidate],
        ).unwrap();
        assert!(frontier.iter().all(|policy| !policy.actions_by_outcome.iter().any(|a| a == "fast")));
        assert!(frontier.iter().all(|policy| policy.experiment_id == "NO_EXPERIMENT_LOTTERY"));
        assert!(frontier.iter().any(|p| p.expected_objectives == vec![8.0, -1.0] && p.cost_units == 0));
        assert!(frontier.iter().any(|p| p.expected_objectives == vec![7.0, -0.5] && p.cost_units == 0));
        assert!(frontier.iter().any(|p| p.expected_objectives == vec![6.0, 0.0] && p.cost_units == 0));
    }

    #[test]
    fn equal_pareto_policies_are_deduplicated_deterministically() {
        let candidate = ParetoExperimentCandidate {
            experiment_id: "coin-flip".into(),
            cost_units: 1,
            outcome_likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        };
        let frontier = pareto_experiment_policies(
            &[0.5, 0.5],
            &pareto_model(true),
            &[candidate],
        ).unwrap();
        let mixed: Vec<_> = frontier.iter()
            .filter(|p| p.expected_objectives == vec![7.0, -0.5])
            .collect();
        assert_eq!(mixed.len(), 1);
        assert_eq!(mixed[0].actions_by_outcome, vec!["balanced", "safe"]);
    }

    #[test]
    fn informative_observation_can_dominate_same_marginal_no_experiment_lottery() {
        let model = MultiObjectiveModel {
            action_ids: vec!["choose-h1".into(), "choose-h2".into()],
            objective_ids: vec!["performance-1".into(), "performance-2".into()],
            utility_by_hypothesis: vec![
                vec![vec![10.0, 0.0], vec![0.0, 10.0]],
                vec![vec![0.0, 10.0], vec![10.0, 0.0]],
            ],
            minimum_by_objective: vec![None, None],
        };
        let candidate = ParetoExperimentCandidate {
            experiment_id: "diagnostic-probe".into(),
            cost_units: 1,
            outcome_likelihoods: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
        };
        let frontier = pareto_experiment_policies(&[0.5, 0.5], &model, &[candidate]).unwrap();
        assert!(frontier.iter().any(|p| p.experiment_id == "diagnostic-probe" && p.expected_objectives == vec![9.0, 9.0]));
        assert!(frontier.iter().any(|p| p.experiment_id == "NO_EXPERIMENT_LOTTERY" && p.expected_objectives == vec![5.0, 5.0]));
    }
    #[test]
    fn pareto_policy_frontier_fails_closed_when_all_actions_are_unsafe() {
        let mut model = pareto_model(true);
        model.minimum_by_objective[1] = Some(2.0);
        let candidate = ParetoExperimentCandidate {
            experiment_id: "probe".into(),
            cost_units: 1,
            outcome_likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        };
        assert_eq!(
            pareto_experiment_policies(&[0.5, 0.5], &model, &[candidate]),
            Err(RobustParetoError::NoFeasibleAction)
        );
    }
    #[test]
    fn invalid_multiobjective_dimensions_are_rejected() {
        let mut model = pareto_model(true);
        model.minimum_by_objective.pop();
        assert_eq!(
            pareto_actions(&[0.5, 0.5], &model),
            Err(RobustParetoError::InvalidObjectiveSet)
        );
    }

    #[test]
    fn infeasible_action_set_fails_closed() {
        let mut model = pareto_model(true);
        model.minimum_by_objective[1] = Some(2.0);
        assert_eq!(
            pareto_actions(&[0.5, 0.5], &model),
            Err(RobustParetoError::NoFeasibleAction)
        );
    }

    #[test]
    fn invalid_likelihood_rows_are_rejected() {
        let candidate = ParetoExperimentCandidate {
            experiment_id: "bad".into(),
            cost_units: 1,
            outcome_likelihoods: vec![vec![0.8, 0.8], vec![0.5, 0.5]],
        };
        assert_eq!(
            pareto_experiment_policies(&[0.5, 0.5], &pareto_model(true), &[candidate]),
            Err(RobustParetoError::InvalidLikelihoodMatrix)
        );
    }

    #[test]
    fn pareto_duplicate_experiment_ids_are_rejected() {
        let a = ParetoExperimentCandidate {
            experiment_id: "same".into(),
            cost_units: 1,
            outcome_likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        };
        let b = ParetoExperimentCandidate { ..a.clone() };
        assert_eq!(
            pareto_experiment_policies(&[0.5, 0.5], &pareto_model(true), &[a, b]),
            Err(RobustParetoError::DuplicateIdentifier)
        );
    }
}
