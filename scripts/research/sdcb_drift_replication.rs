//! Exact two-stage scientific decision design with Markov instrument drift
//! and independently biased measurement channels.
//!
//! Build from repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_drift_replication.rs -o /tmp/sdcb-drift-replication
//! Run:
//!   /tmp/sdcb-drift-replication
//!
//! Each scenario defines a transition matrix per device for the fixed interval
//! between the two stages. Stage one observes the initial instrument state;
//! stage two observes the transitioned state. Instrument transitions are
//! conditionally independent across devices given their previous states and
//! scenario. Observation noise is conditionally independent given scientific
//! hypothesis, current instrument state, experiment, and scenario.
//!
//! This is a bounded exact reference, not a continuous-time drift model,
//! physical sensor model, or production planner.

const STATES: usize = 2;
const OUTCOMES: usize = 2;
const LEAVES: usize = 4;
const TOL: f64 = 1e-9;
const EPS: f64 = 1e-12;
const MAX_HYPOTHESES: usize = 4;
const MAX_DEVICES: usize = 3;
const MAX_SCENARIOS: usize = 4;
const MAX_ACTIONS: usize = 3;
const MAX_CANDIDATES: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentRole {
    Calibration,
    Science,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidDimensions,
    InvalidPrior,
    InvalidScenario,
    InvalidActionSet,
    InvalidCandidate,
    InvalidLikelihoods,
    CalibrationLeaksHypothesis,
    DuplicateIdentifier,
    InvalidTransitionMatrix,
    NoFeasiblePolicy,
    SearchSpaceTooLarge,
    NumericalInconsistency,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scenario {
    pub id: String,
    /// Rows are scientific hypotheses, columns are terminal actions.
    /// Utilities are maximized; abstention is an ordinary explicit action.
    pub utility_by_hypothesis: Vec<Vec<f64>>,
    /// Device -> previous binary state -> next binary state.
    /// One transition occurs between stage-one and stage-two measurements.
    pub transition_by_device: Vec<[[f64; STATES]; STATES]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub hypothesis_count: usize,
    pub device_ids: Vec<String>,
    /// Joint prior, hypothesis-major then device-state bitfield.
    /// joint_index = h * (2 ^ device_count) + sum(state[d] * 2^d).
    pub joint_prior: Vec<f64>,
    pub action_ids: Vec<String>,
    /// Complete ambiguity scenarios; no probabilities are assigned to them.
    pub scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub experiment_id: String,
    pub device_id: String,
    pub role: ExperimentRole,
    pub cost_units: u64,
    /// Scenario -> (hypothesis * STATES + device_state) -> P(binary outcome).
    pub likelihoods_by_scenario: Vec<Vec<[f64; OUTCOMES]>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    pub first_experiment_id: String,
    /// One second-stage experiment for each first-stage outcome [0, 1].
    pub second_experiments_by_first_outcome: [String; OUTCOMES],
    /// Terminal actions indexed by first_outcome * 2 + second_outcome.
    pub actions_by_outcome_pair: [String; LEAVES],
    pub scenario_values: Vec<f64>,
    pub worst_case_value: f64,
    pub baseline_action_id: String,
    pub baseline_worst_case_value: f64,
    pub robust_gain: f64,
    pub expected_cost_by_scenario: Vec<f64>,
    pub maximum_path_cost: u64,
}

fn validate_probabilities(values: &[f64]) -> Result<(), Error> {
    if values.is_empty()
        || values.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (values.iter().sum::<f64>() - 1.0).abs() > TOL
    {
        return Err(Error::InvalidPrior);
    }
    Ok(())
}

fn state_count(device_count: usize) -> usize {
    1usize << device_count
}

fn validate(model: &Model, candidates: &[Candidate]) -> Result<(), Error> {
    let h_count = model.hypothesis_count;
    let d_count = model.device_ids.len();
    if h_count < 2 || h_count > MAX_HYPOTHESES
        || d_count == 0 || d_count > MAX_DEVICES
        || model.device_ids.iter().any(|id| id.trim().is_empty())
        || (0..d_count).any(|i| model.device_ids[..i].contains(&model.device_ids[i]))
    {
        return Err(Error::InvalidDimensions);
    }

    let states = state_count(d_count);
    let joint_count = h_count.checked_mul(states).ok_or(Error::InvalidDimensions)?;
    if model.joint_prior.len() != joint_count {
        return Err(Error::InvalidDimensions);
    }
    validate_probabilities(&model.joint_prior)?;

    let action_count = model.action_ids.len();
    if action_count < 2 || action_count > MAX_ACTIONS
        || model.action_ids.iter().any(|id| id.trim().is_empty())
        || (0..action_count).any(|i| model.action_ids[..i].contains(&model.action_ids[i]))
    {
        return Err(Error::InvalidActionSet);
    }

    if model.scenarios.is_empty() || model.scenarios.len() > MAX_SCENARIOS {
        return Err(Error::InvalidScenario);
    }
    for (index, scenario) in model.scenarios.iter().enumerate() {
        if scenario.id.trim().is_empty()
            || model.scenarios[..index].iter().any(|old| old.id == scenario.id)
            || scenario.utility_by_hypothesis.len() != h_count
            || scenario.utility_by_hypothesis.iter().any(|row| {
                row.len() != action_count || row.iter().any(|u| !u.is_finite())
            })
            || scenario.transition_by_device.len() != d_count
        {
            return Err(Error::InvalidScenario);
        }
        for matrix in &scenario.transition_by_device {
            for row in matrix {
                if row.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                    || (row.iter().sum::<f64>() - 1.0).abs() > TOL
                {
                    return Err(Error::InvalidTransitionMatrix);
                }
            }
        }
    }

    if candidates.len() < 2 || candidates.len() > MAX_CANDIDATES {
        return Err(Error::SearchSpaceTooLarge);
    }
    for (index, candidate) in candidates.iter().enumerate() {
        if candidate.experiment_id.trim().is_empty()
            || candidate.device_id.trim().is_empty()
            || candidate.cost_units == 0
            || !model.device_ids.contains(&candidate.device_id)
        {
            return Err(Error::InvalidCandidate);
        }
        if candidates[..index].iter().any(|old| old.experiment_id == candidate.experiment_id) {
            return Err(Error::DuplicateIdentifier);
        }
        if candidate.likelihoods_by_scenario.len() != model.scenarios.len()
            || candidate.likelihoods_by_scenario.iter().any(|matrix| {
                matrix.len() != h_count * STATES
                    || matrix.iter().any(|row| {
                        row.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                            || (row.iter().sum::<f64>() - 1.0).abs() > TOL
                    })
            })
        {
            return Err(Error::InvalidLikelihoods);
        }

        if candidate.role == ExperimentRole::Calibration {
            for matrix in &candidate.likelihoods_by_scenario {
                for state in 0..STATES {
                    let reference = matrix[state];
                    for hypothesis in 1..h_count {
                        let row = matrix[hypothesis * STATES + state];
                        if (0..OUTCOMES).any(|outcome| (row[outcome] - reference[outcome]).abs() > TOL) {
                            return Err(Error::CalibrationLeaksHypothesis);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn robust_baseline(model: &Model) -> Result<(usize, f64), Error> {
    let states = state_count(model.device_ids.len());
    let mut best: Option<(usize, f64)> = None;
    for action in 0..model.action_ids.len() {
        let scenario_values: Vec<f64> = model.scenarios.iter().map(|scenario| {
            model.joint_prior.iter().enumerate().map(|(joint_index, p)| {
                let hypothesis = joint_index / states;
                p * scenario.utility_by_hypothesis[hypothesis][action]
            }).sum()
        }).collect();
        let worst = scenario_values.iter().copied().fold(f64::INFINITY, f64::min);
        let replace = match best {
            None => true,
            Some((_, old)) if worst > old + EPS => true,
            Some((old_index, old)) if (worst - old).abs() <= EPS =>
                model.action_ids[action] < model.action_ids[old_index],
            _ => false,
        };
        if replace {
            best = Some((action, worst));
        }
    }
    best.ok_or(Error::InvalidActionSet)
}

fn device_index(model: &Model, candidate: &Candidate) -> usize {
    model.device_ids.iter().position(|id| id == &candidate.device_id)
        .expect("candidate device was validated")
}

fn transition_probability(
    scenario: &Scenario,
    device_count: usize,
    previous_code: usize,
    next_code: usize,
) -> f64 {
    let mut probability = 1.0;
    for device in 0..device_count {
        let previous = (previous_code >> device) & 1;
        let next = (next_code >> device) & 1;
        probability *= scenario.transition_by_device[device][previous][next];
    }
    probability
}

fn scenario_policy_value(
    model: &Model,
    candidates: &[Candidate],
    scenario_index: usize,
    first_index: usize,
    second_by_first: [usize; OUTCOMES],
    actions_by_pair: [usize; LEAVES],
) -> f64 {
    let scenario = &model.scenarios[scenario_index];
    let d_count = model.device_ids.len();
    let states = state_count(d_count);
    let first = &candidates[first_index];
    let first_device = device_index(model, first);
    let first_matrix = &first.likelihoods_by_scenario[scenario_index];
    let mut total = 0.0;

    for (joint_index, prior) in model.joint_prior.iter().enumerate() {
        let hypothesis = joint_index / states;
        let previous_code = joint_index % states;
        let first_state = (previous_code >> first_device) & 1;

        for first_outcome in 0..OUTCOMES {
            let p_first = first_matrix[hypothesis * STATES + first_state][first_outcome];
            if p_first <= 0.0 || *prior <= 0.0 {
                continue;
            }
            let second = &candidates[second_by_first[first_outcome]];
            let second_device = device_index(model, second);
            let second_matrix = &second.likelihoods_by_scenario[scenario_index];

            for next_code in 0..states {
                let p_transition = transition_probability(scenario, d_count, previous_code, next_code);
                if p_transition <= 0.0 {
                    continue;
                }
                let second_state = (next_code >> second_device) & 1;
                for second_outcome in 0..OUTCOMES {
                    let p_second = second_matrix[hypothesis * STATES + second_state][second_outcome];
                    let action = actions_by_pair[first_outcome * OUTCOMES + second_outcome];
                    total += prior * p_first * p_transition * p_second
                        * scenario.utility_by_hypothesis[hypothesis][action];
                }
            }
        }
    }
    total
}

fn first_outcome_probability(
    model: &Model,
    candidate: &Candidate,
    scenario_index: usize,
    outcome: usize,
) -> f64 {
    let states = state_count(model.device_ids.len());
    let device = device_index(model, candidate);
    let matrix = &candidate.likelihoods_by_scenario[scenario_index];
    model.joint_prior.iter().enumerate().map(|(joint_index, prior)| {
        let hypothesis = joint_index / states;
        let code = joint_index % states;
        let device_state = (code >> device) & 1;
        prior * matrix[hypothesis * STATES + device_state][outcome]
    }).sum()
}

/// Evaluate a frozen two-stage policy map against one complete scenario.
/// This is exposed for analytic controls and independent benchmark adapters.
pub fn evaluate_fixed_policy(
    model: &Model,
    candidates: &[Candidate],
    scenario_index: usize,
    first_index: usize,
    second_by_first: [usize; OUTCOMES],
    actions_by_pair: [usize; LEAVES],
) -> Result<f64, Error> {
    validate(model, candidates)?;
    if scenario_index >= model.scenarios.len()
        || first_index >= candidates.len()
        || second_by_first.iter().any(|i| *i >= candidates.len())
        || actions_by_pair.iter().any(|a| *a >= model.action_ids.len())
    {
        return Err(Error::InvalidCandidate);
    }
    Ok(scenario_policy_value(
        model, candidates, scenario_index, first_index, second_by_first, actions_by_pair,
    ))
}

/// Exact two-stage maximin policy. Both first-observation branches must fit
/// the hard budget. Returns None when no experiment improves the robust
/// no-experiment action value.
pub fn select(
    model: &Model,
    candidates: &[Candidate],
    budget_units: u64,
) -> Result<Option<Policy>, Error> {
    validate(model, candidates)?;
    let (baseline_action, baseline_value) = robust_baseline(model)?;
    if budget_units == 0 {
        return Ok(None);
    }

    let action_count = model.action_ids.len();
    let action_policy_count = action_count.checked_pow(LEAVES as u32)
        .ok_or(Error::SearchSpaceTooLarge)?;
    if action_policy_count > 4096 {
        return Err(Error::SearchSpaceTooLarge);
    }

    let mut best: Option<(
        usize, [usize; OUTCOMES], [usize; LEAVES], Vec<f64>, f64, Vec<f64>, u64,
    )> = None;

    for first_index in 0..candidates.len() {
        for second0 in 0..candidates.len() {
            for second1 in 0..candidates.len() {
                let first_cost = candidates[first_index].cost_units;
                let path0 = first_cost.checked_add(candidates[second0].cost_units)
                    .ok_or(Error::NoFeasiblePolicy)?;
                let path1 = first_cost.checked_add(candidates[second1].cost_units)
                    .ok_or(Error::NoFeasiblePolicy)?;
                if path0 > budget_units || path1 > budget_units {
                    continue;
                }
                let second_by_first = [second0, second1];
                let max_path = path0.max(path1);

                for action_code in 0..action_policy_count {
                    let mut code = action_code;
                    let mut actions = [0usize; LEAVES];
                    for action in &mut actions {
                        *action = code % action_count;
                        code /= action_count;
                    }
                    let scenario_values: Vec<f64> = (0..model.scenarios.len())
                        .map(|scenario_index| scenario_policy_value(
                            model, candidates, scenario_index, first_index,
                            second_by_first, actions,
                        ))
                        .collect();
                    let worst = scenario_values.iter().copied().fold(f64::INFINITY, f64::min);
                    let expected_costs: Vec<f64> = model.scenarios.iter().enumerate().map(|(s, _)| {
                        first_cost as f64
                            + first_outcome_probability(model, &candidates[first_index], s, 0)
                                * candidates[second0].cost_units as f64
                            + first_outcome_probability(model, &candidates[first_index], s, 1)
                                * candidates[second1].cost_units as f64
                    }).collect();

                    let key = (first_index, second0, second1, actions);
                    let replace = match &best {
                        None => true,
                        Some((old_first, old_second, old_actions, _, old_worst, _, _))
                            if worst > *old_worst + EPS => true,
                        Some((old_first, old_second, old_actions, _, old_worst, _, _))
                            if (worst - *old_worst).abs() <= EPS =>
                        {
                            key < (*old_first, old_second[0], old_second[1], *old_actions)
                        }
                        _ => false,
                    };
                    if replace {
                        best = Some((
                            first_index, second_by_first, actions, scenario_values,
                            worst, expected_costs, max_path,
                        ));
                    }
                }
            }
        }
    }

    let Some((first_index, second, actions, values, worst, costs, max_path)) = best else {
        return Err(Error::NoFeasiblePolicy);
    };
    if !worst.is_finite()
        || values.iter().chain(costs.iter()).any(|value| !value.is_finite())
    {
        return Err(Error::NumericalInconsistency);
    }
    if worst <= baseline_value + EPS {
        return Ok(None);
    }

    Ok(Some(Policy {
        first_experiment_id: candidates[first_index].experiment_id.clone(),
        second_experiments_by_first_outcome: second.map(|i| candidates[i].experiment_id.clone()),
        actions_by_outcome_pair: actions.map(|i| model.action_ids[i].clone()),
        scenario_values: values,
        worst_case_value: worst,
        baseline_action_id: model.action_ids[baseline_action].clone(),
        baseline_worst_case_value: baseline_value,
        robust_gain: worst - baseline_value,
        expected_cost_by_scenario: costs,
        maximum_path_cost: max_path,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Model, Vec<Candidate>) {
        let devices = vec![
            String::from("instrument-a"),
            String::from("instrument-b"),
            String::from("instrument-c"),
        ];
        let mut prior = vec![0.0; 2 * state_count(devices.len())];
        for hypothesis in 0..2 {
            for code in 0..state_count(devices.len()) {
                let state_b = (code >> 1) & 1;
                let state_c = (code >> 2) & 1;
                let p_b = if state_b == 0 { 0.9 } else { 0.1 };
                let p_c = if state_c == 0 { 0.9 } else { 0.1 };
                prior[hypothesis * state_count(devices.len()) + code] = 0.5 * 0.5 * p_b * p_c;
            }
        }

        let transition = |flip_a: f64| vec![
            [[1.0 - flip_a, flip_a], [flip_a, 1.0 - flip_a]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ];
        let scenarios = vec![
            Scenario {
                id: String::from("nominal-low-drift"),
                utility_by_hypothesis: vec![
                    vec![1.0, 0.0, 0.6],
                    vec![0.0, 1.0, 0.6],
                ],
                transition_by_device: transition(0.02),
            },
            Scenario {
                id: String::from("shifted-high-drift"),
                utility_by_hypothesis: vec![
                    vec![1.0, 0.0, 0.6],
                    vec![0.0, 1.0, 0.6],
                ],
                transition_by_device: transition(0.25),
            },
        ];

        let likelihood = |role: ExperimentRole, accuracy: f64| -> Vec<[f64; OUTCOMES]> {
            let mut rows = Vec::new();
            for hypothesis in 0..2 {
                for instrument_state in 0..STATES {
                    let truth = match role {
                        ExperimentRole::Calibration => instrument_state,
                        ExperimentRole::Science => hypothesis ^ instrument_state,
                    };
                    let mut row = [0.0; OUTCOMES];
                    row[truth] = accuracy;
                    row[1 - truth] = 1.0 - accuracy;
                    rows.push(row);
                }
            }
            rows
        };
        let candidate = |id: &str, device: &str, role: ExperimentRole, accuracy: f64| Candidate {
            experiment_id: String::from(id),
            device_id: String::from(device),
            role,
            cost_units: 1,
            likelihoods_by_scenario: vec![
                likelihood(role, accuracy),
                likelihood(role, accuracy),
            ],
        };

        let candidates = vec![
            candidate("calibrate-a", "instrument-a", ExperimentRole::Calibration, 0.95),
            candidate("science-a", "instrument-a", ExperimentRole::Science, 0.95),
            candidate("science-b", "instrument-b", ExperimentRole::Science, 0.80),
            candidate("science-c", "instrument-c", ExperimentRole::Science, 0.80),
        ];
        (
            Model {
                hypothesis_count: 2,
                device_ids: devices,
                joint_prior: prior,
                action_ids: vec![
                    String::from("choose-h0"),
                    String::from("choose-h1"),
                    String::from("abstain"),
                ],
                scenarios,
            },
            candidates,
        )
    }

    #[test]
    fn robust_policy_prefers_independent_devices_over_stale_calibration() {
        let (model, candidates) = fixture();
        let policy = select(&model, &candidates, 2).unwrap().unwrap();
        assert_eq!(policy.first_experiment_id, "science-b");
        assert_eq!(policy.second_experiments_by_first_outcome, [
            String::from("science-c"), String::from("science-c")
        ]);
        assert_eq!(policy.actions_by_outcome_pair, [
            String::from("choose-h0"), String::from("abstain"),
            String::from("abstain"), String::from("choose-h1")
        ]);
        assert!((policy.worst_case_value - 0.77848).abs() < 1e-12);
        assert!((policy.baseline_worst_case_value - 0.60).abs() < 1e-12);
        assert!((policy.robust_gain - 0.17848).abs() < 1e-12);
        assert_eq!(policy.maximum_path_cost, 2);
    }

    #[test]
    fn independent_device_replication_beats_same_device_repeat() {
        let (model, candidates) = fixture();
        let actions = [0usize, 2usize, 2usize, 1usize];
        let same_device = evaluate_fixed_policy(&model, &candidates, 0, 2, [2, 2], actions).unwrap();
        let independent_devices = evaluate_fixed_policy(&model, &candidates, 0, 2, [3, 3], actions).unwrap();
        assert!((same_device - 0.772).abs() < 1e-12);
        assert!((independent_devices - 0.77848).abs() < 1e-12);
        assert!(independent_devices > same_device);
    }

    #[test]
    fn calibration_value_decays_when_the_instrument_drifts() {
        let (model, candidates) = fixture();
        let actions = [0usize, 1usize, 1usize, 0usize];
        let nominal = evaluate_fixed_policy(&model, &candidates, 0, 0, [1, 1], actions).unwrap();
        let shifted = evaluate_fixed_policy(&model, &candidates, 1, 0, [1, 1], actions).unwrap();
        assert!((nominal - 0.8888).abs() < 1e-12);
        assert!((shifted - 0.7025).abs() < 1e-12);
        assert!(shifted < nominal);
    }

    #[test]
    fn calibration_likelihood_must_not_leak_the_scientific_hypothesis() {
        let (model, mut candidates) = fixture();
        candidates[0].likelihoods_by_scenario[0][2] = [0.7, 0.3];
        assert_eq!(select(&model, &candidates, 2), Err(Error::CalibrationLeaksHypothesis));
    }

    #[test]
    fn every_observable_branch_must_fit_the_hard_budget() {
        let (model, candidates) = fixture();
        assert_eq!(select(&model, &candidates, 1), Err(Error::NoFeasiblePolicy));
    }

    #[test]
    fn invalid_instrument_transition_is_rejected() {
        let (mut model, candidates) = fixture();
        model.scenarios[1].transition_by_device[0][0] = [0.8, 0.8];
        assert_eq!(select(&model, &candidates, 2), Err(Error::InvalidTransitionMatrix));
    }
}
