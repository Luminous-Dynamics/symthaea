//! Exact two-stage experiment design with shared common-cause measurement error.
//!
//! Build:
//!   rustc --edition=2024 --test scripts/research/sdcb_common_cause_measurement.rs -o /tmp/sdcb-common-cause
//! Run:
//!   /tmp/sdcb-common-cause
//!
//! The latent state contains one shared binary environment/reference state g
//! plus one binary local state per registered device. A Markov transition for
//! g couples all device channels. Device-local transitions are conditionally
//! independent only after conditioning on (g_old, g_new, c_device_old).
//! Experiment outcomes depend on h, g, and the named device's local state.
//!
//! This bounded reference is not a production metrology model. It supports
//! binary states/outcomes and exactly two stages; it does not infer the model
//! or authenticate physical devices and calibration evidence.

const B: usize = 2;
const OUTCOMES: usize = 2;
const LEAVES: usize = 4;
const TOL: f64 = 1e-9;
const EPS: f64 = 1e-12;
const MAX_HYPOTHESES: usize = 4;
const MAX_DEVICES: usize = 3;
const MAX_SCENARIOS: usize = 4;
const MAX_ACTIONS: usize = 3;
const MAX_CANDIDATES: usize = 3;
type LocalTransition = [[[[f64; B]; B]; B]; B]; // old_g, new_g, old_local, new_local

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    SharedReferenceCalibration,
    ScientificMeasurement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidDimensions,
    InvalidPrior,
    InvalidScenario,
    InvalidActions,
    InvalidCandidate,
    InvalidLikelihoods,
    CalibrationLeaksHypothesis,
    DuplicateIdentifier,
    InvalidCommonTransition,
    InvalidLocalTransition,
    SearchSpaceTooLarge,
    NoFeasiblePolicy,
    Numerical,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scenario {
    pub scenario_id: String,
    /// Rows are scientific hypotheses; columns are terminal actions.
    pub utility_by_hypothesis: Vec<Vec<f64>>,
    /// Shared environment/reference-state Markov transition.
    /// Rows are old g; columns are new g.
    pub common_transition: [[f64; B]; B],
    /// One device-local transition tensor per named device:
    /// [old_g][new_g][old_local_state][new_local_state].
    pub local_transition_by_device: Vec<LocalTransition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub hypothesis_count: usize,
    pub device_ids: Vec<String>,
    /// Joint prior, hypothesis-major. Instrument code uses bit 0 for g
    /// and bit (device_index + 1) for that device's local state.
    pub joint_prior: Vec<f64>,
    pub action_ids: Vec<String>,
    pub scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub experiment_id: String,
    pub device_id: String,
    pub role: Role,
    pub cost_units: u64,
    /// Scenario -> hypothesis * 4 + common_state * 2 + local_device_state
    /// -> P(outcome 0), P(outcome 1).
    pub likelihoods_by_scenario: Vec<Vec<[f64; OUTCOMES]>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    pub first_experiment_id: String,
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

fn validate_probabilities(values: &[f64]) -> bool {
    !values.is_empty()
        && values.iter().all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0)
        && (values.iter().sum::<f64>() - 1.0).abs() <= TOL
}

fn instrument_state_count(device_count: usize) -> usize {
    1usize << (device_count + 1)
}

fn common_state(code: usize) -> usize {
    code & 1
}

fn local_state(code: usize, device: usize) -> usize {
    (code >> (device + 1)) & 1
}

fn validate(model: &Model, candidates: &[Candidate]) -> Result<(), Error> {
    let h = model.hypothesis_count;
    let d = model.device_ids.len();
    if !(2..=MAX_HYPOTHESES).contains(&h)
        || d == 0 || d > MAX_DEVICES
        || model.device_ids.iter().any(|id| id.trim().is_empty())
        || (0..d).any(|i| model.device_ids[..i].contains(&model.device_ids[i]))
    {
        return Err(Error::InvalidDimensions);
    }
    let states = instrument_state_count(d);
    if model.joint_prior.len() != h * states || !validate_probabilities(&model.joint_prior) {
        return Err(Error::InvalidPrior);
    }

    let action_count = model.action_ids.len();
    if !(2..=MAX_ACTIONS).contains(&action_count)
        || model.action_ids.iter().any(|id| id.trim().is_empty())
        || (0..action_count).any(|i| model.action_ids[..i].contains(&model.action_ids[i]))
    {
        return Err(Error::InvalidActions);
    }
    if model.scenarios.is_empty() || model.scenarios.len() > MAX_SCENARIOS {
        return Err(Error::InvalidScenario);
    }
    for (index, scenario) in model.scenarios.iter().enumerate() {
        if scenario.scenario_id.trim().is_empty()
            || model.scenarios[..index].iter().any(|old| old.scenario_id == scenario.scenario_id)
            || scenario.utility_by_hypothesis.len() != h
            || scenario.utility_by_hypothesis.iter().any(|row| {
                row.len() != action_count || row.iter().any(|u| !u.is_finite())
            })
            || scenario.local_transition_by_device.len() != d
        {
            return Err(Error::InvalidScenario);
        }
        if scenario.common_transition.iter().any(|row| {
            row.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                || (row.iter().sum::<f64>() - 1.0).abs() > TOL
        }) {
            return Err(Error::InvalidCommonTransition);
        }
        for tensor in &scenario.local_transition_by_device {
            for old_g in 0..B {
                for new_g in 0..B {
                    for old_local in 0..B {
                        let row = tensor[old_g][new_g][old_local];
                        if row.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                            || (row.iter().sum::<f64>() - 1.0).abs() > TOL
                        {
                            return Err(Error::InvalidLocalTransition);
                        }
                    }
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
                matrix.len() != h * 4
                    || matrix.iter().any(|row| {
                        row.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
                            || (row.iter().sum::<f64>() - 1.0).abs() > TOL
                    })
            })
        {
            return Err(Error::InvalidLikelihoods);
        }
        if candidate.role == Role::SharedReferenceCalibration {
            for matrix in &candidate.likelihoods_by_scenario {
                for common in 0..B {
                    for local in 0..B {
                        let row_index = common * B + local;
                        let reference = matrix[row_index];
                        for hypothesis in 1..h {
                            if (0..OUTCOMES).any(|outcome| {
                                (matrix[hypothesis * 4 + row_index][outcome] - reference[outcome]).abs() > TOL
                            }) {
                                return Err(Error::CalibrationLeaksHypothesis);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn candidate_device(model: &Model, candidate: &Candidate) -> usize {
    model.device_ids.iter().position(|id| id == &candidate.device_id)
        .expect("candidate device validated")
}

fn transition_probability(
    scenario: &Scenario,
    device_count: usize,
    old_code: usize,
    new_code: usize,
) -> f64 {
    let old_g = common_state(old_code);
    let new_g = common_state(new_code);
    let mut probability = scenario.common_transition[old_g][new_g];
    for device in 0..device_count {
        let old_local = local_state(old_code, device);
        let new_local = local_state(new_code, device);
        probability *= scenario.local_transition_by_device[device]
            [old_g][new_g][old_local][new_local];
    }
    probability
}

fn baseline(model: &Model) -> Result<(usize, f64), Error> {
    let states = instrument_state_count(model.device_ids.len());
    let mut best: Option<(usize, f64)> = None;
    for action in 0..model.action_ids.len() {
        let worst = model.scenarios.iter().map(|scenario| {
            model.joint_prior.iter().enumerate().map(|(index, p)| {
                let h = index / states;
                *p * scenario.utility_by_hypothesis[h][action]
            }).sum::<f64>()
        }).fold(f64::INFINITY, f64::min);
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
    best.ok_or(Error::InvalidActions)
}

fn scenario_value(
    model: &Model,
    candidates: &[Candidate],
    scenario_index: usize,
    first_index: usize,
    second: [usize; OUTCOMES],
    actions: [usize; LEAVES],
) -> f64 {
    let scenario = &model.scenarios[scenario_index];
    let devices = model.device_ids.len();
    let states = instrument_state_count(devices);
    let first = &candidates[first_index];
    let first_device = candidate_device(model, first);
    let first_likelihoods = &first.likelihoods_by_scenario[scenario_index];
    let mut total = 0.0;

    for (joint_index, prior) in model.joint_prior.iter().enumerate() {
        if *prior <= 0.0 { continue; }
        let h = joint_index / states;
        let old_code = joint_index % states;
        let old_g = common_state(old_code);
        let old_local = local_state(old_code, first_device);
        let row = h * 4 + old_g * 2 + old_local;
        for y1 in 0..OUTCOMES {
            let p1 = first_likelihoods[row][y1];
            if p1 <= 0.0 { continue; }
            let second_candidate = &candidates[second[y1]];
            let second_device = candidate_device(model, second_candidate);
            let second_likelihoods = &second_candidate.likelihoods_by_scenario[scenario_index];

            for new_code in 0..states {
                let p_transition = transition_probability(scenario, devices, old_code, new_code);
                if p_transition <= 0.0 { continue; }
                let new_g = common_state(new_code);
                let new_local = local_state(new_code, second_device);
                let second_row = h * 4 + new_g * 2 + new_local;
                for y2 in 0..OUTCOMES {
                    let p2 = second_likelihoods[second_row][y2];
                    let action = actions[y1 * OUTCOMES + y2];
                    total += *prior * p1 * p_transition * p2
                        * scenario.utility_by_hypothesis[h][action];
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
    let states = instrument_state_count(model.device_ids.len());
    let device = candidate_device(model, candidate);
    let matrix = &candidate.likelihoods_by_scenario[scenario_index];
    model.joint_prior.iter().enumerate().map(|(index, prior)| {
        let h = index / states;
        let code = index % states;
        let row = h * 4 + common_state(code) * 2 + local_state(code, device);
        *prior * matrix[row][outcome]
    }).sum()
}

pub fn evaluate_fixed_policy(
    model: &Model,
    candidates: &[Candidate],
    scenario_index: usize,
    first: usize,
    second: [usize; OUTCOMES],
    actions: [usize; LEAVES],
) -> Result<f64, Error> {
    validate(model, candidates)?;
    if scenario_index >= model.scenarios.len()
        || first >= candidates.len()
        || second.iter().any(|i| *i >= candidates.len())
        || actions.iter().any(|a| *a >= model.action_ids.len())
    {
        return Err(Error::InvalidCandidate);
    }
    Ok(scenario_value(model, candidates, scenario_index, first, second, actions))
}

/// Exact two-stage maximin policy over a small joint common-cause/device state.
pub fn select(
    model: &Model,
    candidates: &[Candidate],
    budget_units: u64,
) -> Result<Option<Policy>, Error> {
    validate(model, candidates)?;
    let (baseline_action, baseline_value) = baseline(model)?;
    if budget_units == 0 { return Ok(None); }

    let candidate_count = candidates.len();
    let action_count = model.action_ids.len();
    let action_maps = action_count.checked_pow(LEAVES as u32).ok_or(Error::SearchSpaceTooLarge)?;
    if action_maps > 4096 { return Err(Error::SearchSpaceTooLarge); }

    let mut best: Option<(
        usize, [usize; OUTCOMES], [usize; LEAVES], Vec<f64>, f64, Vec<f64>, u64,
    )> = None;

    for first in 0..candidate_count {
        for second0 in 0..candidate_count {
            for second1 in 0..candidate_count {
                let cost0 = candidates[first].cost_units.checked_add(candidates[second0].cost_units)
                    .ok_or(Error::NoFeasiblePolicy)?;
                let cost1 = candidates[first].cost_units.checked_add(candidates[second1].cost_units)
                    .ok_or(Error::NoFeasiblePolicy)?;
                if cost0 > budget_units || cost1 > budget_units { continue; }
                let second = [second0, second1];
                let maximum_path = cost0.max(cost1);

                for action_code in 0..action_maps {
                    let mut code = action_code;
                    let mut actions = [0usize; LEAVES];
                    for action in &mut actions {
                        *action = code % action_count;
                        code /= action_count;
                    }
                    let values: Vec<f64> = (0..model.scenarios.len()).map(|s| {
                        scenario_value(model, candidates, s, first, second, actions)
                    }).collect();
                    let worst = values.iter().copied().fold(f64::INFINITY, f64::min);
                    let costs: Vec<f64> = (0..model.scenarios.len()).map(|s| {
                        candidates[first].cost_units as f64
                            + first_outcome_probability(model, &candidates[first], s, 0)
                                * candidates[second0].cost_units as f64
                            + first_outcome_probability(model, &candidates[first], s, 1)
                                * candidates[second1].cost_units as f64
                    }).collect();
                    let key = (first, second, actions);
                    let replace = match &best {
                        None => true,
                        Some((of, os, oa, _, old, _, _)) if worst > *old + EPS => true,
                        Some((of, os, oa, _, old, _, _)) if (worst - *old).abs() <= EPS =>
                            key < (*of, *os, *oa),
                        _ => false,
                    };
                    if replace {
                        best = Some((first, second, actions, values, worst, costs, maximum_path));
                    }
                }
            }
        }
    }

    let Some((first, second, actions, values, worst, costs, maximum_path)) = best else {
        return Err(Error::NoFeasiblePolicy);
    };
    if !worst.is_finite() || values.iter().chain(costs.iter()).any(|v| !v.is_finite()) {
        return Err(Error::Numerical);
    }
    if worst <= baseline_value + EPS { return Ok(None); }

    Ok(Some(Policy {
        first_experiment_id: candidates[first].experiment_id.clone(),
        second_experiments_by_first_outcome: second.map(|i| candidates[i].experiment_id.clone()),
        actions_by_outcome_pair: actions.map(|i| model.action_ids[i].clone()),
        scenario_values: values,
        worst_case_value: worst,
        baseline_action_id: model.action_ids[baseline_action].clone(),
        baseline_worst_case_value: baseline_value,
        robust_gain: worst - baseline_value,
        expected_cost_by_scenario: costs,
        maximum_path_cost: maximum_path,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_identity() -> LocalTransition {
        let mut tensor = [[[[0.0; B]; B]; B]; B];
        for old_g in 0..B {
            for new_g in 0..B {
                for old_local in 0..B {
                    tensor[old_g][new_g][old_local][old_local] = 1.0;
                }
            }
        }
        tensor
    }

    fn best_terminal_value(
        model: &Model,
        candidates: &[Candidate],
        scenario: usize,
        first: usize,
        second: [usize; OUTCOMES],
    ) -> f64 {
        let mut best = f64::NEG_INFINITY;
        for mut code in 0..2usize.pow(LEAVES as u32) {
            let mut actions = [0usize; LEAVES];
            for action in &mut actions {
                *action = code % 2;
                code /= 2;
            }
            let value = evaluate_fixed_policy(model, candidates, scenario, first, second, actions).unwrap();
            best = best.max(value);
        }
        best
    }

    fn fixture() -> (Model, Vec<Candidate>) {
        let devices = vec![
            String::from("science-a"),
            String::from("science-b"),
            String::from("shared-reference"),
        ];
        let state_count = instrument_state_count(devices.len());
        let mut prior = vec![0.0; 2 * state_count];
        for h in 0..2 {
            for g in 0..2 {
                let code = g; // local device states are known to be zero
                prior[h * state_count + code] = 0.25;
            }
        }
        let scenarios = vec![
            Scenario {
                scenario_id: String::from("stable-common-cause"),
                utility_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                common_transition: [[1.0, 0.0], [0.0, 1.0]],
                local_transition_by_device: vec![local_identity(); 3],
            },
            Scenario {
                scenario_id: String::from("common-cause-drift"),
                utility_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                common_transition: [[0.8, 0.2], [0.2, 0.8]],
                local_transition_by_device: vec![local_identity(); 3],
            },
        ];

        let likelihood = |role: Role, accuracy: f64| {
            let mut rows = vec![[0.0; OUTCOMES]; 2 * 4];
            for h in 0..2 {
                for g in 0..2 {
                    for local in 0..2 {
                        let truth = match role {
                            Role::SharedReferenceCalibration => g ^ local,
                            Role::ScientificMeasurement => h ^ g ^ local,
                        };
                        let row = h * 4 + g * 2 + local;
                        rows[row][truth] = accuracy;
                        rows[row][1 - truth] = 1.0 - accuracy;
                    }
                }
            }
            rows
        };
        let make_candidate = |id: &str, device: &str, role: Role| Candidate {
            experiment_id: id.to_string(),
            device_id: device.to_string(),
            role,
            cost_units: 1,
            likelihoods_by_scenario: vec![
                likelihood(role, 0.95),
                likelihood(role, 0.95),
            ],
        };
        let candidates = vec![
            make_candidate("shared-reference-calibration", "shared-reference", Role::SharedReferenceCalibration),
            make_candidate("science-a", "science-a", Role::ScientificMeasurement),
            make_candidate("science-b", "science-b", Role::ScientificMeasurement),
        ];
        (Model {
            hypothesis_count: 2,
            device_ids: devices,
            joint_prior: prior,
            action_ids: vec![String::from("choose-h0"), String::from("choose-h1")],
            scenarios,
        }, candidates)
    }

    #[test]
    fn shared_reference_then_science_is_the_robust_policy() {
        let (model, candidates) = fixture();
        let policy = select(&model, &candidates, 2).unwrap().unwrap();
        assert_eq!(policy.first_experiment_id, "shared-reference-calibration");
        assert_eq!(policy.second_experiments_by_first_outcome, [
            String::from("science-a"), String::from("science-a")
        ]);
        assert_eq!(policy.actions_by_outcome_pair, [
            String::from("choose-h0"), String::from("choose-h1"),
            String::from("choose-h1"), String::from("choose-h0")
        ]);
        assert!((policy.scenario_values[0] - 0.905).abs() < 1e-12);
        assert!((policy.scenario_values[1] - 0.743).abs() < 1e-12);
        assert!((policy.worst_case_value - 0.743).abs() < 1e-12);
        assert!((policy.baseline_worst_case_value - 0.5).abs() < 1e-12);
        assert!((policy.robust_gain - 0.243).abs() < 1e-12);
        assert_eq!(policy.maximum_path_cost, 2);
    }

    #[test]
    fn same_device_repeat_does_not_resolve_shared_common_cause() {
        let (model, candidates) = fixture();
        let first = 1usize;
        for scenario in 0..2 {
            let best = best_terminal_value(&model, &candidates, scenario, first, [1, 1]);
            assert!((best - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn cross_device_agreement_does_not_resolve_a_shared_common_cause() {
        let (model, candidates) = fixture();
        for scenario in 0..2 {
            let best = best_terminal_value(&model, &candidates, scenario, 1, [1, 2]);
            assert!((best - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn stale_shared_reference_loses_value_when_common_cause_drifts() {
        let (model, candidates) = fixture();
        let actions = [0usize, 1usize, 1usize, 0usize];
        let nominal = evaluate_fixed_policy(&model, &candidates, 0, 0, [1, 1], actions).unwrap();
        let shifted = evaluate_fixed_policy(&model, &candidates, 1, 0, [1, 1], actions).unwrap();
        assert!((nominal - 0.905).abs() < 1e-12);
        assert!((shifted - 0.743).abs() < 1e-12);
        assert!(shifted < nominal);
    }

    #[test]
    fn calibration_likelihood_cannot_leak_scientific_hypothesis() {
        let (model, mut candidates) = fixture();
        candidates[0].likelihoods_by_scenario[0][4] = [0.7, 0.3];
        assert_eq!(select(&model, &candidates, 2), Err(Error::CalibrationLeaksHypothesis));
    }

    #[test]
    fn common_transition_rows_must_normalize() {
        let (mut model, candidates) = fixture();
        model.scenarios[1].common_transition[0] = [0.8, 0.8];
        assert_eq!(select(&model, &candidates, 2), Err(Error::InvalidCommonTransition));
    }

    #[test]
    fn local_transition_rows_must_normalize() {
        let (mut model, candidates) = fixture();
        model.scenarios[1].local_transition_by_device[0][0][1][0] = [0.5, 0.5];
        assert_eq!(select(&model, &candidates, 2), Err(Error::InvalidLocalTransition));
    }

    #[test]
    fn every_contingent_branch_must_obey_the_hard_budget() {
        let (model, candidates) = fixture();
        assert_eq!(select(&model, &candidates, 1), Err(Error::NoFeasiblePolicy));
    }
}
