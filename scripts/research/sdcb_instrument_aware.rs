//! Exact two-stage experiment design with a persistent latent instrument state.
//!
//! Build: rustc --edition=2024 --test scripts/research/sdcb_instrument_aware.rs -o /tmp/sdcb-instrument-aware
//! Run:   /tmp/sdcb-instrument-aware
//!
//! Bounded finite reference only: binary outcomes, two stages, fixed instrument
//! state. Observations are conditionally independent given joint state/scenario;
//! repeated observations can still be dependent marginally through that state.

const O: usize = 2;
const LEAVES: usize = O * O;
const TOL: f64 = 1e-9;
const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role { Science, Calibration }

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Dimensions, Prior, Scenario, Actions, Experiment, Likelihood,
    CalibrationLeaksHypothesis, DuplicateId, NoFeasiblePolicy, SearchTooLarge,
    Numerical,
}

#[derive(Debug, Clone)]
pub struct Scenario {
    pub id: String,
    /// Rows are scientific hypotheses; columns are terminal actions.
    pub utility: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub hypotheses: usize,
    pub instrument_states: usize,
    /// Hypothesis-major joint prior: h * instrument_states + c.
    pub joint_prior: Vec<f64>,
    pub actions: Vec<String>,
    /// Complete ambiguity scenarios are not assigned probabilities.
    pub scenarios: Vec<Scenario>,
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: String,
    pub role: Role,
    pub cost: u64,
    /// Scenario -> joint latent state -> P(outcome 0), P(outcome 1).
    pub likelihoods: Vec<Vec<[f64; O]>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Policy {
    pub first: String,
    /// One second-stage experiment per first-stage outcome [0, 1].
    pub second_by_first: [String; O],
    /// Terminal actions indexed by first_outcome * 2 + second_outcome.
    pub action_by_pair: [String; LEAVES],
    pub scenario_values: Vec<f64>,
    pub worst_case_value: f64,
    pub baseline_action: String,
    pub baseline_worst_case: f64,
    pub robust_gain: f64,
    pub expected_cost_by_scenario: Vec<f64>,
    pub maximum_path_cost: u64,
}

fn validate(model: &Model, candidates: &[Candidate]) -> Result<(), Error> {
    let (h, c) = (model.hypotheses, model.instrument_states);
    let joint = h.checked_mul(c).ok_or(Error::Dimensions)?;
    if h == 0 || h > 4 || !(2..=4).contains(&c) || joint > 8 || model.joint_prior.len() != joint {
        return Err(Error::Dimensions);
    }
    let ps = &model.joint_prior;
    if ps.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || (ps.iter().sum::<f64>() - 1.0).abs() > TOL { return Err(Error::Prior); }
    if model.actions.is_empty() || model.actions.len() > 3
        || model.actions.iter().any(|s| s.trim().is_empty())
        || (0..model.actions.len()).any(|i| model.actions[..i].contains(&model.actions[i])) {
        return Err(Error::Actions);
    }
    if model.scenarios.is_empty() || model.scenarios.len() > 4 { return Err(Error::Scenario); }
    for (i, s) in model.scenarios.iter().enumerate() {
        if s.id.trim().is_empty() || model.scenarios[..i].iter().any(|p| p.id == s.id)
            || s.utility.len() != h
            || s.utility.iter().any(|r| r.len() != model.actions.len() || r.iter().any(|v| !v.is_finite())) {
            return Err(Error::Scenario);
        }
    }
    if !(2..=3).contains(&candidates.len()) { return Err(Error::SearchTooLarge); }
    for (i, e) in candidates.iter().enumerate() {
        if e.id.trim().is_empty() || e.cost == 0 || candidates[..i].iter().any(|p| p.id == e.id) {
            return Err(if e.id.trim().is_empty() || e.cost == 0 { Error::Experiment } else { Error::DuplicateId });
        }
        if e.likelihoods.len() != model.scenarios.len()
            || e.likelihoods.iter().any(|m| m.len() != joint
                || m.iter().any(|r| r.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
                    || (r.iter().sum::<f64>() - 1.0).abs() > TOL)) {
            return Err(Error::Likelihood);
        }
        if e.role == Role::Calibration {
            for m in &e.likelihoods {
                for instrument in 0..c {
                    let reference = m[instrument];
                    for hypothesis in 1..h {
                        if (0..O).any(|y| (m[hypothesis * c + instrument][y] - reference[y]).abs() > TOL) {
                            return Err(Error::CalibrationLeaksHypothesis);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn baseline(model: &Model) -> Result<(usize, f64), Error> {
    let mut best: Option<(usize, f64)> = None;
    for a in 0..model.actions.len() {
        let worst = model.scenarios.iter().map(|s| {
            model.joint_prior.iter().enumerate().map(|(j, p)| {
                *p * s.utility[j / model.instrument_states][a]
            }).sum::<f64>()
        }).fold(f64::INFINITY, f64::min);
        let replace = match best {
            None => true,
            Some((_, v)) if worst > v + EPS => true,
            Some((old, v)) if (worst - v).abs() <= EPS => model.actions[a] < model.actions[old],
            _ => false,
        };
        if replace { best = Some((a, worst)); }
    }
    best.ok_or(Error::Actions)
}

fn decode(mut code: usize, base: usize, len: usize) -> Vec<usize> {
    let mut out = vec![0; len];
    for x in &mut out { *x = code % base; code /= base; }
    out
}

fn value(model: &Model, candidates: &[Candidate], scenario: usize, first: usize,
         second: &[usize; O], actions: &[usize; LEAVES]) -> f64 {
    let mut total = 0.0;
    let c = model.instrument_states;
    for (j, prior) in model.joint_prior.iter().enumerate() {
        let h = j / c;
        let p1 = &candidates[first].likelihoods[scenario][j];
        for y1 in 0..O {
            let p2 = &candidates[second[y1]].likelihoods[scenario][j];
            for y2 in 0..O {
                total += *prior * p1[y1] * p2[y2]
                    * model.scenarios[scenario].utility[h][actions[y1 * O + y2]];
            }
        }
    }
    total
}

fn expected_cost(model: &Model, candidates: &[Candidate], scenario: usize, first: usize,
                 second: &[usize; O]) -> f64 {
    let mut cost = candidates[first].cost as f64;
    for y1 in 0..O {
        let p: f64 = model.joint_prior.iter().enumerate().map(|(j, prior)| {
            *prior * candidates[first].likelihoods[scenario][j][y1]
        }).sum();
        cost += p * candidates[second[y1]].cost as f64;
    }
    cost
}

/// Exact maximin two-stage policy. One policy must serve every ambiguity
/// scenario; every observable branch must obey the hard cumulative budget.
pub fn select(model: &Model, candidates: &[Candidate], budget: u64) -> Result<Policy, Error> {
    validate(model, candidates)?;
    if budget == 0 { return Err(Error::NoFeasiblePolicy); }
    let (base_action, base_value) = baseline(model)?;
    let ne = candidates.len();
    let na = model.actions.len();
    let n_action_maps = na.checked_pow(LEAVES as u32).ok_or(Error::SearchTooLarge)?;
    if n_action_maps > 4096 { return Err(Error::SearchTooLarge); }
    let mut best: Option<(usize, [usize; O], [usize; LEAVES], Vec<f64>, f64, Vec<f64>, u64)> = None;
    for first in 0..ne {
        for code in 0..ne.pow(O as u32) {
            let d = decode(code, ne, O);
            let second = [d[0], d[1]];
            let Some(path0) = candidates[first].cost.checked_add(candidates[second[0]].cost) else { continue };
            let Some(path1) = candidates[first].cost.checked_add(candidates[second[1]].cost) else { continue };
            let paths = [path0, path1];
            if paths.iter().any(|p| *p > budget) { continue; }
            let max_path = *paths.iter().max().ok_or(Error::NoFeasiblePolicy)?;
            for action_code in 0..n_action_maps {
                let d = decode(action_code, na, LEAVES);
                let actions = [d[0], d[1], d[2], d[3]];
                let values: Vec<f64> = (0..model.scenarios.len())
                    .map(|s| value(model, candidates, s, first, &second, &actions)).collect();
                let worst = values.iter().copied().fold(f64::INFINITY, f64::min);
                let costs: Vec<f64> = (0..model.scenarios.len())
                    .map(|s| expected_cost(model, candidates, s, first, &second)).collect();
                let replace = match &best {
                    None => true,
                    Some((_, _, _, _, old, _, _)) if worst > *old + EPS => true,
                    Some((of, os, oa, _, old, _, _)) if (worst - *old).abs() <= EPS =>
                        (first, second, actions) < (*of, *os, *oa),
                    _ => false,
                };
                if replace { best = Some((first, second, actions, values, worst, costs, max_path)); }
            }
        }
    }
    let (first, second, actions, values, worst, costs, max_path) = best.ok_or(Error::NoFeasiblePolicy)?;
    if !worst.is_finite() || values.iter().chain(costs.iter()).any(|x| !x.is_finite())
        || worst + EPS < base_value { return Err(Error::Numerical); }
    Ok(Policy {
        first: candidates[first].id.clone(),
        second_by_first: second.map(|i| candidates[i].id.clone()),
        action_by_pair: actions.map(|i| model.actions[i].clone()),
        scenario_values: values,
        worst_case_value: worst,
        baseline_action: model.actions[base_action].clone(),
        baseline_worst_case: base_value,
        robust_gain: (worst - base_value).max(0.0),
        expected_cost_by_scenario: costs,
        maximum_path_cost: max_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Model, Vec<Candidate>) {
        let model = Model {
            hypotheses: 2, instrument_states: 2, joint_prior: vec![0.25; 4],
            actions: vec!["choose-h0".into(), "choose-h1".into()],
            scenarios: vec![
                Scenario { id: "nominal".into(), utility: vec![vec![1.0, 0.0], vec![0.0, 1.0]] },
                Scenario { id: "shift".into(), utility: vec![vec![1.0, 0.0], vec![0.0, 1.0]] },
            ],
        };
        let likelihood = |role: Role, accuracy: f64| {
            (0..2).flat_map(|h| (0..2).map(move |c| {
                let truth = if role == Role::Calibration { c } else { h ^ c };
                if truth == 0 { [accuracy, 1.0 - accuracy] } else { [1.0 - accuracy, accuracy] }
            })).collect::<Vec<_>>()
        };
        let cal = Candidate { id: "calibration-standard".into(), role: Role::Calibration, cost: 1,
            likelihoods: vec![likelihood(Role::Calibration, 0.95), likelihood(Role::Calibration, 0.90)] };
        let sci = Candidate { id: "science-measurement".into(), role: Role::Science, cost: 1,
            likelihoods: vec![likelihood(Role::Science, 0.95), likelihood(Role::Science, 0.90)] };
        (model, vec![cal, sci])
    }
    #[test]
    fn calibration_then_science_resolves_shared_bias() {
        let (m, c) = fixture();
        let p = select(&m, &c, 2).unwrap();
        assert_eq!(p.first, "calibration-standard");
        assert_eq!(p.second_by_first, [String::from("science-measurement"), String::from("science-measurement")]);
        assert_eq!(p.action_by_pair, [String::from("choose-h0"), String::from("choose-h1"), String::from("choose-h1"), String::from("choose-h0")]);
        assert!((p.scenario_values[0] - 0.905).abs() < 1e-12);
        assert!((p.scenario_values[1] - 0.82).abs() < 1e-12);
        assert!((p.robust_gain - 0.32).abs() < 1e-12);
        assert_eq!(p.maximum_path_cost, 2);
    }
    #[test]
    fn calibration_channel_cannot_depend_on_scientific_hypothesis() {
        let (m, mut c) = fixture();
        c[0].likelihoods[0][2] = [0.7, 0.3];
        assert_eq!(select(&m, &c, 2), Err(Error::CalibrationLeaksHypothesis));
    }
    #[test]
    fn hard_budget_applies_to_every_contingent_branch() {
        let (m, c) = fixture();
        assert_eq!(select(&m, &c, 1), Err(Error::NoFeasiblePolicy));
    }
    #[test]
    fn repeating_science_measurement_does_not_resolve_unknown_xor_offset() {
        let (m, c) = fixture();
        let repeat = [1usize, 1usize];
        let mut best: f64 = 0.0;
        for code in 0..2usize.pow(LEAVES as u32) {
            let d = decode(code, 2, LEAVES);
            let a = [d[0], d[1], d[2], d[3]];
            best = best.max(value(&m, &c, 0, 1, &repeat, &a));
        }
        assert!((best - 0.5).abs() < 1e-12);
    }
    #[test]
    fn invalid_joint_prior_is_rejected() {
        let (mut m, c) = fixture(); m.joint_prior[0] = 0.5;
        assert_eq!(select(&m, &c, 2), Err(Error::Prior));
    }
}
