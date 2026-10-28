//! Fixed-model ambiguity minimax over complete acquisition decision trees.
//! Contrasts with greedy one-step robust EVSI. The latent task is tiny by design: all policy
//! trees up to three queries are enumerated and scored against one fixed model scenario at a time.

use std::{env, process::ExitCode};
type Matrix = [[f64; 3]; 4];
type BeliefSet = [[f64; 3]; 4];
const PRIOR: [f64; 3] = [0.6, 0.3, 0.1];
const LOSS: [[f64; 3]; 3] = [[0., 4., 7.], [5., 0., 3.], [8., 2., 0.]];
const COST: [f64; 4] = [1., 1., 2., 1.];
const QUERY_NAMES: [&str; 4] = ["coarse-shape", "inventory-check", "design-record", "cheap-proxy"];
const NOMINAL: Matrix = [
    [0.88, 0.34, 0.25], [0.35, 0.86, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
];
const Q0_SHIFT: Matrix = [
    [0.88, 0.76, 0.25], [0.35, 0.86, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
];
const Q1_SHIFT: Matrix = [
    [0.88, 0.34, 0.25], [0.83, 0.22, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
];
const COMPOUND: Matrix = [
    [0.88, 0.74, 0.25], [0.79, 0.22, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.83, 0.18],
];
const OUT_OF_SET: Matrix = [
    [0.15, 0.92, 0.25], [0.35, 0.86, 0.31],
    [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
];
const AMBIGUITY: [Matrix; 4] = [NOMINAL, Q0_SHIFT, Q1_SHIFT, COMPOUND];
const SCENARIOS: [(&str, Matrix); 5] = [
    ("iid", NOMINAL), ("q0_conditional_shift", Q0_SHIFT),
    ("q1_asymmetric_error", Q1_SHIFT), ("compound_shift", COMPOUND),
    ("out_of_set_flip", OUT_OF_SET),
];
const EPS: f64 = 1.0e-12;

#[derive(Clone, Debug)]
enum Tree {
    Stop,
    Query { index: usize, zero: Box<Tree>, one: Box<Tree> },
}
impl Tree {
    fn first_query(&self) -> Option<usize> {
        match self { Tree::Stop => None, Tree::Query { index, .. } => Some(*index) }
    }
}
#[derive(Clone)]
struct Plan { risk: [f64; 4], cost: [f64; 4], queries: [f64; 4], tree: Tree }
#[derive(Clone, Copy, Default)]
struct Metrics { loss: f64, regret: f64, cost: f64, queries: f64 }

fn p_observe(matrix: Matrix, q: usize, h: usize, y: bool) -> f64 {
    let p = matrix[q][h];
    if y { p } else { 1. - p }
}
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let sum = v.iter().sum::<f64>();
    if sum <= 0. { PRIOR } else { [v[0]/sum, v[1]/sum, v[2]/sum] }
}
fn posterior(b: [f64; 3], matrix: Matrix, q: usize, y: bool) -> [f64; 3] {
    normalize([
        b[0] * p_observe(matrix, q, 0, y),
        b[1] * p_observe(matrix, q, 1, y),
        b[2] * p_observe(matrix, q, 2, y),
    ])
}
fn action_risk(b: [f64; 3], action: usize) -> f64 {
    (0..3).map(|h| b[h] * LOSS[h][action]).sum()
}
fn best_action(b: [f64; 3]) -> usize {
    let mut best = 0;
    let mut value = action_risk(b, 0);
    for action in 1..3 {
        let candidate = action_risk(b, action);
        if candidate < value - EPS { best = action; value = candidate; }
    }
    best
}
fn available(mask: u8) -> Vec<usize> { (0..4).filter(|q| mask & (1 << q) != 0).collect() }
fn stop_plan(nominal_belief: [f64; 3], scenario_beliefs: BeliefSet) -> Plan {
    let action = best_action(nominal_belief);
    Plan {
        risk: std::array::from_fn(|s| action_risk(scenario_beliefs[s], action)),
        cost: [0.; 4], queries: [0.; 4], tree: Tree::Stop,
    }
}
fn predictive_probability(b: [f64; 3], matrix: Matrix, q: usize, y: bool) -> f64 {
    (0..3).map(|h| b[h] * p_observe(matrix, q, h, y)).sum()
}

/// Enumerate every observation-contingent query tree up to the requested depth.
/// A strategy is one fixed tree shared by all ambiguity scenarios; only scenario probabilities
/// differ when its risk vector is evaluated. This avoids choosing a different adversarial model
/// independently at each branch.
fn enumerate_plans(
    nominal_belief: [f64; 3], scenario_beliefs: BeliefSet, mask: u8, rounds: usize,
) -> Vec<Plan> {
    let mut plans = vec![stop_plan(nominal_belief, scenario_beliefs)];
    if rounds == 0 || mask == 0 { return plans; }

    for query in available(mask) {
        let remainder = mask & !(1 << query);
        let next_nominal_zero = posterior(nominal_belief, NOMINAL, query, false);
        let next_nominal_one = posterior(nominal_belief, NOMINAL, query, true);
        let mut belief_zero: BeliefSet = [[0.; 3]; 4];
        let mut belief_one: BeliefSet = [[0.; 3]; 4];
        let mut probability_zero = [0.; 4];
        let mut probability_one = [0.; 4];

        for s in 0..4 {
            belief_zero[s] = posterior(scenario_beliefs[s], AMBIGUITY[s], query, false);
            belief_one[s] = posterior(scenario_beliefs[s], AMBIGUITY[s], query, true);
            probability_zero[s] = predictive_probability(scenario_beliefs[s], AMBIGUITY[s], query, false);
            probability_one[s] = predictive_probability(scenario_beliefs[s], AMBIGUITY[s], query, true);
        }

        let zero_plans = enumerate_plans(next_nominal_zero, belief_zero, remainder, rounds - 1);
        let one_plans = enumerate_plans(next_nominal_one, belief_one, remainder, rounds - 1);
        for zero_plan in &zero_plans {
            for one_plan in &one_plans {
                let risk = std::array::from_fn(|s| {
                    probability_zero[s] * zero_plan.risk[s] + probability_one[s] * one_plan.risk[s]
                });
                let cost = std::array::from_fn(|s| {
                    COST[query] + probability_zero[s] * zero_plan.cost[s]
                        + probability_one[s] * one_plan.cost[s]
                });
                let queries = std::array::from_fn(|s| {
                    1. + probability_zero[s] * zero_plan.queries[s]
                        + probability_one[s] * one_plan.queries[s]
                });
                plans.push(Plan {
                    risk, cost, queries,
                    tree: Tree::Query {
                        index: query,
                        zero: Box::new(zero_plan.tree.clone()),
                        one: Box::new(one_plan.tree.clone()),
                    },
                });
            }
        }
    }
    plans
}
fn worst(values: [f64; 4]) -> f64 { values.into_iter().fold(f64::NEG_INFINITY, f64::max) }
fn is_better(candidate: &Plan, incumbent: &Plan) -> bool {
    let candidate_risk = worst(candidate.risk);
    let incumbent_risk = worst(incumbent.risk);
    let candidate_cost = worst(candidate.cost);
    let incumbent_cost = worst(incumbent.cost);
    candidate_risk < incumbent_risk - EPS
        || ((candidate_risk - incumbent_risk).abs() <= EPS
            && candidate_cost < incumbent_cost - EPS)
}
fn choose_plan(rounds: usize) -> Plan {
    let plans = enumerate_plans(PRIOR, [PRIOR; 4], 0b1111, rounds.min(3));
    let mut best = 0;
    for i in 1..plans.len() {
        if is_better(&plans[i], &plans[best]) { best = i; }
    }
    plans.swap_remove(best)
}
fn terminal(h: usize, belief: [f64; 3]) -> Metrics {
    let loss = LOSS[h][best_action(belief)];
    let best_possible = LOSS[h].iter().copied().fold(f64::INFINITY, f64::min);
    Metrics { loss, regret: loss - best_possible, cost: 0., queries: 0. }
}
fn evaluate_tree(tree: &Tree, truth: Matrix, h: usize, belief: [f64; 3]) -> Metrics {
    match tree {
        Tree::Stop => terminal(h, belief),
        Tree::Query { index, zero, one } => {
            let p0 = p_observe(truth, *index, h, false);
            let p1 = p_observe(truth, *index, h, true);
            let m0 = evaluate_tree(zero, truth, h, posterior(belief, NOMINAL, *index, false));
            let m1 = evaluate_tree(one, truth, h, posterior(belief, NOMINAL, *index, true));
            Metrics {
                loss: p0 * m0.loss + p1 * m1.loss,
                regret: p0 * m0.regret + p1 * m1.regret,
                cost: COST[*index] + p0 * m0.cost + p1 * m1.cost,
                queries: 1. + p0 * m0.queries + p1 * m1.queries,
            }
        }
    }
}
fn evaluate_plan(tree: &Tree, truth: Matrix) -> Metrics {
    let mut result = Metrics::default();
    for h in 0..3 {
        let m = evaluate_tree(tree, truth, h, PRIOR);
        result.loss += PRIOR[h] * m.loss;
        result.regret += PRIOR[h] * m.regret;
        result.cost += PRIOR[h] * m.cost;
        result.queries += PRIOR[h] * m.queries;
    }
    result
}
fn print_csv(filter: Option<&str>) -> Result<(), String> {
    if let Some(name) = filter {
        if !SCENARIOS.iter().any(|(scenario, _)| *scenario == name) {
            return Err(format!("unknown scenario: {name}"));
        }
    }
    println!("scenario,query_cap,expected_loss,expected_decision_regret,expected_query_cost,expected_query_count,worst_case_ambiguity_regret,planned_first_query");
    for cap in 0..=3 {
        let plan = choose_plan(cap);
        let first = plan.tree.first_query().map(|q| QUERY_NAMES[q]).unwrap_or("none");
        let worst_case = worst(plan.risk);
        for (scenario, matrix) in SCENARIOS {
            if filter.is_some_and(|f| f != scenario) { continue; }
            let m = evaluate_plan(&plan.tree, matrix);
            println!("{scenario},{cap},{:.12},{:.12},{:.12},{:.12},{:.12},{first}",
                m.loss, m.regret, m.cost, m.queries, worst_case);
        }
    }
    Ok(())
}
fn main() -> ExitCode {
    let mut filter = None;
    for arg in env::args().skip(1) {
        if arg == "--help" || arg == "-h" {
            eprintln!("Usage: cargo run --bin robust_tree -- [--scenario=NAME]");
            return ExitCode::SUCCESS;
        } else if let Some(value) = arg.strip_prefix("--scenario=") {
            filter = Some(value);
        } else {
            eprintln!("unknown argument: {arg}");
            return ExitCode::from(2);
        }
    }
    match print_csv(filter) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => { eprintln!("{error}"); ExitCode::from(2) }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 1.0e-8 }

    #[test]
    fn exact_policy_tree_counts_are_stable() {
        assert_eq!(enumerate_plans(PRIOR, [PRIOR; 4], 0b1111, 0).len(), 1);
        assert_eq!(enumerate_plans(PRIOR, [PRIOR; 4], 0b1111, 1).len(), 5);
        assert_eq!(enumerate_plans(PRIOR, [PRIOR; 4], 0b1111, 2).len(), 65);
        assert_eq!(enumerate_plans(PRIOR, [PRIOR; 4], 0b1111, 3).len(), 3137);
    }
    #[test]
    fn horizon_aware_plan_beats_greedy_robust_worst_case_in_this_fixture() {
        let plan = choose_plan(3);
        assert!(worst(plan.risk) < 1.778);
        assert!(near(worst(plan.risk), 1.6090232));
    }
    #[test]
    fn horizon_aware_plan_has_lower_iid_regret_and_cost_than_plugin_greedy() {
        let plan = choose_plan(3);
        let iid = evaluate_plan(&plan.tree, NOMINAL);
        assert!(near(iid.regret, 1.0751888));
        assert!(near(iid.cost, 2.632));
        assert!(iid.regret < 1.091594);
        assert!(iid.cost < 3.87373);
    }
    #[test]
    fn out_of_set_shift_remains_a_failure_control() {
        let plan = choose_plan(3);
        let shifted = evaluate_plan(&plan.tree, OUT_OF_SET);
        assert!(shifted.regret > 2.3);
        assert!(near(shifted.regret, 3.513272));
    }
    #[test]
    fn all_reported_metrics_are_finite_and_query_cap_is_respected() {
        for cap in 0..=3 {
            let plan = choose_plan(cap);
            for (_, matrix) in SCENARIOS {
                let metrics = evaluate_plan(&plan.tree, matrix);
                assert!(metrics.loss.is_finite() && metrics.regret.is_finite());
                assert!(metrics.cost.is_finite() && metrics.queries.is_finite());
                assert!(metrics.queries <= cap as f64 + 1.0e-8);
            }
        }
    }
}
