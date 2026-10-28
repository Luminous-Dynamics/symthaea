//! Robust active-evidence policy trees over calibration-derived likelihood intervals.
//!
//! The calibration counts in this fixture are deterministic synthetic counts rounded around the
//! frozen nominal channel. They are used to stress interval width, not to claim empirical
//! calibration performance. The Hoeffding/Bonferroni coverage argument applies to independent,
//! fixed-sample Bernoulli calibration observations under stationarity.

use std::{env, process::ExitCode};

type Matrix = [[f64; 3]; 4];
type Intervals = [[[f64; 2]; 3]; 4];

const PRIOR: [f64; 3] = [0.6, 0.3, 0.1];
const LOSS: [[f64; 3]; 3] = [[0., 4., 7.], [5., 0., 3.], [8., 2., 0.]];
const COST: [f64; 4] = [1., 1., 2., 1.];
const QUERY_NAMES: [&str; 4] = ["coarse-shape", "inventory-check", "design-record", "cheap-proxy"];
const NOMINAL: Matrix = [
    [0.88, 0.34, 0.25],
    [0.35, 0.86, 0.31],
    [0.24, 0.38, 0.88],
    [0.76, 0.70, 0.18],
];
const SCENARIOS: [(&str, Matrix); 5] = [
    ("iid", NOMINAL),
    ("q0_conditional_shift", [
        [0.88, 0.76, 0.25], [0.35, 0.86, 0.31],
        [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
    ]),
    ("q1_asymmetric_error", [
        [0.88, 0.34, 0.25], [0.83, 0.22, 0.31],
        [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
    ]),
    ("compound_shift", [
        [0.88, 0.74, 0.25], [0.79, 0.22, 0.31],
        [0.24, 0.38, 0.88], [0.76, 0.83, 0.18],
    ]),
    ("out_of_set_flip", [
        [0.15, 0.92, 0.25], [0.35, 0.86, 0.31],
        [0.24, 0.38, 0.88], [0.76, 0.70, 0.18],
    ]),
];
const SAMPLE_SIZES: [usize; 9] = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
const ALPHA: f64 = 0.05;
const CELLS: f64 = 12.0;
const EPS: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
enum Tree {
    Stop,
    Query { index: usize, zero: Box<Tree>, one: Box<Tree> },
}

impl Tree {
    fn first_query(&self) -> Option<usize> {
        match self {
            Self::Stop => None,
            Self::Query { index, .. } => Some(*index),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    loss: f64,
    regret: f64,
    cost: f64,
    queries: f64,
}

fn p_observe(matrix: Matrix, query: usize, hypothesis: usize, outcome: bool) -> f64 {
    let probability = matrix[query][hypothesis];
    if outcome { probability } else { 1.0 - probability }
}

fn normalize(values: [f64; 3]) -> [f64; 3] {
    let sum = values.iter().sum::<f64>();
    if sum <= 0.0 {
        PRIOR
    } else {
        [values[0] / sum, values[1] / sum, values[2] / sum]
    }
}

fn posterior(belief: [f64; 3], query: usize, outcome: bool) -> [f64; 3] {
    normalize([
        belief[0] * p_observe(NOMINAL, query, 0, outcome),
        belief[1] * p_observe(NOMINAL, query, 1, outcome),
        belief[2] * p_observe(NOMINAL, query, 2, outcome),
    ])
}

fn action_risk(belief: [f64; 3], action: usize) -> f64 {
    (0..3).map(|h| belief[h] * LOSS[h][action]).sum()
}

fn best_action(belief: [f64; 3]) -> usize {
    let mut best = 0;
    let mut risk = action_risk(belief, 0);
    for action in 1..3 {
        let candidate = action_risk(belief, action);
        if candidate < risk - EPS {
            best = action;
            risk = candidate;
        }
    }
    best
}

fn available(mask: u8) -> Vec<usize> {
    (0..4).filter(|query| mask & (1_u8 << query) != 0).collect()
}

/// Enumerate all observation-contingent query trees. No query is repeated on a path, which is
/// the key condition for multi-affinity of the per-hypothesis trajectory risk in channel rates.
fn enumerate_trees(belief: [f64; 3], mask: u8, rounds: usize) -> Vec<Tree> {
    let mut trees = vec![Tree::Stop];
    if rounds == 0 || mask == 0 {
        return trees;
    }
    for query in available(mask) {
        let remainder = mask & !(1_u8 << query);
        let zero_belief = posterior(belief, query, false);
        let one_belief = posterior(belief, query, true);
        let zero_trees = enumerate_trees(zero_belief, remainder, rounds - 1);
        let one_trees = enumerate_trees(one_belief, remainder, rounds - 1);
        for zero in &zero_trees {
            for one in &one_trees {
                trees.push(Tree::Query {
                    index: query,
                    zero: Box::new(zero.clone()),
                    one: Box::new(one.clone()),
                });
            }
        }
    }
    trees
}

/// Fixed-sample Hoeffding intervals or a time-uniform sequence built by alpha spending.
///
/// For anytime=true, per-cell/time failure mass is alpha/[CELLS*n*(n+1)]. Since
/// sum_{n>=1} 1/(n*(n+1)) = 1, Hoeffding plus the union bound covers all 12 cells
/// and all sample sizes simultaneously with probability at least 1-alpha, assuming
/// independent stationary Bernoulli observations per cell. This is conservative but
/// valid under optional stopping; the fixed-sample interval alone is not.
fn hoeffding_intervals(n: usize, anytime: bool) -> Intervals {
    let radius = interval_radius(n, anytime);
    let mut intervals = [[[0.0; 2]; 3]; 4];
    for query in 0..4 {
        for hypothesis in 0..3 {
            let successes = (NOMINAL[query][hypothesis] * n as f64).round_ties_even() as usize;
            let estimate = successes as f64 / n as f64;
            intervals[query][hypothesis] = [
                (estimate - radius).max(0.0),
                (estimate + radius).min(1.0),
            ];
        }
    }
    intervals
}

fn interval_radius(n: usize, anytime: bool) -> f64 {
    let log_term = if anytime {
        (2.0 * CELLS * n as f64 * (n + 1) as f64 / ALPHA).ln()
    } else {
        (2.0 * CELLS / ALPHA).ln()
    };
    (log_term / (2.0 * n as f64)).sqrt()
}

const COVERAGE_GRID: [f64; 11] = [
    0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0,
];

/// Stable normalized Binomial(n,p) probability masses, generated outward from a mode.
/// This is used to enumerate exact finite-grid coverage without sampling Monte Carlo trials.
fn binomial_pmf(n: usize, p: f64) -> Vec<f64> {
    let mut mass = vec![0.0; n + 1];
    if p <= 0.0 {
        mass[0] = 1.0;
        return mass;
    }
    if p >= 1.0 {
        mass[n] = 1.0;
        return mass;
    }

    let mode = (((n + 1) as f64) * p).floor() as usize;
    let mode = mode.min(n);
    mass[mode] = 1.0;

    for k in (1..=mode).rev() {
        mass[k - 1] = mass[k]
            * (k as f64 / (n - k + 1) as f64)
            * ((1.0 - p) / p);
    }
    for k in mode..n {
        mass[k + 1] = mass[k]
            * ((n - k) as f64 / (k + 1) as f64)
            * (p / (1.0 - p));
    }

    let total = mass.iter().sum::<f64>();
    for value in &mut mass {
        *value /= total;
    }
    mass
}

fn cell_coverage(n: usize, true_probability: f64, anytime: bool) -> f64 {
    let radius = interval_radius(n, anytime);
    binomial_pmf(n, true_probability)
        .into_iter()
        .enumerate()
        .filter(|(successes, _)| {
            let estimate = *successes as f64 / n as f64;
            let lower = (estimate - radius).max(0.0);
            let upper = (estimate + radius).min(1.0);
            lower <= true_probability && true_probability <= upper
        })
        .map(|(_, probability_mass)| probability_mass)
        .sum()
}

/// Expected loss/cost conditional on a fixed true hypothesis. Policy belief updates always use
/// the nominal channel, while outcome probabilities use the supplied true/vertex matrix.
fn evaluate_conditional(
    tree: &Tree,
    truth: Matrix,
    hypothesis: usize,
    belief: [f64; 3],
) -> Metrics {
    match tree {
        Tree::Stop => {
            let action = best_action(belief);
            let loss = LOSS[hypothesis][action];
            let minimum = LOSS[hypothesis]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            Metrics {
                loss,
                regret: loss - minimum,
                cost: 0.0,
                queries: 0.0,
            }
        }
        Tree::Query { index, zero, one } => {
            let p0 = p_observe(truth, *index, hypothesis, false);
            let p1 = p_observe(truth, *index, hypothesis, true);
            let zero_metrics = evaluate_conditional(
                zero,
                truth,
                hypothesis,
                posterior(belief, *index, false),
            );
            let one_metrics = evaluate_conditional(
                one,
                truth,
                hypothesis,
                posterior(belief, *index, true),
            );
            Metrics {
                loss: p0 * zero_metrics.loss + p1 * one_metrics.loss,
                regret: p0 * zero_metrics.regret + p1 * one_metrics.regret,
                cost: COST[*index] + p0 * zero_metrics.cost + p1 * one_metrics.cost,
                queries: 1.0 + p0 * zero_metrics.queries + p1 * one_metrics.queries,
            }
        }
    }
}

/// Exact worst-case risk/cost over the product box. For each true hypothesis h, a trajectory's
/// probability depends only on the four parameters p(query | h). Because a query is never repeated
/// on a path, conditional expected loss/cost is multi-affine in those four probabilities, so its
/// maximum over that row's interval box is attained at one of 2^4 vertices. The rows are independent
/// box factors, hence max over the full box of E[loss] = sum_h prior[h] * max_row_h E[loss | h].
fn worst_box_metrics(tree: &Tree, n: usize, anytime: bool) -> Metrics {
    let intervals = hoeffding_intervals(n, anytime);
    let mut worst_loss = 0.0;
    let mut worst_cost = 0.0;
    let mut worst_queries = 0.0;

    for hypothesis in 0..3 {
        let mut row_worst_loss = f64::NEG_INFINITY;
        let mut row_worst_cost = f64::NEG_INFINITY;
        let mut row_worst_queries = f64::NEG_INFINITY;
        for vertex in 0..(1_u8 << 4) {
            let mut matrix = NOMINAL;
            for query in 0..4 {
                let endpoint = usize::from(vertex & (1_u8 << query) != 0);
                matrix[query][hypothesis] = intervals[query][hypothesis][endpoint];
            }
            let metrics = evaluate_conditional(tree, matrix, hypothesis, PRIOR);
            row_worst_loss = row_worst_loss.max(metrics.loss);
            row_worst_cost = row_worst_cost.max(metrics.cost);
            row_worst_queries = row_worst_queries.max(metrics.queries);
        }
        worst_loss += PRIOR[hypothesis] * row_worst_loss;
        worst_cost += PRIOR[hypothesis] * row_worst_cost;
        worst_queries += PRIOR[hypothesis] * row_worst_queries;
    }

    Metrics {
        loss: worst_loss,
        regret: worst_loss, // Every true-hypothesis row has a zero-loss terminal action.
        cost: worst_cost,
        queries: worst_queries,
    }
}

fn choose_tree(trees: &[Tree], n: usize, anytime: bool) -> (Tree, Metrics) {
    let mut best_tree = trees[0].clone();
    let mut best_metrics = worst_box_metrics(&best_tree, n, anytime);
    for tree in trees.iter().skip(1) {
        let metrics = worst_box_metrics(tree, n, anytime);
        if metrics.loss < best_metrics.loss - EPS
            || ((metrics.loss - best_metrics.loss).abs() <= EPS
                && metrics.cost < best_metrics.cost - EPS)
        {
            best_tree = tree.clone();
            best_metrics = metrics;
        }
    }
    (best_tree, best_metrics)
}

fn evaluate(tree: &Tree, truth: Matrix) -> Metrics {
    let mut total = Metrics::default();
    for hypothesis in 0..3 {
        let conditional = evaluate_conditional(tree, truth, hypothesis, PRIOR);
        total.loss += PRIOR[hypothesis] * conditional.loss;
        total.regret += PRIOR[hypothesis] * conditional.regret;
        total.cost += PRIOR[hypothesis] * conditional.cost;
        total.queries += PRIOR[hypothesis] * conditional.queries;
    }
    total
}

fn method_name(anytime: bool) -> &'static str {
    if anytime { "time_uniform" } else { "fixed_sample" }
}

fn print_coverage_csv() {
    println!("interval_method,n_per_cell,true_probability,cell_coverage,minimum_grid_cell_coverage,union_bound_lower_12_cells,per_cell_target");
    for anytime in [false, true] {
        for n in SAMPLE_SIZES {
            let values: Vec<(f64, f64)> = COVERAGE_GRID
                .iter()
                .copied()
                .map(|p| (p, cell_coverage(n, p, anytime)))
                .collect();
            let minimum = values
                .iter()
                .map(|(_, coverage)| *coverage)
                .fold(f64::INFINITY, f64::min);
            let union_bound = 1.0 - CELLS * (1.0 - minimum);
            let per_cell_target = 1.0 - ALPHA / CELLS;
            for (true_probability, coverage) in values {
                println!(
                    "{},{n},{true_probability:.12},{coverage:.12},{minimum:.12},{union_bound:.12},{per_cell_target:.12}",
                    method_name(anytime)
                );
            }
        }
    }
}

fn print_csv() {
    println!("interval_method,n_per_cell,confidence_radius,scenario,expected_loss,expected_decision_regret,expected_query_cost,expected_query_count,worst_case_box_regret,worst_case_box_expected_cost,worst_case_box_expected_queries,planned_first_query");
    let trees = enumerate_trees(PRIOR, 0b1111, 3);
    for anytime in [false, true] {
        for n in SAMPLE_SIZES {
            let (tree, worst) = choose_tree(&trees, n, anytime);
            let first = tree
                .first_query()
                .map(|query| QUERY_NAMES[query])
                .unwrap_or("none");
            for (scenario, matrix) in SCENARIOS {
                let metrics = evaluate(&tree, matrix);
                println!(
                    "{},{n},{:.12},{scenario},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{first}",
                    method_name(anytime),
                    interval_radius(n, anytime),
                    metrics.loss,
                    metrics.regret,
                    metrics.cost,
                    metrics.queries,
                    worst.regret,
                    worst.cost,
                    worst.queries
                );
            }
        }
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next() {
        None => {
            print_csv();
            ExitCode::SUCCESS
        }
        Some(argument) if argument == "--coverage-grid" && args.next().is_none() => {
            print_coverage_csv();
            ExitCode::SUCCESS
        }
        Some(argument) if argument == "--help" || argument == "-h" => {
            eprintln!("Usage: cargo run --bin calibrated_box -- [--coverage-grid]");
            ExitCode::SUCCESS
        }
        Some(argument) => {
            eprintln!("unexpected argument: {argument}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(left: f64, right: f64, tolerance: f64) -> bool {
        (left - right).abs() <= tolerance
    }

    #[test]
    fn query_tree_inventory_is_exact_and_no_query_repeats_on_a_path() {
        assert_eq!(enumerate_trees(PRIOR, 0b1111, 0).len(), 1);
        assert_eq!(enumerate_trees(PRIOR, 0b1111, 1).len(), 5);
        assert_eq!(enumerate_trees(PRIOR, 0b1111, 2).len(), 65);
        assert_eq!(enumerate_trees(PRIOR, 0b1111, 3).len(), 3137);
    }

    #[test]
    fn fixed_and_anytime_radii_are_valid_ordered_and_nonincreasing() {
        for anytime in [false, true] {
            let mut previous = f64::INFINITY;
            for n in SAMPLE_SIZES {
                let radius = interval_radius(n, anytime);
                assert!(radius.is_finite() && radius > 0.0);
                assert!(radius < previous);
                previous = radius;
                let intervals = hoeffding_intervals(n, anytime);
                for query in 0..4 {
                    for hypothesis in 0..3 {
                        let low = intervals[query][hypothesis][0];
                        let high = intervals[query][hypothesis][1];
                        assert!((0.0..=1.0).contains(&low));
                        assert!((0.0..=1.0).contains(&high));
                        assert!(low <= NOMINAL[query][hypothesis]);
                        assert!(high >= NOMINAL[query][hypothesis]);
                    }
                }
            }
        }
        for n in SAMPLE_SIZES {
            assert!(interval_radius(n, true) >= interval_radius(n, false));
        }
    }

    #[test]
    fn anytime_radius_matches_the_declared_alpha_spending_schedule() {
        for n in SAMPLE_SIZES {
            let radius = interval_radius(n, true);
            let two_sided_hoeffding_bound = 2.0 * (-2.0 * n as f64 * radius * radius).exp();
            let allotted_cell_time_alpha = ALPHA / (CELLS * n as f64 * (n + 1) as f64);
            assert!(
                (two_sided_hoeffding_bound - allotted_cell_time_alpha).abs() <= 1.0e-15,
                "n={n}: Hoeffding bound does not match alpha spending"
            );
        }
        let partial_sum: f64 = (1..=100_000)
            .map(|n| ALPHA / (CELLS * n as f64 * (n + 1) as f64))
            .sum();
        let telescoping_limit = ALPHA / CELLS;
        assert!(partial_sum <= telescoping_limit + 1.0e-15);
        assert!((telescoping_limit - partial_sum - telescoping_limit / 100_001.0).abs() < 1.0e-12);
    }

    #[test]
    fn exact_binomial_coverage_grid_respects_the_bonferroni_budget() {
        let per_cell_target = 1.0 - ALPHA / CELLS;
        for anytime in [false, true] {
            for n in SAMPLE_SIZES {
                let mut minimum = f64::INFINITY;
                for true_probability in COVERAGE_GRID {
                    let coverage = cell_coverage(n, true_probability, anytime);
                    assert!(coverage.is_finite() && (0.0..=1.0 + 1.0e-10).contains(&coverage));
                    minimum = minimum.min(coverage);
                }
                assert!(
                    minimum + 1.0e-10 >= per_cell_target,
                    "anytime={anytime}, n={n}: minimum grid coverage {minimum} below {per_cell_target}"
                );
                let simultaneous_union_bound = 1.0 - CELLS * (1.0 - minimum);
                assert!(simultaneous_union_bound + 1.0e-10 >= 1.0 - ALPHA);
            }
        }
    }

    #[test]
    fn row_factorization_matches_full_twelve_parameter_vertex_max_for_shallow_tree() {
        let n = 50;
        let tree = enumerate_trees(PRIOR, 0b1111, 1)
            .into_iter()
            .nth(1)
            .expect("one-query tree");
        let factorized = worst_box_metrics(&tree, n, false);
        let intervals = hoeffding_intervals(n, false);
        let mut full_worst_loss = f64::NEG_INFINITY;
        let mut full_worst_cost = f64::NEG_INFINITY;
        for vertex in 0_u16..(1_u16 << 12) {
            let mut matrix = NOMINAL;
            for query in 0..4 {
                for hypothesis in 0..3 {
                    let bit = query * 3 + hypothesis;
                    let endpoint = usize::from(vertex & (1_u16 << bit) != 0);
                    matrix[query][hypothesis] = intervals[query][hypothesis][endpoint];
                }
            }
            let result = evaluate(&tree, matrix);
            full_worst_loss = full_worst_loss.max(result.loss);
            full_worst_cost = full_worst_cost.max(result.cost);
        }
        assert!(near(factorized.loss, full_worst_loss, 1.0e-9));
        assert!(near(factorized.cost, full_worst_cost, 1.0e-9));
    }

    #[test]
    fn insufficient_calibration_abstains_then_more_evidence_expands_tree() {
        let trees = enumerate_trees(PRIOR, 0b1111, 3);
        let (small, small_worst) = choose_tree(&trees, 20, false);
        assert_eq!(small, Tree::Stop);
        assert!(near(small_worst.loss, 2.3, 1.0e-9));
        let (medium, medium_worst) = choose_tree(&trees, 200, false);
        assert_eq!(medium.first_query(), Some(0));
        assert!(near(medium_worst.cost, 1.0, 1.0e-9));
        let (large, large_worst) = choose_tree(&trees, 1000, false);
        assert_eq!(large.first_query(), Some(0));
        assert!(large_worst.cost > 3.0);
        assert!(large_worst.loss < medium_worst.loss);
    }

    #[test]
    fn held_out_shift_metrics_are_finite_and_queries_stay_within_depth() {
        let trees = enumerate_trees(PRIOR, 0b1111, 3);
        for n in SAMPLE_SIZES {
            let (tree, worst) = choose_tree(&trees, n, false);
            assert!(worst.loss.is_finite() && worst.cost.is_finite());
            for (_, matrix) in SCENARIOS {
                let result = evaluate(&tree, matrix);
                assert!(result.loss.is_finite() && result.regret.is_finite());
                assert!(result.cost.is_finite() && result.queries.is_finite());
                assert!(result.cost >= 0.0 && result.queries <= 3.0 + 1.0e-8);
            }
        }
    }
}
