//! Robust action selection from class-conditional calibration counts.
//!
//! Each row records the outcomes observed for one known ground-truth class. The test constructs
//! a simultaneous 95% Bonferroni-Clopper-Pearson confidence region for all nine confusion-matrix cells,
//! intersects each row with the probability simplex, and evaluates the deployed policy against
//! every vertex of that feasible matrix set. Because the fixed-policy risk is linear in the true
//! confusion matrix, its maximum over this polytope occurs at a vertex. This is a finite synthetic
//! oracle, not evidence that calibration remains valid under distribution shift or dependence.

const PROBABILITY_SCALE: u16 = 1_000;
const FAMILYWISE_ALPHA: f64 = 0.05;
const SIMULTANEOUS_CELL_TAIL_ALPHA: f64 = FAMILYWISE_ALPHA / 18.0;
// Bound inversion and exact-enumeration runtime; larger samples require a separately
// qualified asymptotic or high-precision implementation rather than unbounded loops.
const MAX_SUPPORTED_TRIALS_PER_CLASS: u64 = 1_000_000;
const EPSILON: f64 = 1.0e-9;
const PRIOR_WEIGHTS: [u64; 3] = [6, 3, 1];
const DECISION_LOSSES: [[u32; 3]; 2] = [[0, 100, 100], [4, 4, 4]];
const ACQUISITION_COST: f64 = 0.1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Interval {
    lower: u16,
    upper: u16,
}

const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural log of Gamma(x), using the Lanczos approximation for positive x.
fn log_gamma(value: f64) -> f64 {
    let shifted = value - 1.0;
    let mut sum = LANCZOS_COEFFICIENTS[0];
    for (index, coefficient) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        sum += coefficient / (shifted + index as f64);
    }
    let t = shifted + 7.5;
    0.918_938_533_204_672_7 + (shifted + 0.5) * t.ln() - t + sum.ln()
}

fn log_binomial_mass(total: u64, successes: u64, probability: f64) -> f64 {
    if probability <= 0.0 {
        return if successes == 0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if probability >= 1.0 {
        return if successes == total { 0.0 } else { f64::NEG_INFINITY };
    }
    log_gamma(total as f64 + 1.0)
        - log_gamma(successes as f64 + 1.0)
        - log_gamma((total - successes) as f64 + 1.0)
        + successes as f64 * probability.ln()
        + (total - successes) as f64 * (-probability).ln_1p()
}

/// P[X >= successes] evaluated from the boundary mass and stable recurrence.
/// Call at p <= successes / total, where right-tail terms decrease.
fn binomial_upper_tail(total: u64, successes: u64, probability: f64) -> f64 {
    if successes == 0 {
        return 1.0;
    }
    if probability <= 0.0 {
        return 0.0;
    }
    if probability >= 1.0 {
        return 1.0;
    }

    let mut term = log_binomial_mass(total, successes, probability).exp();
    if term == 0.0 {
        return 0.0;
    }
    let mut sum = term;
    for current in successes..total {
        let ratio = (total - current) as f64 / (current + 1) as f64
            * probability / (1.0 - probability);
        term *= ratio;
        sum += term;
        if ratio < 1.0 && term <= sum * 1.0e-15 {
            break;
        }
    }
    sum.min(1.0)
}

/// P[X <= successes] evaluated from the boundary mass and stable recurrence.
/// Call at p >= successes / total, where left-tail terms decrease.
fn binomial_lower_tail(total: u64, successes: u64, probability: f64) -> f64 {
    if successes >= total {
        return 1.0;
    }
    if probability <= 0.0 {
        return 1.0;
    }
    if probability >= 1.0 {
        return 0.0;
    }

    let mut term = log_binomial_mass(total, successes, probability).exp();
    if term == 0.0 {
        return 0.0;
    }
    let mut sum = term;
    for current in (1..=successes).rev() {
        let ratio =
            current as f64 / (total - current + 1) as f64 * (1.0 - probability) / probability;
        term *= ratio;
        sum += term;
        if ratio < 1.0 && term <= sum * 1.0e-15 {
            break;
        }
    }
    sum.min(1.0)
}

/// Two-sided Clopper-Pearson exact binomial interval with per-cell alpha = 0.05 / 9.
/// The endpoints are found by inverting exact binomial tail probabilities. Outward per-mille
/// rounding expands rather than narrows the interval, preserving its coverage guarantee.
fn clopper_pearson_cell_interval(successes: u64, total: u64) -> Option<Interval> {
    if total == 0 || successes > total || total > MAX_SUPPORTED_TRIALS_PER_CLASS {
        return None;
    }

    let lower = if successes == 0 {
        0.0
    } else {
        let mut low = 0.0;
        let mut high = successes as f64 / total as f64;
        for _ in 0..80 {
            let middle = (low + high) / 2.0;
            if binomial_upper_tail(total, successes, middle) < SIMULTANEOUS_CELL_TAIL_ALPHA {
                low = middle;
            } else {
                high = middle;
            }
        }
        low
    };

    let upper = if successes == total {
        1.0
    } else {
        let mut low = successes as f64 / total as f64;
        let mut high = 1.0;
        for _ in 0..80 {
            let middle = (low + high) / 2.0;
            if binomial_lower_tail(total, successes, middle) > SIMULTANEOUS_CELL_TAIL_ALPHA {
                low = middle;
            } else {
                high = middle;
            }
        }
        high
    };

    let scale = f64::from(PROBABILITY_SCALE);
    let interval = Interval {
        lower: ((lower * scale - 1.0e-9).floor().max(0.0) as u16).min(PROBABILITY_SCALE),
        upper: ((upper * scale + 1.0e-9).ceil().max(0.0) as u16).min(PROBABILITY_SCALE),
    };
    if interval.lower > interval.upper {
        return None;
    }
    Some(interval)
}

/// Vertices of a three-outcome row within three marginal intervals and sum(row) = 1000.
/// At a vertex of this two-dimensional polytope, at least two cells are at interval bounds.
fn row_vertices(intervals: [Interval; 3]) -> Vec<[u16; 3]> {
    let mut vertices = Vec::new();
    for (first, second, remaining) in [(0_usize, 1_usize, 2_usize), (0, 2, 1), (1, 2, 0)] {
        for first_value in [intervals[first].lower, intervals[first].upper] {
            for second_value in [intervals[second].lower, intervals[second].upper] {
                let assigned = u32::from(first_value) + u32::from(second_value);
                if assigned > u32::from(PROBABILITY_SCALE) {
                    continue;
                }
                let remaining_value = (u32::from(PROBABILITY_SCALE) - assigned) as u16;
                let allowed = intervals[remaining];
                if !(allowed.lower..=allowed.upper).contains(&remaining_value) {
                    continue;
                }
                let mut row = [0_u16; 3];
                row[first] = first_value;
                row[second] = second_value;
                row[remaining] = remaining_value;
                if !vertices.contains(&row) {
                    vertices.push(row);
                }
            }
        }
    }
    vertices
}

/// Builds a simultaneous confidence region for all nine class-conditional probabilities.
/// Returns None for empty classes, overflowing counts, nonrepresentable counts, or infeasible rows.
fn confusion_intervals(counts: [[u64; 3]; 3]) -> Option<[[Interval; 3]; 3]> {
    let empty = Interval { lower: 0, upper: 0 };
    let mut regions = [[empty; 3]; 3];
    for (class_index, row) in counts.iter().enumerate() {
        let total = row.iter().try_fold(0_u64, |sum, count| sum.checked_add(*count))?;
        if total == 0 || total > MAX_SUPPORTED_TRIALS_PER_CLASS {
            return None;
        }
        for (outcome_index, count) in row.iter().enumerate() {
            regions[class_index][outcome_index] = clopper_pearson_cell_interval(*count, total)?;
        }
        if row_vertices(regions[class_index]).is_empty() {
            return None;
        }
    }
    Some(regions)
}

fn valid_probability_matrix(matrix: [[u16; 3]; 3]) -> bool {
    matrix.iter().all(|row| {
        row.iter().map(|value| u32::from(*value)).sum::<u32>()
            == u32::from(PROBABILITY_SCALE)
    })
}

fn baseline_expected_loss(weights: [u64; 3], decisions: &[[u32; 3]]) -> Option<f64> {
    let total_weight = weights.iter().try_fold(0_u64, |sum, weight| sum.checked_add(*weight))?;
    if total_weight == 0 || decisions.is_empty() {
        return None;
    }
    decisions
        .iter()
        .map(|losses| {
            weights
                .iter()
                .zip(losses.iter())
                .map(|(weight, loss)| *weight as f64 * f64::from(*loss))
                .sum::<f64>()
                / total_weight as f64
        })
        .min_by(f64::total_cmp)
}

/// Fix the deployed decision for each observed outcome using the assumed model.
/// An outcome with zero assumed probability falls back to the caller's safe decision.
fn deployed_policy(
    assumed: [[u16; 3]; 3],
    weights: [u64; 3],
    decisions: &[[u32; 3]],
    fallback_decision: usize,
) -> Option<[usize; 3]> {
    if !valid_probability_matrix(assumed)
        || decisions.is_empty()
        || fallback_decision >= decisions.len()
    {
        return None;
    }
    let mut actions = [fallback_decision; 3];
    for (outcome_index, action) in actions.iter_mut().enumerate() {
        let outcome_mass = weights
            .iter()
            .enumerate()
            .map(|(class_index, weight)| *weight * u64::from(assumed[class_index][outcome_index]))
            .sum::<u64>();
        if outcome_mass == 0 {
            *action = fallback_decision;
            continue;
        }
        *action = (0..decisions.len())
            .min_by(|left, right| {
                let loss_for = |decision_index: usize| {
                    weights
                        .iter()
                        .enumerate()
                        .map(|(class_index, weight)| {
                            *weight as f64
                                * f64::from(assumed[class_index][outcome_index])
                                * f64::from(decisions[decision_index][class_index])
                        })
                        .sum::<f64>()
                };
                loss_for(*left).total_cmp(&loss_for(*right))
            })?;
    }
    Some(actions)
}

fn policy_risk(
    truth: [[u16; 3]; 3],
    weights: [u64; 3],
    decisions: &[[u32; 3]],
    actions: [usize; 3],
) -> Option<f64> {
    if !valid_probability_matrix(truth)
        || decisions.is_empty()
        || actions.iter().any(|action| *action >= decisions.len())
    {
        return None;
    }
    let total_weight = weights.iter().try_fold(0_u64, |sum, weight| sum.checked_add(*weight))?;
    if total_weight == 0 {
        return None;
    }
    let mut loss = 0.0;
    for (class_index, row) in truth.iter().enumerate() {
        for (outcome_index, probability) in row.iter().enumerate() {
            loss += weights[class_index] as f64
                * f64::from(*probability)
                * f64::from(decisions[actions[outcome_index]][class_index]);
        }
    }
    Some(loss / (total_weight as f64 * f64::from(PROBABILITY_SCALE)))
}

/// Exact worst-case deployed-policy loss over the Cartesian product of row-polytope vertices.
fn worst_case_policy_risk(
    regions: [[Interval; 3]; 3],
    weights: [u64; 3],
    decisions: &[[u32; 3]],
    actions: [usize; 3],
) -> Option<f64> {
    let rows = [
        row_vertices(regions[0]),
        row_vertices(regions[1]),
        row_vertices(regions[2]),
    ];
    if rows.iter().any(Vec::is_empty) {
        return None;
    }
    let mut worst = f64::NEG_INFINITY;
    for chair_row in &rows[0] {
        for table_row in &rows[1] {
            for stool_row in &rows[2] {
                let truth = [*chair_row, *table_row, *stool_row];
                worst = worst.max(policy_risk(truth, weights, decisions, actions)?);
            }
        }
    }
    if worst.is_finite() {
        Some(worst)
    } else {
        None
    }
}

fn robust_net_value(
    counts: [[u64; 3]; 3],
    assumed: [[u16; 3]; 3],
    weights: [u64; 3],
    decisions: &[[u32; 3]],
    safe_fallback: usize,
    acquisition_cost: f64,
) -> Option<(f64, f64)> {
    if !acquisition_cost.is_finite() || acquisition_cost < 0.0 {
        return None;
    }
    let regions = confusion_intervals(counts)?;
    let actions = deployed_policy(assumed, weights, decisions, safe_fallback)?;
    let baseline = baseline_expected_loss(weights, decisions)?;
    let worst = worst_case_policy_risk(regions, weights, decisions, actions)?;
    Some((worst, baseline - worst - acquisition_cost))
}

/// Vertices of an uncertainty set for the class prior, represented in per-mille units.
/// A vertex is formed by assigning two cells to interval bounds and solving the sum constraint.
fn prior_vertices(lower: [u16; 3], upper: [u16; 3]) -> Option<Vec<[u16; 3]>> {
    if lower
        .iter()
        .zip(upper.iter())
        .any(|(minimum, maximum)| minimum > maximum || *maximum > PROBABILITY_SCALE)
        || lower.iter().map(|value| u32::from(*value)).sum::<u32>()
            > u32::from(PROBABILITY_SCALE)
        || upper.iter().map(|value| u32::from(*value)).sum::<u32>()
            < u32::from(PROBABILITY_SCALE)
    {
        return None;
    }
    let mut vertices = Vec::new();
    for (first, second, remaining) in [(0_usize, 1_usize, 2_usize), (0, 2, 1), (1, 2, 0)] {
        for first_value in [lower[first], upper[first]] {
            for second_value in [lower[second], upper[second]] {
                let assigned = u32::from(first_value) + u32::from(second_value);
                if assigned > u32::from(PROBABILITY_SCALE) {
                    continue;
                }
                let remaining_value = (u32::from(PROBABILITY_SCALE) - assigned) as u16;
                if !(lower[remaining]..=upper[remaining]).contains(&remaining_value) {
                    continue;
                }
                let mut prior = [0_u16; 3];
                prior[first] = first_value;
                prior[second] = second_value;
                prior[remaining] = remaining_value;
                if !vertices.contains(&prior) {
                    vertices.push(prior);
                }
            }
        }
    }
    if vertices.is_empty() {
        None
    } else {
        Some(vertices)
    }
}

/// Worst-case no-measurement action: choose one action before the unknown prior is revealed.
fn robust_baseline_loss(priors: &[[u16; 3]], decisions: &[[u32; 3]]) -> Option<f64> {
    if priors.is_empty() || decisions.is_empty() {
        return None;
    }
    decisions
        .iter()
        .map(|losses| {
            priors
                .iter()
                .map(|prior| {
                    prior
                        .iter()
                        .zip(losses.iter())
                        .map(|(probability, loss)| {
                            f64::from(*probability) * f64::from(*loss)
                        })
                        .sum::<f64>()
                        / f64::from(PROBABILITY_SCALE)
                })
                .max_by(f64::total_cmp)
                .unwrap_or(f64::INFINITY)
        })
        .min_by(f64::total_cmp)
}

/// Maximize fixed-policy risk over both plausible priors and plausible confusion matrices.
/// For fixed prior, risk is linear in each matrix row; for fixed matrix, it is linear in prior.
/// Therefore the maximum over this Cartesian product is attained at a pair of vertices.
fn worst_case_policy_risk_over_priors(
    regions: [[Interval; 3]; 3],
    priors: &[[u16; 3]],
    decisions: &[[u32; 3]],
    actions: [usize; 3],
) -> Option<f64> {
    if priors.is_empty() {
        return None;
    }
    let rows = [
        row_vertices(regions[0]),
        row_vertices(regions[1]),
        row_vertices(regions[2]),
    ];
    if rows.iter().any(Vec::is_empty) {
        return None;
    }
    let mut worst = f64::NEG_INFINITY;
    for chair_row in &rows[0] {
        for table_row in &rows[1] {
            for stool_row in &rows[2] {
                let truth = [*chair_row, *table_row, *stool_row];
                for prior in priors {
                    let mut loss = 0.0;
                    for (class_index, row) in truth.iter().enumerate() {
                        for (outcome_index, probability) in row.iter().enumerate() {
                            loss += f64::from(prior[class_index])
                                * f64::from(*probability)
                                * f64::from(decisions[actions[outcome_index]][class_index]);
                        }
                    }
                    worst = worst.max(loss / (f64::from(PROBABILITY_SCALE) * f64::from(PROBABILITY_SCALE)));
                }
            }
        }
    }
    if worst.is_finite() {
        Some(worst)
    } else {
        None
    }
}

fn robust_net_value_over_prior_uncertainty(
    regions: [[Interval; 3]; 3],
    prior_lower: [u16; 3],
    prior_upper: [u16; 3],
    nominal_weights: [u64; 3],
    assumed: [[u16; 3]; 3],
    decisions: &[[u32; 3]],
    safe_fallback: usize,
    acquisition_cost: f64,
) -> Option<(f64, f64, f64)> {
    if !acquisition_cost.is_finite() || acquisition_cost < 0.0 {
        return None;
    }
    let priors = prior_vertices(prior_lower, prior_upper)?;
    let actions = deployed_policy(assumed, nominal_weights, decisions, safe_fallback)?;
    let baseline = robust_baseline_loss(&priors, decisions)?;
    let worst_policy = worst_case_policy_risk_over_priors(regions, &priors, decisions, actions)?;
    Some((
        baseline,
        worst_policy,
        baseline - worst_policy - acquisition_cost,
    ))
}

fn worst_policy_risk_from_counts(
    counts: [[u64; 3]; 3],
    weights: [u64; 3],
    decisions: &[[u32; 3]],
    actions: [usize; 3],
) -> Option<f64> {
    let regions = confusion_intervals(counts)?;
    worst_case_policy_risk(regions, weights, decisions, actions)
}

/// Predictive value of collecting one additional labeled calibration sample from a known class.
/// The current deployed action policy is held fixed; only the confidence region is updated.
/// Outcome probabilities are taken from the explicitly supplied assumed calibration row.
fn expected_risk_after_one_calibration_sample(
    counts: [[u64; 3]; 3],
    class_index: usize,
    assumed: [[u16; 3]; 3],
    weights: [u64; 3],
    decisions: &[[u32; 3]],
    actions: [usize; 3],
) -> Option<f64> {
    if class_index >= 3 || !valid_probability_matrix(assumed) {
        return None;
    }
    let mut expected_risk = 0.0;
    for (outcome_index, probability) in assumed[class_index].iter().enumerate() {
        if *probability == 0 {
            continue;
        }
        let mut updated = counts;
        updated[class_index][outcome_index] =
            updated[class_index][outcome_index].checked_add(1)?;
        let risk = worst_policy_risk_from_counts(updated, weights, decisions, actions)?;
        expected_risk += risk * f64::from(*probability) / f64::from(PROBABILITY_SCALE);
    }
    Some(expected_risk)
}

/// Select the next calibration class by expected reduction in robust decision risk per unit cost.
/// Returns None if no candidate has positive expected value or if the cost contract is invalid.
fn select_next_calibration_class(
    counts: [[u64; 3]; 3],
    assumed: [[u16; 3]; 3],
    weights: [u64; 3],
    decisions: &[[u32; 3]],
    actions: [usize; 3],
    class_costs: [f64; 3],
) -> Option<(usize, f64)> {
    if class_costs.iter().any(|cost| !cost.is_finite() || *cost <= 0.0) {
        return None;
    }
    let current_risk = worst_policy_risk_from_counts(counts, weights, decisions, actions)?;
    let mut best: Option<(usize, f64)> = None;
    for (class_index, class_cost) in class_costs.iter().enumerate() {
        let expected_risk = expected_risk_after_one_calibration_sample(
            counts,
            class_index,
            assumed,
            weights,
            decisions,
            actions,
        )?;
        let improvement = current_risk - expected_risk;
        if improvement <= EPSILON {
            continue;
        }
        let ratio = improvement / *class_cost;
        if best.is_none_or(|(_, best_ratio)| ratio > best_ratio + EPSILON) {
            best = Some((class_index, ratio));
        }
    }
    best
}

fn nominal_counts(per_class: u64) -> [[u64; 3]; 3] {
    [
        [per_class * 95 / 100, per_class * 5 / 100, 0],
        [per_class * 5 / 100, per_class * 95 / 100, 0],
        [per_class * 5 / 100, per_class * 95 / 100, 0],
    ]
}

fn assumed_sensor() -> [[u16; 3]; 3] {
    [[950, 50, 0], [50, 950, 0], [50, 950, 0]]
}

fn wilson_cell_interval_negative_control(successes: u64, total: u64) -> Option<Interval> {
    if total == 0 || successes > total || total > MAX_SUPPORTED_TRIALS_PER_CLASS {
        return None;
    }
    let z: f64 = 2.772_921_294_608_662;
    let n = total as f64;
    let proportion = successes as f64 / n;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let center = (proportion + z_squared / (2.0 * n)) / denominator;
    let half_width = z
        * (proportion * (1.0 - proportion) / n + z_squared / (4.0 * n * n)).sqrt()
        / denominator;
    let lower = (center - half_width).clamp(0.0, 1.0);
    let upper = (center + half_width).clamp(0.0, 1.0);
    Some(Interval {
        lower: ((lower * f64::from(PROBABILITY_SCALE) - 1.0e-9).floor().max(0.0) as u16)
            .min(PROBABILITY_SCALE),
        upper: ((upper * f64::from(PROBABILITY_SCALE) + 1.0e-9).ceil().max(0.0) as u16)
            .min(PROBABILITY_SCALE),
    })
}

fn multinomial_mass(total: u64, counts: [u64; 3], truth: [f64; 3]) -> f64 {
    if counts
        .iter()
        .zip(truth.iter())
        .any(|(count, probability)| *count > 0 && *probability == 0.0)
    {
        return 0.0;
    }
    let mut log_probability = log_gamma(total as f64 + 1.0);
    for (count, probability) in counts.iter().zip(truth.iter()) {
        log_probability -= log_gamma(*count as f64 + 1.0);
        if *count > 0 {
            log_probability += *count as f64 * probability.ln();
        }
    }
    log_probability.exp()
}

/// Exact finite multinomial enumeration for one row of the confusion matrix.
fn exact_row_coverage(
    total: u64,
    truth: [f64; 3],
    interval_method: fn(u64, u64) -> Option<Interval>,
) -> f64 {
    let mut covered_mass = 0.0;
    let mut total_mass = 0.0;
    for first in 0..=total {
        for second in 0..=(total - first) {
            let third = total - first - second;
            let counts = [first, second, third];
            let mass = multinomial_mass(total, counts, truth);
            total_mass += mass;
            let intervals = [
                interval_method(first, total).expect("valid first-cell count"),
                interval_method(second, total).expect("valid second-cell count"),
                interval_method(third, total).expect("valid third-cell count"),
            ];
            if truth.iter().zip(intervals.iter()).all(|(probability, interval)| {
                *probability * f64::from(PROBABILITY_SCALE) + 1.0e-9
                    >= f64::from(interval.lower)
                    && *probability * f64::from(PROBABILITY_SCALE) - 1.0e-9
                        <= f64::from(interval.upper)
            }) {
                covered_mass += mass;
            }
        }
    }
    assert!((total_mass - 1.0).abs() < 1.0e-10, "multinomial mass must sum to one");
    covered_mass
}

fn exact_matrix_coverage(
    trials_per_class: u64,
    truth: [[f64; 3]; 3],
    interval_method: fn(u64, u64) -> Option<Interval>,
) -> f64 {
    truth
        .iter()
        .map(|row| exact_row_coverage(trials_per_class, *row, interval_method))
        .product()
}

fn expected_binomial_interval_width(
    total: u64,
    truth_probability: f64,
    interval_method: fn(u64, u64) -> Option<Interval>,
) -> f64 {
    (0..=total)
        .map(|successes| {
            let probability_mass = log_binomial_mass(total, successes, truth_probability).exp();
            let interval = interval_method(successes, total).expect("valid interval input");
            probability_mass * f64::from(interval.upper - interval.lower)
        })
        .sum()
}

/// Expected sum of the nine cell-interval widths under the true matrix, in per-mille units.
fn expected_matrix_interval_width(
    trials_per_class: u64,
    truth: [[f64; 3]; 3],
    interval_method: fn(u64, u64) -> Option<Interval>,
) -> f64 {
    truth
        .iter()
        .flat_map(|row| row.iter())
        .map(|probability| {
            expected_binomial_interval_width(trials_per_class, *probability, interval_method)
        })
        .sum()
}

#[test]
fn coverage_width_frontier_compares_exact_and_wilson_regions() {
    let rare_event = [
        [0.99, 0.009, 0.001],
        [0.005, 0.99, 0.005],
        [0.001, 0.009, 0.99],
    ];
    let cases = [
        (10_u64, 0.997_500, 0.739_700, 4_082.703, 3_970.067),
        (20_u64, 0.996_676, 0.918_596, 2_396.361, 2_575.044),
        (50_u64, 0.995_195, 0.840_230, 1_121.239, 1_293.004),
    ];

    for (trials, expected_cp_coverage, expected_wilson_coverage, expected_cp_width, expected_wilson_width) in cases {
        let cp_coverage =
            exact_matrix_coverage(trials, rare_event, clopper_pearson_cell_interval);
        let wilson_coverage =
            exact_matrix_coverage(trials, rare_event, wilson_cell_interval_negative_control);
        let cp_width =
            expected_matrix_interval_width(trials, rare_event, clopper_pearson_cell_interval);
        let wilson_width =
            expected_matrix_interval_width(trials, rare_event, wilson_cell_interval_negative_control);

        assert!((cp_coverage - expected_cp_coverage).abs() < 1.0e-5);
        assert!((wilson_coverage - expected_wilson_coverage).abs() < 1.0e-5);
        assert!((cp_width - expected_cp_width).abs() < 0.1);
        assert!((wilson_width - expected_wilson_width).abs() < 0.1);
        assert!(cp_coverage >= 0.95);
        assert!(wilson_coverage < 0.95);
        // Width is not universally better: the exact region costs more at n=10, then
        // becomes narrower than Wilson for this rare-event matrix at n=20 and n=50.
        if trials >= 20 {
            assert!(cp_width < wilson_width);
        } else {
            assert!(cp_width > wilson_width);
        }
    }
}

#[test]
fn prior_shift_uncertainty_can_reverse_a_nominally_positive_acquisition_decision() {
    let known_matrix = [
        [
            Interval { lower: 950, upper: 950 },
            Interval { lower: 50, upper: 50 },
            Interval { lower: 0, upper: 0 },
        ],
        [
            Interval { lower: 50, upper: 50 },
            Interval { lower: 950, upper: 950 },
            Interval { lower: 0, upper: 0 },
        ],
        [
            Interval { lower: 50, upper: 50 },
            Interval { lower: 950, upper: 950 },
            Interval { lower: 0, upper: 0 },
        ],
    ];
    let nominal = [600, 300, 100];
    let nominal_loss = policy_risk(
        [[950, 50, 0], [50, 950, 0], [50, 950, 0]],
        nominal,
        &DECISION_LOSSES,
        [0, 1, 1],
    )
    .expect("valid nominal model");
    assert!((nominal_loss - 3.64).abs() <= EPSILON);
    assert!((4.0 - nominal_loss - ACQUISITION_COST - 0.26).abs() <= EPSILON);

    let broad = robust_net_value_over_prior_uncertainty(
        known_matrix,
        [200, 100, 0],
        [800, 700, 500],
        [6, 3, 1],
        assumed_sensor(),
        &DECISION_LOSSES,
        1,
        ACQUISITION_COST,
    )
    .expect("feasible broad prior region");
    assert!((broad.0 - 4.0).abs() <= EPSILON);
    assert!((broad.1 - 7.08).abs() <= EPSILON);
    assert!((broad.2 + 3.18).abs() <= EPSILON);
    assert!(broad.2 < 0.0);

    let narrow = robust_net_value_over_prior_uncertainty(
        known_matrix,
        [600, 250, 50],
        [700, 350, 150],
        [6, 3, 1],
        assumed_sensor(),
        &DECISION_LOSSES,
        1,
        ACQUISITION_COST,
    )
    .expect("feasible narrow prior region");
    assert!((narrow.0 - 4.0).abs() <= EPSILON);
    assert!((narrow.1 - 3.64).abs() <= EPSILON);
    assert!((narrow.2 - 0.26).abs() <= EPSILON);
    assert!(narrow.2 > 0.0);
}

#[test]
fn joint_matrix_and_prevalence_uncertainty_compose_conservatively() {
    let regions = confusion_intervals(nominal_counts(1_000)).expect("valid calibration counts");
    let result = robust_net_value_over_prior_uncertainty(
        regions,
        [200, 100, 0],
        [800, 700, 500],
        [6, 3, 1],
        assumed_sensor(),
        &DECISION_LOSSES,
        1,
        ACQUISITION_COST,
    )
    .expect("feasible joint uncertainty region");
    assert!((result.0 - 4.0).abs() <= EPSILON);
    assert!((result.1 - 8.8648).abs() <= EPSILON);
    assert!((result.2 + 4.9648).abs() <= EPSILON);
    assert!(result.2 < 0.0);
}

#[test]
fn joint_prior_and_matrix_robustness_is_invariant_to_class_relabeling() {
    let counts = nominal_counts(1_000);
    let lower = [200, 100, 0];
    let upper = [800, 700, 500];
    let nominal_weights = [6_u64, 3, 1];
    let assumed = assumed_sensor();
    let base_regions = confusion_intervals(counts).expect("valid calibration counts");
    let base = robust_net_value_over_prior_uncertainty(
        base_regions,
        lower,
        upper,
        nominal_weights,
        assumed,
        &DECISION_LOSSES,
        1,
        ACQUISITION_COST,
    )
    .expect("valid joint uncertainty region");
    assert!((base.0 - 4.0).abs() <= EPSILON);
    assert!((base.1 - 8.8648).abs() <= EPSILON);
    assert!((base.2 + 4.9648).abs() <= EPSILON);

    for order in [
        [0_usize, 1_usize, 2_usize],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let permute_prior = |values: [u16; 3]| [values[order[0]], values[order[1]], values[order[2]]];
        let permute_weights =
            |values: [u64; 3]| [values[order[0]], values[order[1]], values[order[2]]];
        let permute_matrix = |matrix: [[u16; 3]; 3]| {
            [matrix[order[0]], matrix[order[1]], matrix[order[2]]]
        };
        let permute_counts = |matrix: [[u64; 3]; 3]| {
            [matrix[order[0]], matrix[order[1]], matrix[order[2]]]
        };
        let permute_decisions = |decisions: &[[u32; 3]]| {
            decisions
                .iter()
                .map(|losses| [losses[order[0]], losses[order[1]], losses[order[2]]])
                .collect::<Vec<_>>()
        };
        let permuted_regions =
            confusion_intervals(permute_counts(counts)).expect("permuted counts are valid");
        let permuted_decisions = permute_decisions(&DECISION_LOSSES);
        let result = robust_net_value_over_prior_uncertainty(
            permuted_regions,
            permute_prior(lower),
            permute_prior(upper),
            permute_weights(nominal_weights),
            permute_matrix(assumed),
            &permuted_decisions,
            1,
            ACQUISITION_COST,
        )
        .expect("permuted uncertainty region is valid");
        assert!((result.0 - base.0).abs() <= EPSILON);
        assert!((result.1 - base.1).abs() <= EPSILON);
        assert!((result.2 - base.2).abs() <= EPSILON);
    }
}

#[test]
fn active_calibration_allocation_targets_the_class_with_highest_marginal_decision_value() {
    let counts = [[19, 1, 0], [1, 19, 0], [1, 19, 0]];
    let actions = deployed_policy(assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, 1)
        .expect("valid assumed model");
    let current_risk = worst_policy_risk_from_counts(
        counts,
        PRIOR_WEIGHTS,
        &DECISION_LOSSES,
        actions,
    )
    .expect("valid initial calibration region");
    assert!((current_risk - 15.5128).abs() <= EPSILON);

    let expected_after_chair = expected_risk_after_one_calibration_sample(
        counts, 0, assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, actions,
    ).expect("chair calibration is valid");
    let expected_after_table = expected_risk_after_one_calibration_sample(
        counts, 1, assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, actions,
    ).expect("table calibration is valid");
    let expected_after_stool = expected_risk_after_one_calibration_sample(
        counts, 2, assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, actions,
    ).expect("stool calibration is valid");

    assert!((expected_after_chair - 15.48736).abs() <= EPSILON);
    assert!((expected_after_table - 15.20752).abs() <= EPSILON);
    assert!((expected_after_stool - 15.41104).abs() <= EPSILON);

    let equal_costs = select_next_calibration_class(
        counts, assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, actions, [1.0, 1.0, 1.0],
    ).expect("a positive-value next sample exists");
    assert_eq!(equal_costs.0, 1, "equal costs should prioritize table, the highest risk reduction");
    assert!((equal_costs.1 - 0.30528).abs() <= EPSILON);

    let unequal_costs = select_next_calibration_class(
        counts, assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, actions, [1.0, 1.0, 0.25],
    ).expect("a positive-value next sample exists");
    assert_eq!(unequal_costs.0, 2, "lower stool-label cost changes the optimal acquisition");
    assert!((unequal_costs.1 - 0.40704).abs() <= EPSILON);
}

#[test]
fn active_calibration_allocator_fails_closed_for_invalid_costs_and_classes() {
    let counts = [[19, 1, 0], [1, 19, 0], [1, 19, 0]];
    let actions = deployed_policy(assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, 1)
        .expect("valid assumed model");
    assert_eq!(
        expected_risk_after_one_calibration_sample(
            counts, 3, assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, actions,
        ),
        None
    );
    assert_eq!(
        select_next_calibration_class(
            counts, assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, actions, [1.0, 0.0, 1.0],
        ),
        None
    );
    assert_eq!(
        select_next_calibration_class(
            counts, assumed_sensor(), PRIOR_WEIGHTS, &DECISION_LOSSES, actions, [1.0, f64::NAN, 1.0],
        ),
        None
    );
}

#[test]
fn invalid_or_infeasible_prior_regions_fail_closed() {
    assert_eq!(prior_vertices([800, 300, 100], [900, 400, 200]), None);
    assert_eq!(prior_vertices([300, 300, 300], [300, 300, 300]), None);
    assert_eq!(
        robust_baseline_loss(&[], &DECISION_LOSSES),
        None
    );
    let exact_matrix = [
        [
            Interval { lower: 950, upper: 950 },
            Interval { lower: 50, upper: 50 },
            Interval { lower: 0, upper: 0 },
        ],
        [
            Interval { lower: 50, upper: 50 },
            Interval { lower: 950, upper: 950 },
            Interval { lower: 0, upper: 0 },
        ],
        [
            Interval { lower: 50, upper: 50 },
            Interval { lower: 950, upper: 950 },
            Interval { lower: 0, upper: 0 },
        ],
    ];
    assert_eq!(
        robust_net_value_over_prior_uncertainty(
            exact_matrix,
            [200, 100, 0],
            [800, 700, 500],
            [6, 3, 1],
            assumed_sensor(),
            &DECISION_LOSSES,
            1,
            -0.1,
        ),
        None
    );
}

#[test]
fn exact_coverage_qualification_catches_rare_event_undercoverage() {
    let cases = [
        (
            "nominal-sparse",
            [
                [0.95, 0.05, 0.0],
                [0.05, 0.95, 0.0],
                [0.05, 0.95, 0.0],
            ],
            0.999_012,
        ),
        (
            "asymmetric-sparse",
            [
                [0.95, 0.05, 0.0],
                [0.15, 0.85, 0.0],
                [0.20, 0.80, 0.0],
            ],
            0.995_752,
        ),
        (
            "balanced-dense",
            [
                [0.70, 0.20, 0.10],
                [0.10, 0.70, 0.20],
                [0.20, 0.10, 0.70],
            ],
            0.980_770,
        ),
        (
            "rare-class-event",
            [
                [0.99, 0.009, 0.001],
                [0.005, 0.99, 0.005],
                [0.001, 0.009, 0.99],
            ],
            0.996_676,
        ),
    ];

    for (name, truth, expected_coverage) in cases {
        let actual = exact_matrix_coverage(20, truth, clopper_pearson_cell_interval);
        assert!(
            (actual - expected_coverage).abs() < 1.0e-5,
            "{name}: expected exact-enumeration coverage {expected_coverage}, got {actual}"
        );
        assert!(
            actual >= 0.95,
            "{name}: Bonferroni-Clopper-Pearson region under-covered: {actual}"
        );
    }

    let rare_event = [
        [0.99, 0.009, 0.001],
        [0.005, 0.99, 0.005],
        [0.001, 0.009, 0.99],
    ];
    let wilson_coverage =
        exact_matrix_coverage(20, rare_event, wilson_cell_interval_negative_control);
    assert!(
        wilson_coverage < 0.95,
        "negative control must reproduce Wilson undercoverage at rare-event boundary, got {wilson_coverage}"
    );
}

#[test]
fn exact_coverage_grid_qualifies_rare_and_asymmetric_probability_matrices() {
    let cases: [(&str, [[f64; 3]; 3], [f64; 3]); 4] = [
        (
            "nominal-sparse",
            [
                [0.95, 0.05, 0.0],
                [0.05, 0.95, 0.0],
                [0.05, 0.95, 0.0],
            ],
            [0.996_918, 0.999_012, 0.997_734],
        ),
        (
            "asymmetric-sparse",
            [
                [0.95, 0.05, 0.0],
                [0.15, 0.85, 0.0],
                [0.20, 0.80, 0.0],
            ],
            [0.996_727, 0.995_752, 0.993_215],
        ),
        (
            "balanced-dense",
            [
                [0.70, 0.20, 0.10],
                [0.10, 0.70, 0.20],
                [0.20, 0.10, 0.70],
            ],
            [0.989_228, 0.980_770, 0.974_754],
        ),
        (
            "rare-class-event",
            [
                [0.99, 0.009, 0.001],
                [0.005, 0.99, 0.005],
                [0.001, 0.009, 0.99],
            ],
            [0.997_500, 0.996_676, 0.995_195],
        ),
    ];
    let sample_sizes = [10_u64, 20, 50];

    for (name, truth, expected_by_sample_size) in cases {
        for (index, trials_per_class) in sample_sizes.iter().enumerate() {
            let actual = exact_matrix_coverage(
                *trials_per_class,
                truth,
                clopper_pearson_cell_interval,
            );
            let expected = expected_by_sample_size[index];
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "{name} at {trials_per_class} trials/class: expected {expected}, got {actual}"
            );
            assert!(
                actual >= 0.95,
                "{name} at {trials_per_class} trials/class under-covered: {actual}"
            );
        }
    }

    let rare_event = [
        [0.99, 0.009, 0.001],
        [0.005, 0.99, 0.005],
        [0.001, 0.009, 0.99],
    ];
    let expected_wilson_coverage = [0.739_700, 0.918_596, 0.840_230];
    for (index, trials_per_class) in sample_sizes.iter().enumerate() {
        let actual = exact_matrix_coverage(
            *trials_per_class,
            rare_event,
            wilson_cell_interval_negative_control,
        );
        let expected = expected_wilson_coverage[index];
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "Wilson negative control at {trials_per_class} trials/class: expected {expected}, got {actual}"
        );
        assert!(
            actual < 0.95,
            "Wilson negative control must expose rare-event undercoverage: {actual}"
        );
    }
}

#[test]
fn class_conditional_intervals_are_simultaneous_and_outward_rounded() {
    let intervals = confusion_intervals(nominal_counts(1_000)).expect("valid counts");
    assert_eq!(
        intervals[0],
        [
            Interval { lower: 927, upper: 968 },
            Interval { lower: 32, upper: 73 },
            Interval { lower: 0, upper: 6 },
        ]
    );
    assert_eq!(
        intervals[1],
        [
            Interval { lower: 32, upper: 73 },
            Interval { lower: 927, upper: 968 },
            Interval { lower: 0, upper: 6 },
        ]
    );
}

#[test]
fn robust_gate_rejects_small_or_moderate_calibration_samples() {
    let low = robust_net_value(
        nominal_counts(1_000),
        assumed_sensor(),
        PRIOR_WEIGHTS,
        &DECISION_LOSSES,
        1,
        ACQUISITION_COST,
    )
    .expect("valid low-sample matrix");
    assert!((low.0 - 4.5784).abs() <= EPSILON);
    assert!((low.1 + 0.6784).abs() <= EPSILON);
    assert!(low.1 <= 0.0);

    let medium = robust_net_value(
        nominal_counts(10_000),
        assumed_sensor(),
        PRIOR_WEIGHTS,
        &DECISION_LOSSES,
        1,
        ACQUISITION_COST,
    )
    .expect("valid medium-sample matrix");
    assert!((medium.0 - 3.9256).abs() <= EPSILON);
    assert!((medium.1 + 0.0256).abs() <= EPSILON);
    assert!(medium.1 <= 0.0);
}

#[test]
fn enough_class_conditional_calibration_can_certify_positive_worst_case_value() {
    let result = robust_net_value(
        nominal_counts(100_000),
        assumed_sensor(),
        PRIOR_WEIGHTS,
        &DECISION_LOSSES,
        1,
        ACQUISITION_COST,
    )
    .expect("valid high-sample matrix");
    assert!((result.0 - 3.7216).abs() <= EPSILON);
    assert!((result.1 - 0.1784).abs() <= EPSILON);
    assert!(result.1 > 0.0);
}

#[test]
fn class_conditional_false_positive_drift_is_not_hidden_by_average_accuracy() {
    let asymmetric = [
        [9_500, 500, 0],
        [1_500, 8_500, 0],
        [2_000, 8_000, 0],
    ];
    let result = robust_net_value(
        asymmetric,
        assumed_sensor(),
        PRIOR_WEIGHTS,
        &DECISION_LOSSES,
        1,
        ACQUISITION_COST,
    )
    .expect("valid class-conditional counts");
    assert!((result.0 - 8.4088).abs() <= EPSILON);
    assert!((result.1 + 4.5088).abs() <= EPSILON);
    assert!(result.1 < 0.0);
}

#[test]
fn malformed_counts_assumptions_and_costs_fail_closed() {
    assert_eq!(
        confusion_intervals([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),
        None
    );
    assert_eq!(
        confusion_intervals([[u64::MAX, 1, 0], [1, 1, 1], [1, 1, 1]]),
        None
    );
    assert_eq!(clopper_pearson_cell_interval(1, 1_000_001), None);
    assert_eq!(wilson_cell_interval_negative_control(1, 1_000_001), None);
    let malformed_assumed = [[900, 50, 0], [50, 950, 0], [50, 950, 0]];
    assert_eq!(
        robust_net_value(
            nominal_counts(1_000),
            malformed_assumed,
            PRIOR_WEIGHTS,
            &DECISION_LOSSES,
            1,
            ACQUISITION_COST
        ),
        None
    );
    assert_eq!(
        robust_net_value(
            nominal_counts(1_000),
            assumed_sensor(),
            PRIOR_WEIGHTS,
            &DECISION_LOSSES,
            1,
            f64::NAN
        ),
        None
    );
}

#[test]
fn impossible_observation_falls_back_to_safe_hold() {
    let actions = deployed_policy(
        assumed_sensor(),
        PRIOR_WEIGHTS,
        &DECISION_LOSSES,
        1,
    )
    .expect("valid policy");
    assert_eq!(actions, [0, 1, 1]);
}
