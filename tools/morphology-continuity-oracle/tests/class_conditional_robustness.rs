//! Robust action selection from class-conditional calibration counts.
//!
//! Each row records the outcomes observed for one known ground-truth class. The test constructs
//! a nominal simultaneous 95% Bonferroni-Wilson confidence region for all nine confusion-matrix cells,
//! intersects each row with the probability simplex, and evaluates the deployed policy against
//! every vertex of that feasible matrix set. Because the fixed-policy risk is linear in the true
//! confusion matrix, its maximum over this polytope occurs at a vertex. This is a finite synthetic
//! oracle, not evidence that calibration remains valid under distribution shift or dependence.

const PROBABILITY_SCALE: u16 = 1_000;
const FAMILYWISE_ALPHA: f64 = 0.05;
const SIMULTANEOUS_CELL_TAIL_ALPHA: f64 = FAMILYWISE_ALPHA / 18.0;
const MAX_EXACT_F64_INTEGER: u64 = (1_u64 << 53) - 1;
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
    0.999_999_999_999_809_93,
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
    if total == 0 || successes > total || total > MAX_EXACT_F64_INTEGER {
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
        if total == 0 || total > MAX_EXACT_F64_INTEGER {
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
