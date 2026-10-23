//! Cost-aware evidence selection when sensor reliability is estimated from finite calibration data.
//!
//! This fixture converts binomial correctness counts into a conservative, outward-rounded
//! Wilson score interval, then evaluates the minimum expected value of sample information (EVSI)
//! over every per-mille reliability in that interval. The resulting decision remains synthetic:
//! the interval assumes independent, identically distributed Bernoulli calibration outcomes and
//! does not establish coverage under distribution shift or dependent samples. The link from one
//! scalar reliability to a confusion matrix is valid only for the explicitly declared symmetric
//! sensor family; real use would require class-conditional calibration.

const PROBABILITY_SCALE: u16 = 1_000;
const WILSON_Z_95: f64 = 1.959_963_984_540_054;
const COST_MILLIUNITS: u32 = 1_000;
const COST_PENALTY_PER_MILLIUNIT: f64 = 0.0001;
const EPSILON: f64 = 1.0e-12;
const PRIOR_WEIGHTS: [u64; 3] = [6, 3, 1];
const DECISION_LOSSES: [[u32; 3]; 2] = [[0, 100, 100], [4, 4, 4]];
const MAX_EXACT_F64_INTEGER: u64 = (1_u64 << 53) - 1;

/// Returns a two-sided 95% Wilson score interval on a per-mille grid.
///
/// Bounds are rounded *outward*: floor the lower endpoint and ceil the upper endpoint. This
/// avoids narrowing the computed interval during discretization. Counts must be valid, nonempty,
/// and within the exact-integer range of f64 so the numerical contract is explicit.
fn wilson_interval_per_mille(correct: u64, total: u64) -> Option<(u16, u16)> {
    if total == 0 || correct > total || total > MAX_EXACT_F64_INTEGER {
        return None;
    }

    let n = total as f64;
    let proportion = correct as f64 / n;
    let z_squared = WILSON_Z_95 * WILSON_Z_95;
    let denominator = 1.0 + z_squared / n;
    let center = (proportion + z_squared / (2.0 * n)) / denominator;
    let half_width = WILSON_Z_95
        * (proportion * (1.0 - proportion) / n + z_squared / (4.0 * n * n)).sqrt()
        / denominator;
    let lower = (center - half_width).clamp(0.0, 1.0);
    let upper = (center + half_width).clamp(0.0, 1.0);

    let lower_per_mille = (lower * f64::from(PROBABILITY_SCALE)).floor() as u16;
    let upper_per_mille = (upper * f64::from(PROBABILITY_SCALE)).ceil() as u16;
    if lower_per_mille > upper_per_mille || upper_per_mille > PROBABILITY_SCALE {
        return None;
    }
    Some((lower_per_mille, upper_per_mille))
}

fn likelihoods_for_reliability(reliability: u16) -> Option<[[u16; 3]; 3]> {
    if reliability > PROBABILITY_SCALE {
        return None;
    }
    let error = PROBABILITY_SCALE - reliability;
    Some([
        [reliability, error, 0],
        [error, reliability, 0],
        [error, reliability, 0],
    ])
}

fn baseline_expected_loss() -> f64 {
    // The restore-chair decision has prior expected loss 40; safe-hold has loss 4.
    4.0
}

fn expected_loss_after_measurement(reliability: u16) -> Option<f64> {
    let likelihoods = likelihoods_for_reliability(reliability)?;
    let total_weight = PRIOR_WEIGHTS.iter().sum::<u64>() as f64;
    let mut expected_loss = 0.0;

    for outcome_index in 0..3 {
        let best_joint_loss = DECISION_LOSSES
            .iter()
            .map(|decision_losses| {
                PRIOR_WEIGHTS
                    .iter()
                    .enumerate()
                    .map(|(hypothesis_index, weight)| {
                        *weight as f64
                            * f64::from(decision_losses[hypothesis_index])
                            * f64::from(likelihoods[hypothesis_index][outcome_index])
                    })
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
        expected_loss += best_joint_loss / (total_weight * f64::from(PROBABILITY_SCALE));
    }
    Some(expected_loss)
}

fn evsi(reliability: u16) -> Option<f64> {
    Some((baseline_expected_loss() - expected_loss_after_measurement(reliability)?).max(0.0))
}

/// Minimum EVSI over the entire closed integer per-mille interval, not just its midpoint.
fn worst_case_evsi(lower: u16, upper: u16) -> Option<f64> {
    if lower > upper || upper > PROBABILITY_SCALE {
        return None;
    }
    (lower..=upper)
        .map(evsi)
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .min_by(f64::total_cmp)
}

/// Acquire evidence only if its conservative value exceeds the declared cost penalty.
fn should_acquire(correct: u64, total: u64) -> Option<bool> {
    let (lower, upper) = wilson_interval_per_mille(correct, total)?;
    let worst_case_net_value =
        worst_case_evsi(lower, upper)? - f64::from(COST_MILLIUNITS) * COST_PENALTY_PER_MILLIUNIT;
    Some(worst_case_net_value > EPSILON)
}

#[test]
fn wilson_bounds_are_known_and_rounded_outward() {
    assert_eq!(wilson_interval_per_mille(95, 100), Some((888, 979)));
    assert_eq!(wilson_interval_per_mille(99, 100), Some((945, 999)));
    assert_eq!(wilson_interval_per_mille(100, 100), Some((963, 1_000)));
    assert_eq!(wilson_interval_per_mille(20, 20), Some((838, 1_000)));
}

#[test]
fn a_95_percent_point_estimate_is_not_enough_when_its_interval_crosses_the_decision_threshold() {
    let interval = wilson_interval_per_mille(95, 100).expect("valid calibration counts");
    assert_eq!(interval, (888, 979));
    assert_eq!(worst_case_evsi(interval.0, interval.1), Some(0.0));
    assert_eq!(should_acquire(95, 100), Some(false));
}

#[test]
fn additional_calibration_evidence_can_make_worst_case_net_value_positive() {
    let interval_99 = wilson_interval_per_mille(99, 100).expect("valid calibration counts");
    let worst_99 = worst_case_evsi(interval_99.0, interval_99.1).expect("valid interval");
    assert!((worst_99 - 0.156).abs() <= 1.0e-9);
    assert_eq!(should_acquire(99, 100), Some(true));

    let interval_100 = wilson_interval_per_mille(100, 100).expect("valid calibration counts");
    let worst_100 = worst_case_evsi(interval_100.0, interval_100.1).expect("valid interval");
    assert!((worst_100 - 0.8904).abs() <= 1.0e-9);
    assert_eq!(should_acquire(100, 100), Some(true));
}

#[test]
fn a_small_perfect_calibration_sample_still_abstains() {
    // Perfect observed accuracy on only 20 trials leaves a broad interval whose lower endpoint
    // does not clear the decision-sensitive reliability threshold.
    let interval = wilson_interval_per_mille(20, 20).expect("valid calibration counts");
    assert_eq!(interval, (838, 1_000));
    assert_eq!(worst_case_evsi(interval.0, interval.1), Some(0.0));
    assert_eq!(should_acquire(20, 20), Some(false));
}

#[test]
fn malformed_counts_and_reliability_intervals_fail_closed() {
    assert_eq!(wilson_interval_per_mille(0, 0), None);
    assert_eq!(wilson_interval_per_mille(101, 100), None);
    assert_eq!(
        wilson_interval_per_mille(1, MAX_EXACT_F64_INTEGER + 1),
        None
    );
    assert_eq!(worst_case_evsi(951, 950), None);
    assert_eq!(worst_case_evsi(0, 1_001), None);
    assert_eq!(evsi(1_001), None);
}

#[test]
fn robust_value_is_computed_across_the_full_interval_not_at_the_point_estimate() {
    let interval = wilson_interval_per_mille(99, 100).expect("valid calibration counts");
    let all_values: Vec<f64> = (interval.0..=interval.1)
        .map(|reliability| evsi(reliability).expect("per-mille reliability is bounded"))
        .collect();
    let observed_minimum = all_values
        .into_iter()
        .min_by(f64::total_cmp)
        .expect("nonempty interval");
    assert!((observed_minimum - 0.156).abs() <= EPSILON);
    assert!((observed_minimum - evsi(990).expect("valid point estimate")).abs() > 0.1);
}
