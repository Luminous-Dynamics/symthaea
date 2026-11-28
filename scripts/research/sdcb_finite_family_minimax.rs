//! Small exact-oracle fixture for finite-family Gaussian robust testing.
//!
//! Families: B = {N(-1, 1), N(1, 1)}, A = {N(-1/2, 1), N(1/2, 1)}.
//! For alpha=0.05, the GLRT + worst-null p-value test is also the
//! least-favourable-mixture Neyman-Pearson test and is exactly maximin
//! for this symmetric two-point fixture. This is not a general solver.
//!
//! Run:
//!   rustc --edition=2024 --test scripts/research/sdcb_finite_family_minimax.rs -o /tmp/sdcb-minimax-tests
//!   /tmp/sdcb-minimax-tests
//!   rustc --edition=2024 scripts/research/sdcb_finite_family_minimax.rs -o /tmp/sdcb-minimax
//!   /tmp/sdcb-minimax

use std::f64::consts::{LN_2, PI, SQRT_2, TAU};

#[derive(Clone, Copy, Debug, PartialEq)]
struct NormalLaw {
    mean: f64,
    variance: f64,
}

const NULLS: [NormalLaw; 2] = [
    NormalLaw { mean: -1.0, variance: 1.0 },
    NormalLaw { mean: 1.0, variance: 1.0 },
];

const ALTERNATIVES: [NormalLaw; 2] = [
    NormalLaw { mean: -0.5, variance: 1.0 },
    NormalLaw { mean: 0.5, variance: 1.0 },
];

const ALPHA: f64 = 0.05;
const FROZEN_RADIUS: f64 = 0.103_318_479_672_272_44;
const FROZEN_POWER: f64 = 0.072_652_743_731_872_36;
const ROBUST_PAIR_THRESHOLD: f64 = -2.644_853_626_951_473;
const ROBUST_PAIR_WORST_POWER: f64 = 0.000_830_849_737_217_821_2;

fn valid_law(law: NormalLaw) -> bool {
    law.mean.is_finite() && law.variance.is_finite() && law.variance > 0.0
}

/// Numerical Recipes-style erfc approximation; adequate for a small reference
/// fixture, not a certified special-function implementation.
fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let z = x.abs() / SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * z);
    let mut polynomial = 0.170_872_77;
    for coefficient in [
        -0.822_152_23,
        1.488_515_87,
        -1.135_203_98,
        0.278_868_07,
        -0.186_288_06,
        0.096_784_18,
        0.374_091_96,
        1.000_023_68,
    ] {
        polynomial = coefficient + t * polynomial;
    }
    let erfc = t * (-z * z - 1.265_512_23 + t * polynomial).exp();
    if x >= 0.0 { 1.0 - 0.5 * erfc } else { 0.5 * erfc }
}

fn log_pdf(x: f64, law: NormalLaw) -> f64 {
    -0.5 * (((x - law.mean) * (x - law.mean)) / law.variance
        + law.variance.ln()
        + TAU.ln())
}

fn max_log_pdf(x: f64, family: &[NormalLaw]) -> Option<f64> {
    if family.is_empty() || family.iter().any(|law| !valid_law(*law)) || !x.is_finite() {
        return None;
    }
    family
        .iter()
        .map(|law| log_pdf(x, *law))
        .reduce(f64::max)
}

/// Candidate generalized likelihood-ratio score max_A log f - max_B log f.
fn glrt_score(x: f64, nulls: &[NormalLaw], alternatives: &[NormalLaw]) -> Option<f64> {
    let a = max_log_pdf(x, alternatives)?;
    let b = max_log_pdf(x, nulls)?;
    Some(a - b)
}

fn abs_normal_probability(radius: f64, law: NormalLaw) -> Option<f64> {
    if !valid_law(law) || !radius.is_finite() || radius < 0.0 {
        return None;
    }
    let sd = law.variance.sqrt();
    Some(normal_cdf((radius - law.mean) / sd) - normal_cdf((-radius - law.mean) / sd))
}

/// Exact component p-value for this fixture's statistic, which is decreasing
/// in |x|. General families require their own validated score-tail calculation.
fn component_p_value(x: f64, null: NormalLaw) -> Option<f64> {
    if !x.is_finite() {
        return None;
    }
    abs_normal_probability(x.abs(), null)
}

fn family_p_sup(x: f64, nulls: &[NormalLaw]) -> Option<f64> {
    if nulls.is_empty() {
        return None;
    }
    let mut worst: f64 = 0.0;
    for null in nulls {
        worst = worst.max(component_p_value(x, *null)?);
    }
    Some(worst)
}

fn critical_radius(alpha: f64, null: NormalLaw) -> Option<f64> {
    if !(alpha > 0.0 && alpha < 1.0) || !valid_law(null) {
        return None;
    }
    let mut low = 0.0;
    let mut high = 16.0 * null.variance.sqrt() + null.mean.abs();
    if abs_normal_probability(high, null)? < alpha {
        return None;
    }
    for _ in 0..100 {
        let mid = 0.5 * (low + high);
        if abs_normal_probability(mid, null)? < alpha {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(0.5 * (low + high))
}

fn family_intersects(nulls: &[NormalLaw], alternatives: &[NormalLaw]) -> bool {
    nulls.iter().any(|b| {
        alternatives.iter().any(|a| {
            // Exact equality is only valid for canonical fixture parameters.
            // Do not generalize this into tolerance-based law equivalence.
            a.mean == b.mean && a.variance == b.variance
        })
    })
}

fn log_cosh(x: f64) -> f64 {
    let magnitude = x.abs();
    magnitude + (-2.0 * magnitude).exp().ln_1p() - LN_2
}

/// Equal-weight mixture log(q_A/q_B) for the frozen symmetric family.
fn mixture_log_likelihood_ratio(x: f64) -> f64 {
    0.375 + log_cosh(0.5 * x) - log_cosh(x)
}

fn mixture_rejection_region(x: f64, radius: f64) -> bool {
    x.abs() <= radius
}

/// Independent Monte Carlo rank p-value, with conservative inclusive ties.
fn monte_carlo_rank_p_value(observed_score: f64, calibration_scores: &[f64]) -> Option<f64> {
    if !observed_score.is_finite()
        || calibration_scores.is_empty()
        || calibration_scores.iter().any(|score| !score.is_finite())
    {
        return None;
    }
    let exceedances = calibration_scores
        .iter()
        .filter(|score| **score >= observed_score)
        .count();
    Some((1 + exceedances) as f64 / (calibration_scores.len() + 1) as f64)
}

fn inverse_normal_cdf(probability: f64) -> Option<f64> {
    if !(probability > 0.0 && probability < 1.0) {
        return None;
    }
    let mut low = -12.0;
    let mut high = 12.0;
    for _ in 0..120 {
        let mid = 0.5 * (low + high);
        if normal_cdf(mid) < probability {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(0.5 * (low + high))
}

fn robust_single_pair_baseline(alpha: f64) -> Option<(f64, f64, f64)> {
    // Nominal pair A+=N(0.5,1) against B+=N(1,1): reject for X <= c.
    // Calibrate c against the least-favourable null member N(-1,1).
    let z = inverse_normal_cdf(alpha)?;
    let threshold = z - 1.0;
    let worst_alt_power = normal_cdf(threshold - 0.5);
    let other_alt_power = normal_cdf(threshold + 0.5);
    Some((threshold, worst_alt_power, other_alt_power))
}

fn maximin_reference() -> Option<(f64, f64)> {
    let radius = critical_radius(ALPHA, NULLS[0])?;
    let power = abs_normal_probability(radius, ALTERNATIVES[0])?;
    Some((radius, power))
}

fn main() {
    if family_intersects(&NULLS, &ALTERNATIVES) {
        println!("status=FAMILY_INTERSECTION");
        return;
    }
    let (radius, power) = maximin_reference().expect("valid frozen fixture");
    let (threshold, pair_worst, pair_other) =
        robust_single_pair_baseline(ALPHA).expect("valid alpha");
    println!("status=FINITE_FAMILY_ONLY");
    println!("alpha={ALPHA:.8}");
    println!("critical_radius={radius:.12}");
    println!("null_size_minus1={:.8}", abs_normal_probability(radius, NULLS[0]).unwrap());
    println!("null_size_plus1={:.8}", abs_normal_probability(radius, NULLS[1]).unwrap());
    println!("alternative_power_minus_half={power:.8}");
    println!(
        "alternative_power_plus_half={:.8}",
        abs_normal_probability(radius, ALTERNATIVES[1]).unwrap()
    );
    println!("robust_pair_threshold={threshold:.12}");
    println!("robust_pair_worst_power={pair_worst:.8}");
    println!("robust_pair_other_power={pair_other:.8}");
    println!("rust_cdf_is_approximate=true");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.12}, expected={expected:.12}, tolerance={tolerance}"
        );
    }

    #[test]
    fn frozen_families_are_disjoint() {
        assert!(!family_intersects(&NULLS, &ALTERNATIVES));
    }

    #[test]
    fn exact_cross_family_duplicate_triggers_overlap_gate() {
        let alternatives = [ALTERNATIVES[0], NULLS[0]];
        assert!(family_intersects(&NULLS, &alternatives));
    }

    #[test]
    fn glrt_score_is_symmetric_and_decreases_with_absolute_value() {
        let at_zero = glrt_score(0.0, &NULLS, &ALTERNATIVES).unwrap();
        let at_positive = glrt_score(0.25, &NULLS, &ALTERNATIVES).unwrap();
        let at_negative = glrt_score(-0.25, &NULLS, &ALTERNATIVES).unwrap();
        let farther = glrt_score(0.75, &NULLS, &ALTERNATIVES).unwrap();
        close(at_positive, at_negative, 1.0e-12);
        assert!(at_zero > at_positive);
        assert!(at_positive > farther);
        close(at_zero, 0.375, 1.0e-12);
    }

    #[test]
    fn family_p_sup_is_the_maximum_component_tail() {
        let x = 0.08;
        let p_left = component_p_value(x, NULLS[0]).unwrap();
        let p_right = component_p_value(x, NULLS[1]).unwrap();
        let p_sup = family_p_sup(x, &NULLS).unwrap();
        close(p_left, p_right, 2.0e-7);
        close(p_sup, p_left.max(p_right), 1.0e-12);
    }

    #[test]
    fn maximin_test_has_five_percent_size_under_each_null() {
        let radius = critical_radius(ALPHA, NULLS[0]).unwrap();
        for null in NULLS {
            close(abs_normal_probability(radius, null).unwrap(), ALPHA, 2.0e-6);
        }
    }

    #[test]
    fn maximin_test_matches_frozen_worst_case_power() {
        let radius = critical_radius(ALPHA, NULLS[0]).unwrap();
        for alternative in ALTERNATIVES {
            close(
                abs_normal_probability(radius, alternative).unwrap(),
                FROZEN_POWER,
                2.0e-6,
            );
        }
        close(radius, FROZEN_RADIUS, 2.0e-6);
    }

    #[test]
    fn mixture_lr_and_glrt_induce_the_same_rejection_order() {
        let mut prior_glrt = f64::INFINITY;
        let mut prior_mix = f64::INFINITY;
        for i in 0..200 {
            let x = i as f64 / 20.0;
            let glrt = glrt_score(x, &NULLS, &ALTERNATIVES).unwrap();
            let mix = mixture_log_likelihood_ratio(x);
            assert!(glrt < prior_glrt || i == 0);
            assert!(mix < prior_mix || i == 0);
            prior_glrt = glrt;
            prior_mix = mix;
        }
    }

    #[test]
    fn worst_null_recalibration_of_single_pair_is_frozen() {
        let (threshold, worst, other) = robust_single_pair_baseline(ALPHA).unwrap();
        close(threshold, ROBUST_PAIR_THRESHOLD, 2.0e-6);
        close(worst, ROBUST_PAIR_WORST_POWER, 2.0e-6);
        close(other, 0.015_982_275_968_578_4, 2.0e-6);
        close(normal_cdf(threshold + 1.0), ALPHA, 2.0e-6);
        close(normal_cdf(threshold - 1.0), 0.000_133_772_014_598_695_45, 2.0e-6);
    }

    #[test]
    fn expanding_null_family_cannot_reduce_p_sup() {
        let x = 0.1;
        let before = family_p_sup(x, &NULLS).unwrap();
        let expanded = [NULLS[0], NULLS[1], NormalLaw { mean: 0.0, variance: 1.0 }];
        let after = family_p_sup(x, &expanded).unwrap();
        assert!(after >= before);
    }

    #[test]
    fn monte_carlo_rank_uses_inclusive_ties() {
        let p = monte_carlo_rank_p_value(1.0, &[1.0, 0.5, 1.0, 0.0]).unwrap();
        close(p, 3.0 / 5.0, 1.0e-12);
        assert_eq!(monte_carlo_rank_p_value(1.0, &[]), None);
    }

    #[test]
    fn invalid_alpha_fails_closed() {
        assert_eq!(critical_radius(0.0, NULLS[0]), None);
        assert_eq!(critical_radius(1.0, NULLS[0]), None);
        assert_eq!(inverse_normal_cdf(f64::NAN), None);
    }

    #[test]
    fn invalid_gaussian_variance_fails_closed() {
        assert!(max_log_pdf(
            0.0,
            &[NormalLaw { mean: 0.0, variance: 0.0 }]
        )
        .is_none());
    }
}
