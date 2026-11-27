//! Gaussian predictive distribution -> finite categorical likelihoods.
//! Standalone numerical adapter; not a planner. Thresholds must represent
//! actual instrument quantization or a frozen benchmark outcome partition.
//! Standard-normal tails use direct 64-point Gauss-Legendre integration.

use std::sync::OnceLock;

const N: usize = 64;
const CUTOFF: f64 = 12.0;
const SUM_TOL: f64 = 1.0e-11;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    NonFiniteMean, InvalidVariance, NonFiniteThreshold,
    ThresholdsNotStrictlyIncreasing, NumericFailure,
    ProbabilityNormalizationFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianBinLikelihoods {
    pub mean: f64,
    pub variance: f64,
    pub thresholds: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub probability_sum: f64,
}

fn legendre_rule(n: usize) -> Result<Vec<(f64, f64)>, Error> {
    let mut rule = Vec::with_capacity(n);
    for i in 0..((n + 1) / 2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut converged = false;
        for _ in 0..100 {
            let mut p0 = 1.0;
            let mut p1 = 0.0;
            for j in 1..=n {
                let p2 = p1;
                p1 = p0;
                p0 = ((2 * j - 1) as f64 * z * p1 - (j - 1) as f64 * p2) / j as f64;
            }
            let dp = n as f64 * (z * p0 - p1) / (z * z - 1.0);
            if !dp.is_finite() || dp == 0.0 { return Err(Error::NumericFailure); }
            let next = z - p0 / dp;
            if (next - z).abs() <= 2.0 * f64::EPSILON * next.abs().max(1.0) {
                z = next;
                converged = true;
                break;
            }
            z = next;
        }
        if !converged { return Err(Error::NumericFailure); }
        let mut p0 = 1.0;
        let mut p1 = 0.0;
        for j in 1..=n {
            let p2 = p1;
            p1 = p0;
            p0 = ((2 * j - 1) as f64 * z * p1 - (j - 1) as f64 * p2) / j as f64;
        }
        let dp = n as f64 * (z * p0 - p1) / (z * z - 1.0);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);
        if !w.is_finite() || w <= 0.0 { return Err(Error::NumericFailure); }
        rule.push((-z, w));
        if n - 1 - i != i { rule.push((z, w)); }
    }
    rule.sort_by(|a, b| a.0.total_cmp(&b.0));
    if rule.len() != n { return Err(Error::NumericFailure); }
    Ok(rule)
}

fn cached_legendre_rule() -> Result<&'static [(f64, f64)], Error> {
    static RULE: OnceLock<Result<Vec<(f64, f64)>, Error>> = OnceLock::new();
    match RULE.get_or_init(|| legendre_rule(N)) {
        Ok(rule) => Ok(rule.as_slice()),
        Err(error) => Err(error.clone()),
    }
}

/// Standard-normal probability for the half-open interval (lower, upper].
/// The 64-point Gauss–Legendre rule is initialized once and reused across calls.
pub fn standard_normal_interval_probability(
    lower: f64,
    upper: f64,
) -> Result<f64, Error> {
    if lower.is_nan() || upper.is_nan() || upper < lower {
        return Err(Error::NumericFailure);
    }
    if lower == upper {
        return Ok(0.0);
    }
    phi_interval(lower, upper, cached_legendre_rule()?)
}

fn integrate_phi(a: f64, b: f64, rule: &[(f64, f64)]) -> Result<f64, Error> {
    if !a.is_finite() || a < 0.0 || b.is_nan() || b <= a {
        return Err(Error::NumericFailure);
    }
    let upper = b.min(CUTOFF);
    if upper <= a { return Ok(0.0); }
    let mid = 0.5 * (a + upper);
    let radius = 0.5 * (upper - a);
    let mut sum = 0.0;
    for (node, weight) in rule {
        let x = mid + radius * node;
        sum += weight * (-0.5 * x * x).exp();
    }
    let p = radius * INV_SQRT_2PI * sum;
    if p.is_finite() && p >= 0.0 { Ok(p) } else { Err(Error::NumericFailure) }
}

fn q_nonnegative(x: f64, rule: &[(f64, f64)]) -> Result<f64, Error> {
    if x.is_nan() || x < 0.0 { return Err(Error::NumericFailure); }
    if x >= CUTOFF { return Ok(0.0); }
    integrate_phi(x, CUTOFF, rule)
}

/// Standard-normal mass in (lower, upper], with infinite outer bounds.
fn phi_interval(lower: f64, upper: f64, rule: &[(f64, f64)]) -> Result<f64, Error> {
    if lower.is_nan() || upper.is_nan() || upper <= lower {
        return Err(Error::NumericFailure);
    }
    if lower >= 0.0 {
        let ql = q_nonnegative(lower, rule)?;
        let qu = if upper.is_infinite() { 0.0 } else { q_nonnegative(upper, rule)? };
        return Ok((ql - qu).max(0.0));
    }
    if upper <= 0.0 {
        let ql = if upper.is_infinite() { 0.0 } else { q_nonnegative(-upper, rule)? };
        let qu = if lower.is_infinite() { 0.0 } else { q_nonnegative(-lower, rule)? };
        return Ok((ql - qu).max(0.0));
    }
    let left = if lower.is_infinite() { 0.0 } else { q_nonnegative(-lower, rule)? };
    let right = if upper.is_infinite() { 0.0 } else { q_nonnegative(upper, rule)? };
    Ok((1.0 - left - right).max(0.0))
}

/// Map N(mean, variance) to bins (-inf,e0], (e0,e1], ..., (ek,+inf).
pub fn gaussian_bin_likelihoods(
    mean: f64,
    variance: f64,
    thresholds: &[f64],
) -> Result<GaussianBinLikelihoods, Error> {
    if !mean.is_finite() { return Err(Error::NonFiniteMean); }
    if !variance.is_finite() || variance < 0.0 { return Err(Error::InvalidVariance); }
    for (i, edge) in thresholds.iter().enumerate() {
        if !edge.is_finite() { return Err(Error::NonFiniteThreshold); }
        if i > 0 && thresholds[i - 1] >= *edge {
            return Err(Error::ThresholdsNotStrictlyIncreasing);
        }
    }
    let mut probabilities = vec![0.0; thresholds.len() + 1];
    if variance == 0.0 {
        probabilities[thresholds.partition_point(|edge| mean > *edge)] = 1.0;
    } else {
        let sd = variance.sqrt();
        if !sd.is_finite() || sd <= 0.0 { return Err(Error::NumericFailure); }
        let rule = cached_legendre_rule()?;
        let mut lower = f64::NEG_INFINITY;
        for (i, edge) in thresholds.iter().enumerate() {
            let upper = (*edge - mean) / sd;
            probabilities[i] = phi_interval(lower, upper, rule)?;
            lower = upper;
        }
        probabilities[thresholds.len()] = phi_interval(lower, f64::INFINITY, rule)?;
    }
    let sum: f64 = probabilities.iter().sum();
    if !sum.is_finite()
        || probabilities.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (sum - 1.0).abs() > SUM_TOL
    {
        return Err(Error::ProbabilityNormalizationFailure);
    }
    Ok(GaussianBinLikelihoods {
        mean, variance, thresholds: thresholds.to_vec(), probabilities, probability_sum: sum
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a:f64,b:f64,t:f64) {
        assert!((a-b).abs()<=t,"actual={a:.17}, expected={b:.17}, tolerance={t:e}");
    }

    #[test]
    fn cached_interval_kernel_matches_direct_tail_known_answers() {
        close(
            standard_normal_interval_probability(8.0, 9.0).unwrap(),
            6.21983198586573e-16,
            1.0e-17,
        );
        close(
            standard_normal_interval_probability(f64::NEG_INFINITY, f64::INFINITY).unwrap(),
            1.0,
            1.0e-15,
        );
    }

    #[test]
    fn standard_normal_binary_threshold_is_balanced() {
        let r=gaussian_bin_likelihoods(0.0,1.0,&[0.0]).unwrap();
        close(r.probabilities[0],0.5,2.0e-14);
        close(r.probabilities[1],0.5,2.0e-14);
    }

    #[test]
    fn standard_normal_three_bins_match_known_probabilities() {
        let r=gaussian_bin_likelihoods(0.0,1.0,&[-1.0,1.0]).unwrap();
        close(r.probabilities[0],0.15865525393145707,3.0e-14);
        close(r.probabilities[1],0.6826894921370859,3.0e-14);
        close(r.probabilities[2],0.15865525393145707,3.0e-14);
    }

    #[test]
    fn extreme_positive_tail_is_not_lost_to_cdf_cancellation() {
        let r=gaussian_bin_likelihoods(0.0,1.0,&[8.0,9.0]).unwrap();
        close(r.probabilities[1],6.21983198586573e-16,1.0e-17);
        assert!(r.probabilities[2]>0.0);
    }

    #[test]
    fn shifted_scaled_gaussian_uses_standardized_thresholds() {
        let r=gaussian_bin_likelihoods(2.0,4.0,&[2.0]).unwrap();
        close(r.probabilities[0],0.5,2.0e-14);
        close(r.probabilities[1],0.5,2.0e-14);
    }

    #[test]
    fn deterministic_measurement_uses_half_open_bins() {
        assert_eq!(gaussian_bin_likelihoods(1.0,0.0,&[1.0]).unwrap().probabilities,vec![1.0,0.0]);
        assert_eq!(gaussian_bin_likelihoods(1.0001,0.0,&[1.0]).unwrap().probabilities,vec![0.0,1.0]);
    }

    #[test]
    fn no_thresholds_produce_one_all_real_bin() {
        assert_eq!(gaussian_bin_likelihoods(3.0,2.0,&[]).unwrap().probabilities,vec![1.0]);
    }

    #[test]
    fn invalid_inputs_and_unsorted_thresholds_fail_closed() {
        assert_eq!(gaussian_bin_likelihoods(0.0,-1.0,&[0.0]),Err(Error::InvalidVariance));
        assert_eq!(gaussian_bin_likelihoods(0.0,1.0,&[1.0,0.0]),Err(Error::ThresholdsNotStrictlyIncreasing));
        assert_eq!(gaussian_bin_likelihoods(0.0,1.0,&[f64::NAN]),Err(Error::NonFiniteThreshold));
    }

    #[test]
    fn output_is_a_valid_finite_likelihood_row() {
        let r=gaussian_bin_likelihoods(-0.4,0.7,&[-1.0,0.0,1.2]).unwrap();
        assert_eq!(r.probabilities.len(),4);
        assert!(r.probabilities.iter().all(|p| *p>=0.0 && *p<=1.0));
        close(r.probability_sum,1.0,SUM_TOL);
    }
}
