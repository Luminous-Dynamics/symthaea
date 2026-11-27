//! Bivariate Gaussian rectangle likelihood reference.
//!
//! Run from the repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_bivariate_rectangle.rs -o /tmp/sdcb-bivariate-rectangle
//!   /tmp/sdcb-bivariate-rectangle
//!
//! This bounded two-dimensional evaluator conditions one channel on the other.
//! Its inner normal interval calculation reuses the scalar #13944 reference.

#[path = "sdcb_gaussian_bin_likelihoods.rs"]
mod univariate;

const CUTOFF: f64 = 12.0;
const TRUNCATION_BOUND: f64 = 7.1e-33;
const MAX_CONDITION: f64 = 1.0e8;
const REL_TOL: f64 = 2.0e-10;
const ABS_TOL: f64 = 1.0e-32;
const MAX_DEPTH: usize = 24;
const MAX_OUTER_EVALUATIONS: u64 = 1_000_000;
const MAX_CELLS: usize = 64;
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RectangleError {
    NonFiniteMean, NonFiniteCovariance, AsymmetricCovariance,
    InvalidVariance, CovarianceNotPositiveSemidefinite,
    IllConditionedCovarianceUnsupported, InvalidBounds,
    NonFiniteBoundIsNaN, IntegrationDidNotConverge,
    EvaluationBudgetExceeded, ProbabilityBelowResolution,
    NumericFailure, InvalidThresholds, TooManyJointCells,
    JointNormalizationFailure, MarginalizationFailure,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BivariateRectangleEstimate {
    pub probability: f64,
    /// Outer refinement diagnostic, not a certified total error bound.
    pub outer_quadrature_error_indicator: f64,
    /// Union-bound allowance for truncating both independent normal coordinates.
    pub truncation_error_bound: f64,
    pub correlation: Option<f64>,
    pub correlation_condition_estimate: Option<f64>,
    pub covariance_rank: u8,
    pub outer_integrand_evaluations: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BivariateJointBins {
    /// Row-major: first-channel bin, then second-channel bin.
    pub probabilities: Vec<Vec<f64>>,
    pub first_channel_marginals: Vec<f64>,
    pub second_channel_marginals: Vec<f64>,
    pub probability_sum: f64,
    pub marginalization_error: f64,
}

fn normal_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard-normal mass in (lower, upper], clipped to +/-12 sigma.
fn normal_interval(lower: f64, upper: f64) -> Result<f64, RectangleError> {
    if lower.is_nan() || upper.is_nan() || upper < lower {
        return Err(RectangleError::InvalidBounds);
    }
    if upper == lower { return Ok(0.0); }
    let lo = lower.max(-CUTOFF);
    let hi = upper.min(CUTOFF);
    if hi <= lo { return Ok(0.0); }
    let mass = univariate::standard_normal_interval_probability(lo, hi)
        .map_err(|_| RectangleError::NumericFailure)?;
    if mass.is_finite() && (0.0..=1.0).contains(&mass) {
        Ok(mass)
    } else {
        Err(RectangleError::NumericFailure)
    }
}

fn deterministic_inside(value: f64, lower: f64, upper: f64) -> bool {
    value > lower && value <= upper
}

fn constrain_standard_normal(
    mean: f64, loading: f64, lower: f64, upper: f64,
    z_lower: &mut f64, z_upper: &mut f64,
) -> bool {
    if loading == 0.0 { return deterministic_inside(mean, lower, upper); }
    let x_lower = (lower - mean) / loading;
    let x_upper = (upper - mean) / loading;
    if loading > 0.0 {
        *z_lower = (*z_lower).max(x_lower);
        *z_upper = (*z_upper).min(x_upper);
    } else {
        *z_lower = (*z_lower).max(x_upper);
        *z_upper = (*z_upper).min(x_lower);
    }
    *z_lower < *z_upper
}

fn rank_one_rectangle(
    mean: [f64; 2], loading: [f64; 2], lower: [f64; 2], upper: [f64; 2],
    correlation: Option<f64>,
) -> Result<BivariateRectangleEstimate, RectangleError> {
    let mut z_lower = f64::NEG_INFINITY;
    let mut z_upper = f64::INFINITY;
    for i in 0..2 {
        if !constrain_standard_normal(mean[i], loading[i], lower[i], upper[i], &mut z_lower, &mut z_upper) {
            return Ok(BivariateRectangleEstimate {
                probability: 0.0, outer_quadrature_error_indicator: 0.0,
                truncation_error_bound: TRUNCATION_BOUND, correlation,
                correlation_condition_estimate: None, covariance_rank: 1, outer_integrand_evaluations: 0,
            });
        }
    }
    if z_lower >= z_upper {
        return Ok(BivariateRectangleEstimate {
            probability: 0.0, outer_quadrature_error_indicator: 0.0,
            truncation_error_bound: TRUNCATION_BOUND, correlation,
            correlation_condition_estimate: None, covariance_rank: 1, outer_integrand_evaluations: 0,
        });
    }
    if z_lower >= CUTOFF || z_upper <= -CUTOFF {
        return Err(RectangleError::ProbabilityBelowResolution);
    }
    let probability = normal_interval(z_lower, z_upper)?;
    if probability <= TRUNCATION_BOUND { return Err(RectangleError::ProbabilityBelowResolution); }
    Ok(BivariateRectangleEstimate {
        probability, outer_quadrature_error_indicator: 0.0,
        truncation_error_bound: TRUNCATION_BOUND, correlation,
        correlation_condition_estimate: None, covariance_rank: 1, outer_integrand_evaluations: 0,
    })
}

fn within_outer_evaluation_budget(current: u64, additional: u64) -> bool {
    additional <= MAX_OUTER_EVALUATIONS.saturating_sub(current)
}

fn simpson(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> f64 {
    (b - a) * (fa + 4.0 * fm + fb) / 6.0
}

fn refine<F>(
    f: &mut F, a: f64, b: f64, fa: f64, fm: f64, fb: f64,
    whole: f64, abs_tol: f64, depth: usize, evals: &mut u64,
) -> Result<(f64, f64), RectangleError>
where F: FnMut(f64) -> Result<f64, RectangleError> {
    if !within_outer_evaluation_budget(*evals, 2) { return Err(RectangleError::EvaluationBudgetExceeded); }
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    if lm == a || rm == b { return Err(RectangleError::NumericFailure); }
    let fl = f(lm)?;
    let fr = f(rm)?;
    *evals += 2;
    let left = simpson(a, m, fa, fl, fm);
    let right = simpson(m, b, fm, fr, fb);
    let refined = left + right;
    let delta = refined - whole;
    let error = delta.abs() / 15.0;
    if error <= abs_tol {
        let corrected = refined + delta / 15.0;
        return if corrected.is_finite() && corrected >= 0.0 {
            Ok((corrected, error))
        } else { Err(RectangleError::NumericFailure) };
    }
    if depth == 0 { return Err(RectangleError::IntegrationDidNotConverge); }
    let (lv, le) = refine(f, a, m, fa, fl, fm, left, abs_tol * 0.5, depth - 1, evals)?;
    let (rv, re) = refine(f, m, b, fm, fr, fb, right, abs_tol * 0.5, depth - 1, evals)?;
    Ok((lv + rv, le + re))
}

fn add_transition_points(
    points: &mut Vec<f64>, threshold: f64, rho: f64, conditional_sd: f64,
    lower: f64, upper: f64,
) {
    if !threshold.is_finite() || rho.abs() < 1.0e-8 { return; }
    let center = threshold / rho;
    let width = conditional_sd / rho.abs();
    for k in [-8.0, -4.0, 0.0, 4.0, 8.0] {
        let x = center + k * width;
        if x > lower && x < upper { points.push(x); }
    }
}

/// Compute P(lower < Y <= upper) for a two-channel Gaussian.
/// Rank-deficient PSD covariance is integrated on its exact support; no jitter is used.
pub fn bivariate_gaussian_rectangle(
    mean: [f64; 2], covariance: [[f64; 2]; 2],
    lower: [f64; 2], upper: [f64; 2],
) -> Result<BivariateRectangleEstimate, RectangleError> {
    if mean.iter().any(|x| !x.is_finite()) { return Err(RectangleError::NonFiniteMean); }
    if covariance.iter().flatten().any(|x| !x.is_finite()) {
        return Err(RectangleError::NonFiniteCovariance);
    }
    if lower.iter().chain(upper.iter()).any(|x| x.is_nan()) {
        return Err(RectangleError::NonFiniteBoundIsNaN);
    }
    if lower[0] > upper[0] || lower[1] > upper[1] {
        return Err(RectangleError::InvalidBounds);
    }
    if covariance[0][1] != covariance[1][0] {
        return Err(RectangleError::AsymmetricCovariance);
    }
    let v1 = covariance[0][0];
    let v2 = covariance[1][1];
    if v1 < 0.0 || v2 < 0.0 { return Err(RectangleError::InvalidVariance); }
    let c12 = covariance[0][1];

    if v1 == 0.0 && v2 == 0.0 {
        if c12 != 0.0 { return Err(RectangleError::CovarianceNotPositiveSemidefinite); }
        let probability = if deterministic_inside(mean[0], lower[0], upper[0])
            && deterministic_inside(mean[1], lower[1], upper[1]) { 1.0 } else { 0.0 };
        return Ok(BivariateRectangleEstimate {
            probability, outer_quadrature_error_indicator: 0.0, truncation_error_bound: 0.0,
            correlation: None, correlation_condition_estimate: None, covariance_rank: 0,
            outer_integrand_evaluations: 0,
        });
    }

    if v1 == 0.0 || v2 == 0.0 {
        if c12 != 0.0 { return Err(RectangleError::CovarianceNotPositiveSemidefinite); }
        let loading = if v1 == 0.0 { [0.0, v2.sqrt()] } else { [v1.sqrt(), 0.0] };
        return rank_one_rectangle(mean, loading, lower, upper, None);
    }

    let s1 = v1.sqrt();
    let s2 = v2.sqrt();
    if !s1.is_finite() || !s2.is_finite() || s1 <= 0.0 || s2 <= 0.0 {
        return Err(RectangleError::NumericFailure);
    }
    let rho = (c12 / s1) / s2;
    if !rho.is_finite() { return Err(RectangleError::NumericFailure); }
    if rho.abs() > 1.0 {
        return Err(RectangleError::CovarianceNotPositiveSemidefinite);
    }
    if rho.abs() >= 1.0 {
        return rank_one_rectangle(mean, [s1, c12 / s1], lower, upper, Some(rho.signum()));
    }
    let condition = (1.0 + rho.abs()) / (1.0 - rho.abs());
    if condition > MAX_CONDITION {
        return Err(RectangleError::IllConditionedCovarianceUnsupported);
    }

    if lower[0] == upper[0] || lower[1] == upper[1] {
        return Ok(BivariateRectangleEstimate {
            probability: 0.0, outer_quadrature_error_indicator: 0.0,
            truncation_error_bound: TRUNCATION_BOUND, correlation: Some(rho),
            correlation_condition_estimate: Some(condition), covariance_rank: 2,
            outer_integrand_evaluations: 0,
        });
    }

    let a1 = ((lower[0] - mean[0]) / s1).max(-CUTOFF);
    let b1 = ((upper[0] - mean[0]) / s1).min(CUTOFF);
    if b1 <= a1 { return Err(RectangleError::ProbabilityBelowResolution); }
    let a2 = (lower[1] - mean[1]) / s2;
    let b2 = (upper[1] - mean[1]) / s2;
    let conditional_sd = (1.0 - rho * rho).sqrt();
    if !conditional_sd.is_finite() || conditional_sd <= 0.0 {
        return Err(RectangleError::NumericFailure);
    }

    // Break at conditional transition locations to expose narrow high-rho changes.
    let mut points = vec![a1, b1];
    add_transition_points(&mut points, a2, rho, conditional_sd, a1, b1);
    add_transition_points(&mut points, b2, rho, conditional_sd, a1, b1);
    points.sort_by(f64::total_cmp);
    points.dedup_by(|a, b| (*a - *b).abs() <= 8.0 * f64::EPSILON * a.abs().max(b.abs()).max(1.0));

    let max_width = (2.0 * conditional_sd / rho.abs().max(0.01)).max(0.01).min(0.5);
    let full_width = b1 - a1;
    let mut evals = 0_u64;
    let mut probability = 0.0;
    let mut error_indicator = 0.0;
    let mut integrand = |z1: f64| -> Result<f64, RectangleError> {
        let lo = (a2 - rho * z1) / conditional_sd;
        let hi = (b2 - rho * z1) / conditional_sd;
        Ok(normal_pdf(z1) * normal_interval(lo, hi)?)
    };

    // First estimate the total probability coarsely, then distribute one
    // global absolute error budget across all segments. A local relative
    // tolerance would over-refine negligible far-tail pieces unnecessarily.
    let mut segments: Vec<(f64, f64, f64, f64, f64, f64)> = Vec::new();
    let mut rough_probability = 0.0;
    for pair in points.windows(2) {
        let start = pair[0];
        let end = pair[1];
        if end <= start { continue; }
        let count = ((end - start) / max_width).ceil().max(1.0) as usize;
        for i in 0..count {
            let a = start + (end - start) * i as f64 / count as f64;
            let b = start + (end - start) * (i + 1) as f64 / count as f64;
            if !within_outer_evaluation_budget(evals, 3) {
                return Err(RectangleError::EvaluationBudgetExceeded);
            }
            let midpoint = 0.5 * (a + b);
            let fa = integrand(a)?;
            let fm = integrand(midpoint)?;
            let fb = integrand(b)?;
            evals += 3;
            let whole = simpson(a, b, fa, fm, fb);
            rough_probability += whole;
            segments.push((a, b, fa, fm, fb, whole));
        }
    }
    if !rough_probability.is_finite() || rough_probability < 0.0 {
        return Err(RectangleError::NumericFailure);
    }
    let global_tolerance = ABS_TOL.max(REL_TOL * rough_probability.abs());
    for (a, b, fa, fm, fb, whole) in segments {
        let local_tolerance = global_tolerance * ((b - a) / full_width);
        let (value, error) = refine(
            &mut integrand, a, b, fa, fm, fb, whole,
            local_tolerance, MAX_DEPTH, &mut evals,
        )?;
        probability += value;
        error_indicator += error;
    }

    if !probability.is_finite() || probability < 0.0 || probability > 1.0 + 1.0e-10 {
        return Err(RectangleError::NumericFailure);
    }
    if probability <= 10.0 * TRUNCATION_BOUND {
        return Err(RectangleError::ProbabilityBelowResolution);
    }
    Ok(BivariateRectangleEstimate {
        probability: probability.min(1.0),
        outer_quadrature_error_indicator: error_indicator,
        truncation_error_bound: TRUNCATION_BOUND,
        correlation: Some(rho),
        correlation_condition_estimate: Some(condition),
        covariance_rank: 2,
        outer_integrand_evaluations: evals,
    })
}

/// Full row-major joint categorical table for two correlated Gaussian channels.
pub fn bivariate_gaussian_joint_bins(
    mean: [f64; 2], covariance: [[f64; 2]; 2],
    thresholds_1: &[f64], thresholds_2: &[f64],
) -> Result<BivariateJointBins, RectangleError> {
    for edges in [thresholds_1, thresholds_2] {
        for (i, x) in edges.iter().enumerate() {
            if !x.is_finite() || (i > 0 && edges[i - 1] >= *x) {
                return Err(RectangleError::InvalidThresholds);
            }
        }
    }
    let n1 = thresholds_1.len() + 1;
    let n2 = thresholds_2.len() + 1;
    if n1.saturating_mul(n2) > MAX_CELLS { return Err(RectangleError::TooManyJointCells); }

    let mut e1 = vec![f64::NEG_INFINITY];
    e1.extend_from_slice(thresholds_1);
    e1.push(f64::INFINITY);
    let mut e2 = vec![f64::NEG_INFINITY];
    e2.extend_from_slice(thresholds_2);
    e2.push(f64::INFINITY);

    let mut probabilities = vec![vec![0.0; n2]; n1];
    for i in 0..n1 {
        for j in 0..n2 {
            probabilities[i][j] = bivariate_gaussian_rectangle(
                mean, covariance, [e1[i], e2[j]], [e1[i + 1], e2[j + 1]],
            )?.probability;
        }
    }
    let first_channel_marginals: Vec<f64> =
        probabilities.iter().map(|row| row.iter().sum()).collect();
    let second_channel_marginals: Vec<f64> = (0..n2)
        .map(|j| probabilities.iter().map(|row| row[j]).sum()).collect();
    let probability_sum: f64 = probabilities.iter().flatten().sum();

    let scalar_1 = univariate::gaussian_bin_likelihoods(mean[0], covariance[0][0], thresholds_1)
        .map_err(|_| RectangleError::NumericFailure)?;
    let scalar_2 = univariate::gaussian_bin_likelihoods(mean[1], covariance[1][1], thresholds_2)
        .map_err(|_| RectangleError::NumericFailure)?;
    let marginalization_error = first_channel_marginals.iter().zip(&scalar_1.probabilities)
        .chain(second_channel_marginals.iter().zip(&scalar_2.probabilities))
        .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);

    if (probability_sum - 1.0).abs() > 1.0e-9 {
        return Err(RectangleError::JointNormalizationFailure);
    }
    if marginalization_error > 1.0e-9 {
        return Err(RectangleError::MarginalizationFailure);
    }
    Ok(BivariateJointBins {
        probabilities, first_channel_marginals, second_channel_marginals,
        probability_sum, marginalization_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "actual={a:.17e}, expected={b:.17e}, tol={tol:e}");
    }

    #[test]
    fn independent_channels_reduce_to_product_of_marginals() {
        let r = bivariate_gaussian_rectangle(
            [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [-1.0, -0.5], [1.0, 0.8],
        ).unwrap();
        let p1 = univariate::gaussian_bin_likelihoods(0.0, 1.0, &[-1.0, 1.0]).unwrap().probabilities[1];
        let p2 = univariate::gaussian_bin_likelihoods(0.0, 1.0, &[-0.5, 0.8]).unwrap().probabilities[1];
        close(r.probability, p1 * p2, 2.0e-10);
    }

    #[test]
    fn positive_orthant_matches_arcsine_identity() {
        let rho: f64 = 0.6;
        let r = bivariate_gaussian_rectangle(
            [0.0, 0.0], [[1.0, rho], [rho, 1.0]],
            [0.0, 0.0], [f64::INFINITY, f64::INFINITY],
        ).unwrap();
        close(r.probability, 0.25 + rho.asin() / (2.0 * std::f64::consts::PI), 2.0e-10);
    }

    #[test]
    fn negative_orthant_matches_arcsine_identity() {
        let rho: f64 = -0.75;
        let r = bivariate_gaussian_rectangle(
            [0.0, 0.0], [[1.0, rho], [rho, 1.0]],
            [0.0, 0.0], [f64::INFINITY, f64::INFINITY],
        ).unwrap();
        close(r.probability, 0.25 + rho.asin() / (2.0 * std::f64::consts::PI), 2.0e-10);
    }

    #[test]
    fn asymmetric_rectangle_matches_independent_fixture() {
        let r = bivariate_gaussian_rectangle(
            [0.3, -0.2], [[1.0, 0.65], [0.65, 1.5]], [-0.5, -1.1], [1.2, 0.7],
        ).unwrap();
        close(r.probability, 0.35895496926483, 5.0e-10);
    }

    #[test]
    fn correlated_rare_tail_is_preserved() {
        let r = bivariate_gaussian_rectangle(
            [0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], [8.0, 8.0], [9.0, 9.0],
        ).unwrap();
        close(r.probability, 1.7805614713759896e-21, 2.0e-28);
        assert!(r.probability > 0.0);
    }

    #[test]
    fn joint_bins_preserve_correlation_and_marginals() {
        let r = bivariate_gaussian_joint_bins(
            [0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]], &[0.0], &[0.0],
        ).unwrap();
        close(r.probabilities[1][1], 0.35241638234956674, 2.0e-10);
        close(r.probabilities[0][0], 0.35241638234956674, 2.0e-10);
        close(r.probabilities[0][1], 0.14758361765043326, 2.0e-10);
        close(r.probability_sum, 1.0, 1.0e-9);
        assert!(r.marginalization_error < 1.0e-9);
    }

    #[test]
    fn exact_duplicate_channels_are_integrated_on_rank_one_support() {
        let r = bivariate_gaussian_rectangle(
            [0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]],
            [0.0, 0.0], [1.0, 1.0],
        ).unwrap();
        close(r.probability, 0.3413447460685429, 2.0e-13);
        assert_eq!(r.covariance_rank, 1);
        assert_eq!(r.correlation, Some(1.0));
    }

    #[test]
    fn perfect_anticorrelation_uses_intersected_support() {
        let r = bivariate_gaussian_rectangle(
            [0.0, 0.0], [[1.0, -1.0], [-1.0, 1.0]],
            [0.0, -1.0], [1.0, 0.0],
        ).unwrap();
        close(r.probability, 0.3413447460685429, 2.0e-13);
        assert_eq!(r.covariance_rank, 1);
        assert_eq!(r.correlation, Some(-1.0));
    }

    #[test]
    fn deterministic_coordinate_is_respected_for_rank_one_covariance() {
        let r = bivariate_gaussian_rectangle(
            [2.0, 0.0], [[0.0, 0.0], [0.0, 1.0]],
            [1.0, -0.5], [2.0, 0.5],
        ).unwrap();
        close(r.probability, 0.3829249225480262, 2.0e-13);
        assert_eq!(r.covariance_rank, 1);
        assert_eq!(r.correlation, None);
        let excluded = bivariate_gaussian_rectangle(
            [2.0001, 0.0], [[0.0, 0.0], [0.0, 1.0]],
            [1.0, -0.5], [2.0, 0.5],
        ).unwrap();
        assert_eq!(excluded.probability, 0.0);
    }

    #[test]
    fn rank_zero_covariance_is_a_deterministic_rectangle_test() {
        let r = bivariate_gaussian_rectangle(
            [0.0, 2.0], [[0.0, 0.0], [0.0, 0.0]],
            [-1.0, 1.0], [0.0, 3.0],
        ).unwrap();
        assert_eq!(r.probability, 1.0);
        assert_eq!(r.covariance_rank, 0);
    }

    #[test]
    fn non_psd_covariance_is_rejected_without_repair() {
        assert_eq!(
            bivariate_gaussian_rectangle(
                [0.0, 0.0], [[1.0, 1.01], [1.01, 1.0]],
                [-1.0, -1.0], [1.0, 1.0],
            ),
            Err(RectangleError::CovarianceNotPositiveSemidefinite)
        );
    }

    #[test]
    fn even_small_covariance_asymmetry_is_not_silently_averaged() {
        assert_eq!(
            bivariate_gaussian_rectangle(
                [0.0, 0.0], [[1.0, 0.2], [0.2000000000001, 1.0]],
                [-1.0, -1.0], [1.0, 1.0],
            ),
            Err(RectangleError::AsymmetricCovariance)
        );
    }

    #[test]
    fn zero_variance_with_nonzero_cross_covariance_is_not_psd() {
        assert_eq!(
            bivariate_gaussian_rectangle(
                [0.0, 0.0], [[0.0, 0.1], [0.1, 1.0]],
                [-1.0, -1.0], [1.0, 1.0],
            ),
            Err(RectangleError::CovarianceNotPositiveSemidefinite)
        );
    }

    #[test]
    fn ill_conditioned_covariance_is_unsupported_without_jitter() {
        assert_eq!(
            bivariate_gaussian_rectangle([0.0, 0.0], [[1.0, 0.999999999], [0.999999999, 1.0]],
                [-1.0, -1.0], [1.0, 1.0]),
            Err(RectangleError::IllConditionedCovarianceUnsupported)
        );
    }

    #[test]
    fn zero_width_rectangle_has_zero_mass() {
        let r = bivariate_gaussian_rectangle(
            [0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]], [0.2, -1.0], [0.2, 1.0],
        ).unwrap();
        assert_eq!(r.probability, 0.0);
    }

    #[test]
    fn asymmetric_covariance_and_reversed_bounds_fail_closed() {
        assert_eq!(
            bivariate_gaussian_rectangle([0.0, 0.0], [[1.0, 0.2], [0.21, 1.0]],
                [-1.0, -1.0], [1.0, 1.0]),
            Err(RectangleError::AsymmetricCovariance)
        );
        assert_eq!(
            bivariate_gaussian_rectangle([0.0, 0.0], [[1.0, 0.2], [0.2, 1.0]],
                [1.0, -1.0], [0.0, 1.0]),
            Err(RectangleError::InvalidBounds)
        );
    }

    #[test]
    fn outer_evaluation_budget_rejects_overrun_before_each_batch() {
        assert!(within_outer_evaluation_budget(MAX_OUTER_EVALUATIONS - 2, 2));
        assert!(!within_outer_evaluation_budget(MAX_OUTER_EVALUATIONS - 2, 3));
        assert!(!within_outer_evaluation_budget(MAX_OUTER_EVALUATIONS, 1));
    }

    #[test]
    fn integration_reports_diagnostics_and_stays_bounded() {
        let r = bivariate_gaussian_rectangle(
            [0.0, 0.0], [[1.0, 0.4], [0.4, 1.0]], [-1.0, -1.0], [1.0, 1.0],
        ).unwrap();
        assert!(r.outer_integrand_evaluations <= MAX_OUTER_EVALUATIONS);
        assert!(r.outer_quadrature_error_indicator.is_finite());
        assert!(r.truncation_error_bound > 0.0);
    }
}
