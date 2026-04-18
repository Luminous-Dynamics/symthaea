// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Statistics & Probability Engine
//!
//! Probability distributions, descriptive statistics, hypothesis testing,
//! Bayesian inference, and regression — all with HDC encoding and Phi coupling.
//!
//! ## Capabilities
//!
//! - **Descriptive**: mean, variance, std, median, quartiles, skewness, kurtosis
//! - **Distributions**: PDF, CDF, sampling, moments for 8 families
//! - **Hypothesis testing**: t-test, chi-squared goodness-of-fit
//! - **Bayesian inference**: conjugate priors (Normal-Normal, Beta-Binomial)
//! - **Correlation**: Pearson, Spearman rank
//! - **Linear regression**: y = ax + b via least-squares
//!
//! ## HDC Integration
//!
//! Distributions are encoded into [`BinaryHV`] by binding a distribution-type
//! primitive with parameter encodings (mean, variance). Bayesian posterior update
//! maps naturally: `prior.encode().bind(&likelihood.encode()) ≈ posterior.encode()`.
//!
//! ## References
//!
//! - Abramowitz & Stegun (1964) — error function approximation (Eq. 7.1.26)
//! - Lanczos (1964) — gamma function approximation
//! - Wilson & Hilferty (1931) — chi-squared normal approximation
//! - Kanerva (2009) — hyperdimensional computing encodings

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use serde::{Deserialize, Serialize};

// ─── PRNG (Xorshift64) ─────────────────────────────────────────────────────

/// Simple xorshift64 pseudo-random number generator (Marsaglia 2003).
///
/// Deterministic from seed — suitable for reproducible sampling.
/// Period: 2^64 - 1. NOT cryptographically secure.
#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG from the given seed.
    ///
    /// Seed 0 is mapped to 1 to avoid the zero fixed point.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate the next pseudo-random u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a uniform f64 in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Generate a standard normal sample via the Box-Muller transform.
    ///
    /// Returns two independent N(0,1) samples; we discard the second for simplicity.
    pub fn next_standard_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 1e-15 {
                return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }
}

// ─── ProbabilityDistribution Trait ──────────────────────────────────────────

/// Trait for probability distributions with HDC encoding.
///
/// Every distribution can compute its PDF/CDF, moments, draw samples,
/// and encode itself into a [`BinaryHV`] for hyperdimensional reasoning.
pub trait ProbabilityDistribution {
    /// Probability density (continuous) or mass (discrete) function.
    fn pdf(&self, x: f64) -> f64;

    /// Cumulative distribution function P(X ≤ x).
    fn cdf(&self, x: f64) -> f64;

    /// Expected value E[X].
    fn dist_mean(&self) -> f64;

    /// Variance Var(X).
    fn dist_variance(&self) -> f64;

    /// Draw a single sample using the provided PRNG.
    fn sample(&self, rng: &mut Xorshift64) -> f64;

    /// Encode this distribution as a BinaryHV.
    ///
    /// The encoding binds a distribution-type primitive with parameter
    /// encodings derived from the mean and variance, following the HDC
    /// algebra: `type_hv ⊕ mean_hv ⊕ var_hv`.
    fn encode(&self) -> BinaryHV;

    /// Standard deviation √Var(X).
    fn dist_std(&self) -> f64 {
        self.dist_variance().sqrt()
    }
}

impl ProbabilityDistribution for Distribution {
    fn pdf(&self, x: f64) -> f64 {
        Distribution::pdf(self, x)
    }

    fn cdf(&self, x: f64) -> f64 {
        Distribution::cdf(self, x)
    }

    fn dist_mean(&self) -> f64 {
        self.mean_val()
    }

    fn dist_variance(&self) -> f64 {
        self.variance_val()
    }

    fn sample(&self, rng: &mut Xorshift64) -> f64 {
        Distribution::sample(self, rng)
    }

    fn encode(&self) -> BinaryHV {
        Distribution::encode(self)
    }
}

// ─── Descriptive Statistics ──────────────────────────────────────────────────

/// Compute the mean of a slice
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the variance (population)
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
}

/// Compute the sample variance (Bessel-corrected)
pub fn sample_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

/// Standard deviation (population)
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Median
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Quartiles (Q1, Q2/median, Q3)
pub fn quartiles(data: &[f64]) -> (f64, f64, f64) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let q2 = median(data);

    let lower = &sorted[..n / 2];
    let upper = if n.is_multiple_of(2) {
        &sorted[n / 2..]
    } else {
        &sorted[n / 2 + 1..]
    };

    let q1 = if lower.is_empty() { q2 } else { median(lower) };
    let q3 = if upper.is_empty() { q2 } else { median(upper) };

    (q1, q2, q3)
}

/// Skewness (Fisher-Pearson)
pub fn skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 {
        return 0.0;
    }
    let m3: f64 = data.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n;
    m3
}

/// Excess kurtosis
pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 {
        return 0.0;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 {
        return 0.0;
    }
    let m4: f64 = data.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    m4 - 3.0
}

// ─── Probability Distributions ───────────────────────────────────────────────

/// Distribution families
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    /// Normal(μ, σ²)
    Normal { mu: f64, sigma: f64 },
    /// Uniform(a, b)
    Uniform { a: f64, b: f64 },
    /// Bernoulli(p)
    Bernoulli { p: f64 },
    /// Binomial(n, p)
    Binomial { n: u64, p: f64 },
    /// Poisson(λ)
    Poisson { lambda: f64 },
    /// Exponential(λ)
    Exponential { lambda: f64 },
    /// Beta(α, β)
    Beta { alpha: f64, beta: f64 },
    /// Gamma(α, β)
    Gamma { alpha: f64, beta: f64 },
}

impl Distribution {
    /// Probability density/mass function
    pub fn pdf(&self, x: f64) -> f64 {
        match self {
            Distribution::Normal { mu, sigma } => {
                let z = (x - mu) / sigma;
                (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
            }
            Distribution::Uniform { a, b } => {
                if x >= *a && x <= *b {
                    1.0 / (b - a)
                } else {
                    0.0
                }
            }
            Distribution::Bernoulli { p } => {
                if (x - 1.0).abs() < 1e-10 {
                    *p
                } else if x.abs() < 1e-10 {
                    1.0 - p
                } else {
                    0.0
                }
            }
            Distribution::Binomial { n, p } => {
                let k = x.round() as u64;
                if k > *n || x < 0.0 {
                    return 0.0;
                }
                let coeff = binomial_coefficient(*n, k);
                coeff as f64 * p.powi(k as i32) * (1.0 - p).powi((*n - k) as i32)
            }
            Distribution::Poisson { lambda } => {
                let k = x.round() as u64;
                if x < 0.0 {
                    return 0.0;
                }
                (-lambda).exp() * lambda.powi(k as i32) / factorial(k) as f64
            }
            Distribution::Exponential { lambda } => {
                if x < 0.0 {
                    0.0
                } else {
                    lambda * (-lambda * x).exp()
                }
            }
            Distribution::Beta { alpha, beta } => {
                if !(0.0..=1.0).contains(&x) {
                    return 0.0;
                }
                let b_val = beta_function(*alpha, *beta);
                x.powf(alpha - 1.0) * (1.0 - x).powf(beta - 1.0) / b_val
            }
            Distribution::Gamma { alpha, beta } => {
                if x < 0.0 {
                    return 0.0;
                }
                beta.powf(*alpha) / gamma_function(*alpha) * x.powf(alpha - 1.0) * (-beta * x).exp()
            }
        }
    }

    /// Cumulative distribution function
    pub fn cdf(&self, x: f64) -> f64 {
        match self {
            Distribution::Normal { mu, sigma } => {
                0.5 * (1.0 + erf((x - mu) / (sigma * std::f64::consts::SQRT_2)))
            }
            Distribution::Uniform { a, b } => {
                if x < *a {
                    0.0
                } else if x > *b {
                    1.0
                } else {
                    (x - a) / (b - a)
                }
            }
            Distribution::Exponential { lambda } => {
                if x < 0.0 {
                    0.0
                } else {
                    1.0 - (-lambda * x).exp()
                }
            }
            _ => {
                // Numerical CDF via trapezoidal rule for distributions without closed form
                let lower = self.mean_val() - 6.0 * self.std_val();
                let upper = x;
                if upper <= lower {
                    return 0.0;
                }
                let n = 1000;
                let h = (upper - lower) / n as f64;
                let mut sum = 0.0;
                for i in 0..n {
                    let x0 = lower + i as f64 * h;
                    let x1 = x0 + h;
                    sum += (self.pdf(x0) + self.pdf(x1)) / 2.0 * h;
                }
                sum.max(0.0).min(1.0)
            }
        }
    }

    /// Mean
    pub fn mean_val(&self) -> f64 {
        match self {
            Distribution::Normal { mu, .. } => *mu,
            Distribution::Uniform { a, b } => (a + b) / 2.0,
            Distribution::Bernoulli { p } => *p,
            Distribution::Binomial { n, p } => *n as f64 * p,
            Distribution::Poisson { lambda } => *lambda,
            Distribution::Exponential { lambda } => 1.0 / lambda,
            Distribution::Beta { alpha, beta } => alpha / (alpha + beta),
            Distribution::Gamma { alpha, beta } => alpha / beta,
        }
    }

    /// Variance
    pub fn variance_val(&self) -> f64 {
        match self {
            Distribution::Normal { sigma, .. } => sigma * sigma,
            Distribution::Uniform { a, b } => (b - a).powi(2) / 12.0,
            Distribution::Bernoulli { p } => p * (1.0 - p),
            Distribution::Binomial { n, p } => *n as f64 * p * (1.0 - p),
            Distribution::Poisson { lambda } => *lambda,
            Distribution::Exponential { lambda } => 1.0 / (lambda * lambda),
            Distribution::Beta { alpha, beta } => {
                (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0))
            }
            Distribution::Gamma { alpha, beta } => alpha / (beta * beta),
        }
    }

    /// Standard deviation
    pub fn std_val(&self) -> f64 {
        self.variance_val().sqrt()
    }

    /// Draw a single sample from this distribution.
    ///
    /// Uses the provided [`Xorshift64`] PRNG for deterministic, reproducible sampling.
    /// Continuous distributions use inverse-CDF or Box-Muller transforms;
    /// discrete distributions use the appropriate counting methods.
    pub fn sample(&self, rng: &mut Xorshift64) -> f64 {
        match self {
            Distribution::Normal { mu, sigma } => {
                // Box-Muller via the PRNG's helper
                mu + sigma * rng.next_standard_normal()
            }
            Distribution::Uniform { a, b } => {
                // Inverse CDF: F^{-1}(u) = a + u(b - a)
                a + rng.next_f64() * (b - a)
            }
            Distribution::Bernoulli { p } => {
                if rng.next_f64() < *p {
                    1.0
                } else {
                    0.0
                }
            }
            Distribution::Binomial { n, p } => {
                // Sum of n Bernoulli trials
                let mut count = 0u64;
                for _ in 0..*n {
                    if rng.next_f64() < *p {
                        count += 1;
                    }
                }
                count as f64
            }
            Distribution::Poisson { lambda } => {
                // Knuth's algorithm: generate exponential inter-arrivals
                let l = (-lambda).exp();
                let mut k = 0u64;
                let mut p_val = 1.0;
                loop {
                    k += 1;
                    p_val *= rng.next_f64();
                    if p_val < l {
                        break;
                    }
                }
                (k - 1) as f64
            }
            Distribution::Exponential { lambda } => {
                // Inverse CDF: -ln(1-u)/λ
                let u = rng.next_f64();
                -(1.0 - u).ln() / lambda
            }
            Distribution::Beta { alpha, beta } => {
                // Joehnk's method for small parameters, or via Gamma ratio:
                // If X ~ Gamma(α,1), Y ~ Gamma(β,1), then X/(X+Y) ~ Beta(α,β)
                let x = sample_gamma(*alpha, 1.0, rng);
                let y = sample_gamma(*beta, 1.0, rng);
                if x + y > 1e-15 {
                    x / (x + y)
                } else {
                    0.5
                }
            }
            Distribution::Gamma { alpha, beta } => sample_gamma(*alpha, *beta, rng),
        }
    }

    /// HDC encoding of the distribution
    pub fn encode(&self) -> BinaryHV {
        let dist_name = match self {
            Distribution::Normal { .. } => "DIST_NORMAL",
            Distribution::Uniform { .. } => "DIST_UNIFORM",
            Distribution::Bernoulli { .. } => "DIST_BERNOULLI",
            Distribution::Binomial { .. } => "DIST_BINOMIAL",
            Distribution::Poisson { .. } => "DIST_POISSON",
            Distribution::Exponential { .. } => "DIST_EXPONENTIAL",
            Distribution::Beta { .. } => "DIST_BETA",
            Distribution::Gamma { .. } => "DIST_GAMMA",
        };
        let type_hv = BinaryHV::random(seed_from_name(dist_name));
        let mean_hv = BinaryHV::random(seed_from_name(&format!(
            "MEAN_{}",
            self.mean_val().to_bits()
        )));
        let var_hv = BinaryHV::random(seed_from_name(&format!(
            "VAR_{}",
            self.variance_val().to_bits()
        )));
        type_hv.bind(&mean_hv).bind(&var_hv)
    }
}

// ─── Hypothesis Testing ──────────────────────────────────────────────────────

/// Result of a hypothesis test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    /// Name of the test performed
    pub test_name: String,
    /// Computed test statistic (t, χ², etc.)
    pub test_statistic: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// Whether H₀ is rejected at the chosen significance level
    pub reject: bool,
    /// Confidence level (1 − α)
    pub confidence_level: f64,
    /// Confidence interval (lower, upper) for the estimated parameter.
    ///
    /// For t-tests this is the CI for the mean (or difference in means).
    /// For chi-squared this is `(0.0, 0.0)` (not applicable).
    pub confidence_interval: (f64, f64),
    /// Phi (consciousness coupling) — higher when test is informative
    pub phi: f64,
}

/// One-sample t-test: test if mean equals μ₀.
///
/// Uses the normal approximation for the p-value (accurate for n > 30).
/// The confidence interval is for the population mean.
pub fn one_sample_t_test(data: &[f64], mu0: f64, alpha: f64) -> HypothesisTestResult {
    let n = data.len() as f64;
    let m = mean(data);
    let s = sample_variance(data).sqrt();
    let se = s / n.sqrt();
    let t = if se > 1e-15 { (m - mu0) / se } else { 0.0 };

    // Approximate p-value using normal approximation (good for df > 30)
    let p_value = 2.0 * (1.0 - normal_cdf(t.abs()));

    // CI for the mean using z-critical (normal approx)
    let z_crit = z_critical(alpha);
    let ci = (m - z_crit * se, m + z_crit * se);

    HypothesisTestResult {
        test_name: "One-sample t-test".to_string(),
        test_statistic: t,
        p_value,
        reject: p_value < alpha,
        confidence_level: 1.0 - alpha,
        confidence_interval: ci,
        phi: if p_value < alpha { 0.3 } else { 0.1 },
    }
}

/// Two-sample t-test (equal variances assumed, pooled).
///
/// Tests H₀: μ₁ = μ₂. The confidence interval is for the difference μ₁ − μ₂.
pub fn two_sample_t_test(data1: &[f64], data2: &[f64], alpha: f64) -> HypothesisTestResult {
    let n1 = data1.len() as f64;
    let n2 = data2.len() as f64;
    let m1 = mean(data1);
    let m2 = mean(data2);
    let s1 = sample_variance(data1);
    let s2 = sample_variance(data2);

    let sp =
        ((((n1 - 1.0) * s1 + (n2 - 1.0) * s2) / (n1 + n2 - 2.0)) * (1.0 / n1 + 1.0 / n2)).sqrt();

    let t = if sp > 1e-15 { (m1 - m2) / sp } else { 0.0 };

    let p_value = 2.0 * (1.0 - normal_cdf(t.abs()));

    // CI for difference in means
    let z_crit = z_critical(alpha);
    let diff = m1 - m2;
    let ci = (diff - z_crit * sp, diff + z_crit * sp);

    HypothesisTestResult {
        test_name: "Two-sample t-test".to_string(),
        test_statistic: t,
        p_value,
        reject: p_value < alpha,
        confidence_level: 1.0 - alpha,
        confidence_interval: ci,
        phi: if p_value < alpha { 0.3 } else { 0.1 },
    }
}

/// Chi-squared goodness-of-fit test
pub fn chi_squared_test(observed: &[f64], expected: &[f64], alpha: f64) -> HypothesisTestResult {
    assert_eq!(observed.len(), expected.len());

    let chi2: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(o, e)| if *e > 1e-15 { (o - e).powi(2) / e } else { 0.0 })
        .sum();

    let df = (observed.len() - 1) as f64;
    // Approximate p-value using Wilson-Hilferty normal approximation
    let z = (chi2 / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df));
    let z = z / (2.0 / (9.0 * df)).sqrt();
    let p_value = 1.0 - normal_cdf(z);

    HypothesisTestResult {
        test_name: "Chi-squared goodness-of-fit".to_string(),
        test_statistic: chi2,
        p_value,
        reject: p_value < alpha,
        confidence_level: 1.0 - alpha,
        confidence_interval: (0.0, 0.0), // Not applicable for chi-squared GOF
        phi: if p_value < alpha { 0.3 } else { 0.1 },
    }
}

// ─── Bayesian Inference ──────────────────────────────────────────────────────

/// Result of Bayesian posterior update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianResult {
    pub prior: Distribution,
    pub posterior: Distribution,
    pub description: String,
    pub phi: f64,
    pub encoding: BinaryHV,
}

/// Normal-Normal conjugate update.
/// Prior: N(μ₀, σ₀²), Likelihood: N(x̄, σ²/n) → Posterior: N(μ₁, σ₁²)
pub fn normal_normal_update(
    prior_mu: f64,
    prior_sigma: f64,
    data_mean: f64,
    data_sigma: f64,
    n: usize,
) -> BayesianResult {
    let prior_precision = 1.0 / (prior_sigma * prior_sigma);
    let likelihood_precision = n as f64 / (data_sigma * data_sigma);

    let posterior_precision = prior_precision + likelihood_precision;
    let posterior_mu =
        (prior_precision * prior_mu + likelihood_precision * data_mean) / posterior_precision;
    let posterior_sigma = (1.0 / posterior_precision).sqrt();

    let prior = Distribution::Normal {
        mu: prior_mu,
        sigma: prior_sigma,
    };
    let posterior = Distribution::Normal {
        mu: posterior_mu,
        sigma: posterior_sigma,
    };

    let encoding = prior.encode().bind(&posterior.encode());

    BayesianResult {
        prior,
        posterior,
        description: format!(
            "Normal-Normal update: N({:.3}, {:.3}²) → N({:.3}, {:.3}²) after {} observations",
            prior_mu, prior_sigma, posterior_mu, posterior_sigma, n
        ),
        phi: 0.3 + 0.1 * (n as f64).ln().min(2.0),
        encoding,
    }
}

/// Beta-Binomial conjugate update.
/// Prior: Beta(α, β), observe k successes in n trials → Posterior: Beta(α+k, β+n-k)
pub fn beta_binomial_update(
    prior_alpha: f64,
    prior_beta: f64,
    successes: u64,
    trials: u64,
) -> BayesianResult {
    let posterior_alpha = prior_alpha + successes as f64;
    let posterior_beta = prior_beta + (trials - successes) as f64;

    let prior = Distribution::Beta {
        alpha: prior_alpha,
        beta: prior_beta,
    };
    let posterior = Distribution::Beta {
        alpha: posterior_alpha,
        beta: posterior_beta,
    };

    let encoding = prior.encode().bind(&posterior.encode());

    BayesianResult {
        prior,
        posterior,
        description: format!(
            "Beta-Binomial update: Beta({:.1}, {:.1}) → Beta({:.1}, {:.1}) after {}/{} successes",
            prior_alpha, prior_beta, posterior_alpha, posterior_beta, successes, trials
        ),
        phi: 0.3 + 0.1 * (trials as f64).ln().min(2.0),
        encoding,
    }
}

// ─── Correlation ─────────────────────────────────────────────────────────────

/// Pearson correlation coefficient
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let mx = mean(x);
    let my = mean(y);

    let cov: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / n;
    let sx = std_dev(x);
    let sy = std_dev(y);

    if sx < 1e-15 || sy < 1e-15 {
        0.0
    } else {
        cov / (sx * sy)
    }
}

/// Spearman rank correlation
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let rx = rank(x);
    let ry = rank(y);
    pearson_correlation(&rx, &ry)
}

fn rank(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        ranks[idx] = (rank + 1) as f64;
    }
    ranks
}

// ─── Linear Regression ──────────────────────────────────────────────────────

/// Result of a linear regression y = a + bx
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionResult {
    pub intercept: f64,
    pub slope: f64,
    pub r_squared: f64,
    pub residual_std: f64,
    pub phi: f64,
}

/// Simple linear regression: y = a + bx (least-squares)
pub fn linear_regression(x: &[f64], y: &[f64]) -> LinearRegressionResult {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let mx = mean(x);
    let my = mean(y);

    let sxx: f64 = x.iter().map(|xi| (xi - mx).powi(2)).sum();
    let sxy: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mx) * (yi - my))
        .sum();

    let slope = if sxx > 1e-15 { sxy / sxx } else { 0.0 };
    let intercept = my - slope * mx;

    // R²
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| {
            let predicted = intercept + slope * xi;
            (yi - predicted).powi(2)
        })
        .sum();
    let ss_tot: f64 = y.iter().map(|yi| (yi - my).powi(2)).sum();
    let r_squared = if ss_tot > 1e-15 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    let residual_std = if n > 2.0 {
        (ss_res / (n - 2.0)).sqrt()
    } else {
        0.0
    };

    LinearRegressionResult {
        intercept,
        slope,
        r_squared,
        residual_std,
        phi: 0.2 + 0.3 * r_squared,
    }
}

// ─── Helper Functions ────────────────────────────────────────────────────────

/// Error function (Abramowitz and Stegun approximation)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Normal CDF using error function
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Approximate z-critical value for a two-tailed test at significance α.
///
/// Uses the rational approximation of the inverse normal CDF
/// (Abramowitz & Stegun, Eq. 26.2.23, |ε| < 4.5 × 10⁻⁴).
fn z_critical(alpha: f64) -> f64 {
    // We need the (1 - α/2) quantile of the standard normal
    let p = 1.0 - alpha / 2.0;
    inverse_normal_cdf(p)
}

/// Inverse standard normal CDF (probit) via rational approximation.
///
/// Accurate to ~4.5 × 10⁻⁴ for p ∈ (0, 1). Uses the Abramowitz & Stegun
/// approximation (Eq. 26.2.23) with the substitution t = √(-2 ln(1-p)).
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let p_inner = if p < 0.5 { p } else { 1.0 - p };

    let t = (-2.0 * p_inner.ln()).sqrt();

    // Rational approximation coefficients (A&S 26.2.23)
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;

    sign * (t - numerator / denominator)
}

fn factorial(n: u64) -> u64 {
    (1..=n).product::<u64>().max(1)
}

fn binomial_coefficient(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Lanczos approximation of the gamma function
fn gamma_function(x: f64) -> f64 {
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_function(1.0 - x))
    } else {
        let x = x - 1.0;
        let g = 7.0;
        let c = [
            0.999_999_999_999_810,
            676.520_368_121_885_1,
            -1_259.139_216_722_403,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507_343_278_686_905,
            -0.138_571_095_265_720,
            9.984_369_578_019_572e-6,
            1.505_632_735_149_312e-7,
        ];

        let mut sum = c[0];
        for (i, &ci) in c.iter().enumerate().skip(1) {
            sum += ci / (x + i as f64);
        }

        let t = x + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
    }
}

/// Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
fn beta_function(a: f64, b: f64) -> f64 {
    gamma_function(a) * gamma_function(b) / gamma_function(a + b)
}

/// Sample from Gamma(alpha, beta) using Marsaglia & Tsang's method (2000).
///
/// For α ≥ 1 uses the rejection method directly.
/// For α < 1 uses the Ahrens-Dieter boost: Gamma(α,β) = Gamma(α+1,β) × U^(1/α).
fn sample_gamma(alpha: f64, beta: f64, rng: &mut Xorshift64) -> f64 {
    if alpha < 1.0 {
        // Boost: X ~ Gamma(alpha+1, beta), then X * U^(1/alpha) ~ Gamma(alpha, beta)
        let u = rng.next_f64();
        return sample_gamma(alpha + 1.0, beta, rng) * u.powf(1.0 / alpha);
    }

    // Marsaglia & Tsang (2000) for alpha >= 1
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = rng.next_standard_normal();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.next_f64();
        // Squeeze test
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v / beta;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v / beta;
        }
    }
}

// ─── Bootstrap ───────────────────────────────────────────────────────────────

/// LCG PRNG for reproducible bootstrap resampling.
///
/// next = (6364136223846793005 * seed + 1442695040888963407) mod 2^64
#[inline]
fn lcg_next(seed: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Draw n samples with replacement from `data` using LCG with given seed.
fn bootstrap_resample(data: &[f64], n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let len = data.len();
    (0..n)
        .map(|_| {
            s = lcg_next(s);
            data[(s as usize) % len]
        })
        .collect()
}

/// Percentile of sorted data (linear interpolation).
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (idx - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

/// Non-parametric bootstrap confidence interval for the mean.
///
/// Resamples with replacement `n_bootstrap` times, returns `(lower, upper)` at
/// the given confidence level using the percentile method.
pub fn bootstrap_mean_ci(
    data: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64) {
    bootstrap_statistic_ci(data, |d| mean(d), n_bootstrap, confidence, seed)
}

/// Non-parametric bootstrap CI for the standard deviation.
pub fn bootstrap_std_ci(
    data: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64) {
    bootstrap_statistic_ci(data, |d| std_dev(d), n_bootstrap, confidence, seed)
}

/// Non-parametric bootstrap CI for the median.
pub fn bootstrap_median_ci(
    data: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64) {
    bootstrap_statistic_ci(data, |d| median(d), n_bootstrap, confidence, seed)
}

/// Generic non-parametric bootstrap CI for any statistic.
pub fn bootstrap_statistic_ci<F>(
    data: &[f64],
    stat: F,
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    let mut bootstrap_stats: Vec<f64> = (0..n_bootstrap)
        .map(|i| {
            let s = lcg_next(seed.wrapping_add(i as u64));
            let sample = bootstrap_resample(data, n, s);
            stat(&sample)
        })
        .collect();
    bootstrap_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lo = percentile_sorted(&bootstrap_stats, alpha / 2.0);
    let hi = percentile_sorted(&bootstrap_stats, 1.0 - alpha / 2.0);
    (lo, hi)
}

/// Parametric bootstrap: fit Normal to data, return bootstrap distribution of the mean.
pub fn parametric_bootstrap_normal(
    mu: f64,
    sigma: f64,
    n: usize,
    n_bootstrap: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Xorshift64::new(seed);
    let dist = Distribution::Normal { mu, sigma };
    (0..n_bootstrap)
        .map(|_| {
            let samples: Vec<f64> = (0..n).map(|_| dist.sample(&mut rng)).collect();
            mean(&samples)
        })
        .collect()
}

// ─── AR(1) Model ─────────────────────────────────────────────────────────────

/// First-order autoregressive model: x_t = μ + φ(x_{t-1} - μ) + σε_t
#[derive(Debug, Clone)]
pub struct AR1Model {
    /// Autoregressive coefficient (|φ| < 1 for stationarity).
    pub phi: f64,
    /// Innovation standard deviation.
    pub sigma: f64,
    /// Process mean.
    pub mean: f64,
}

impl AR1Model {
    /// Fit AR(1) via Yule-Walker equations.
    ///
    /// φ = cov(x_t, x_{t-1}) / var(x_t)
    /// σ = std(x_t) * sqrt(1 - φ²)
    pub fn fit(data: &[f64]) -> Self {
        if data.len() < 2 {
            return Self { phi: 0.0, sigma: 1.0, mean: 0.0 };
        }
        let mu = mean(data);
        let n = data.len();

        // Compute lag-0 variance and lag-1 covariance
        let var: f64 = data.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / n as f64;
        let cov: f64 = data[..n - 1]
            .iter()
            .zip(data[1..].iter())
            .map(|(&a, &b)| (a - mu) * (b - mu))
            .sum::<f64>()
            / (n - 1) as f64;

        let phi = if var > 1e-15 { cov / var } else { 0.0 };
        let sigma = var.sqrt() * (1.0 - phi * phi).max(0.0).sqrt();

        Self { phi, sigma, mean: mu }
    }

    /// Predict next value: x_{t+1} = μ + φ(x_t - μ)
    pub fn predict_next(&self, current: f64) -> f64 {
        self.mean + self.phi * (current - self.mean)
    }

    /// Multi-step forecast (exponential decay toward mean).
    pub fn forecast(&self, current: f64, steps: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(steps);
        let mut x = current;
        for _ in 0..steps {
            x = self.predict_next(x);
            result.push(x);
        }
        result
    }

    /// True if the process is stationary (|φ| < 1).
    pub fn is_stationary(&self) -> bool {
        self.phi.abs() < 1.0
    }

    /// Autocorrelation at lag k: ρ(k) = φ^k
    pub fn acf(&self, lag: usize) -> f64 {
        self.phi.powi(lag as i32)
    }

    /// Simulate n observations from the AR(1) process.
    pub fn simulate(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Xorshift64::new(seed);
        let mut result = Vec::with_capacity(n);
        let mut x = self.mean;
        for _ in 0..n {
            let eps = rng.next_standard_normal();
            x = self.mean + self.phi * (x - self.mean) + self.sigma * eps;
            result.push(x);
        }
        result
    }
}

// ─── Distribution Fitting (MLE) ──────────────────────────────────────────────

/// MLE estimate for Normal distribution: (μ̂, σ̂)
pub fn fit_normal_mle(data: &[f64]) -> (f64, f64) {
    let mu = mean(data);
    let sigma = std_dev(data);
    (mu, sigma)
}

/// MLE estimate for Exponential distribution: λ̂ = 1/mean
pub fn fit_exponential_mle(data: &[f64]) -> f64 {
    let m = mean(data);
    if m > 1e-15 { 1.0 / m } else { 1.0 }
}

/// MLE estimate for Gamma via moment matching.
///
/// k̂ = mean² / var,  θ̂ = var / mean
pub fn fit_gamma_mle(data: &[f64]) -> (f64, f64) {
    let m = mean(data);
    let v = variance(data);
    if m < 1e-15 || v < 1e-15 {
        return (1.0, 1.0);
    }
    let k = m * m / v;
    let theta = v / m;
    (k, theta)
}

/// MLE estimate for Poisson distribution: λ̂ = mean
pub fn fit_poisson_mle(data: &[f64]) -> f64 {
    mean(data)
}

/// Akaike Information Criterion: AIC = -2 * log_likelihood + 2 * k
pub fn aic(log_likelihood: f64, n_params: usize) -> f64 {
    -2.0 * log_likelihood + 2.0 * n_params as f64
}

/// Bayesian Information Criterion: BIC = -2 * log_likelihood + k * ln(n)
pub fn bic(log_likelihood: f64, n_params: usize, n: usize) -> f64 {
    -2.0 * log_likelihood + n_params as f64 * (n as f64).ln()
}

// ─── Multivariate Statistics ────────────────────────────────────────────────

/// Compute the sample covariance matrix from column-oriented data.
///
/// `columns` is a slice of variables, each of which is a slice of observations.
/// All columns must have the same length (panics otherwise).
/// Returns an n×n covariance matrix (Vec<Vec<f64>>) with Bessel correction.
pub fn covariance_matrix(columns: &[&[f64]]) -> Vec<Vec<f64>> {
    let p = columns.len();
    if p == 0 {
        return vec![];
    }
    let n = columns[0].len();
    assert!(n >= 2, "Need at least 2 observations for covariance");
    for col in columns {
        assert_eq!(col.len(), n, "All columns must have the same length");
    }

    let means: Vec<f64> = columns.iter().map(|c| mean(c)).collect();
    let mut cov = vec![vec![0.0; p]; p];

    for i in 0..p {
        for j in i..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += (columns[i][k] - means[i]) * (columns[j][k] - means[j]);
            }
            let c = sum / (n - 1) as f64; // Bessel correction
            cov[i][j] = c;
            cov[j][i] = c; // Symmetric
        }
    }
    cov
}

/// Compute the Pearson correlation matrix from column-oriented data.
///
/// Returns an n×n matrix where entry (i,j) is the Pearson correlation
/// between columns i and j. Diagonal entries are 1.0.
pub fn correlation_matrix(columns: &[&[f64]]) -> Vec<Vec<f64>> {
    let cov = covariance_matrix(columns);
    let p = cov.len();
    let mut corr = vec![vec![0.0; p]; p];

    let stds: Vec<f64> = (0..p).map(|i| cov[i][i].sqrt()).collect();

    for i in 0..p {
        for j in 0..p {
            if stds[i] > 1e-15 && stds[j] > 1e-15 {
                corr[i][j] = cov[i][j] / (stds[i] * stds[j]);
            } else if i == j {
                corr[i][j] = 1.0;
            }
        }
    }
    corr
}

/// Result of Principal Component Analysis.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Eigenvalues (variance explained by each component), sorted descending.
    pub eigenvalues: Vec<f64>,
    /// Proportion of variance explained by each component.
    pub variance_explained: Vec<f64>,
    /// Cumulative proportion of variance explained.
    pub cumulative_variance: Vec<f64>,
    /// Number of components needed to explain ≥ threshold variance (default 0.95).
    pub components_for_95pct: usize,
    /// Principal component loadings (eigenvectors as rows), shape k × p.
    pub loadings: Vec<Vec<f64>>,
    /// Projected data (scores), shape n × k.
    pub scores: Vec<Vec<f64>>,
}

/// Perform Principal Component Analysis on column-oriented data.
///
/// `columns`: slice of p variables, each with n observations.
/// `k`: number of principal components to retain (clamped to min(n, p)).
///
/// Uses the covariance matrix + SVD approach via `HdcMatrix::svd()`.
pub fn pca(columns: &[&[f64]], k: usize) -> PcaResult {
    use crate::hdc::linear_algebra::HdcMatrix;

    let p = columns.len();
    assert!(p > 0, "PCA needs at least 1 variable");
    let n = columns[0].len();
    assert!(n >= 2, "PCA needs at least 2 observations");
    let k = k.min(p).min(n);

    // Center the data
    let means: Vec<f64> = columns.iter().map(|c| mean(c)).collect();
    let centered: Vec<Vec<f64>> = columns
        .iter()
        .zip(means.iter())
        .map(|(col, &m)| col.iter().map(|&x| x - m).collect())
        .collect();

    // Build covariance matrix and compute its eigendecomposition via SVD
    let cov = covariance_matrix(columns);
    let cov_flat: Vec<f64> = cov.iter().flat_map(|row| row.iter().copied()).collect();
    let cov_matrix = HdcMatrix::new(cov_flat, p, p);

    let (singular_values, _u, v, _result) = cov_matrix.svd();

    // Eigenvalues of cov = singular values (cov is symmetric PSD)
    let total_var: f64 = singular_values.iter().sum();
    let variance_explained: Vec<f64> = singular_values
        .iter()
        .map(|&s| if total_var > 1e-15 { s / total_var } else { 0.0 })
        .collect();

    let mut cumulative = Vec::with_capacity(singular_values.len());
    let mut cum = 0.0;
    for &ve in &variance_explained {
        cum += ve;
        cumulative.push(cum);
    }

    let components_for_95pct = cumulative
        .iter()
        .position(|&c| c >= 0.95)
        .map(|i| i + 1)
        .unwrap_or(p);

    // Extract top-k loadings from V (eigenvectors as columns)
    let mut loadings = Vec::with_capacity(k);
    for j in 0..k {
        let mut loading = Vec::with_capacity(p);
        for i in 0..p {
            loading.push(v.get(i, j));
        }
        loadings.push(loading);
    }

    // Project data onto principal components: scores = centered_data × loadings^T
    let mut scores = Vec::with_capacity(n);
    for obs in 0..n {
        let mut score_row = Vec::with_capacity(k);
        for pc in 0..k {
            let mut s = 0.0;
            for var in 0..p {
                s += centered[var][obs] * loadings[pc][var];
            }
            score_row.push(s);
        }
        scores.push(score_row);
    }

    PcaResult {
        eigenvalues: singular_values[..k].to_vec(),
        variance_explained: variance_explained[..k].to_vec(),
        cumulative_variance: cumulative[..k].to_vec(),
        components_for_95pct,
        loadings,
        scores,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    // ── Descriptive statistics ───────────────────────────────────────────

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < TOL);
        assert!((mean(&[10.0]) - 10.0).abs() < TOL);
    }

    #[test]
    fn test_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((variance(&data) - 4.0).abs() < TOL);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev(&data) - 2.0).abs() < TOL);
    }

    #[test]
    fn test_median() {
        assert!((median(&[1.0, 3.0, 5.0]) - 3.0).abs() < TOL);
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < TOL);
    }

    #[test]
    fn test_quartiles() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (q1, q2, q3) = quartiles(&data);
        assert!((q2 - 4.5).abs() < TOL);
        assert!(q1 < q2);
        assert!(q2 < q3);
    }

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric distribution should have ~0 skewness
        let data: Vec<f64> = (0..100).map(|i| i as f64 - 50.0).collect();
        assert!(skewness(&data).abs() < 0.1);
    }

    // ── Distributions ────────────────────────────────────────────────────

    #[test]
    fn test_normal_pdf() {
        let n = Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        // PDF at 0 should be 1/√(2π)
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((n.pdf(0.0) - expected).abs() < TOL);
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        let n = Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        assert!((n.cdf(0.0) - 0.5).abs() < TOL);
        // CDF(-x) + CDF(x) ≈ 1
        assert!((n.cdf(-1.0) + n.cdf(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normal_mean_var() {
        let n = Distribution::Normal {
            mu: 5.0,
            sigma: 3.0,
        };
        assert!((n.mean_val() - 5.0).abs() < TOL);
        assert!((n.variance_val() - 9.0).abs() < TOL);
    }

    #[test]
    fn test_uniform_pdf() {
        let u = Distribution::Uniform { a: 0.0, b: 1.0 };
        assert!((u.pdf(0.5) - 1.0).abs() < TOL);
        assert!((u.pdf(-0.1) - 0.0).abs() < TOL);
        assert!((u.pdf(1.1) - 0.0).abs() < TOL);
    }

    #[test]
    fn test_exponential_properties() {
        let e = Distribution::Exponential { lambda: 2.0 };
        assert!((e.mean_val() - 0.5).abs() < TOL);
        assert!((e.variance_val() - 0.25).abs() < TOL);
        assert!((e.cdf(0.0) - 0.0).abs() < TOL);
        assert!((e.cdf(f64::INFINITY) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_bernoulli() {
        let b = Distribution::Bernoulli { p: 0.7 };
        assert!((b.pdf(1.0) - 0.7).abs() < TOL);
        assert!((b.pdf(0.0) - 0.3).abs() < TOL);
        assert!((b.mean_val() - 0.7).abs() < TOL);
    }

    #[test]
    fn test_binomial_properties() {
        let b = Distribution::Binomial { n: 10, p: 0.5 };
        assert!((b.mean_val() - 5.0).abs() < TOL);
        assert!((b.variance_val() - 2.5).abs() < TOL);
    }

    #[test]
    fn test_poisson() {
        let p = Distribution::Poisson { lambda: 4.0 };
        assert!((p.mean_val() - 4.0).abs() < TOL);
        assert!((p.variance_val() - 4.0).abs() < TOL);
    }

    #[test]
    fn test_beta_mean() {
        let b = Distribution::Beta {
            alpha: 2.0,
            beta: 5.0,
        };
        assert!((b.mean_val() - 2.0 / 7.0).abs() < TOL);
    }

    #[test]
    fn test_distribution_encoding() {
        let n1 = Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        let n2 = Distribution::Normal {
            mu: 10.0,
            sigma: 2.0,
        };
        let sim = n1.encode().similarity(&n2.encode());
        assert!(
            sim < 0.6,
            "Different distributions should have low similarity"
        );
    }

    // ── Hypothesis testing ───────────────────────────────────────────────

    #[test]
    fn test_one_sample_t_test() {
        // Data clearly centered at 5, test against μ=5
        let data: Vec<f64> = vec![4.8, 5.1, 5.0, 4.9, 5.2, 5.0, 4.9, 5.1];
        let result = one_sample_t_test(&data, 5.0, 0.05);
        assert!(!result.reject, "Should not reject H0: μ=5");
    }

    #[test]
    fn test_one_sample_t_test_reject() {
        // Data centered at 10, test against μ=5
        let data: Vec<f64> = vec![9.5, 10.2, 10.1, 9.8, 10.3, 10.0, 9.9, 10.1];
        let result = one_sample_t_test(&data, 5.0, 0.05);
        assert!(result.reject, "Should reject H0: μ=5");
    }

    #[test]
    fn test_two_sample_t_test() {
        let data1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = two_sample_t_test(&data1, &data2, 0.05);
        assert!(result.reject, "Clearly different means");
    }

    #[test]
    fn test_chi_squared() {
        // Fair die: observed ≈ expected
        let observed = vec![16.0, 18.0, 17.0, 15.0, 17.0, 17.0];
        let expected = vec![16.67, 16.67, 16.67, 16.67, 16.67, 16.67];
        let result = chi_squared_test(&observed, &expected, 0.05);
        assert!(!result.reject, "Fair die should not be rejected");
    }

    // ── Bayesian inference ───────────────────────────────────────────────

    #[test]
    fn test_normal_normal_update() {
        let result = normal_normal_update(0.0, 1.0, 5.0, 1.0, 10);
        // Posterior should be pulled toward data mean (5.0)
        if let Distribution::Normal { mu, sigma } = &result.posterior {
            assert!(
                *mu > 0.0 && *mu < 5.0,
                "Posterior mean should be between prior and data"
            );
            assert!(*sigma < 1.0, "Posterior should be more precise than prior");
        } else {
            panic!("Expected Normal posterior");
        }
    }

    #[test]
    fn test_beta_binomial_update() {
        // Uniform prior Beta(1,1), observe 7/10 successes
        let result = beta_binomial_update(1.0, 1.0, 7, 10);
        if let Distribution::Beta { alpha, beta } = &result.posterior {
            assert!((alpha - 8.0).abs() < TOL);
            assert!((beta - 4.0).abs() < TOL);
            // Posterior mean = 8/12 ≈ 0.667
            assert!((result.posterior.mean_val() - 8.0 / 12.0).abs() < TOL);
        } else {
            panic!("Expected Beta posterior");
        }
    }

    // ── Correlation ──────────────────────────────────────────────────────

    #[test]
    fn test_pearson_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson_correlation(&x, &y) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_pearson_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson_correlation(&x, &y) - (-1.0)).abs() < TOL);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((spearman_correlation(&x, &y) - 1.0).abs() < TOL);
    }

    // ── Linear regression ────────────────────────────────────────────────

    #[test]
    fn test_linear_regression_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0]; // y = 1 + 2x
        let result = linear_regression(&x, &y);
        assert!((result.slope - 2.0).abs() < TOL);
        assert!((result.intercept - 1.0).abs() < TOL);
        assert!((result.r_squared - 1.0).abs() < TOL);
    }

    #[test]
    fn test_linear_regression_noisy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1]; // ~2x with noise
        let result = linear_regression(&x, &y);
        assert!((result.slope - 2.0).abs() < 0.2);
        assert!(result.r_squared > 0.95);
    }

    // ── Sampling ──────────────────────────────────────────────────────────

    #[test]
    fn test_normal_sampling_mean() {
        // Draw 10,000 samples from N(5, 2) and check empirical mean
        let dist = Distribution::Normal {
            mu: 5.0,
            sigma: 2.0,
        };
        let mut rng = Xorshift64::new(42);
        let samples: Vec<f64> = (0..10_000).map(|_| dist.sample(&mut rng)).collect();
        let m = mean(&samples);
        assert!(
            (m - 5.0).abs() < 0.1,
            "Empirical mean {m} should be near 5.0"
        );
    }

    #[test]
    fn test_exponential_sampling_mean() {
        let dist = Distribution::Exponential { lambda: 3.0 };
        let mut rng = Xorshift64::new(99);
        let samples: Vec<f64> = (0..10_000).map(|_| dist.sample(&mut rng)).collect();
        let m = mean(&samples);
        // E[X] = 1/λ = 1/3 ≈ 0.333
        assert!(
            (m - 1.0 / 3.0).abs() < 0.02,
            "Empirical mean {m} should be near 0.333"
        );
    }

    #[test]
    fn test_beta_sampling_mean() {
        let dist = Distribution::Beta {
            alpha: 2.0,
            beta: 5.0,
        };
        let mut rng = Xorshift64::new(7);
        let samples: Vec<f64> = (0..10_000).map(|_| dist.sample(&mut rng)).collect();
        let m = mean(&samples);
        // E[X] = α/(α+β) = 2/7 ≈ 0.286
        assert!(
            (m - 2.0 / 7.0).abs() < 0.02,
            "Empirical mean {m} should be near 0.286"
        );
    }

    #[test]
    fn test_poisson_sampling_mean() {
        let dist = Distribution::Poisson { lambda: 4.0 };
        let mut rng = Xorshift64::new(123);
        let samples: Vec<f64> = (0..10_000).map(|_| dist.sample(&mut rng)).collect();
        let m = mean(&samples);
        assert!(
            (m - 4.0).abs() < 0.15,
            "Empirical mean {m} should be near 4.0"
        );
    }

    // ── Confidence intervals ────────────────────────────────────────────

    #[test]
    fn test_confidence_interval_contains_true_mean() {
        // Data drawn from N(10, 1) — CI at 95% should contain 10
        let data: Vec<f64> = vec![9.8, 10.1, 10.0, 9.9, 10.2, 10.0, 9.9, 10.1, 10.0, 9.8];
        let result = one_sample_t_test(&data, 10.0, 0.05);
        let (lo, hi) = result.confidence_interval;
        assert!(
            lo < 10.0 && hi > 10.0,
            "95% CI ({lo:.3}, {hi:.3}) should contain true mean 10.0"
        );
    }

    #[test]
    fn test_two_sample_ci_excludes_zero_when_different() {
        let data1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let result = two_sample_t_test(&data1, &data2, 0.05);
        let (lo, hi) = result.confidence_interval;
        // Both bounds should be negative (data1 < data2, so diff is negative)
        assert!(
            hi < 0.0,
            "CI ({lo:.3}, {hi:.3}) should be entirely below 0 for clearly different groups"
        );
    }

    // ── Trait usage ─────────────────────────────────────────────────────

    #[test]
    fn test_probability_distribution_trait() {
        // Use via trait object to verify the trait works
        let dist = Distribution::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        let trait_obj: &dyn ProbabilityDistribution = &dist;
        assert!((trait_obj.dist_mean() - 0.0).abs() < TOL);
        assert!((trait_obj.dist_variance() - 1.0).abs() < TOL);
        assert!((trait_obj.dist_std() - 1.0).abs() < TOL);
        let expected_pdf = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((trait_obj.pdf(0.0) - expected_pdf).abs() < TOL);
    }

    // ── Kurtosis ────────────────────────────────────────────────────────

    #[test]
    fn test_kurtosis_normal_like() {
        // Excess kurtosis of a uniform distribution = -1.2
        let data: Vec<f64> = (0..10_000).map(|i| i as f64 / 10_000.0).collect();
        let k = kurtosis(&data);
        assert!(
            (k - (-1.2)).abs() < 0.05,
            "Uniform excess kurtosis {k} should be near -1.2"
        );
    }

    // ── Helper functions ─────────────────────────────────────────────────

    #[test]
    fn test_erf() {
        assert!(erf(0.0).abs() < TOL);
        assert!((erf(f64::INFINITY) - 1.0).abs() < TOL);
        // erf(1) ≈ 0.8427
        assert!((erf(1.0) - 0.8427).abs() < 0.001);
    }

    #[test]
    fn test_gamma_function() {
        // Γ(1) = 1
        assert!((gamma_function(1.0) - 1.0).abs() < TOL);
        // Γ(2) = 1
        assert!((gamma_function(2.0) - 1.0).abs() < TOL);
        // Γ(5) = 4! = 24
        assert!((gamma_function(5.0) - 24.0).abs() < 0.001);
        // Γ(0.5) = √π
        assert!((gamma_function(0.5) - std::f64::consts::PI.sqrt()).abs() < TOL);
    }

    #[test]
    fn test_inverse_normal_cdf() {
        // Φ⁻¹(0.5) = 0.0
        assert!(inverse_normal_cdf(0.5).abs() < 0.001);
        // Φ⁻¹(0.975) ≈ 1.96
        assert!((inverse_normal_cdf(0.975) - 1.96).abs() < 0.01);
        // Φ⁻¹(0.025) ≈ -1.96
        assert!((inverse_normal_cdf(0.025) + 1.96).abs() < 0.01);
    }

    #[test]
    fn test_xorshift_determinism() {
        let mut rng1 = Xorshift64::new(42);
        let mut rng2 = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    // ── Bootstrap CI ─────────────────────────────────────────────────────

    #[test]
    fn test_bootstrap_mean_ci_contains_true_mean() {
        // Data from N(5, 1) — CI should contain 5.0
        let data: Vec<f64> = vec![4.8, 5.1, 5.0, 4.9, 5.2, 5.0, 4.9, 5.1, 5.3, 4.7,
                                   5.1, 4.8, 5.0, 5.2, 4.9, 5.0, 5.1, 4.8, 5.3, 5.0];
        let (lo, hi) = bootstrap_mean_ci(&data, 1000, 0.95, 42);
        assert!(
            lo < 5.0 && hi > 5.0,
            "Bootstrap 95% CI ({:.3}, {:.3}) should contain 5.0",
            lo, hi
        );
    }

    #[test]
    fn test_bootstrap_median_ci_width() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let (lo, hi) = bootstrap_median_ci(&data, 500, 0.95, 99);
        assert!(hi > lo, "CI should have positive width");
        assert!(lo > 0.0 && hi < 22.0, "CI should be in reasonable range");
    }

    // ── AR(1) Model ──────────────────────────────────────────────────────

    #[test]
    fn test_ar1_stationarity() {
        // phi = 0.5 → stationary
        let m = AR1Model { phi: 0.5, sigma: 1.0, mean: 0.0 };
        assert!(m.is_stationary());
        // phi = 1.1 → non-stationary
        let m2 = AR1Model { phi: 1.1, sigma: 1.0, mean: 0.0 };
        assert!(!m2.is_stationary());
    }

    #[test]
    fn test_ar1_acf_matches_phi_power() {
        let m = AR1Model { phi: 0.7, sigma: 1.0, mean: 0.0 };
        for lag in 0..5 {
            let expected = 0.7f64.powi(lag as i32);
            assert!(
                (m.acf(lag) - expected).abs() < 1e-10,
                "ACF at lag {} = {} != {}",
                lag, m.acf(lag), expected
            );
        }
    }

    #[test]
    fn test_ar1_fit_recovers_phi() {
        // Simulate AR(1) with phi=0.6 and fit
        let m = AR1Model { phi: 0.6, sigma: 0.5, mean: 2.0 };
        let data = m.simulate(2000, 1234);
        let fitted = AR1Model::fit(&data);
        assert!(
            (fitted.phi - 0.6).abs() < 0.05,
            "Fitted phi {} should be near 0.6",
            fitted.phi
        );
    }

    #[test]
    fn test_ar1_forecast_decays_to_mean() {
        let m = AR1Model { phi: 0.5, sigma: 1.0, mean: 10.0 };
        let forecast = m.forecast(20.0, 20);
        // After many steps, should converge to mean
        let last = *forecast.last().unwrap();
        assert!(
            (last - 10.0).abs() < 1.0,
            "Long-run forecast {} should be near mean 10.0",
            last
        );
    }

    // ── MLE Fitting ──────────────────────────────────────────────────────

    #[test]
    fn test_fit_normal_mle_recovers_params() {
        let data: Vec<f64> = vec![2.8, 3.1, 3.0, 2.9, 3.2, 3.0, 2.9, 3.1, 3.0, 2.95];
        let (mu, sigma) = fit_normal_mle(&data);
        assert!((mu - 3.0).abs() < 0.1, "MLE mu {} should be near 3.0", mu);
        assert!(sigma > 0.0, "MLE sigma should be positive");
    }

    #[test]
    fn test_fit_exponential_mle() {
        // lambda = 2.0, mean = 0.5
        let data: Vec<f64> = vec![0.3, 0.6, 0.4, 0.5, 0.7, 0.45, 0.55, 0.5, 0.35, 0.6];
        let lambda = fit_exponential_mle(&data);
        assert!(lambda > 0.0, "lambda must be positive");
        // Should be near 2.0
        assert!((lambda - 2.0).abs() < 0.5, "lambda {} should be near 2.0", lambda);
    }

    #[test]
    fn test_fit_gamma_moment_matching() {
        // k=2, theta=3 → mean=6, var=18
        // Moment matching: k = mean²/var = 36/18 = 2
        let data: Vec<f64> = vec![4.0, 6.0, 7.0, 5.0, 8.0, 6.0, 5.5, 7.0, 6.5, 5.0];
        let (k, theta) = fit_gamma_mle(&data);
        assert!(k > 0.0 && theta > 0.0, "Gamma params must be positive");
    }

    #[test]
    fn test_aic_bic_ordering() {
        // Higher log-likelihood → lower AIC/BIC
        let ll_good = -50.0;
        let ll_bad = -100.0;
        let n = 100;
        assert!(aic(ll_good, 2) < aic(ll_bad, 2));
        assert!(bic(ll_good, 2, n) < bic(ll_bad, 2, n));
        // More params → higher BIC (penalty)
        assert!(bic(ll_good, 5, n) > bic(ll_good, 2, n));
    }

    // ── Multivariate Statistics ─────────────────────────────────────────

    #[test]
    fn test_covariance_matrix_identity() {
        // Independent variables with unit variance
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let cov = covariance_matrix(&[&x, &y]);

        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);
        // Diagonal = sample variance
        assert!((cov[0][0] - sample_variance(&x)).abs() < TOL);
        assert!((cov[1][1] - sample_variance(&y)).abs() < TOL);
        // Symmetric
        assert!((cov[0][1] - cov[1][0]).abs() < TOL);
        // x and y are perfectly anti-correlated
        assert!(cov[0][1] < 0.0);
    }

    #[test]
    fn test_covariance_matrix_identical_columns() {
        let x = [1.0, 3.0, 5.0, 7.0];
        let cov = covariance_matrix(&[&x, &x]);
        // Cov(X, X) = Var(X)
        assert!((cov[0][0] - cov[0][1]).abs() < TOL);
        assert!((cov[0][0] - cov[1][1]).abs() < TOL);
    }

    #[test]
    fn test_correlation_matrix_properties() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0]; // Perfectly correlated (y = 2x)
        let z = [5.0, 4.0, 3.0, 2.0, 1.0]; // Perfectly anti-correlated

        let corr = correlation_matrix(&[&x, &y, &z]);
        // Diagonal = 1.0
        assert!((corr[0][0] - 1.0).abs() < TOL);
        assert!((corr[1][1] - 1.0).abs() < TOL);
        // x-y perfect positive correlation
        assert!((corr[0][1] - 1.0).abs() < TOL);
        // x-z perfect negative correlation
        assert!((corr[0][2] - (-1.0)).abs() < TOL);
        // Symmetric
        assert!((corr[0][1] - corr[1][0]).abs() < TOL);
    }

    #[test]
    fn test_pca_basic() {
        // 2D data with most variance along x-axis
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [1.1, 2.0, 3.1, 4.0, 4.9, 6.1, 7.0, 7.9, 9.1, 10.0];

        let result = pca(&[&x, &y], 2);
        // First PC should explain most variance
        assert!(
            result.variance_explained[0] > 0.95,
            "First PC should explain >95% variance, got {:.4}",
            result.variance_explained[0]
        );
        // Eigenvalues should be non-negative and sorted descending
        assert!(result.eigenvalues[0] >= result.eigenvalues[1]);
        assert!(result.eigenvalues[1] >= 0.0);
        // Scores should have correct shape: 10 observations × 2 components
        assert_eq!(result.scores.len(), 10);
        assert_eq!(result.scores[0].len(), 2);
        // Cumulative variance of all components should be ~1.0
        assert!((result.cumulative_variance[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pca_loadings_orthogonal() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 3.0, 1.0, 2.0, 4.0];
        let z = [2.0, 4.0, 1.0, 3.0, 5.0];

        let result = pca(&[&x, &y, &z], 3);
        // Loadings should be approximately orthogonal
        for i in 0..result.loadings.len() {
            for j in (i + 1)..result.loadings.len() {
                let dot: f64 = result.loadings[i]
                    .iter()
                    .zip(result.loadings[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(
                    dot.abs() < 0.1,
                    "Loadings {i} and {j} should be orthogonal, dot={dot:.4}"
                );
            }
        }
    }
}
