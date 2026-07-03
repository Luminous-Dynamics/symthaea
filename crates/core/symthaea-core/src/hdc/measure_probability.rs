// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Measure theory and probability: σ-algebras, Lebesgue integration, martingales,
//! Brownian motion, Itô's lemma, conditional expectation, LLN/CLT, and KS statistic.

// ─── σ-Algebras and Measures ─────────────────────────────────────────────────

/// A finite σ-algebra over the set {0, …, n-1}.
/// Each set is represented as an indicator vector of length n.
#[derive(Clone, Debug)]
pub struct SigmaAlgebra {
    pub n: usize,
    /// Collection of sets, each as indicator vector (true = element included).
    pub sets: Vec<Vec<bool>>,
}

impl SigmaAlgebra {
    /// Construct a σ-algebra.
    pub fn new(n: usize, sets: Vec<Vec<bool>>) -> Self {
        SigmaAlgebra { n, sets }
    }

    /// Complement of a set.
    fn complement(&self, set: &[bool]) -> Vec<bool> {
        set.iter().map(|&b| !b).collect()
    }

    /// Union of two sets.
    fn union(a: &[bool], b: &[bool]) -> Vec<bool> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x || y).collect()
    }

    /// Check if a given set is already in the collection.
    fn contains_set(&self, set: &[bool]) -> bool {
        self.sets.iter().any(|s| s == set)
    }

    /// Validate that this is a proper σ-algebra:
    /// 1. Contains Ω (all-true) and ∅ (all-false)
    /// 2. Closed under complement
    /// 3. Closed under countable (finite here) union
    pub fn is_valid_sigma_algebra(&self) -> bool {
        let omega: Vec<bool> = vec![true; self.n];
        let empty: Vec<bool> = vec![false; self.n];

        // Must contain Ω and ∅
        if !self.contains_set(&omega) || !self.contains_set(&empty) {
            return false;
        }

        // Closed under complement
        for set in &self.sets {
            let comp = self.complement(set);
            if !self.contains_set(&comp) {
                return false;
            }
        }

        // Closed under pairwise union (sufficient for finite collections)
        for a in &self.sets {
            for b in &self.sets {
                let u = Self::union(a, b);
                if !self.contains_set(&u) {
                    return false;
                }
            }
        }

        true
    }
}

/// A measure space: a σ-algebra equipped with a measure for each set.
#[derive(Clone, Debug)]
pub struct MeasureSpace {
    pub algebra: SigmaAlgebra,
    /// Measure values aligned with `algebra.sets`.
    pub measure: Vec<f64>,
}

impl MeasureSpace {
    pub fn new(algebra: SigmaAlgebra, measure: Vec<f64>) -> Self {
        MeasureSpace { algebra, measure }
    }

    /// Check that this is a valid probability measure:
    /// - Non-negativity
    /// - P(Ω) = 1
    /// - σ-additivity (for disjoint sets A, B: P(A∪B) = P(A) + P(B))
    pub fn is_probability_measure(&self) -> bool {
        let n = self.algebra.n;
        let omega: Vec<bool> = vec![true; n];

        // Find Ω index and check P(Ω) = 1
        let omega_idx = self.algebra.sets.iter().position(|s| s == &omega);
        if let Some(idx) = omega_idx {
            if (self.measure[idx] - 1.0).abs() > 1e-10 {
                return false;
            }
        } else {
            return false;
        }

        // Non-negativity
        if self.measure.iter().any(|&m| m < -1e-10) {
            return false;
        }

        // σ-additivity: for disjoint sets A, B in algebra, P(A∪B) = P(A)+P(B)
        let sets = &self.algebra.sets;
        for (i, a) in sets.iter().enumerate() {
            for (j, b) in sets.iter().enumerate() {
                // Check disjointness
                let disjoint = a.iter().zip(b.iter()).all(|(&x, &y)| !(x && y));
                if !disjoint {
                    continue;
                }
                let u = SigmaAlgebra::union(a, b);
                if let Some(k) = sets.iter().position(|s| s == &u)
                    && (self.measure[k] - self.measure[i] - self.measure[j]).abs() > 1e-10
                {
                    return false;
                }
            }
        }

        true
    }
}

// ─── Lebesgue Integration (numerical) ────────────────────────────────────────

/// Approximate Lebesgue integral of f on [a, b] using n-point Riemann sum.
pub fn lebesgue_integral<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if n == 0 || a >= b {
        return 0.0;
    }
    let dx = (b - a) / n as f64;
    (0..n).map(|i| f(a + (i as f64 + 0.5) * dx) * dx).sum()
}

/// L^p norm: (∫|f|^p)^(1/p) on [a, b] with n subintervals.
pub fn lp_norm<F>(f: F, p: f64, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let integral = lebesgue_integral(|x| f(x).abs().powf(p), a, b, n);
    integral.powf(1.0 / p)
}

/// L² inner product ⟨f, g⟩ = ∫f(x)g(x)dx on [a, b].
pub fn l2_inner_product<F, G>(f: F, g: G, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    lebesgue_integral(|x| f(x) * g(x), a, b, n)
}

// ─── Martingales ──────────────────────────────────────────────────────────────

/// A discrete-time martingale.
/// `values[t]` is the value at time t.
/// `filtration[t]` is the index set of "known" time steps at time t (σ-field approximation).
#[derive(Clone, Debug)]
pub struct Martingale {
    pub values: Vec<f64>,
    pub filtration: Vec<Vec<usize>>,
}

impl Martingale {
    pub fn new(values: Vec<f64>, filtration: Vec<Vec<usize>>) -> Self {
        Martingale { values, filtration }
    }

    /// Check the martingale property: E[X_{n+1} | F_n] = X_n for each n.
    /// Uses the provided `expectation_fn` to compute conditional expectations.
    pub fn is_martingale(&self, expectation_fn: &dyn Fn(&[f64]) -> f64) -> bool {
        let n = self.values.len();
        if n <= 1 {
            return true;
        }
        for t in 0..n - 1 {
            let known: Vec<f64> = self.filtration[t].iter().map(|&i| self.values[i]).collect();
            let cond_exp = expectation_fn(&known);
            if (cond_exp - self.values[t]).abs() > 1e-8 {
                return false;
            }
        }
        true
    }
}

/// Doob's optional stopping bound: E[X_τ] ≤ max|X_n| (crude bound).
pub fn optional_stopping_bound(martingale_values: &[f64], _stopping_time: usize) -> f64 {
    martingale_values
        .iter()
        .cloned()
        .map(f64::abs)
        .fold(0.0f64, f64::max)
}

/// Doob's maximal inequality: P(max X_n ≥ λ) ≤ E[|X_N|] / λ.
pub fn doobs_maximal_inequality(values: &[f64], lambda: f64) -> f64 {
    if lambda <= 0.0 || values.is_empty() {
        return 1.0;
    }
    let e_xn = values.iter().cloned().map(f64::abs).sum::<f64>() / values.len() as f64;
    (e_xn / lambda).min(1.0)
}

// ─── Brownian Motion (simulation) ────────────────────────────────────────────

/// Simple LCG random number generator producing values in [0, 1).
fn lcg_next(state: &mut u64) -> f64 {
    // Parameters from Numerical Recipes
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// Box-Muller transform to generate N(0,1) samples.
fn normal_sample(state: &mut u64) -> f64 {
    loop {
        let u1 = lcg_next(state);
        let u2 = lcg_next(state);
        if u1 > 0.0 {
            return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
}

/// Simulate a standard Brownian motion path W(t) for t in [0, n_steps * dt].
/// Uses W(t + dt) = W(t) + sqrt(dt) * Z where Z ~ N(0,1).
pub fn brownian_motion(n_steps: usize, dt: f64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut path = vec![0.0f64; n_steps + 1];
    let sqrt_dt = dt.sqrt();
    for i in 1..=n_steps {
        path[i] = path[i - 1] + sqrt_dt * normal_sample(&mut state);
    }
    path
}

/// Geometric Brownian Motion: S(t+dt) = S(t) * exp((μ - σ²/2)*dt + σ*sqrt(dt)*Z).
pub fn geometric_brownian_motion(
    s0: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    n: usize,
    seed: u64,
) -> Vec<f64> {
    let mut state = seed;
    let mut path = vec![s0; n + 1];
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let diffusion_scale = sigma * dt.sqrt();
    for i in 1..=n {
        let z = normal_sample(&mut state);
        path[i] = path[i - 1] * (drift + diffusion_scale * z).exp();
    }
    path
}

/// Quadratic variation: sum of squared increments.
pub fn quadratic_variation(path: &[f64]) -> f64 {
    path.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum()
}

// ─── Itô's Lemma (symbolic application) ──────────────────────────────────────

/// Drift coefficient of f(X_t, t) by Itô's lemma:
/// df = (∂f/∂t + μ·∂f/∂x + ½σ²·∂²f/∂x²) dt + σ·∂f/∂x dW
pub fn ito_drift(f_x: f64, f_t: f64, f_xx: f64, mu: f64, sigma: f64) -> f64 {
    f_t + mu * f_x + 0.5 * sigma * sigma * f_xx
}

/// Diffusion coefficient of f(X_t, t) by Itô's lemma: σ · ∂f/∂x.
pub fn ito_diffusion(f_x: f64, sigma: f64) -> f64 {
    sigma * f_x
}

// ─── Conditional Expectation (discrete) ──────────────────────────────────────

/// Compute the conditional expectation E[X | σ(partition)].
/// For each partition cell, replace values with the average over that cell.
/// Returns a vector aligned with x: result[i] = average of x over the cell containing i.
pub fn conditional_expectation(x: &[f64], partition: &[Vec<usize>]) -> Vec<f64> {
    let n = x.len();
    let mut result = vec![0.0f64; n];
    let mut assigned = vec![false; n];

    for cell in partition {
        if cell.is_empty() {
            continue;
        }
        let avg = cell.iter().map(|&i| x[i]).sum::<f64>() / cell.len() as f64;
        for &i in cell {
            if i < n {
                result[i] = avg;
                assigned[i] = true;
            }
        }
    }

    // Any unassigned elements get their own value
    for i in 0..n {
        if !assigned[i] {
            result[i] = x[i];
        }
    }

    result
}

// ─── Law of Large Numbers / CLT ───────────────────────────────────────────────

/// Running empirical means: result[i] = mean of samples[0..=i].
pub fn empirical_mean_convergence(samples: &[f64]) -> Vec<f64> {
    let mut running = Vec::with_capacity(samples.len());
    let mut sum = 0.0f64;
    for (i, &x) in samples.iter().enumerate() {
        sum += x;
        running.push(sum / (i + 1) as f64);
    }
    running
}

/// CLT standardization: (sample_mean - 0) / (sample_std / sqrt(n)).
pub fn clt_standardized(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();
    if std < 1e-15 {
        return 0.0;
    }
    mean / (std / n.sqrt())
}

/// Kolmogorov-Smirnov statistic: max |F_n(x) - Φ(x)| where Φ is the standard normal CDF.
pub fn kolmogorov_smirnov_statistic(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut max_d = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let empirical_cdf = (i + 1) as f64 / n as f64;
        let normal_cdf = standard_normal_cdf(x);
        let d = (empirical_cdf - normal_cdf).abs();
        if d > max_d {
            max_d = d;
        }
        // Also check left limit
        let empirical_left = i as f64 / n as f64;
        let d_left = (empirical_left - normal_cdf).abs();
        if d_left > max_d {
            max_d = d_left;
        }
    }
    max_d
}

/// Approximation to the standard normal CDF using the Abramowitz & Stegun method.
pub fn standard_normal_cdf(x: f64) -> f64 {
    // Rational approximation (max error ≈ 7.5e-8)
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf_positive = 1.0 - pdf * poly;
    if x >= 0.0 {
        cdf_positive
    } else {
        1.0 - cdf_positive
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trivial_sigma_algebra(n: usize) -> SigmaAlgebra {
        // Trivial σ-algebra: {∅, Ω}
        let empty = vec![false; n];
        let omega = vec![true; n];
        SigmaAlgebra::new(n, vec![empty, omega])
    }

    #[test]
    fn test_trivial_sigma_algebra_valid() {
        let sa = make_trivial_sigma_algebra(3);
        assert!(sa.is_valid_sigma_algebra());
    }

    #[test]
    fn test_invalid_sigma_algebra_missing_complement() {
        // {∅, Ω, {0}} — missing complement {1,2}
        let sets = vec![
            vec![false, false, false],
            vec![true, true, true],
            vec![true, false, false],
        ];
        let sa = SigmaAlgebra::new(3, sets);
        assert!(!sa.is_valid_sigma_algebra());
    }

    #[test]
    fn test_power_set_sigma_algebra() {
        // Power set of {0,1} is always a valid σ-algebra
        let sets = vec![
            vec![false, false],
            vec![true, false],
            vec![false, true],
            vec![true, true],
        ];
        let sa = SigmaAlgebra::new(2, sets);
        assert!(sa.is_valid_sigma_algebra());
    }

    #[test]
    fn test_brownian_motion_zero_mean() {
        // Average endpoint over many BM paths should be ≈ 0
        let n_paths = 500;
        let n_steps = 100;
        let dt = 0.01;
        let mut endpoint_sum = 0.0;
        for seed in 0..n_paths as u64 {
            let path = brownian_motion(n_steps, dt, seed * 7919 + 1);
            endpoint_sum += path[n_steps];
        }
        let mean = endpoint_sum / n_paths as f64;
        assert!(mean.abs() < 0.15, "BM mean endpoint ≈ 0, got {}", mean);
    }

    #[test]
    fn test_brownian_motion_quadratic_variation() {
        // E[QV] = T for standard BM (n*dt)
        let n_paths = 200;
        let n_steps = 200;
        let dt = 0.01;
        let t = n_steps as f64 * dt;
        let mut qv_sum = 0.0;
        for seed in 0..n_paths as u64 {
            let path = brownian_motion(n_steps, dt, seed * 6271 + 3);
            qv_sum += quadratic_variation(&path);
        }
        let mean_qv = qv_sum / n_paths as f64;
        assert!(
            (mean_qv - t).abs() < 0.15,
            "Mean quadratic variation ≈ T={}, got {}",
            t,
            mean_qv
        );
    }

    #[test]
    fn test_gbm_positive() {
        // GBM paths are always positive
        let path = geometric_brownian_motion(100.0, 0.05, 0.2, 0.01, 252, 42);
        assert!(
            path.iter().all(|&v| v > 0.0),
            "All GBM values must be positive"
        );
    }

    #[test]
    fn test_ito_drift_log_process() {
        // For f(S,t) = log(S), f_x=1/S, f_xx=-1/S², f_t=0
        // ito_drift = mu*(1/S) + 0.5*sigma²*(-1/S²)*S² = mu - sigma²/2
        let s = 100.0f64;
        let mu = 0.05f64;
        let sigma = 0.2f64;
        let f_x = 1.0 / s;
        let f_t = 0.0;
        let f_xx = -1.0 / (s * s);
        // Multiply back by S since dS = mu*S dt + sigma*S dW → dx = f_x * dX
        // The "mu" passed here is already the coefficient on dS/S, but ito_drift
        // takes the drift of the SDE directly: dS = mu dt + sigma dW (not GBM).
        // For GBM: drift of log = mu - sigma²/2 using f_x=1/S, f_xx=-1/S², scaled
        // coefficient mu_s = mu*S, sigma_s = sigma*S
        let mu_s = mu * s;
        let sigma_s = sigma * s;
        let drift = ito_drift(f_x, f_t, f_xx, mu_s, sigma_s);
        let expected = mu - 0.5 * sigma * sigma;
        assert!(
            (drift - expected).abs() < 1e-10,
            "Itô drift of log(S) = mu - σ²/2 = {}, got {}",
            expected,
            drift
        );
    }

    #[test]
    fn test_conditional_expectation_idempotent() {
        // E[E[X|G]|G] = E[X|G]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let partition = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let ce1 = conditional_expectation(&x, &partition);
        let ce2 = conditional_expectation(&ce1, &partition);
        for (a, b) in ce1.iter().zip(ce2.iter()) {
            assert!((a - b).abs() < 1e-10, "CE is idempotent");
        }
    }

    #[test]
    fn test_conditional_expectation_values() {
        let x = vec![2.0, 4.0, 6.0];
        let partition = vec![vec![0, 1, 2]];
        let ce = conditional_expectation(&x, &partition);
        // All should equal (2+4+6)/3 = 4
        for &v in &ce {
            assert!((v - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_empirical_mean_convergence() {
        // For constant sequence, running mean is constant
        let samples = vec![5.0; 100];
        let means = empirical_mean_convergence(&samples);
        for &m in &means {
            assert!((m - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lp_norm_l2_sin() {
        // ||sin||_2 on [0, π] = sqrt(π/2)
        let norm = lp_norm(|x: f64| x.sin(), 2.0, 0.0, std::f64::consts::PI, 10000);
        let expected = (std::f64::consts::PI / 2.0).sqrt();
        assert!(
            (norm - expected).abs() < 0.01,
            "L2 norm of sin on [0,π]: expected {}, got {}",
            expected,
            norm
        );
    }

    #[test]
    fn test_lebesgue_integral_constant() {
        // ∫ 3 dx on [0, 2] = 6
        let val = lebesgue_integral(|_| 3.0, 0.0, 2.0, 1000);
        assert!((val - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_standard_normal_cdf() {
        // Φ(0) = 0.5, Φ(∞) → 1, Φ(-∞) → 0
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-5);
        assert!(standard_normal_cdf(5.0) > 0.999);
        assert!(standard_normal_cdf(-5.0) < 0.001);
    }

    #[test]
    fn test_probability_measure_trivial() {
        let sa = make_trivial_sigma_algebra(3);
        // Ω has measure 1, ∅ has measure 0
        let measure = vec![0.0, 1.0]; // aligned with sets: [∅, Ω]
        let ms = MeasureSpace::new(sa, measure);
        assert!(ms.is_probability_measure());
    }

    #[test]
    fn test_doobs_inequality_bound() {
        let values = vec![0.0, 0.5, 1.0, 0.8, 0.3];
        let bound = doobs_maximal_inequality(&values, 0.5);
        // Should be in [0,1]
        assert!(bound >= 0.0 && bound <= 1.0);
    }
}
