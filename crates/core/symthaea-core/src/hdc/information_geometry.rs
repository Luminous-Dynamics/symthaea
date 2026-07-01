// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Information geometry: Fisher information matrices, natural gradients,
//! KL divergence geometry, statistical manifold geodesics, α-connections,
//! and exponential families.

// ─── Fisher Information Matrix ────────────────────────────────────────────────

/// Fisher Information Matrix for the normal distribution N(μ, σ²).
/// FIM = diag(1/σ², 2/σ²) in (μ, σ) parameterisation.
/// Layout: [[I_μμ, I_μσ], [I_σμ, I_σσ]]
pub fn fisher_information_normal(mu: f64, sigma: f64) -> [[f64; 2]; 2] {
    let _ = mu; // μ doesn't affect FIM for normal
    let s2 = sigma * sigma;
    [[1.0 / s2, 0.0], [0.0, 2.0 / s2]]
}

/// Fisher Information for Bernoulli(p): I(p) = 1 / (p(1-p)).
pub fn fisher_information_bernoulli(p: f64) -> f64 {
    1.0 / (p * (1.0 - p))
}

/// Fisher Information for Poisson(λ): I(λ) = 1/λ.
pub fn fisher_information_poisson(lambda: f64) -> f64 {
    1.0 / lambda
}

/// Fisher Information for Exponential(λ): I(λ) = 1/λ².
pub fn fisher_information_exponential(lambda: f64) -> f64 {
    1.0 / (lambda * lambda)
}

/// Cramér-Rao lower bound: variance ≥ 1/I(θ).
pub fn cramer_rao_bound(fisher: f64) -> f64 {
    1.0 / fisher
}

// ─── Natural Gradient ─────────────────────────────────────────────────────────

/// Solve a linear system A·x = b via Gaussian elimination with partial pivoting.
/// Returns x (the solution vector).
fn gaussian_eliminate(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    // Build augmented matrix [A | b]
    let mut mat: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    // Forward elimination
    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                mat[r1][col]
                    .abs()
                    .partial_cmp(&mat[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        mat.swap(col, pivot_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-14 {
            continue; // Singular or near-singular — leave zero
        }

        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for k in col..=n {
                let v = mat[col][k] * factor;
                mat[row][k] -= v;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = mat[i][n];
        for j in (i + 1)..n {
            sum -= mat[i][j] * x[j];
        }
        if mat[i][i].abs() > 1e-14 {
            x[i] = sum / mat[i][i];
        }
    }
    x
}

/// Compute the natural gradient: F⁻¹ · gradient.
/// `fim` is the Fisher Information Matrix (n×n as Vec<Vec<f64>>).
pub fn natural_gradient(gradient: &[f64], fim: &[Vec<f64>]) -> Vec<f64> {
    gaussian_eliminate(fim, gradient)
}

/// Compute one natural gradient descent step:
/// params_new = params - lr * F⁻¹ · gradient.
pub fn natural_gradient_step(
    params: &[f64],
    gradient: &[f64],
    fim: &[Vec<f64>],
    lr: f64,
) -> Vec<f64> {
    let ng = natural_gradient(gradient, fim);
    params
        .iter()
        .zip(ng.iter())
        .map(|(&p, &g)| p - lr * g)
        .collect()
}

// ─── KL Divergence Geometry ───────────────────────────────────────────────────

/// KL divergence KL(N(μ₁,σ₁²) ‖ N(μ₂,σ₂²)).
/// = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²) / (2σ₂²) - 1/2
pub fn kl_divergence_normal(mu1: f64, s1: f64, mu2: f64, s2: f64) -> f64 {
    let log_ratio = (s2 / s1).ln();
    let diff = mu1 - mu2;
    log_ratio + (s1 * s1 + diff * diff) / (2.0 * s2 * s2) - 0.5
}

/// KL divergence KL(p ‖ q) for categorical distributions.
/// Skips terms where p_i = 0.
pub fn kl_divergence_categorical(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter(|&(&pi, _)| pi > 1e-15)
        .map(|(&pi, &qi)| pi * (pi / qi.max(1e-300)).ln())
        .sum()
}

/// Symmetrized KL divergence: (KL(p‖q) + KL(q‖p)) / 2.
pub fn symmetrized_kl(p: &[f64], q: &[f64]) -> f64 {
    (kl_divergence_categorical(p, q) + kl_divergence_categorical(q, p)) / 2.0
}

/// Jeffrey's divergence = symmetrized KL.
pub fn jeffrey_divergence(p: &[f64], q: &[f64]) -> f64 {
    symmetrized_kl(p, q)
}

// ─── Statistical Manifold Geodesics ──────────────────────────────────────────

/// Exponential (e-) geodesic: straight line in natural parameter space θ.
/// θ(t) = (1-t)·θ₀ + t·θ₁
pub fn exponential_family_geodesic(theta0: &[f64], theta1: &[f64], t: f64) -> Vec<f64> {
    theta0
        .iter()
        .zip(theta1.iter())
        .map(|(&a, &b)| (1.0 - t) * a + t * b)
        .collect()
}

/// Mixture (m-) geodesic: straight line in expectation parameter space η.
/// η(t) = (1-t)·η₀ + t·η₁
pub fn mixture_geodesic(eta0: &[f64], eta1: &[f64], t: f64) -> Vec<f64> {
    eta0.iter()
        .zip(eta1.iter())
        .map(|(&a, &b)| (1.0 - t) * a + t * b)
        .collect()
}

/// Convert N(μ, σ²) from natural parameters to expectation parameters.
/// Natural: θ₁ = μ/σ², θ₂ = -1/(2σ²)
/// Expectation: η₁ = μ, η₂ = μ² + σ²
pub fn exponential_to_expectation_normal(mu: f64, sigma: f64) -> (f64, f64) {
    let eta1 = mu;
    let eta2 = mu * mu + sigma * sigma;
    (eta1, eta2)
}

/// Convert expectation parameters back to (μ, σ) for N(μ, σ²).
/// η₁ = μ, η₂ = μ² + σ² → μ = η₁, σ = sqrt(η₂ - η₁²)
pub fn expectation_to_natural_normal(eta1: f64, eta2: f64) -> (f64, f64) {
    let mu = eta1;
    let var = eta2 - eta1 * eta1;
    let sigma = var.max(0.0).sqrt();
    (mu, sigma)
}

// ─── α-Connections ────────────────────────────────────────────────────────────

/// α-connection Christoffel symbols Γ^(α)_{ijk}.
/// For a flat manifold, all Christoffel symbols vanish.
/// For a non-flat manifold with provided FIM and its derivative, compute:
///   Γ^(α)_{ijk} = (1-α)/2 · Γ^(0)_{ijk} + (1+α)/2 · Γ^(0)_{kij}
/// where Γ^(0) are Levi-Civita (Riemannian) symbols.
///
/// `dfim[k][i][j]` = ∂g_{ij}/∂θ^k (derivative of FIM entry)
pub fn alpha_connection_christoffel(
    alpha: f64,
    fim: &[Vec<f64>],
    dfim: &[Vec<Vec<f64>>],
) -> Vec<Vec<Vec<f64>>> {
    let n = fim.len();
    // Compute Levi-Civita Christoffel symbols of the first kind:
    // Γ^(0)_{ijk} = ½(∂_i g_{jk} + ∂_j g_{ik} - ∂_k g_{ij})
    // Then Γ^(α)_{ijk} = (1-α)/2 Γ^(0)_{ijk} + (1+α)/2 Γ^(0)_{kij}

    let lc = |i: usize, j: usize, k: usize| -> f64 {
        if i < dfim.len() && j < n && k < n && i < n && j < dfim[i].len() && k < dfim[i][j].len() {
            let d_i_jk = if i < dfim.len() && j < dfim[i].len() && k < dfim[i][j].len() {
                dfim[i][j][k]
            } else {
                0.0
            };
            let d_j_ik = if j < dfim.len() && i < dfim[j].len() && k < dfim[j][i].len() {
                dfim[j][i][k]
            } else {
                0.0
            };
            let d_k_ij = if k < dfim.len() && i < dfim[k].len() && j < dfim[k][i].len() {
                dfim[k][i][j]
            } else {
                0.0
            };
            0.5 * (d_i_jk + d_j_ik - d_k_ij)
        } else {
            0.0
        }
    };

    let mut result = vec![vec![vec![0.0f64; n]; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let gamma_ijk = lc(i, j, k);
                let gamma_kij = lc(k, i, j);
                result[i][j][k] = (1.0 - alpha) / 2.0 * gamma_ijk + (1.0 + alpha) / 2.0 * gamma_kij;
            }
        }
    }
    result
}

// ─── Exponential Family ───────────────────────────────────────────────────────

/// An exponential family distribution parameterised by sufficient statistics,
/// log-partition function, and base measure.
#[derive(Clone, Debug)]
pub struct ExponentialFamily {
    pub sufficient_stats: Vec<f64>,
    pub log_partition: f64,
    pub base_measure: f64,
}

/// Convert natural parameters θ to mean (expectation) parameters η.
/// Supported families: "normal", "bernoulli", "poisson".
pub fn natural_to_mean_params(theta: &[f64], family: &str) -> Vec<f64> {
    match family {
        "normal" => {
            // θ = [θ₁, θ₂] = [μ/σ², -1/(2σ²)]
            // σ² = -1/(2θ₂), μ = θ₁ * σ²
            let sigma2 = if theta.len() > 1 && theta[1].abs() > 1e-15 {
                -1.0 / (2.0 * theta[1])
            } else {
                1.0
            };
            let mu = if !theta.is_empty() {
                theta[0] * sigma2
            } else {
                0.0
            };
            vec![mu, mu * mu + sigma2] // η₁ = μ, η₂ = μ² + σ²
        }
        "bernoulli" => {
            // θ = log(p/(1-p)) → p = sigmoid(θ)
            let p = if !theta.is_empty() {
                1.0 / (1.0 + (-theta[0]).exp())
            } else {
                0.5
            };
            vec![p]
        }
        "poisson" => {
            // θ = log(λ) → λ = exp(θ)
            let lambda = if !theta.is_empty() {
                theta[0].exp()
            } else {
                1.0
            };
            vec![lambda]
        }
        _ => theta.to_vec(),
    }
}

/// Log-likelihood of observations x under the exponential family.
/// Supported families: "normal", "bernoulli", "poisson".
pub fn log_likelihood(x: &[f64], theta: &[f64], family: &str) -> f64 {
    match family {
        "normal" => {
            // θ = [μ, σ] directly for simplicity (mean parameterisation)
            let mu = if !theta.is_empty() { theta[0] } else { 0.0 };
            let sigma = if theta.len() > 1 {
                theta[1].max(1e-10)
            } else {
                1.0
            };
            let n = x.len() as f64;
            let log_2pi = (2.0 * std::f64::consts::PI).ln();
            let sse: f64 = x.iter().map(|&xi| (xi - mu).powi(2)).sum();
            -0.5 * n * (log_2pi + 2.0 * sigma.ln()) - sse / (2.0 * sigma * sigma)
        }
        "bernoulli" => {
            let p = if !theta.is_empty() {
                theta[0].clamp(1e-10, 1.0 - 1e-10)
            } else {
                0.5
            };
            x.iter()
                .map(|&xi| xi * p.ln() + (1.0 - xi) * (1.0 - p).ln())
                .sum()
        }
        "poisson" => {
            let lambda = if !theta.is_empty() {
                theta[0].max(1e-10)
            } else {
                1.0
            };
            x.iter()
                .map(|&xi| xi * lambda.ln() - lambda - log_factorial(xi as usize))
                .sum()
        }
        _ => 0.0,
    }
}

fn log_factorial(n: usize) -> f64 {
    (1..=n).map(|k| (k as f64).ln()).sum()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_normal_identity() {
        // N(0, 1): FIM = [[1, 0], [0, 2]]
        let fim = fisher_information_normal(0.0, 1.0);
        assert!((fim[0][0] - 1.0).abs() < 1e-10, "I_μμ = 1 for N(0,1)");
        assert!((fim[0][1]).abs() < 1e-10, "I_μσ = 0");
        assert!((fim[1][0]).abs() < 1e-10, "I_σμ = 0");
        assert!((fim[1][1] - 2.0).abs() < 1e-10, "I_σσ = 2 for N(0,1)");
    }

    #[test]
    fn test_fisher_normal_scaling() {
        // N(μ, σ²): I_μμ = 1/σ²
        let fim = fisher_information_normal(3.0, 2.0);
        assert!((fim[0][0] - 0.25).abs() < 1e-10);
        assert!((fim[1][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_normal_self_is_zero() {
        let kl = kl_divergence_normal(1.0, 2.0, 1.0, 2.0);
        assert!(kl.abs() < 1e-10, "KL(P‖P) = 0, got {}", kl);
    }

    #[test]
    fn test_kl_divergence_normal_nonnegative() {
        // KL divergence is always non-negative
        let kl = kl_divergence_normal(0.0, 1.0, 1.0, 2.0);
        assert!(kl >= -1e-10, "KL ≥ 0, got {}", kl);
    }

    #[test]
    fn test_kl_categorical_self_zero() {
        let p = vec![0.2, 0.5, 0.3];
        let kl = kl_divergence_categorical(&p, &p);
        assert!(kl.abs() < 1e-10, "KL(p‖p) = 0");
    }

    #[test]
    fn test_kl_categorical_nonnegative() {
        let p = vec![0.1, 0.6, 0.3];
        let q = vec![0.4, 0.4, 0.2];
        let kl = kl_divergence_categorical(&p, &q);
        assert!(kl >= -1e-10, "KL ≥ 0");
    }

    #[test]
    fn test_natural_gradient_direction() {
        // For FIM = identity, natural gradient = ordinary gradient
        let fim = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let grad = vec![0.5, -0.3];
        let ng = natural_gradient(&grad, &fim);
        assert!((ng[0] - 0.5).abs() < 1e-10);
        assert!((ng[1] + 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_natural_gradient_step_reduces() {
        // With FIM = identity and small lr, step should move in gradient direction
        let fim = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let params = vec![1.0, 2.0];
        let grad = vec![1.0, -1.0];
        let new_params = natural_gradient_step(&params, &grad, &fim, 0.1);
        assert!((new_params[0] - 0.9).abs() < 1e-10);
        assert!((new_params[1] - 2.1).abs() < 1e-10);
    }

    #[test]
    fn test_cramer_rao_normal() {
        // For N(μ, σ²) with known σ: FIM for μ = 1/σ², Cramér-Rao = σ²
        let sigma = 2.0f64;
        let fim = fisher_information_normal(0.0, sigma)[0][0]; // I_μμ
        let cr = cramer_rao_bound(fim);
        assert!((cr - sigma * sigma).abs() < 1e-10, "Cramér-Rao = σ²");
    }

    #[test]
    fn test_exponential_to_expectation_roundtrip() {
        let (mu, sigma) = (1.5f64, 0.8f64);
        let (eta1, eta2) = exponential_to_expectation_normal(mu, sigma);
        let (mu_back, sigma_back) = expectation_to_natural_normal(eta1, eta2);
        assert!((mu_back - mu).abs() < 1e-10, "μ roundtrip");
        assert!((sigma_back - sigma).abs() < 1e-10, "σ roundtrip");
    }

    #[test]
    fn test_e_geodesic_endpoints() {
        let theta0 = vec![0.0, 1.0];
        let theta1 = vec![2.0, 3.0];
        let at0 = exponential_family_geodesic(&theta0, &theta1, 0.0);
        let at1 = exponential_family_geodesic(&theta0, &theta1, 1.0);
        assert!((at0[0] - theta0[0]).abs() < 1e-10);
        assert!((at1[0] - theta1[0]).abs() < 1e-10);
    }

    #[test]
    fn test_symmetrized_kl_symmetric() {
        let p = vec![0.3, 0.4, 0.3];
        let q = vec![0.1, 0.7, 0.2];
        let skl_pq = symmetrized_kl(&p, &q);
        let skl_qp = symmetrized_kl(&q, &p);
        assert!(
            (skl_pq - skl_qp).abs() < 1e-10,
            "Symmetrized KL is symmetric"
        );
    }

    #[test]
    fn test_fisher_bernoulli() {
        // I(0.5) = 1/(0.5*0.5) = 4
        let i = fisher_information_bernoulli(0.5);
        assert!((i - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_natural_to_mean_bernoulli() {
        // θ = 0 → p = 0.5
        let mean = natural_to_mean_params(&[0.0], "bernoulli");
        assert!((mean[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_natural_to_mean_poisson() {
        // θ = ln(3) → λ = 3
        let mean = natural_to_mean_params(&[3.0f64.ln()], "poisson");
        assert!((mean[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_alpha_connection_zero_gradient_is_zero() {
        // For flat manifold (zero FIM derivative), all Christoffel symbols vanish
        let fim = vec![vec![1.0f64, 0.0], vec![0.0, 2.0]];
        let dfim = vec![
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        ];
        let christoffel = alpha_connection_christoffel(0.0, &fim, &dfim);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    assert!(
                        christoffel[i][j][k].abs() < 1e-10,
                        "Flat manifold: Γ^(0)_{{{}{}{}}} = 0",
                        i,
                        j,
                        k
                    );
                }
            }
        }
    }
}
