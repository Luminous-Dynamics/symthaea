// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Information Geometry — the natural geometry of probability and consciousness.
//!
//! The Fisher information metric gives a Riemannian geometry to the space of
//! probability distributions. Consciousness, as a probability distribution
//! over states, lives in this geometric space.
//!
//! g_ij(θ) = E[∂ᵢ log p(x|θ) × ∂ⱼ log p(x|θ)]  (Fisher metric)
//!
//! References:
//! - Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
//! - Ay, N. et al. (2017). *Information Geometry*. Springer.
//! - Tononi, G. (2008). Consciousness as Integrated Information (geometric Φ).

/// Fisher information for a single parameter: I(θ) = E[(d/dθ log p)²]
///
/// For discrete distribution p_i(θ): I = Σ_i (dp_i/dθ)² / p_i
pub fn fisher_information(probs: &[f64], dprobs_dtheta: &[f64]) -> f64 {
    probs
        .iter()
        .zip(dprobs_dtheta.iter())
        .filter(|(&p, _)| p > 1e-15)
        .map(|(&p, &dp)| dp * dp / p)
        .sum()
}

/// Fisher information matrix for multiple parameters.
/// g_ij = Σ_k (∂p_k/∂θ_i)(∂p_k/∂θ_j) / p_k
///
/// `jacobian[k * n_params + i]` = ∂p_k/∂θ_i
pub fn fisher_metric(probs: &[f64], jacobian: &[f64], n_params: usize) -> Vec<f64> {
    let n_states = probs.len();
    let mut g = vec![0.0; n_params * n_params];

    for i in 0..n_params {
        for j in i..n_params {
            let mut val = 0.0;
            for k in 0..n_states {
                if probs[k] > 1e-15 {
                    let dpi = jacobian[k * n_params + i];
                    let dpj = jacobian[k * n_params + j];
                    val += dpi * dpj / probs[k];
                }
            }
            g[i * n_params + j] = val;
            g[j * n_params + i] = val;
        }
    }

    g
}

/// Fisher information for a Gaussian distribution N(μ, σ²).
/// I = [[1/σ², 0], [0, 2/σ²]] (block diagonal in μ, σ)
pub fn fisher_gaussian(sigma: f64) -> [f64; 4] {
    let s2 = sigma * sigma;
    [1.0 / s2, 0.0, 0.0, 2.0 / s2]
}

/// Geodesic distance on the statistical manifold (Fisher-Rao distance).
///
/// For two distributions p and q on the probability simplex:
/// d(p, q) = 2 arccos(Σ √(p_i × q_i))  (Bhattacharyya angle)
pub fn fisher_rao_distance(p: &[f64], q: &[f64]) -> f64 {
    let bc: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi * qi).sqrt())
        .sum();
    2.0 * bc.clamp(0.0, 1.0).acos()
}

/// KL divergence: D_KL(p || q) = Σ p_i ln(p_i / q_i)
/// This is NOT a distance (asymmetric) but induces the Fisher metric locally.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > 1e-15 && qi > 1e-15)
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum()
}

/// Scalar curvature of the Fisher metric at a point on the probability simplex.
///
/// For a 2D probability simplex (3 states, 2 free parameters):
/// R = 2/(n-1) where n is the number of states.
/// This is the curvature of the sphere (positive curvature).
pub fn simplex_scalar_curvature(n_states: usize) -> f64 {
    if n_states <= 1 {
        return 0.0;
    }
    2.0 / (n_states as f64 - 1.0)
}

/// Information-geometric Phi (Φ_IG): the "consciousness curvature" of a state.
///
/// Measures how much a probability distribution deviates from the product
/// of its marginals — i.e., how integrated the information is.
///
/// Φ_IG = D_KL(p_joint || Π p_marginal) evaluated with Fisher metric weighting.
///
/// For a bipartite system with joint distribution p(a,b) and marginals p(a), p(b):
/// Φ = Σ_{a,b} p(a,b) ln[p(a,b) / (p(a) × p(b))]
pub fn phi_information_geometric(
    joint: &[f64], // p(a,b) flattened: joint[a * n_b + b]
    n_a: usize,
    n_b: usize,
) -> f64 {
    // Compute marginals
    let mut marginal_a = vec![0.0; n_a];
    let mut marginal_b = vec![0.0; n_b];

    for a in 0..n_a {
        for b in 0..n_b {
            let p = joint[a * n_b + b];
            marginal_a[a] += p;
            marginal_b[b] += p;
        }
    }

    // Φ = Σ p(a,b) ln(p(a,b) / (p(a) × p(b)))
    let mut phi = 0.0;
    for a in 0..n_a {
        for b in 0..n_b {
            let p_joint = joint[a * n_b + b];
            let p_product = marginal_a[a] * marginal_b[b];
            if p_joint > 1e-15 && p_product > 1e-15 {
                phi += p_joint * (p_joint / p_product).ln();
            }
        }
    }

    phi
}

/// Natural gradient: the gradient of a function in the Fisher metric.
/// ∇̃f = G⁻¹ ∇f  where G is the Fisher metric and ∇f is the Euclidean gradient.
///
/// Natural gradient descent converges faster because it follows geodesics
/// on the statistical manifold rather than straight lines in parameter space.
pub fn natural_gradient(
    fisher_metric_inv: &[f64],
    euclidean_gradient: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut nat_grad = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            nat_grad[i] += fisher_metric_inv[i * n + j] * euclidean_gradient[j];
        }
    }
    nat_grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_info_uniform() {
        // Uniform distribution: all dp/dθ = 0 → I = 0
        let probs = vec![0.25; 4];
        let dprobs = vec![0.0; 4];
        assert_eq!(fisher_information(&probs, &dprobs), 0.0);
    }

    #[test]
    fn test_fisher_gaussian_scaling() {
        // I_μμ = 1/σ² → larger σ = less information
        let g1 = fisher_gaussian(1.0);
        let g2 = fisher_gaussian(2.0);
        assert!(g1[0] > g2[0], "Narrower Gaussian has more Fisher info");
        assert!((g1[0] - 1.0).abs() < 1e-14, "I_μμ(σ=1) = 1");
        assert!((g2[0] - 0.25).abs() < 1e-14, "I_μμ(σ=2) = 0.25");
    }

    #[test]
    fn test_fisher_rao_self_distance() {
        let p = vec![0.3, 0.3, 0.4];
        let d = fisher_rao_distance(&p, &p);
        assert!(d.abs() < 1e-10, "Distance to self = 0: got {:.6}", d);
    }

    #[test]
    fn test_fisher_rao_triangle_inequality() {
        let p = vec![0.5, 0.5];
        let q = vec![0.3, 0.7];
        let r = vec![0.1, 0.9];
        let dpq = fisher_rao_distance(&p, &q);
        let dqr = fisher_rao_distance(&q, &r);
        let dpr = fisher_rao_distance(&p, &r);
        assert!(dpr <= dpq + dqr + 1e-10, "Triangle inequality violated");
    }

    #[test]
    fn test_kl_divergence_nonneg() {
        let p = vec![0.3, 0.7];
        let q = vec![0.5, 0.5];
        let d = kl_divergence(&p, &q);
        assert!(
            d >= -1e-14,
            "KL divergence should be non-negative: {:.6}",
            d
        );
    }

    #[test]
    fn test_phi_ig_independent() {
        // Independent: p(a,b) = p(a)*p(b) → Φ = 0
        let joint = vec![0.15, 0.35, 0.1, 0.4]; // p(a) = [0.5, 0.5], p(b) = [0.25, 0.75]
                                                // Actually: [0.5*0.5, 0.5*0.5, 0.5*0.5, 0.5*0.5] for independence
        let independent = vec![0.25, 0.25, 0.25, 0.25];
        let phi = phi_information_geometric(&independent, 2, 2);
        assert!(
            phi.abs() < 1e-10,
            "Independent system Φ = 0: got {:.6}",
            phi
        );
    }

    #[test]
    fn test_phi_ig_correlated() {
        // Maximally correlated: p(0,0) = p(1,1) = 0.5
        let correlated = vec![0.5, 0.0, 0.0, 0.5];
        let phi = phi_information_geometric(&correlated, 2, 2);
        assert!(
            phi > 0.5,
            "Correlated system should have high Φ: {:.4}",
            phi
        );
    }

    #[test]
    fn test_simplex_curvature() {
        // 2-simplex (3 states): R = 2/2 = 1 (unit sphere)
        let r = simplex_scalar_curvature(3);
        assert!((r - 1.0).abs() < 1e-14);
    }
}
