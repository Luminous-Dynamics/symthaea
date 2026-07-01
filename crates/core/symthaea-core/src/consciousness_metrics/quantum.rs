// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quantum-Inspired Entropy
//!
//! von Neumann entropy for quantum-inspired HDC analysis.

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Quantum entropy result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEntropyResult {
    /// von Neumann entropy S = -Tr(ρ log ρ)
    pub von_neumann_entropy: f64,
    /// Purity Tr(ρ²) - 1 for pure state, 1/d for maximally mixed
    pub purity: f64,
    /// Linear entropy S_L = 1 - Tr(ρ²)
    pub linear_entropy: f64,
    /// Eigenvalue spectrum of density matrix
    pub eigenvalues: Vec<f64>,
}

/// Quantum-inspired entropy calculator
///
/// Treats HDC vectors as quantum-like states and computes
/// von Neumann entropy and related measures.
#[derive(Debug, Clone, Default)]
pub struct QuantumEntropyCalculator {
    /// Dimension for density matrix (subsampled from full HDC dimension)
    pub subsample_dim: usize,
}

impl QuantumEntropyCalculator {
    /// Create calculator with default subsampling
    pub fn new() -> Self {
        Self {
            subsample_dim: 64, // Small enough for eigendecomposition
        }
    }

    /// Create calculator with custom dimension
    pub fn with_dimension(dim: usize) -> Self {
        Self {
            subsample_dim: dim.min(256), // Cap for computational feasibility
        }
    }

    /// Construct density matrix from HDC vector
    ///
    /// ρ = |ψ⟩⟨ψ| where ψ is the normalized HDC vector
    fn construct_density_matrix(&self, hv: &ContinuousHV) -> Vec<Vec<f64>> {
        let d = self.subsample_dim.min(hv.values.len());
        let step = hv.values.len() / d;

        // Subsample and normalize
        let mut psi: Vec<f64> = (0..d).map(|i| hv.values[i * step] as f64).collect();

        let norm: f64 = psi.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut psi {
                *x /= norm;
            }
        }

        // Construct ρ = |ψ⟩⟨ψ|
        let mut rho = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                rho[i][j] = psi[i] * psi[j];
            }
        }

        rho
    }

    /// Construct mixed density matrix from multiple HDC vectors
    ///
    /// ρ = Σ_i p_i |ψ_i⟩⟨ψ_i| (uniform weights)
    fn construct_mixed_density_matrix(&self, hvs: &[ContinuousHV]) -> Vec<Vec<f64>> {
        if hvs.is_empty() {
            return vec![vec![0.0; self.subsample_dim]; self.subsample_dim];
        }

        let d = self.subsample_dim;
        let mut rho = vec![vec![0.0; d]; d];
        let weight = 1.0 / hvs.len() as f64;

        for hv in hvs {
            let rho_i = self.construct_density_matrix(hv);
            for i in 0..d {
                for j in 0..d {
                    rho[i][j] += weight * rho_i[i][j];
                }
            }
        }

        rho
    }

    /// Compute eigenvalues of density matrix using power iteration
    fn compute_eigenvalues(&self, rho: &[Vec<f64>]) -> Vec<f64> {
        let d = rho.len();
        if d == 0 {
            return vec![];
        }

        // Simple power iteration for top eigenvalues
        // For a proper implementation, use nalgebra or similar
        let mut eigenvalues = Vec::new();
        let mut remaining = rho.to_vec();

        for _ in 0..d.min(10) {
            // Get top 10 eigenvalues
            let (lambda, v) = self.power_iteration(&remaining, 50);
            if lambda < 1e-10 {
                break;
            }
            eigenvalues.push(lambda);

            // Deflate: A = A - λvv^T
            for i in 0..d {
                for j in 0..d {
                    remaining[i][j] -= lambda * v[i] * v[j];
                }
            }
        }

        eigenvalues
    }

    /// Power iteration for largest eigenvalue
    fn power_iteration(&self, matrix: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
        let d = matrix.len();
        if d == 0 {
            return (0.0, vec![]);
        }

        // Initialize with random vector
        let mut v: Vec<f64> = (0..d).map(|i| ((i * 7 + 3) % 17) as f64 / 17.0).collect();

        // Normalize
        let mut norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in &mut v {
                *x /= norm;
            }
        }

        let mut lambda = 0.0;

        for _ in 0..max_iter {
            // Multiply: v' = Av
            let mut v_new = vec![0.0; d];
            for i in 0..d {
                for j in 0..d {
                    v_new[i] += matrix[i][j] * v[j];
                }
            }

            // Compute eigenvalue (Rayleigh quotient)
            lambda = v.iter().zip(v_new.iter()).map(|(a, b)| a * b).sum();

            // Normalize
            norm = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                break;
            }
            for x in &mut v_new {
                *x /= norm;
            }

            v = v_new;
        }

        (lambda.abs(), v)
    }

    /// Compute von Neumann entropy S = -Tr(ρ log ρ) = -Σ λ_i log λ_i
    pub fn von_neumann_entropy(&self, hv: &ContinuousHV) -> f64 {
        let rho = self.construct_density_matrix(hv);
        let eigenvalues = self.compute_eigenvalues(&rho);

        let mut s = 0.0;
        for lambda in &eigenvalues {
            if *lambda > 1e-10 {
                s -= lambda * lambda.log2();
            }
        }
        s.max(0.0)
    }

    /// Compute purity Tr(ρ²)
    pub fn purity(&self, hv: &ContinuousHV) -> f64 {
        let rho = self.construct_density_matrix(hv);
        let d = rho.len();

        // Tr(ρ²) = Σ_ij ρ_ij ρ_ji
        let mut trace = 0.0;
        for i in 0..d {
            for j in 0..d {
                trace += rho[i][j] * rho[j][i];
            }
        }
        trace
    }

    /// Full quantum entropy analysis
    pub fn analyze(&self, hv: &ContinuousHV) -> QuantumEntropyResult {
        let rho = self.construct_density_matrix(hv);
        let eigenvalues = self.compute_eigenvalues(&rho);

        let von_neumann = {
            let mut s = 0.0;
            for lambda in &eigenvalues {
                if *lambda > 1e-10 {
                    s -= lambda * lambda.log2();
                }
            }
            s.max(0.0)
        };

        let purity = {
            let d = rho.len();
            let mut trace = 0.0;
            for i in 0..d {
                for j in 0..d {
                    trace += rho[i][j] * rho[j][i];
                }
            }
            trace
        };

        let linear_entropy = 1.0 - purity;

        QuantumEntropyResult {
            von_neumann_entropy: von_neumann,
            purity,
            linear_entropy,
            eigenvalues,
        }
    }

    /// Analyze entanglement between two HDC vectors
    pub fn entanglement_entropy(&self, hv1: &ContinuousHV, hv2: &ContinuousHV) -> f64 {
        // Create joint state
        let joint = hv1.bind(hv2);

        // Create mixed state from marginals
        let rho_mixed = self.construct_mixed_density_matrix(&[hv1.clone(), hv2.clone()]);
        let eigenvalues = self.compute_eigenvalues(&rho_mixed);

        let mut s = 0.0;
        for lambda in &eigenvalues {
            if *lambda > 1e-10 {
                s -= lambda * lambda.log2();
            }
        }

        // Entanglement entropy is the difference from pure state
        let pure_entropy = self.von_neumann_entropy(&joint);
        (s - pure_entropy).abs()
    }
}
