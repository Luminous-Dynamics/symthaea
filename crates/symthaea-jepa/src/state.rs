// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! JEPA engine state: weight storage, energy tracking, telemetry.

use serde::{Deserialize, Serialize};

/// Persistent state for the JEPA engine.
///
/// Holds flattened weight matrices for the context encoder, predictor, and
/// target encoder, plus energy and loss tracking for thermodynamic telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepaState {
    /// Total energy spent by JEPA forward passes (joules, monotonically increasing).
    pub total_energy_spent: f64,

    /// Cumulative cosine loss across all forward passes.
    pub cumulative_loss: f64,

    /// Number of forward passes completed.
    pub steps: u64,

    /// Most recent latent prediction error (cosine loss from last forward pass).
    pub last_pe: f32,

    /// Running variance of each latent dimension (for collapse detection).
    /// Updated via Welford's online algorithm.
    pub latent_variance: Vec<f32>,

    /// Welford mean accumulator for variance computation.
    latent_mean: Vec<f32>,

    /// Welford M2 accumulator.
    latent_m2: Vec<f32>,
}

impl JepaState {
    /// Create a fresh state for a given latent dimension.
    pub fn new(latent_dim: usize) -> Self {
        Self {
            total_energy_spent: 0.0,
            cumulative_loss: 0.0,
            steps: 0,
            last_pe: 1.0, // Start with maximum uncertainty
            latent_variance: vec![1.0; latent_dim],
            latent_mean: vec![0.0; latent_dim],
            latent_m2: vec![0.0; latent_dim],
        }
    }

    /// Update running variance estimate via Welford's online algorithm.
    pub fn update_variance(&mut self, latent: &[f32]) {
        self.steps += 1;
        let n = self.steps as f32;
        for (i, &x) in latent.iter().enumerate() {
            if i >= self.latent_mean.len() {
                break;
            }
            let delta = x - self.latent_mean[i];
            self.latent_mean[i] += delta / n;
            let delta2 = x - self.latent_mean[i];
            self.latent_m2[i] += delta * delta2;
            self.latent_variance[i] = if self.steps > 1 {
                self.latent_m2[i] / (n - 1.0)
            } else {
                1.0
            };
        }
    }

    /// Check if any latent dimension has collapsed (variance below floor).
    pub fn has_collapse(&self, floor: f32) -> bool {
        self.steps > 10 && self.latent_variance.iter().any(|&v| v < floor)
    }

    /// Compute variance regularization penalty.
    /// Returns penalty proportional to how many dimensions are below the floor.
    pub fn variance_penalty(&self, floor: f32) -> f32 {
        if self.steps <= 10 {
            return 0.0;
        }
        let violations: f32 = self
            .latent_variance
            .iter()
            .map(|&v| if v < floor { floor - v } else { 0.0 })
            .sum();
        violations / self.latent_variance.len() as f32
    }
}
