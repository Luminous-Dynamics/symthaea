// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Haptic-Semantic Alignment: Proprioceptive-Semantic Binder for FEP.

use symthaea_core::hdc::ContinuousHV;

/// Maps a platform's high-dimensional proprioceptive state to a semantic constraint vector.
///
/// This binder allows the FEP module to project physical configurations (e.g., limb strain,
/// balance) into HDV semantic space. The Cognitive Loop can then use this binding to
/// "feel" physical constraints and modulate planning based on somatic integrity.
pub struct HapticSemanticBinder {
    /// Semantic mapping matrix for projection
    projection_matrix: Vec<f32>,
    /// Dimension of the incoming proprioceptive state
    state_dim: usize,
    /// Dimension of the target semantic space
    semantic_dim: usize,
}

impl HapticSemanticBinder {
    pub fn new(state_dim: usize, semantic_dim: usize) -> Self {
        // Initialize with random projection weights
        let mut rng = symthaea_core::hdc::deterministic_seeds::seed_from_name("haptic_binder");
        let projection_matrix = (0..(state_dim * semantic_dim))
            .map(|_| {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        Self {
            projection_matrix,
            state_dim,
            semantic_dim,
        }
    }

    /// Project proprioceptive state HV into semantic semantic constraint space, modulated by actuator health.
    pub fn bind_with_health(
        &self,
        proprioception: &[f32],
        actuator_health: &[f32],
    ) -> ContinuousHV {
        let mut semantic_data = vec![0.0f32; self.semantic_dim];

        // Linear projection, but dampen contributions from unhealthy actuators (actuator_health < 1.0)
        for i in 0..self.semantic_dim {
            for j in 0..self.state_dim {
                let health = actuator_health.get(j).copied().unwrap_or(1.0);
                semantic_data[i] +=
                    proprioception[j] * health * self.projection_matrix[i * self.state_dim + j];
            }
        }

        // Return as semantic hypervector
        ContinuousHV::from_vec(semantic_data)
    }

    /// Project proprioceptive state HV into semantic semantic constraint space.
    pub fn bind(&self, proprioception: &[f32]) -> ContinuousHV {
        self.bind_with_health(proprioception, &vec![1.0; self.state_dim])
    }
}
