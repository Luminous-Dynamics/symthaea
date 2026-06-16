// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Crystallized Dreaming — Dreaming with Spatial Macros
//!
//! Extends the Counterfactual Dream Engine to support spatial transformation
//! macro-operators as dreamable actions.

use crate::DreamableAction;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::grid_encoder::GridTransform;

/// A sequence of spatial transformations (crystallized logic) as a dreamable action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrystallizedAction {
    /// The chain of spatial transforms to apply
    pub transforms: Vec<GridTransform>,
}

impl DreamableAction for CrystallizedAction {
    fn perturb(&self, seed: u64) -> Self {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(&seed.to_le_bytes());

        let mut transforms = self.transforms.clone();

        // Simple perturbation:
        // 1. Add a random transform
        // 2. Remove a random transform
        // 3. Swap two transforms

        let mut bytes = [0u8; 4];
        let mut xof = hasher.finalize_xof();
        xof.fill(&mut bytes);
        let rand_val = u32::from_le_bytes(bytes);

        match rand_val % 3 {
            0 => {
                // Add Rotate90 as a common helpful transform
                transforms.push(GridTransform::Rotate90);
            }
            1 if !transforms.is_empty() => {
                // Remove last
                transforms.pop();
            }
            2 if transforms.len() >= 2 => {
                // Swap first and last
                let len = transforms.len();
                transforms.swap(0, len - 1);
            }
            _ => {
                // No-op or fallback: Reverse
                transforms.reverse();
            }
        }

        Self { transforms }
    }

    fn predict_outcome(&self, state: &[f32]) -> Vec<f32> {
        // A simple linear prediction model for dreaming
        // In a real system, this would use a learned world model or GridEncoder simulation
        let influence = self.magnitude();
        state
            .iter()
            .map(|&s| (s * 0.9 + influence * 0.1).clamp(-1.0, 1.0))
            .collect()
    }

    fn magnitude(&self) -> f32 {
        // Complexity of the macro chain
        self.transforms.len() as f32 * 0.2
    }
}

impl CrystallizedAction {
    /// Create from a single transform
    pub fn single(t: GridTransform) -> Self {
        Self {
            transforms: vec![t],
        }
    }

    /// Create from a chain
    pub fn chain(transforms: Vec<GridTransform>) -> Self {
        Self { transforms }
    }
}
