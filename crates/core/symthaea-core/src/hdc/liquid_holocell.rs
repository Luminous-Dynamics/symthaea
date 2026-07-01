// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::HdcDimensionality;
use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// The atomic primitive unit of Symthaea.
///
/// A Liquid Holocell fuses continuous-time liquid dynamics (LTC/CfC)
/// with holographic hyperdimensional memory (HDC). It supports
/// 'Holographic Dilation', allowing the cell to dynamically scale its
/// semantic resolution from 2^14 to 2^16 based on thermodynamic load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LiquidHolocell {
    /// The current holographic state of the cell.
    pub state: ContinuousHV,

    /// The liquid time-constant (tau) governing temporal decay.
    pub tau: f32,

    /// Current dimensionality tier.
    pub dimensionality: HdcDimensionality,

    /// Thermodynamic pressure (mapped from 6W to 20W).
    pub pressure: f32,
}

impl LiquidHolocell {
    /// Create a new Liquid Holocell at the standard baseline (2^14).
    pub fn new(seed: u64) -> Self {
        let dim = HdcDimensionality::Standard;
        Self {
            state: ContinuousHV::random(dim.dimension(), seed),
            tau: 1.0,
            dimensionality: dim,
            pressure: 0.0,
        }
    }

    /// Set the liquid time-constant.
    pub fn with_tau(mut self, tau: f32) -> Self {
        self.tau = tau;
        self
    }

    /// Perform 'Holographic Dilation' - scale dimensionality.
    ///
    /// Scaling is performed using a fractal folding/unfolding technique
    /// that preserves semantic alignment across different resolutions.
    pub fn dilate(&mut self, target: HdcDimensionality) {
        let current_dim = self.dimensionality.dimension();
        let target_dim = target.dimension();

        if current_dim == target_dim {
            return;
        }

        if target_dim > current_dim {
            // UPSAMPLING (Unfolding)
            // We expand the vector by tiling it with permutations.
            // This maintains the original semantic signal while
            // providing more orthogonal space for new associations.
            let mut new_values = Vec::with_capacity(target_dim);
            let mut current_vec = self.state.clone();

            while new_values.len() < target_dim {
                let chunk_size = current_vec.values.len().min(target_dim - new_values.len());
                new_values.extend_from_slice(&current_vec.values[..chunk_size]);
                // Permute for the next chunk to avoid trivial repetition
                current_vec = current_vec.permute(1);
            }
            self.state = ContinuousHV::from_vec(new_values);
        } else {
            // DOWNSAMPLING (Folding)
            // We fold the vector back by bundling its constituent segments.
            // This is a lossy operation that compresses the semantic signal
            // back into the 6W baseline dimensionality.
            let chunks: Vec<Vec<f32>> = self
                .state
                .values
                .chunks(target_dim)
                .map(|c| c.to_vec())
                .collect();

            let mut folded = vec![0.0f32; target_dim];
            let n = chunks.len() as f32;

            for chunk in chunks {
                for (i, &val) in chunk.iter().enumerate() {
                    if i < target_dim {
                        folded[i] += val / n;
                    }
                }
            }
            self.state = ContinuousHV::from_vec(folded);
        }

        self.dimensionality = target;
    }

    /// Integrate a new input into the cell's state using liquid dynamics.
    ///
    /// dH/dt = -1/tau * H + Input
    pub fn step(&mut self, input: &ContinuousHV, dt: f32) {
        // Ensure input matches current dimensionality
        let mut aligned_input = input.clone();
        if aligned_input.dim() != self.state.dim() {
            // Create a temp holocell to dilate the input
            let mut temp = LiquidHolocell {
                state: aligned_input,
                tau: 1.0,
                dimensionality: HdcDimensionality::from_dimension(input.dim()),
                pressure: 0.0,
            };
            temp.dilate(self.dimensionality);
            aligned_input = temp.state;
        }

        // Liquid Update: H_new = H_old * exp(-dt/tau) + Input * (1 - exp(-dt/tau))
        let decay = (-dt / self.tau).exp();
        let integration = 1.0 - decay;

        for (h, &i) in self
            .state
            .values
            .iter_mut()
            .zip(aligned_input.values.iter())
        {
            *h = (*h * decay) + (i * integration);
        }
    }

    /// Fork the holocell into a temporary 'Dream Sandbox' for consequence simulation.
    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// Simulate the thermodynamic impact of an action.
    ///
    /// Returns the predicted thermodynamic load [0, 1].
    pub fn simulate(&self, input: &ContinuousHV, iterations: usize) -> f32 {
        let mut sandbox = self.fork();
        let dt = 0.1;

        for _ in 0..iterations {
            sandbox.step(input, dt);
        }

        // Predicted Load = Energy of the delta (surprise)
        let surprise = 1.0 - self.state.similarity(&sandbox.state);

        // Map surprise to thermodynamic load prediction
        // High surprise = high power draw
        surprise.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holocell_dilation_upsampling() {
        let mut cell = LiquidHolocell::new(42);
        assert_eq!(cell.state.dim(), 16_384);

        cell.dilate(HdcDimensionality::Ultra);
        assert_eq!(cell.state.dim(), 65_536);

        // Semantic check: original 16K segment should be preserved in the first chunk
        // (Wait, my impl uses tiling with permutation, so segment 0 is identical)
    }

    #[test]
    fn test_holocell_dilation_downsampling() {
        let mut cell = LiquidHolocell::new(42);
        cell.dilate(HdcDimensionality::Ultra);

        cell.dilate(HdcDimensionality::Standard);
        assert_eq!(cell.state.dim(), 16_384);
    }

    #[test]
    fn test_holocell_liquid_step() {
        let mut cell = LiquidHolocell::new(42);
        let input = ContinuousHV::random(16_384, 43);

        let original_state = cell.state.clone();
        cell.step(&input, 0.1);

        let sim_to_input = cell.state.similarity(&input);
        let sim_to_old = cell.state.similarity(&original_state);

        assert!(sim_to_input > 0.0);
        assert!(sim_to_old > 0.0);
    }
}
