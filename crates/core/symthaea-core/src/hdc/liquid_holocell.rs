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

    /// Change the HDC resolution using the canonical ContinuousHV dilation operator.
    ///
    /// Upsampling is a redundant embedding, not an increase in independent
    /// information capacity. Downsampling folds the permuted chunks back into
    /// their original coordinate alignment before averaging. Keeping this
    /// operation delegated to ContinuousHV::dilate avoids a second, subtly
    /// different implementation of the holographic transform.
    pub fn dilate(&mut self, target: HdcDimensionality) {
        let target_dim = target.dimension();
        if self.state.dim() == target_dim {
            self.dimensionality = target;
            return;
        }

        self.state = self.state.dilate(target_dim);
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

        // The first chunk is the original representation.
        let original = ContinuousHV::random(16_384, 42);
        assert_eq!(&cell.state.values[..16_384], &original.values);
    }

    #[test]
    fn test_holocell_dilation_downsampling() {
        let mut cell = LiquidHolocell::new(42);
        let original = cell.state.clone();

        cell.dilate(HdcDimensionality::Ultra);
        assert_eq!(cell.state.dim(), 65_536);
        cell.dilate(HdcDimensionality::Standard);

        assert_eq!(cell.state.dim(), 16_384);
        assert_eq!(
            cell.state.values,
            original.values,
            "upsample/downsample must undo the permutation before folding"
        );
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
