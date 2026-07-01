// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metric controller — maps HDC thought vector to metric commands.
//!
//! Projects the 16,384D thought vector down to 12 control channels
//! (3 amplifiers × 4 parameters each), with Phi-dependent amplitude gating.

use crate::types::{GravcraftConfig, MetricCommand};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

/// Maps thought hypervectors to metric distortion commands.
pub struct MetricController {
    /// Projection weights: 12 random HVs, one per output channel
    projections: Vec<ContinuousHV>,
    config: GravcraftConfig,
}

impl MetricController {
    pub fn new(genesis: &GenesisSeed, config: &GravcraftConfig) -> Self {
        let projections = (0..12)
            .map(|i| genesis.hv(&format!("gravcraft::projection::{}", i), HDC_DIMENSION))
            .collect();
        Self {
            projections,
            config: config.clone(),
        }
    }

    /// Compute metric command from thought vector and current Phi.
    ///
    /// Phi directly modulates maximum amplitude: higher consciousness =
    /// more authority to bend spacetime.
    pub fn forward(&self, thought_hv: &ContinuousHV, phi: f64) -> MetricCommand {
        // Project thought vector to 12 scalar channels
        let channels: Vec<f64> = self
            .projections
            .iter()
            .map(|proj| thought_hv.similarity(proj) as f64)
            .collect();

        // Phi-gated maximum amplitude
        let max_amp = self.config.max_metric_amplitude * phi.clamp(0.0, 1.0);

        // Parse into 3 amplifier commands
        let mut amplifiers = [(0.0, 0.0, 0.0, 0.0); 3];
        for i in 0..3 {
            let base = i * 4;
            let amplitude = channels[base].tanh() * max_amp;
            let focal = channels[base + 1].tanh().abs() * 20.0 + 1.0; // 1-21 meters
            let azimuth = channels[base + 2] * std::f64::consts::PI; // -π to π
            let elevation = channels[base + 3].tanh() * std::f64::consts::FRAC_PI_2; // -π/2 to π/2

            amplifiers[i] = (amplitude, focal, azimuth, elevation);
        }

        MetricCommand { amplifiers }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_output_bounded() {
        let genesis = GenesisSeed::from_phrase("gravcraft test");
        let config = GravcraftConfig::default();
        let controller = MetricController::new(&genesis, &config);

        let thought = ContinuousHV::random(HDC_DIMENSION, 0xDEAD);
        let cmd = controller.forward(&thought, 0.8);

        for (amp, focal, _az, el) in &cmd.amplifiers {
            assert!(amp.abs() <= config.max_metric_amplitude + 0.01);
            assert!(*focal >= 0.0);
            assert!(el.abs() <= std::f64::consts::FRAC_PI_2 + 0.01);
        }
    }

    #[test]
    fn test_phi_zero_kills_amplitude() {
        let genesis = GenesisSeed::from_phrase("gravcraft test");
        let config = GravcraftConfig::default();
        let controller = MetricController::new(&genesis, &config);

        let thought = ContinuousHV::random(HDC_DIMENSION, 0xBEEF);
        let cmd = controller.forward(&thought, 0.0);

        for (amp, _, _, _) in &cmd.amplifiers {
            assert!(
                amp.abs() < 1e-10,
                "Phi=0 should kill all amplitude: {}",
                amp
            );
        }
    }

    #[test]
    fn test_controller_deterministic() {
        let genesis = GenesisSeed::from_phrase("determinism");
        let config = GravcraftConfig::default();
        let c1 = MetricController::new(&genesis, &config);
        let c2 = MetricController::new(&genesis, &config);

        let thought = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd1 = c1.forward(&thought, 0.7);
        let cmd2 = c2.forward(&thought, 0.7);

        assert_eq!(cmd1.amplifiers[0].0, cmd2.amplifiers[0].0);
    }
}
