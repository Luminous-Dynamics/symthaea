// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adaptive invariant guard featuring cold-start calibration loops.

use std::cell::Cell;
use symthaea_broca::generator::BrocaGenerator;
use symthaea_core::hdc::ContinuousHV;

pub struct AxiomaticInvariantGuard {
    moving_coherence: Cell<Option<f32>>,
    momentum: f32,
    sensitivity: f32,
}

impl AxiomaticInvariantGuard {
    pub fn new(momentum: f32, sensitivity: f32) -> Self {
        Self {
            moving_coherence: Cell::new(None),
            momentum: momentum.clamp(0.0, 1.0),
            sensitivity,
        }
    }

    /// Dynamically evaluates trajectory coherence with cold-start auto-calibration
    pub fn verify_emission_path(
        &self,
        generator: &mut BrocaGenerator,
        thought_hv: &ContinuousHV,
        _token_id: u32,
    ) -> (bool, f32, f32) {
        let output_hv = generator.controller().output_hv();
        let coherence = output_hv.similarity(thought_hv);

        // COLD-START CALIBRATION: If this is step zero, snap baseline directly to reality
        let current_baseline = match self.moving_coherence.get() {
            Some(base) => base,
            None => {
                self.moving_coherence.set(Some(coherence));
                generator.config_mut().gating.enable_soft_veto = false;
                return (true, coherence, coherence);
            }
        };

        // Update the online moving average profile smoothly via exponential decay
        let updated_baseline =
            (current_baseline * self.momentum) + (coherence * (1.0 - self.momentum));
        self.moving_coherence.set(Some(updated_baseline));

        // Dynamic Veto Logic: Only trigger if variance drops below your tracking sensitivity standard deviation
        if coherence < (current_baseline - self.sensitivity) {
            generator.config_mut().gating.enable_soft_veto = true;
            return (false, coherence, current_baseline);
        }

        generator.config_mut().gating.enable_soft_veto = false;
        (true, coherence, current_baseline)
    }
}
