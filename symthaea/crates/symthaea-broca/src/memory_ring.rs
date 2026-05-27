// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Receptive hyperdimensional context tracker for extended memory horizons.

use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

pub struct HdcContextRing {
    context_vector: ContinuousHV,
    decay_factor: f32,
}

impl HdcContextRing {
    pub fn new(decay_factor: f32) -> Self {
        Self {
            context_vector: ContinuousHV::zero(HDC_DIMENSION),
            decay_factor: decay_factor.clamp(0.0, 1.0),
        }
    }

    /// Roll the current token output hypervector into the foundational context ring
    pub fn push_trajectory(&mut self, output_hv: &ContinuousHV) {
        let mut composite = vec![0.0f32; HDC_DIMENSION];
        let current_slice = self.context_vector.as_slice();
        let incoming_slice = output_hv.as_slice();

        for i in 0..HDC_DIMENSION {
            composite[i] = (current_slice[i] * self.decay_factor)
                + (incoming_slice[i] * (1.0 - self.decay_factor));
        }
        self.context_vector = ContinuousHV::from_slice(&composite);
    }

    pub fn current_context(&self) -> &ContinuousHV {
        &self.context_vector
    }

    pub fn clear(&mut self) {
        self.context_vector = ContinuousHV::zero(HDC_DIMENSION);
    }
}
