// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Somatic Bridge — Bridges Broca's strategic intent to Soma's kinetic execution.
//!
//! Allows the Sovereign Architect to synthesize and broadcast motor-control
//! hypervectors to the robotic substrate.

use anyhow::Result;
use symthaea_core::hdc::ContinuousHV;

#[derive(Clone)]
pub struct SomaticBridge {
    pub hdc_dim: usize,
}

impl SomaticBridge {
    pub fn new(dim: usize) -> Self {
        Self { hdc_dim: dim }
    }

    /// Broadcast a kinetic nucleus to the Soma substrate.
    pub fn broadcast_kinetic_nucleus(&self, nucleus: &ContinuousHV) -> Result<()> {
        println!("🦾 Somatic Bridge: Broadcasting Kinetic Nucleus to Soma engine...");
        // (In real: we would push this to an Iceoryx2 ring-buffer for the motor controller)
        let norm = nucleus.norm();
        println!(
            "   └─ Kinetic Energy: {:.4} | Dimension: {}",
            norm, self.hdc_dim
        );
        Ok(())
    }

    /// Convert Somatic Prediction Error (from body) to cognitive curiosity.
    pub fn interpret_somatic_pe(&self, pe: f32) -> f32 {
        // High PE -> High Curiosity spike
        pe.clamp(0.0, 1.0)
    }
}
