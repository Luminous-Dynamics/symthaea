// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Accessors for the memetic immune system (plan Phase 2).
//!
//! Exposes the live memetic-defense telemetry so callers/tests can observe
//! screening without reaching into the loop internals. Feature-gated behind
//! `social-fabric` (the same gate as the immune field itself).

#![cfg(feature = "social-fabric")]

use crate::cognitive_loop::CognitiveLoopService;
use symthaea_memetics::MemeticTelemetry;

impl CognitiveLoopService {
    /// Current memetic immune telemetry: memes seen/rejected/accepted, rolling
    /// mean resonance and contagion index, and immune-memory size.
    pub fn memetic_telemetry(&self) -> MemeticTelemetry {
        self.memetic_immune.telemetry()
    }

    /// Vaccinate the memetic immune system against a known pathogen signature,
    /// so future variants that resonate with it are rejected (mutation-tolerant).
    pub fn vaccinate_meme(&mut self, pathogen: symthaea_core::hdc::BinaryHV) {
        self.memetic_immune.vaccinate(pathogen);
    }
}
