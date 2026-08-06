// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Detritivore Bridge — Mk0 Plastic Shredder & Extruder Translation Layer.
//!
//! Maps shredder torque, extruder temperature, and material feedstock intake
//! into HDC perception space. This provides the "material metabolism"
//! feedback for the Mk0 Bootstrapper Protocol.

use symthaea_core::embodiment::{
    EmbodimentBridge, EmbodimentPlatform, EmbodimentResult, MotorSafetyLevel,
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Telemetry from the Detritivore recycling system.
#[derive(Debug, Clone, Default)]
pub struct DetritivoreTelemetry {
    /// Shredder motor current/torque (detects jam or empty)
    pub shredder_torque: f32,
    /// Extruder nozzle temperature in Celsius
    pub extruder_temp: f32,
    /// Feedstock intake mass in grams
    pub feedstock_mass: f32,
    /// Filament output yield (0.0 to 1.0)
    pub yield_efficiency: f32,
    /// Contamination detection flag (0.0 = clean, 1.0 = metallic/foreign debris)
    pub contamination_index: f32,
}

/// EmbodimentBridge for the Mk0-Detritivore recycler.
pub struct DetritivoreEmbodiment {
    /// Ethics-layer veto set by `apply_moral_gate`, composed most-restrictive-wins
    /// into `safety_level()`. Added 2026-07-31 -- see that method's note.
    moral_safety: Option<MotorSafetyLevel>,
    pub genesis: GenesisSeed,
    pub telemetry: DetritivoreTelemetry,
    steps: usize,
}

impl DetritivoreEmbodiment {
    /// Create a new Detritivore embodiment.
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            moral_safety: None,
            genesis: genesis.clone(),
            telemetry: DetritivoreTelemetry::default(),
            steps: 0,
        }
    }

    /// Update the internal telemetry from real or simulated hardware.
    pub fn update_hardware_state(&mut self, telemetry: DetritivoreTelemetry) {
        self.telemetry = telemetry;
    }
}

impl EmbodimentBridge for DetritivoreEmbodiment {
    fn step(&mut self, _thought_hv: &ContinuousHV, _dt: f32, _phi: f64) -> EmbodimentResult {
        self.steps += 1;

        // Detritivore actions:
        // - Shredder speed modulation
        // - Extruder PID target
        // - Cooling fan activation

        EmbodimentResult {
            num_actuators: 2, // Shredder Motor, Extruder Heater
            control_effort: self.telemetry.shredder_torque + (self.telemetry.extruder_temp / 200.0),
            success: self.telemetry.contamination_index < 0.5, // Failure if contaminated
            prediction_error: 0.0,
            safety_level: self.safety_level(),
            epistemic_grounding: 0,
            observation_confidence: 1.0,
        }
    }

    fn encode_perception(&mut self) -> ContinuousHV {
        let mut bundle = Vec::new();
        let dim = 16384;

        // 1. Encode Shredder Torque (Resistance/Friction)
        let torque_hv = ContinuousHV::random(dim, (self.telemetry.shredder_torque * 100.0) as u64);
        bundle.push(torque_hv);

        // 2. Encode Feedstock Level (Material Input)
        let mass_hv = ContinuousHV::random(dim, (self.telemetry.feedstock_mass) as u64);
        bundle.push(mass_hv);

        // 3. Encode Thermal State (Metabolism Stability)
        let temp_hv = ContinuousHV::random(dim, (self.telemetry.extruder_temp) as u64);
        bundle.push(temp_hv);

        // 4. Encode Contamination (Environmental Surprise)
        let contamination_hv =
            ContinuousHV::random(dim, (self.telemetry.contamination_index * 1000.0) as u64);
        bundle.push(contamination_hv);

        if bundle.is_empty() {
            return ContinuousHV::zero(dim);
        }

        let refs: Vec<&ContinuousHV> = bundle.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// NOTE (2026-07-31): this is driven by waste-processing line (contamination + thermal limits) ONLY -- it does **not**
    /// consult the `phi` argument `step()` receives, which is bound as `_phi` and
    /// discarded. That is a separate, deliberate-looking gap from the moral one fixed
    /// here, and is NOT changed by this commit: whether a waste-processing line
    /// should be Phi-gated at all is a design question, not a bug to patch silently.
    /// The moral veto below composes most-restrictive-wins (derived Ord is
    /// Green < Yellow < Orange < Red).
    fn safety_level(&self) -> MotorSafetyLevel {
        let base = {
            if self.telemetry.contamination_index > 0.8 {
                MotorSafetyLevel::Red // Stop: metal detected in shredder
            } else if self.telemetry.extruder_temp > 260.0 {
                MotorSafetyLevel::Yellow // Caution: thermal limit
            } else {
                MotorSafetyLevel::Green
            }
        };
        match self.moral_safety {
            Some(m) => base.max(m),
            None => base,
        }
    }

    fn platform(&self) -> EmbodimentPlatform {
        EmbodimentPlatform::Detritivore
    }

    fn num_actuators(&self) -> usize {
        2
    }
    fn total_steps(&self) -> usize {
        self.steps
    }
    fn reset(&mut self) {
        self.steps = 0;
        // Must clear the moral veto, or it persists across reset and the bridge stays
        // gated forever. Caught by the contract's assertion 3, whose failure message
        // names this exact omission.
        self.moral_safety = None;
    }

    fn telemetry(&self) -> symthaea_core::embodiment::EmbodimentTelemetry {
        Default::default()
    }

    fn set_safety_override(&mut self, _level: MotorSafetyLevel) {}
    fn clear_safety_override(&mut self) {}
    /// Apply an ethics-layer verdict.
    ///
    /// FIXED 2026-07-31. This was an **explicitly empty override** -- literally
    /// `fn apply_moral_gate(&mut self, _gate: ..) {}` -- which is worse than
    /// inheriting the trait default, because a grep-based audit for
    /// `fn apply_moral_gate` counts it as compliant. A static sweep of the fleet did
    /// exactly that and reported this bridge as honouring the contract; only running
    /// `tests/embodiment_moral_contract.rs` against it revealed otherwise.
    fn apply_moral_gate(&mut self, gate: symthaea_core::embodiment::MoralGateInput) {
        use symthaea_core::embodiment::MoralGateInput as MGI;
        self.moral_safety = if gate.ahimsa_violated || gate.verdict == MGI::VERDICT_BLOCKED {
            Some(MotorSafetyLevel::Red)
        } else if gate.consent_violation {
            Some(MotorSafetyLevel::Orange)
        } else if gate.verdict == MGI::VERDICT_CAUTION {
            Some(MotorSafetyLevel::Yellow)
        } else {
            None
        };
    }
    fn platform_telemetry_bytes(&self) -> Vec<u8> {
        Vec::new()
    }
}
