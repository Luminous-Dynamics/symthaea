// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Helios Bridge — Mk0 Solar Microgrid Translation Layer.
//!
//! Maps solar inverter telemetry (voltage, current, power) and battery
//! state-of-charge (SoC) into HDC perception space. This provides the
//! "energy metabolism" feedback for the Mk0 Bootstrapper Protocol.

use symthaea_core::embodiment::{
    EmbodimentBridge, EmbodimentPlatform, EmbodimentResult, MotorSafetyLevel,
};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;

/// Telemetry from the Helios solar microgrid.
#[derive(Debug, Clone, Default)]
pub struct HeliosTelemetry {
    pub pv_voltage: f32,
    pub pv_current: f32,
    pub battery_soc: f32, // 0.0 to 1.0
    pub load_power: f32,
    pub temperature_c: f32,
}

/// EmbodimentBridge for the Mk0-Helios solar microgrid.
pub struct HeliosEmbodiment {
    /// Ethics-layer veto set by `apply_moral_gate`, composed most-restrictive-wins
    /// into `safety_level()`. Added 2026-07-31 -- see that method's note.
    moral_safety: Option<MotorSafetyLevel>,
    pub genesis: GenesisSeed,
    pub telemetry: HeliosTelemetry,
    steps: usize,
}

impl HeliosEmbodiment {
    /// Create a new Helios embodiment.
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            moral_safety: None,
            genesis: genesis.clone(),
            telemetry: HeliosTelemetry::default(),
            steps: 0,
        }
    }

    /// Update the internal telemetry from real or simulated hardware.
    pub fn update_hardware_state(&mut self, telemetry: HeliosTelemetry) {
        self.telemetry = telemetry;
    }
}

impl EmbodimentBridge for HeliosEmbodiment {
    fn step(&mut self, _thought_hv: &ContinuousHV, _dt: f32, _phi: f64) -> EmbodimentResult {
        self.steps += 1;

        // Helios is primarily an observational/infrastructure platform.
        // Actions might include load-shedding or inverter mode switching,
        // but for Mk0-baseline we focus on metabolism sensing.
        EmbodimentResult {
            num_actuators: 0,
            control_effort: 0.0,
            success: true,
            prediction_error: 0.0,
            safety_level: self.safety_level(),
            epistemic_grounding: 0,
            observation_confidence: 1.0,
        }
    }

    fn encode_perception(&mut self) -> ContinuousHV {
        let mut bundle = Vec::new();
        let dim = 16384; // Unified HDC dimension

        // 1. Encode PV Power (Metabolism Input)
        let pv_power = self.telemetry.pv_voltage * self.telemetry.pv_current;
        let power_hv = ContinuousHV::random(dim, (pv_power * 100.0) as u64);
        bundle.push(power_hv);

        // 2. Encode Battery SoC (Reserve Level)
        let soc_hv = ContinuousHV::random(dim, (self.telemetry.battery_soc * 1000.0) as u64);
        bundle.push(soc_hv);

        // 3. Encode Load (Metabolism Output)
        let load_hv = ContinuousHV::random(dim, (self.telemetry.load_power * 10.0) as u64);
        bundle.push(load_hv);

        if bundle.is_empty() {
            return ContinuousHV::zero(dim);
        }

        let refs: Vec<&ContinuousHV> = bundle.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// NOTE (2026-07-31): this is driven by solar microgrid (battery state-of-charge) ONLY -- it does **not**
    /// consult the `phi` argument `step()` receives, which is bound as `_phi` and
    /// discarded. That is a separate, deliberate-looking gap from the moral one fixed
    /// here, and is NOT changed by this commit: whether a solar microgrid
    /// should be Phi-gated at all is a design question, not a bug to patch silently.
    /// The moral veto below composes most-restrictive-wins (derived Ord is
    /// Green < Yellow < Orange < Red).
    fn safety_level(&self) -> MotorSafetyLevel {
        let base = {
            if self.telemetry.battery_soc < 0.1 {
                MotorSafetyLevel::Red // Critical reserve
            } else if self.telemetry.battery_soc < 0.3 {
                MotorSafetyLevel::Yellow // Low reserve
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
        EmbodimentPlatform::Infrastructure
    }

    fn num_actuators(&self) -> usize {
        0
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
