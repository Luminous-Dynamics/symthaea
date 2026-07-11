// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ground-truth field environment for the agricultural stewardship platform.
//!
//! Fixes the actuator-derived-sensing bug from
//! `SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md`: `light_level` was
//! `0.55 + canopy_sensor_mast_torque * 0.35`, `terrain_roughness` was
//! `0.2 + drive_request * 0.15`, and `human_proximity` was
//! `drive_request * 0.2` -- all three were pure functions of the robot's own
//! commanded actuation, so a stationary robot always read zero human
//! proximity regardless of whether a human was actually standing next to it,
//! and driving harder always read as rougher terrain and closer humans for
//! no physical reason. This module gives all three an independent
//! ground-truth source; the simulator then senses them through a lagged
//! model instead of writing them directly from actuator commands.
use rand::Rng;
use symthaea_core::genesis::{GenesisSeed, ShakeRng};

/// Average rate (per second) a human enters proximity while none is present.
const HUMAN_ARRIVAL_RATE_PER_SEC: f64 = 0.03;
/// Average rate (per second) a present human leaves proximity.
const HUMAN_DEPARTURE_RATE_PER_SEC: f64 = 0.1;
/// Terrain roughness mean-reverts toward this baseline, independent of drive.
const TERRAIN_BASELINE: f64 = 0.25;

pub struct FieldEnvironment {
    /// Ground-truth ambient light, 0..1, drifting on a slow independent
    /// cycle. The canopy_sensor_mast actuator affects how quickly the
    /// simulator's sensed channel tracks this, not the ground truth itself.
    pub ambient_light: f64,
    /// Ground-truth terrain roughness at the robot's current location, 0..1.
    pub terrain_roughness: f64,
    pub human_present: bool,
    /// Ground-truth proximity intensity of a present human, 0..1.
    /// Meaningless (always 0) while no human is present.
    pub human_intensity: f64,
    rng: ShakeRng,
    light_phase: f64,
}

impl FieldEnvironment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            ambient_light: 0.7,
            terrain_roughness: TERRAIN_BASELINE,
            human_present: false,
            human_intensity: 0.0,
            rng: genesis.domain("agribot::field_environment"),
            light_phase: 0.0,
        }
    }

    /// Advance the ground-truth world by `dt` seconds.
    pub fn step(&mut self, dt: f64) {
        let dt = dt.max(0.0);

        // Ambient light: slow independent drift, not tied to the robot's
        // own mast deployment.
        self.light_phase += dt * 0.02;
        let diurnal = 0.55 + 0.35 * self.light_phase.sin();
        let light_noise = (self.rng.r#gen::<f64>() - 0.5) * 0.02;
        self.ambient_light = (diurnal + light_noise).clamp(0.0, 1.0);

        // Terrain roughness: mean-reverting random walk toward a baseline,
        // unrelated to the robot's own drive command.
        let terrain_noise = (self.rng.r#gen::<f64>() - 0.5) * 0.4 * dt;
        self.terrain_roughness = (self.terrain_roughness
            + (TERRAIN_BASELINE - self.terrain_roughness) * 0.1 * dt
            + terrain_noise)
            .clamp(0.0, 1.0);

        // Human presence: independent arrival/departure process.
        if self.human_present {
            if self.rng.r#gen::<f64>() < HUMAN_DEPARTURE_RATE_PER_SEC * dt {
                self.human_present = false;
                self.human_intensity = 0.0;
            } else {
                let drift = (self.rng.r#gen::<f64>() - 0.5) * 0.5 * dt;
                self.human_intensity = (self.human_intensity + drift).clamp(0.0, 1.0);
            }
        } else if self.rng.r#gen::<f64>() < HUMAN_ARRIVAL_RATE_PER_SEC * dt {
            self.human_present = true;
            self.human_intensity = self.rng.gen_range(0.4..0.95);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_after_many_steps() {
        let genesis = GenesisSeed::from_phrase("test-field-env");
        let mut env = FieldEnvironment::new(&genesis);
        for _ in 0..5000 {
            env.step(0.01);
            assert!(env.ambient_light.is_finite());
            assert!(env.terrain_roughness.is_finite());
            assert!(env.human_intensity.is_finite());
        }
    }

    #[test]
    fn test_no_human_without_arrival() {
        let genesis = GenesisSeed::from_phrase("test-field-env-human");
        let mut env = FieldEnvironment::new(&genesis);
        env.step(0.01);
        if !env.human_present {
            assert_eq!(env.human_intensity, 0.0);
        }
    }
}
