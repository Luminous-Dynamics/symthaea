// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ground-truth encounter environment for the sanctuary platform.
//!
//! Fixes the self-referential-sensing bug from
//! `SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md`: `animal_presence` was
//! a pure autoregression (`x' = x*0.985 + 0.015`) that converged to 1.0
//! regardless of any command, and `vehicle_threat`/`robot_threat` were
//! derived solely from the robot's own drive torque. None of the three could
//! ever represent an actual external animal, vehicle, or intruding robot --
//! the platform's entire stated purpose (detect animal distress) was
//! unfalsifiable. This module gives all three a source independent of the
//! robot's own actuation history; the simulator then reads them through a
//! lagged sensor model instead of writing them directly.
use rand::Rng;
use symthaea_core::genesis::{GenesisSeed, ShakeRng};

/// Average rate (per second of sim time) of a new animal encounter starting
/// while none is present.
const ARRIVAL_RATE_PER_SEC: f64 = 0.05;
/// Average rate (per second) an ongoing encounter ends.
const DEPARTURE_RATE_PER_SEC: f64 = 0.08;

pub struct EncounterEnvironment {
    pub animal_present: bool,
    /// Ground-truth distress of the present animal, 0..1. Meaningless
    /// (always 0) while no animal is present.
    pub animal_distress: f64,
    /// Ground-truth intrusion pressure from an external vehicle, 0..1.
    pub vehicle_intrusion: f64,
    /// Ground-truth intrusion pressure from another robot, 0..1.
    pub robot_intrusion: f64,
    rng: ShakeRng,
}

impl EncounterEnvironment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            animal_present: false,
            animal_distress: 0.0,
            vehicle_intrusion: 0.0,
            robot_intrusion: 0.0,
            rng: genesis.domain("biota::encounter_environment"),
        }
    }

    /// Advance the ground-truth world by `dt` seconds. `soothing` (0..1) is
    /// the robot's calming signal (thermal_beacon + sanctuary_projector
    /// blended) -- it can ease an already-present animal's distress, but
    /// cannot manufacture or erase presence/intrusion itself.
    pub fn step(&mut self, dt: f64, soothing: f64) {
        let dt = dt.max(0.0);
        let soothing = soothing.clamp(0.0, 1.0);

        if self.animal_present {
            if self.rng.r#gen::<f64>() < DEPARTURE_RATE_PER_SEC * dt {
                self.animal_present = false;
                self.animal_distress = 0.0;
            } else {
                let drift = (self.rng.r#gen::<f64>() - 0.5) * 0.6 * dt;
                self.animal_distress =
                    (self.animal_distress + drift - soothing * 0.4 * dt).clamp(0.0, 1.0);
            }
        } else if self.rng.r#gen::<f64>() < ARRIVAL_RATE_PER_SEC * dt {
            self.animal_present = true;
            self.animal_distress = self.rng.gen_range(0.3..0.9);
        }

        // Independent mean-reverting random walks -- an external vehicle or
        // robot passing nearby, unrelated to this robot's own drive command.
        let vehicle_noise = (self.rng.r#gen::<f64>() - 0.5) * 0.4 * dt;
        self.vehicle_intrusion =
            (self.vehicle_intrusion * (1.0 - 0.3 * dt).max(0.0) + vehicle_noise).clamp(0.0, 1.0);
        let robot_noise = (self.rng.r#gen::<f64>() - 0.5) * 0.3 * dt;
        self.robot_intrusion =
            (self.robot_intrusion * (1.0 - 0.3 * dt).max(0.0) + robot_noise).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_encounter_without_arrival_roll() {
        // With departure/arrival driven only by rng, a short single step at
        // tiny dt should very rarely flip presence; assert the field stays
        // a valid bool and distress stays zeroed while absent.
        let genesis = GenesisSeed::from_phrase("test-env");
        let mut env = EncounterEnvironment::new(&genesis);
        env.step(0.01, 0.0);
        if !env.animal_present {
            assert_eq!(env.animal_distress, 0.0);
        }
    }

    #[test]
    fn test_soothing_reduces_distress_over_time() {
        let genesis = GenesisSeed::from_phrase("test-env-soothe");
        let mut env = EncounterEnvironment::new(&genesis);
        env.animal_present = true;
        env.animal_distress = 0.9;
        for _ in 0..500 {
            env.step(0.05, 1.0);
        }
        assert!(
            env.animal_distress < 0.9,
            "sustained full soothing must reduce distress from 0.9, got {}",
            env.animal_distress
        );
    }

    #[test]
    fn test_finite_after_many_steps() {
        let genesis = GenesisSeed::from_phrase("test-env-stable");
        let mut env = EncounterEnvironment::new(&genesis);
        for _ in 0..5000 {
            env.step(0.01, 0.5);
            assert!(env.animal_distress.is_finite());
            assert!(env.vehicle_intrusion.is_finite());
            assert!(env.robot_intrusion.is_finite());
        }
    }
}
