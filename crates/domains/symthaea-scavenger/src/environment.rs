// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ground-truth site environment for the disassembly/salvage platform.
//!
//! Fixes the actuator-derived-sensing bug from
//! `SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md`: `human_proximity` was
//! `channels[HUMAN_PROXIMITY]*0.94 + effective_cutter.abs()*0.03` -- a pure
//! function of the robot's own cutter usage, not any sensor. A robot that
//! never ran its cutter always read zero human proximity regardless of
//! whether a human was actually standing next to it, and cutting harder
//! always read as a closer human for no physical reason -- dangerous, since
//! this channel gates `human_guard` (suppresses cutter aggression) and
//! feeds `incident_risk` directly. This module gives it an independent
//! ground-truth source; the simulator then senses it through a lagged model
//! instead of writing it directly from the cutter command.
use rand::Rng;
use symthaea_core::genesis::{GenesisSeed, ShakeRng};

/// Average rate (per second) a human enters proximity while none is present.
const HUMAN_ARRIVAL_RATE_PER_SEC: f64 = 0.03;
/// Average rate (per second) a present human leaves proximity.
const HUMAN_DEPARTURE_RATE_PER_SEC: f64 = 0.1;

pub struct SiteEnvironment {
    pub human_present: bool,
    /// Ground-truth proximity intensity of a present human, 0..1.
    /// Meaningless (always 0) while no human is present.
    pub human_intensity: f64,
    rng: ShakeRng,
}

impl SiteEnvironment {
    pub fn new(genesis: &GenesisSeed) -> Self {
        Self {
            human_present: false,
            human_intensity: 0.0,
            rng: genesis.domain("scavenger::site_environment"),
        }
    }

    /// Advance the ground-truth world by `dt` seconds.
    pub fn step(&mut self, dt: f64) {
        let dt = dt.max(0.0);
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
        let genesis = GenesisSeed::from_phrase("test-site-env");
        let mut env = SiteEnvironment::new(&genesis);
        for _ in 0..5000 {
            env.step(0.01);
            assert!(env.human_intensity.is_finite());
        }
    }

    #[test]
    fn test_no_human_without_arrival() {
        let genesis = GenesisSeed::from_phrase("test-site-env-human");
        let mut env = SiteEnvironment::new(&genesis);
        env.step(0.01);
        if !env.human_present {
            assert_eq!(env.human_intensity, 0.0);
        }
    }
}
