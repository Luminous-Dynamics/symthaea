// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Periodic wind-gust disturbance source.
//!
//! Emits a bounded lateral force that ramps up, holds, then decays so the
//! controller has to reject it. The magnitude drives the `danger_level` fed
//! into the consciousness pipeline — stronger gust → higher danger → more
//! caution → lower motor gain.

/// State machine for a single gust pattern.
pub struct WindGustSource {
    period: f64,
    duration: f64,
    peak_force: f64,
    phase_jitter_seed: u64,
}

impl Default for WindGustSource {
    fn default() -> Self {
        Self {
            period: 8.0,
            duration: 2.5,
            peak_force: 0.025, // N — ~10× Crazyflie weight during peak
            phase_jitter_seed: 0xC0FFEE,
        }
    }
}

impl WindGustSource {
    /// World-frame force in Newtons at time `t`, and a normalized "intensity"
    /// in `[0,1]` for the danger channel.
    pub fn sample(&self, t: f64) -> ([f64; 3], f64) {
        let cycle = (t / self.period).floor() as i64;
        let phase = t - cycle as f64 * self.period;

        if phase > self.duration {
            return ([0.0; 3], 0.0);
        }

        // Smooth triangular envelope
        let half = self.duration * 0.5;
        let env = if phase < half {
            phase / half
        } else {
            (self.duration - phase) / half
        }
        .clamp(0.0, 1.0);

        // Pseudo-random direction per cycle (xorshift on cycle + seed)
        let mut h = (cycle as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.phase_jitter_seed;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        let theta = (h as f64 / u64::MAX as f64) * std::f64::consts::TAU;

        let fx = theta.cos() * self.peak_force * env;
        let fy = theta.sin() * self.peak_force * env;

        ([fx, fy, 0.0], env)
    }
}
