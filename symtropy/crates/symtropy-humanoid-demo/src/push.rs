// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Periodic horizontal push perturbations applied to the torso.
//!
//! Modeled on the flight demo's gust source: triangular envelope over a
//! brief window, pseudo-random direction per cycle.

pub struct PushSource {
    period: f64,
    duration: f64,
    peak_force: f64,
    seed: u64,
}

impl Default for PushSource {
    fn default() -> Self {
        Self {
            period: 6.5,
            duration: 0.9,
            peak_force: 220.0,
            seed: 0x5A5A5A5A5A5A_5A5A,
        }
    }
}

impl PushSource {
    /// World-frame push force (Newtons) and normalized intensity at time `t`.
    pub fn sample(&self, t: f64) -> ([f64; 3], f64) {
        let cycle = (t / self.period).floor() as i64;
        let phase = t - cycle as f64 * self.period;

        if phase > self.duration {
            return ([0.0; 3], 0.0);
        }

        let half = self.duration * 0.5;
        let env = if phase < half {
            phase / half
        } else {
            (self.duration - phase) / half
        }
        .clamp(0.0, 1.0);

        // Pseudo-random horizontal direction per cycle
        let mut h = (cycle as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.seed;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        let theta = (h as f64 / u64::MAX as f64) * std::f64::consts::TAU;

        let fx = theta.cos() * self.peak_force * env;
        let fy = theta.sin() * self.peak_force * env;
        ([fx, fy, 0.0], env)
    }
}
