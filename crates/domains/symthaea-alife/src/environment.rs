// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! An exogenous resource signal an [`crate::Organism`] observes but does not control.
//!
//! Phase 0's core anti-theater requirement (`ALIFE_PLAN_2026-07-08.md`, "no unfalsifiable
//! convergence") is that every organism has a real input channel it cannot influence.
//! `Environment::resource_at` depends only on the tick counter and a fixed seed — never on
//! anything the organism does — so there is no possible self-referential loop here.

/// A slowly-varying resource concentration in `[0, 1]`, plus small deterministic noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Environment {
    /// Mean resource level.
    pub mean: f64,
    /// Oscillation amplitude around the mean.
    pub amplitude: f64,
    /// Period of the oscillation, in ticks.
    pub period: f64,
    /// Seed for the deterministic noise term (independent per-environment, reproducible).
    pub noise_seed: u64,
    /// Noise amplitude.
    pub noise_amplitude: f64,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            mean: 0.5,
            amplitude: 0.3,
            period: 200.0,
            noise_seed: 0xA5A5_1234_DEAD_BEEF,
            noise_amplitude: 0.02,
        }
    }
}

impl Environment {
    /// Resource level at tick `t`. Pure function of `(self, t)` — no organism state feeds in.
    pub fn resource_at(&self, t: u64) -> f64 {
        let phase = std::f64::consts::TAU * (t as f64) / self.period;
        let base = self.mean + self.amplitude * phase.sin();
        let noise = (xorshift_unit(self.noise_seed ^ t) - 0.5) * 2.0 * self.noise_amplitude;
        (base + noise).clamp(0.0, 1.0)
    }
}

/// Deterministic pseudo-random value in `[0, 1)` from a 64-bit key. Not cryptographic — only
/// used for reproducible, seed-driven test noise.
pub(crate) fn xorshift_unit(mut x: u64) -> f64 {
    if x == 0 {
        x = 0x9E37_79B9_7F4A_7C15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x as f64) / (u64::MAX as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_stays_in_unit_range() {
        let env = Environment::default();
        for t in 0..2000u64 {
            let r = env.resource_at(t);
            assert!(
                (0.0..=1.0).contains(&r),
                "resource_at({t}) = {r} out of range"
            );
        }
    }

    #[test]
    fn resource_is_a_pure_function_of_tick() {
        let env = Environment::default();
        for t in [0u64, 1, 50, 500, 12345] {
            assert_eq!(
                env.resource_at(t),
                env.resource_at(t),
                "not deterministic at t={t}"
            );
        }
    }

    #[test]
    fn resource_actually_oscillates() {
        // Sanity check the amplitude isn't accidentally zeroed out by clamping/noise.
        let env = Environment {
            noise_amplitude: 0.0,
            ..Environment::default()
        };
        let min = (0..400)
            .map(|t| env.resource_at(t))
            .fold(f64::MAX, f64::min);
        let max = (0..400)
            .map(|t| env.resource_at(t))
            .fold(f64::MIN, f64::max);
        assert!(
            max - min > 0.4,
            "expected real oscillation, got range {min}..{max}"
        );
    }
}
