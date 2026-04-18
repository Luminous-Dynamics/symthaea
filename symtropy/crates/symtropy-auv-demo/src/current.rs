// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Underwater-current disturbance: a slowly rotating lateral force whose
//! magnitude varies with depth (strong near the thermocline at 8–15 m,
//! calmer near the surface and in the deep).

use std::f64::consts::TAU;

pub struct UnderwaterCurrent {
    pub rotation_period: f64,
    pub peak_force: f64,
    pub thermocline_depth: f64,
    pub thermocline_width: f64,
}

impl Default for UnderwaterCurrent {
    fn default() -> Self {
        Self {
            rotation_period: 18.0,
            peak_force: 90.0,
            thermocline_depth: 12.0,
            thermocline_width: 5.0,
        }
    }
}

impl UnderwaterCurrent {
    /// World-frame current force (Newtons) and a normalized intensity `[0,1]`.
    pub fn sample(&self, t: f64, depth: f64) -> ([f64; 3], f64) {
        // Depth-dependent magnitude (Gaussian bump centered at thermocline)
        let d = (depth - self.thermocline_depth) / self.thermocline_width;
        let env = (-d * d).exp();

        let phase = (t / self.rotation_period) * TAU;
        let fx = self.peak_force * env * phase.cos();
        let fy = self.peak_force * env * phase.sin();
        // Small vertical component (upwelling/downwelling cycles)
        let fz = 0.15 * self.peak_force * env * (2.0 * phase).sin();

        ([fx, fy, fz], env)
    }
}
