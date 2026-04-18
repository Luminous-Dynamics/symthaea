// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rough-terrain perturbation source: a periodic "rough patch" window
//! that raises `roughness` ∈ [0,1]. The simulator has no external-force
//! API, so this drives observation noise + the danger signal rather
//! than mutating physics directly.

pub struct TerrainField {
    pub period: f64,
    pub duration: f64,
    pub peak: f64,
}

impl Default for TerrainField {
    fn default() -> Self {
        Self {
            period: 5.0,
            duration: 1.8,
            peak: 1.0,
        }
    }
}

impl TerrainField {
    /// Return `(roughness, forward_x_of_current_patch_if_active)`.
    pub fn sample(&self, t: f64) -> (f64, Option<f64>) {
        let cycle = (t / self.period).floor() as i64;
        let phase = t - cycle as f64 * self.period;
        if phase > self.duration {
            return (0.0, None);
        }
        let half = self.duration * 0.5;
        let env = if phase < half {
            phase / half
        } else {
            (self.duration - phase) / half
        }
        .clamp(0.0, 1.0)
            * self.peak;
        // Rough patch is placed at the robot's forward position of ~(cycle+1)*1.5 m
        // (very loose — used only as a visual cue).
        let x = (cycle as f64 + 1.0) * 1.5;
        (env, Some(x))
    }
}
