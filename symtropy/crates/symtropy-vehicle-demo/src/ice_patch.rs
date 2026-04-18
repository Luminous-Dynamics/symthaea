// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Track-side ice patches: circular low-friction zones that also emit a small
//! lateral gust force. An ice patch is "active" when the car enters it.

/// A single low-friction zone on the track.
#[derive(Clone, Copy, Debug)]
pub struct IcePatch {
    pub center: [f64; 2],
    pub radius: f64,
    /// Friction multiplier inside the patch (1.0 = dry asphalt, 0.2 = black ice).
    pub friction: f64,
    /// Lateral gust (world-frame newtons) applied while the car is inside.
    pub gust: [f64; 2],
}

/// A set of ice patches distributed around the track.
pub struct IceField {
    pub patches: Vec<IcePatch>,
}

impl IceField {
    /// Pre-seeded distribution for the default figure-8 track.
    pub fn default_layout() -> Self {
        Self {
            patches: vec![
                IcePatch {
                    center: [15.0, 8.0],
                    radius: 4.0,
                    friction: 0.22,
                    gust: [0.0, -180.0],
                },
                IcePatch {
                    center: [-12.0, -5.0],
                    radius: 3.5,
                    friction: 0.30,
                    gust: [120.0, 0.0],
                },
                IcePatch {
                    center: [0.0, 14.0],
                    radius: 3.0,
                    friction: 0.35,
                    gust: [-90.0, 60.0],
                },
                IcePatch {
                    center: [8.0, -14.0],
                    radius: 4.0,
                    friction: 0.28,
                    gust: [0.0, 160.0],
                },
            ],
        }
    }

    /// Return `(friction, external_force, intensity)` for the car's position.
    ///
    /// `intensity` is in `[0,1]` — 1.0 at patch center, 0.0 outside any patch.
    pub fn sample(&self, pos: [f64; 2]) -> (f64, [f64; 2], f64) {
        let mut worst_friction = 1.0f64;
        let mut accum_force = [0.0f64; 2];
        let mut worst_intensity = 0.0f64;

        for p in &self.patches {
            let dx = pos[0] - p.center[0];
            let dy = pos[1] - p.center[1];
            let d = (dx * dx + dy * dy).sqrt();
            if d < p.radius {
                // Linear falloff from center (1.0) to edge (0.0)
                let t = 1.0 - d / p.radius;
                // Blend friction toward patch value
                let blended = 1.0 + (p.friction - 1.0) * t;
                if blended < worst_friction {
                    worst_friction = blended;
                }
                accum_force[0] += p.gust[0] * t;
                accum_force[1] += p.gust[1] * t;
                if t > worst_intensity {
                    worst_intensity = t;
                }
            }
        }
        (worst_friction, accum_force, worst_intensity)
    }
}
