// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[derive(Debug, Clone, Copy)]
pub struct GaitGenome {
    pub walk_v: f32,
    pub walk_s: f32,
    pub walk_k: f32,
    pub climb_v: f32,
    pub climb_s: f32,
    pub climb_k: f32,
    pub vel_gain: f32,
    pub stride_gain: f32,
}

impl GaitGenome {
    pub fn default() -> Self {
        Self {
            // UPGRADED: Scaled base target values to match high-frequency liquid neural oscillations
            walk_v: 3.25,
            walk_s: 0.32,
            walk_k: 0.45,
            climb_v: 2.10,
            climb_s: 0.38,
            climb_k: 0.55,
            vel_gain: 0.45,
            stride_gain: 0.08,
        }
    }
}

pub struct Rng {
    pub state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 33) as f32 / (1u32 << 31) as f32
    }

    pub fn random_mutation(&mut self, base: f32, variance: f32) -> f32 {
        base + (self.next_f32() * 2.0 - 1.0) * variance
    }
}
