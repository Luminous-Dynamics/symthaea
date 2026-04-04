// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared visual and physical constants for Sol Atlas renderers.

// ─── Camera defaults ────────────────────────────────────────────

pub const CAMERA_ZOOM_MIN: f32 = 1.8;
pub const CAMERA_ZOOM_MAX: f32 = 8.0;
pub const CAMERA_INITIAL_DISTANCE: f32 = 4.2;
pub const CAMERA_INITIAL_THETA: f32 = -0.5;
pub const CAMERA_INERTIA_DECAY: f32 = 0.92;
pub const CAMERA_AUTO_ROTATE_SPEED: f32 = 0.0008;
pub const CAMERA_DRIFT_AMPLITUDE: f32 = 0.01;
pub const CAMERA_PHI_CLAMP_DEG: f32 = 85.0;

// ─── Globe geometry ─────────────────────────────────────────────

pub const GLOBE_RADIUS: f32 = 1.0;
pub const GLOBE_LAT_SEGMENTS: u32 = 128;
pub const GLOBE_LON_SEGMENTS: u32 = 128;

// ─── Starfield ──────────────────────────────────────────────────

pub const STARFIELD_COUNT: u32 = 3000;
pub const STARFIELD_RADIUS: f32 = 50.0;
pub const STARFIELD_SEED: u32 = 0xDEAD_BEEF;

// ─── Eight Harmonies palette (RGB 0.0-1.0) ──────────────────────

pub const HARMONY_COLORS: [[f32; 3]; 8] = [
    [0.40, 0.49, 0.92],  // 1. Resonant Coherence — #667eea
    [0.91, 0.12, 0.39],  // 2. Pan-Sentient Flourishing — #E91E63
    [0.00, 0.54, 0.48],  // 3. Integral Wisdom — #00897B
    [1.00, 0.84, 0.00],  // 4. Infinite Play — #FFD700
    [0.00, 0.87, 1.00],  // 5. Universal Interconnectedness — #00ddff
    [0.49, 0.23, 0.93],  // 6. Sacred Reciprocity — #7c3aed
    [0.23, 0.51, 0.96],  // 7. Evolutionary Progression — #3b82f6
    [0.06, 0.09, 0.16],  // 8. Sacred Stillness — deep indigo
];

// ─── Mycelix brand colors ───────────────────────────────────────

pub const MYCELIX_LIME: [f32; 3] = [0.486, 0.988, 0.0];
pub const MYCELIX_CYAN: [f32; 3] = [0.0, 0.87, 1.0];
pub const WISDOM_GREEN: [f32; 3] = [0.133, 0.286, 0.133];
pub const SACRED_GOLD: [f32; 3] = [1.0, 0.84, 0.0];
pub const SACRED_STILLNESS_INDIGO: [f32; 3] = [0.06, 0.09, 0.16];

// ─── Earth shader colors ────────────────────────────────────────

pub const EARTH_LAND_COLOR: [f32; 3] = WISDOM_GREEN;
pub const EARTH_OCEAN_COLOR: [f32; 3] = SACRED_STILLNESS_INDIGO;
pub const EARTH_COAST_COLOR: [f32; 3] = SACRED_GOLD;

// ─── Bloom post-processing ──────────────────────────────────────

pub const BLOOM_THRESHOLD: f32 = 0.8;
pub const BLOOM_INTENSITY: f32 = 0.15;

// ─── Sacred geometry ────────────────────────────────────────────

pub const SACRED_BREATHING_PERIOD: f32 = 8.0; // seconds per cycle
pub const GOLDEN_RATIO: f32 = 1.618_034;

// ─── Celestial body PBR defaults ────────────────────────────────

/// (roughness, metalness, emission) for common celestial objects.
pub mod celestial {
    pub const SUN: (f32, f32, f32) = (1.0, 0.0, 2.0);
    pub const MOON: (f32, f32, f32) = (0.95, 0.0, 0.0);
    pub const VENUS: (f32, f32, f32) = (0.8, 0.0, 0.0);
    pub const MARS: (f32, f32, f32) = (0.9, 0.1, 0.0);
    pub const JUPITER: (f32, f32, f32) = (0.7, 0.0, 0.0);
    pub const SATURN: (f32, f32, f32) = (0.75, 0.0, 0.0);
}
