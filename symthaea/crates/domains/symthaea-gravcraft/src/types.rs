// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for the gravcraft platform.

use serde::{Deserialize, Serialize};

/// State of a single gravity amplifier.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct AmplifierState {
    /// Metric perturbation amplitude (0 = flat, 1 = max distortion)
    pub amplitude: f64,
    /// Focal distance from craft center (meters)
    pub focal_distance: f64,
    /// Azimuthal angle (radians, 0..2π)
    pub azimuth: f64,
    /// Elevation angle (radians, -π/2..π/2)
    pub elevation: f64,
}

/// Full state of the gravcraft.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravcraftState {
    /// Position in world frame (meters)
    pub position: [f64; 3],
    /// Coordinate velocity (m/s)
    pub velocity: [f64; 3],
    /// Scalar summary of metric deviation from flat (dimensionless)
    pub metric_perturbation: f64,
    /// Kretschner scalar proxy — tidal force magnitude
    pub tidal_tensor_norm: f64,
    /// State of each gravity amplifier
    pub amplifiers: [AmplifierState; 3],
    /// Simulation time (seconds)
    pub timestamp: f64,
}

impl Default for GravcraftState {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            metric_perturbation: 0.0,
            tidal_tensor_norm: 0.0,
            amplifiers: [AmplifierState::default(); 3],
            timestamp: 0.0,
        }
    }
}

impl GravcraftState {
    /// Flatten state to channel array for HDC encoding.
    pub fn to_channels(&self) -> [f32; 16] {
        [
            self.position[0] as f32,
            self.position[1] as f32,
            self.position[2] as f32,
            self.velocity[0] as f32,
            self.velocity[1] as f32,
            self.velocity[2] as f32,
            self.metric_perturbation as f32,
            self.tidal_tensor_norm as f32,
            self.amplifiers[0].amplitude as f32,
            self.amplifiers[0].focal_distance as f32,
            self.amplifiers[1].amplitude as f32,
            self.amplifiers[1].focal_distance as f32,
            self.amplifiers[2].amplitude as f32,
            self.amplifiers[2].focal_distance as f32,
            self.timestamp as f32,
            0.0, // padding
        ]
    }
}

/// Metric distortion command — 12 control channels for 3 amplifiers.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetricCommand {
    /// Per-amplifier commands: (amplitude, focal_distance, azimuth, elevation)
    pub amplifiers: [(f64, f64, f64, f64); 3],
}

/// Configuration for the gravcraft platform.
#[derive(Debug, Clone)]
pub struct GravcraftConfig {
    /// Maximum metric perturbation amplitude (dimensionless, typically 0.01-0.1)
    pub max_metric_amplitude: f64,
    /// Bubble radius for Alcubierre shaping (meters)
    pub bubble_radius: f64,
    /// Wall thickness parameter σ (1/meters, higher = thinner wall)
    pub wall_sigma: f64,
    /// RK4 substeps per simulation step
    pub rk4_substeps: usize,
}

impl Default for GravcraftConfig {
    fn default() -> Self {
        Self {
            max_metric_amplitude: 0.05,
            bubble_radius: 5.0,
            wall_sigma: 2.0,
            rk4_substeps: 4,
        }
    }
}
