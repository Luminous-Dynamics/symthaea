// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gravity amplifier model — per-unit metric contribution.
//!
//! Each amplifier generates a directional metric perturbation using
//! an Alcubierre-like shaping function centered on its focal point.

use crate::types::GravcraftConfig;

/// Alcubierre shaping function f(r_s).
///
/// f(r_s) = (tanh(σ(r_s + R)) - tanh(σ(r_s - R))) / (2·tanh(σR))
///
/// Properties:
/// - f ≈ 1 for r_s << R (inside bubble)
/// - f ≈ 0 for r_s >> R (outside bubble)
/// - Smooth transition of width ~1/σ around r_s = R
///
/// Reference: Alcubierre, *Class. Quantum Grav.* 11 L73 (1994).
pub fn alcubierre_shaping(r_s: f64, radius: f64, sigma: f64) -> f64 {
    if radius <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let tanh_plus = (sigma * (r_s + radius)).tanh();
    let tanh_minus = (sigma * (r_s - radius)).tanh();
    let tanh_denom = (sigma * radius).tanh();
    if tanh_denom.abs() < 1e-15 {
        return 0.0;
    }
    (tanh_plus - tanh_minus) / (2.0 * tanh_denom)
}

/// Derivative of the shaping function df/dr_s.
///
/// Used for computing Christoffel symbols (metric gradients).
pub fn alcubierre_shaping_derivative(r_s: f64, radius: f64, sigma: f64) -> f64 {
    if radius <= 0.0 || sigma <= 0.0 {
        return 0.0;
    }
    let sech2_plus = 1.0 - (sigma * (r_s + radius)).tanh().powi(2);
    let sech2_minus = 1.0 - (sigma * (r_s - radius)).tanh().powi(2);
    let tanh_denom = (sigma * radius).tanh();
    if tanh_denom.abs() < 1e-15 {
        return 0.0;
    }
    sigma * (sech2_plus - sech2_minus) / (2.0 * tanh_denom)
}

/// Metric perturbation from a single gravity amplifier.
///
/// Returns the shift vector components (g_tx, g_ty, g_tz) at a given position.
///
/// The amplifier at `focal_point` creates a directional metric perturbation
/// along the focal direction. The perturbation is:
///   β^i = amplitude × f(|r - focal|) × direction^i
pub fn amplifier_metric_shift(
    position: [f64; 3],
    focal_point: [f64; 3],
    amplitude: f64,
    config: &GravcraftConfig,
) -> [f64; 3] {
    // Distance from position to focal point
    let dx = position[0] - focal_point[0];
    let dy = position[1] - focal_point[1];
    let dz = position[2] - focal_point[2];
    let r = (dx * dx + dy * dy + dz * dz).sqrt();

    if r < 1e-10 {
        return [0.0; 3];
    }

    // Shaping function value at this distance
    let f = alcubierre_shaping(r, config.bubble_radius, config.wall_sigma);

    // Direction from focal point to position (normalized)
    let dir = [dx / r, dy / r, dz / r];

    // Metric shift: β^i = amplitude × f(r) × direction
    [
        amplitude * f * dir[0],
        amplitude * f * dir[1],
        amplitude * f * dir[2],
    ]
}

/// Combined metric shift from all 3 amplifiers.
///
/// Returns the total shift vector and scalar metric perturbation magnitude.
pub fn combined_metric(
    position: [f64; 3],
    craft_position: [f64; 3],
    amplifier_cmds: &[(f64, f64, f64, f64); 3], // (amplitude, focal_dist, azimuth, elevation)
    config: &GravcraftConfig,
) -> ([f64; 3], f64) {
    let mut total_shift = [0.0; 3];

    for &(amplitude, focal_dist, azimuth, elevation) in amplifier_cmds {
        // Compute focal point from craft position + spherical offset
        let cos_el = elevation.cos();
        let focal = [
            craft_position[0] + focal_dist * cos_el * azimuth.cos(),
            craft_position[1] + focal_dist * cos_el * azimuth.sin(),
            craft_position[2] + focal_dist * elevation.sin(),
        ];

        let shift = amplifier_metric_shift(position, focal, amplitude, config);
        total_shift[0] += shift[0];
        total_shift[1] += shift[1];
        total_shift[2] += shift[2];
    }

    let magnitude =
        (total_shift[0].powi(2) + total_shift[1].powi(2) + total_shift[2].powi(2)).sqrt();

    (total_shift, magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shaping_center_is_one() {
        let f = alcubierre_shaping(0.0, 5.0, 2.0);
        assert!((f - 1.0).abs() < 0.01, "f(0) = {}, expected ~1.0", f);
    }

    #[test]
    fn test_shaping_far_is_zero() {
        let f = alcubierre_shaping(50.0, 5.0, 2.0);
        assert!(f.abs() < 0.01, "f(50) = {}, expected ~0.0", f);
    }

    #[test]
    fn test_shaping_bounded() {
        for r in 0..100 {
            let f = alcubierre_shaping(r as f64 * 0.5, 5.0, 2.0);
            assert!(f >= -0.01 && f <= 1.01, "f({}) = {} out of bounds", r, f);
        }
    }

    #[test]
    fn test_amplifier_metric_at_focal() {
        let config = GravcraftConfig::default();
        let pos = [0.0, 0.0, 0.0];
        let focal = [10.0, 0.0, 0.0];
        let shift = amplifier_metric_shift(pos, focal, 0.05, &config);
        // Shift should point from focal toward position (negative x direction)
        assert!(
            shift[0] < 0.0,
            "Shift should point toward focal: {:?}",
            shift
        );
    }

    #[test]
    fn test_combined_metric_three_amplifiers() {
        let config = GravcraftConfig::default();
        let craft_pos = [0.0, 0.0, 0.0];
        let cmds = [
            (0.05, 10.0, 0.0, 0.0),   // Forward (+x)
            (0.05, 10.0, 2.094, 0.0), // 120° (left-back)
            (0.05, 10.0, 4.189, 0.0), // 240° (right-back)
        ];
        let (shift, mag) = combined_metric(craft_pos, craft_pos, &cmds, &config);
        // Three symmetric amplifiers should partially cancel
        assert!(
            mag < 0.15,
            "Symmetric amplifiers should partially cancel: mag={}",
            mag
        );
        assert!(shift.iter().all(|v| v.is_finite()));
    }
}
