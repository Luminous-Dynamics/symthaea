// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cycle-count based animation: breathing oscillation and easing functions.

use std::f64::consts::PI;

/// Breathing period in cycles. At 234Hz this is ~0.51s.
const BREATH_PERIOD: f64 = 120.0;

/// Compute breathing oscillation phase from cycle count.
/// Returns a value in [0, 1] representing one full breath cycle.
pub fn breathing_phase(cycle_count: u64) -> f64 {
    let t = (cycle_count as f64) * 2.0 * PI / BREATH_PERIOD;
    // sin maps to [-1,1], remap to [0,1]
    (t.sin() + 1.0) / 2.0
}

/// Smooth step easing: 3t² - 2t³
pub fn smooth_step(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Layer fade-in based on consciousness level and threshold.
/// Returns opacity in [0, 1].
pub fn layer_opacity(consciousness: f64, threshold: f64, fade_width: f64) -> f32 {
    if consciousness <= threshold {
        return 0.0;
    }
    let t = ((consciousness - threshold) / fade_width).min(1.0);
    smooth_step(t as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breathing_bounded() {
        for c in 0..300 {
            let p = breathing_phase(c);
            assert!(p >= 0.0 && p <= 1.0, "cycle {c}: {p}");
        }
    }

    #[test]
    fn breathing_periodic() {
        let a = breathing_phase(0);
        let b = breathing_phase(120);
        // One full period apart, should be ~equal
        assert!((a - b).abs() < 0.01);
    }

    #[test]
    fn smooth_step_bounds() {
        assert_eq!(smooth_step(0.0), 0.0);
        assert_eq!(smooth_step(1.0), 1.0);
        assert!((smooth_step(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn layer_opacity_below_threshold() {
        assert_eq!(layer_opacity(0.1, 0.3, 0.2), 0.0);
    }

    #[test]
    fn layer_opacity_above_threshold() {
        let o = layer_opacity(0.6, 0.3, 0.2);
        assert!((o - 1.0).abs() < 0.01);
    }

    #[test]
    fn layer_opacity_mid_fade() {
        let o = layer_opacity(0.4, 0.3, 0.2);
        assert!(o > 0.0 && o < 1.0);
    }
}
